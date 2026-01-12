"""
Differentiable Gaussian Splatting Rasterizer (Pure PyTorch).
Tile-based rendering with alpha blending.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple
import math

from .utils import build_covariance_3d, compute_cov2d
from .camera import Camera


class RenderOutput(NamedTuple):
    """Output from rasterization."""
    rendered_image: torch.Tensor  # [3, H, W]
    radii: torch.Tensor           # [N] Gaussian radii in pixels
    viewspace_points: torch.Tensor  # [N, 3] Points in view space (for gradients)
    visibility_filter: torch.Tensor  # [N] Boolean mask of visible Gaussians


class GaussianRasterizer:
    """
    Pure-PyTorch differentiable Gaussian rasterizer.
    
    This is a simplified implementation that trades some speed for simplicity.
    For production use, consider using a CUDA-accelerated version.
    """
    
    def __init__(
        self,
        tile_size: int = 16,
        max_gaussians_per_tile: int = 256,
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        """
        Args:
            tile_size: Size of screen tiles for binning
            max_gaussians_per_tile: Maximum Gaussians to render per tile
            bg_color: Background color (RGB, 0-1)
        """
        self.tile_size = tile_size
        self.max_gaussians_per_tile = max_gaussians_per_tile
        self.bg_color = bg_color
    
    def forward(
        self,
        means3D: torch.Tensor,      # [N, 3]
        scales: torch.Tensor,       # [N, 3]
        rotations: torch.Tensor,    # [N, 4]
        colors: torch.Tensor,       # [N, 3]
        opacities: torch.Tensor,    # [N, 1]
        camera: Camera,
        sh_degree: int = 0,
    ) -> RenderOutput:
        """
        Render Gaussians to image.
        
        Args:
            means3D: Gaussian centers in world space
            scales: Log-scale values (will be exp'd)
            rotations: Quaternions
            colors: RGB colors [0, 1]
            opacities: Opacity values [0, 1]
            camera: Camera object
            sh_degree: Active SH degree (for view-dependent coloring)
        
        Returns:
            RenderOutput with rendered image and auxiliary data
        """
        device = means3D.device
        N = means3D.shape[0]
        H, W = camera.image_height, camera.image_width
        
        # =====================================================================
        # Step 1: Transform to camera space and project
        # =====================================================================
        
        # Transform to camera space
        means_cam = self._transform_points(means3D, camera.world_view_transform)
        
        # Frustum culling
        valid_z = means_cam[:, 2] > 0.2  # In front of camera
        
        # Project to screen
        means2D, depths = self._project_points(means_cam, camera)
        
        # =====================================================================
        # Step 2: Compute 2D covariance for each Gaussian
        # =====================================================================
        
        # Build 3D covariance
        cov3D = build_covariance_3d(scales, rotations)
        
        # Project to 2D
        cov2D = compute_cov2d(
            cov3D,
            camera.world_view_transform,
            means_cam,
            camera.focal_x,
            camera.focal_y,
            camera.tan_fovx,
            camera.tan_fovy
        )
        
        # Compute radii from eigenvalues
        det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] ** 2
        det = torch.clamp(det, min=1e-6)
        
        mid = 0.5 * (cov2D[:, 0, 0] + cov2D[:, 1, 1])
        diff = torch.sqrt(torch.clamp(mid ** 2 - det, min=0))
        lambda1 = mid + diff
        lambda2 = mid - diff
        
        radii = 3.0 * torch.sqrt(torch.max(lambda1, lambda2))
        radii = torch.ceil(radii).int()
        
        # =====================================================================
        # Step 3: Filter visible Gaussians
        # =====================================================================
        
        # Visibility: in frustum and radius > 0
        visible = valid_z & (radii > 0)
        visible = visible & (means2D[:, 0] > -radii.float()) & (means2D[:, 0] < W + radii.float())
        visible = visible & (means2D[:, 1] > -radii.float()) & (means2D[:, 1] < H + radii.float())
        
        if not visible.any():
            # No visible Gaussians
            bg = torch.tensor(self.bg_color, device=device).view(3, 1, 1)
            return RenderOutput(
                rendered_image=bg.expand(3, H, W),
                radii=radii,
                viewspace_points=means2D,
                visibility_filter=visible
            )
        
        # =====================================================================
        # Step 4: Render using per-pixel alpha blending
        # =====================================================================
        
        rendered = self._render_gaussians(
            means2D[visible],
            cov2D[visible],
            colors[visible],
            opacities[visible],
            depths[visible],
            H, W,
            device
        )
        
        return RenderOutput(
            rendered_image=rendered,
            radii=radii,
            viewspace_points=torch.cat([means2D, depths.unsqueeze(-1)], dim=-1),
            visibility_filter=visible
        )
    
    def _transform_points(self, points: torch.Tensor, view_matrix: torch.Tensor) -> torch.Tensor:
        """Transform points from world to camera space."""
        ones = torch.ones(points.shape[0], 1, device=points.device)
        points_h = torch.cat([points, ones], dim=-1)  # [N, 4]
        points_cam = (view_matrix @ points_h.T).T[:, :3]  # [N, 3]
        return points_cam
    
    def _project_points(self, points_cam: torch.Tensor, camera: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project camera-space points to screen coordinates."""
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        # Perspective division
        u = camera.focal_x * x / z + camera.image_width / 2
        v = camera.focal_y * y / z + camera.image_height / 2
        
        means2D = torch.stack([u, v], dim=-1)
        return means2D, z
    
    def _render_gaussians(
        self,
        means2D: torch.Tensor,    # [M, 2]
        cov2D: torch.Tensor,      # [M, 2, 2]
        colors: torch.Tensor,     # [M, 3]
        opacities: torch.Tensor,  # [M, 1]
        depths: torch.Tensor,     # [M]
        H: int, W: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Render Gaussians using vectorized per-pixel computation.
        
        This is the core differentiable rendering loop.
        """
        M = means2D.shape[0]
        
        # Sort by depth (front to back for proper alpha blending)
        sorted_indices = torch.argsort(depths)
        means2D = means2D[sorted_indices]
        cov2D = cov2D[sorted_indices]
        colors = colors[sorted_indices]
        opacities = opacities[sorted_indices]
        
        # Precompute inverse covariance
        det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
        det = torch.clamp(det, min=1e-6)
        inv_cov = torch.zeros_like(cov2D)
        inv_cov[:, 0, 0] = cov2D[:, 1, 1] / det
        inv_cov[:, 1, 1] = cov2D[:, 0, 0] / det
        inv_cov[:, 0, 1] = -cov2D[:, 0, 1] / det
        inv_cov[:, 1, 0] = -cov2D[:, 1, 0] / det
        
        # For memory efficiency, render in tiles
        tile_h = self.tile_size
        tile_w = self.tile_size
        n_tiles_h = (H + tile_h - 1) // tile_h
        n_tiles_w = (W + tile_w - 1) // tile_w
        
        # Initialize output
        output = torch.zeros(3, H, W, device=device)
        accumulated_alpha = torch.zeros(1, H, W, device=device)
        
        # Render each tile
        for ty in range(n_tiles_h):
            for tx in range(n_tiles_w):
                y0, y1 = ty * tile_h, min((ty + 1) * tile_h, H)
                x0, x1 = tx * tile_w, min((tx + 1) * tile_w, W)
                
                tile_output, tile_alpha = self._render_tile(
                    means2D, inv_cov, colors, opacities,
                    x0, x1, y0, y1, device
                )
                
                output[:, y0:y1, x0:x1] = tile_output
                accumulated_alpha[:, y0:y1, x0:x1] = tile_alpha
        
        # Add background color
        bg = torch.tensor(self.bg_color, device=device).view(3, 1, 1)
        output = output + bg * (1 - accumulated_alpha)
        
        return output
    
    def _render_tile(
        self,
        means2D: torch.Tensor,
        inv_cov: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        x0: int, x1: int,
        y0: int, y1: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render a single tile."""
        th, tw = y1 - y0, x1 - x0
        M = means2D.shape[0]
        
        # Create pixel grid for this tile
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device, dtype=torch.float32),
            torch.arange(x0, x1, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pixels = torch.stack([xx, yy], dim=-1)  # [th, tw, 2]
        
        # Find Gaussians that overlap this tile (approximate)
        tile_center = torch.tensor([[(x0 + x1) / 2, (y0 + y1) / 2]], device=device)
        dist_to_tile = torch.norm(means2D - tile_center, dim=-1)
        tile_radius = math.sqrt(tw**2 + th**2) / 2 + 50  # Add margin
        tile_mask = dist_to_tile < tile_radius
        
        if not tile_mask.any():
            return (
                torch.zeros(3, th, tw, device=device),
                torch.zeros(1, th, tw, device=device)
            )
        
        # Filter to relevant Gaussians
        means_tile = means2D[tile_mask]      # [K, 2]
        inv_cov_tile = inv_cov[tile_mask]    # [K, 2, 2]
        colors_tile = colors[tile_mask]      # [K, 3]
        opacities_tile = opacities[tile_mask]  # [K, 1]
        K = means_tile.shape[0]
        
        # Limit number of Gaussians per tile
        if K > self.max_gaussians_per_tile:
            K = self.max_gaussians_per_tile
            means_tile = means_tile[:K]
            inv_cov_tile = inv_cov_tile[:K]
            colors_tile = colors_tile[:K]
            opacities_tile = opacities_tile[:K]
        
        # Compute Gaussian values at each pixel
        # pixels: [th, tw, 2], means: [K, 2]
        diff = pixels.unsqueeze(2) - means_tile.view(1, 1, K, 2)  # [th, tw, K, 2]
        
        # Mahalanobis distance: d.T @ inv_cov @ d
        # [th, tw, K, 2] @ [K, 2, 2] -> [th, tw, K, 2]
        tmp = torch.einsum('hwki,kij->hwkj', diff, inv_cov_tile)
        # [th, tw, K, 2] * [th, tw, K, 2] -> [th, tw, K]
        mahal = (tmp * diff).sum(dim=-1)
        
        # Gaussian weight
        gaussian = torch.exp(-0.5 * mahal)  # [th, tw, K]
        
        # Alpha values
        alpha = gaussian * opacities_tile.squeeze(-1).view(1, 1, K)  # [th, tw, K]
        alpha = torch.clamp(alpha, max=0.99)
        
        # Alpha blending (front to back, already sorted)
        output = torch.zeros(th, tw, 3, device=device)
        T = torch.ones(th, tw, device=device)  # Transmittance
        
        for k in range(K):
            weight = alpha[:, :, k] * T
            output += weight.unsqueeze(-1) * colors_tile[k]
            T = T * (1 - alpha[:, :, k])
        
        accumulated_alpha = 1 - T
        
        return output.permute(2, 0, 1), accumulated_alpha.unsqueeze(0)


def render(
    gaussians: 'GaussianModel',
    camera: Camera,
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scaling_modifier: float = 1.0
) -> RenderOutput:
    """
    Convenience function to render Gaussians.
    
    Args:
        gaussians: GaussianModel instance
        camera: Camera to render from
        bg_color: Background color
        scaling_modifier: Scale factor for Gaussian sizes
    
    Returns:
        RenderOutput
    """
    rasterizer = GaussianRasterizer(bg_color=bg_color)
    
    # Get view direction for each Gaussian
    viewdirs = camera.get_view_direction(gaussians.xyz)
    
    # Get colors (potentially view-dependent)
    colors = gaussians.get_colors(viewdirs)
    
    # Apply scaling modifier
    scales = gaussians._scaling
    if scaling_modifier != 1.0:
        scales = scales + math.log(scaling_modifier)
    
    return rasterizer.forward(
        means3D=gaussians.xyz,
        scales=scales,
        rotations=gaussians._rotation,
        colors=colors,
        opacities=gaussians.opacity,
        camera=camera,
        sh_degree=gaussians.active_sh_degree
    )
