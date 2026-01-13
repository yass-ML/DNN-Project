"""
Differentiable Gaussian Splatting Rasterizer.

This module provides two rasterization backends:
1. gsplat (CUDA-accelerated) - Used when gsplat and CUDA toolkit are available
2. Pure PyTorch (CPU/GPU) - Fallback when gsplat is not available

gsplat provides up to 4x less GPU memory usage and 15% faster training.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple, Dict, Any
import math

from .utils import build_covariance_3d, compute_cov2d
from .camera import Camera

# Try to import gsplat
GSPLAT_AVAILABLE = False
try:
    from gsplat import rasterization as gsplat_rasterization
    # Test if CUDA backend works
    import torch
    if torch.cuda.is_available():
        GSPLAT_AVAILABLE = True
except ImportError:
    pass
except Exception as e:
    print(f"gsplat import failed: {e}")

if not GSPLAT_AVAILABLE:
    print("Note: Using pure PyTorch rasterizer (gsplat/CUDA not available)")


class RenderOutput(NamedTuple):
    """Output from rasterization."""

    rendered_image: torch.Tensor  # [3, H, W]
    radii: torch.Tensor  # [N] Gaussian radii in pixels
    viewspace_points: torch.Tensor  # [N, 2] or [N, 3] Points in screen space
    visibility_filter: torch.Tensor  # [N] Boolean mask of visible Gaussians
    meta: Optional[Dict[str, Any]] = None  # Additional metadata


# =============================================================================
# Pure PyTorch Rasterizer (Fallback)
# =============================================================================

class PyTorchGaussianRasterizer:
    """
    Pure-PyTorch differentiable Gaussian rasterizer.

    This is a simplified implementation that works without CUDA toolkit.
    For production use with CUDA, gsplat is recommended.
    """

    def __init__(
        self,
        tile_size: int = 16,
        max_gaussians_per_tile: int = 256,
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        **kwargs,  # Accept extra args for compatibility
    ):
        self.tile_size = tile_size
        self.max_gaussians_per_tile = max_gaussians_per_tile
        self.bg_color = bg_color

    def forward(
        self,
        means3D: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        camera: Camera,
        sh_degree: Optional[int] = None,
    ) -> RenderOutput:
        """Render Gaussians to image using pure PyTorch."""
        device = means3D.device
        N = means3D.shape[0]
        H, W = camera.image_height, camera.image_width

        # Transform to camera space
        means_cam = self._transform_points(means3D, camera.world_view_transform)

        # Frustum culling
        valid_z = means_cam[:, 2] > 0.2

        # Project to screen
        means2D, depths = self._project_points(means_cam, camera)

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
            camera.tan_fovy,
        )

        # Compute radii from eigenvalues
        det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] ** 2
        det = torch.clamp(det, min=1e-6)

        mid = 0.5 * (cov2D[:, 0, 0] + cov2D[:, 1, 1])
        diff = torch.sqrt(torch.clamp(mid**2 - det, min=0))
        lambda1 = mid + diff
        lambda2 = mid - diff

        radii = 3.0 * torch.sqrt(torch.max(lambda1, lambda2))
        radii = torch.ceil(radii).int()

        # Visibility filter
        visible = valid_z & (radii > 0)
        visible = (
            visible
            & (means2D[:, 0] > -radii.float())
            & (means2D[:, 0] < W + radii.float())
        )
        visible = (
            visible
            & (means2D[:, 1] > -radii.float())
            & (means2D[:, 1] < H + radii.float())
        )

        if not visible.any():
            bg = torch.tensor(self.bg_color, device=device).view(3, 1, 1)
            return RenderOutput(
                rendered_image=bg.expand(3, H, W),
                radii=radii,
                viewspace_points=means2D,
                visibility_filter=visible,
                meta=None,
            )

        # Handle SH colors (use only DC component for pure PyTorch)
        if colors.dim() == 3:
            colors = colors[:, 0, :]  # Just use DC

        rendered = self._render_gaussians(
            means2D[visible],
            cov2D[visible],
            colors[visible],
            opacities[visible],
            depths[visible],
            H,
            W,
            device,
        )

        return RenderOutput(
            rendered_image=rendered,
            radii=radii,
            viewspace_points=torch.cat([means2D, depths.unsqueeze(-1)], dim=-1),
            visibility_filter=visible,
            meta=None,
        )

    def _transform_points(
        self, points: torch.Tensor, view_matrix: torch.Tensor
    ) -> torch.Tensor:
        ones = torch.ones(points.shape[0], 1, device=points.device)
        points_h = torch.cat([points, ones], dim=-1)
        points_cam = (view_matrix @ points_h.T).T[:, :3]
        return points_cam

    def _project_points(
        self, points_cam: torch.Tensor, camera: Camera
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        u = camera.focal_x * x / z + camera.image_width / 2
        v = camera.focal_y * y / z + camera.image_height / 2
        means2D = torch.stack([u, v], dim=-1)
        return means2D, z

    def _render_gaussians(
        self,
        means2D: torch.Tensor,
        cov2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        M = means2D.shape[0]

        # Sort by depth
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

        tile_h = self.tile_size
        tile_w = self.tile_size
        n_tiles_h = (H + tile_h - 1) // tile_h
        n_tiles_w = (W + tile_w - 1) // tile_w

        output = torch.zeros(3, H, W, device=device)
        accumulated_alpha = torch.zeros(1, H, W, device=device)

        for ty in range(n_tiles_h):
            for tx in range(n_tiles_w):
                y0, y1 = ty * tile_h, min((ty + 1) * tile_h, H)
                x0, x1 = tx * tile_w, min((tx + 1) * tile_w, W)

                tile_output, tile_alpha = self._render_tile(
                    means2D, inv_cov, colors, opacities, x0, x1, y0, y1, device
                )

                output[:, y0:y1, x0:x1] = tile_output
                accumulated_alpha[:, y0:y1, x0:x1] = tile_alpha

        bg = torch.tensor(self.bg_color, device=device).view(3, 1, 1)
        output = output + bg * (1 - accumulated_alpha)

        return output

    def _render_tile(
        self,
        means2D: torch.Tensor,
        inv_cov: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        th, tw = y1 - y0, x1 - x0

        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device, dtype=torch.float32),
            torch.arange(x0, x1, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pixels = torch.stack([xx, yy], dim=-1)

        tile_center = torch.tensor([[(x0 + x1) / 2, (y0 + y1) / 2]], device=device)
        dist_to_tile = torch.norm(means2D - tile_center, dim=-1)
        tile_radius = math.sqrt(tw**2 + th**2) / 2 + 50
        tile_mask = dist_to_tile < tile_radius

        if not tile_mask.any():
            return (
                torch.zeros(3, th, tw, device=device),
                torch.zeros(1, th, tw, device=device),
            )

        means_tile = means2D[tile_mask]
        inv_cov_tile = inv_cov[tile_mask]
        colors_tile = colors[tile_mask]
        opacities_tile = opacities[tile_mask]
        K = means_tile.shape[0]

        if K > self.max_gaussians_per_tile:
            K = self.max_gaussians_per_tile
            means_tile = means_tile[:K]
            inv_cov_tile = inv_cov_tile[:K]
            colors_tile = colors_tile[:K]
            opacities_tile = opacities_tile[:K]

        diff = pixels.unsqueeze(2) - means_tile.view(1, 1, K, 2)
        tmp = torch.einsum("hwki,kij->hwkj", diff, inv_cov_tile)
        mahal = (tmp * diff).sum(dim=-1)
        gaussian = torch.exp(-0.5 * mahal)

        if opacities_tile.dim() == 2:
            opacities_tile = opacities_tile.squeeze(-1)
        alpha = gaussian * opacities_tile.view(1, 1, K)
        alpha = torch.clamp(alpha, max=0.99)

        output = torch.zeros(th, tw, 3, device=device)
        T = torch.ones(th, tw, device=device)

        for k in range(K):
            weight = alpha[:, :, k] * T
            output += weight.unsqueeze(-1) * colors_tile[k]
            T = T * (1 - alpha[:, :, k])

        accumulated_alpha = 1 - T
        return output.permute(2, 0, 1), accumulated_alpha.unsqueeze(0)


# =============================================================================
# gsplat Rasterizer (CUDA-accelerated)
# =============================================================================

class GsplatRasterizer:
    """CUDA-accelerated Gaussian rasterizer using gsplat."""

    def __init__(
        self,
        tile_size: int = 16,
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        near_plane: float = 0.01,
        far_plane: float = 100.0,
        eps2d: float = 0.3,
        packed: bool = False,
        absgrad: bool = True,
        rasterize_mode: str = "classic",
        **kwargs,
    ):
        self.tile_size = tile_size
        self.bg_color = bg_color
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.eps2d = eps2d
        self.packed = packed
        self.absgrad = absgrad
        self.rasterize_mode = rasterize_mode

    def forward(
        self,
        means3D: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        camera: Camera,
        sh_degree: Optional[int] = None,
    ) -> RenderOutput:
        device = means3D.device
        N = means3D.shape[0]
        H, W = camera.image_height, camera.image_width

        activated_scales = torch.exp(scales)
        if opacities.dim() == 2:
            opacities = opacities.squeeze(-1)
        opacities = torch.clamp(opacities, min=0.0, max=1.0)

        viewmat = camera.world_view_transform.unsqueeze(0)
        K = torch.tensor(
            [
                [camera.focal_x, 0, camera.image_width / 2],
                [0, camera.focal_y, camera.image_height / 2],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        backgrounds = torch.tensor(
            [self.bg_color], dtype=torch.float32, device=device
        )

        if colors.dim() == 2:
            colors_input = colors
            sh_degree_input = None
        else:
            colors_input = colors
            sh_degree_input = sh_degree

        render_colors, render_alphas, meta = gsplat_rasterization(
            means=means3D,
            quats=rotations,
            scales=activated_scales,
            opacities=opacities,
            colors=colors_input,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            eps2d=self.eps2d,
            sh_degree=sh_degree_input,
            packed=self.packed,
            tile_size=self.tile_size,
            backgrounds=backgrounds,
            render_mode="RGB",
            absgrad=self.absgrad,
            rasterize_mode=self.rasterize_mode,
        )

        rendered_image = render_colors[0].permute(2, 0, 1)

        radii = meta.get("radii", torch.zeros(N, dtype=torch.int32, device=device))
        if radii.dim() > 1:
            radii = radii.squeeze(0)

        means2d = meta.get("means2d", torch.zeros(N, 2, device=device))
        if means2d.dim() > 2:
            means2d = means2d.squeeze(0)

        visibility_filter = radii > 0

        return RenderOutput(
            rendered_image=rendered_image,
            radii=radii,
            viewspace_points=means2d,
            visibility_filter=visibility_filter,
            meta=meta,
        )


# =============================================================================
# Unified Rasterizer Interface
# =============================================================================

class GaussianRasterizer:
    """
    Unified Gaussian rasterizer that automatically selects the best backend.
    
    Uses gsplat (CUDA) when available, falls back to pure PyTorch otherwise.
    """

    def __init__(self, **kwargs):
        if GSPLAT_AVAILABLE:
            self._impl = GsplatRasterizer(**kwargs)
        else:
            self._impl = PyTorchGaussianRasterizer(**kwargs)

    def forward(self, *args, **kwargs) -> RenderOutput:
        return self._impl.forward(*args, **kwargs)


def render(
    gaussians: "GaussianModel",
    camera: Camera,
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scaling_modifier: float = 1.0,
    **kwargs,
) -> RenderOutput:
    """
    Convenience function to render Gaussians.

    Automatically uses gsplat if available, otherwise falls back to PyTorch.
    """
    rasterizer = GaussianRasterizer(bg_color=bg_color, **kwargs)

    viewdirs = camera.get_view_direction(gaussians.xyz)
    colors = gaussians.get_colors(viewdirs)

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
        sh_degree=gaussians.active_sh_degree,
    )


def render_with_sh(
    gaussians: "GaussianModel",
    camera: Camera,
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scaling_modifier: float = 1.0,
    **kwargs,
) -> RenderOutput:
    """Render with SH coefficients passed directly (for gsplat optimization)."""
    rasterizer = GaussianRasterizer(bg_color=bg_color, **kwargs)

    sh_features = gaussians.features

    scales = gaussians._scaling
    if scaling_modifier != 1.0:
        scales = scales + math.log(scaling_modifier)

    return rasterizer.forward(
        means3D=gaussians.xyz,
        scales=scales,
        rotations=gaussians._rotation,
        colors=sh_features,
        opacities=gaussians.opacity,
        camera=camera,
        sh_degree=gaussians.active_sh_degree,
    )
