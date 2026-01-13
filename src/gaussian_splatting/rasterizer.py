"""
Differentiable Gaussian Splatting Rasterizer using gsplat.
CUDA-accelerated tile-based rendering with alpha blending.

This module wraps the gsplat library for efficient GPU-accelerated rasterization.
gsplat provides up to 4x less GPU memory usage and 15% faster training compared
to the original 3DGS implementation.
"""

import torch
from typing import Tuple, Optional, NamedTuple, Dict, Any
import math

try:
    from gsplat import rasterization

    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not installed. Install with: pip install gsplat")

from .camera import Camera


class RenderOutput(NamedTuple):
    """Output from rasterization."""

    rendered_image: torch.Tensor  # [3, H, W]
    radii: torch.Tensor  # [N] Gaussian radii in pixels
    viewspace_points: torch.Tensor  # [N, 2] Points in screen space (for gradients)
    visibility_filter: torch.Tensor  # [N] Boolean mask of visible Gaussians
    meta: Optional[Dict[str, Any]] = None  # Additional metadata from gsplat


class GaussianRasterizer:
    """
    CUDA-accelerated differentiable Gaussian rasterizer using gsplat.

    gsplat provides efficient tile-based rasterization with automatic
    gradient computation for all Gaussian parameters.
    """

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
    ):
        """
        Args:
            tile_size: Size of screen tiles for binning (default: 16)
            bg_color: Background color (RGB, 0-1)
            near_plane: Near clipping plane
            far_plane: Far clipping plane
            eps2d: Epsilon for 2D covariance regularization
            packed: Whether to use packed mode for sparse gradients
            absgrad: Whether to compute absolute gradients for densification
            rasterize_mode: "classic" or "antialiased"
        """
        if not GSPLAT_AVAILABLE:
            raise RuntimeError(
                "gsplat is not installed. Install with: pip install gsplat"
            )

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
        means3D: torch.Tensor,  # [N, 3]
        scales: torch.Tensor,  # [N, 3] (log-scale, will be exp'd)
        rotations: torch.Tensor,  # [N, 4] quaternions
        colors: torch.Tensor,  # [N, 3] or [N, K, 3] for SH
        opacities: torch.Tensor,  # [N, 1] or [N]
        camera: Camera,
        sh_degree: Optional[int] = None,
    ) -> RenderOutput:
        """
        Render Gaussians to image using gsplat.

        Args:
            means3D: Gaussian centers in world space [N, 3]
            scales: Log-scale values [N, 3] (will be exp'd internally)
            rotations: Quaternions [N, 4]
            colors: RGB colors [N, 3] or SH coefficients [N, K, 3]
            opacities: Opacity values [N, 1] or [N] in [0, 1]
            camera: Camera object with intrinsics and extrinsics
            sh_degree: Active SH degree for view-dependent coloring (None for RGB)

        Returns:
            RenderOutput with rendered image and auxiliary data
        """
        device = means3D.device
        N = means3D.shape[0]
        H, W = camera.image_height, camera.image_width

        # Prepare inputs for gsplat
        # gsplat expects scales to be activated (not log-scale)
        activated_scales = torch.exp(scales)  # [N, 3]

        # gsplat expects opacities as [N], not [N, 1]
        if opacities.dim() == 2:
            opacities = opacities.squeeze(-1)  # [N]

        # Ensure opacities are in valid range
        opacities = torch.clamp(opacities, min=0.0, max=1.0)

        # Build view matrix [1, 4, 4] - gsplat expects batched input
        viewmat = camera.world_view_transform.unsqueeze(0)  # [1, 4, 4]

        # Build camera intrinsic matrix [1, 3, 3]
        K = torch.tensor(
            [
                [camera.focal_x, 0, camera.image_width / 2],
                [0, camera.focal_y, camera.image_height / 2],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)  # [1, 3, 3]

        # Background color tensor
        backgrounds = torch.tensor(
            [self.bg_color], dtype=torch.float32, device=device
        )  # [1, 3]

        # Handle colors - gsplat can handle both RGB and SH
        if colors.dim() == 2:
            # RGB colors [N, 3]
            colors_input = colors
            sh_degree_input = None
        else:
            # SH coefficients [N, K, 3]
            colors_input = colors
            sh_degree_input = sh_degree

        # Call gsplat rasterization
        render_colors, render_alphas, meta = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4] - gsplat normalizes internally
            scales=activated_scales,  # [N, 3]
            opacities=opacities,  # [N]
            colors=colors_input,  # [N, 3] or [N, K, 3]
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
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

        # render_colors: [1, H, W, 3] -> [3, H, W]
        rendered_image = render_colors[0].permute(2, 0, 1)  # [3, H, W]

        # Extract auxiliary outputs from meta
        radii = meta.get("radii", torch.zeros(N, dtype=torch.int32, device=device))
        if radii.dim() > 1:
            radii = radii.squeeze(0)  # Remove batch dimension if present

        # means2d for gradient computation
        means2d = meta.get("means2d", torch.zeros(N, 2, device=device))
        if means2d.dim() > 2:
            means2d = means2d.squeeze(0)  # Remove batch dimension if present

        # Visibility filter: Gaussians with radius > 0 are visible
        visibility_filter = radii > 0

        return RenderOutput(
            rendered_image=rendered_image,
            radii=radii,
            viewspace_points=means2d,
            visibility_filter=visibility_filter,
            meta=meta,
        )


def render(
    gaussians: "GaussianModel",
    camera: Camera,
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scaling_modifier: float = 1.0,
    **kwargs,
) -> RenderOutput:
    """
    Convenience function to render Gaussians using gsplat.

    Args:
        gaussians: GaussianModel instance
        camera: Camera to render from
        bg_color: Background color (RGB, 0-1)
        scaling_modifier: Scale factor for Gaussian sizes
        **kwargs: Additional arguments passed to GaussianRasterizer

    Returns:
        RenderOutput with rendered image and auxiliary data
    """
    rasterizer = GaussianRasterizer(bg_color=bg_color, **kwargs)

    # Get view direction for each Gaussian (for view-dependent colors)
    viewdirs = camera.get_view_direction(gaussians.xyz)

    # Get colors (potentially view-dependent via SH)
    colors = gaussians.get_colors(viewdirs)

    # Apply scaling modifier to log-scales
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
    """
    Render Gaussians using spherical harmonics for view-dependent appearance.

    This function passes the SH coefficients directly to gsplat, which handles
    the view-dependent color computation internally (more efficient than
    computing colors on the CPU/Python side).

    Args:
        gaussians: GaussianModel instance with SH features
        camera: Camera to render from
        bg_color: Background color (RGB, 0-1)
        scaling_modifier: Scale factor for Gaussian sizes
        **kwargs: Additional arguments passed to GaussianRasterizer

    Returns:
        RenderOutput with rendered image and auxiliary data
    """
    rasterizer = GaussianRasterizer(bg_color=bg_color, **kwargs)

    # Get SH features directly (gsplat will handle the evaluation)
    sh_features = gaussians.features  # [N, K, 3]

    # Apply scaling modifier to log-scales
    scales = gaussians._scaling
    if scaling_modifier != 1.0:
        scales = scales + math.log(scaling_modifier)

    return rasterizer.forward(
        means3D=gaussians.xyz,
        scales=scales,
        rotations=gaussians._rotation,
        colors=sh_features,  # Pass SH coefficients
        opacities=gaussians.opacity,
        camera=camera,
        sh_degree=gaussians.active_sh_degree,
    )
