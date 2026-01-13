"""
3D Gaussian Model for Gaussian Splatting.
Stores and manages all Gaussian parameters with proper activations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from .utils import (
    build_covariance_3d,
    inverse_sigmoid,
    sh_degree_to_components,
    eval_sh,
    C0,
    quaternion_to_rotation_matrix,
)
from .scene import PointCloud


class GaussianModel(nn.Module):
    """
    3D Gaussian representation for a scene.
    Each Gaussian has: position, color (SH), opacity, scale, and rotation.
    """

    def __init__(self, sh_degree: int = 3):
        """
        Initialize Gaussian model.

        Args:
            sh_degree: Maximum spherical harmonics degree (0-3)
        """
        super().__init__()
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0  # Start with just DC, increase during training

        # Learnable parameters (initialized in create_from_pcd)
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(0, (sh_degree + 1) ** 2 - 1, 3))
        self._scaling = nn.Parameter(torch.empty(0, 3))
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._opacity = nn.Parameter(torch.empty(0, 1))

        # For densification (registered as buffers to move with model)
        self.register_buffer("xyz_gradient_accum", None)
        self.register_buffer("denom", None)
        self.register_buffer("max_radii2D", None)

        # Spatial learning rate reference (for position LR scheduling)
        self.spatial_lr_scale = 1.0

    @property
    def num_gaussians(self) -> int:
        """Return number of Gaussians."""
        return self._xyz.shape[0]

    @property
    def device(self) -> torch.device:
        """Return device of parameters."""
        return self._xyz.device

    def create_from_pcd(self, pcd: PointCloud, spatial_lr_scale: float = 1.0):
        """
        Initialize Gaussians from a point cloud.

        Args:
            pcd: PointCloud with positions and colors
            spatial_lr_scale: Scale for spatial learning rate
        """
        self.spatial_lr_scale = spatial_lr_scale

        points = torch.tensor(pcd.points, dtype=torch.float32)
        colors = (
            torch.tensor(pcd.colors, dtype=torch.float32) / 255.0
        )  # Normalize to [0, 1]

        n_points = points.shape[0]
        device = points.device

        print(f"Initializing {n_points} Gaussians from point cloud")

        # Position
        self._xyz = nn.Parameter(points)

        # Color as SH DC coefficient
        # RGB in [0, 1] -> SH DC = (rgb - 0.5) / C0
        fused_color = (colors - 0.5) / C0
        self._features_dc = nn.Parameter(fused_color.unsqueeze(1))  # [N, 1, 3]

        # Higher-order SH (initialized to 0)
        num_rest = (self.max_sh_degree + 1) ** 2 - 1
        self._features_rest = nn.Parameter(torch.zeros(n_points, num_rest, 3))

        # Scale: estimate from local point density
        dist = self._compute_nearest_neighbor_dist(points)
        dist = torch.clamp(dist, min=1e-7)
        scales = torch.log(dist).unsqueeze(-1).repeat(1, 3)
        self._scaling = nn.Parameter(scales)

        # Rotation: identity quaternion (w, x, y, z) = (1, 0, 0, 0)
        rots = torch.zeros(n_points, 4)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots)

        # Opacity: start at 0.1 (inverse sigmoid)
        opacities = inverse_sigmoid(0.1 * torch.ones(n_points, 1))
        self._opacity = nn.Parameter(opacities)

        # Initialize gradient tracking for densification (as buffers to move with model)
        self.register_buffer(
            "xyz_gradient_accum", torch.zeros(n_points, 1, device=device)
        )
        self.register_buffer("denom", torch.zeros(n_points, 1, device=device))
        self.register_buffer("max_radii2D", torch.zeros(n_points, device=device))

    def _compute_nearest_neighbor_dist(self, points: torch.Tensor) -> torch.Tensor:
        """Compute distance to nearest neighbor for each point."""
        # Use simple O(n^2) for small point clouds, sample for large
        n = points.shape[0]
        if n > 10000:
            # Sample subset for efficiency
            indices = torch.randperm(n)[: min(n, 5000)]
            sampled = points[indices]
            dists = torch.cdist(points, sampled)
            # Set diagonal-ish to inf to exclude self
            k = min(3, sampled.shape[0])
            min_dists, _ = dists.topk(k, dim=1, largest=False)
            return min_dists[:, -1]  # Use k-th nearest
        else:
            dists = torch.cdist(points, points)
            # Set diagonal to inf
            dists.fill_diagonal_(float("inf"))
            min_dists, _ = dists.min(dim=1)
            return min_dists

    # =========================================================================
    # Property accessors with activations
    # =========================================================================

    @property
    def xyz(self) -> torch.Tensor:
        """Return Gaussian positions [N, 3]."""
        return self._xyz

    @property
    def features(self) -> torch.Tensor:
        """Return all SH features [N, (deg+1)^2, 3]."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def opacity(self) -> torch.Tensor:
        """Return activated opacity [N, 1] in [0, 1]."""
        return torch.sigmoid(self._opacity)

    @property
    def scaling(self) -> torch.Tensor:
        """Return activated scale [N, 3] (exponential)."""
        return torch.exp(self._scaling)

    @property
    def rotation(self) -> torch.Tensor:
        """Return normalized rotation quaternions [N, 4]."""
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    def get_covariance(self) -> torch.Tensor:
        """Compute 3D covariance matrices [N, 3, 3]."""
        return build_covariance_3d(self._scaling, self._rotation)

    def get_colors(self, viewdirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get RGB colors, optionally view-dependent.

        Args:
            viewdirs: Normalized view directions [N, 3]

        Returns:
            colors: RGB colors [N, 3] in [0, 1]
        """
        if viewdirs is None or self.active_sh_degree == 0:
            # Just return DC component
            return torch.clamp(self._features_dc.squeeze(1) * C0 + 0.5, 0, 1)

        # Evaluate spherical harmonics
        sh_features = self.features[:, : sh_degree_to_components(self.active_sh_degree)]
        colors = eval_sh(self.active_sh_degree, sh_features, viewdirs)
        colors = torch.clamp(colors + 0.5, 0, 1)
        return colors

    # =========================================================================
    # Densification and Pruning
    # =========================================================================

    def add_densification_stats(
        self,
        viewspace_point_tensor: torch.Tensor,
        radii: torch.Tensor,
        absgrad: torch.Tensor = None,
    ):
        """
        Accumulate gradient statistics for densification.

        Args:
            viewspace_point_tensor: Projected 2D points with gradients
            radii: 2D radii of Gaussians (can be [N] or [N, 2] or [1, N] or [1, N, 2])
            absgrad: Optional pre-computed absolute gradients from gsplat
                     (when using gsplat with absgrad=True)
        """
        # Handle radii shape - gsplat may return [N, 2] for x/y radii
        # We need a single value per Gaussian for max_radii2D tracking
        if radii.dim() == 3:
            radii = radii.squeeze(0)  # [1, N, 2] -> [N, 2]
        if radii.dim() == 2:
            radii = radii.max(dim=-1).values  # [N, 2] -> [N], take max of x/y radii

        # Use absgrad if provided (gsplat), otherwise compute from .grad
        if absgrad is not None:
            # gsplat provides absolute gradients directly
            # Handle potential batch dimension (1, N, 2) -> (N, 2)
            if absgrad.dim() == 3:
                absgrad = absgrad.squeeze(0)
            grad_norm = absgrad.norm(dim=-1, keepdim=True)
        elif viewspace_point_tensor.grad is not None:
            # Legacy: compute from gradient tensor
            grad = viewspace_point_tensor.grad[:, :2]
            grad_norm = torch.norm(grad, dim=-1, keepdim=True)
        else:
            return  # No gradients available

        self.xyz_gradient_accum += grad_norm
        self.denom += 1
        self.max_radii2D = torch.max(self.max_radii2D, radii)

    def densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        extent: float = 1.0,
        max_screen_size: float = 20.0,
    ):
        """
        Densify Gaussians with high gradients and prune low-opacity ones.

        Args:
            grad_threshold: Threshold for gradient-based densification
            min_opacity: Minimum opacity to keep a Gaussian
            extent: Scene extent for scale thresholding
            max_screen_size: Maximum 2D size before pruning
        """
        if self.denom is None or self.xyz_gradient_accum is None:
            return

        # Compute average gradient
        grads = self.xyz_gradient_accum / (self.denom + 1e-7)
        grads[grads.isnan()] = 0.0

        # Find Gaussians to densify (high gradient)
        selected = grads.squeeze() >= grad_threshold

        # Split large Gaussians
        scales = self.scaling
        max_scale = scales.max(dim=-1).values
        large_mask = max_scale > extent * 0.01
        split_mask = selected & large_mask

        # Clone small Gaussians
        clone_mask = selected & ~large_mask

        # Perform densification
        self._densify_and_clone(clone_mask)
        self._densify_and_split(split_mask)

        # Prune
        prune_mask = (self.opacity < min_opacity).squeeze()
        if self.max_radii2D is not None:
            prune_mask = prune_mask | (self.max_radii2D > max_screen_size)

        self._prune(prune_mask)

        # Reset stats
        device = self.device
        n = self.num_gaussians
        self.register_buffer("xyz_gradient_accum", torch.zeros(n, 1, device=device))
        self.register_buffer("denom", torch.zeros(n, 1, device=device))
        self.register_buffer("max_radii2D", torch.zeros(n, device=device))

    def _densify_and_clone(self, mask: torch.Tensor):
        """Clone Gaussians at mask positions."""
        if not mask.any():
            return

        # Clone parameters
        new_xyz = self._xyz[mask].clone()
        new_features_dc = self._features_dc[mask].clone()
        new_features_rest = self._features_rest[mask].clone()
        new_scaling = self._scaling[mask].clone()
        new_rotation = self._rotation[mask].clone()
        new_opacity = self._opacity[mask].clone()

        # Add small offset to position
        new_xyz += torch.randn_like(new_xyz) * self.scaling[mask] * 0.5

        self._concat_params(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_scaling,
            new_rotation,
            new_opacity,
        )

    def _densify_and_split(self, mask: torch.Tensor, n_splits: int = 2):
        """Split large Gaussians at mask positions into n_splits parts."""
        if not mask.any():
            return

        n_selected = mask.sum().item()

        # Sample new positions around old ones
        stds = self.scaling[mask].repeat(n_splits, 1)
        means = self._xyz[mask].repeat(n_splits, 1)
        samples = torch.randn((n_splits * n_selected, 3), device=self.device)
        new_xyz = means + samples * stds

        # Reduce scale
        new_scaling = self._scaling[mask].repeat(n_splits, 1) - np.log(n_splits)

        # Copy other parameters
        new_features_dc = self._features_dc[mask].repeat(n_splits, 1, 1)
        new_features_rest = self._features_rest[mask].repeat(n_splits, 1, 1)
        new_rotation = self._rotation[mask].repeat(n_splits, 1)
        new_opacity = self._opacity[mask].repeat(n_splits, 1)

        self._concat_params(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_scaling,
            new_rotation,
            new_opacity,
        )

        # Remove original Gaussians that were split
        self._prune(mask)

    def _concat_params(
        self, xyz, features_dc, features_rest, scaling, rotation, opacity
    ):
        """Concatenate new parameters to existing ones."""
        self._xyz = nn.Parameter(torch.cat([self._xyz, xyz], dim=0))
        self._features_dc = nn.Parameter(
            torch.cat([self._features_dc, features_dc], dim=0)
        )
        self._features_rest = nn.Parameter(
            torch.cat([self._features_rest, features_rest], dim=0)
        )
        self._scaling = nn.Parameter(torch.cat([self._scaling, scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, rotation], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, opacity], dim=0))

        # Extend gradient tracking
        n_new = xyz.shape[0]
        device = self.device
        self.register_buffer(
            "xyz_gradient_accum",
            torch.cat(
                [self.xyz_gradient_accum, torch.zeros(n_new, 1, device=device)], dim=0
            ),
        )
        self.register_buffer(
            "denom",
            torch.cat([self.denom, torch.zeros(n_new, 1, device=device)], dim=0),
        )
        self.register_buffer(
            "max_radii2D",
            torch.cat([self.max_radii2D, torch.zeros(n_new, device=device)], dim=0),
        )

    def _prune(self, mask: torch.Tensor):
        """Remove Gaussians at mask positions."""
        keep = ~mask

        self._xyz = nn.Parameter(self._xyz[keep])
        self._features_dc = nn.Parameter(self._features_dc[keep])
        self._features_rest = nn.Parameter(self._features_rest[keep])
        self._scaling = nn.Parameter(self._scaling[keep])
        self._rotation = nn.Parameter(self._rotation[keep])
        self._opacity = nn.Parameter(self._opacity[keep])

        if self.xyz_gradient_accum is not None:
            self.register_buffer("xyz_gradient_accum", self.xyz_gradient_accum[keep])
            self.register_buffer("denom", self.denom[keep])
            self.register_buffer("max_radii2D", self.max_radii2D[keep])

    def reset_opacity(self):
        """Reset opacity to initial value (used periodically during training)."""
        new_opacity = inverse_sigmoid(
            torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01)
        )
        self._opacity = nn.Parameter(new_opacity)

    def oneup_sh_degree(self):
        """Increase active SH degree by 1."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(f"SH degree increased to {self.active_sh_degree}")

    # =========================================================================
    # Optimizer setup
    # =========================================================================

    def get_optimizer_param_groups(
        self,
        position_lr: float = 0.00016,
        feature_lr: float = 0.0025,
        opacity_lr: float = 0.05,
        scaling_lr: float = 0.005,
        rotation_lr: float = 0.001,
    ) -> list:
        """
        Get parameter groups for optimizer with per-parameter learning rates.

        Returns:
            List of parameter group dicts for optimizer
        """
        return [
            {
                "params": [self._xyz],
                "lr": position_lr * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self._features_dc], "lr": feature_lr, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": rotation_lr, "name": "rotation"},
        ]
