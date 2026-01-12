"""
Mathematical utilities for 3D Gaussian Splatting.
Includes spherical harmonics, quaternion operations, covariance computation, and SSIM loss.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ============================================================================
# Spherical Harmonics
# ============================================================================

# SH constants for degree 0-3
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics at given directions.
    
    Args:
        deg: Maximum SH degree (0-3)
        sh: SH coefficients [N, C, 3] where C = (deg+1)^2
        dirs: Normalized directions [N, 3]
    
    Returns:
        colors: RGB colors [N, 3]
    """
    assert deg >= 0 and deg <= 3
    assert sh.shape[1] >= (deg + 1) ** 2
    
    result = C0 * sh[:, 0]
    
    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
        
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = result + \
                C2[0] * xy * sh[:, 4] + \
                C2[1] * yz * sh[:, 5] + \
                C2[2] * (2.0 * zz - xx - yy) * sh[:, 6] + \
                C2[3] * xz * sh[:, 7] + \
                C2[4] * (xx - yy) * sh[:, 8]
            
            if deg > 2:
                result = result + \
                    C3[0] * y * (3 * xx - yy) * sh[:, 9] + \
                    C3[1] * xy * z * sh[:, 10] + \
                    C3[2] * y * (4 * zz - xx - yy) * sh[:, 11] + \
                    C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12] + \
                    C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + \
                    C3[5] * z * (xx - yy) * sh[:, 14] + \
                    C3[6] * x * (xx - 3 * yy) * sh[:, 15]
    
    return result


def sh_degree_to_components(deg: int) -> int:
    """Return number of SH components for given degree."""
    return (deg + 1) ** 2


# ============================================================================
# Quaternion Operations
# ============================================================================

def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions to unit length. Input: [..., 4]"""
    return F.normalize(q, p=2, dim=-1)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        q: Quaternions [N, 4] in (w, x, y, z) format
    
    Returns:
        R: Rotation matrices [N, 3, 3]
    """
    q = normalize_quaternion(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.
    
    Args:
        R: Rotation matrices [N, 3, 3]
    
    Returns:
        q: Quaternions [N, 4] in (w, x, y, z) format
    """
    batch_size = R.shape[0]
    q = torch.zeros((batch_size, 4), device=R.device, dtype=R.dtype)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    s = torch.sqrt(torch.clamp(trace + 1, min=1e-10)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s
    
    return normalize_quaternion(q)


# ============================================================================
# Covariance Computation
# ============================================================================

def build_scaling_rotation(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Build scale-rotation matrix L such that covariance = L @ L.T
    
    Args:
        scaling: Log-scale values [N, 3]
        rotation: Quaternions [N, 4]
    
    Returns:
        L: Scale-rotation matrices [N, 3, 3]
    """
    L = torch.zeros((scaling.shape[0], 3, 3), device=scaling.device, dtype=scaling.dtype)
    R = quaternion_to_rotation_matrix(rotation)
    
    # Apply exponential to get actual scale
    s = torch.exp(scaling)
    
    # L = R @ S where S is diagonal scale matrix
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    
    L = R @ L
    return L


def build_covariance_3d(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Build 3D covariance matrices from scaling and rotation.
    
    Args:
        scaling: Log-scale values [N, 3]
        rotation: Quaternions [N, 4]
    
    Returns:
        cov3d: 3D covariance matrices [N, 3, 3]
    """
    L = build_scaling_rotation(scaling, rotation)
    cov3d = L @ L.transpose(-1, -2)
    return cov3d


def compute_cov2d(
    cov3d: torch.Tensor,
    viewmatrix: torch.Tensor,
    points_cam: torch.Tensor,
    focal_x: float,
    focal_y: float,
    tan_fovx: float,
    tan_fovy: float
) -> torch.Tensor:
    """
    Project 3D covariance to 2D screen space.
    
    Args:
        cov3d: 3D covariance matrices [N, 3, 3]
        viewmatrix: Camera view matrix [4, 4]
        points_cam: Points in camera space [N, 3]
        focal_x, focal_y: Focal lengths in pixels
        tan_fovx, tan_fovy: Tangent of half FoV
    
    Returns:
        cov2d: 2D covariance matrices [N, 2, 2]
    """
    N = cov3d.shape[0]
    
    # Limit the projected Gaussians to the visible frustum
    t = points_cam
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[:, 0] / t[:, 2]
    tytz = t[:, 1] / t[:, 2]
    
    tx = torch.clamp(txtz, -limx, limx) * t[:, 2]
    ty = torch.clamp(tytz, -limy, limy) * t[:, 2]
    tz = t[:, 2]
    
    # Jacobian of the projection
    J = torch.zeros((N, 2, 3), device=cov3d.device, dtype=cov3d.dtype)
    J[:, 0, 0] = focal_x / tz
    J[:, 0, 2] = -focal_x * tx / (tz * tz)
    J[:, 1, 1] = focal_y / tz
    J[:, 1, 2] = -focal_y * ty / (tz * tz)
    
    # Transform covariance to camera space: W @ cov3d @ W.T
    W = viewmatrix[:3, :3].unsqueeze(0)  # [1, 3, 3]
    cov_cam = W @ cov3d @ W.transpose(-1, -2)
    
    # Project to 2D: J @ cov_cam @ J.T
    cov2d = J @ cov_cam @ J.transpose(-1, -2)
    
    # Add low-pass filter for anti-aliasing
    cov2d[:, 0, 0] = cov2d[:, 0, 0] + 0.3
    cov2d[:, 1, 1] = cov2d[:, 1, 1] + 0.3
    
    return cov2d


# ============================================================================
# Loss Functions
# ============================================================================

def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple L1 loss."""
    return torch.abs(pred - target).mean()


def gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords = coords - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        img1, img2: Images [B, C, H, W]
        window_size: Size of Gaussian window
        size_average: Return mean SSIM if True
    
    Returns:
        SSIM value(s)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    device = img1.device
    channel = img1.shape[1]
    
    # Create Gaussian window
    window = gaussian_window(window_size, 1.5, device)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.repeat(channel, 1, 1, 1)
    
    padding = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=[1, 2, 3])


def dssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute D-SSIM loss = (1 - SSIM) / 2"""
    return (1 - ssim(pred, target)) / 2


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float = 0.2
) -> torch.Tensor:
    """Combined L1 + D-SSIM loss."""
    return (1 - lambda_dssim) * l1_loss(pred, target) + lambda_dssim * dssim_loss(pred, target)


# ============================================================================
# Inverse Sigmoid for Initialization
# ============================================================================

def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Inverse sigmoid function for opacity initialization."""
    return torch.log(x / (1 - x))


# ============================================================================
# Point Cloud Utilities
# ============================================================================

def random_point_colors(n: int, device: torch.device) -> torch.Tensor:
    """Generate random initial colors (as SH DC component)."""
    # RGB in [0, 1] range, convert to SH DC = (color - 0.5) / C0
    rgb = torch.rand(n, 3, device=device) * 0.5 + 0.25  # Clamp to avoid extremes
    return (rgb - 0.5) / C0


def points_to_sh_dc(colors: torch.Tensor) -> torch.Tensor:
    """Convert RGB colors [N, 3] to SH DC coefficients [N, 1, 3]."""
    return ((colors - 0.5) / C0).unsqueeze(1)


def sh_dc_to_colors(sh_dc: torch.Tensor) -> torch.Tensor:
    """Convert SH DC coefficients [N, 1, 3] to RGB colors [N, 3]."""
    return sh_dc.squeeze(1) * C0 + 0.5
