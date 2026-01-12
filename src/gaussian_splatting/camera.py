"""
Camera models for 3D Gaussian Splatting.
Handles intrinsics, extrinsics, and projection.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CameraInfo:
    """Raw camera information from COLMAP or other sources."""
    uid: int
    R: np.ndarray  # [3, 3] rotation matrix (world to camera)
    T: np.ndarray  # [3] translation vector
    FovX: float
    FovY: float
    image: np.ndarray  # [H, W, 3] RGB image
    image_path: str
    image_name: str
    width: int
    height: int


class Camera:
    """
    Camera class for rendering.
    Stores all camera parameters needed for Gaussian Splatting.
    """
    
    def __init__(
        self,
        uid: int,
        R: np.ndarray,
        T: np.ndarray,
        FovX: float,
        FovY: float,
        image: torch.Tensor,
        image_name: str,
        width: int,
        height: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.uid = uid
        self.image_name = image_name
        self.device = device
        
        # Store image as tensor [3, H, W]
        self.original_image = image.to(device)
        self.image_width = width
        self.image_height = height
        
        # Store FoV
        self.FovX = FovX
        self.FovY = FovY
        
        # Compute derived values
        self.zfar = 100.0
        self.znear = 0.01
        
        # Compute world-to-camera transform
        self.R = torch.tensor(R, dtype=torch.float32, device=device)
        self.T = torch.tensor(T, dtype=torch.float32, device=device)
        
        # View matrix (world to camera)
        self.world_view_transform = self._get_world_to_view_matrix()
        
        # Projection matrix
        self.projection_matrix = self._get_projection_matrix()
        
        # Full projection (view @ proj)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        
        # Camera center in world coordinates
        self.camera_center = self._get_camera_center()
        
        # Focal length in pixels
        self.focal_x = self.image_width / (2 * np.tan(FovX / 2))
        self.focal_y = self.image_height / (2 * np.tan(FovY / 2))
        
        # Tangent of half FoV
        self.tan_fovx = np.tan(FovX / 2)
        self.tan_fovy = np.tan(FovY / 2)
    
    def _get_world_to_view_matrix(self) -> torch.Tensor:
        """Compute 4x4 world-to-camera transformation matrix."""
        Rt = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        return Rt
    
    def _get_projection_matrix(self) -> torch.Tensor:
        """Compute OpenGL-style projection matrix."""
        znear = self.znear
        zfar = self.zfar
        
        tanHalfFovY = np.tan(self.FovY / 2)
        tanHalfFovX = np.tan(self.FovX / 2)
        
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        
        P = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        
        P[0, 0] = 2 * znear / (right - left)
        P[1, 1] = 2 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -2 * zfar * znear / (zfar - znear)
        P[3, 2] = -1.0
        
        return P
    
    def _get_camera_center(self) -> torch.Tensor:
        """Compute camera center in world coordinates."""
        # Camera center = -R^T @ T
        return -self.R.T @ self.T
    
    def get_view_direction(self, points: torch.Tensor) -> torch.Tensor:
        """
        Get normalized view direction from camera to points.
        
        Args:
            points: World coordinates [N, 3]
        
        Returns:
            directions: Normalized vectors [N, 3]
        """
        directions = points - self.camera_center.unsqueeze(0)
        return torch.nn.functional.normalize(directions, dim=-1)


def focal2fov(focal: float, pixels: int) -> float:
    """Convert focal length to field of view."""
    return 2 * np.arctan(pixels / (2 * focal))


def fov2focal(fov: float, pixels: int) -> float:
    """Convert field of view to focal length."""
    return pixels / (2 * np.tan(fov / 2))


def load_camera_from_info(info: CameraInfo, device: torch.device) -> Camera:
    """Create Camera object from CameraInfo."""
    # Convert image to tensor [3, H, W]
    if info.image is not None:
        image = torch.from_numpy(info.image).float().permute(2, 0, 1) / 255.0
    else:
        image = torch.zeros(3, info.height, info.width)
    
    return Camera(
        uid=info.uid,
        R=info.R,
        T=info.T,
        FovX=info.FovX,
        FovY=info.FovY,
        image=image,
        image_name=info.image_name,
        width=info.width,
        height=info.height,
        device=device
    )
