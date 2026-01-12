"""
Scene management and data loading for 3D Gaussian Splatting.
Supports COLMAP output format and simple image-based loading.
"""

import os
import struct
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, NamedTuple
from collections import namedtuple
from pathlib import Path

from .camera import CameraInfo, focal2fov


class PointCloud(NamedTuple):
    """Point cloud data from SfM."""
    points: np.ndarray  # [N, 3] positions
    colors: np.ndarray  # [N, 3] RGB colors (0-255)
    normals: Optional[np.ndarray] = None  # [N, 3] normals (optional)


# ============================================================================
# COLMAP Binary Format Reading
# ============================================================================

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str):
    """Read and unpack bytes from binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(format_char_sequence, data)


def read_cameras_binary(path: str) -> dict:
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODELS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = {
                "id": camera_id,
                "model": CAMERA_MODELS[model_id].model_name,
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras


def read_images_binary(path: str) -> dict:
    """Read COLMAP images.bin file."""
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            # Skip point2D data (x, y, point3D_id)
            fid.read(24 * num_points2D)
            
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
            }
    return images


def read_points3D_binary(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.bin file."""
    points = []
    colors = []
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            # error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_length)  # Skip track data
            points.append(xyz)
            colors.append(rgb)
    return np.array(points), np.array(colors)


def read_cameras_text(path: str) -> dict:
    """Read COLMAP cameras.txt file."""
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elements = line.split()
            camera_id = int(elements[0])
            model = elements[1]
            width = int(elements[2])
            height = int(elements[3])
            params = np.array([float(x) for x in elements[4:]])
            cameras[camera_id] = {
                "id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_text(path: str) -> dict:
    """Read COLMAP images.txt file."""
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
    
    for i in range(0, len(lines), 2):
        elements = lines[i].split()
        image_id = int(elements[0])
        qvec = np.array([float(x) for x in elements[1:5]])
        tvec = np.array([float(x) for x in elements[5:8]])
        camera_id = int(elements[8])
        image_name = elements[9]
        images[image_id] = {
            "id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": image_name,
        }
    return images


def read_points3D_text(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.txt file."""
    points = []
    colors = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elements = line.split()
            xyz = np.array([float(x) for x in elements[1:4]])
            rgb = np.array([int(x) for x in elements[4:7]])
            points.append(xyz)
            colors.append(rgb)
    return np.array(points), np.array(colors)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


# ============================================================================
# Scene Class
# ============================================================================

class Scene:
    """
    Scene class that loads and manages training data.
    Supports COLMAP format or simple image-based loading.
    """
    
    def __init__(
        self,
        source_path: str,
        images_folder: str = "images",
        eval_split: float = 0.0,
        resolution_scale: float = 1.0
    ):
        """
        Initialize scene from source path.
        
        Args:
            source_path: Path to scene data (COLMAP output or image folder)
            images_folder: Name of images subfolder
            eval_split: Fraction of images to use for evaluation (0 = all train)
            resolution_scale: Scale factor for image resolution
        """
        self.source_path = source_path
        self.images_folder = images_folder
        self.resolution_scale = resolution_scale
        self.eval_split = eval_split
        
        self.train_cameras: List[CameraInfo] = []
        self.test_cameras: List[CameraInfo] = []
        self.point_cloud: Optional[PointCloud] = None
        
        self._load_scene()
    
    def _load_scene(self):
        """Load scene data, detecting format automatically."""
        sparse_path = os.path.join(self.source_path, "sparse", "0")
        
        if os.path.exists(sparse_path):
            # COLMAP format
            print(f"Loading COLMAP scene from {self.source_path}")
            self._load_colmap(sparse_path)
        else:
            # Simple image-based format (requires pre-computed cameras.json)
            cameras_json = os.path.join(self.source_path, "cameras.json")
            if os.path.exists(cameras_json):
                print(f"Loading scene from cameras.json")
                self._load_json(cameras_json)
            else:
                # Load just images - create synthetic point cloud from center
                print(f"No camera data found. Creating synthetic scene from images.")
                self._load_images_only()
    
    def _load_colmap(self, sparse_path: str):
        """Load scene from COLMAP sparse reconstruction."""
        # Try binary format first
        cameras_bin = os.path.join(sparse_path, "cameras.bin")
        images_bin = os.path.join(sparse_path, "images.bin")
        points_bin = os.path.join(sparse_path, "points3D.bin")
        
        if os.path.exists(cameras_bin):
            cameras = read_cameras_binary(cameras_bin)
            images = read_images_binary(images_bin)
            points, colors = read_points3D_binary(points_bin)
        else:
            # Fall back to text format
            cameras = read_cameras_text(os.path.join(sparse_path, "cameras.txt"))
            images = read_images_text(os.path.join(sparse_path, "images.txt"))
            points, colors = read_points3D_text(os.path.join(sparse_path, "points3D.txt"))
        
        # Create point cloud
        self.point_cloud = PointCloud(points=points, colors=colors)
        
        # Load images and create camera info
        images_path = os.path.join(self.source_path, self.images_folder)
        all_cameras = []
        
        for idx, (img_id, img_data) in enumerate(sorted(images.items())):
            cam_data = cameras[img_data["camera_id"]]
            
            # Load image
            img_path = os.path.join(images_path, img_data["name"])
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            image = np.array(Image.open(img_path))
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # Apply resolution scale
            if self.resolution_scale != 1.0:
                new_h = int(image.shape[0] * self.resolution_scale)
                new_w = int(image.shape[1] * self.resolution_scale)
                image = np.array(Image.fromarray(image).resize((new_w, new_h)))
            
            # Get camera intrinsics
            height, width = image.shape[:2]
            params = cam_data["params"]
            
            if cam_data["model"] == "PINHOLE":
                fx, fy, cx, cy = params
            elif cam_data["model"] == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1:3]
            else:
                # Default to using first param as focal
                fx = fy = params[0]
                cx, cy = width / 2, height / 2
            
            # Apply resolution scale to intrinsics
            fx *= self.resolution_scale
            fy *= self.resolution_scale
            
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            
            # Get extrinsics
            R = qvec2rotmat(img_data["qvec"])
            T = img_data["tvec"]
            
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovX=FovX,
                FovY=FovY,
                image=image,
                image_path=img_path,
                image_name=img_data["name"],
                width=width,
                height=height,
            )
            all_cameras.append(cam_info)
        
        # Split into train/test
        self._split_cameras(all_cameras)
        
        print(f"Loaded {len(self.train_cameras)} train and {len(self.test_cameras)} test cameras")
        print(f"Point cloud: {len(self.point_cloud.points)} points")
    
    def _load_json(self, json_path: str):
        """Load scene from cameras.json file."""
        import json
        with open(json_path, "r") as f:
            data = json.load(f)
        
        images_path = os.path.join(self.source_path, self.images_folder)
        all_cameras = []
        
        for idx, cam_data in enumerate(data["cameras"]):
            img_path = os.path.join(images_path, cam_data["image_name"])
            if not os.path.exists(img_path):
                continue
            
            image = np.array(Image.open(img_path))
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            height, width = image.shape[:2]
            
            cam_info = CameraInfo(
                uid=idx,
                R=np.array(cam_data["R"]),
                T=np.array(cam_data["T"]),
                FovX=cam_data["FovX"],
                FovY=cam_data["FovY"],
                image=image,
                image_path=img_path,
                image_name=cam_data["image_name"],
                width=width,
                height=height,
            )
            all_cameras.append(cam_info)
        
        # Load point cloud if available
        if "points" in data:
            points = np.array(data["points"]["xyz"])
            colors = np.array(data["points"]["rgb"])
            self.point_cloud = PointCloud(points=points, colors=colors)
        else:
            self._create_synthetic_point_cloud(all_cameras)
        
        self._split_cameras(all_cameras)
    
    def _load_images_only(self):
        """Load images without camera data - create synthetic cameras."""
        images_path = os.path.join(self.source_path, self.images_folder)
        if not os.path.exists(images_path):
            images_path = self.source_path
        
        image_files = sorted([
            f for f in os.listdir(images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if not image_files:
            raise ValueError(f"No images found in {images_path}")
        
        all_cameras = []
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(images_path, img_name)
            image = np.array(Image.open(img_path))
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            height, width = image.shape[:2]
            
            # Create synthetic camera (looking at origin, arranged in circle)
            angle = 2 * np.pi * idx / len(image_files)
            radius = 3.0
            
            # Camera position
            cam_pos = np.array([
                radius * np.cos(angle),
                0.0,
                radius * np.sin(angle)
            ])
            
            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(np.array([0, 1, 0]), forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            
            R = np.stack([right, up, forward], axis=1).T
            T = -R @ cam_pos
            
            # Default FoV
            FovX = np.pi / 3  # 60 degrees
            FovY = FovX * height / width
            
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovX=FovX,
                FovY=FovY,
                image=image,
                image_path=img_path,
                image_name=img_name,
                width=width,
                height=height,
            )
            all_cameras.append(cam_info)
        
        self._create_synthetic_point_cloud(all_cameras)
        self._split_cameras(all_cameras)
        
        print(f"Created synthetic scene with {len(all_cameras)} cameras")
        print("Warning: Without real camera poses, results will be poor.")
        print("Consider running COLMAP on your images first.")
    
    def _create_synthetic_point_cloud(self, cameras: List[CameraInfo]):
        """Create a simple synthetic point cloud."""
        # Create a grid of points around origin
        n = 10
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        z = np.linspace(-1, 1, n)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        colors = np.ones_like(points) * 128  # Gray
        
        self.point_cloud = PointCloud(points=points, colors=colors.astype(np.uint8))
    
    def _split_cameras(self, cameras: List[CameraInfo]):
        """Split cameras into train and test sets."""
        if self.eval_split <= 0:
            self.train_cameras = cameras
            self.test_cameras = []
        else:
            n_test = max(1, int(len(cameras) * self.eval_split))
            # Take every Nth camera for test
            step = len(cameras) // n_test
            test_indices = set(range(0, len(cameras), step)[:n_test])
            
            self.train_cameras = [c for i, c in enumerate(cameras) if i not in test_indices]
            self.test_cameras = [c for i, c in enumerate(cameras) if i in test_indices]
    
    def get_train_cameras(self) -> List[CameraInfo]:
        """Return training cameras."""
        return self.train_cameras
    
    def get_test_cameras(self) -> List[CameraInfo]:
        """Return test cameras."""
        return self.test_cameras
    
    def get_point_cloud(self) -> PointCloud:
        """Return initial point cloud."""
        return self.point_cloud
