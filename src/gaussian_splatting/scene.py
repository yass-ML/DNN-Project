import os
import json
from typing import List, NamedTuple

class Camera(NamedTuple):
    image_path: str
    uid: int
    R: 'np.ndarray'
    T: 'np.ndarray'
    FovY: float
    FovX: float
    image: 'np.ndarray'
    # TODO: Add projection matrix, image width/height, etc.

class Scene:
    def __init__(self, source_path: str, images: str = "images"):
        """
        Initialize scene, load COLMAP data or other SfM outputs.
        """
        self.source_path = source_path
        self.loaded_iter = None
        self.cameras: List[Camera] = []
        self.point_cloud = None
        
        self._load_scene_info()

    def _load_scene_info(self):
        """
        Load camera intrinsics, extrinsics, and sparse point cloud.
        """
        # TODO: parsing logic for standard formats (COLMAP, etc.)
        # This usually involves reading cameras.txt, images.txt, points3D.txt
        print(f"Loading scene from {self.source_path}...")
        self.cameras = [] # Placeholder
        self.point_cloud = [] # Placeholder

    def get_train_cameras(self):
        return self.cameras

    def get_test_cameras(self):
        # Stub: split logic usually goes here
        return []
