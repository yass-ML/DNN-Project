"""
High-level pipeline for 3D Gaussian Splatting.
Orchestrates scene loading, training, and export.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch

from .gaussian_model import GaussianModel
from .scene import Scene
from .trainer import Trainer, TrainingConfig
from .export import export_ply


@dataclass
class GaussianConfig:
    """Configuration for Gaussian Splatting pipeline."""

    # Training iterations
    iterations: int = 30000

    # Learning rates
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    # Loss
    lambda_dssim: float = 0.2

    # Densification
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.000002
    densification_interval: int = 100

    # SH degree
    sh_degree: int = 3

    # Resolution scale (lower for faster training)
    resolution_scale: float = 1.0

    # Background color
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Output
    output_path: str = "./output"
    save_interval: int = 5000


class GaussianSplattingPipeline:
    """
    Main pipeline for 3D Gaussian Splatting.
    Handles the full workflow from images to Unity-compatible export.
    """

    def __init__(self, config: GaussianConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.model: Optional[GaussianModel] = None
        self.scene: Optional[Scene] = None
        self.trainer: Optional[Trainer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Gaussian Splatting Pipeline initialized")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

    def load_scene(self, source_path: str, images_folder: str = "images"):
        """
        Load scene data from disk.

        Args:
            source_path: Path to scene data (COLMAP output or image folder)
            images_folder: Subfolder containing images
        """
        print(f"\nLoading scene from {source_path}...")

        self.scene = Scene(
            source_path=source_path,
            images_folder=images_folder,
            resolution_scale=self.config.resolution_scale,
        )

        if self.scene.point_cloud is None:
            raise ValueError("Failed to load point cloud from scene")

        print(f"Scene loaded:")
        print(f"  - Training images: {len(self.scene.train_cameras)}")
        print(f"  - Point cloud: {len(self.scene.point_cloud.points)} points")

    def initialize_model(self):
        """Initialize Gaussian model from scene point cloud."""
        if self.scene is None:
            raise ValueError("Scene not loaded. Call load_scene() first.")

        print(f"\nInitializing Gaussian model (SH degree {self.config.sh_degree})...")

        self.model = GaussianModel(sh_degree=self.config.sh_degree)
        self.model.create_from_pcd(self.scene.point_cloud)
        self.model = self.model.to(self.device)

        print(f"Model initialized with {self.model.num_gaussians} Gaussians")

    def train(self) -> GaussianModel:
        """
        Run training loop.

        Returns:
            Trained GaussianModel
        """
        if self.scene is None:
            raise ValueError("Scene not loaded. Call load_scene() first.")
        if self.model is None:
            self.initialize_model()

        # Create training config
        train_config = TrainingConfig(
            iterations=self.config.iterations,
            position_lr_init=self.config.position_lr_init,
            position_lr_final=self.config.position_lr_final,
            feature_lr=self.config.feature_lr,
            opacity_lr=self.config.opacity_lr,
            scaling_lr=self.config.scaling_lr,
            rotation_lr=self.config.rotation_lr,
            lambda_dssim=self.config.lambda_dssim,
            densify_from_iter=self.config.densify_from_iter,
            densify_until_iter=self.config.densify_until_iter,
            densify_grad_threshold=self.config.densify_grad_threshold,
            densification_interval=self.config.densification_interval,
            bg_color=self.config.bg_color,
            save_interval=self.config.save_interval,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            scene=self.scene,
            config=train_config,
            output_path=self.config.output_path,
            device=self.device,
        )

        # Train
        self.model = self.trainer.train()
        return self.model

    def export(self, output_path: str, include_sh: bool = True):
        """
        Export trained model to PLY file.

        Args:
            output_path: Path to save PLY file
            include_sh: Include spherical harmonics coefficients
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print(f"\nExporting model to {output_path}...")
        export_ply(
            self.model,
            output_path,
            include_sh=include_sh,
            max_sh_degree=self.config.sh_degree,
        )

        print(f"\n{'=' * 50}")
        print("Export complete!")
        print(f"{'=' * 50}")
        print(f"\nTo use in Unity:")
        print(f"1. Install UnityGaussianSplatting package")
        print(f"   (https://github.com/aras-p/UnityGaussianSplatting)")
        print(f"2. Drag {output_path} into your Unity project")
        print(f"3. Create a GaussianSplatting object and assign the asset")

    def run(self, source_path: str, output_path: str, images_folder: str = "images"):
        """
        Run full pipeline: load, train, export.

        Args:
            source_path: Path to scene data
            output_path: Path to save PLY file
            images_folder: Subfolder containing images
        """
        self.load_scene(source_path, images_folder)
        self.train()
        self.export(output_path)
