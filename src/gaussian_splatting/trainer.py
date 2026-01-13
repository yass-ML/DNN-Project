"""
Training pipeline for 3D Gaussian Splatting.
Handles optimization loop, densification, and learning rate scheduling.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from typing import List, Optional, Tuple
import random
from tqdm import tqdm
import os

from .gaussian_model import GaussianModel
from .scene import Scene
from .camera import Camera, load_camera_from_info
from .rasterizer import render
from .utils import combined_loss, l1_loss, dssim_loss


class TrainingConfig:
    """Configuration for training."""

    def __init__(
        self,
        iterations: int = 30000,
        position_lr_init: float = 0.00016,
        position_lr_final: float = 0.0000016,
        position_lr_delay_mult: float = 0.01,
        position_lr_max_steps: int = 30000,
        feature_lr: float = 0.0025,
        opacity_lr: float = 0.05,
        scaling_lr: float = 0.005,
        rotation_lr: float = 0.001,
        lambda_dssim: float = 0.2,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15000,
        densify_grad_threshold: float = 0.000002,
        densification_interval: int = 100,
        opacity_reset_interval: int = 3000,
        sh_degree_increase_interval: int = 1000,
        min_opacity: float = 0.0001,
        prune_opacity_from_iter: int = 3000,
        test_interval: int = 1000,
        save_interval: int = 5000,
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.iterations = iterations
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.position_lr_max_steps = position_lr_max_steps
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr
        self.lambda_dssim = lambda_dssim
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densify_grad_threshold = densify_grad_threshold
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.sh_degree_increase_interval = sh_degree_increase_interval
        self.min_opacity = min_opacity
        self.prune_opacity_from_iter = prune_opacity_from_iter
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.bg_color = bg_color


class Trainer:
    """
    Training manager for Gaussian Splatting.
    """

    def __init__(
        self,
        model: GaussianModel,
        scene: Scene,
        config: TrainingConfig,
        output_path: str = "./output",
        device: torch.device = None,
    ):
        self.model = model
        self.scene = scene
        self.config = config
        self.output_path = output_path
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move model to device
        self.model = self.model.to(self.device)

        # Load cameras
        self.train_cameras = [
            load_camera_from_info(cam_info, self.device)
            for cam_info in scene.get_train_cameras()
        ]
        self.test_cameras = [
            load_camera_from_info(cam_info, self.device)
            for cam_info in scene.get_test_cameras()
        ]

        print(f"Training with {len(self.train_cameras)} cameras")

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Setup optimizer
        self.optimizer = None
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup optimizer with per-parameter learning rates."""
        param_groups = self.model.get_optimizer_param_groups(
            position_lr=self.config.position_lr_init,
            feature_lr=self.config.feature_lr,
            opacity_lr=self.config.opacity_lr,
            scaling_lr=self.config.scaling_lr,
            rotation_lr=self.config.rotation_lr,
        )
        self.optimizer = optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def _update_learning_rate(self, iteration: int):
        """Update learning rate with exponential decay for position."""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # Exponential decay
                lr = self._get_expon_lr(
                    iteration,
                    self.config.position_lr_init,
                    self.config.position_lr_final,
                    self.config.position_lr_max_steps,
                )
                param_group["lr"] = lr * self.model.spatial_lr_scale

    def _get_expon_lr(
        self, step: int, lr_init: float, lr_final: float, max_steps: int
    ) -> float:
        """Compute exponentially decaying learning rate."""
        if step > max_steps:
            return lr_final
        t = step / max_steps
        return lr_init * (lr_final / lr_init) ** t

    def train(self) -> GaussianModel:
        """
        Run the training loop.

        Returns:
            Trained GaussianModel
        """
        if len(self.train_cameras) == 0:
            raise ValueError("No training cameras available!")

        print(f"\nStarting training for {self.config.iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Initial Gaussians: {self.model.num_gaussians}")

        progress = tqdm(range(1, self.config.iterations + 1), desc="Training")

        for iteration in progress:
            self._train_step(iteration, progress)

        print(f"\nTraining complete!")
        print(f"Final Gaussians: {self.model.num_gaussians}")

        return self.model

    def _train_step(self, iteration: int, progress: tqdm):
        """Execute single training step."""
        self.optimizer.zero_grad()

        # Update learning rate
        self._update_learning_rate(iteration)

        # Increase SH degree periodically
        if iteration % self.config.sh_degree_increase_interval == 0:
            self.model.oneup_sh_degree()

        # Select random camera
        camera = random.choice(self.train_cameras)

        # Render
        output = render(self.model, camera, bg_color=self.config.bg_color)

        rendered_image = output.rendered_image
        gt_image = camera.original_image

        # Compute loss
        loss = combined_loss(
            rendered_image.unsqueeze(0), gt_image.unsqueeze(0), self.config.lambda_dssim
        )

        # Backward
        loss.backward()

        # Densification and pruning
        with torch.no_grad():
            if iteration < self.config.densify_until_iter:
                # Track gradients for densification
                absgrad = None
                if output.meta is not None:
                    means2d = output.meta.get("means2d", None)
                    if means2d is not None and hasattr(means2d, "absgrad"):
                        absgrad = means2d.absgrad

                # print(f"absgrad is none: {absgrad is None}")
                self.model.add_densification_stats(
                    output.viewspace_points, output.radii.float(), absgrad=absgrad
                )

                # Densification
                if (
                    iteration >= self.config.densify_from_iter
                    and iteration % self.config.densification_interval == 0
                ):
                    num_before = self.model.num_gaussians

                    # Don't prune by opacity until prune_opacity_from_iter
                    effective_min_opacity = (
                        self.config.min_opacity
                        if iteration >= self.config.prune_opacity_from_iter
                        else 0.0  # Disable opacity pruning early in training
                    )
                    self.model.densify_and_prune(
                        grad_threshold=self.config.densify_grad_threshold,
                        min_opacity=effective_min_opacity,
                        max_screen_size=0,  # Disable screen size pruning
                    )

                    num_after = self.model.num_gaussians
                    if iteration % 500 == 0:
                        print(f"\n[Densify] {num_before} -> {num_after} Gaussians")

                    # Recreate optimizer for new parameters
                    self._setup_optimizer()

                # Opacity reset - skip if Gaussian count is too low
                if iteration % self.config.opacity_reset_interval == 0:
                    if self.model.num_gaussians > 5000:  # Only reset if we have enough
                        self.model.reset_opacity()

        # Optimizer step
        self.optimizer.step()

        # Update progress bar
        if iteration % 100 == 0:
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "gaussians": self.model.num_gaussians,
                }
            )

        # Test/Save periodically
        if iteration % self.config.test_interval == 0:
            self._test_step(iteration)

        if iteration % self.config.save_interval == 0:
            self._save_checkpoint(iteration)

    def _test_step(self, iteration: int):
        """Evaluate on test cameras."""
        if len(self.test_cameras) == 0:
            return

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for camera in self.test_cameras:
                output = render(self.model, camera, bg_color=self.config.bg_color)
                loss = l1_loss(output.rendered_image, camera.original_image)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_cameras)
        print(f"\n[Iter {iteration}] Test L1 Loss: {avg_loss:.4f}")

        self.model.train()

    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_path, f"checkpoint_{iteration}.pt")
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"\nSaved checkpoint to {checkpoint_path}")

    def render_image(
        self, camera_index: int = 0, use_test: bool = False
    ) -> torch.Tensor:
        """
        Render an image from a specific camera.

        Args:
            camera_index: Index of camera to use
            use_test: Use test cameras instead of train

        Returns:
            Rendered image [3, H, W]
        """
        cameras = self.test_cameras if use_test else self.train_cameras
        if camera_index >= len(cameras):
            camera_index = 0

        camera = cameras[camera_index]

        self.model.eval()
        with torch.no_grad():
            output = render(self.model, camera, bg_color=self.config.bg_color)
        self.model.train()

        return output.rendered_image
