"""
Model Service for FaceNet Backend.

Handles model loading and inference for the API.
"""

from pathlib import Path
from typing import Optional, Union
import threading

import numpy as np
from PIL import Image
import torch

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference import FaceNetInference
from src.inference_transfer import FaceNetTransferInference


def is_transfer_learning_checkpoint(model_path: str) -> bool:
    """
    Check if a checkpoint is from transfer learning (facenet_pytorch).

    Transfer learning checkpoints have keys starting with 'conv2d_1a'
    while custom model checkpoints have keys starting with 'stem'.
    """
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Check for facenet_pytorch style keys
        keys = list(state_dict.keys())
        if keys and any(k.startswith("conv2d_1a") for k in keys):
            return True
        return False
    except Exception:
        return False


class ModelService:
    """
    Singleton service for FaceNet model inference.

    Thread-safe model loading and inference.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model: Optional[Union[FaceNetInference, FaceNetTransferInference]] = None
        self.model_path: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = 1.0
        self.is_transfer_learning = False
        self._initialized = True

    def load_model(self, model_path: str, threshold: float = 1.0) -> bool:
        """
        Load model from checkpoint.

        Automatically detects if the checkpoint is from transfer learning
        (facenet_pytorch) or the custom model implementation.

        Args:
            model_path: Path to model checkpoint
            threshold: Distance threshold for same person

        Returns:
            True if successful, False otherwise
        """
        try:
            # Detect model type
            self.is_transfer_learning = is_transfer_learning_checkpoint(model_path)

            if self.is_transfer_learning:
                print(f"Detected transfer learning model: {model_path}")
                # Count classes from scia_images folder
                scia_path = Path(__file__).parent.parent.parent / "data" / "scia_images"
                if scia_path.exists():
                    num_classes = len(
                        [
                            d
                            for d in scia_path.iterdir()
                            if d.is_dir() and not d.name.startswith(".")
                        ]
                    )
                else:
                    num_classes = 64  # Default fallback

                self.model = FaceNetTransferInference(
                    model_path=model_path,
                    num_classes=num_classes,
                    device=self.device,
                    threshold=threshold,
                )
            else:
                print(f"Detected custom model: {model_path}")
                self.model = FaceNetInference(
                    model_path=model_path, device=self.device, threshold=threshold
                )

            self.model_path = model_path
            self.threshold = threshold
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def get_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Generate embedding for an image."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.get_embedding(image)

    def compare_faces(
        self,
        image1: Union[str, Image.Image, np.ndarray],
        image2: Union[str, Image.Image, np.ndarray],
    ) -> dict:
        """Compare two face images."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.compare_faces(image1, image2)

    def set_threshold(self, threshold: float):
        """Update the distance threshold."""
        self.threshold = threshold
        if self.model:
            self.model.threshold = threshold

    def get_status(self) -> dict:
        """Get service status."""
        return {
            "model_loaded": self.is_loaded(),
            "model_path": self.model_path,
            "device": self.device,
            "threshold": self.threshold,
            "model_type": "transfer_learning"
            if self.is_transfer_learning
            else "custom",
        }

    def find_top_matches(
        self,
        query_image: Union[str, Image.Image, np.ndarray],
        gallery_dir: str,
        top_n: int = 5,
    ) -> list:
        """
        Find top N matching identities from a gallery directory.

        Args:
            query_image: Query face image
            gallery_dir: Path to directory containing identity folders
            top_n: Number of top matches to return

        Returns:
            List of dicts with identity info, distance, similarity, and image path
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        gallery_path = Path(gallery_dir)
        if not gallery_path.exists():
            raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")

        # Get query embedding
        query_embedding = self.get_embedding(query_image)

        # Collect all identities and their embeddings
        results = []

        for identity_dir in gallery_path.iterdir():
            if not identity_dir.is_dir() or identity_dir.name.startswith("."):
                continue

            identity_name = identity_dir.name

            # Get all images for this identity
            image_files = (
                list(identity_dir.glob("*.jpg"))
                + list(identity_dir.glob("*.jpeg"))
                + list(identity_dir.glob("*.png"))
            )

            if not image_files:
                continue

            # Compute embeddings for all images of this identity
            identity_embeddings = []
            valid_image_paths = []

            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    emb = self.get_embedding(img)
                    identity_embeddings.append(emb)
                    valid_image_paths.append(str(img_path))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            if not identity_embeddings:
                continue

            # Average embedding for this identity
            avg_embedding = np.mean(identity_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(
                avg_embedding
            )  # Re-normalize

            # Compute distance and similarity
            diff = query_embedding - avg_embedding
            distance = float(np.sum(diff**2))
            similarity = float(np.dot(query_embedding, avg_embedding))

            # Pick best representative image (closest to average)
            best_img_idx = 0
            best_img_dist = float("inf")
            for idx, emb in enumerate(identity_embeddings):
                dist = float(np.sum((emb - avg_embedding) ** 2))
                if dist < best_img_dist:
                    best_img_dist = dist
                    best_img_idx = idx

            results.append(
                {
                    "identity": identity_name,
                    "distance": distance,
                    "similarity": similarity,
                    "is_match": distance < self.threshold,
                    "confidence": max(0.0, 1.0 - distance / self.threshold)
                    if distance < self.threshold
                    else 0.0,
                    "image_path": valid_image_paths[best_img_idx],
                    "num_images": len(identity_embeddings),
                }
            )

        # Sort by distance (ascending) and return top N
        results.sort(key=lambda x: x["distance"])
        return results[:top_n]


# Global instance
model_service = ModelService()


def get_model_service() -> ModelService:
    """Get the model service singleton."""
    return model_service
