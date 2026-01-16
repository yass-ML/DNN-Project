"""
Inference Utilities for FaceNet.

Provides utilities for:
- Loading trained model
- Generating embeddings for single images
- Comparing two face images
- Batch processing
"""

import os
import sys
from pathlib import Path
from typing import Union, List, Tuple

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import InceptionResNetV1


class FaceNetInference:
    """
    FaceNet inference class for generating embeddings and comparing faces.

    Args:
        model_path: Path to trained model checkpoint
        embedding_dim: Embedding dimension (default: 128)
        device: Device to use ('cuda' or 'cpu')
        threshold: Distance threshold for same/different person decision
    """

    def __init__(
        self,
        model_path: str,
        embedding_dim: int = 128,
        device: str = "cuda",
        threshold: float = 1.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.embedding_dim = embedding_dim

        # Load model
        self.model = self._load_model(model_path, embedding_dim)

        # Setup transforms
        self.transform = T.Compose(
            [
                T.Resize((160, 160)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_model(self, model_path: str, embedding_dim: int) -> InceptionResNetV1:
        """Load model from checkpoint."""
        model = InceptionResNetV1(embedding_dim=embedding_dim)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")

        model = model.to(self.device)
        model.eval()

        return model

    def _preprocess(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return self.transform(image)

    @torch.no_grad()
    def get_embedding(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            L2-normalized embedding of shape (embedding_dim,)
        """
        # Preprocess
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        # Forward pass
        embedding = self.model(tensor)

        return embedding.cpu().numpy().squeeze()

    @torch.no_grad()
    def get_embeddings_batch(
        self, images: List[Union[str, Path, Image.Image]]
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of images.

        Args:
            images: List of image paths, PIL Images, or numpy arrays

        Returns:
            L2-normalized embeddings of shape (N, embedding_dim)
        """
        tensors = [self._preprocess(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        embeddings = self.model(batch)

        return embeddings.cpu().numpy()

    def compute_distance(
        self, emb1: np.ndarray, emb2: np.ndarray, squared: bool = True
    ) -> float:
        """
        Compute L2 distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding
            squared: If True, return squared distance

        Returns:
            L2 distance (or squared L2 distance)
        """
        diff = emb1 - emb2
        distance = np.sum(diff**2)

        if not squared:
            distance = np.sqrt(distance)

        return float(distance)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Note: For L2-normalized embeddings, cosine similarity = dot product.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity in range [-1, 1]
        """
        return float(np.dot(emb1, emb2))

    def compare_faces(
        self,
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image],
        return_embeddings: bool = False,
    ) -> dict:
        """
        Compare two face images.

        Args:
            image1: First face image
            image2: Second face image
            return_embeddings: If True, include embeddings in result

        Returns:
            Dictionary with comparison results:
            - distance: Squared L2 distance
            - similarity: Cosine similarity
            - is_same_person: Boolean decision
            - confidence: Confidence score (0-1)
        """
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)

        distance = self.compute_distance(emb1, emb2, squared=True)
        similarity = self.compute_similarity(emb1, emb2)

        is_same = distance < self.threshold

        # Confidence: how far from threshold (normalized)
        confidence = (
            1.0 - min(distance / (2 * self.threshold), 1.0)
            if is_same
            else min(distance / (2 * self.threshold), 1.0)
        )

        result = {
            "distance": distance,
            "similarity": similarity,
            "is_same_person": is_same,
            "confidence": confidence,
            "threshold": self.threshold,
        }

        if return_embeddings:
            result["embedding1"] = emb1
            result["embedding2"] = emb2

        return result

    def is_same_person(
        self,
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image],
    ) -> bool:
        """
        Simple boolean check if two images are of the same person.

        Args:
            image1: First face image
            image2: Second face image

        Returns:
            True if same person, False otherwise
        """
        return self.compare_faces(image1, image2)["is_same_person"]

    def find_matches(
        self,
        query_image: Union[str, Path, Image.Image],
        gallery_images: List[Union[str, Path, Image.Image]],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find top-k matches for a query image in a gallery.

        Args:
            query_image: Query face image
            gallery_images: List of gallery images
            top_k: Number of top matches to return

        Returns:
            List of (index, distance) tuples for top-k matches
        """
        query_emb = self.get_embedding(query_image)
        gallery_embs = self.get_embeddings_batch(gallery_images)

        # Compute distances
        distances = np.sum((gallery_embs - query_emb) ** 2, axis=1)

        # Get top-k
        top_k_idx = np.argsort(distances)[:top_k]

        return [(int(idx), float(distances[idx])) for idx in top_k_idx]

    def verify_embedding_normalization(self, embedding: np.ndarray) -> bool:
        """Verify that embedding has unit L2 norm."""
        norm = np.linalg.norm(embedding)
        return np.abs(norm - 1.0) < 1e-5


def load_model(
    model_path: str, embedding_dim: int = 128, device: str = "cuda"
) -> FaceNetInference:
    """
    Factory function to create FaceNetInference instance.

    Args:
        model_path: Path to trained model
        embedding_dim: Embedding dimension
        device: Device to use

    Returns:
        FaceNetInference instance
    """
    return FaceNetInference(model_path, embedding_dim, device)


def get_embedding(model_path: str, image_path: str) -> np.ndarray:
    """
    Convenience function to get embedding for a single image.

    Args:
        model_path: Path to trained model
        image_path: Path to face image

    Returns:
        128-dimensional L2-normalized embedding
    """
    inference = FaceNetInference(model_path)
    return inference.get_embedding(image_path)


def compare_images(model_path: str, image1_path: str, image2_path: str) -> dict:
    """
    Convenience function to compare two face images.

    Args:
        model_path: Path to trained model
        image1_path: Path to first face image
        image2_path: Path to second face image

    Returns:
        Comparison results dictionary
    """
    inference = FaceNetInference(model_path)
    return inference.compare_faces(image1_path, image2_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FaceNet Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--image1", type=str, required=True, help="Path to first image")
    parser.add_argument(
        "--image2", type=str, default=None, help="Path to second image (for comparison)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Distance threshold for same person",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Create inference object
    inference = FaceNetInference(
        args.model_path, device=args.device, threshold=args.threshold
    )

    if args.image2:
        # Compare two images
        print(f"\nComparing images:")
        print(f"  Image 1: {args.image1}")
        print(f"  Image 2: {args.image2}")

        result = inference.compare_faces(args.image1, args.image2)

        print(f"\nResults:")
        print(f"  Squared L2 Distance: {result['distance']:.4f}")
        print(f"  Cosine Similarity: {result['similarity']:.4f}")
        print(f"  Same Person: {result['is_same_person']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Threshold: {result['threshold']:.4f}")
    else:
        # Get embedding for single image
        print(f"\nGenerating embedding for: {args.image1}")

        embedding = inference.get_embedding(args.image1)

        print(f"\nEmbedding:")
        print(f"  Shape: {embedding.shape}")
        print(f"  L2 Norm: {np.linalg.norm(embedding):.6f}")
        print(f"  Min/Max: {embedding.min():.4f} / {embedding.max():.4f}")
        print(f"  First 10 values: {embedding[:10]}")
