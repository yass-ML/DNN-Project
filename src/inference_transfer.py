"""
Inference Utilities for Transfer Learning FaceNet.

This module is designed for models trained with facenet_pytorch's
InceptionResnetV1 using transfer learning (classify=True mode).

Provides utilities for:
- Loading transfer learning model
- Generating embeddings for single images
- Comparing two face images
- Batch processing
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Union, List, Tuple

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# facenet_pytorch model
from facenet_pytorch import InceptionResnetV1


class FaceNetTransferInference:
    """
    FaceNet inference class for transfer learning models.

    Uses facenet_pytorch's InceptionResnetV1 which was fine-tuned
    with classify=True mode.

    Args:
        model_path: Path to trained model checkpoint
        num_classes: Number of identity classes the model was trained on
        device: Device to use ('cuda' or 'cpu')
        threshold: Distance threshold for same/different person decision
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int,
        device: str = "cuda",
        threshold: float = 1.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.num_classes = num_classes

        # Load model
        self.model = self._load_model(model_path, num_classes)

        # Setup transforms (same as used in training)
        self.transform = T.Compose(
            [
                T.Resize((160, 160)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_model(self, model_path: str, num_classes: int) -> InceptionResnetV1:
        """Load model from checkpoint."""
        # Create model with same architecture as training
        # classify=True adds the classification head
        model = InceptionResnetV1(
            pretrained=None,  # Don't load pretrained weights
            classify=True,
            num_classes=num_classes,
        )

        if os.path.exists(model_path):
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model path {model_path} not found.")

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

        For classification models, we extract the embedding from the
        layer before the classification head.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            L2-normalized embedding of shape (512,)
        """
        # Preprocess
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        # Temporarily disable classification to get embeddings
        original_classify = self.model.classify
        self.model.classify = False

        # Forward pass to get embeddings
        embedding = self.model(tensor)

        # Restore classification mode
        self.model.classify = original_classify

        # L2 normalize
        embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)

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
            L2-normalized embeddings of shape (N, 512)
        """
        tensors = [self._preprocess(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        # Temporarily disable classification to get embeddings
        original_classify = self.model.classify
        self.model.classify = False

        embeddings = self.model(batch)

        # Restore classification mode
        self.model.classify = original_classify

        # L2 normalize
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)

        return embeddings.cpu().numpy()

    @torch.no_grad()
    def predict_identity(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[int, float]:
        """
        Predict the identity class for an image.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            Tuple of (predicted_class_index, confidence)
        """
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)

        confidence, predicted = torch.max(probs, dim=1)

        return int(predicted.item()), float(confidence.item())

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

        # Decision based on threshold
        is_same_person = distance < self.threshold

        # Compute confidence (higher similarity / lower distance = higher confidence)
        # Map distance to confidence: 0 distance -> 1.0 confidence
        # threshold distance -> 0.5 confidence
        confidence = max(0, 1 - (distance / (2 * self.threshold)))

        result = {
            "distance": distance,
            "similarity": similarity,
            "is_same_person": is_same_person,
            "confidence": confidence,
            "threshold": self.threshold,
        }

        if return_embeddings:
            result["embedding1"] = emb1
            result["embedding2"] = emb2

        return result

    def verify(
        self,
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image],
    ) -> bool:
        """
        Simple verification: are these the same person?

        Args:
            image1: First face image
            image2: Second face image

        Returns:
            True if same person, False otherwise
        """
        result = self.compare_faces(image1, image2)
        return result["is_same_person"]


def get_num_classes_from_dataset(data_dir: str) -> int:
    """Count the number of identity folders in the dataset directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    num_classes = len([d for d in data_path.iterdir() if d.is_dir()])
    return num_classes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face verification using transfer learning model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/facenet_transfer_learning.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of identity classes (if not provided, will try to infer from data_dir)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/scia_images/",
        help="Path to training data directory (used to infer num_classes)",
    )
    parser.add_argument(
        "--image1",
        type=str,
        required=True,
        help="Path to first image",
    )
    parser.add_argument(
        "--image2",
        type=str,
        required=True,
        help="Path to second image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Distance threshold for same/different person decision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine number of classes
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        try:
            num_classes = get_num_classes_from_dataset(args.data_dir)
            print(f"Inferred {num_classes} classes from {args.data_dir}")
        except ValueError:
            raise ValueError(
                "Could not infer num_classes. Please provide --num_classes or valid --data_dir"
            )

    # Initialize inference
    inference = FaceNetTransferInference(
        model_path=args.model_path,
        num_classes=num_classes,
        device=args.device,
        threshold=args.threshold,
    )

    # Compare faces
    print(f"\nComparing faces:")
    print(f"  Image 1: {args.image1}")
    print(f"  Image 2: {args.image2}")
    print(f"  Threshold: {args.threshold}")

    result = inference.compare_faces(args.image1, args.image2)

    print(f"\nResults:")
    print(f"  Distance: {result['distance']:.4f}")
    print(f"  Similarity: {result['similarity']:.4f}")
    print(f"  Same person: {result['is_same_person']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # Also show identity predictions
    print(f"\nIdentity predictions:")
    class1, conf1 = inference.predict_identity(args.image1)
    class2, conf2 = inference.predict_identity(args.image2)
    print(f"  Image 1: Class {class1} (confidence: {conf1:.2%})")
    print(f"  Image 2: Class {class2} (confidence: {conf2:.2%})")


if __name__ == "__main__":
    main()
