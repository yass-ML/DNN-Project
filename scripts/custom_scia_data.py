"""
Custom SCIA Dataset Processing Script.

This script:
1. Loads single images from identity folders in ./data/scia_images/
2. Applies data augmentation to create N versions of each image
3. Generates embeddings using the trained FaceNet model
4. Calculates and saves the optimal threshold for each person
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import InceptionResNetV1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process custom SCIA dataset with augmentation"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/scia_images/",
        help="Path to SCIA images directory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--n_augmentations",
        type=int,
        default=10,
        help="Number of augmented versions to create per image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/scia_processed/",
        help="Directory to save processed data and thresholds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--save_augmented",
        action="store_true",
        help="Save augmented images to disk",
        default=False,
    )
    return parser.parse_args()


def get_augmentation_transforms(image_size: int = 160) -> T.Compose:
    """
    Get augmentation transforms for creating variations of the image.

    These augmentations simulate real-world variations:
    - Slight rotations
    - Brightness/contrast changes
    - Small crops and resizes
    - Horizontal flips
    """
    return T.Compose(
        [
            T.Resize((image_size + 40, image_size + 40)),
            T.RandomCrop(image_size + 20),
            T.RandomRotation(degrees=10),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_eval_transform(image_size: int = 160) -> T.Compose:
    """Get standard evaluation transform (no augmentation)."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_model(
    model_path: str, embedding_dim: int, device: torch.device
) -> InceptionResNetV1:
    """Load the trained FaceNet model."""
    model = InceptionResNetV1(embedding_dim=embedding_dim)

    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    return model


def scan_dataset(data_dir: Path) -> Dict[str, Path]:
    """
    Scan the SCIA dataset directory.

    Returns:
        Dictionary mapping identity name to image path
    """
    identities = {}

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    for identity_dir in sorted(data_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        # Find the single image in the identity folder
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        images = [
            f
            for f in identity_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if len(images) == 0:
            print(f"Warning: No images found in {identity_dir}")
            continue
        elif len(images) > 1:
            print(f"Warning: Multiple images found in {identity_dir}, using first one")

        identities[identity_dir.name] = images[0]

    print(f"Found {len(identities)} identities in {data_dir}")
    return identities


def augment_image(
    image: Image.Image,
    n_augmentations: int,
    aug_transform: T.Compose,
    eval_transform: T.Compose,
) -> List[torch.Tensor]:
    """
    Create N augmented versions of an image plus the original.

    Returns:
        List of tensors (original + n augmented versions)
    """
    augmented = []

    # Add original image (with eval transform)
    augmented.append(eval_transform(image))

    # Add augmented versions
    for _ in range(n_augmentations):
        augmented.append(aug_transform(image))

    return augmented


@torch.no_grad()
def compute_embeddings(
    model: InceptionResNetV1,
    images: List[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Compute embeddings for a list of image tensors."""
    # Stack images into a batch
    batch = torch.stack(images).to(device)

    # Compute embeddings
    embeddings = model(batch)

    return embeddings.cpu().numpy()


def compute_intra_class_distances(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise squared L2 distances within a set of embeddings.

    These represent distances between different augmented versions of the same person.
    """
    n = len(embeddings)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            diff = embeddings[i] - embeddings[j]
            dist = np.sum(diff**2)
            distances.append(dist)

    return np.array(distances)


def find_optimal_threshold(
    intra_distances: np.ndarray,
    margin: float = 0.1,
) -> Tuple[float, dict]:
    """
    Find optimal threshold for a single identity.

    The threshold should be set so that all augmented versions of the same
    person are considered the same identity (distance < threshold).

    Args:
        intra_distances: Distances between augmented versions of the same person
        margin: Safety margin above max intra-class distance

    Returns:
        threshold: Recommended threshold for this identity
        stats: Dictionary with distance statistics
    """
    stats = {
        "min_distance": float(np.min(intra_distances)),
        "max_distance": float(np.max(intra_distances)),
        "mean_distance": float(np.mean(intra_distances)),
        "std_distance": float(np.std(intra_distances)),
        "median_distance": float(np.median(intra_distances)),
    }

    # Set threshold above the maximum intra-class distance with a margin
    # This ensures all augmented versions are correctly identified as the same person
    threshold = stats["mean_distance"] * (1 + margin)

    return threshold, stats


def save_results(
    output_dir: Path,
    identity_name: str,
    embeddings: np.ndarray,
    threshold: float,
    stats: dict,
):
    """Save embeddings, threshold, and stats for an identity."""
    identity_dir = output_dir / identity_name
    identity_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(identity_dir / "embeddings.npy", embeddings)

    # Save threshold and stats to text file
    with open(identity_dir / "threshold.txt", "w") as f:
        f.write(f"Identity: {identity_name}\n")
        f.write(f"Recommended Threshold: {threshold:.6f}\n")
        f.write(f"\nDistance Statistics:\n")
        f.write(f"  Min Distance: {stats['min_distance']:.6f}\n")
        f.write(f"  Max Distance: {stats['max_distance']:.6f}\n")
        f.write(f"  Mean Distance: {stats['mean_distance']:.6f}\n")
        f.write(f"  Std Distance: {stats['std_distance']:.6f}\n")
        f.write(f"  Median Distance: {stats['median_distance']:.6f}\n")
        f.write(f"\nNumber of embeddings: {len(embeddings)}\n")

    print(f"  Saved results to {identity_dir}")


def save_global_summary(
    output_dir: Path,
    all_thresholds: Dict[str, float],
    all_stats: Dict[str, dict],
):
    """Save a global summary with recommended thresholds for all identities."""
    summary_file = output_dir / "global_thresholds.txt"

    thresholds = list(all_thresholds.values())

    with open(summary_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("GLOBAL THRESHOLD SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Number of identities: {len(all_thresholds)}\n\n")

        f.write("Recommended Global Thresholds:\n")
        f.write(f"  Conservative (min): {min(thresholds):.6f}\n")
        f.write(f"  Average: {np.mean(thresholds):.6f}\n")
        f.write(f"  Median: {np.median(thresholds):.6f}\n")
        f.write(f"  Liberal (max): {max(thresholds):.6f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Per-Identity Thresholds:\n")
        f.write("-" * 60 + "\n")

        for identity, threshold in sorted(all_thresholds.items()):
            stats = all_stats[identity]
            f.write(f"\n{identity}:\n")
            f.write(f"  Threshold: {threshold:.6f}\n")
            f.write(f"  Max intra-class dist: {stats['max_distance']:.6f}\n")
            f.write(f"  Mean intra-class dist: {stats['mean_distance']:.6f}\n")

    print(f"\nGlobal summary saved to {summary_file}")


def main():
    args = parse_args()

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, 128, device)

    # Scan dataset
    identities = scan_dataset(data_dir)

    if len(identities) == 0:
        print("No identities found. Exiting.")
        return

    # Setup transforms
    aug_transform = get_augmentation_transforms()
    eval_transform = get_eval_transform()

    # Process each identity
    all_thresholds = {}
    all_stats = {}

    print(
        f"\nProcessing {len(identities)} identities with {args.n_augmentations} augmentations each...\n"
    )

    for identity_name, image_path in tqdm(
        identities.items(), desc="Processing identities"
    ):
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Create augmented versions
        augmented_images = augment_image(
            image, args.n_augmentations, aug_transform, eval_transform
        )

        # Save augmented images if requested
        if args.save_augmented:
            aug_dir = output_dir / identity_name / "augmented"
            aug_dir.mkdir(parents=True, exist_ok=True)

            # We need to save before normalization, so re-augment for saving
            save_transform = T.Compose(
                [
                    T.Resize((160 + 40, 160 + 40)),
                    T.RandomCrop(160 + 20),
                    T.RandomRotation(degrees=15),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
                    ),
                    T.RandomPerspective(distortion_scale=0.2, p=0.5),
                    T.Resize((160, 160)),
                ]
            )

            # Save original
            image.resize((160, 160)).save(aug_dir / "original.png")

            # Save augmented
            for i in range(args.n_augmentations):
                aug_img = save_transform(image)
                aug_img.save(aug_dir / f"augmented_{i:02d}.png")

        # Compute embeddings
        embeddings = compute_embeddings(model, augmented_images, device)

        # Compute intra-class distances
        intra_distances = compute_intra_class_distances(embeddings)

        # Find optimal threshold
        threshold, stats = find_optimal_threshold(intra_distances)

        # Save results
        save_results(output_dir, identity_name, embeddings, threshold, stats)

        all_thresholds[identity_name] = threshold
        all_stats[identity_name] = stats

    # Save global summary
    save_global_summary(output_dir, all_thresholds, all_stats)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
