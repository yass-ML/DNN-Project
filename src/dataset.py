"""
Dataset and Data Loading for FaceNet Training.

Implements identity-based sampling to facilitate online triplet mining.
Uses P identities x K samples per identity per batch.
"""

import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import torchvision.transforms as T


class FaceDataset(Dataset):
    """
    Face dataset for FaceNet training.

    Expected directory structure:
        root/
        ├── person_001/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── person_002/
        │   └── ...

    Args:
        root: Root directory containing identity folders
        transform: Image transforms to apply
        min_samples_per_identity: Minimum samples required per identity
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        min_samples_per_identity: int = 2,
    ):
        self.root = Path(root)
        self.transform = transform or self._default_transform()
        self.min_samples_per_identity = min_samples_per_identity

        # Build dataset
        self.samples: List[Tuple[str, int]] = []  # (image_path, label)
        self.identity_to_indices: Dict[int, List[int]] = {}
        self.identity_to_name: Dict[int, str] = {}

        self._scan_directory()

    def _default_transform(self) -> Callable:
        """Default transform: resize, normalize to [-1, 1]."""
        return T.Compose(
            [
                T.Resize((160, 160)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _scan_directory(self):
        """Scan root directory and build sample list."""
        if not self.root.exists():
            raise ValueError(f"Dataset root does not exist: {self.root}")

        identity_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])

        current_label = 0
        sample_idx = 0

        for identity_dir in identity_dirs:
            # Get all image files for this identity
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
            images = [
                f
                for f in identity_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            # Skip identities with too few samples
            if len(images) < self.min_samples_per_identity:
                continue

            # Add identity mapping
            self.identity_to_name[current_label] = identity_dir.name
            self.identity_to_indices[current_label] = []

            # Add all samples for this identity
            for img_path in sorted(images):
                self.samples.append((str(img_path), current_label))
                self.identity_to_indices[current_label].append(sample_idx)
                sample_idx += 1

            current_label += 1

        print(
            f"Loaded {len(self.samples)} samples from {len(self.identity_to_indices)} identities"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample by index."""
        img_path, label = self.samples[idx]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_num_identities(self) -> int:
        """Return number of unique identities."""
        return len(self.identity_to_indices)

    def get_identity_samples(self, label: int) -> List[int]:
        """Get all sample indices for a given identity."""
        return self.identity_to_indices.get(label, [])


class IdentitySampler(Sampler):
    """
    Batch sampler for identity-based sampling.

    Samples P identities and K samples per identity for each batch.
    This ensures each batch has enough samples per identity for triplet mining.

    Args:
        dataset: FaceDataset instance
        p_identities: Number of identities per batch
        k_samples: Number of samples per identity
        num_iterations: Number of batches per epoch
    """

    def __init__(
        self,
        dataset: FaceDataset,
        p_identities: int = 32,
        k_samples: int = 4,
        num_iterations: Optional[int] = None,
    ):
        self.dataset = dataset
        self.p_identities = p_identities
        self.k_samples = k_samples

        # Filter identities that have at least k_samples
        self.valid_identities = [
            label
            for label, indices in dataset.identity_to_indices.items()
            if len(indices) >= k_samples
        ]

        if len(self.valid_identities) < p_identities:
            raise ValueError(
                f"Not enough identities with >= {k_samples} samples. "
                f"Found {len(self.valid_identities)}, need {p_identities}"
            )

        if num_iterations is None:
            # Default: enough iterations to see all identities once per epoch
            self.num_iterations = len(self.valid_identities) // p_identities
        else:
            self.num_iterations = num_iterations

        print(
            f"IdentitySampler: {len(self.valid_identities)} valid identities, "
            f"{self.num_iterations} iterations per epoch, "
            f"batch size = {p_identities * k_samples}"
        )

    def __iter__(self):
        """Generate batches of indices."""
        for _ in range(self.num_iterations):
            batch_indices = []

            # Randomly select P identities
            selected_identities = random.sample(
                self.valid_identities, self.p_identities
            )

            for identity in selected_identities:
                # Get all indices for this identity
                identity_indices = self.dataset.identity_to_indices[identity]

                # Randomly select K samples (with replacement if necessary)
                if len(identity_indices) >= self.k_samples:
                    selected = random.sample(identity_indices, self.k_samples)
                else:
                    selected = random.choices(identity_indices, k=self.k_samples)

                batch_indices.extend(selected)

            yield batch_indices

    def __len__(self) -> int:
        return self.num_iterations


def get_transforms(mode: str = "train", image_size: int = 160) -> Callable:
    """
    Get transforms for training or evaluation.

    Args:
        mode: 'train' or 'eval'
        image_size: Target image size

    Returns:
        Transform composition
    """
    if mode == "train":
        return T.Compose(
            [
                T.Resize((image_size + 20, image_size + 20)),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )


def get_dataloader(
    root: str,
    mode: str = "train",
    p_identities: int = 32,
    k_samples: int = 4,
    num_workers: int = 4,
    num_iterations: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader with identity-based sampling.

    Args:
        root: Dataset root directory
        mode: 'train' or 'eval'
        p_identities: Identities per batch
        k_samples: Samples per identity
        num_workers: DataLoader workers
        num_iterations: Batches per epoch

    Returns:
        DataLoader instance
    """
    transform = get_transforms(mode)
    dataset = FaceDataset(root, transform=transform)

    if mode == "train":
        sampler = IdentitySampler(
            dataset,
            p_identities=p_identities,
            k_samples=k_samples,
            num_iterations=num_iterations,
        )
        return DataLoader(
            dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=p_identities * k_samples,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


class LFWPairsDataset(Dataset):
    """
    Dataset for LFW-style pair verification.

    Expected format (pairs.txt):
        # Matched pairs (same person)
        person_name\timg1_num\timg2_num
        # Mismatched pairs (different people)
        person1_name\timg1_num\tperson2_name\timg2_num

    Args:
        root: Root directory containing identity folders
        pairs_file: Path to pairs.txt file
        transform: Image transforms
    """

    def __init__(
        self, root: str, pairs_file: str, transform: Optional[Callable] = None
    ):
        self.root = Path(root)
        self.pairs_file = Path(pairs_file)
        self.transform = transform or get_transforms("eval")

        self.pairs: List[Tuple[str, str, int]] = []  # (img1, img2, same_person)
        self._load_pairs()

    def _load_pairs(self):
        """Load pairs from file."""
        with open(self.pairs_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")

                if len(parts) == 3:
                    # Same person: name, img1, img2
                    name, n1, n2 = parts
                    img1 = self.root / name / f"{name}_{int(n1):04d}.jpg"
                    img2 = self.root / name / f"{name}_{int(n2):04d}.jpg"
                    self.pairs.append((str(img1), str(img2), 1))

                elif len(parts) == 4:
                    # Different people: name1, img1, name2, img2
                    name1, n1, name2, n2 = parts
                    img1 = self.root / name1 / f"{name1}_{int(n1):04d}.jpg"
                    img2 = self.root / name2 / f"{name2}_{int(n2):04d}.jpg"
                    self.pairs.append((str(img1), str(img2), 0))

        print(f"Loaded {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get a pair of images and label."""
        img1_path, img2_path, same = self.pairs[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, same


if __name__ == "__main__":
    # Create a dummy dataset for testing
    import tempfile
    import numpy as np

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create dummy identities
        for i in range(10):  # 10 identities
            identity_dir = root / f"person_{i:03d}"
            identity_dir.mkdir()

            # Create dummy images
            for j in range(5):  # 5 images per identity
                img = Image.fromarray(
                    np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
                )
                img.save(identity_dir / f"img_{j}.jpg")

        print("Created dummy dataset")

        # Test FaceDataset
        dataset = FaceDataset(root)
        print(f"Dataset length: {len(dataset)}")
        print(f"Number of identities: {dataset.get_num_identities()}")

        # Get a sample
        img, label = dataset[0]
        print(f"Sample shape: {img.shape}, label: {label}")

        # Test IdentitySampler
        sampler = IdentitySampler(dataset, p_identities=4, k_samples=3)

        # Get one batch
        for batch_indices in sampler:
            print(f"Batch size: {len(batch_indices)}")
            batch_labels = [dataset[i][1] for i in batch_indices]
            print(f"Batch labels: {batch_labels}")
            break

        # Test DataLoader
        loader = get_dataloader(
            str(root),
            mode="train",
            p_identities=4,
            k_samples=3,
            num_workers=0,
            num_iterations=5,
        )

        for batch_idx, (images, labels) in enumerate(loader):
            print(
                f"Batch {batch_idx}: images shape = {images.shape}, labels = {labels}"
            )
            if batch_idx >= 2:
                break

        print("\n✓ Dataset tests passed!")
