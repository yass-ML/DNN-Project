"""
Evaluation Script for FaceNet.

Implements LFW-style verification protocol:
- Generate embeddings for image pairs
- Compute squared L2 distance
- Find optimal threshold via cross-validation
- Report accuracy, ROC-AUC, and EER
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import KFold

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import InceptionResNetV1
from src.dataset import LFWPairsDataset, FaceDataset, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FaceNet model")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to evaluation data directory"
    )
    parser.add_argument(
        "--pairs_file",
        type=str,
        default=None,
        help="Path to pairs file for LFW-style evaluation",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--n_folds", type=int, default=10, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )

    return parser.parse_args()


def load_model(
    model_path: str, embedding_dim: int, device: torch.device
) -> InceptionResNetV1:
    """Load model from checkpoint."""
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


@torch.no_grad()
def compute_embeddings(
    model: InceptionResNetV1, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute embeddings for all images in dataloader."""
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Computing embeddings"):
        images = images.to(device)
        embeddings = model(images)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.vstack(all_embeddings), np.concatenate(all_labels)


@torch.no_grad()
def compute_pair_embeddings(
    model: InceptionResNetV1, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute embeddings for image pairs."""
    embeddings1 = []
    embeddings2 = []
    labels = []

    for img1, img2, same in tqdm(dataloader, desc="Computing pair embeddings"):
        img1 = img1.to(device)
        img2 = img2.to(device)

        emb1 = model(img1)
        emb2 = model(img2)

        embeddings1.append(emb1.cpu().numpy())
        embeddings2.append(emb2.cpu().numpy())
        labels.append(same.numpy())

    if len(embeddings1) == 0:
        raise ValueError(
            "No pairs were loaded. Check that your pairs file and data directory are correct."
        )

    return np.vstack(embeddings1), np.vstack(embeddings2), np.concatenate(labels)


def compute_squared_l2_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute squared L2 distance between embeddings."""
    diff = emb1 - emb2
    return np.sum(diff**2, axis=1)


def find_optimal_threshold(
    distances: np.ndarray, labels: np.ndarray, num_thresholds: int = 1000
) -> Tuple[float, float]:
    """Find optimal threshold that maximizes accuracy."""
    thresholds = np.linspace(0, 4, num_thresholds)

    best_acc = 0
    best_threshold = 0

    for threshold in thresholds:
        predictions = distances < threshold
        acc = accuracy_score(labels, predictions)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc


def evaluate_with_cross_validation(
    distances: np.ndarray, labels: np.ndarray, n_folds: int = 10
) -> dict:
    """Evaluate using k-fold cross-validation."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies = []
    thresholds = []

    for train_idx, test_idx in kfold.split(distances):
        # Find threshold on training fold
        train_distances = distances[train_idx]
        train_labels = labels[train_idx]
        threshold, _ = find_optimal_threshold(train_distances, train_labels)

        # Evaluate on test fold
        test_distances = distances[test_idx]
        test_labels = labels[test_idx]
        predictions = test_distances < threshold
        acc = accuracy_score(test_labels, predictions)

        accuracies.append(acc)
        thresholds.append(threshold)

    return {
        "accuracy": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "threshold": np.mean(thresholds),
        "threshold_std": np.std(thresholds),
        "fold_accuracies": accuracies,
    }


def compute_roc_metrics(distances: np.ndarray, labels: np.ndarray) -> dict:
    """Compute ROC curve and related metrics."""
    # For ROC, higher score = more similar, so we negate distance
    scores = -distances

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Find Equal Error Rate (EER)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # Find TAR at specific FAR values
    tar_at_far = {}
    for target_far in [0.001, 0.01, 0.1]:
        idx = np.searchsorted(fpr, target_far)
        if idx < len(tpr):
            tar_at_far[f"TAR@FAR={target_far}"] = tpr[idx]
        else:
            tar_at_far[f"TAR@FAR={target_far}"] = tpr[-1]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "eer": eer,
        **tar_at_far,
    }


def evaluate_verification(
    model: InceptionResNetV1,
    dataloader: DataLoader,
    device: torch.device,
    n_folds: int = 10,
) -> dict:
    """Full verification evaluation pipeline."""
    print("\nRunning verification evaluation...")

    # Compute embeddings for pairs
    emb1, emb2, labels = compute_pair_embeddings(model, dataloader, device)

    # Compute distances
    distances = compute_squared_l2_distance(emb1, emb2)

    print(f"\nDistance statistics:")
    print(f"  Same person mean distance: {distances[labels == 1].mean():.4f}")
    print(f"  Diff person mean distance: {distances[labels == 0].mean():.4f}")

    # Cross-validation evaluation
    cv_results = evaluate_with_cross_validation(distances, labels, n_folds)

    print(f"\nCross-validation results ({n_folds}-fold):")
    print(
        f"  Accuracy: {cv_results['accuracy'] * 100:.2f}% ± {cv_results['accuracy_std'] * 100:.2f}%"
    )
    print(
        f"  Threshold: {cv_results['threshold']:.4f} ± {cv_results['threshold_std']:.4f}"
    )

    # ROC metrics
    roc_results = compute_roc_metrics(distances, labels)

    print(f"\nROC metrics:")
    print(f"  AUC: {roc_results['roc_auc']:.4f}")
    print(f"  EER: {roc_results['eer'] * 100:.2f}%")
    print(f"  TAR@FAR=0.001: {roc_results['TAR@FAR=0.001'] * 100:.2f}%")
    print(f"  TAR@FAR=0.01: {roc_results['TAR@FAR=0.01'] * 100:.2f}%")

    return {**cv_results, **roc_results, "distances": distances, "labels": labels}


def evaluate_identification(
    model: InceptionResNetV1,
    gallery_loader: DataLoader,
    probe_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate identification (1:N matching) accuracy."""
    print("\nRunning identification evaluation...")

    # Compute gallery embeddings
    print("Computing gallery embeddings...")
    gallery_emb, gallery_labels = compute_embeddings(model, gallery_loader, device)

    # Compute probe embeddings
    print("Computing probe embeddings...")
    probe_emb, probe_labels = compute_embeddings(model, probe_loader, device)

    # Compute distances between all probes and gallery
    # Shape: (num_probes, num_gallery)
    distances = np.zeros((len(probe_emb), len(gallery_emb)))
    for i, probe in enumerate(probe_emb):
        distances[i] = np.sum((gallery_emb - probe) ** 2, axis=1)

    # Rank-1 identification: closest match
    closest_idx = np.argmin(distances, axis=1)
    predictions = gallery_labels[closest_idx]
    rank1_acc = accuracy_score(probe_labels, predictions)

    # Rank-k identification
    rank_k_accs = {}
    for k in [1, 5, 10, 20]:
        if k > len(gallery_emb):
            continue
        top_k_idx = np.argsort(distances, axis=1)[:, :k]
        top_k_labels = gallery_labels[top_k_idx]
        correct = np.any(top_k_labels == probe_labels[:, np.newaxis], axis=1)
        rank_k_accs[f"rank_{k}"] = np.mean(correct)

    print(f"\nIdentification results:")
    print(f"  Rank-1: {rank1_acc * 100:.2f}%")
    for k, acc in rank_k_accs.items():
        print(f"  {k}: {acc * 100:.2f}%")

    return {"rank1": rank1_acc, **rank_k_accs}


def save_results(results: dict, output_path: str):
    """Save evaluation results to file."""
    import json

    # Filter out non-serializable items
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            if len(v) < 100:  # Only save small arrays
                serializable[k] = v.tolist()
        elif isinstance(v, (int, float, str, list)):
            serializable[k] = v

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    args = parse_args()

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, args.embedding_dim, device)

    transform = get_transforms("eval")

    if args.pairs_file:
        # LFW-style pair verification
        dataset = LFWPairsDataset(args.data_dir, args.pairs_file, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        results = evaluate_verification(model, dataloader, device, args.n_folds)
    else:
        # General evaluation on dataset
        dataset = FaceDataset(args.data_dir, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Compute all embeddings
        embeddings, labels = compute_embeddings(model, dataloader, device)

        print(f"\nComputed {len(embeddings)} embeddings")
        print(f"Number of identities: {len(np.unique(labels))}")

        # Verify L2 normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Embedding L2 norm: {norms.mean():.4f} ± {norms.std():.6f}")

        results = {
            "num_embeddings": len(embeddings),
            "num_identities": len(np.unique(labels)),
        }

    # Save results
    save_results(results, output_dir / "evaluation_results.json")


if __name__ == "__main__":
    main()
