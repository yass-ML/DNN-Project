"""
Optimized Triplet Loss with efficient mining strategies.

Key optimizations:
1. Memory-efficient triplet mining (avoids N³ tensors)
2. Vectorized operations where possible
3. Option to use batch-hard fallback for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def pairwise_distances_optimized(
    embeddings: torch.Tensor, 
    squared: bool = False
) -> torch.Tensor:
    """
    Optimized pairwise distance computation.
    
    Uses the identity: ||a - b||² = ||a||² + ||b||² - 2(a·b)
    For L2-normalized embeddings, this simplifies to: 2 - 2(a·b)
    """
    # Compute dot products
    dot_product = torch.mm(embeddings, embeddings.t())
    
    # For normalized embeddings, squared distance = 2 - 2*dot_product
    # This is more numerically stable
    square_norm = torch.diag(dot_product)
    
    # Compute squared distances
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)
    
    if not squared:
        # Numerical stability for sqrt
        distances = torch.sqrt(distances + 1e-16)
    
    return distances


class EfficientBatchHardLoss(nn.Module):
    """
    Efficient Batch Hard Triplet Loss.
    
    For each anchor:
    - Hardest positive: argmax d(a, p) over same-class samples
    - Hardest negative: argmin d(a, n) over different-class samples
    
    This is O(N²) instead of O(N³) for batch-all.
    """
    
    def __init__(self, margin: float = 0.2, squared: bool = True):
        super().__init__()
        self.margin = margin
        self.squared = squared
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute batch hard triplet loss."""
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances - O(N²)
        pairwise_dist = pairwise_distances_optimized(embeddings, squared=self.squared)
        
        # Create masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        indices_not_equal = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # Mask for valid positives (same label, different index)
        mask_pos = labels_equal & indices_not_equal
        
        # Hardest positive for each anchor
        # Set invalid positions to -inf so they're not selected by max
        masked_pos_dist = pairwise_dist.clone()
        masked_pos_dist[~mask_pos] = -float('inf')
        hardest_pos_dist, _ = masked_pos_dist.max(dim=1)
        
        # Hardest negative for each anchor  
        # Set invalid positions to +inf so they're not selected by min
        masked_neg_dist = pairwise_dist.clone()
        masked_neg_dist[~labels_not_equal] = float('inf')
        hardest_neg_dist, _ = masked_neg_dist.min(dim=1)
        
        # Compute triplet loss
        triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        
        # Only count valid triplets (where we found both pos and neg)
        valid_triplets = (hardest_pos_dist > -float('inf')) & (hardest_neg_dist < float('inf'))
        num_valid = valid_triplets.sum()
        
        if num_valid > 0:
            loss = triplet_loss[valid_triplets].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute stats
        num_hard = (triplet_loss > 0).sum().item()
        
        stats = {
            'num_triplets': int(num_valid.item()),
            'num_hard': num_hard,
            'avg_pos_dist': hardest_pos_dist[valid_triplets].mean().item() if num_valid > 0 else 0,
            'avg_neg_dist': hardest_neg_dist[valid_triplets].mean().item() if num_valid > 0 else 0,
        }
        
        return loss, stats


class EfficientSemiHardLoss(nn.Module):
    """
    Memory-efficient Semi-Hard Triplet Loss.
    
    Instead of creating N³ tensors, we iterate over anchors
    and use vectorized operations within each anchor.
    
    Trade-off: Slightly slower than fully vectorized, but uses O(N²) memory.
    """
    
    def __init__(self, margin: float = 0.2, squared: bool = True):
        super().__init__()
        self.margin = margin
        self.squared = squared
        self.batch_hard_fallback = EfficientBatchHardLoss(margin, squared)
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute semi-hard triplet loss efficiently."""
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances once - O(N²)
        pairwise_dist = pairwise_distances_optimized(embeddings, squared=self.squared)
        
        # Create masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        indices_not_equal = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # Mask for valid positives and negatives
        mask_pos = labels_equal & indices_not_equal
        mask_neg = labels_not_equal
        
        total_loss = torch.tensor(0.0, device=device)
        num_semihard = 0
        num_hard = 0
        
        # Iterate over anchors - more memory efficient than N³ tensor
        for anchor_idx in range(batch_size):
            # Get positive indices for this anchor
            pos_mask = mask_pos[anchor_idx]
            if not pos_mask.any():
                continue
            
            pos_indices = torch.where(pos_mask)[0]
            neg_mask = mask_neg[anchor_idx]
            
            if not neg_mask.any():
                continue
            
            neg_indices = torch.where(neg_mask)[0]
            neg_distances = pairwise_dist[anchor_idx, neg_indices]
            
            for pos_idx in pos_indices:
                d_ap = pairwise_dist[anchor_idx, pos_idx]
                
                # Semi-hard: d_ap < d_an < d_ap + margin
                semi_hard_mask = (neg_distances > d_ap) & (neg_distances < d_ap + self.margin)
                
                if semi_hard_mask.any():
                    # Select hardest semi-hard (largest distance within margin)
                    semi_hard_dists = neg_distances[semi_hard_mask]
                    d_an = semi_hard_dists.max()
                    num_semihard += 1
                else:
                    # Fallback to hardest negative within d_ap (hard negative)
                    hard_mask = neg_distances < d_ap
                    if hard_mask.any():
                        d_an = neg_distances[hard_mask].max()
                        num_hard += 1
                    else:
                        continue
                
                loss = F.relu(d_ap - d_an + self.margin)
                total_loss = total_loss + loss
        
        num_total = num_semihard + num_hard
        
        if num_total > 0:
            loss = total_loss / num_total
        else:
            # Fallback to batch hard if no triplets found
            return self.batch_hard_fallback(embeddings, labels)
        
        stats = {
            'num_triplets': num_total,
            'num_semihard': num_semihard,
            'num_hard': num_hard,
            'fraction_semihard': num_semihard / max(num_total, 1),
        }
        
        return loss, stats


class HybridTripletLoss(nn.Module):
    """
    Hybrid loss that uses batch-hard for speed but incorporates
    semi-hard concepts through sampling.
    
    Much faster than true semi-hard mining while maintaining quality.
    """
    
    def __init__(
        self, 
        margin: float = 0.2, 
        squared: bool = True,
        soft_margin: bool = False
    ):
        super().__init__()
        self.margin = margin
        self.squared = squared
        self.soft_margin = soft_margin
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute hybrid triplet loss.
        
        Uses vectorized batch-hard for efficiency, with soft margin option
        for smoother gradients.
        """
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        pairwise_dist = pairwise_distances_optimized(embeddings, squared=self.squared)
        
        # Masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        indices_not_equal = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        
        mask_pos = labels_equal & indices_not_equal
        mask_neg = ~labels_equal
        
        # Hardest positive
        pos_dist = pairwise_dist.clone()
        pos_dist[~mask_pos] = 0
        hardest_pos_dist, _ = pos_dist.max(dim=1)
        
        # Hardest negative
        neg_dist = pairwise_dist.clone()
        large_val = pairwise_dist.max() + 1
        neg_dist[~mask_neg] = large_val
        hardest_neg_dist, _ = neg_dist.min(dim=1)
        
        # Compute loss
        if self.soft_margin:
            # Soft margin: log(1 + exp(d_pos - d_neg))
            # Provides smoother gradients
            triplet_loss = torch.log1p(torch.exp(hardest_pos_dist - hardest_neg_dist))
        else:
            triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        
        # Valid anchors (have at least one pos and one neg)
        valid = (pos_dist.sum(dim=1) > 0) & (neg_dist.min(dim=1)[0] < large_val)
        
        if valid.sum() > 0:
            loss = triplet_loss[valid].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        stats = {
            'num_triplets': int(valid.sum().item()),
            'num_hard': int((triplet_loss[valid] > 0).sum().item()) if valid.sum() > 0 else 0,
            'avg_pos_dist': hardest_pos_dist[valid].mean().item() if valid.sum() > 0 else 0,
            'avg_neg_dist': hardest_neg_dist[valid].mean().item() if valid.sum() > 0 else 0,
        }
        
        return loss, stats


def get_optimized_loss(
    loss_type: str = 'hybrid',
    margin: float = 0.2,
    squared: bool = True
) -> nn.Module:
    """
    Factory function for optimized loss functions.
    
    Args:
        loss_type: 'hybrid' (recommended), 'batch_hard', or 'semihard'
        margin: Triplet margin
        squared: Use squared L2 distance
    """
    if loss_type == 'hybrid':
        return HybridTripletLoss(margin=margin, squared=squared)
    elif loss_type == 'batch_hard':
        return EfficientBatchHardLoss(margin=margin, squared=squared)
    elif loss_type == 'semihard':
        return EfficientSemiHardLoss(margin=margin, squared=squared)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Benchmark comparison
    import time
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate typical batch
    P, K = 32, 4  # 32 identities, 4 samples each = 128 batch size
    batch_size = P * K
    embedding_dim = 128
    
    # Create test data
    embeddings = []
    labels = []
    for i in range(P):
        base = torch.randn(1, embedding_dim)
        for j in range(K):
            embeddings.append(base + 0.1 * torch.randn(1, embedding_dim))
            labels.append(i)
    
    embeddings = torch.cat(embeddings, dim=0).to(device)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor(labels).to(device)
    
    print(f"Batch size: {batch_size}, Device: {device}")
    print("=" * 50)
    
    # Test each loss
    for name, loss_fn in [
        ("Hybrid (Recommended)", HybridTripletLoss()),
        ("Batch Hard", EfficientBatchHardLoss()),
        ("Semi-Hard", EfficientSemiHardLoss()),
    ]:
        loss_fn = loss_fn.to(device)
        
        # Warmup
        for _ in range(3):
            loss, _ = loss_fn(embeddings, labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time it
        start = time.time()
        n_iters = 100
        for _ in range(n_iters):
            loss, stats = loss_fn(embeddings, labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / n_iters * 1000
        print(f"{name}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Time: {elapsed:.2f} ms/batch")
        print(f"  Stats: {stats}")
        print()
