"""
Triplet Loss and Online Mining Strategies for FaceNet.

Implementation based on "FaceNet: A Unified Embedding for Face Recognition and Clustering"
(Schroff et al., 2015) - Section 3.2

Key concepts:
- Triplet Loss: L = [||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin]+
- Semi-Hard Negative Mining: Select negatives where d(a,p) < d(a,n) < d(a,p) + margin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def pairwise_distances(embeddings: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """
    Compute pairwise distance matrix.
    
    Args:
        embeddings: Tensor of shape (N, D) with L2-normalized embeddings
        squared: If True, return squared distances
    
    Returns:
        Distance matrix of shape (N, N)
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    # For L2-normalized vectors: ||a||^2 = ||b||^2 = 1
    # So: ||a - b||^2 = 2 - 2*a·b = 2*(1 - a·b)
    dot_product = torch.mm(embeddings, embeddings.t())
    
    # Get squared norms
    square_norm = torch.diag(dot_product)
    
    # ||a - b||^2 = ||a||^2 - 2*a·b + ||b||^2
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    
    # Ensure numerical stability
    distances = torch.clamp(distances, min=0.0)
    
    if not squared:
        # Add epsilon for numerical stability before sqrt
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    
    return distances


def get_anchor_positive_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    Get a mask for valid anchor-positive pairs.
    
    Returns mask[a, p] = 1 if a and p have same label and a != p
    """
    # Check that i and j are distinct
    indices_not_equal = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    
    # Check if labels are equal
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    return indices_not_equal & labels_equal


def get_anchor_negative_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    Get a mask for valid anchor-negative pairs.
    
    Returns mask[a, n] = 1 if a and n have different labels
    """
    return labels.unsqueeze(0) != labels.unsqueeze(1)


def get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    Get a 3D mask for valid triplets.
    
    Returns mask[a, p, n] = 1 if:
    - a, p, n are distinct
    - a and p have same label
    - a and n have different labels
    """
    # Check that i, j, k are distinct
    indices_not_equal = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k
    
    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_not_equal_k = (~label_equal).unsqueeze(1)
    
    valid_labels = i_equal_j & i_not_equal_k
    
    return distinct_indices & valid_labels


class TripletLoss(nn.Module):
    """
    Basic Triplet Loss.
    
    L = max(||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin, 0)
    
    Args:
        margin: Margin for triplet loss (default: 0.2 as per paper)
        squared: If True, use squared L2 distance
    """
    
    def __init__(self, margin: float = 0.2, squared: bool = True):
        super().__init__()
        self.margin = margin
        self.squared = squared
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss for pre-selected triplets.
        
        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)
            negative: Negative embeddings (N, D)
        
        Returns:
            Scalar loss value
        """
        if self.squared:
            distance_positive = torch.sum((anchor - positive) ** 2, dim=1)
            distance_negative = torch.sum((anchor - negative) ** 2, dim=1)
        else:
            distance_positive = torch.norm(anchor - positive, p=2, dim=1)
            distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class BatchAllTripletLoss(nn.Module):
    """
    Batch All Triplet Loss.
    
    Computes triplet loss over all valid triplets in the batch and averages
    over the positive valid triplets.
    
    Args:
        margin: Margin for triplet loss (default: 0.2)
        squared: If True, use squared L2 distance
        soft: If True, use soft margin (log-sum-exp)
    """
    
    def __init__(self, margin: float = 0.2, squared: bool = True, soft: bool = False):
        super().__init__()
        self.margin = margin
        self.squared = squared
        self.soft = soft
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute batch all triplet loss.
        
        Args:
            embeddings: L2-normalized embeddings (N, D)
            labels: Ground truth labels (N,)
        
        Returns:
            loss: Scalar loss value
            fraction_positive: Fraction of positive triplets
        """
        # Compute pairwise distances
        pairwise_dist = pairwise_distances(embeddings, squared=self.squared)
        
        # Get anchor-positive distances: shape (N, N, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        
        # Get anchor-negative distances: shape (N, 1, N)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        
        # Compute triplet loss: shape (N, N, N)
        if self.soft:
            triplet_loss = torch.log1p(torch.exp(anchor_positive_dist - anchor_negative_dist))
        else:
            triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Get valid triplet mask
        mask = get_triplet_mask(labels).float()
        
        # Apply mask
        triplet_loss = triplet_loss * mask
        
        # Remove negative losses (easy triplets)
        triplet_loss = F.relu(triplet_loss)
        
        # Count positive triplets
        valid_triplets = (triplet_loss > 1e-16).float()
        num_positive_triplets = valid_triplets.sum()
        num_valid_triplets = mask.sum()
        
        fraction_positive = num_positive_triplets / (num_valid_triplets + 1e-16)
        
        # Average over positive triplets
        loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        
        return loss, fraction_positive.item()


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss.
    
    For each anchor, selects:
    - Hardest positive: max d(a, p) for p with same label
    - Hardest negative: min d(a, n) for n with different label
    
    Args:
        margin: Margin for triplet loss (default: 0.2)
        squared: If True, use squared L2 distance
        soft: If True, use soft margin
    """
    
    def __init__(self, margin: float = 0.2, squared: bool = True, soft: bool = False):
        super().__init__()
        self.margin = margin
        self.squared = squared
        self.soft = soft
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute batch hard triplet loss.
        
        Args:
            embeddings: L2-normalized embeddings (N, D)
            labels: Ground truth labels (N,)
        
        Returns:
            Scalar loss value
        """
        pairwise_dist = pairwise_distances(embeddings, squared=self.squared)
        
        # Get hardest positive for each anchor
        # Mask out invalid positives
        mask_anchor_positive = get_anchor_positive_triplet_mask(labels).float()
        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        
        # Get max distance for each anchor
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)
        
        # Get hardest negative for each anchor
        # Add max distance to invalid negatives so they're never selected
        mask_anchor_negative = get_anchor_negative_triplet_mask(labels).float()
        max_dist = pairwise_dist.max()
        anchor_negative_dist = pairwise_dist + (1.0 - mask_anchor_negative) * max_dist
        
        # Get min distance for each anchor
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)
        
        # Compute loss
        if self.soft:
            triplet_loss = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))
        else:
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return triplet_loss.mean()


class OnlineTripletMiner(nn.Module):
    """
    Online Triplet Mining with Semi-Hard Negative Selection.
    
    As described in Section 3.2 of the FaceNet paper:
    "We select semi-hard negatives, i.e., negatives that are farther away from the anchor
    than the positive exemplar, but still hard because they violate the margin."
    
    Semi-hard condition: d(a, p) < d(a, n) < d(a, p) + margin
    
    Args:
        margin: Margin for triplet loss (default: 0.2)
        squared: If True, use squared L2 distance
    """
    
    def __init__(self, margin: float = 0.2, squared: bool = True):
        super().__init__()
        self.margin = margin
        self.squared = squared
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute triplet loss using semi-hard negative mining.
        
        Args:
            embeddings: L2-normalized embeddings (N, D)
            labels: Ground truth labels (N,)
        
        Returns:
            loss: Scalar loss value
            stats: Dictionary with mining statistics
        """
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        pairwise_dist = pairwise_distances(embeddings, squared=self.squared)
        
        # Get masks
        anchor_positive_mask = get_anchor_positive_triplet_mask(labels)
        anchor_negative_mask = get_anchor_negative_triplet_mask(labels)
        
        # For each anchor-positive pair, find semi-hard negatives
        total_loss = torch.tensor(0.0, device=device)
        num_valid_triplets = 0
        num_semihard = 0
        num_hard = 0
        
        for anchor_idx in range(batch_size):
            anchor_label = labels[anchor_idx]
            
            # Find all positive indices for this anchor
            positive_mask = anchor_positive_mask[anchor_idx]
            positive_indices = torch.where(positive_mask)[0]
            
            if len(positive_indices) == 0:
                continue
            
            # Find all negative indices for this anchor
            negative_mask = anchor_negative_mask[anchor_idx]
            negative_indices = torch.where(negative_mask)[0]
            
            if len(negative_indices) == 0:
                continue
            
            for pos_idx in positive_indices:
                d_ap = pairwise_dist[anchor_idx, pos_idx]
                
                # Get distances to all negatives
                d_an = pairwise_dist[anchor_idx, negative_indices]
                
                # Semi-hard negatives: d_ap < d_an < d_ap + margin
                semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + self.margin)
                semi_hard_negatives = negative_indices[semi_hard_mask]
                
                if len(semi_hard_negatives) > 0:
                    # Select the hardest semi-hard negative (closest to margin)
                    semi_hard_distances = d_an[semi_hard_mask]
                    hardest_semi_hard_idx = semi_hard_distances.argmax()
                    neg_idx = semi_hard_negatives[hardest_semi_hard_idx]
                    d_an_selected = semi_hard_distances[hardest_semi_hard_idx]
                    num_semihard += 1
                else:
                    # Fallback to hard negative (d_an < d_ap)
                    hard_mask = d_an < d_ap
                    if hard_mask.any():
                        hard_negatives = negative_indices[hard_mask]
                        hard_distances = d_an[hard_mask]
                        hardest_idx = hard_distances.argmax()
                        neg_idx = hard_negatives[hardest_idx]
                        d_an_selected = hard_distances[hardest_idx]
                        num_hard += 1
                    else:
                        # All negatives are easy - skip this pair
                        continue
                
                # Compute triplet loss for this triplet
                loss = F.relu(d_ap - d_an_selected + self.margin)
                total_loss = total_loss + loss
                num_valid_triplets += 1
        
        # Average loss
        if num_valid_triplets > 0:
            loss = total_loss / num_valid_triplets
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        stats = {
            'num_triplets': num_valid_triplets,
            'num_semihard': num_semihard,
            'num_hard': num_hard,
            'fraction_semihard': num_semihard / max(num_valid_triplets, 1),
        }
        
        return loss, stats


class SemiHardTripletLoss(nn.Module):
    """
    Vectorized Semi-Hard Triplet Loss (more efficient implementation).
    
    Implements semi-hard negative mining as described in Section 3.2:
    For each anchor-positive pair, select negatives where:
    d(a, p) < d(a, n) < d(a, p) + margin
    
    Args:
        margin: Margin for triplet loss (default: 0.2)
        squared: If True, use squared L2 distance
    """
    
    def __init__(self, margin: float = 0.2, squared: bool = True):
        super().__init__()
        self.margin = margin
        self.squared = squared
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute semi-hard triplet loss.
        
        Args:
            embeddings: L2-normalized embeddings (N, D)
            labels: Ground truth labels (N,)
        
        Returns:
            loss: Scalar loss value
            stats: Mining statistics
        """
        pairwise_dist = pairwise_distances(embeddings, squared=self.squared)
        
        # Anchor-positive mask and distances
        ap_mask = get_anchor_positive_triplet_mask(labels)
        an_mask = get_anchor_negative_triplet_mask(labels)
        
        # Shape: (N, N) - all anchor-positive distances
        ap_distances = pairwise_dist * ap_mask.float()
        
        # For each anchor, get all positives
        # Shape: (N, 1) - we'll compare against all negatives
        
        # Strategy: For each (anchor, positive) pair, find semi-hard negatives
        # Reshape for broadcasting
        # ap_distances: (N, N) -> we need (N, N, 1) for anchor-positive
        # an_distances: (N, N) -> we need (N, 1, N) for anchor-negative
        
        ap_dist_3d = pairwise_dist.unsqueeze(2)  # (N, N, 1) - anchor to all others
        an_dist_3d = pairwise_dist.unsqueeze(1)  # (N, 1, N) - anchor to all others
        
        # Semi-hard condition: d(a,p) < d(a,n) < d(a,p) + margin
        # Shape: (N, N, N) representing (anchor, positive, negative)
        semi_hard_mask = (an_dist_3d > ap_dist_3d) & (an_dist_3d < ap_dist_3d + self.margin)
        
        # Valid triplet mask
        triplet_mask = get_triplet_mask(labels)
        
        # Combined mask: valid triplets AND semi-hard
        combined_mask = triplet_mask & semi_hard_mask
        
        # Compute loss for all semi-hard triplets
        triplet_loss = ap_dist_3d - an_dist_3d + self.margin
        triplet_loss = F.relu(triplet_loss)
        
        # Apply mask
        triplet_loss = triplet_loss * combined_mask.float()
        
        # Count triplets
        num_semihard = combined_mask.sum().item()
        num_total_valid = triplet_mask.sum().item()
        
        if num_semihard > 0:
            loss = triplet_loss.sum() / num_semihard
        else:
            # Fallback to batch hard if no semi-hard triplets found
            batch_hard = BatchHardTripletLoss(margin=self.margin, squared=self.squared)
            loss = batch_hard(embeddings, labels)
            num_semihard = 0
        
        stats = {
            'num_triplets': int(num_semihard),
            'num_total_valid': int(num_total_valid),
            'fraction_semihard': num_semihard / max(num_total_valid, 1),
        }
        
        return loss, stats


def get_triplet_loss(loss_type: str = 'semihard', margin: float = 0.2, 
                     squared: bool = True) -> nn.Module:
    """
    Factory function to create triplet loss module.
    
    Args:
        loss_type: One of 'basic', 'batch_all', 'batch_hard', 'semihard'
        margin: Margin for triplet loss
        squared: If True, use squared L2 distance
    
    Returns:
        Loss module
    """
    if loss_type == 'basic':
        return TripletLoss(margin=margin, squared=squared)
    elif loss_type == 'batch_all':
        return BatchAllTripletLoss(margin=margin, squared=squared)
    elif loss_type == 'batch_hard':
        return BatchHardTripletLoss(margin=margin, squared=squared)
    elif loss_type == 'semihard':
        return SemiHardTripletLoss(margin=margin, squared=squared)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test triplet loss implementations
    import torch
    
    torch.manual_seed(42)
    
    # Create dummy embeddings and labels
    # Simulate P=4 identities, K=3 samples each
    P, K = 4, 3
    batch_size = P * K
    embedding_dim = 128
    
    # Create embeddings (simulating different identities having similar embeddings)
    embeddings = []
    labels = []
    for i in range(P):
        # Each identity gets K samples
        base = torch.randn(1, embedding_dim)
        for j in range(K):
            # Add small noise to create samples of same identity
            embeddings.append(base + 0.1 * torch.randn(1, embedding_dim))
            labels.append(i)
    
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
    labels = torch.tensor(labels)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels: {labels}")
    
    # Test batch all loss
    print("\n--- Batch All Triplet Loss ---")
    batch_all_loss = BatchAllTripletLoss(margin=0.2)
    loss, frac = batch_all_loss(embeddings, labels)
    print(f"Loss: {loss.item():.4f}, Fraction positive: {frac:.4f}")
    
    # Test batch hard loss
    print("\n--- Batch Hard Triplet Loss ---")
    batch_hard_loss = BatchHardTripletLoss(margin=0.2)
    loss = batch_hard_loss(embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
    
    # Test semi-hard mining
    print("\n--- Semi-Hard Triplet Loss ---")
    semihard_loss = SemiHardTripletLoss(margin=0.2)
    loss, stats = semihard_loss(embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
    
    # Test online miner
    print("\n--- Online Triplet Miner ---")
    miner = OnlineTripletMiner(margin=0.2)
    loss, stats = miner(embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
    
    print("\n✓ All triplet loss tests passed!")
