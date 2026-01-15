"""
Training Script for FaceNet.

Implements the training loop with:
- Triplet loss with semi-hard negative mining
- Learning rate scheduling
- Progress tracking and logging
- Model checkpointing
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import InceptionResNetV1
from src.loss import SemiHardTripletLoss, BatchHardTripletLoss, BatchAllTripletLoss
from src.dataset import FaceDataset, IdentitySampler, get_transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train FaceNet model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Path to validation data directory')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--p_identities', type=int, default=32,
                        help='Number of identities per batch')
    parser.add_argument('--k_samples', type=int, default=4,
                        help='Number of samples per identity')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', 
                        default=[30, 60, 80],
                        help='Epochs to decay learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                        help='Learning rate decay factor')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    
    # Loss arguments
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Triplet loss margin (default: 0.2)')
    parser.add_argument('--loss_type', type=str, default='semihard',
                        choices=['semihard', 'batch_hard', 'batch_all'],
                        help='Triplet loss type')
    
    # Misc arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--iterations_per_epoch', type=int, default=1000,
                        help='Iterations per epoch')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Tensorboard log directory')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with small dataset')
    
    return parser.parse_args()


def get_loss_fn(loss_type: str, margin: float):
    """Get loss function based on type."""
    if loss_type == 'semihard':
        return SemiHardTripletLoss(margin=margin)
    elif loss_type == 'batch_hard':
        return BatchHardTripletLoss(margin=margin)
    elif loss_type == 'batch_all':
        return BatchAllTripletLoss(margin=margin)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_scheduler(optimizer, args):
    """Get learning rate scheduler."""
    def lr_lambda(epoch):
        factor = 1.0
        for decay_epoch in args.lr_decay_epochs:
            if epoch >= decay_epoch:
                factor *= args.lr_decay_factor
        return factor
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, 
                writer, args):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_triplets = 0
    total_semihard = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        embeddings = model(images)
        
        # Compute loss
        if args.loss_type in ['semihard']:
            loss, stats = loss_fn(embeddings, labels)
            total_triplets += stats.get('num_triplets', 0)
            total_semihard += stats.get('num_semihard', stats.get('num_triplets', 0))
        elif args.loss_type == 'batch_all':
            loss, frac = loss_fn(embeddings, labels)
            stats = {'fraction_positive': frac}
        else:
            loss = loss_fn(embeddings, labels)
            stats = {}
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            if 'num_triplets' in stats:
                writer.add_scalar('train/num_triplets', stats['num_triplets'], global_step)
            if 'fraction_semihard' in stats:
                writer.add_scalar('train/fraction_semihard', stats['fraction_semihard'], global_step)
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'total_triplets': total_triplets,
        'total_semihard': total_semihard,
    }


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, args):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_batches = 0
    
    for images, labels in tqdm(dataloader, desc='Validating'):
        images = images.to(device)
        labels = labels.to(device)
        
        embeddings = model(images)
        
        if args.loss_type in ['semihard']:
            loss, _ = loss_fn(embeddings, labels)
        elif args.loss_type == 'batch_all':
            loss, _ = loss_fn(embeddings, labels)
        else:
            loss = loss_fn(embeddings, labels)
        
        total_loss += loss.item()
        total_batches += 1
    
    return total_loss / max(total_batches, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)
    print(f"Saved checkpoint to {path}")


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Create model
    print("Creating model...")
    model = InceptionResNetV1(
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.data_dir}...")
    train_transform = get_transforms('train')
    train_dataset = FaceDataset(args.data_dir, transform=train_transform)
    
    # Adjust for debug mode
    if args.debug:
        args.iterations_per_epoch = 10
        args.epochs = 2
        args.p_identities = min(args.p_identities, 4)
    
    train_sampler = IdentitySampler(
        train_dataset,
        p_identities=args.p_identities,
        k_samples=args.k_samples,
        num_iterations=args.iterations_per_epoch
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Validation loader (optional)
    val_loader = None
    if args.val_dir:
        val_transform = get_transforms('eval')
        val_dataset = FaceDataset(args.val_dir, transform=val_transform)
        val_sampler = IdentitySampler(
            val_dataset,
            p_identities=args.p_identities,
            k_samples=args.k_samples,
            num_iterations=100  # Fixed number for validation
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create loss function
    print(f"Using {args.loss_type} triplet loss with margin={args.margin}")
    loss_fn = get_loss_fn(args.loss_type, args.margin)
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = get_scheduler(optimizer, args)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.p_identities * args.k_samples} "
          f"({args.p_identities} identities × {args.k_samples} samples)")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_stats = train_epoch(
            model, train_loader, loss_fn, optimizer, 
            device, epoch, writer, args
        )
        
        # Step scheduler
        scheduler.step()
        
        # Validate
        val_loss = None
        if val_loader:
            val_loss = validate(model, val_loader, loss_fn, device, args)
            writer.add_scalar('val/loss', val_loss, epoch)
        
        # Log epoch stats
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Triplets: {train_stats['total_triplets']}, "
              f"Semi-hard: {train_stats['total_semihard']}")
        if val_loss:
            print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        current_loss = val_loss if val_loss else train_stats['loss']
        if current_loss < best_loss:
            best_loss = current_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss,
                save_dir / 'best_model.pth'
            )
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_stats['loss'],
                save_dir / f'checkpoint_epoch_{epoch}.pth'
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs, train_stats['loss'],
        save_dir / 'final_model.pth'
    )
    
    writer.close()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {save_dir}")
    print(f"Tensorboard logs saved to {log_dir}")


if __name__ == '__main__':
    main()
