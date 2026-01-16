"""
Optimized Training Script for FaceNet.

Key optimizations:
1. Automatic Mixed Precision (AMP) for 1.5-3x speedup
2. Gradient accumulation for effective larger batches
3. Efficient zero_grad with set_to_none=True
4. torch.compile for PyTorch 2.0+ (optional)
5. Improved learning rate scheduling
6. Better optimizer choices (AdamW)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import InceptionResNetV1
from src.loss_optimized import get_optimized_loss, HybridTripletLoss
from src.dataset_optimized import (
    FaceDatasetOptimized, 
    IdentitySamplerOptimized, 
    get_transforms_optimized,
    get_optimized_dataloader
)
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train FaceNet model (Optimized)')
    
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
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (lower for AdamW)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer to use')
    
    # Loss arguments
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Triplet loss margin (default: 0.2)')
    parser.add_argument('--loss_type', type=str, default='hybrid',
                        choices=['hybrid', 'batch_hard', 'semihard'],
                        help='Triplet loss type (hybrid recommended)')
    
    # Optimization arguments
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--no_amp', action='store_false', dest='amp',
                        help='Disable AMP')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--cache_images', action='store_true',
                        help='Cache images in RAM (for small datasets)')
    
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


def get_optimizer(model, args):
    """Get optimizer based on args."""
    if args.optimizer == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True  # Slightly better than vanilla momentum
        )
    elif args.optimizer == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:  # adamw
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )


def get_scheduler(optimizer, args, steps_per_epoch):
    """Get learning rate scheduler."""
    # Cosine annealing with warm restarts
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(1000, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model, 
    dataloader, 
    loss_fn, 
    optimizer, 
    scheduler,
    scaler,
    device, 
    epoch, 
    writer, 
    args
):
    """Train for one epoch with optimizations."""
    model.train()
    
    total_loss = 0.0
    total_triplets = 0
    
    # AMP context
    amp_context = autocast() if args.amp and device.type == 'cuda' else nullcontext()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move to device (non_blocking for async transfer)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with AMP
        with amp_context:
            embeddings = model(images)
            loss, stats = loss_fn(embeddings, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation
        
        # Backward pass
        if args.amp and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % args.gradient_accumulation == 0:
            if args.amp and device.type == 'cuda':
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Zero gradients efficiently
            optimizer.zero_grad(set_to_none=True)
            
            # Step scheduler
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item() * args.gradient_accumulation
        total_triplets += stats.get('num_triplets', 0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * args.gradient_accumulation:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('train/loss', loss.item() * args.gradient_accumulation, global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            
            if 'num_triplets' in stats:
                writer.add_scalar('train/num_triplets', stats['num_triplets'], global_step)
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'total_triplets': total_triplets,
    }


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, args):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_batches = 0
    
    amp_context = autocast() if args.amp and device.type == 'cuda' else nullcontext()
    
    for images, labels in tqdm(dataloader, desc='Validating'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with amp_context:
            embeddings = model(images)
            loss, _ = loss_fn(embeddings, labels)
        
        total_loss += loss.item()
        total_batches += 1
    
    return total_loss / max(total_batches, 1)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }, path)
    print(f"Saved checkpoint to {path}")


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn autotuner
        torch.backends.cudnn.benchmark = True
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"AMP enabled: {args.amp}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        args.amp = False  # AMP only on CUDA
    
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
    
    # Optionally compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.data_dir}...")
    
    # Adjust for debug mode
    if args.debug:
        args.iterations_per_epoch = 10
        args.epochs = 2
        args.p_identities = min(args.p_identities, 8)
    
    train_loader = get_optimized_dataloader(
        args.data_dir,
        mode='train',
        p_identities=args.p_identities,
        k_samples=args.k_samples,
        num_workers=args.num_workers,
        num_iterations=args.iterations_per_epoch,
        cache_images=args.cache_images,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Validation loader
    val_loader = None
    if args.val_dir:
        val_loader = get_optimized_dataloader(
            args.val_dir,
            mode='eval',
            p_identities=args.p_identities,
            k_samples=args.k_samples,
            num_workers=args.num_workers,
            num_iterations=100,
            persistent_workers=True
        )
    
    # Create loss function
    print(f"Using {args.loss_type} triplet loss with margin={args.margin}")
    loss_fn = get_optimized_loss(args.loss_type, args.margin)
    
    # Create optimizer
    print(f"Using {args.optimizer.upper()} optimizer with lr={args.lr}")
    optimizer = get_optimizer(model, args)
    
    # Create scheduler
    scheduler = get_scheduler(optimizer, args, len(train_loader))
    
    # Create gradient scaler for AMP
    scaler = GradScaler() if args.amp and device.type == 'cuda' else None
    
    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Batch size: {args.p_identities * args.k_samples} "
          f"({args.p_identities} identities × {args.k_samples} samples)")
    if args.gradient_accumulation > 1:
        print(f"Effective batch size: {args.p_identities * args.k_samples * args.gradient_accumulation}")
    print(f"{'='*50}\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_stats = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            scaler, device, epoch, writer, args
        )
        
        # Validate
        val_loss = None
        if val_loader:
            val_loss = validate(model, val_loader, loss_fn, device, args)
            writer.add_scalar('val/loss', val_loss, epoch)
        
        # Log epoch stats
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Triplets: {train_stats['total_triplets']}")
        if val_loss:
            print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s ({len(train_loader)/epoch_time:.1f} batches/sec)")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        current_loss = val_loss if val_loss else train_stats['loss']
        if current_loss < best_loss:
            best_loss = current_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_loss,
                save_dir / 'best_model.pth'
            )
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, train_stats['loss'],
                save_dir / f'checkpoint_epoch_{epoch}.pth'
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, scaler, args.epochs, train_stats['loss'],
        save_dir / 'final_model.pth'
    )
    
    writer.close()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {save_dir}")
    print(f"Tensorboard logs saved to {log_dir}")


if __name__ == '__main__':
    main()
