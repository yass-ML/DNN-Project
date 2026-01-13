#!/usr/bin/env python3
"""
3D Gaussian Splatting for Unity Export.

This script trains a Gaussian Splatting model from input images
and exports it in a format compatible with Unity.

Usage:
    # Training from COLMAP data
    python main.py --mode train --source ./data/my_scene --output ./output/scene.ply

    # Training with custom parameters
    python main.py --mode train --source ./data/my_scene --output ./output/scene.ply \\
        --iterations 15000 --sh_degree 2

Requirements:
    - Images in source_path/images/
    - COLMAP sparse reconstruction in source_path/sparse/0/

    OR

    - Just images (will create synthetic cameras, lower quality)
"""

import argparse
import os
import sys
import torch

from src.gaussian_splatting.pipeline import GaussianSplattingPipeline, GaussianConfig
from src.gaussian_splatting.export import export_ply


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting -> Unity Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "export", "info"],
        default="train",
        help="Mode: train (default), export (from checkpoint), info (show GPU info)",
    )

    # Data paths
    parser.add_argument(
        "--source",
        type=str,
        default="./data",
        help="Path to source data (COLMAP output or images folder)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="images",
        help="Name of images subfolder within source",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/model.ply",
        help="Output path for PLY file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for checkpoints and logs",
    )

    # Training parameters
    parser.add_argument(
        "--iterations", type=int, default=30000, help="Number of training iterations"
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Spherical harmonics degree (0-3)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution scale (0.5 = half resolution, faster training)",
    )

    # Learning rates
    parser.add_argument("--position_lr", type=float, default=0.00016)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # Densification
    parser.add_argument(
        "--densify_until",
        type=int,
        default=15000,
        help="Stop densification after this iteration",
    )

    # Misc
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5000,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--white_bg", action="store_true", help="Use white background instead of black"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (for export mode)",
    )

    return parser.parse_args()


def print_gpu_info():
    """Print GPU information."""
    print("\n=== GPU Information ===")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"Device: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Multi Processors: {props.multi_processor_count}")
    else:
        print("CUDA Available: No")
        print("Training will run on CPU (very slow)")
    print()


def main():
    args = parse_args()

    # Info mode
    if args.mode == "info":
        print_gpu_info()
        return

    print("=" * 60)
    print("  3D Gaussian Splatting for Unity")
    print("=" * 60)
    print_gpu_info()

    # Build config
    bg_color = (1.0, 1.0, 1.0) if args.white_bg else (0.0, 0.0, 0.0)

    config = GaussianConfig(
        iterations=args.iterations,
        position_lr_init=args.position_lr,
        feature_lr=args.feature_lr,
        opacity_lr=args.opacity_lr,
        scaling_lr=args.scaling_lr,
        rotation_lr=args.rotation_lr,
        sh_degree=args.sh_degree,
        resolution_scale=args.resolution,
        densify_until_iter=args.densify_until,
        bg_color=bg_color,
        output_path=args.output_dir,
        save_interval=args.save_interval,
    )

    if args.mode == "train":
        print(f"\nSource: {args.source}")
        print(f"Output: {args.output}")
        print(f"Iterations: {args.iterations}")
        print(f"SH Degree: {args.sh_degree}")
        print(f"Resolution Scale: {args.resolution}")
        print()

        # Create pipeline and run
        pipeline = GaussianSplattingPipeline(config)
        pipeline.run(
            source_path=args.source, output_path=args.output, images_folder=args.images
        )

    elif args.mode == "export":
        if args.checkpoint is None:
            print("Error: --checkpoint required for export mode")
            sys.exit(1)

        print(f"\nExporting from checkpoint: {args.checkpoint}")
        print(f"Output: {args.output}")

        # Load checkpoint and export
        from src.gaussian_splatting.gaussian_model import GaussianModel

        checkpoint = torch.load(args.checkpoint)
        model = GaussianModel(sh_degree=args.sh_degree)
        model.load_state_dict(checkpoint["model_state_dict"])

        export_ply(model, args.output, include_sh=True, max_sh_degree=args.sh_degree)

    print("\n" + "=" * 60)
    print("  Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
