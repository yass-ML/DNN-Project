import argparse
import os
from src.gaussian_splatting.pipeline import GaussianSplattingPipeline, GaussianConfig
from src.gaussian_splatting.scene import Scene

def main():
    parser = argparse.ArgumentParser(description="Gaussian Splatting -> Unity Pipeline")
    parser.add_argument('--mode', type=str, choices=['train'], default='train', help='Mode to run (currently only train)')
    parser.add_argument('--source', type=str, default="./data", help='Path to source data (images/SfM)')
    parser.add_argument('--output_path', type=str, default="./data/models/model.ply", help='Path to save the compatible PLY file')
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("=== Gaussian Splatting Training ===")
        # 1. Load Scene
        scene = Scene(args.source)
        # 2. Init Pipeline
        pipeline = GaussianSplattingPipeline(GaussianConfig())
        # 3. Train
        pipeline.train(scene)
        # 4. Save
        print(f"Exporting model to {args.output_path} for Unity import...")
        pipeline.model.save_ply(args.output_path)
        print("Done! You can now drag and drop the .ply file into your Unity project (requires GS plugin).")

if __name__ == "__main__":
    main()
