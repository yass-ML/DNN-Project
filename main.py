import argparse
import os
from src.gaussian_splatting.pipeline import GaussianSplattingPipeline, GaussianConfig
from src.gaussian_splatting.scene import Scene
from src.rl_navigation.environment import NavigationEnv
from src.rl_navigation.agent import RLAgent

def main():
    parser = argparse.ArgumentParser(description="Gaussian Splatting + RL Navigation Project")
    parser.add_argument('--mode', type=str, choices=['train_gs', 'train_rl', 'demo'], required=True, help='Mode to run')
    parser.add_argument('--source', type=str, default="./data", help='Path to source data (images/SfM)')
    parser.add_argument('--model_path', type=str, default="./data/models", help='Path to save/load models')
    
    args = parser.parse_args()

    if args.mode == 'train_gs':
        print("=== Mode: Train Gaussian Splatting Model ===")
        # 1. Load Scene
        scene = Scene(args.source)
        # 2. Init Pipeline
        pipeline = GaussianSplattingPipeline(GaussianConfig())
        # 3. Train
        pipeline.train(scene)
        # 4. Save
        pipeline.model.save_ply(os.path.join(args.model_path, "model.ply"))

    elif args.mode == 'train_rl':
        print("=== Mode: Train RL Agent ===")
        # 1. Load GS Model (Stub)
        gs_pipeline = GaussianSplattingPipeline(GaussianConfig()) # Should load trained weights
        
        # 2. Setup Environment
        start = [0, 0, 0]
        target = [5, 5, 0]
        env = NavigationEnv(gs_pipeline, start, target)
        
        # 3. Init Agent
        agent = RLAgent(state_dim=6, action_dim=3)
        
        # 4. Train
        agent.train(env)

    elif args.mode == 'demo':
        print("=== Mode: Demo ===")
        print("Loading models and running visual demo...")
        # TODO: Load everything and run a loop showing the agent moving in the GS environment

if __name__ == "__main__":
    main()
