# 3D Navigation with Gaussian Splatting and Reinforcement Learning

## Project Overview
This project aims to create an autonomous navigation system that learns to navigate a 3D environment reconstructed from 2D images.
It consists of two main components:
1.  **3D Reconstruction (Gaussian Splatting)**: Takes a set of images (e.g., a street, a hiking trail) and reconstructs a high-fidelity 3D scene using 3D Gaussian Splatting.
2.  **Navigation (Reinforcement Learning)**: An RL agent is trained to navigate this reconstructed 3D environment from a starting point to a destination.

## Architecture

### 1. Gaussian Splatting (`src/gaussian_splatting/`)
-   **Pipeline**: Handles the optimization loop of 3D Gaussians.
-   **Scene**: Loads input data (images + COLMAP poses) and initializes the point cloud.
-   **Output**: A `.ply` file representing the 3D scene which allows for real-time rendering.

### 2. RL Navigation (`src/rl_navigation/`)
-   **Environment (`NavigationEnv`)**: A custom Gym environment. It uses the GS pipeline to render observations (RGB images) or provides state vectors to the agent. It defines the reward function (e.g., distance to target, collision avoidance).
-   **Agent**: A generic RL agent (currently a simple Policy Network skeleton) that learns to map observations to movement actions.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd DNN-Project
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Gaussian Splatting often requires custom CUDA kernels (diff-gaussian-rasterization). You may need to install those separately from standard submodules repositories.*

## Usage

### 1. Train Gaussian Splatting Model
Reconstruct the 3D scene from your image dataset.
```bash
python main.py --mode train_gs --source ./data/my_scene
```

### 2. Train RL Agent
Train the agent to navigate the reconstructed scene.
```bash
python main.py --mode train_rl
```

### 3. Demo
Run a visualization of the agent performing in the environment.
```bash
python main.py --mode demo
```

## Project Status & TODOs

### Initial Setup (Done)
- [x] Project Skeleton Created
- [x] Core File Structure (`src/`, `data/`)
- [x] Entry point (`main.py`)

### Gaussian Splatting Module (TODO)
- [ ] Implement actual `load_colmap` parsing logic in `scene.py`.
- [ ] Implement `render()` function using rasterization kernels in `pipeline.py`.
- [ ] Implement loss functions (L1, D-SSIM) in `pipeline.py`.

### RL Navigation Module (TODO)
- [ ] Connect `NavigationEnv.render()` to the GS renderer to produce image observations.
- [ ] Switch Agent to a Convolutional Neural Network (CNN) to handle image inputs.
- [ ] Implement a robust RL algorithm (PPO or SAC) instead of the skeleton policy.

## Workflow Example
1.  **Capture**: Take 50-100 photos of a generic street path.
2.  **Process**: Run COLMAP to get sparse structure and positions.
3.  **Train GS**: Run `train_gs` to get a 3D model.
4.  **Train RL**: Run `train_rl` to teach the agent to walk from A to B in that model.
