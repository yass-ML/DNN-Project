# 3D Reconstruction for Unity (Gaussian Splatting)

## Project Overview
This project focuses on creating high-fidelity 3D environments from 2D images using Gaussian Splatting, specifically optimized for export and use within **Unity**.

## Workflow
1.  **Capture**: Take photos of a scene (street, room, nature trail).
2.  **Process**: Train a Gaussian Splatting model to reconstruct the scene.
3.  **Export**: Generate a `.ply` file.
4.  **Interactive Experience**: Import the model into Unity to explore the environment with a playable character.

## Architecture

### `src/gaussian_splatting/`
-   **Pipeline**: Handles the optimization of the 3D Gaussian cloud.
-   **Output**: Produces a strictly formatted `.ply` file compatible with standard Unity Gaussian Splatting renderers.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd DNN-Project
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training & Export
Run the training pipeline. This will load your images/colmap data and produce a 3D model.
```bash
python main.py --mode train --source ./data/my_scene --output_path ./data/models/scene.ply
```

### 2. Import into Unity
1.  Open your Unity Project.
2.  Install a Gaussian Splatting renderer package (e.g., [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)).
3.  Drag and drop the generated `scene.ply` into your Unity Assets folder.
4.  Create a "Gaussian Splat" object in your scene and assign the asset.
5.  Add a Character Controller to run around your scanned world!
