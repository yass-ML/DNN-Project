# FaceNet: Face Recognition using Deep Neural Networks

Implementation of the FaceNet architecture following the paper ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832) (Schroff et al., 2015).

## Overview

FaceNet learns a mapping from face images to a compact Euclidean space where distances directly correspond to face similarity. The key contributions implemented here:

- **Inception-ResNet-v1** architecture with 128-dimensional embeddings
- **L2 normalized embeddings** constrained to a unit hypersphere (||f(x)||₂ = 1)
- **Triplet Loss** with **Online Semi-Hard Negative Mining**
- **Identity-based batch sampling** (P identities × K samples)

## Project Structure

```
DNN-Project/
├── src/
│   ├── __init__.py
│   ├── model.py          # Inception-ResNet-v1 architecture
│   ├── loss.py           # Triplet loss + mining strategies
│   ├── dataset.py        # Data loading + identity sampling
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script
│   └── inference.py      # Inference utilities
├── app/
│   ├── backend/
│   │   ├── main.py       # FastAPI server
│   │   └── model_service.py
│   ├── frontend/
│   │   ├── app.py        # Streamlit UI
│   │   └── utils.py
│   └── requirements.txt
├── configs/
│   └── default.yaml      # Training configuration
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd DNN-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For frontend demo
pip install -r app/requirements.txt
```

## Data Preparation

Organize your face dataset in the following structure:

```
data/
├── train/
│   ├── person_001/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── person_002/
│   │   └── ...
│   └── ...
└── val/
    └── ...
```

Each identity should have its own folder with at least 4 images (for K=4 sampling).

## Training

```bash
# Basic training
python src/train.py --data_dir data/train --epochs 100

# With validation
python src/train.py \
    --data_dir data/train \
    --val_dir data/val \
    --epochs 100 \
    --p_identities 32 \
    --k_samples 4 \
    --margin 0.2 \
    --lr 0.05

# Debug mode (quick test)
python src/train.py --data_dir data/train --debug
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Path to training data |
| `--val_dir` | None | Path to validation data |
| `--epochs` | 100 | Number of training epochs |
| `--p_identities` | 32 | Identities per batch |
| `--k_samples` | 4 | Samples per identity |
| `--lr` | 0.05 | Initial learning rate |
| `--margin` | 0.2 | Triplet loss margin |
| `--loss_type` | semihard | Loss type: semihard, batch_hard, batch_all |

## Evaluation

```bash
# Evaluate on dataset
python src/eval.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/val

# LFW-style pair verification
python src/eval.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/lfw \
    --pairs_file data/pairs.txt
```

## Inference

```bash
# Compare two face images
python src/inference.py \
    --model_path checkpoints/best_model.pth \
    --image1 face1.jpg \
    --image2 face2.jpg

# Get embedding for single image
python src/inference.py \
    --model_path checkpoints/best_model.pth \
    --image1 face.jpg
```

### Python API

```python
from src.inference import FaceNetInference

# Load model
facenet = FaceNetInference("checkpoints/best_model.pth")

# Get embedding (128-dim, L2-normalized)
embedding = facenet.get_embedding("face.jpg")
print(f"Embedding shape: {embedding.shape}")  # (128,)
print(f"L2 norm: {np.linalg.norm(embedding)}")  # ≈ 1.0

# Compare two faces
result = facenet.compare_faces("face1.jpg", "face2.jpg")
print(f"Distance: {result['distance']:.4f}")
print(f"Same person: {result['is_same_person']}")
```

## Frontend Demo

### Start Backend

```bash
cd app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend

```bash
cd app/frontend
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Technical Details

### Model Architecture

- **Input**: 160×160 RGB face images
- **Backbone**: Inception-ResNet-v1
- **Output**: 128-dimensional L2-normalized embedding

### Triplet Loss

The triplet loss minimizes the distance between an anchor and a positive (same identity) while maximizing the distance to a negative (different identity):

```
L = max(||f(a) - f(p)||² - ||f(a) - f(n)||² + α, 0)
```

Where α = 0.2 is the margin.

### Semi-Hard Negative Mining

Instead of using all triplets or hardest negatives, we select semi-hard negatives:

```
d(a, p) < d(a, n) < d(a, p) + α
```

This prevents training from collapsing due to hard negatives early in training.

## References

- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

## License

MIT License
