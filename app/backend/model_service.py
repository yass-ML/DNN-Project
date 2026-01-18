"""
Model Service for FaceNet Backend.

Handles model loading and inference for the API.
"""

import os
from pathlib import Path
from typing import Optional, Union
import threading

import numpy as np
from PIL import Image
import torch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import InceptionResNetV1
from src.inference import FaceNetInference
from src.inference_transfer import FaceNetTransferInference


def is_transfer_learning_checkpoint(model_path: str) -> bool:
    """
    Check if a checkpoint is from transfer learning (facenet_pytorch).
    
    Transfer learning checkpoints have keys starting with 'conv2d_1a'
    while custom model checkpoints have keys starting with 'stem'.
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Check for facenet_pytorch style keys
        keys = list(state_dict.keys())
        if keys and any(k.startswith('conv2d_1a') for k in keys):
            return True
        return False
    except Exception:
        return False


class ModelService:
    """
    Singleton service for FaceNet model inference.
    
    Thread-safe model loading and inference.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model: Optional[Union[FaceNetInference, FaceNetTransferInference]] = None
        self.model_path: Optional[str] = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.threshold = 1.0
        self.is_transfer_learning = False
        self._initialized = True
    
    def load_model(self, model_path: str, threshold: float = 1.0) -> bool:
        """
        Load model from checkpoint.
        
        Automatically detects if the checkpoint is from transfer learning
        (facenet_pytorch) or the custom model implementation.
        
        Args:
            model_path: Path to model checkpoint
            threshold: Distance threshold for same person
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Detect model type
            self.is_transfer_learning = is_transfer_learning_checkpoint(model_path)
            
            if self.is_transfer_learning:
                print(f"Detected transfer learning model: {model_path}")
                # Count classes from scia_images folder
                scia_path = Path(__file__).parent.parent.parent / "data" / "scia_images"
                if scia_path.exists():
                    num_classes = len([d for d in scia_path.iterdir() 
                                      if d.is_dir() and not d.name.startswith('.')])
                else:
                    num_classes = 64  # Default fallback
                
                self.model = FaceNetTransferInference(
                    model_path=model_path,
                    num_classes=num_classes,
                    device=self.device,
                    threshold=threshold
                )
            else:
                print(f"Detected custom model: {model_path}")
                self.model = FaceNetInference(
                    model_path=model_path,
                    device=self.device,
                    threshold=threshold
                )
            
            self.model_path = model_path
            self.threshold = threshold
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Generate embedding for an image."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.get_embedding(image)
    
    def compare_faces(self, image1: Union[str, Image.Image, np.ndarray],
                      image2: Union[str, Image.Image, np.ndarray]) -> dict:
        """Compare two face images."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.compare_faces(image1, image2)
    
    def set_threshold(self, threshold: float):
        """Update the distance threshold."""
        self.threshold = threshold
        if self.model:
            self.model.threshold = threshold
    
    def get_status(self) -> dict:
        """Get service status."""
        return {
            'model_loaded': self.is_loaded(),
            'model_path': self.model_path,
            'device': self.device,
            'threshold': self.threshold,
            'model_type': 'transfer_learning' if self.is_transfer_learning else 'custom',
        }


# Global instance
model_service = ModelService()


def get_model_service() -> ModelService:
    """Get the model service singleton."""
    return model_service
