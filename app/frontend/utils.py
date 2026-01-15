"""
Utility functions for Streamlit frontend.
"""

import io
from typing import Tuple, Optional
import numpy as np
from PIL import Image


def resize_image(image: Image.Image, max_size: int = 400) -> Image.Image:
    """Resize image maintaining aspect ratio."""
    width, height = image.size
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


def pil_to_bytes(image: Image.Image, format: str = 'JPEG') -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def create_comparison_visual(image1: Image.Image, image2: Image.Image,
                             is_same: bool, confidence: float) -> Image.Image:
    """Create a side-by-side comparison image with result overlay."""
    # Resize both images to the same height
    target_height = 300
    
    w1, h1 = image1.size
    new_w1 = int(w1 * target_height / h1)
    image1 = image1.resize((new_w1, target_height))
    
    w2, h2 = image2.size
    new_w2 = int(w2 * target_height / h2)
    image2 = image2.resize((new_w2, target_height))
    
    # Create combined image
    gap = 20
    combined_width = new_w1 + gap + new_w2
    combined = Image.new('RGB', (combined_width, target_height), color='white')
    
    combined.paste(image1, (0, 0))
    combined.paste(image2, (new_w1 + gap, 0))
    
    return combined


def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Get color based on confidence level."""
    if confidence >= 0.8:
        return (0, 200, 0)  # Green
    elif confidence >= 0.5:
        return (255, 165, 0)  # Orange
    else:
        return (255, 0, 0)  # Red


def format_embedding(embedding: np.ndarray, num_values: int = 5) -> str:
    """Format embedding for display."""
    values = embedding[:num_values]
    values_str = ", ".join(f"{v:.4f}" for v in values)
    return f"[{values_str}, ... ({len(embedding)} dims)]"
