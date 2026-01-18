"""
FastAPI Backend for FaceNet Face Recognition.

Endpoints:
- POST /embed: Generate embedding for an uploaded image
- POST /compare: Compare two face images
- GET /health: Health check
- POST /load_model: Load a model checkpoint
"""

import os
import sys
from pathlib import Path
from typing import Optional
import tempfile
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import uvicorn

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.backend.model_service import get_model_service


# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class EmbeddingResponse(BaseModel):
    embedding: list
    embedding_dim: int
    l2_norm: float


class ComparisonResponse(BaseModel):
    distance: float
    similarity: float
    is_same_person: bool
    confidence: float
    threshold: float


class LoadModelRequest(BaseModel):
    model_path: str
    threshold: float = 1.0


class LoadModelResponse(BaseModel):
    success: bool
    message: str


# Create FastAPI app
app = FastAPI(
    title="FaceNet API",
    description="Face recognition API using FaceNet embeddings",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image_from_upload(file: UploadFile) -> Image.Image:
    """Load PIL Image from uploaded file."""
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return image


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    service = get_model_service()
    status = service.get_status()

    return HealthResponse(
        status="healthy", model_loaded=status["model_loaded"], device=status["device"]
    )


@app.post("/load_model", response_model=LoadModelResponse)
async def load_model(request: LoadModelRequest):
    """Load a model checkpoint."""
    service = get_model_service()

    if not os.path.exists(request.model_path):
        raise HTTPException(
            status_code=404, detail=f"Model not found: {request.model_path}"
        )

    success = service.load_model(request.model_path, request.threshold)

    if success:
        return LoadModelResponse(success=True, message="Model loaded successfully")
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")


@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embedding(image: UploadFile = File(...)):
    """
    Generate embedding for an uploaded face image.

    Args:
        image: Face image file (JPEG, PNG)

    Returns:
        128-dimensional L2-normalized embedding
    """
    service = get_model_service()

    if not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Load image
        pil_image = load_image_from_upload(image)

        # Generate embedding
        embedding = service.get_embedding(pil_image)

        return EmbeddingResponse(
            embedding=embedding.tolist(),
            embedding_dim=len(embedding),
            l2_norm=float(np.linalg.norm(embedding)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=ComparisonResponse)
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    """
    Compare two face images.

    Args:
        image1: First face image
        image2: Second face image

    Returns:
        Comparison results including distance, similarity, and same person decision
    """
    service = get_model_service()

    if not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Load images
        pil_image1 = load_image_from_upload(image1)
        pil_image2 = load_image_from_upload(image2)

        # Compare faces
        result = service.compare_faces(pil_image1, pil_image2)

        return ComparisonResponse(
            distance=result["distance"],
            similarity=result["similarity"],
            is_same_person=result["is_same_person"],
            confidence=result["confidence"],
            threshold=result["threshold"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_threshold")
async def set_threshold(threshold: float = Form(...)):
    """Update the distance threshold for same person decision."""
    service = get_model_service()
    service.set_threshold(threshold)
    return {"success": True, "threshold": threshold}


@app.get("/status")
async def get_status():
    """Get detailed service status."""
    service = get_model_service()
    return service.get_status()


def main():
    """Run the API server."""
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
