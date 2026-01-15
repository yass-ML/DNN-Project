"""
Streamlit Frontend for FaceNet Face Recognition.

Features:
- Upload two face images
- Compare faces using the FaceNet model
- Display similarity score and verification result
"""

import os
import sys
from pathlib import Path
import io

import streamlit as st
import requests
from PIL import Image
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.frontend.utils import resize_image, format_embedding, get_confidence_color

# Configuration
API_URL = os.environ.get("FACENET_API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="FaceNet Face Recognition",
    page_icon="👤",
    layout="wide"
)


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def load_model_via_api(model_path: str, threshold: float = 1.0):
    """Load model via API."""
    try:
        response = requests.post(
            f"{API_URL}/load_model",
            json={"model_path": model_path, "threshold": threshold}
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def compare_faces_api(image1: Image.Image, image2: Image.Image) -> dict:
    """Compare two faces via API."""
    # Convert images to bytes
    buf1 = io.BytesIO()
    image1.save(buf1, format='JPEG')
    buf1.seek(0)
    
    buf2 = io.BytesIO()
    image2.save(buf2, format='JPEG')
    buf2.seek(0)
    
    files = {
        'image1': ('image1.jpg', buf1, 'image/jpeg'),
        'image2': ('image2.jpg', buf2, 'image/jpeg')
    }
    
    try:
        response = requests.post(f"{API_URL}/compare", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_embedding_api(image: Image.Image) -> dict:
    """Get embedding via API."""
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    buf.seek(0)
    
    files = {'image': ('image.jpg', buf, 'image/jpeg')}
    
    try:
        response = requests.post(f"{API_URL}/embed", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    # Header
    st.title("👤 FaceNet Face Recognition")
    st.markdown("""
    Upload two face images to compare them using FaceNet embeddings.
    The model generates 128-dimensional face embeddings and computes 
    the L2 distance to determine if they are the same person.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API status
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.info(f"Make sure the API is running at {API_URL}")
        
        st.divider()
        
        # Model loading
        st.subheader("Model")
        model_path = st.text_input(
            "Model Path",
            value="checkpoints/best_model.pth",
            help="Path to the trained model checkpoint"
        )
        threshold = st.slider(
            "Threshold",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Distance threshold for same person decision"
        )
        
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                result = load_model_via_api(model_path, threshold)
                if result.get("success"):
                    st.success("Model loaded!")
                else:
                    st.error(f"Failed: {result.get('message', 'Unknown error')}")
        
        st.divider()
        
        # Info
        st.subheader("About")
        st.markdown("""
        **FaceNet** (Schroff et al., 2015)
        
        - 128-dim L2-normalized embeddings
        - Triplet loss with semi-hard mining
        - Inception-ResNet-v1 architecture
        """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Image 1")
        uploaded_file1 = st.file_uploader(
            "Upload first face image",
            type=['jpg', 'jpeg', 'png'],
            key='image1'
        )
        
        if uploaded_file1:
            image1 = Image.open(uploaded_file1).convert('RGB')
            image1_display = resize_image(image1, max_size=400)
            st.image(image1_display, caption="Image 1")
    
    with col2:
        st.subheader("📷 Image 2")
        uploaded_file2 = st.file_uploader(
            "Upload second face image",
            type=['jpg', 'jpeg', 'png'],
            key='image2'
        )
        
        if uploaded_file2:
            image2 = Image.open(uploaded_file2).convert('RGB')
            image2_display = resize_image(image2, max_size=400)
            st.image(image2_display, caption="Image 2")
    
    # Compare button
    st.divider()
    
    if uploaded_file1 and uploaded_file2:
        if st.button("🔍 Compare Faces", type="primary", use_container_width=True):
            if not api_healthy:
                st.error("API is not available. Please start the backend server.")
            else:
                with st.spinner("Comparing faces..."):
                    # Reset file pointers
                    uploaded_file1.seek(0)
                    uploaded_file2.seek(0)
                    image1 = Image.open(uploaded_file1).convert('RGB')
                    image2 = Image.open(uploaded_file2).convert('RGB')
                    
                    result = compare_faces_api(image1, image2)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display results
                    st.subheader("📊 Results")
                    
                    # Main result
                    is_same = result['is_same_person']
                    confidence = result['confidence']
                    
                    if is_same:
                        st.success(f"✅ **Same Person** (Confidence: {confidence:.1%})")
                    else:
                        st.error(f"❌ **Different People** (Confidence: {confidence:.1%})")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("L2 Distance", f"{result['distance']:.4f}")
                    with col2:
                        st.metric("Similarity", f"{result['similarity']:.4f}")
                    with col3:
                        st.metric("Threshold", f"{result['threshold']:.4f}")
                    with col4:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Distance visualization
                    st.subheader("📈 Distance Visualization")
                    
                    # Progress bar showing distance relative to threshold
                    distance = result['distance']
                    threshold_val = result['threshold']
                    
                    # Normalize for visualization (0-2*threshold range)
                    max_display = 2 * threshold_val
                    progress = min(distance / max_display, 1.0)
                    
                    st.progress(progress)
                    
                    if distance < threshold_val:
                        st.caption(f"Distance ({distance:.4f}) < Threshold ({threshold_val:.4f}) → Same Person")
                    else:
                        st.caption(f"Distance ({distance:.4f}) ≥ Threshold ({threshold_val:.4f}) → Different People")
    else:
        st.info("👆 Upload two face images to compare them")
    
    # Footer
    st.divider()
    st.caption("Built with FaceNet (Schroff et al., 2015) | Streamlit + FastAPI")


if __name__ == "__main__":
    main()
