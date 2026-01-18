"""
Streamlit Frontend for FaceNet Face Recognition.

Features:
- Upload two face images
- Compare faces using the FaceNet model
- Display similarity score and verification result
- Search for matching identities in a gallery
"""

import os
import sys
from pathlib import Path
import io

import streamlit as st
import requests
from PIL import Image

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import resize_image

# Configuration
API_URL = os.environ.get("FACENET_API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(page_title="FaceNet Face Recognition", page_icon="👤", layout="wide")


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def load_model_via_api(model_path: str, threshold: float = 1.0):
    """Load model via API."""
    try:
        response = requests.post(
            f"{API_URL}/load_model",
            json={"model_path": model_path, "threshold": threshold},
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def compare_faces_api(image1: Image.Image, image2: Image.Image) -> dict:
    """Compare two faces via API."""
    # Convert images to bytes
    buf1 = io.BytesIO()
    image1.save(buf1, format="JPEG")
    buf1.seek(0)

    buf2 = io.BytesIO()
    image2.save(buf2, format="JPEG")
    buf2.seek(0)

    files = {
        "image1": ("image1.jpg", buf1, "image/jpeg"),
        "image2": ("image2.jpg", buf2, "image/jpeg"),
    }

    try:
        response = requests.post(f"{API_URL}/compare", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_embedding_api(image: Image.Image) -> dict:
    """Get embedding via API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    files = {"image": ("image.jpg", buf, "image/jpeg")}

    try:
        response = requests.post(f"{API_URL}/embed", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def search_identity_api(
    image: Image.Image, top_n: int = 5, gallery_path: str = "data/scia_images"
) -> dict:
    """Search for matching identities via API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    files = {"image": ("image.jpg", buf, "image/jpeg")}
    data = {"top_n": top_n, "gallery_path": gallery_path}

    try:
        response = requests.post(f"{API_URL}/search", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_identities_api(gallery_path: str = "data/scia_images") -> dict:
    """Get list of all identities via API."""
    try:
        response = requests.get(
            f"{API_URL}/identities", params={"gallery_path": gallery_path}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def load_image_from_path(image_path: str) -> Image.Image:
    """Load image from a file path."""
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def main():
    # Header
    st.title("👤 FaceNet Face Recognition")

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
            value="checkpoints/facenet_transfer_learning.pth",
            help="Path to the trained model checkpoint",
        )
        threshold = st.slider(
            "Threshold",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Distance threshold for same person decision",
        )

        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                result = load_model_via_api(model_path, threshold)
                if result.get("success"):
                    st.success("Model loaded!")
                else:
                    st.error(f"Failed: {result.get('message', 'Unknown error')}")

    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Compare Faces", "🔎 Identity Search"])

    # Tab 1: Face Comparison
    with tab1:
        st.markdown("""
        Upload two face images to compare them using FaceNet embeddings.
        The model generates 512-dimensional face embeddings and computes 
        the L2 distance to determine if they are the same person.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Image 1")
            uploaded_file1 = st.file_uploader(
                "Upload first face image",
                type=["jpg", "jpeg", "png"],
                key="compare_image1",
            )

            if uploaded_file1:
                image1 = Image.open(uploaded_file1).convert("RGB")
                image1_display = resize_image(image1, max_size=400)
                st.image(image1_display, caption="Image 1")

        with col2:
            st.subheader("📷 Image 2")
            uploaded_file2 = st.file_uploader(
                "Upload second face image",
                type=["jpg", "jpeg", "png"],
                key="compare_image2",
            )

            if uploaded_file2:
                image2 = Image.open(uploaded_file2).convert("RGB")
                image2_display = resize_image(image2, max_size=400)
                st.image(image2_display, caption="Image 2")

        # Compare button
        st.divider()

        if uploaded_file1 and uploaded_file2:
            if st.button(
                "🔍 Compare Faces",
                type="primary",
                use_container_width=True,
                key="compare_btn",
            ):
                if not api_healthy:
                    st.error("API is not available. Please start the backend server.")
                else:
                    with st.spinner("Comparing faces..."):
                        # Reset file pointers
                        uploaded_file1.seek(0)
                        uploaded_file2.seek(0)
                        image1 = Image.open(uploaded_file1).convert("RGB")
                        image2 = Image.open(uploaded_file2).convert("RGB")

                        result = compare_faces_api(image1, image2)

                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Display results
                        st.subheader("📊 Results")

                        # Main result
                        is_same = result["is_same_person"]
                        confidence = result["confidence"]

                        if is_same:
                            st.success(
                                f"✅ **Same Person** (Confidence: {confidence:.1%})"
                            )
                        else:
                            st.error(
                                f"❌ **Different People** (Confidence: {confidence:.1%})"
                            )

                        # Metrics
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

                        with m_col1:
                            st.metric("L2 Distance", f"{result['distance']:.4f}")
                        with m_col2:
                            st.metric("Similarity", f"{result['similarity']:.4f}")
                        with m_col3:
                            st.metric("Threshold", f"{result['threshold']:.4f}")
                        with m_col4:
                            st.metric("Confidence", f"{confidence:.1%}")

                        # Distance visualization
                        st.subheader("📈 Distance Visualization")

                        # Progress bar showing distance relative to threshold
                        distance = result["distance"]
                        threshold_val = result["threshold"]

                        # Normalize for visualization (0-2*threshold range)
                        max_display = 2 * threshold_val
                        progress = min(distance / max_display, 1.0)

                        st.progress(progress)

                        if distance < threshold_val:
                            st.caption(
                                f"Distance ({distance:.4f}) < Threshold ({threshold_val:.4f}) → Same Person"
                            )
                        else:
                            st.caption(
                                f"Distance ({distance:.4f}) ≥ Threshold ({threshold_val:.4f}) → Different People"
                            )
        else:
            st.info("👆 Upload two face images to compare them")

    # Tab 2: Identity Search
    with tab2:
        st.markdown("""
        Upload a face image to find the most similar identities from the gallery.
        The model will compare your image against all known identities and return the top matches.
        """)

        col_upload, col_settings = st.columns([2, 1])

        with col_upload:
            st.subheader("📷 Query Image")
            query_file = st.file_uploader(
                "Upload a face image to search",
                type=["jpg", "jpeg", "png"],
                key="search_image",
            )

            if query_file:
                query_image = Image.open(query_file).convert("RGB")
                query_display = resize_image(query_image, max_size=300)
                st.image(query_display, caption="Query Image")

        with col_settings:
            st.subheader("⚙️ Search Settings")
            top_n = st.slider(
                "Number of matches",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of top matching identities to return",
            )
            gallery_path = st.text_input(
                "Gallery Path",
                value="data/scia_images",
                help="Path to the gallery directory with identity folders",
            )

        st.divider()

        if query_file:
            if st.button(
                "🔎 Search Identities",
                type="primary",
                use_container_width=True,
                key="search_btn",
            ):
                if not api_healthy:
                    st.error("API is not available. Please start the backend server.")
                else:
                    with st.spinner("Searching for matching identities..."):
                        query_file.seek(0)
                        query_image = Image.open(query_file).convert("RGB")
                        result = search_identity_api(
                            query_image, top_n=top_n, gallery_path=gallery_path
                        )

                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    elif "detail" in result:
                        st.error(f"Error: {result['detail']}")
                    else:
                        st.subheader(f"🎯 Top {len(result['matches'])} Matches")

                        if not result["matches"]:
                            st.warning("No matches found in the gallery.")
                        else:
                            # Display matches in a grid
                            for i, match in enumerate(result["matches"]):
                                with st.container():
                                    match_col1, match_col2 = st.columns([1, 2])

                                    with match_col1:
                                        # Load and display the match image
                                        try:
                                            match_img = Image.open(
                                                match["image_path"]
                                            ).convert("RGB")
                                            match_img_display = resize_image(
                                                match_img, max_size=150
                                            )
                                            st.image(match_img_display)
                                        except Exception as e:
                                            st.error(
                                                f"Could not load image: {match['image_path']}"
                                            )

                                    with match_col2:
                                        # Format identity name nicely
                                        identity_display = (
                                            match["identity"]
                                            .replace(".", " ")
                                            .replace("-", " ")
                                            .title()
                                        )

                                        # Show match status
                                        if match["is_match"]:
                                            st.success(
                                                f"**#{i + 1} {identity_display}** ✅"
                                            )
                                        else:
                                            st.info(f"**#{i + 1} {identity_display}**")

                                        # Metrics in columns
                                        metric_col1, metric_col2, metric_col3 = (
                                            st.columns(3)
                                        )
                                        with metric_col1:
                                            st.metric(
                                                "Distance", f"{match['distance']:.4f}"
                                            )
                                        with metric_col2:
                                            st.metric(
                                                "Similarity",
                                                f"{match['similarity']:.4f}",
                                            )
                                        with metric_col3:
                                            st.metric(
                                                "Confidence",
                                                f"{match['confidence']:.1%}",
                                            )

                                        st.caption(
                                            f"📸 {match['num_images']} image(s) in gallery"
                                        )

                                    st.divider()
        else:
            st.info("👆 Upload a face image to search for matching identities")

    # Footer
    st.divider()
    st.caption("Built with FaceNet (Schroff et al., 2015) | Streamlit + FastAPI")


if __name__ == "__main__":
    main()
