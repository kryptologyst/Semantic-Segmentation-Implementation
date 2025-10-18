"""
Streamlit web interface for semantic segmentation.

This module provides a user-friendly web interface for testing and
demonstrating the semantic segmentation capabilities.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.segmentation_model import SemanticSegmentationModel
from src.data_utils import create_synthetic_dataset, load_sample_image
from src.visualization import visualize_segmentation, create_color_map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Semantic Segmentation Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str = "fcn_resnet50") -> SemanticSegmentationModel:
    """Load and cache the segmentation model."""
    try:
        model = SemanticSegmentationModel(model_name=model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Semantic Segmentation Demo</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases modern semantic segmentation using PyTorch and torchvision.
    Upload an image or use our synthetic dataset to see how the model segments different objects.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "FCN ResNet-50": "fcn_resnet50",
            "FCN ResNet-101": "fcn_resnet101", 
            "DeepLabV3 ResNet-50": "deeplabv3_resnet50",
            "DeepLabV3 ResNet-101": "deeplabv3_resnet101",
            "LRASPP MobileNet V3": "lraspp_mobilenet_v3_large"
        }
        
        selected_model = st.selectbox(
            "Select Model Architecture",
            options=list(model_options.keys()),
            index=0
        )
        
        model_name = model_options[selected_model]
        
        # Device selection
        device_options = ["Auto", "CPU", "GPU"]
        device_choice = st.selectbox("Device", device_options, index=0)
        
        device = None if device_choice == "Auto" else device_choice.lower()
        
        # Visualization options
        st.header("🎨 Visualization Options")
        alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.6, 0.1)
        show_confidence = st.checkbox("Show Confidence Scores", value=False)
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(model_name)
        
        if model is None:
            st.error("Failed to load model. Please try again.")
            return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Input Image")
        
        # Image input options
        input_option = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Sample Image", "Generate Synthetic Data"]
        )
        
        input_image = None
        
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image to perform semantic segmentation"
            )
            
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert('RGB')
                st.image(input_image, caption="Uploaded Image", use_column_width=True)
        
        elif input_option == "Use Sample Image":
            if st.button("Load Sample Image"):
                input_image = load_sample_image()
                st.image(input_image, caption="Sample Image", use_column_width=True)
        
        elif input_option == "Generate Synthetic Data":
            if st.button("Generate Synthetic Image"):
                # Create a temporary synthetic dataset
                with tempfile.TemporaryDirectory() as temp_dir:
                    create_synthetic_dataset(temp_dir, num_images=1)
                    image_path = Path(temp_dir) / "images" / "image_0000.png"
                    input_image = Image.open(image_path)
                    st.image(input_image, caption="Synthetic Image", use_column_width=True)
    
    with col2:
        st.header("🎯 Segmentation Results")
        
        if input_image is not None:
            # Perform segmentation
            with st.spinner("Performing segmentation..."):
                try:
                    results = model.predict(input_image)
                    segmentation = results['segmentation']
                    confidence = results['confidence']
                    
                    # Display results
                    st.success("Segmentation completed successfully!")
                    
                    # Model info
                    model_info = model.get_model_info()
                    st.markdown("### Model Information")
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.metric("Model", model_info['model_name'])
                        st.metric("Device", model_info['device'])
                    
                    with col_info2:
                        st.metric("Parameters", f"{model_info['total_parameters']:,}")
                        st.metric("Classes", model_info['num_classes'])
                    
                    # Class distribution
                    class_counts = model.get_class_counts(segmentation)
                    st.markdown("### Class Distribution")
                    
                    if class_counts:
                        # Create a bar chart
                        classes = list(class_counts.keys())
                        counts = list(class_counts.values())
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.bar(range(len(classes)), counts)
                        ax.set_xlabel('Classes')
                        ax.set_ylabel('Pixel Count')
                        ax.set_title('Segmentation Class Distribution')
                        ax.set_xticks(range(len(classes)))
                        ax.set_xticklabels(classes, rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Visualization
                    st.markdown("### Visualization")
                    
                    # Create visualization
                    fig = visualize_segmentation(
                        input_image,
                        segmentation,
                        class_names=model.class_names,
                        alpha=alpha
                    )
                    st.pyplot(fig)
                    
                    # Confidence visualization
                    if show_confidence:
                        st.markdown("### Confidence Scores")
                        conf_fig, conf_ax = plt.subplots(figsize=(8, 6))
                        conf_im = conf_ax.imshow(confidence.numpy(), cmap='viridis')
                        conf_ax.set_title('Confidence Scores')
                        conf_ax.axis('off')
                        plt.colorbar(conf_im, ax=conf_ax)
                        st.pyplot(conf_fig)
                    
                    # Download results
                    st.markdown("### Download Results")
                    
                    # Convert segmentation to PIL Image for download
                    seg_image = Image.fromarray(segmentation.numpy().astype(np.uint8))
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        st.download_button(
                            label="Download Segmentation Mask",
                            data=seg_image.tobytes(),
                            file_name="segmentation_mask.png",
                            mime="image/png"
                        )
                    
                    with col_dl2:
                        # Create colored segmentation
                        color_map = create_color_map(len(model.class_names))
                        colored_seg = apply_color_map(segmentation.numpy(), color_map)
                        colored_image = Image.fromarray(colored_seg)
                        
                        st.download_button(
                            label="Download Colored Segmentation",
                            data=colored_image.tobytes(),
                            file_name="colored_segmentation.png",
                            mime="image/png"
                        )
                
                except Exception as e:
                    st.error(f"Segmentation failed: {e}")
                    logger.error(f"Segmentation error: {e}")
        
        else:
            st.info("Please select an input image to see segmentation results.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with PyTorch, Streamlit, and modern semantic segmentation models</p>
        <p>For more information, check out the project repository</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
