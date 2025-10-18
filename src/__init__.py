"""
Semantic Segmentation Implementation

A modern, production-ready semantic segmentation system using PyTorch and torchvision.
Supports multiple models, visualization, and easy integration with web interfaces.
"""

__version__ = "1.0.0"
__author__ = "AI Projects Team"
__email__ = "ai-projects@example.com"

from .segmentation_model import SemanticSegmentationModel
from .data_utils import create_synthetic_dataset, load_sample_image
from .visualization import visualize_segmentation, create_color_map

__all__ = [
    "SemanticSegmentationModel",
    "create_synthetic_dataset", 
    "load_sample_image",
    "visualize_segmentation",
    "create_color_map"
]
