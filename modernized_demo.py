#!/usr/bin/env python3
"""
Modernized semantic segmentation implementation.

This is the updated version of the original 0209.py script with modern
practices, error handling, and enhanced functionality.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from src.segmentation_model import SemanticSegmentationModel
from src.visualization import visualize_segmentation, create_color_map
from src.data_utils import load_sample_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(image_path: Optional[str] = None):
    """
    Main function demonstrating semantic segmentation.
    
    Args:
        image_path: Path to input image. If None, uses sample image.
    """
    logger.info("Starting semantic segmentation demo...")
    
    try:
        # Initialize the modern segmentation model
        logger.info("Loading FCN ResNet-50 model...")
        model = SemanticSegmentationModel(model_name="fcn_resnet50")
        
        # Get model information
        model_info = model.get_model_info()
        logger.info(f"Model loaded: {model_info['model_name']} on {model_info['device']}")
        logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        
        # Load input image
        if image_path and Path(image_path).exists():
            logger.info(f"Loading image: {image_path}")
            input_image = Image.open(image_path).convert('RGB')
        else:
            if image_path:
                logger.warning(f"Image not found: {image_path}, using sample image")
            else:
                logger.info("Using sample image")
            input_image = load_sample_image()
        
        logger.info(f"Image size: {input_image.size}")
        
        # Perform semantic segmentation
        logger.info("Performing semantic segmentation...")
        results = model.predict(input_image)
        segmentation = results['segmentation']
        confidence = results['confidence']
        
        # Get class statistics
        class_counts = model.get_class_counts(segmentation)
        logger.info(f"Segmentation completed. Found {len(class_counts)} classes:")
        
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / segmentation.numel()) * 100
            logger.info(f"  {class_name}: {count:,} pixels ({percentage:.1f}%)")
        
        # Create comprehensive visualization
        logger.info("Creating visualization...")
        fig = visualize_segmentation(
            input_image,
            segmentation,
            class_names=model.class_names,
            alpha=0.6
        )
        
        # Save results
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save visualization
        viz_path = output_dir / "segmentation_result.png"
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {viz_path}")
        
        # Save segmentation mask
        mask_path = output_dir / "segmentation_mask.png"
        mask_image = Image.fromarray(segmentation.numpy().astype(np.uint8))
        mask_image.save(mask_path)
        logger.info(f"Segmentation mask saved to: {mask_path}")
        
        # Show confidence statistics
        avg_confidence = confidence.mean().item()
        min_confidence = confidence.min().item()
        max_confidence = confidence.max().item()
        
        logger.info(f"Confidence statistics:")
        logger.info(f"  Average: {avg_confidence:.3f}")
        logger.info(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")
        
        # Display the results
        plt.show()
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        sys.exit(1)


def create_legacy_color_map(seg_map: np.ndarray) -> np.ndarray:
    """
    Legacy color map function for backward compatibility.
    
    Args:
        seg_map: Segmentation map
        
    Returns:
        Colored segmentation image
    """
    logger.warning("Using legacy color map function. Consider using create_color_map() instead.")
    
    # Use the modern color map function
    num_classes = seg_map.max() + 1
    color_map = create_color_map(num_classes, seed=10)
    
    h, w = seg_map.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label in range(num_classes):
        color_seg[seg_map == label] = color_map[label]
    
    return color_seg


if __name__ == "__main__":
    # Allow command line argument for image path
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(image_path)
