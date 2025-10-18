"""
Visualization utilities for semantic segmentation.

This module provides functions for visualizing segmentation results,
creating color maps, and generating comparison plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def create_color_map(
    num_classes: int,
    seed: int = 42
) -> np.ndarray:
    """
    Create a consistent color map for segmentation classes.
    
    Args:
        num_classes: Number of classes to create colors for
        seed: Random seed for reproducible colors
        
    Returns:
        Array of RGB colors (num_classes, 3)
    """
    np.random.seed(seed)
    
    # Use distinct colors for better visualization
    colors = []
    
    # Predefined colors for common classes
    predefined_colors = [
        [0, 0, 0],        # Background - black
        [128, 0, 0],     # Person - dark red
        [0, 128, 0],     # Vehicle - dark green
        [0, 0, 128],     # Road - dark blue
        [128, 128, 0],   # Building - olive
        [128, 0, 128],   # Sky - purple
        [0, 128, 128],   # Vegetation - teal
        [192, 192, 192], # Sidewalk - silver
    ]
    
    # Use predefined colors first, then generate random ones
    for i in range(num_classes):
        if i < len(predefined_colors):
            colors.append(predefined_colors[i])
        else:
            # Generate random but distinct colors
            color = np.random.randint(50, 255, 3)
            colors.append(color.tolist())
    
    return np.array(colors, dtype=np.uint8)


def apply_color_map(
    segmentation: Union[torch.Tensor, np.ndarray],
    color_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply a color map to a segmentation mask.
    
    Args:
        segmentation: Segmentation mask (H, W) with class indices
        color_map: Color map array (num_classes, 3). If None, creates one.
        
    Returns:
        Colored segmentation image (H, W, 3)
    """
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.numpy()
    
    if color_map is None:
        num_classes = int(segmentation.max()) + 1
        color_map = create_color_map(num_classes)
    
    h, w = segmentation.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(color_map)):
        mask = segmentation == class_id
        colored_seg[mask] = color_map[class_id]
    
    return colored_seg


def visualize_segmentation(
    image: Union[str, Path, Image.Image, np.ndarray],
    segmentation: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    color_map: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
    alpha: float = 0.6
) -> plt.Figure:
    """
    Create a comprehensive visualization of segmentation results.
    
    Args:
        image: Original input image
        segmentation: Segmentation mask
        class_names: List of class names
        color_map: Color map for visualization
        save_path: Path to save the figure
        figsize: Figure size (width, height)
        alpha: Transparency for overlay
        
    Returns:
        Matplotlib figure object
    """
    # Load and prepare image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image_array = np.array(image)
    
    # Prepare segmentation
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.numpy()
    
    # Create color map if not provided
    if color_map is None:
        num_classes = int(segmentation.max()) + 1
        color_map = create_color_map(num_classes)
    
    # Apply color map
    colored_seg = apply_color_map(segmentation, color_map)
    
    # Create overlay
    overlay = image_array.copy()
    mask = segmentation > 0  # Exclude background
    overlay[mask] = (
        alpha * colored_seg[mask] + 
        (1 - alpha) * image_array[mask]
    ).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image
    axes[0].imshow(image_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(colored_seg)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Class distribution
    unique_classes, counts = np.unique(segmentation, return_counts=True)
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_classes]
    
    # Filter class names to match unique classes
    filtered_names = [class_names[i] if i < len(class_names) else f'Class {i}' 
                     for i in unique_classes]
    
    axes[3].bar(range(len(unique_classes)), counts)
    axes[3].set_title('Class Distribution')
    axes[3].set_xlabel('Classes')
    axes[3].set_ylabel('Pixel Count')
    axes[3].set_xticks(range(len(unique_classes)))
    axes[3].set_xticklabels(filtered_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    return fig


def create_comparison_plot(
    images: List[Union[str, Path, Image.Image, np.ndarray]],
    segmentations: List[Union[torch.Tensor, np.ndarray]],
    titles: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """
    Create a comparison plot of multiple segmentation results.
    
    Args:
        images: List of input images
        segmentations: List of segmentation masks
        titles: List of titles for each subplot
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    num_samples = len(images)
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        
        # Load image
        if isinstance(images[i], (str, Path)):
            image = Image.open(images[i]).convert('RGB')
        elif isinstance(images[i], np.ndarray):
            image = Image.fromarray(images[i])
        else:
            image = images[i]
        
        image_array = np.array(image)
        
        # Prepare segmentation
        if isinstance(segmentations[i], torch.Tensor):
            seg = segmentations[i].numpy()
        else:
            seg = segmentations[i]
        
        # Apply color map
        colored_seg = apply_color_map(seg)
        
        # Create overlay
        overlay = image_array.copy()
        mask = seg > 0
        overlay[mask] = (0.6 * colored_seg[mask] + 0.4 * image_array[mask]).astype(np.uint8)
        
        # Plot
        axes[row, col].imshow(overlay)
        title = titles[i] if titles and i < len(titles) else f'Sample {i+1}'
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    return fig


def save_segmentation_results(
    image: Union[str, Path, Image.Image, np.ndarray],
    segmentation: Union[torch.Tensor, np.ndarray],
    output_dir: Union[str, Path],
    prefix: str = "result",
    color_map: Optional[np.ndarray] = None
) -> Dict[str, Path]:
    """
    Save segmentation results to files.
    
    Args:
        image: Original input image
        segmentation: Segmentation mask
        output_dir: Directory to save results
        prefix: Prefix for output filenames
        color_map: Color map for visualization
        
    Returns:
        Dictionary mapping result type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Prepare segmentation
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.numpy()
    
    # Create color map
    if color_map is None:
        num_classes = int(segmentation.max()) + 1
        color_map = create_color_map(num_classes)
    
    # Apply color map
    colored_seg = apply_color_map(segmentation, color_map)
    
    # Save files
    results = {}
    
    # Save original image
    orig_path = output_path / f"{prefix}_original.png"
    image.save(orig_path)
    results['original'] = orig_path
    
    # Save segmentation mask (as grayscale)
    mask_path = output_path / f"{prefix}_mask.png"
    mask_image = Image.fromarray(segmentation.astype(np.uint8))
    mask_image.save(mask_path)
    results['mask'] = mask_path
    
    # Save colored segmentation
    colored_path = output_path / f"{prefix}_colored.png"
    colored_image = Image.fromarray(colored_seg)
    colored_image.save(colored_path)
    results['colored'] = colored_path
    
    # Save overlay
    overlay_path = output_path / f"{prefix}_overlay.png"
    overlay = np.array(image)
    mask = segmentation > 0
    overlay[mask] = (0.6 * colored_seg[mask] + 0.4 * overlay[mask]).astype(np.uint8)
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(overlay_path)
    results['overlay'] = overlay_path
    
    logger.info(f"Segmentation results saved to {output_path}")
    return results
