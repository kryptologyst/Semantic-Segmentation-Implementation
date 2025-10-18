"""
Data utilities for semantic segmentation.

This module provides functions for creating synthetic datasets,
loading sample images, and data preprocessing utilities.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SyntheticSegmentationDataset(Dataset):
    """
    A synthetic dataset generator for semantic segmentation.
    
    Creates artificial images with geometric shapes and corresponding
    segmentation masks for testing and demonstration purposes.
    """
    
    def __init__(
        self,
        size: int = 100,
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 5
    ):
        """
        Initialize the synthetic dataset.
        
        Args:
            size: Number of synthetic images to generate
            image_size: Size of generated images (height, width)
            num_classes: Number of segmentation classes
        """
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_names = [
            'background', 'rectangle', 'circle', 'triangle', 'text'
        ][:num_classes]
        
        logger.info(f"Created synthetic dataset with {size} images")
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        """
        Generate a synthetic image and its segmentation mask.
        
        Args:
            idx: Index of the sample to generate
            
        Returns:
            Tuple of (image, segmentation_mask)
        """
        # Set random seed for reproducibility
        np.random.seed(idx)
        
        # Create base image and mask
        image = Image.new('RGB', self.image_size, color=(50, 50, 50))
        mask = Image.new('L', self.image_size, color=0)  # Background class
        
        draw_img = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)
        
        # Generate random shapes
        num_shapes = np.random.randint(3, 8)
        
        for _ in range(num_shapes):
            shape_type = np.random.randint(1, self.num_classes)
            color = self._get_random_color()
            
            if shape_type == 1:  # Rectangle
                self._draw_rectangle(draw_img, draw_mask, color, shape_type)
            elif shape_type == 2:  # Circle
                self._draw_circle(draw_img, draw_mask, color, shape_type)
            elif shape_type == 3:  # Triangle
                self._draw_triangle(draw_img, draw_mask, color, shape_type)
            elif shape_type == 4:  # Text
                self._draw_text(draw_img, draw_mask, color, shape_type)
        
        return image, mask
    
    def _get_random_color(self) -> Tuple[int, int, int]:
        """Generate a random RGB color."""
        return (
            np.random.randint(100, 255),
            np.random.randint(100, 255),
            np.random.randint(100, 255)
        )
    
    def _draw_rectangle(
        self,
        draw_img: ImageDraw.Draw,
        draw_mask: ImageDraw.Draw,
        color: Tuple[int, int, int],
        class_id: int
    ) -> None:
        """Draw a random rectangle."""
        x1 = np.random.randint(0, self.image_size[1] - 100)
        y1 = np.random.randint(0, self.image_size[0] - 100)
        x2 = x1 + np.random.randint(50, 150)
        y2 = y1 + np.random.randint(50, 150)
        
        draw_img.rectangle([x1, y1, x2, y2], fill=color)
        draw_mask.rectangle([x1, y1, x2, y2], fill=class_id)
    
    def _draw_circle(
        self,
        draw_img: ImageDraw.Draw,
        draw_mask: ImageDraw.Draw,
        color: Tuple[int, int, int],
        class_id: int
    ) -> None:
        """Draw a random circle."""
        center_x = np.random.randint(50, self.image_size[1] - 50)
        center_y = np.random.randint(50, self.image_size[0] - 50)
        radius = np.random.randint(20, 80)
        
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw_img.ellipse(bbox, fill=color)
        draw_mask.ellipse(bbox, fill=class_id)
    
    def _draw_triangle(
        self,
        draw_img: ImageDraw.Draw,
        draw_mask: ImageDraw.Draw,
        color: Tuple[int, int, int],
        class_id: int
    ) -> None:
        """Draw a random triangle."""
        x1 = np.random.randint(0, self.image_size[1] - 100)
        y1 = np.random.randint(0, self.image_size[0] - 100)
        x2 = x1 + np.random.randint(50, 150)
        y2 = y1 + np.random.randint(50, 150)
        x3 = x1 + np.random.randint(0, 100)
        y3 = y2
        
        points = [(x1, y1), (x2, y2), (x3, y3)]
        draw_img.polygon(points, fill=color)
        draw_mask.polygon(points, fill=class_id)
    
    def _draw_text(
        self,
        draw_img: ImageDraw.Draw,
        draw_mask: ImageDraw.Draw,
        color: Tuple[int, int, int],
        class_id: int
    ) -> None:
        """Draw random text."""
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        text = f"Text{np.random.randint(1, 100)}"
        x = np.random.randint(0, self.image_size[1] - 100)
        y = np.random.randint(0, self.image_size[0] - 50)
        
        draw_img.text((x, y), text, fill=color, font=font)
        
        # Approximate text bounding box for mask
        bbox = draw_img.textbbox((x, y), text, font=font)
        draw_mask.rectangle(bbox, fill=class_id)


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_images: int = 50,
    image_size: Tuple[int, int] = (512, 512),
    num_classes: int = 5
) -> Path:
    """
    Create a synthetic segmentation dataset and save it to disk.
    
    Args:
        output_dir: Directory to save the dataset
        num_images: Number of images to generate
        image_size: Size of generated images
        num_classes: Number of segmentation classes
        
    Returns:
        Path to the created dataset directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    dataset = SyntheticSegmentationDataset(
        size=num_images,
        image_size=image_size,
        num_classes=num_classes
    )
    
    logger.info(f"Generating {num_images} synthetic images...")
    
    for i in range(num_images):
        image, mask = dataset[i]
        
        # Save image and mask
        image.save(images_dir / f"image_{i:04d}.png")
        mask.save(masks_dir / f"mask_{i:04d}.png")
    
    logger.info(f"Dataset saved to {output_path}")
    return output_path


def load_sample_image() -> Image.Image:
    """
    Load a sample image for testing.
    
    Returns:
        PIL Image object
    """
    # Create a simple test image
    image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    draw = ImageDraw.Draw(image)
    
    # Draw some shapes
    draw.rectangle([50, 50, 200, 200], fill=(255, 0, 0))  # Red rectangle
    draw.ellipse([300, 100, 450, 250], fill=(0, 255, 0))  # Green circle
    draw.polygon([(100, 300), (200, 300), (150, 400)], fill=(0, 0, 255))  # Blue triangle
    
    return image


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for the loader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
