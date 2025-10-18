"""
Test suite for semantic segmentation implementation.

This module contains comprehensive tests for all major components
of the semantic segmentation system.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from src.segmentation_model import SemanticSegmentationModel
from src.data_utils import (
    SyntheticSegmentationDataset, 
    create_synthetic_dataset,
    load_sample_image
)
from src.visualization import (
    create_color_map,
    apply_color_map,
    visualize_segmentation
)
from src.config import Config


class TestSemanticSegmentationModel:
    """Test cases for SemanticSegmentationModel."""
    
    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        # Test default initialization
        model = SemanticSegmentationModel()
        assert model.model_name == "fcn_resnet50"
        assert model.device.type in ["cpu", "cuda"]
        
        # Test with different model
        model = SemanticSegmentationModel(model_name="fcn_resnet101")
        assert model.model_name == "fcn_resnet101"
    
    def test_model_info(self):
        """Test model information retrieval."""
        model = SemanticSegmentationModel()
        info = model.get_model_info()
        
        assert "model_name" in info
        assert "device" in info
        assert "total_parameters" in info
        assert "num_classes" in info
        assert info["total_parameters"] > 0
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        model = SemanticSegmentationModel()
        
        # Test with PIL Image
        image = Image.new('RGB', (256, 256), color=(255, 0, 0))
        tensor = model.preprocess_image(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 520, 520)
        assert tensor.device == model.device
    
    def test_predict(self):
        """Test model prediction."""
        model = SemanticSegmentationModel()
        
        # Create a test image
        image = Image.new('RGB', (256, 256), color=(255, 0, 0))
        
        results = model.predict(image)
        
        assert "segmentation" in results
        assert "logits" in results
        assert "confidence" in results
        assert "probabilities" in results
        
        # Check output shapes
        assert results["segmentation"].shape == (520, 520)
        assert results["logits"].shape[0] == model.get_model_info()["num_classes"]
    
    def test_class_counts(self):
        """Test class counting functionality."""
        model = SemanticSegmentationModel()
        
        # Create a mock segmentation tensor
        segmentation = torch.zeros((10, 10), dtype=torch.long)
        segmentation[0:5, 0:5] = 1  # Class 1
        segmentation[5:10, 5:10] = 2  # Class 2
        
        counts = model.get_class_counts(segmentation)
        
        assert isinstance(counts, dict)
        assert len(counts) > 0


class TestSyntheticDataset:
    """Test cases for synthetic dataset generation."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = SyntheticSegmentationDataset(size=10)
        
        assert len(dataset) == 10
        assert dataset.image_size == (512, 512)
        assert dataset.num_classes == 5
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = SyntheticSegmentationDataset(size=5)
        
        image, mask = dataset[0]
        
        assert isinstance(image, Image.Image)
        assert isinstance(mask, Image.Image)
        assert image.size == (512, 512)
        assert mask.size == (512, 512)
        assert image.mode == 'RGB'
        assert mask.mode == 'L'
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = create_synthetic_dataset(
                output_dir=temp_dir,
                num_images=5,
                image_size=(256, 256),
                num_classes=3
            )
            
            assert dataset_path.exists()
            assert (dataset_path / "images").exists()
            assert (dataset_path / "masks").exists()
            
            # Check that images were created
            images_dir = dataset_path / "images"
            masks_dir = dataset_path / "masks"
            
            assert len(list(images_dir.glob("*.png"))) == 5
            assert len(list(masks_dir.glob("*.png"))) == 5


class TestVisualization:
    """Test cases for visualization functions."""
    
    def test_create_color_map(self):
        """Test color map creation."""
        color_map = create_color_map(num_classes=5)
        
        assert color_map.shape == (5, 3)
        assert color_map.dtype == np.uint8
        assert np.all(color_map >= 0) and np.all(color_map <= 255)
    
    def test_apply_color_map(self):
        """Test color map application."""
        # Create a simple segmentation mask
        segmentation = np.zeros((10, 10), dtype=np.uint8)
        segmentation[0:5, 0:5] = 1
        segmentation[5:10, 5:10] = 2
        
        color_map = create_color_map(num_classes=3)
        colored_seg = apply_color_map(segmentation, color_map)
        
        assert colored_seg.shape == (10, 10, 3)
        assert colored_seg.dtype == np.uint8
    
    def test_visualize_segmentation(self):
        """Test segmentation visualization."""
        # Create test image and segmentation
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[0:50, 0:50] = 1
        
        fig = visualize_segmentation(image, segmentation)
        
        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots


class TestDataUtils:
    """Test cases for data utilities."""
    
    def test_load_sample_image(self):
        """Test sample image loading."""
        image = load_sample_image()
        
        assert isinstance(image, Image.Image)
        assert image.size == (512, 512)
        assert image.mode == 'RGB'


class TestConfig:
    """Test cases for configuration management."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = Config()
        
        assert isinstance(config.to_dict(), dict)
    
    def test_config_get(self):
        """Test configuration value retrieval."""
        config = Config()
        
        # Test with default value
        value = config.get("nonexistent.key", "default")
        assert value == "default"
    
    def test_model_config(self):
        """Test model configuration retrieval."""
        config = Config()
        model_config = config.get_model_config()
        
        assert isinstance(model_config, dict)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Create model
        model = SemanticSegmentationModel()
        
        # Create synthetic image
        image = load_sample_image()
        
        # Perform segmentation
        results = model.predict(image)
        
        # Visualize results
        fig = visualize_segmentation(
            image, 
            results['segmentation'],
            class_names=model.class_names
        )
        
        assert fig is not None
        
        # Test class counts
        counts = model.get_class_counts(results['segmentation'])
        assert isinstance(counts, dict)
    
    def test_synthetic_data_pipeline(self):
        """Test synthetic data generation and processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create synthetic dataset
            dataset_path = create_synthetic_dataset(
                output_dir=temp_dir,
                num_images=3
            )
            
            # Load and process images
            images_dir = dataset_path / "images"
            image_path = list(images_dir.glob("*.png"))[0]
            
            model = SemanticSegmentationModel()
            results = model.predict(image_path)
            
            assert results['segmentation'].shape == (520, 520)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
