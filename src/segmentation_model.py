"""
Core semantic segmentation model implementation.

This module provides a modern, type-safe implementation of semantic segmentation
using PyTorch and torchvision models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSegmentationModel:
    """
    A modern semantic segmentation model wrapper with enhanced functionality.
    
    This class provides a clean interface for semantic segmentation using
    pre-trained models from torchvision, with support for multiple architectures
    and enhanced visualization capabilities.
    
    Attributes:
        model: The underlying PyTorch segmentation model
        device: The device (CPU/GPU) the model is running on
        transform: Image preprocessing pipeline
        class_names: List of class names for the model
    """
    
    def __init__(
        self,
        model_name: str = "fcn_resnet50",
        device: Optional[str] = None,
        weights: Optional[str] = None
    ) -> None:
        """
        Initialize the semantic segmentation model.
        
        Args:
            model_name: Name of the model architecture to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
            weights: Specific weights to load (None for default)
        """
        self.device = self._setup_device(device)
        self.model_name = model_name
        self.model = self._load_model(model_name, weights)
        self.transform = self._create_transform()
        self.class_names = self._get_class_names()
        
        logger.info(f"Initialized {model_name} on {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup the computation device."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _load_model(self, model_name: str, weights: Optional[str]) -> nn.Module:
        """Load the specified segmentation model."""
        try:
            # Map model names to actual model constructors
            model_constructors = {
                "fcn_resnet50": models.segmentation.fcn_resnet50,
                "fcn_resnet101": models.segmentation.fcn_resnet101,
                "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50,
                "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101,
                "lraspp_mobilenet_v3_large": models.segmentation.lraspp_mobilenet_v3_large,
            }
            
            if model_name not in model_constructors:
                raise ValueError(f"Unsupported model: {model_name}")
            
            constructor = model_constructors[model_name]
            
            # Use modern weights parameter if available, fallback to pretrained
            try:
                if weights:
                    model = constructor(weights=weights)
                else:
                    # Use default weights for the model
                    model = constructor(weights="DEFAULT")
            except TypeError:
                # Fallback for older torchvision versions
                model = constructor(pretrained=True)
            
            model.eval()
            model.to(self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _create_transform(self) -> transforms.Compose:
        """Create the image preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_class_names(self) -> List[str]:
        """Get class names for the model."""
        # COCO classes for segmentation
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for segmentation.
        
        Args:
            image: Path to image file, PIL Image, or image array
            
        Returns:
            Preprocessed image tensor ready for model inference
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Apply transforms and add batch dimension
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image: Union[str, Path, Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: Input image (path, PIL Image, or array)
            
        Returns:
            Dictionary containing segmentation results
        """
        # Preprocess the image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Handle different model output formats
            if isinstance(output, dict):
                if 'out' in output:
                    segmentation_logits = output['out']
                elif 'logits' in output:
                    segmentation_logits = output['logits']
                else:
                    # Take the first tensor if it's a dict
                    segmentation_logits = list(output.values())[0]
            else:
                segmentation_logits = output
            
            # Get predicted classes
            predicted_classes = segmentation_logits.argmax(dim=1)
            
            # Get confidence scores
            probabilities = torch.softmax(segmentation_logits, dim=1)
            confidence = probabilities.max(dim=1)[0]
        
        return {
            'segmentation': predicted_classes.squeeze(0).cpu(),
            'logits': segmentation_logits.squeeze(0).cpu(),
            'confidence': confidence.squeeze(0).cpu(),
            'probabilities': probabilities.squeeze(0).cpu()
        }
    
    def get_class_counts(self, segmentation: torch.Tensor) -> Dict[str, int]:
        """
        Count pixels for each class in the segmentation.
        
        Args:
            segmentation: Segmentation tensor
            
        Returns:
            Dictionary mapping class names to pixel counts
        """
        unique_classes, counts = torch.unique(segmentation, return_counts=True)
        
        class_counts = {}
        for class_id, count in zip(unique_classes.tolist(), counts.tolist()):
            if class_id < len(self.class_names):
                class_counts[self.class_names[class_id]] = count
        
        return class_counts
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': len(self.class_names)
        }
