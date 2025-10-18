#!/usr/bin/env python3
"""
Command-line interface for semantic segmentation.

This script provides a CLI for running semantic segmentation
on images with various options and configurations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from src.segmentation_model import SemanticSegmentationModel
from src.visualization import visualize_segmentation, save_segmentation_results
from src.data_utils import create_synthetic_dataset, load_sample_image
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segment a single image
  python cli.py segment --input image.jpg --output results/

  # Use different model
  python cli.py segment --input image.jpg --model deeplabv3_resnet50

  # Generate synthetic dataset
  python cli.py generate-data --output data/synthetic --num-images 50

  # Show model information
  python cli.py info --model fcn_resnet101
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Perform semantic segmentation')
    segment_parser.add_argument('--input', '-i', required=True, help='Input image path')
    segment_parser.add_argument('--output', '-o', help='Output directory')
    segment_parser.add_argument('--model', '-m', default='fcn_resnet50', 
                               help='Model architecture')
    segment_parser.add_argument('--device', '-d', help='Device (cpu/cuda)')
    segment_parser.add_argument('--alpha', '-a', type=float, default=0.6,
                               help='Overlay transparency')
    segment_parser.add_argument('--save-all', action='store_true',
                               help='Save all visualization types')
    
    # Generate data command
    data_parser = subparsers.add_parser('generate-data', help='Generate synthetic dataset')
    data_parser.add_argument('--output', '-o', required=True, help='Output directory')
    data_parser.add_argument('--num-images', '-n', type=int, default=50,
                            help='Number of images to generate')
    data_parser.add_argument('--image-size', type=int, nargs=2, default=[512, 512],
                            help='Image size (height width)')
    data_parser.add_argument('--num-classes', type=int, default=5,
                            help='Number of classes')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', '-m', default='fcn_resnet50',
                            help='Model architecture')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with sample image')
    demo_parser.add_argument('--model', '-m', default='fcn_resnet50',
                            help='Model architecture')
    demo_parser.add_argument('--output', '-o', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'segment':
            run_segmentation(args)
        elif args.command == 'generate-data':
            run_data_generation(args)
        elif args.command == 'info':
            run_model_info(args)
        elif args.command == 'demo':
            run_demo(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def run_segmentation(args):
    """Run semantic segmentation on input image."""
    logger.info(f"Loading model: {args.model}")
    
    # Initialize model
    model = SemanticSegmentationModel(
        model_name=args.model,
        device=args.device
    )
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Processing image: {input_path}")
    
    # Perform segmentation
    results = model.predict(input_path)
    segmentation = results['segmentation']
    
    # Show basic info
    class_counts = model.get_class_counts(segmentation)
    logger.info(f"Detected {len(class_counts)} classes")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {class_name}: {count:,} pixels")
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        if args.save_all:
            saved_files = save_segmentation_results(
                input_path, segmentation, output_dir
            )
            logger.info(f"Results saved to {output_dir}")
            for result_type, file_path in saved_files.items():
                logger.info(f"  {result_type}: {file_path}")
        else:
            # Just save visualization
            fig = visualize_segmentation(
                input_path,
                segmentation,
                class_names=model.class_names,
                alpha=args.alpha
            )
            viz_path = output_dir / f"segmentation_result.png"
            fig.savefig(viz_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {viz_path}")
    else:
        # Just show visualization
        fig = visualize_segmentation(
            input_path,
            segmentation,
            class_names=model.class_names,
            alpha=args.alpha
        )
        fig.show()


def run_data_generation(args):
    """Generate synthetic dataset."""
    logger.info(f"Generating {args.num_images} synthetic images...")
    
    dataset_path = create_synthetic_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        image_size=tuple(args.image_size),
        num_classes=args.num_classes
    )
    
    logger.info(f"Dataset created at: {dataset_path}")
    logger.info(f"Images: {dataset_path / 'images'}")
    logger.info(f"Masks: {dataset_path / 'masks'}")


def run_model_info(args):
    """Show model information."""
    logger.info(f"Loading model: {args.model}")
    
    model = SemanticSegmentationModel(model_name=args.model)
    info = model.get_model_info()
    
    print(f"\nModel Information:")
    print(f"  Architecture: {info['model_name']}")
    print(f"  Device: {info['device']}")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"  Number of Classes: {info['num_classes']}")
    
    print(f"\nSupported Classes:")
    for i, class_name in enumerate(model.class_names[:10]):  # Show first 10
        print(f"  {i}: {class_name}")
    if len(model.class_names) > 10:
        print(f"  ... and {len(model.class_names) - 10} more")


def run_demo(args):
    """Run demo with sample image."""
    logger.info("Running demo with sample image...")
    
    # Load model
    model = SemanticSegmentationModel(model_name=args.model)
    
    # Create sample image
    image = load_sample_image()
    logger.info("Created sample image")
    
    # Perform segmentation
    results = model.predict(image)
    segmentation = results['segmentation']
    
    # Show results
    class_counts = model.get_class_counts(segmentation)
    logger.info(f"Detected {len(class_counts)} classes")
    
    # Save or show visualization
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig = visualize_segmentation(
            image,
            segmentation,
            class_names=model.class_names
        )
        demo_path = output_dir / "demo_result.png"
        fig.savefig(demo_path, dpi=300, bbox_inches='tight')
        logger.info(f"Demo results saved to {demo_path}")
    else:
        fig = visualize_segmentation(
            image,
            segmentation,
            class_names=model.class_names
        )
        fig.show()


if __name__ == "__main__":
    main()
