#!/usr/bin/env python3
"""
Data exploration script for Tiny-ImageNet dataset
"""

import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import get_default_config
from dataset import (load_class_mappings, load_train_data, load_val_data, 
                    analyze_dataset, verify_dataset_paths)
from visualize import plot_class_distribution, plot_sample_images, set_style
from utils import count_dataset_samples


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Explore Tiny-ImageNet dataset')
    
    parser.add_argument('--dataset-path', type=str, 
                       default='/home/wangyiheng/sais/tiny-imagenet-200',
                       help='Path to Tiny-ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/exploration',
                       help='Output directory for plots')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots to files')
    
    return parser.parse_args()


def main():
    """Main exploration function"""
    args = parse_args()
    
    # Set plot style
    set_style()
    
    print("Exploring Tiny-ImageNet Dataset")
    print("=" * 35)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create config
    config = get_default_config()
    config.data.dataset_path = args.dataset_path
    
    # Verify dataset
    print("\\n1. Verifying dataset paths...")
    if not verify_dataset_paths(config):
        print("Dataset verification failed!")
        return
    
    # Count samples
    print("\\n2. Counting dataset samples...")
    sample_counts = count_dataset_samples(args.dataset_path)
    for split, count in sample_counts.items():
        print(f"  {split.capitalize()}: {count:,} samples")
    
    # Load class mappings
    print("\\n3. Loading class information...")
    class_ids, class_names = load_class_mappings(args.dataset_path)
    print(f"  Number of classes: {len(class_ids)}")
    print(f"  First 5 classes:")
    for i in range(min(5, len(class_ids))):
        class_name = class_names.get(class_ids[i], 'Unknown')
        print(f"    {class_ids[i]}: {class_name}")
    
    # Load training and validation data
    print("\\n4. Loading dataset...")
    train_data, train_labels = load_train_data(args.dataset_path, class_ids)
    val_data, val_labels = load_val_data(args.dataset_path, class_ids)
    
    # Analyze dataset
    print("\\n5. Dataset analysis...")
    analyze_dataset(train_labels, val_labels)
    
    # Plot class distribution
    print("\\n6. Creating visualizations...")
    if args.save_plots:
        dist_path = os.path.join(args.output_dir, 'class_distribution.png')
        plot_class_distribution(train_labels, val_labels, class_names, class_ids, dist_path)
    else:
        plot_class_distribution(train_labels, val_labels, class_names, class_ids)
    
    # Plot sample images from training set
    if args.save_plots:
        train_samples_path = os.path.join(args.output_dir, 'train_samples.png')
        plot_sample_images(train_data, train_labels, class_names, class_ids,
                          title="Training Set Samples", save_path=train_samples_path)
    else:
        plot_sample_images(train_data, train_labels, class_names, class_ids,
                          title="Training Set Samples")
    
    # Plot sample images from validation set
    if args.save_plots:
        val_samples_path = os.path.join(args.output_dir, 'val_samples.png')
        plot_sample_images(val_data, val_labels, class_names, class_ids,
                          title="Validation Set Samples", save_path=val_samples_path)
    else:
        plot_sample_images(val_data, val_labels, class_names, class_ids,
                          title="Validation Set Samples")
    
    print(f"\\nDataset exploration completed!")
    if args.save_plots:
        print(f"Plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
