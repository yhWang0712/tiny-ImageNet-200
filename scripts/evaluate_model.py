#!/usr/bin/env python3
"""
Evaluation script for trained models
"""

import argparse
import os
import sys

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

import torch

from config import Config
from dataset import create_data_loaders
from models import create_model
from train import load_model, get_device
import evaluate as eval_module  # Use alias to avoid conflicts
from utils import set_seed, create_output_directories


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained Tiny-ImageNet classifier')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset-path', type=str, 
                       default='./tiny-imagenet-200',
                       help='Path to Tiny-ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda:2',
                       help='Device to use for evaluation (auto, cpu, cuda, cuda:0, cuda:1, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("Loading model for evaluation...")
    print("=" * 40)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config')
    
    if config is None:
        print("Error: No configuration found in checkpoint")
        return
    
    # Update paths for evaluation
    config.data.dataset_path = args.dataset_path
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    
    # Get device
    device = get_device(args.device)
    
    # Create data loaders
    print("Creating data loaders...")
    _, val_loader, class_ids, class_names = create_data_loaders(config)
    
    # Create and load model
    print("Loading model...")
    model = create_model(config.model)
    model, history = load_model(model, args.checkpoint)
    model = model.to(device)
    
    print(f"Model: {config.model.model_name}")
    print(f"Classes: {len(class_ids)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Run comprehensive evaluation
    print("\\nRunning evaluation...")
    results = eval_module.comprehensive_evaluation(
        model, val_loader, device, class_names, class_ids, config, args.output_dir
    )
    
    print(f"\\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
