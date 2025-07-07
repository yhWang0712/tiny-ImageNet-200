#!/usr/bin/env python3
"""
Training script for Tiny-ImageNet classification
"""

import argparse
import os
import sys

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

import torch

from config import Config, get_default_config
from dataset import create_data_loaders, verify_dataset_paths
from models import create_model, print_model_info
from train import get_device, train_model, save_model, load_model
import evaluate as eval_module  # Use alias to avoid conflicts
from visualize import plot_training_history, set_style, create_summary_report
from utils import (set_seed, create_output_directories, check_system_info,
                  save_experiment_log, get_model_summary, format_time)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Tiny-ImageNet classifier')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'efficientnet_b0', 'efficientnet-b0', 'mobilenet_v2', 'vit'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Don\'t use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'cosine_warm', 'none'],
                       help='Learning rate scheduler')
    
    # Data arguments
    parser.add_argument('--dataset-path', type=str, 
                       default='/home/wangyiheng/sais/tiny-imagenet-200',
                       help='Path to Tiny-ImageNet dataset')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (auto, cpu, cuda, cuda:0, cuda:1, cuda:2, cuda:3, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save training plots')
    parser.add_argument('--no-save-plots', dest='save_plots', action='store_false',
                       help='Don\'t save training plots')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after training')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (for evaluation only)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint for evaluation')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # 标准化模型名称（将连字符转换为下划线）
    args.model = args.model.replace('-', '_')
    
    # Set style for plots
    set_style()
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}"
    
    print(f"Starting experiment: {args.experiment_name}")
    print("=" * 60)
    
    # Display system and device information
    print("🔧 System Information:")
    print(f"   • Python: {sys.version.split()[0]}")
    print(f"   • PyTorch: {torch.__version__}")
    print(f"   • CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   • CUDA Version: {torch.version.cuda}")
        print(f"   • Available GPUs: {torch.cuda.device_count()}")
    print()
    
    # Check system info
    system_info = check_system_info()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    output_dirs = create_output_directories(args.output_dir)
    
    # Create experiment-specific directories
    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    exp_dirs = create_output_directories(exp_dir)
    
    # Create configuration
    config = get_default_config()
    
    # Normalize model name (convert hyphens to underscores)
    model_name = args.model.replace('-', '_')
    
    # Update config with command line arguments
    config.model.model_name = model_name
    config.model.pretrained = args.pretrained
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.optimizer = args.optimizer
    config.training.scheduler = args.scheduler
    config.data.batch_size = args.batch_size
    config.data.dataset_path = args.dataset_path
    config.data.num_workers = args.num_workers
    config.device = args.device
    config.seed = args.seed
    
    # Update output paths
    config.training.checkpoint_dir = exp_dirs['checkpoints']
    config.training.plots_dir = exp_dirs['plots']
    
    # Save configuration
    config_path = os.path.join(exp_dirs['configs'], 'config.yaml')
    config.save_yaml(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Verify dataset
    print("\\nVerifying dataset...")
    if not verify_dataset_paths(config):
        print("Dataset verification failed! Please check the dataset path.")
        return
    
    # Get device
    device = get_device(config.device)
    
    # Create data loaders
    print("\\nCreating data loaders...")
    train_loader, val_loader, class_ids, class_names = create_data_loaders(config)
    
    # Update number of classes in config
    config.model.num_classes = len(class_ids)
    
    # Create model
    print("\\nCreating model...")
    model = create_model(config.model)
    model = model.to(device)
    
    # Print model information
    print_model_info(model, config.model.model_name)
    model_summary = get_model_summary(model)
    
    # Training
    if not args.skip_training:
        print("\\nStarting training...")
        model, history = train_model(model, train_loader, val_loader, config, device)
        
        # Save model
        model_path = os.path.join(exp_dirs['checkpoints'], f'{args.experiment_name}_best.pth')
        save_model(model, model_path, config, history)
        
        # Plot training history
        if args.save_plots:
            plot_path = os.path.join(exp_dirs['plots'], 'training_history.png')
            plot_training_history(history, config.model.model_name, plot_path)
        else:
            plot_training_history(history, config.model.model_name)
    
    else:
        # Load model for evaluation
        if args.checkpoint is None:
            print("Error: --checkpoint required when --skip-training is used")
            return
        
        model, history = load_model(model, args.checkpoint)
        model = model.to(device)
    
    # Evaluation
    if args.evaluate or args.skip_training:
        print("\\nRunning comprehensive evaluation...")
        eval_dir = os.path.join(exp_dirs['evaluation'])
        results = eval_module.comprehensive_evaluation(
            model, val_loader, device, class_names, class_ids, config, eval_dir
        )
        
        # Create summary report
        summary_path = os.path.join(exp_dir, 'experiment_summary.txt')
        create_summary_report(
            config, history if not args.skip_training else {}, 
            results['accuracy'], model_summary['total_params_millions'], summary_path
        )
        
        # Save experiment log
        log_path = os.path.join(exp_dirs['logs'], 'experiment_log.json')
        save_experiment_log(config, history if not args.skip_training else {}, results, log_path)
    
    print(f"\\nExperiment completed! Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()
