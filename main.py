#!/usr/bin/env python3
"""
Main entry point for Tiny-ImageNet classification project
"""

import argparse
import subprocess
import sys
import os


def run_exploration():
    """Run dataset exploration"""
    print("Running dataset exploration...")
    subprocess.run([sys.executable, "scripts/explore_data.py"])


def run_training(args):
    """Run model training"""
    cmd = [sys.executable, "scripts/train_model.py"]
    
    if args.model:
        cmd.extend(["--model", args.model])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.experiment_name:
        cmd.extend(["--experiment-name", args.experiment_name])
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Display training configuration
    print("🚀 Starting Training")
    print("=" * 50)
    print(f"📊 Configuration:")
    print(f"   • Model: {args.model}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Learning Rate: {args.lr}")
    print(f"   • Batch Size: {args.batch_size}")
    print(f"   • Device: {args.device}")
    if args.experiment_name:
        print(f"   • Experiment: {args.experiment_name}")
    print()
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("=" * 50)
    subprocess.run(cmd)


def run_evaluation(args):
    """Run model evaluation"""
    if not args.checkpoint:
        print("Error: Checkpoint path required for evaluation")
        return
    
    cmd = [sys.executable, "scripts/evaluate_model.py", "--checkpoint", args.checkpoint]
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Display evaluation configuration
    print("🔍 Starting Evaluation")
    print("=" * 50)
    print(f"📊 Configuration:")
    print(f"   • Checkpoint: {args.checkpoint}")
    print(f"   • Device: {args.device}")
    print()
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("=" * 50)
    subprocess.run(cmd)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Tiny-ImageNet Classification Project')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Explore command
    explore_parser = subparsers.add_parser('explore', help='Explore the dataset')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, default='resnet34',
                             choices=['resnet18', 'resnet34', 'vit', 'efficientnet-b0', 'mobilenet_v2'],
                             help='Model to train')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=512,
                             help='Batch size')
    train_parser.add_argument('--experiment-name', type=str,
                             help='Experiment name')
    train_parser.add_argument('--device', type=str, default='cuda:2',
                             help='Device to use (auto, cpu, cuda, cuda:0, cuda:1, etc.)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--device', type=str, default='cuda:2',
                            help='Device to use (auto, cpu, cuda, cuda:0, cuda:1, etc.)')
    
    # Quick commands
    quick_parser = subparsers.add_parser('quick', help='Quick training (20 epochs)')
    quick_parser.add_argument('--model', type=str, default='resnet18',
                             help='Model to train quickly')
    quick_parser.add_argument('--device', type=str, default='cuda:2',
                             help='Device to use (auto, cpu, cuda, cuda:0, cuda:1, etc.)')
    
    args = parser.parse_args()
    
    if args.command == 'explore':
        run_exploration()
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'quick':
        # Quick training with fewer epochs
        quick_args = type('args', (), {
            'model': args.model,
            'epochs': 20,
            'lr': 0.001,
            'batch_size': 64,
            'experiment_name': f'quick_{args.model}',
            'device': args.device
        })()
        run_training(quick_args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
