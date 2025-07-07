"""
Utility functions for Tiny-ImageNet project
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def create_output_directories(base_dir: str = "outputs") -> dict:
    """
    Create output directories for the project
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary with created directory paths
    """
    directories = {
        'base': base_dir,
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'plots': os.path.join(base_dir, 'plots'),
        'logs': os.path.join(base_dir, 'logs'),
        'evaluation': os.path.join(base_dir, 'evaluation'),
        'configs': os.path.join(base_dir, 'configs')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print("Output directories created:")
    for name, path in directories.items():
        print(f"  {name}: {path}")
    
    return directories


def check_system_info() -> dict:
    """
    Check and display system information
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'cuda_available': torch.cuda.is_available()
    }
    
    if info['cuda_available']:
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    
    print("System Information:")
    print(f"  Platform: {info['platform']}")
    print(f"  Python: {info['python_version']}")
    print(f"  CPU cores: {info['cpu_count']}")
    print(f"  RAM: {info['memory_gb']} GB")
    print(f"  CUDA available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"  CUDA version: {info['cuda_version']}")
        print(f"  GPU: {info['gpu_name']}")
        print(f"  GPU memory: {info['gpu_memory_gb']} GB")
    
    return info


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"


def count_dataset_samples(dataset_path: str) -> dict:
    """
    Count samples in the dataset
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with sample counts
    """
    counts = {}
    
    # Count training samples
    train_path = os.path.join(dataset_path, 'train')
    if os.path.exists(train_path):
        train_count = 0
        for class_dir in os.listdir(train_path):
            class_images_dir = os.path.join(train_path, class_dir, 'images')
            if os.path.exists(class_images_dir):
                train_count += len([f for f in os.listdir(class_images_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts['train'] = train_count
    
    # Count validation samples
    val_path = os.path.join(dataset_path, 'val', 'images')
    if os.path.exists(val_path):
        counts['val'] = len([f for f in os.listdir(val_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Count test samples
    test_path = os.path.join(dataset_path, 'test', 'images')
    if os.path.exists(test_path):
        counts['test'] = len([f for f in os.listdir(test_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    return counts


def estimate_training_time(num_samples: int, batch_size: int, num_epochs: int, 
                          time_per_batch: float = 0.1) -> str:
    """
    Estimate training time
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        time_per_batch: Estimated time per batch in seconds
        
    Returns:
        Formatted estimated time string
    """
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * num_epochs
    estimated_seconds = total_batches * time_per_batch
    
    return format_time(estimated_seconds)


def get_model_summary(model: torch.nn.Module) -> dict:
    """
    Get model summary information
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6,
        'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
    }
    
    return summary


def save_experiment_log(config, history: dict, results: dict, log_path: str) -> None:
    """
    Save experiment log as JSON
    
    Args:
        config: Configuration object
        history: Training history
        results: Evaluation results
        log_path: Path to save log file
    """
    import json
    from datetime import datetime
    
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'data': config.data.__dict__,
            'device': config.device,
            'seed': config.seed
        },
        'history': history,
        'results': results
    }
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Experiment log saved to: {log_path}")


def load_experiment_log(log_path: str) -> dict:
    """
    Load experiment log from JSON
    
    Args:
        log_path: Path to log file
        
    Returns:
        Dictionary with experiment data
    """
    import json
    
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    return log_data


def cleanup_checkpoints(checkpoint_dir: str, keep_best: bool = True, keep_latest: bool = True) -> None:
    """
    Clean up old checkpoints to save disk space
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Whether to keep best model checkpoint
        keep_latest: Whether to keep latest checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if len(checkpoint_files) <= 2:  # Keep at least 2 files
        return
    
    # Sort by modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    
    files_to_keep = set()
    
    if keep_latest:
        files_to_keep.add(checkpoint_files[-1])  # Most recent
    
    if keep_best:
        # Find file with "best" in name
        for f in checkpoint_files:
            if "best" in f.lower():
                files_to_keep.add(f)
                break
    
    # Remove old files
    for f in checkpoint_files:
        if f not in files_to_keep:
            file_path = os.path.join(checkpoint_dir, f)
            os.remove(file_path)
            print(f"Removed old checkpoint: {f}")


def check_dependencies() -> dict:
    """
    Check if required dependencies are installed
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'torch': False,
        'torchvision': False,
        'numpy': False,
        'matplotlib': False,
        'sklearn': False,
        'tqdm': False,
        'seaborn': False,
        'PIL': False,
        'efficientnet_pytorch': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'PIL':
                import PIL
            elif dep == 'sklearn':
                import sklearn
            elif dep == 'efficientnet_pytorch':
                import efficientnet_pytorch
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    print("Dependency Status:")
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    return dependencies
