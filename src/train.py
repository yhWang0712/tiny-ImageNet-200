"""
Training utilities for Tiny-ImageNet classification
"""

import os
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import Config
except ImportError:
    from config import Config


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get device for training
    
    Args:
        device_config: Device configuration ("auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.)
        
    Returns:
        PyTorch device
    """
    print(f"🖥️  Device Configuration: {device_config}")
    
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("🔍 Auto-detecting device...")
    elif device_config == "cpu":
        device = torch.device("cpu")
        print("🖳  Forcing CPU usage")
    elif device_config.startswith("cuda"):
        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            # Check if specific GPU index is provided
            if ":" in device_config:
                gpu_id = int(device_config.split(":")[1])
                if gpu_id >= torch.cuda.device_count():
                    available_gpus = torch.cuda.device_count()
                    print(f"⚠️  Warning: GPU {gpu_id} not available (only {available_gpus} GPUs found)")
                    print(f"🔧 Falling back to GPU 0")
                    device = torch.device("cuda:0")
                else:
                    device = torch.device(device_config)
                    print(f"🎯 Targeting specific GPU: {gpu_id}")
            else:
                # Just "cuda" without specific ID
                device = torch.device("cuda")
                print("🔧 Using default CUDA device")
    else:
        print(f"⚠️  Warning: Unknown device config '{device_config}', using auto-detection")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Display final device information
    print(f"✅ Selected device: {device}")
    
    if device.type == "cuda":
        gpu_id = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(gpu_id)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        
        print(f"🚀 GPU Information:")
        print(f"   • GPU ID: {gpu_id}")
        print(f"   • GPU Name: {gpu_name}")
        print(f"   • Total Memory: {total_memory:.1f} GB")
        
        try:
            torch.cuda.set_device(gpu_id)
            reserved_memory = torch.cuda.memory_reserved(gpu_id) / 1024**3
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3
            free_memory = total_memory - reserved_memory
            print(f"   • Reserved Memory: {reserved_memory:.1f} GB")
            print(f"   • Allocated Memory: {allocated_memory:.1f} GB")
            print(f"   • Free Memory: {free_memory:.1f} GB")
        except Exception as e:
            print(f"   • Memory details unavailable: {str(e)[:50]}...")
            
        # Show all available GPUs
        if torch.cuda.device_count() > 1:
            print(f"📊 Available GPUs ({torch.cuda.device_count()} total):")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                status = "🎯 SELECTED" if i == gpu_id else "💤 Available"
                print(f"   • GPU {i}: {name} ({memory:.1f} GB) {status}")
    else:
        print("🖳  Using CPU for computation")
    
    print("=" * 60)
    return device


def get_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """
    Get optimizer based on configuration
    
    Args:
        model: PyTorch model
        config: Configuration object
        
    Returns:
        PyTorch optimizer
    """
    optimizer_name = config.training.optimizer.lower()
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(optimizer: optim.Optimizer, config: Config) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler based on configuration
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration object
        
    Returns:
        PyTorch learning rate scheduler or None
    """
    scheduler_name = config.training.scheduler.lower()
    
    if scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma
        )
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.t_max
        )
    elif scheduler_name == "cosine_warm":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.t_0,
            T_mult=config.training.t_mult
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int) -> Tuple[float, float]:
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (data, target) in enumerate(train_pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        current_acc = 100. * correct / total
        train_pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                  device: torch.device, epoch: int) -> Tuple[float, float]:
    """
    Validate model for one epoch
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        for data, target in val_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            current_acc = 100. * correct / total
            val_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    epoch_loss = val_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Config, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Train model according to configuration
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        device: Device to use
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Training history
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'learning_rates': [],
        'best_val_acc': 0.0,
        'total_time': 0.0
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Starting training for {config.training.num_epochs} epochs...")
    print_gpu_memory()
    
    start_time = time.time()
    
    for epoch in range(config.training.num_epochs):
        epoch_start = time.time()
        
        # Train epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate epoch
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        if scheduler is not None:
            scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Record history
        history['train_losses'].append(train_loss)
        history['train_accuracies'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch {epoch+1}/{config.training.num_epochs} ({epoch_time:.1f}s) - '
              f'LR: {current_lr:.6f} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Print GPU memory usage periodically
        if (epoch + 1) % 10 == 0:
            print_gpu_memory()
    
    total_time = time.time() - start_time
    history['total_time'] = total_time
    history['best_val_acc'] = best_val_acc
    
    print(f"\\nTraining completed! Total time: {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights")
    
    return model, history


def save_model(model: nn.Module, save_path: str, config: Config, history: Dict) -> None:
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        save_path: Path to save model
        config: Configuration object
        history: Training history
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'model_name': config.model.model_name
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


def load_model(model: nn.Module, checkpoint_path: str) -> Tuple[nn.Module, Dict]:
    """
    Load model from checkpoint
    
    Args:
        model: PyTorch model (with same architecture)
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Tuple of (loaded_model, history)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    history = checkpoint.get('history', {})
    
    print(f"Model loaded from: {checkpoint_path}")
    
    return model, history
