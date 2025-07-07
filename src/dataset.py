"""
Dataset utilities for Tiny-ImageNet
"""

import os
import random
from typing import List, Tuple, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    from .config import Config
except ImportError:
    from config import Config


class TinyImageNetDataset(Dataset):
    """Tiny-ImageNet dataset class"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels (integers)
            transform: Pytorch transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_class_mappings(dataset_path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load class mappings from wnids.txt and words.txt
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (class_ids, class_names_dict)
    """
    wnids_path = os.path.join(dataset_path, 'wnids.txt')
    words_path = os.path.join(dataset_path, 'words.txt')
    
    # Read class IDs
    with open(wnids_path, 'r') as f:
        class_ids = [line.strip() for line in f.readlines()]
    
    # Read class names
    class_names = {}
    with open(words_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                class_names[parts[0]] = parts[1]
    
    return class_ids, class_names


def load_train_data(dataset_path: str, class_ids: List[str]) -> Tuple[List[str], List[int]]:
    """
    Load training data paths and labels
    
    Args:
        dataset_path: Path to dataset directory
        class_ids: List of class IDs
        
    Returns:
        Tuple of (image_paths, labels)
    """
    train_path = os.path.join(dataset_path, 'train')
    train_data = []
    train_labels = []
    
    print("Loading training data...")
    for i, class_id in enumerate(class_ids):
        class_dir = os.path.join(train_path, class_id, 'images')
        if os.path.exists(class_dir):
            image_files = os.listdir(class_dir)
            for img_file in image_files:
                if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    train_data.append(img_path)
                    train_labels.append(i)
        
        if i % 50 == 0:
            print(f"Processed {i+1}/{len(class_ids)} classes")
    
    return train_data, train_labels


def load_val_data(dataset_path: str, class_ids: List[str]) -> Tuple[List[str], List[int]]:
    """
    Load validation data paths and labels
    
    Args:
        dataset_path: Path to dataset directory
        class_ids: List of class IDs
        
    Returns:
        Tuple of (image_paths, labels)
    """
    val_path = os.path.join(dataset_path, 'val')
    val_annotations_path = os.path.join(val_path, 'val_annotations.txt')
    val_images_path = os.path.join(val_path, 'images')
    
    val_data = []
    val_labels = []
    
    print("Loading validation data...")
    with open(val_annotations_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                class_id = parts[1]
                
                img_path = os.path.join(val_images_path, img_name)
                if os.path.exists(img_path) and class_id in class_ids:
                    val_data.append(img_path)
                    val_labels.append(class_ids.index(class_id))
    
    return val_data, val_labels


def get_transforms(config: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.RandomHorizontalFlip(p=config.data.horizontal_flip_prob),
        transforms.RandomRotation(degrees=config.data.rotation_degrees),
        transforms.ColorJitter(**config.data.color_jitter_params),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.mean, std=config.data.std)
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.mean, std=config.data.std)
    ])
    
    return train_transform, val_transform


def create_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, str]]:
    """
    Create train and validation data loaders
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, class_ids, class_names)
    """
    # Load class mappings
    class_ids, class_names = load_class_mappings(config.data.dataset_path)
    
    # Load data
    train_data, train_labels = load_train_data(config.data.dataset_path, class_ids)
    val_data, val_labels = load_val_data(config.data.dataset_path, class_ids)
    
    # Get transforms
    train_transform, val_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = TinyImageNetDataset(train_data, train_labels, train_transform)
    val_dataset = TinyImageNetDataset(val_data, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available()  # Only use pin_memory if CUDA is available
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available()  # Only use pin_memory if CUDA is available
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Number of classes: {len(class_ids)}")
    
    return train_loader, val_loader, class_ids, class_names


def analyze_dataset(train_labels: List[int], val_labels: List[int]) -> None:
    """
    Analyze and print dataset statistics
    
    Args:
        train_labels: Training labels
        val_labels: Validation labels
    """
    print("=== Dataset Statistics ===")
    
    # Training set statistics
    train_counter = Counter(train_labels)
    print(f"Training set:")
    print(f"  Samples per class (unique counts): {len(set(train_counter.values()))}")
    print(f"  Average samples per class: {np.mean(list(train_counter.values())):.1f}")
    print(f"  Min samples per class: {min(train_counter.values())}")
    print(f"  Max samples per class: {max(train_counter.values())}")
    
    # Validation set statistics
    val_counter = Counter(val_labels)
    print(f"Validation set:")
    print(f"  Samples per class (unique counts): {len(set(val_counter.values()))}")
    print(f"  Average samples per class: {np.mean(list(val_counter.values())):.1f}")
    print(f"  Min samples per class: {min(val_counter.values())}")
    print(f"  Max samples per class: {max(val_counter.values())}")


def verify_dataset_paths(config: Config) -> bool:
    """
    Verify that dataset paths exist
    
    Args:
        config: Configuration object
        
    Returns:
        True if all paths exist, False otherwise
    """
    dataset_path = config.data.dataset_path
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')
    wnids_path = os.path.join(dataset_path, 'wnids.txt')
    words_path = os.path.join(dataset_path, 'words.txt')
    
    paths = {
        'Dataset root': dataset_path,
        'Training directory': train_path,
        'Validation directory': val_path,
        'Test directory': test_path,
        'Class IDs file': wnids_path,
        'Class names file': words_path
    }
    
    print("Verifying dataset paths:")
    all_exist = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    
    return all_exist
