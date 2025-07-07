"""
Visualization utilities for training and results
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from .config import Config
except ImportError:
    from config import Config


def plot_training_history(history: Dict, model_name: str = "", save_path: Optional[str] = None) -> None:
    """
    Plot training history including loss, accuracy, and learning rate
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title(f'{model_name} - Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rates' in history and len(history['learning_rates']) > 0:
        axes[2].plot(epochs, history['learning_rates'], 'g-', label='Learning Rate', linewidth=2)
        axes[2].set_title(f'{model_name} - Learning Rate Schedule')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No Learning Rate\\nData Available', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title(f'{model_name} - Learning Rate')
    
    # Add best accuracy annotation
    if 'best_val_acc' in history:
        best_acc = history['best_val_acc']
        axes[1].axhline(y=best_acc, color='red', linestyle='--', alpha=0.7)
        axes[1].text(0.02, 0.98, f'Best Val Acc: {best_acc:.2f}%', 
                    transform=axes[1].transAxes, va='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def compare_models(histories: Dict[str, Dict], save_path: Optional[str] = None) -> None:
    """
    Compare training histories of multiple models
    
    Args:
        histories: Dictionary of {model_name: history}
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for (model_name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Training loss
        axes[0, 0].plot(epochs, history['train_losses'], 
                       color=color, label=f'{model_name}', linewidth=2)
        
        # Validation loss
        axes[0, 1].plot(epochs, history['val_losses'], 
                       color=color, label=f'{model_name}', linewidth=2)
        
        # Training accuracy
        axes[1, 0].plot(epochs, history['train_accuracies'], 
                       color=color, label=f'{model_name}', linewidth=2)
        
        # Validation accuracy
        axes[1, 1].plot(epochs, history['val_accuracies'], 
                       color=color, label=f'{model_name}', linewidth=2)
    
    # Set titles and labels
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Accuracy Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Accuracy Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    plt.show()


def plot_class_distribution(train_labels: List[int], val_labels: List[int], 
                           class_names: Dict[str, str], class_ids: List[str],
                           save_path: Optional[str] = None) -> None:
    """
    Plot class distribution in train and validation sets
    
    Args:
        train_labels: Training labels
        val_labels: Validation labels
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        save_path: Path to save the plot
    """
    from collections import Counter
    
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    
    # Get counts for all classes
    train_counts = [train_counter.get(i, 0) for i in range(len(class_ids))]
    val_counts = [val_counter.get(i, 0) for i in range(len(class_ids))]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Training set distribution
    axes[0].bar(range(len(class_ids)), train_counts, alpha=0.7, color='blue')
    axes[0].set_title('Training Set - Class Distribution')
    axes[0].set_xlabel('Class Index')
    axes[0].set_ylabel('Number of Samples')
    axes[0].grid(True, alpha=0.3)
    
    # Validation set distribution
    axes[1].bar(range(len(class_ids)), val_counts, alpha=0.7, color='red')
    axes[1].set_title('Validation Set - Class Distribution')
    axes[1].set_xlabel('Class Index')
    axes[1].set_ylabel('Number of Samples')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    
    plt.show()


def plot_sample_images(image_paths: List[str], labels: List[int], 
                      class_names: Dict[str, str], class_ids: List[str],
                      num_samples: int = 12, title: str = "Sample Images",
                      save_path: Optional[str] = None) -> None:
    """
    Plot sample images from the dataset
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        num_samples: Number of samples to show
        title: Plot title
        save_path: Path to save the plot
    """
    import random
    from PIL import Image
    
    # Randomly select samples
    indices = random.sample(range(len(image_paths)), min(num_samples, len(image_paths)))
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle(title, fontsize=16)
    
    for i, idx in enumerate(indices):
        row = i // 4
        col = i % 4
        
        try:
            # Load and display image
            img = Image.open(image_paths[idx]).convert('RGB')
            axes[row, col].imshow(img)
            
            # Get class name
            class_id = class_ids[labels[idx]]
            class_name = class_names.get(class_id, 'Unknown')[:20]
            
            axes[row, col].set_title(class_name, fontsize=10)
            axes[row, col].axis('off')
            
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Error loading\\n{str(e)[:30]}', 
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(indices), 12):
        row = i // 4
        col = i % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images plot saved to: {save_path}")
    
    plt.show()


def create_summary_report(config: Config, history: Dict, accuracy: float, 
                         model_params: float, output_path: str) -> None:
    """
    Create a text summary report of the experiment
    
    Args:
        config: Configuration used
        history: Training history
        accuracy: Final validation accuracy
        model_params: Number of model parameters (in millions)
        output_path: Path to save the report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Tiny-ImageNet Classification Experiment Report\\n")
        f.write("=" * 50 + "\\n\\n")
        
        # Model information
        f.write("MODEL CONFIGURATION\\n")
        f.write("-" * 20 + "\\n")
        f.write(f"Model: {config.model.model_name}\\n")
        f.write(f"Number of classes: {config.model.num_classes}\\n")
        f.write(f"Pretrained: {config.model.pretrained}\\n")
        f.write(f"Parameters: {model_params:.2f}M\\n\\n")
        
        # Training configuration
        f.write("TRAINING CONFIGURATION\\n")
        f.write("-" * 22 + "\\n")
        f.write(f"Epochs: {config.training.num_epochs}\\n")
        f.write(f"Batch size: {config.data.batch_size}\\n")
        f.write(f"Learning rate: {config.training.learning_rate}\\n")
        f.write(f"Optimizer: {config.training.optimizer}\\n")
        f.write(f"Scheduler: {config.training.scheduler}\\n")
        f.write(f"Weight decay: {config.training.weight_decay}\\n\\n")
        
        # Results
        f.write("RESULTS\\n")
        f.write("-" * 8 + "\\n")
        f.write(f"Final validation accuracy: {accuracy * 100:.2f}%\\n")
        f.write(f"Best validation accuracy: {history.get('best_val_acc', 0):.2f}%\\n")
        f.write(f"Total training time: {history.get('total_time', 0):.1f} seconds\\n\\n")
        
        # Training summary
        if 'train_losses' in history and len(history['train_losses']) > 0:
            f.write("TRAINING SUMMARY\\n")
            f.write("-" * 16 + "\\n")
            f.write(f"Initial train loss: {history['train_losses'][0]:.4f}\\n")
            f.write(f"Final train loss: {history['train_losses'][-1]:.4f}\\n")
            f.write(f"Initial val loss: {history['val_losses'][0]:.4f}\\n")
            f.write(f"Final val loss: {history['val_losses'][-1]:.4f}\\n")
            f.write(f"Initial train acc: {history['train_accuracies'][0]:.2f}%\\n")
            f.write(f"Final train acc: {history['train_accuracies'][-1]:.2f}%\\n")
            f.write(f"Initial val acc: {history['val_accuracies'][0]:.2f}%\\n")
            f.write(f"Final val acc: {history['val_accuracies'][-1]:.2f}%\\n")
    
    print(f"Summary report saved to: {output_path}")


def set_style() -> None:
    """Set matplotlib and seaborn style for consistent plots"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.unicode_minus'] = False
