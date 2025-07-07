"""
Evaluation utilities for Tiny-ImageNet classification
"""

import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .config import Config
except ImportError:
    from config import Config


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device,
                  class_names: Dict[str, str], class_ids: List[str]) -> Tuple[float, List[int], List[int]]:
    """
    Evaluate model on given dataset
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        
    Returns:
        Tuple of (accuracy, predictions, labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Overall accuracy: {accuracy * 100:.2f}%")
    
    return accuracy, all_predictions, all_labels


def print_classification_report(labels: List[int], predictions: List[int],
                               class_names: Dict[str, str], class_ids: List[str],
                               num_classes_to_show: int = 20) -> None:
    """
    Print detailed classification report
    
    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        num_classes_to_show: Number of classes to show in detail
    """
    print(f"\\nClassification Report (first {num_classes_to_show} classes):")
    
    # Filter to first N classes for readability
    filtered_labels = [l for l in labels if l < num_classes_to_show]
    filtered_predictions = [p for p, l in zip(predictions, labels) if l < num_classes_to_show]
    
    if len(filtered_labels) > 0:
        target_names = [
            class_names.get(class_ids[i], f'Class_{i}')[:20] 
            for i in range(min(num_classes_to_show, len(class_ids)))
        ]
        
        print(classification_report(
            filtered_labels,
            filtered_predictions,
            target_names=target_names,
            digits=3
        ))


def plot_confusion_matrix(labels: List[int], predictions: List[int],
                         class_names: Dict[str, str], class_ids: List[str],
                         num_classes: int = 10, save_path: str = None) -> None:
    """
    Plot confusion matrix for subset of classes
    
    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        num_classes: Number of classes to include in matrix
        save_path: Path to save plot
    """
    # Filter to first N classes
    filtered_labels = [l for l in labels if l < num_classes]
    filtered_predictions = [p for p, l in zip(predictions, labels) if l < num_classes]
    
    if len(filtered_labels) == 0:
        print("No samples found for confusion matrix")
        return
    
    # Generate confusion matrix
    cm = confusion_matrix(filtered_labels, filtered_predictions)
    
    # Create class labels
    class_labels = [
        class_names.get(class_ids[i], f'Class_{i}')[:10]
        for i in range(min(num_classes, len(class_ids)))
    ]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'Confusion Matrix (First {num_classes} Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def visualize_predictions(model: nn.Module, data_loader: DataLoader, device: torch.device,
                         class_names: Dict[str, str], class_ids: List[str],
                         config: Config, num_samples: int = 8, save_path: str = None) -> None:
    """
    Visualize model predictions on sample images
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        config: Configuration object
        num_samples: Number of samples to visualize
        save_path: Path to save plot
    """
    model.eval()
    
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
        confidence = torch.max(probabilities, 1)[0]
    
    # Denormalize images for visualization
    mean = torch.tensor(config.data.mean).view(3, 1, 1).to(device)
    std = torch.tensor(config.data.std).view(3, 1, 1).to(device)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.cpu().permute(1, 2, 0).numpy()
        
        axes[i].imshow(img)
        
        # Get class names
        true_class = class_names.get(class_ids[labels[i].item()], 'Unknown')[:15]
        pred_class = class_names.get(class_ids[predicted[i].item()], 'Unknown')[:15]
        conf = confidence[i].item()
        
        # Set title color based on correctness
        color = 'green' if labels[i] == predicted[i] else 'red'
        axes[i].set_title(f'True: {true_class}\\nPred: {pred_class}\\nConf: {conf:.3f}',
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions (Green=Correct, Red=Incorrect)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to: {save_path}")
    
    plt.show()


def analyze_class_performance(labels: List[int], predictions: List[int],
                             class_names: Dict[str, str], class_ids: List[str],
                             top_k: int = 10) -> None:
    """
    Analyze per-class performance
    
    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        top_k: Number of best/worst classes to show
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    num_classes = len(class_ids)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), zero_division=0
    )
    
    # Create results dataframe-like structure
    results = []
    for i in range(num_classes):
        class_name = class_names.get(class_ids[i], f'Class_{i}')
        results.append({
            'class_id': i,
            'class_name': class_name[:20],
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        })
    
    # Sort by F1 score
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    print(f"\\nTop {top_k} performing classes (by F1-score):")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 70)
    for i in range(min(top_k, len(results))):
        r = results[i]
        print(f"{r['class_name']:<20} {r['precision']:<10.3f} {r['recall']:<10.3f} "
              f"{r['f1']:<10.3f} {r['support']:<10}")
    
    print(f"\\nBottom {top_k} performing classes (by F1-score):")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 70)
    for i in range(max(0, len(results) - top_k), len(results)):
        r = results[i]
        print(f"{r['class_name']:<20} {r['precision']:<10.3f} {r['recall']:<10.3f} "
              f"{r['f1']:<10.3f} {r['support']:<10}")


def plot_top_errors(model: nn.Module, data_loader: DataLoader, device: torch.device,
                   class_names: Dict[str, str], class_ids: List[str],
                   config: Config, num_errors: int = 16, save_path: str = None) -> None:
    """
    Plot images with highest prediction errors (lowest confidence on wrong predictions)
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        config: Configuration object
        num_errors: Number of error examples to show
        save_path: Path to save plot
    """
    model.eval()
    
    errors = []
    
    print("Finding prediction errors...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Finding errors'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)
            confidence = torch.max(probabilities, 1)[0]
            
            # Find incorrect predictions
            incorrect_mask = predicted != labels
            
            if incorrect_mask.any():
                incorrect_images = images[incorrect_mask]
                incorrect_labels = labels[incorrect_mask]
                incorrect_predictions = predicted[incorrect_mask]
                incorrect_confidence = confidence[incorrect_mask]
                
                for img, true_label, pred_label, conf in zip(
                    incorrect_images, incorrect_labels, incorrect_predictions, incorrect_confidence
                ):
                    errors.append({
                        'image': img.cpu(),
                        'true_label': true_label.cpu().item(),
                        'pred_label': pred_label.cpu().item(),
                        'confidence': conf.cpu().item()
                    })
    
    if len(errors) == 0:
        print("No errors found!")
        return
    
    # Sort by confidence (highest confidence errors are most interesting)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Take top errors
    top_errors = errors[:num_errors]
    
    # Plot
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.ravel()
    
    mean = torch.tensor(config.data.mean).view(3, 1, 1)
    std = torch.tensor(config.data.std).view(3, 1, 1)
    
    for i, error in enumerate(top_errors):
        if i >= rows * cols:
            break
        
        # Denormalize image
        img = error['image'] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img)
        
        # Get class names
        true_class = class_names.get(class_ids[error['true_label']], 'Unknown')[:15]
        pred_class = class_names.get(class_ids[error['pred_label']], 'Unknown')[:15]
        conf = error['confidence']
        
        axes[i].set_title(f'True: {true_class}\\nPred: {pred_class}\\nConf: {conf:.3f}',
                         color='red', fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(top_errors), rows * cols):
        axes[i].axis('off')
    
    plt.suptitle('Highest Confidence Errors', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top errors visualization saved to: {save_path}")
    
    plt.show()


def comprehensive_evaluation(model: nn.Module, val_loader: DataLoader, device: torch.device,
                           class_names: Dict[str, str], class_ids: List[str],
                           config: Config, output_dir: str = "outputs/evaluation") -> Dict:
    """
    Perform comprehensive model evaluation
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to use
        class_names: Dictionary mapping class IDs to names
        class_ids: List of class IDs
        config: Configuration object
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic evaluation
    accuracy, predictions, labels = evaluate_model(model, val_loader, device, class_names, class_ids)
    
    # Classification report
    print_classification_report(labels, predictions, class_names, class_ids)
    
    # Confusion matrix
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, predictions, class_names, class_ids, 
                         num_classes=15, save_path=confusion_matrix_path)
    
    # Prediction visualization
    predictions_path = os.path.join(output_dir, "predictions.png")
    visualize_predictions(model, val_loader, device, class_names, class_ids, 
                         config, save_path=predictions_path)
    
    # Class performance analysis
    analyze_class_performance(labels, predictions, class_names, class_ids)
    
    # Top errors
    errors_path = os.path.join(output_dir, "top_errors.png")
    plot_top_errors(model, val_loader, device, class_names, class_ids, 
                   config, save_path=errors_path)
    
    results = {
        'accuracy': accuracy,
        'num_samples': len(labels),
        'num_classes': len(class_ids),
        'predictions': predictions,
        'labels': labels
    }
    
    print(f"\\nEvaluation completed! Results saved to: {output_dir}")
    
    return results
