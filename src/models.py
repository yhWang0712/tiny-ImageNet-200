"""
Model definitions for Tiny-ImageNet classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig


class SimpleViT(nn.Module):
    """Simplified Vision Transformer implementation"""
    
    def __init__(self, img_size: int = 64, patch_size: int = 8, num_classes: int = 200,
                 dim: int = 256, depth: int = 6, heads: int = 8, mlp_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        # Image patches configuration
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = x.shape[0]
        
        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, 3 * self.patch_size ** 2)
        
        # Patch embedding
        patches = self.patch_embedding(patches)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Add positional encoding
        x += self.pos_embedding
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Classification using CLS token
        x = self.norm(x[:, 0])
        x = self.classifier(x)
        
        return x


def create_resnet18(num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    """
    Create ResNet-18 model adapted for Tiny-ImageNet,包含18层（包括8个残差块，每块2层），2222结构
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        ResNet-18 model
    """
    model = models.resnet18(pretrained=pretrained)
    
    # Adapt for smaller input size (64x64)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool to preserve feature map size
    
    # Modify final layer for number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def create_resnet34(num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    """
    Create ResNet-34 model adapted for Tiny-ImageNet,包含34层（包括16个残差块，每块2层），3463结构
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        ResNet-34 model
    """
    model = models.resnet34(pretrained=pretrained)
    
    # Adapt for smaller input size (64x64)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Modify final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def create_efficientnet_b0(num_classes: int = 200) -> Optional[nn.Module]:
    """
    Create EfficientNet-B0 model
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        EfficientNet-B0 model or None if not available
    """
    try:
        from efficientnet_pytorch import EfficientNet
        
        model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Modify classifier
        num_features = model._fc.in_features
        model._fc = nn.Linear(num_features, num_classes)
        
        return model
    
    except ImportError:
        print("EfficientNet not available. Install with: pip install efficientnet-pytorch")
        return None


def create_mobilenet_v2(num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    """
    Create MobileNet-V2 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        MobileNet-V2 model
    """
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # Modify classifier
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def create_model(config: ModelConfig) -> nn.Module:
    """
    Create model based on configuration
    
    Args:
        config: Model configuration
        
    Returns:
        PyTorch model
    """
    model_name = config.model_name.lower()
    
    if model_name == "resnet18":
        model = create_resnet18(config.num_classes, config.pretrained)
        
    elif model_name == "resnet34":
        model = create_resnet34(config.num_classes, config.pretrained)
        
    elif model_name == "efficientnet_b0":
        model = create_efficientnet_b0(config.num_classes)
        if model is None:
            raise ValueError("EfficientNet not available")
            
    elif model_name == "mobilenet_v2":
        model = create_mobilenet_v2(config.num_classes, config.pretrained)
        
    elif model_name == "vit":
        model = SimpleViT(
            img_size=64,  # Tiny-ImageNet image size
            patch_size=config.vit_patch_size,
            num_classes=config.num_classes,
            dim=config.vit_dim,
            depth=config.vit_depth,
            heads=config.vit_heads,
            mlp_dim=config.vit_mlp_dim,
            dropout=config.dropout_rate
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params) in millions
    """
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    return total_params, trainable_params


def print_model_info(model: nn.Module, model_name: str = "") -> None:
    """
    Print model information
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    total_params, trainable_params = count_parameters(model)
    
    print(f"=== {model_name} Model Information ===")
    print(f"Total parameters: {total_params:.2f}M")
    print(f"Trainable parameters: {trainable_params:.2f}M")
    print(f"Model structure:")
    print(model)


def get_available_models() -> list:
    """Get list of available model names"""
    return ["resnet18", "resnet34", "efficientnet_b0", "mobilenet_v2", "vit"]
