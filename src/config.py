"""
Configuration settings for Tiny-ImageNet classification
"""

import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_path: str = "/home/wangyiheng/sais/tiny-imagenet-200"  # Relative path to dataset
    batch_size: int = 256
    num_workers: int = 4
    image_size: Tuple[int, int] = (64, 64)
    
    # Normalization values (ImageNet pretrained)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data augmentation parameters
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 15
    color_jitter_params: dict = None
    
    def __post_init__(self):
        if self.color_jitter_params is None:
            self.color_jitter_params = {
                'brightness': 0.2,
                'contrast': 0.2, 
                'saturation': 0.2,
                'hue': 0  # 禁止使用色调调整
            }


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "resnet18"  # Options: resnet18, efficientnet_b0, vit
    num_classes: int = 200
    pretrained: bool = True
    dropout_rate: float = 0.1
    
    # ViT specific parameters
    vit_patch_size: int = 8
    vit_dim: int = 256
    vit_depth: int = 6
    vit_heads: int = 8
    vit_mlp_dim: int = 512


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer: str = "adamw"  # Options: adam, adamw, sgd
    momentum: float = 0.9  # For SGD
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # Options: step, cosine, cosine_warm
    step_size: int = 10  # For StepLR
    gamma: float = 0.1  # For StepLR
    t_max: int = 50  # For CosineAnnealingLR
    t_0: int = 10  # For CosineAnnealingWarmRestarts
    t_mult: int = 2  # For CosineAnnealingWarmRestarts
    
    # Model saving
    save_best_only: bool = True
    checkpoint_dir: str = "outputs/checkpoints"
    save_plots: bool = True
    plots_dir: str = "outputs/plots"


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda
    seed: int = 42
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config objects from dictionaries
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            **{k: v for k, v in config_dict.items() if k not in ['data', 'model', 'training']}
        )
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        import yaml
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'device': self.device,
            'seed': self.seed
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
