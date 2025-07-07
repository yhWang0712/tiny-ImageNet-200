#!/usr/bin/env python3
"""
Test script to verify project setup
"""

import os
import sys

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from config import Config, get_default_config
        print("✓ config module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
    
    try:
        from dataset import TinyImageNetDataset, load_class_mappings
        print("✓ dataset module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dataset: {e}")
    
    try:
        from models import create_model, get_available_models
        print("✓ models module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models: {e}")
    
    try:
        from train import get_device, train_model
        print("✓ train module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import train: {e}")
    
    try:
        from evaluate import evaluate_model
        print("✓ evaluate module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluate: {e}")
    
    try:
        from visualize import plot_training_history
        print("✓ visualize module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import visualize: {e}")
    
    try:
        from utils import set_seed, check_system_info
        print("✓ utils module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")


def test_config():
    """Test configuration system"""
    print("\\nTesting configuration system...")
    
    try:
        from config import get_default_config
        config = get_default_config()
        print(f"✓ Default config created")
        print(f"  Model: {config.model.model_name}")
        print(f"  Epochs: {config.training.num_epochs}")
        print(f"  Batch size: {config.data.batch_size}")
    except Exception as e:
        print(f"✗ Failed to create config: {e}")


def test_models():
    """Test model creation"""
    print("\\nTesting model creation...")
    
    try:
        from models import get_available_models, create_model
        from config import ModelConfig
        
        available_models = get_available_models()
        print(f"✓ Available models: {available_models}")
        
        # Test creating a simple model
        model_config = ModelConfig(model_name="resnet18", num_classes=200)
        model = create_model(model_config)
        print(f"✓ ResNet-18 model created successfully")
        
    except Exception as e:
        print(f"✗ Failed to create model: {e}")


def test_utilities():
    """Test utility functions"""
    print("\\nTesting utilities...")
    
    try:
        from utils import check_system_info, set_seed
        
        set_seed(42)
        print("✓ Random seed set successfully")
        
        system_info = check_system_info()
        print("✓ System info retrieved successfully")
        
    except Exception as e:
        print(f"✗ Failed to test utilities: {e}")


def main():
    """Main test function"""
    print("Tiny-ImageNet Project Setup Test")
    print("=" * 40)
    
    # Change to project directory
    os.chdir(project_root)
    
    test_imports()
    test_config()
    test_models()
    test_utilities()
    
    print("\\n" + "=" * 40)
    print("Setup test completed!")
    print("\\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Explore dataset: python main.py explore")
    print("3. Run quick training: python main.py train")


if __name__ == '__main__':
    main()
