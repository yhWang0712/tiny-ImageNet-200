#!/usr/bin/env python3
"""
测试脚本：验证数据加载是否正常工作
"""

import os
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from config import Config, get_default_config
from dataset import create_data_loaders

def test_data_loading():
    """测试数据加载是否正常"""
    print("测试数据加载...")
    
    try:
        # 获取默认配置
        config = get_default_config()
        
        # 减少批次大小以便快速测试
        config.data.batch_size = 8
        config.data.num_workers = 2
        
        print(f"数据集路径: {config.data.dataset_path}")
        print(f"批次大小: {config.data.batch_size}")
        print(f"色相参数: {config.data.color_jitter_params['hue']}")
        
        # 创建数据加载器
        train_loader, val_loader, class_ids, class_names = create_data_loaders(config)
        
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        print(f"类别数: {len(class_ids)}")
        
        # 测试加载一个批次
        print("\\n测试加载训练数据批次...")
        train_iter = iter(train_loader)
        batch_data, batch_labels = next(train_iter)
        
        print(f"批次数据形状: {batch_data.shape}")
        print(f"批次标签形状: {batch_labels.shape}")
        print(f"数据类型: {batch_data.dtype}")
        print(f"数据范围: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
        
        print("\\n✅ 数据加载测试成功！")
        
    except Exception as e:
        print(f"\\n❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = test_data_loading()
    if success:
        print("\\n🎉 所有测试通过！现在可以安全地开始训练了。")
    else:
        print("\\n⚠️  请先解决数据加载问题再开始训练。")
