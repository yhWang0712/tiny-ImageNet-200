# Tiny-ImageNet 分类项目

一个模块化的 Python 项目，用于在 Tiny-ImageNet 数据集上训练和评估深度学习模型。

## 项目结构

```
tiny-imagenet-project/
├── src/                    # 源代码模块
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── dataset.py         # 数据集加载和预处理
│   ├── models.py          # 模型定义
│   ├── train.py           # 训练工具
│   ├── evaluate.py        # 评估工具
│   ├── visualize.py       # 可视化工具
│   └── utils.py           # 通用工具
├── scripts/               # 可执行脚本
│   ├── train_model.py     # 主训练脚本
│   ├── evaluate_model.py  # 模型评估脚本
│   └── explore_data.py    # 数据集探索脚本
├── configs/               # 配置文件
│   ├── default.yaml       # 默认配置
│   ├── vit.yaml          # Vision Transformer 配置
│   └── fast.yaml         # 快速实验配置
├── outputs/               # 输出目录（运行时创建）
│   ├── checkpoints/       # 模型检查点
│   ├── plots/            # 训练图表
│   ├── logs/             # 实验日志
│   └── evaluation/       # 评估结果
├── requirements.txt       # Python 依赖
└── README.md             # 本文件
```

## 功能特性

### 支持的模型
- **ResNet-18/34**: 适配 64x64 输入尺寸
- **Vision Transformer (ViT)**: 自定义实现
- **MobileNet-V2**: 轻量级移动端友好模型
- **EfficientNet-B0**: 高效卷积网络（可选）

### 主要功能
- **模块化设计**: 清晰的关注点分离
- **灵活配置**: 基于 YAML 的配置系统
- **数据增强**: 全面的数据预处理
- **学习率调度**: 多种调度器选项
- **全面评估**: 详细的性能分析
- **可视化**: 训练曲线、混淆矩阵、错误分析
- **实验跟踪**: 自动日志记录和检查点保存

## 安装说明

1. **克隆或创建项目目录**:
   ```bash
   # 克隆仓库
   git clone <your-repo-url>
   cd tiny-imagenet-project
   ```

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

3. **下载数据集**（如果还没有）:
   ```bash
   # 下载 Tiny-ImageNet 数据集到 ./tiny-imagenet-200/
   # 如果数据集在不同位置，请更新 configs/*.yaml
   ```

## 数据集设置

Tiny-ImageNet 数据集应下载并放置在项目目录中：

```bash
# 选项 1: 直接下载
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

# 选项 2: 如果已有数据集
# 确保数据集在 ./tiny-imagenet-200/ 目录中
```

期望的数据集结构：
```
tiny-imagenet-200/
├── train/           # 按类别组织的训练图像
├── val/            # 验证图像  
├── test/           # 测试图像
├── wnids.txt       # 类别 ID
└── words.txt       # 类别名称
```

## 快速开始

### 1. 探索数据集
```bash
python scripts/explore_data.py --dataset-path ./tiny-imagenet-200
```

### 2. 训练模型
```bash
# 使用默认设置训练 ResNet-18
python scripts/train_model.py --model resnet18 --epochs 50

# 使用自定义配置训练 Vision Transformer
python scripts/train_model.py --model vit --epochs 100 --lr 0.0003

# 快速实验（更少的轮次）
python scripts/train_model.py --model resnet18 --epochs 20 --experiment-name quick_test
```

### 3. 评估训练好的模型
```bash
python scripts/evaluate_model.py --checkpoint outputs/experiment_name/checkpoints/model_best.pth
```

## 配置

项目使用 YAML 配置文件以便于实验：

### 配置示例 (configs/default.yaml)
```yaml
# 数据设置
data:
  dataset_path: "./tiny-imagenet-200"
  batch_size: 32
  num_workers: 4

# 模型设置
model:
  model_name: "resnet18"
  pretrained: true

# 训练设置
training:
  num_epochs: 50
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
```

## 训练选项

### 命令行参数
```bash
python scripts/train_model.py --help
```

主要选项：
- `--model`: 模型架构（resnet18, vit 等）
- `--epochs`: 训练轮次
- `--batch-size`: 训练批次大小
- `--lr`: 学习率
- `--optimizer`: 优化器（adam, adamw, sgd）
- `--scheduler`: 学习率调度器
- `--experiment-name`: 用于组织结果的名称

### 模型选项
1. **ResNet-18**: 快速、可靠的基线
2. **ResNet-34**: 更深的 ResNet 变体
3. **Vision Transformer**: 基于注意力的模型
4. **MobileNet-V2**: 高效的移动端模型
5. **EfficientNet-B0**: 最先进的效率（需要额外安装）

## 输出结构

每个实验会创建有组织的输出：
```
outputs/experiment_name/
├── checkpoints/
│   └── model_best.pth          # 最佳模型检查点
├── plots/
│   ├── training_history.png    # 损失/准确率曲线
│   └── ...
├── evaluation/
│   ├── confusion_matrix.png    # 混淆矩阵
│   ├── predictions.png         # 样本预测
│   └── top_errors.png         # 预测错误
├── logs/
│   └── experiment_log.json     # 详细实验日志
└── experiment_summary.txt      # 人类可读摘要
```

## 示例工作流

### 1. 比较多个模型
```bash
# 训练不同模型
python scripts/train_model.py --model resnet18 --experiment-name resnet18_baseline
python scripts/train_model.py --model vit --experiment-name vit_experiment --epochs 100
python scripts/train_model.py --model mobilenet_v2 --experiment-name mobilenet_baseline

# 分别评估
python scripts/evaluate_model.py --checkpoint outputs/resnet18_baseline/checkpoints/*_best.pth
python scripts/evaluate_model.py --checkpoint outputs/vit_experiment/checkpoints/*_best.pth
python scripts/evaluate_model.py --checkpoint outputs/mobilenet_baseline/checkpoints/*_best.pth
```

### 2. 超参数调优
```bash
# 尝试不同学习率
python scripts/train_model.py --lr 0.001 --experiment-name lr_0001
python scripts/train_model.py --lr 0.0001 --experiment-name lr_00001

# 尝试不同优化器
python scripts/train_model.py --optimizer adamw --experiment-name adamw
python scripts/train_model.py --optimizer sgd --experiment-name sgd
```

### 3. 快速调试
```bash
# 用于测试的快速实验
python scripts/train_model.py --epochs 5 --batch-size 64 --experiment-name debug
```

## 自定义

### 添加新模型
1. 在 `src/models.py` 中添加模型定义
2. 更新 `create_model()` 函数
3. 在脚本中添加模型名称到选择项

### 自定义数据增强
修改 `src/dataset.py` 中的 `get_transforms()`:
```python
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # 在此添加您的自定义变换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 自定义评估指标
在 `src/evaluate.py` 中添加函数并在 `comprehensive_evaluation()` 中调用。

## 结果解读

### 训练历史图表
- **损失曲线**: 应随时间递减
- **准确率曲线**: 应随时间递增
- **学习率**: 显示调度器行为

### 评估输出
- **总体准确率**: 主要性能指标
- **每类性能**: 识别困难的类别
- **混淆矩阵**: 显示分类模式
- **错误分析**: 突出模型弱点

## 获得最佳结果的技巧

1. **从预训练模型开始**（ResNet-18）作为基线
2. **使用数据增强**提高泛化能力
3. **监控验证准确率**检测过拟合
4. **尝试不同学习率**（0.001, 0.0001）
5. **使用余弦退火**进行学习率调度
6. **训练足够的轮次**（50+ 轮以获得好结果）

## 故障排除

### 常见问题
1. **CUDA 内存不足**: 减少批次大小
2. **训练缓慢**: 增加 num_workers，使用 GPU
3. **准确率低**: 尝试不同模型、学习率
4. **导入错误**: 检查 Python 路径和依赖
5. **数据加载溢出错误**: 如果遇到 `OverflowError` 或色相调整错误，请检查配置文件中的 `hue` 参数是否在合理范围内（推荐 ≤ 0.05）

### 性能优化
- 如果可用，使用 GPU (`--device cuda`)
- 如果内存允许，增加批次大小
- 为数据加载使用更多工作进程
- 启用混合精度训练（未来功能）

## 高级用法

### 自定义配置
创建您自己的 YAML 配置文件并修改设置：
```bash
cp configs/default.yaml configs/my_experiment.yaml
# 编辑 configs/my_experiment.yaml
# 使用 --config 标志（未来功能）
```

### 实验跟踪
所有实验都会自动记录：
- 使用的配置
- 训练历史
- 最终结果
- 系统信息

## 未来增强

计划功能：
- 混合精度训练
- 多 GPU 支持
- 超参数优化
- 模型集成
- 自定义配置文件加载
- TensorBoard 集成

## 贡献

扩展此项目：
1. 遵循模块化结构
2. 为新功能添加测试
3. 更新文档
4. 使用类型提示和文档字符串

## 许可证

本项目用于教育目的。请遵守 Tiny-ImageNet 数据集许可证。
