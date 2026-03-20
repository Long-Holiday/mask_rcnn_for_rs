# 增强版Mask R-CNN遥感影像实例分割

基于PyTorch实现的多模态遥感影像实例分割系统，包含三个主要改进：
1. 基于KMeans的自适应锚框生成
2. 双分支骨干网络（ResNet主干 + SWIR空间注意力分支）
3. 交叉注意力掩码分支（使用NIR影像增强）

## 项目结构

```
code/
├── modules/                      # 核心模块（高内聚低耦合设计）
│   ├── anchor_generator.py      # KMeans锚框生成模块
│   ├── dual_branch_backbone.py  # 双分支骨干网络
│   ├── cross_attention_mask.py  # 交叉注意力掩码模块
│   └── dataset.py               # 多模态数据加载器
│
├── models/                       # 模型定义
│   └── enhanced_mask_rcnn.py    # 增强版Mask R-CNN集成
│
├── train.py                      # 训练脚本
├── evaluate.py                   # 评估脚本
├── ablation_study.py            # 消融实验脚本
├── requirements.txt             # 依赖包
└── 使用说明.md                  # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用流程

### 步骤1: 生成最优锚框（独立运行）

```bash
python modules/anchor_generator.py
```

这将：
- 分析数据集中所有边界框的尺寸分布
- 使用KMeans聚类计算最优锚框大小
- 生成可视化结果：`outputs/anchors/anchor_visualization.png`
- 保存锚框配置：`outputs/anchors/anchor_config.json`

### 步骤2: 训练模型

基础训练：
```bash
python train.py
```

自定义参数训练：
```bash
python train.py \
    --data_root ./instance_segmentation_dataset \
    --batch_size 4 \
    --num_epochs 50 \
    --lr 0.005
```

从检查点恢复训练：
```bash
python train.py --resume ./outputs/training/checkpoint_latest.pth
```

### 步骤3: 评估模型

```bash
python evaluate.py --checkpoint ./outputs/training/checkpoint_best.pth
```

### 步骤4: 消融实验

运行所有消融实验：
```bash
python ablation_study.py --experiments all --num_epochs 30
```

运行特定实验：
```bash
python ablation_study.py --experiments baseline no_cross_attention
```

可用的消融实验：
- `baseline`: 完整模型（所有组件）
- `no_cross_attention`: 不使用交叉注意力
- `fusion_concat`: 使用拼接融合代替相加融合
- `fusion_weighted`: 使用可学习加权融合

## 模块说明

### 1. KMeans锚框生成模块 (`anchor_generator.py`)

**设计目标**: 独立的锚框优化模块，与主模型解耦

**核心功能**:
- 从COCO格式标注文件中提取所有边界框尺寸
- 使用KMeans聚类找到最优锚框大小
- 生成可视化和配置文件

**独立运行**:
```python
from modules.anchor_generator import AnchorGenerator

generator = AnchorGenerator(
    annotation_path='./instance_segmentation_dataset/annotations/instances.json',
    n_clusters=9
)
config = generator.run(output_dir='./outputs/anchors')
```

### 2. 双分支骨干网络 (`dual_branch_backbone.py`)

**设计目标**: 融合RGB和SWIR两种模态的特征

**架构**:
- **主干**: ResNet50处理RGB影像（3通道，可选预训练权重）
- **分支**: 轻量级空间注意力网络处理SWIR影像（单通道）
- **融合**: 支持三种融合方式（add/concat/weighted）

**关键组件**:
- `SpatialAttention`: 空间注意力模块（仅空间注意力，无通道注意力）
- `SWIRBranch`: SWIR特征提取分支（单通道输入）
- `DualBranchBackbone`: 双分支主网络

**使用示例**:
```python
from modules.dual_branch_backbone import DualBranchBackbone

backbone = DualBranchBackbone(
    rgb_channels=3,
    swir_channels=1,  # SWIR单通道
    pretrained=True,
    fusion_method='add'
)

features = backbone(rgb_images, swir_images)
# 输出: {'res2': ..., 'res3': ..., 'res4': ..., 'res5': ...}
```

### 3. 交叉注意力掩码模块 (`cross_attention_mask.py`)

**设计目标**: 使用NIR影像增强掩码预测

**架构**:
- **NIR特征提取器**: 轻量级CNN提取NIR特征作为Key/Value（单通道输入）
- **交叉注意力**: 多头注意力机制，Query来自RoI特征
- **掩码预测头**: 集成交叉注意力的掩码分支

**关键组件**:
- `CrossAttention`: 多头交叉注意力
- `NIRFeatureExtractor`: NIR特征提取（单通道输入）
- `EnhancedMaskRCNNHead`: 增强掩码预测头

**使用示例**:
```python
from modules.cross_attention_mask import EnhancedMaskRCNNHead

mask_head = EnhancedMaskRCNNHead(
    in_channels=256,
    num_classes=4,
    nir_channels=1,  # NIR单通道
    use_cross_attention=True
)

mask_pred = mask_head(roi_features, nir_images)
```

### 4. 多模态数据加载器 (`dataset.py`)

**设计目标**: 同步加载RGB、SWIR、NIR三种模态

**影像格式**:
- **RGB**: 3通道彩色影像
- **SWIR**: 单通道灰度影像
- **NIR**: 单通道灰度影像

**功能**:
- 支持COCO格式标注
- 自动划分训练/验证集
- 数据增强（翻转、归一化等）
- 自定义collate函数处理多模态batch

**使用示例**:
```python
from modules.dataset import MultiModalRemoteSensingDataset, get_transforms, collate_fn
from torch.utils.data import DataLoader

dataset = MultiModalRemoteSensingDataset(
    root_dir='./instance_segmentation_dataset',
    annotation_file='./instance_segmentation_dataset/annotations/instances.json',
    split='train',
    transforms=get_transforms(train=True)
)

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
```

## 消融实验设计

模块化设计使得消融实验非常简单：

### 1. 测试交叉注意力的贡献
```bash
# 有交叉注意力
python train.py --output_dir ./outputs/with_ca

# 无交叉注意力
python train.py --no_cross_attention --output_dir ./outputs/without_ca
```

### 2. 测试不同融合方式
```bash
# 相加融合
python train.py --fusion_method add

# 拼接融合
python train.py --fusion_method concat

# 加权融合
python train.py --fusion_method weighted
```

### 3. 测试锚框优化的影响
```bash
# 使用KMeans锚框
python train.py

# 使用默认锚框（修改代码中anchor_config_path=None）
```

## 训练配置

默认配置（可在`train.py`中修改）：

```python
{
    'batch_size': 2,
    'num_epochs': 50,
    'learning_rate': 0.005,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'lr_milestones': [30, 40],
    'lr_gamma': 0.1,
    'backbone_pretrained': True,
    'fusion_method': 'add',
    'use_cross_attention': True
}
```

## 输出文件

训练过程会生成以下文件：

```
outputs/
├── anchors/                          # 锚框生成结果
│   ├── anchor_config.json           # 锚框配置
│   └── anchor_visualization.png     # 可视化
│
├── training/                         # 训练输出
│   ├── logs/                        # TensorBoard日志
│   ├── config.json                  # 训练配置
│   ├── checkpoint_best.pth          # 最佳模型
│   ├── checkpoint_latest.pth        # 最新模型
│   └── checkpoint_epoch_*.pth       # 定期保存
│
├── evaluation/                       # 评估结果
│   └── results.json                 # 检测结果
│
└── ablation/                         # 消融实验
    ├── baseline/
    ├── no_cross_attention/
    ├── fusion_concat/
    ├── fusion_weighted/
    └── summary.json                 # 实验总结
```

## 监控训练

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir ./outputs/training/logs
```

## 模型特点

### 高内聚低耦合设计

1. **独立模块**: 每个改进点都是独立的模块，可单独测试
2. **清晰接口**: 模块间通过标准的tensor接口交互
3. **易于消融**: 通过配置参数即可开关各个组件
4. **便于扩展**: 可轻松添加新的特征融合方式或注意力机制

### 改进点对应的模块

| 改进点 | 模块文件 | 类名 | 消融方法 |
|--------|---------|------|---------|
| KMeans锚框 | `anchor_generator.py` | `AnchorGenerator` | 设置`anchor_config_path=None` |
| 双分支骨干网络 | `dual_branch_backbone.py` | `DualBranchBackbone` | 修改`fusion_method`参数 |
| 交叉注意力 | `cross_attention_mask.py` | `CrossAttention` | 设置`use_cross_attention=False` |

## 注意事项

1. **GPU内存**: 建议至少8GB显存，batch_size=2
2. **数据格式**: 
   - 确保RGB、SWIR、NIR影像文件名一致
   - RGB影像为3通道PNG格式
   - SWIR和NIR影像为单通道PNG格式（灰度图）
3. **类别数量**: 根据实际数据集修改`num_classes`（包括背景类）
4. **预训练权重**: 首次运行会自动下载ResNet预训练权重

## 测试模块

每个模块都包含独立的测试函数：

```bash
# 测试双分支骨干网络
python modules/dual_branch_backbone.py

# 测试交叉注意力模块
python modules/cross_attention_mask.py

# 测试数据加载器
python modules/dataset.py

# 测试完整模型
python models/enhanced_mask_rcnn.py
```

## 常见问题

### Q: 如何修改锚框数量？
A: 在`anchor_generator.py`中修改`n_clusters`参数

### Q: 如何使用单模态训练？
A: 修改数据加载器，将SWIR或NIR设置为RGB的副本

### Q: 如何添加新的融合方式？
A: 在`DualBranchBackbone._fuse_features()`中添加新的分支

### Q: 训练速度慢怎么办？
A: 
- 减小batch_size
- 设置`backbone_pretrained=False`加快初始化
- 减小输入图像尺寸（修改`min_size`和`max_size`）

## 引用

如果使用本代码，请引用相关论文：
- Mask R-CNN: He et al., "Mask R-CNN", ICCV 2017
- ResNet: He et al., "Deep Residual Learning", CVPR 2016
- Attention: Vaswani et al., "Attention is All You Need", NeurIPS 2017
