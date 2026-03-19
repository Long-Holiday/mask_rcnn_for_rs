"""
双分支骨干网络模块
主干: ResNet处理可见光影像
分支: 空间注意力层处理短波红外影像
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class SpatialAttention(nn.Module):
    """空间注意力模块，用于SWIR分支（仅空间注意力，无通道注意力）"""
    
    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: 输入通道数
        """
        super().__init__()
        
        # 空间注意力：使用平均池化和最大池化生成空间注意力图
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            空间注意力加权后的特征 [B, C, H, W]
        """
        # 生成空间注意力图
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_att = self.spatial_attention(spatial_input)  # [B, 1, H, W]
        
        # 应用空间注意力
        x = x * spatial_att
        
        return x


class SWIRBranch(nn.Module):
    """SWIR分支：轻量级特征提取 + 空间注意力（单通道输入）"""
    
    def __init__(self, in_channels: int = 1, out_channels_list: list = [256, 512, 1024, 2048]):
        """
        Args:
            in_channels: 输入通道数（SWIR影像通道数，默认为1）
            out_channels_list: 各层输出通道数，与ResNet对齐 [res2, res3, res4, res5]
        """
        super().__init__()
        
        # 轻量级卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 各阶段的空间注意力模块（4个阶段对应res2-res5）
        self.stage1 = self._make_stage(64, out_channels_list[0])
        self.stage2 = self._make_stage(out_channels_list[0], out_channels_list[1], stride=2)
        self.stage3 = self._make_stage(out_channels_list[1], out_channels_list[2], stride=2)
        self.stage4 = self._make_stage(out_channels_list[2], out_channels_list[3], stride=2)
        
    def _make_stage(self, in_channels, out_channels, stride=1):
        """创建一个阶段：卷积 + 空间注意力"""
        layers = []
        
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 空间注意力
        layers.append(SpatialAttention(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: SWIR影像 [B, C, H, W]
        Returns:
            多尺度特征字典 {stage_name: feature}
        """
        features = {}
        
        x = self.conv1(x)
        features['stem'] = x
        
        x = self.stage1(x)
        features['res2'] = x
        
        x = self.stage2(x)
        features['res3'] = x
        
        x = self.stage3(x)
        features['res4'] = x
        
        x = self.stage4(x)
        features['res5'] = x
        
        return features


class DualBranchBackbone(nn.Module):
    """双分支骨干网络：ResNet主干 + SWIR空间注意力分支"""
    
    def __init__(self, 
                 rgb_channels: int = 3,
                 swir_channels: int = 1,
                 pretrained: bool = True,
                 fusion_method: str = 'add'):
        """
        Args:
            rgb_channels: RGB影像通道数
            swir_channels: SWIR影像通道数（默认为1，单通道）
            pretrained: 是否使用预训练ResNet
            fusion_method: 特征融合方式 ('add', 'concat', 'weighted')
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # 主干：ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50(weights=None)
        
        # 提取ResNet各阶段
        self.rgb_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.rgb_res2 = resnet.layer1  # 256
        self.rgb_res3 = resnet.layer2  # 512
        self.rgb_res4 = resnet.layer3  # 1024
        self.rgb_res5 = resnet.layer4  # 2048
        
        # SWIR分支
        self.swir_branch = SWIRBranch(
            in_channels=swir_channels,
            out_channels_list=[256, 512, 1024, 2048]
        )
        
        # 特征融合层
        if fusion_method == 'concat':
            self.fusion_res2 = nn.Conv2d(512, 256, 1)
            self.fusion_res3 = nn.Conv2d(1024, 512, 1)
            self.fusion_res4 = nn.Conv2d(2048, 1024, 1)
            self.fusion_res5 = nn.Conv2d(4096, 2048, 1)
        elif fusion_method == 'weighted':
            self.weight_res2 = nn.Parameter(torch.ones(2))
            self.weight_res3 = nn.Parameter(torch.ones(2))
            self.weight_res4 = nn.Parameter(torch.ones(2))
            self.weight_res5 = nn.Parameter(torch.ones(2))
    
    def _fuse_features(self, rgb_feat, swir_feat, stage_name):
        """特征融合（自动对齐空间尺寸）"""
        # 如果空间尺寸不匹配，将SWIR特征resize到RGB特征的尺寸
        if rgb_feat.shape[2:] != swir_feat.shape[2:]:
            swir_feat = F.interpolate(
                swir_feat, 
                size=rgb_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        if self.fusion_method == 'add':
            return rgb_feat + swir_feat
        
        elif self.fusion_method == 'concat':
            # 对齐空间尺寸后再concat
            fused = torch.cat([rgb_feat, swir_feat], dim=1)
            if stage_name == 'res2':
                return self.fusion_res2(fused)
            elif stage_name == 'res3':
                return self.fusion_res3(fused)
            elif stage_name == 'res4':
                return self.fusion_res4(fused)
            elif stage_name == 'res5':
                return self.fusion_res5(fused)
        
        elif self.fusion_method == 'weighted':
            if stage_name == 'res2':
                weights = F.softmax(self.weight_res2, dim=0)
            elif stage_name == 'res3':
                weights = F.softmax(self.weight_res3, dim=0)
            elif stage_name == 'res4':
                weights = F.softmax(self.weight_res4, dim=0)
            elif stage_name == 'res5':
                weights = F.softmax(self.weight_res5, dim=0)
            
            # 空间尺寸已在函数开头对齐
            return weights[0] * rgb_feat + weights[1] * swir_feat
        
        return rgb_feat + swir_feat
    
    def forward(self, rgb_images, swir_images):
        """
        Args:
            rgb_images: RGB影像 [B, 3, H, W]
            swir_images: SWIR影像 [B, 3, H, W]
        Returns:
            融合后的多尺度特征字典
        """
        # RGB主干特征提取
        rgb_x = self.rgb_stem(rgb_images)
        rgb_res2 = self.rgb_res2(rgb_x)
        rgb_res3 = self.rgb_res3(rgb_res2)
        rgb_res4 = self.rgb_res4(rgb_res3)
        rgb_res5 = self.rgb_res5(rgb_res4)
        
        # SWIR分支特征提取
        swir_features = self.swir_branch(swir_images)
        
        # 特征融合
        fused_features = {
            'res2': self._fuse_features(rgb_res2, swir_features['res2'], 'res2'),
            'res3': self._fuse_features(rgb_res3, swir_features['res3'], 'res3'),
            'res4': self._fuse_features(rgb_res4, swir_features['res4'], 'res4'),
            'res5': self._fuse_features(rgb_res5, swir_features['res5'], 'res5'),
        }
        
        return fused_features


def test_dual_branch_backbone():
    """测试双分支骨干网络"""
    print("测试双分支骨干网络...")
    
    # 创建模型
    model = DualBranchBackbone(
        rgb_channels=3,
        swir_channels=1,
        pretrained=False,
        fusion_method='add'
    )
    
    # 测试输入
    batch_size = 2
    rgb_input = torch.randn(batch_size, 3, 1024, 1024)
    swir_input = torch.randn(batch_size, 1, 1024, 1024)  # 单通道SWIR
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        features = model(rgb_input, swir_input)
    
    # 打印特征尺寸
    print("\n输出特征尺寸:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n测试通过！")


if __name__ == '__main__':
    test_dual_branch_backbone()
