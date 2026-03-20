"""
三模态并行骨干网络模块
主干: 标准ResNet50处理可见光影像
分支: 轻量化注意力（通道+空间）处理短波红外影像
NIR影像移动到掩码生成头，不再在骨干网络中融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ChannelAttention(nn.Module):
    """通道注意力模块（轻量化）"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Args:
            in_channels: 输入通道数
            reduction_ratio: 通道压缩比例
        """
        super().__init__()
        
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            通道注意力加权后的特征 [B, C, H, W]
        """
        # 平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1]
        
        # 融合并生成通道注意力权重
        channel_att = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        
        # 应用通道注意力
        return x * channel_att


class SpatialAttention(nn.Module):
    """空间注意力模块（轻量化）"""
    
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: 卷积核大小
        """
        super().__init__()
        
        # 空间注意力：使用平均池化和最大池化生成空间注意力图
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
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
        return x * spatial_att


class LightweightAttention(nn.Module):
    """轻量化注意力模块：通道注意力 + 空间注意力"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Args:
            in_channels: 输入通道数
            reduction_ratio: 通道压缩比例
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            注意力加权后的特征 [B, C, H, W]
        """
        # 先通道注意力，后空间注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class LightweightBranch(nn.Module):
    """轻量化分支：用于SWIR的轻量级特征提取（单通道 input）"""
    
    def __init__(self, in_channels: int = 1, out_channels_list: list = [256, 512, 1024, 2048]):
        """
        Args:
            in_channels: 输入通道数（SWIR影像通道数，默认为1）
            out_channels_list: 各层输出通道数，与ResNet对齐 [res2, res3, res4, res5]
        """
        super().__init__()
        
        # 轻量级初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 各阶段的轻量化注意力模块（4个阶段对应res2-res5）
        self.stage1 = self._make_stage(64, out_channels_list[0])
        self.stage2 = self._make_stage(out_channels_list[0], out_channels_list[1], stride=2)
        self.stage3 = self._make_stage(out_channels_list[1], out_channels_list[2], stride=2)
        self.stage4 = self._make_stage(out_channels_list[2], out_channels_list[3], stride=2)
        
    def _make_stage(self, in_channels, out_channels, stride=1):
        """创建一个阶段：卷积 + 轻量化注意力（通道+空间）"""
        layers = []
        
        # 轻量级卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 轻量化注意力（通道 + 空间）
        layers.append(LightweightAttention(out_channels))
        
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


class AdaptiveFusionModule(nn.Module):
    """自适应特征融合模块（双模态：RGB + SWIR）"""
    
    def __init__(self, channels: int):
        """
        Args:
            channels: 特征通道数
        """
        super().__init__()
        
        # 全局上下文提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 自适应权重生成网络（双模态）
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征校准
        self.calibration = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, rgb_feat, swir_feat):
        """
        Args:
            rgb_feat: RGB特征 [B, C, H, W]
            swir_feat: SWIR特征 [B, C, H, W]
        Returns:
            自适应融合后的特征 [B, C, H, W]
        """
        # 拼接双模态特征
        concat_feat = torch.cat([rgb_feat, swir_feat], dim=1)  # [B, 2C, H, W]
        
        # 全局池化获取上下文
        global_context = self.global_pool(concat_feat)  # [B, 2C, 1, 1]
        
        # 生成自适应权重
        weights = self.weight_net(global_context)  # [B, 2, 1, 1]
        
        # 加权融合
        fused = weights[:, 0:1] * rgb_feat + weights[:, 1:2] * swir_feat
        
        # 特征校准
        fused = self.calibration(fused)
        
        return fused


class TrimodalBackbone(nn.Module):
    """双模态并行骨干网络：可见光ResNet主干 + SWIR轻量化注意力分支（移除NIR）"""
    
    def __init__(self, 
                 rgb_channels: int = 3,
                 nir_channels: int = 1,
                 swir_channels: int = 1,
                 pretrained: bool = True,
                 fusion_method: str = 'adaptive'):
        """
        Args:
            rgb_channels: RGB影像通道数
            nir_channels: NIR影像通道数（保留参数以兼容，但不使用）
            swir_channels: SWIR影像通道数（默认为1，单通道）
            pretrained: 是否使用预训练ResNet
            fusion_method: 特征融合方式 ('add', 'concat', 'weighted', 'adaptive')
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # 主干：标准ResNet50（可见光）
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
        
        # SWIR轻量化分支（通道+空间注意力）
        self.swir_branch = LightweightBranch(
            in_channels=swir_channels,
            out_channels_list=[256, 512, 1024, 2048]
        )
        
        # 在每个stage后加入SWIR的轻量化注意力
        self.swir_attention_res2 = LightweightAttention(256)
        self.swir_attention_res3 = LightweightAttention(512)
        self.swir_attention_res4 = LightweightAttention(1024)
        self.swir_attention_res5 = LightweightAttention(2048)
        
        # 特征融合层（双模态）
        if fusion_method == 'concat':
            # 双模态拼接：RGB + SWIR
            self.fusion_res2 = nn.Conv2d(512, 256, 1)
            self.fusion_res3 = nn.Conv2d(1024, 512, 1)
            self.fusion_res4 = nn.Conv2d(2048, 1024, 1)
            self.fusion_res5 = nn.Conv2d(4096, 2048, 1)
        elif fusion_method == 'weighted':
            # 双模态加权融合
            self.weight_res2 = nn.Parameter(torch.ones(2))
            self.weight_res3 = nn.Parameter(torch.ones(2))
            self.weight_res4 = nn.Parameter(torch.ones(2))
            self.weight_res5 = nn.Parameter(torch.ones(2))
        elif fusion_method == 'adaptive':
            # 自适应融合模块（双模态）
            self.adaptive_fusion_res2 = AdaptiveFusionModule(256)
            self.adaptive_fusion_res3 = AdaptiveFusionModule(512)
            self.adaptive_fusion_res4 = AdaptiveFusionModule(1024)
            self.adaptive_fusion_res5 = AdaptiveFusionModule(2048)
    
    def _fuse_features(self, rgb_feat, swir_feat, stage_name):
        """双模态特征融合（自动对齐空间尺寸）"""
        # 对齐SWIR特征到RGB特征的空间尺寸
        if swir_feat.shape[2:] != rgb_feat.shape[2:]:
            swir_feat = F.interpolate(
                swir_feat, 
                size=rgb_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        if self.fusion_method == 'add':
            # 简单相加融合
            return rgb_feat + swir_feat
        
        elif self.fusion_method == 'concat':
            # 双模态拼接后通过1x1卷积降维
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
            # 可学习的加权融合
            if stage_name == 'res2':
                weights = F.softmax(self.weight_res2, dim=0)
            elif stage_name == 'res3':
                weights = F.softmax(self.weight_res3, dim=0)
            elif stage_name == 'res4':
                weights = F.softmax(self.weight_res4, dim=0)
            elif stage_name == 'res5':
                weights = F.softmax(self.weight_res5, dim=0)
            
            return weights[0] * rgb_feat + weights[1] * swir_feat
        
        elif self.fusion_method == 'adaptive':
            # 自适应融合
            if stage_name == 'res2':
                return self.adaptive_fusion_res2(rgb_feat, swir_feat)
            elif stage_name == 'res3':
                return self.adaptive_fusion_res3(rgb_feat, swir_feat)
            elif stage_name == 'res4':
                return self.adaptive_fusion_res4(rgb_feat, swir_feat)
            elif stage_name == 'res5':
                return self.adaptive_fusion_res5(rgb_feat, swir_feat)
        
        return rgb_feat + swir_feat
    
    def forward(self, rgb_images, nir_images, swir_images):
        """
        Args:
            rgb_images: RGB影像 [B, 3, H, W]
            nir_images: NIR影像 [B, 1, H, W]（保留参数以兼容，但不使用）
            swir_images: SWIR影像 [B, 1, H, W]
        Returns:
            融合后的多尺度特征字典
        """
        # RGB主干特征提取（标准ResNet50）
        rgb_x = self.rgb_stem(rgb_images)
        rgb_res2 = self.rgb_res2(rgb_x)
        rgb_res3 = self.rgb_res3(rgb_res2)
        rgb_res4 = self.rgb_res4(rgb_res3)
        rgb_res5 = self.rgb_res5(rgb_res4)
        
        # SWIR分支特征提取
        swir_features = self.swir_branch(swir_images)
        
        # 在每个stage后应用SWIR的轻量化注意力
        swir_res2 = self.swir_attention_res2(swir_features['res2'])
        swir_res3 = self.swir_attention_res3(swir_features['res3'])
        swir_res4 = self.swir_attention_res4(swir_features['res4'])
        swir_res5 = self.swir_attention_res5(swir_features['res5'])
        
        # 双模态特征融合（RGB + SWIR）
        fused_features = {
            'res2': self._fuse_features(rgb_res2, swir_res2, 'res2'),
            'res3': self._fuse_features(rgb_res3, swir_res3, 'res3'),
            'res4': self._fuse_features(rgb_res4, swir_res4, 'res4'),
            'res5': self._fuse_features(rgb_res5, swir_res5, 'res5'),
        }
        
        return fused_features


def test_trimodal_backbone():
    """测试双模态并行骨干网络（RGB + SWIR，移除NIR）"""
    print("测试双模态并行骨干网络（RGB + SWIR）...")
    
    # 创建模型
    model = TrimodalBackbone(
        rgb_channels=3,
        nir_channels=1,
        swir_channels=1,
        pretrained=False,
        fusion_method='adaptive'
    )
    
    # 测试输入
    batch_size = 2
    rgb_input = torch.randn(batch_size, 3, 1024, 1024)
    nir_input = torch.randn(batch_size, 1, 1024, 1024)  # 保留但不使用
    swir_input = torch.randn(batch_size, 1, 1024, 1024)  # 单通道SWIR
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        features = model(rgb_input, nir_input, swir_input)
    
    # 打印特征尺寸
    print("\n输出特征尺寸:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n测试通过！（NIR已移除，仅使用RGB + SWIR）")


if __name__ == '__main__':
    test_trimodal_backbone()
