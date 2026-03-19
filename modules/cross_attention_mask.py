"""
交叉注意力掩码分支模块
使用NIR影像作为Key/Value，增强掩码预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """交叉注意力模块"""
    
    def __init__(self, 
                 query_dim: int,
                 key_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            query_dim: Query特征维度
            key_dim: Key/Value特征维度
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()
        
        assert query_dim % num_heads == 0, "query_dim必须能被num_heads整除"
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query投影
        self.q_proj = nn.Linear(query_dim, query_dim)
        
        # Key和Value投影
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [B, N_q, query_dim] 掩码特征
            key: [B, N_k, key_dim] NIR特征
            value: [B, N_k, key_dim] NIR特征
            mask: 可选的注意力掩码
        Returns:
            增强后的query特征 [B, N_q, query_dim]
        """
        batch_size = query.size(0)
        
        # 线性投影
        Q = self.q_proj(query)  # [B, N_q, query_dim]
        K = self.k_proj(key)    # [B, N_k, query_dim]
        V = self.v_proj(value)  # [B, N_k, query_dim]
        
        # 重塑为多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_q, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_k, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_k, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, N_q, N_k]
        
        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N_q, N_k]
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, N_q, head_dim]
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, N_q, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, -1, self.query_dim)  # [B, N_q, query_dim]
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class NIRFeatureExtractor(nn.Module):
    """NIR特征提取器，用于生成Key/Value（单通道输入）"""
    
    def __init__(self, in_channels: int = 1, feature_dim: int = 256):
        """
        Args:
            in_channels: NIR影像通道数（默认为1，单通道）
            feature_dim: 输出特征维度
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, nir_images):
        """
        Args:
            nir_images: NIR影像 [B, C, H, W]
        Returns:
            NIR特征 [B, feature_dim, H', W']
        """
        return self.conv_layers(nir_images)


class CrossAttentionMaskHead(nn.Module):
    """带交叉注意力的掩码预测头"""
    
    def __init__(self,
                 in_channels: int = 256,
                 num_classes: int = 80,
                 num_convs: int = 4,
                 nir_channels: int = 1,
                 num_attention_heads: int = 8,
                 use_cross_attention: bool = True):
        """
        Args:
            in_channels: 输入特征通道数
            num_classes: 类别数
            num_convs: 卷积层数量
            nir_channels: NIR影像通道数（默认为1，单通道）
            num_attention_heads: 交叉注意力头数
            use_cross_attention: 是否使用交叉注意力
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_cross_attention = use_cross_attention
        
        # 标准掩码卷积层
        self.mask_convs = nn.ModuleList()
        for i in range(num_convs):
            self.mask_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # NIR特征提取器
        if use_cross_attention:
            self.nir_extractor = NIRFeatureExtractor(nir_channels, in_channels)
            
            # 交叉注意力模块
            self.cross_attention = CrossAttention(
                query_dim=in_channels,
                key_dim=in_channels,
                num_heads=num_attention_heads,
                dropout=0.1
            )
            
            # 特征融合层
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        
        # 上采样层
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        
        # 最终预测层
        self.predictor = nn.Conv2d(in_channels, num_classes, 1)
        
    def forward(self, roi_features, nir_images=None, roi_boxes=None):
        """
        Args:
            roi_features: RoI特征 [N, C, H, W]
            nir_images: NIR影像 [B, C, H_img, W_img]
            roi_boxes: RoI边界框坐标，用于从NIR提取对应区域
        Returns:
            掩码预测 [N, num_classes, H*2, W*2]
        """
        x = roi_features
        
        # 标准卷积层
        for conv in self.mask_convs:
            x = conv(x)
        
        # 交叉注意力增强
        if self.use_cross_attention and nir_images is not None:
            # 提取NIR特征
            nir_features = self.nir_extractor(nir_images)  # [B, C, H', W']
            
            # 将特征展平为序列
            N, C, H, W = x.shape
            x_flat = x.view(N, C, H * W).transpose(1, 2)  # [N, H*W, C]
            
            # 从NIR特征中提取对应RoI区域（简化版本：使用全局特征）
            B, C_nir, H_nir, W_nir = nir_features.shape
            nir_flat = nir_features.view(B, C_nir, H_nir * W_nir).transpose(1, 2)  # [B, H'*W', C]
            
            # 扩展NIR特征以匹配RoI数量
            # 简化处理：假设每个RoI使用相同的NIR特征
            nir_flat_expanded = nir_flat[0:1].expand(N, -1, -1)  # [N, H'*W', C]
            
            # 交叉注意力
            attn_output, _ = self.cross_attention(
                query=x_flat,
                key=nir_flat_expanded,
                value=nir_flat_expanded
            )  # [N, H*W, C]
            
            # 重塑回空间维度
            attn_output = attn_output.transpose(1, 2).view(N, C, H, W)  # [N, C, H, W]
            
            # 融合原始特征和注意力特征
            x = torch.cat([x, attn_output], dim=1)
            x = self.fusion_conv(x)
        
        # 上采样
        x = self.deconv(x)
        x = F.relu(x)
        
        # 预测掩码
        mask_pred = self.predictor(x)
        
        return mask_pred


class EnhancedMaskRCNNHead(nn.Module):
    """增强版Mask R-CNN头，集成交叉注意力"""
    
    def __init__(self,
                 in_channels: int = 256,
                 num_classes: int = 80,
                 roi_size: int = 14,
                 nir_channels: int = 1,
                 use_cross_attention: bool = True):
        """
        Args:
            in_channels: 输入特征通道数
            num_classes: 类别数
            roi_size: RoI Align输出尺寸
            nir_channels: NIR影像通道数（默认为1，单通道）
            use_cross_attention: 是否使用交叉注意力
        """
        super().__init__()
        
        self.mask_head = CrossAttentionMaskHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_convs=4,
            nir_channels=nir_channels,
            num_attention_heads=8,
            use_cross_attention=use_cross_attention
        )
        
    def forward(self, roi_features, nir_images=None, roi_boxes=None):
        """
        Args:
            roi_features: RoI特征 [N, C, H, W]
            nir_images: NIR影像 [B, C, H, W]
            roi_boxes: RoI边界框
        Returns:
            掩码预测 [N, num_classes, H*2, W*2]
        """
        return self.mask_head(roi_features, nir_images, roi_boxes)


def test_cross_attention_mask():
    """测试交叉注意力掩码模块"""
    print("测试交叉注意力掩码模块...")
    
    # 创建模型
    model = EnhancedMaskRCNNHead(
        in_channels=256,
        num_classes=80,
        roi_size=14,
        nir_channels=1,  # NIR单通道
        use_cross_attention=True
    )
    
    # 测试输入
    batch_size = 2
    num_rois = 10
    roi_features = torch.randn(num_rois, 256, 14, 14)
    nir_images = torch.randn(batch_size, 1, 1024, 1024)  # NIR单通道
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        mask_pred = model(roi_features, nir_images)
    
    print(f"\nRoI特征输入: {roi_features.shape}")
    print(f"NIR影像输入: {nir_images.shape}")
    print(f"掩码预测输出: {mask_pred.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    print("\n测试通过！")


if __name__ == '__main__':
    test_cross_attention_mask()
