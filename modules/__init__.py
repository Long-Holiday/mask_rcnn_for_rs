"""
模块包初始化
"""
from .anchor_generator import AnchorGenerator
from .dual_branch_backbone import TrimodalBackbone, SpatialAttention, LightweightBranch
from .cross_attention_mask import (
    CrossAttention, 
    NIRFeatureExtractor, 
    EnhancedMaskRCNNHead,
    CrossAttentionMaskHead
)
from .dataset import (
    MultiModalRemoteSensingDataset,
    get_transforms,
    get_single_channel_transforms,
    collate_fn
)

__all__ = [
    'AnchorGenerator',
    'TrimodalBackbone',
    'SpatialAttention',
    'LightweightBranch',
    'CrossAttention',
    'NIRFeatureExtractor',
    'EnhancedMaskRCNNHead',
    'CrossAttentionMaskHead',
    'MultiModalRemoteSensingDataset',
    'get_transforms',
    'get_single_channel_transforms',
    'collate_fn'
]
