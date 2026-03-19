"""
模型包初始化
"""
from .enhanced_mask_rcnn import EnhancedMaskRCNN, build_enhanced_mask_rcnn

__all__ = [
    'EnhancedMaskRCNN',
    'build_enhanced_mask_rcnn'
]
