"""
增强版Mask R-CNN模型
集成双分支骨干网络和交叉注意力掩码分支
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
import sys
sys.path.append('..')

from modules.dual_branch_backbone import DualBranchBackbone
from modules.cross_attention_mask import EnhancedMaskRCNNHead
from typing import Dict, List, Tuple, Optional




class BackboneWithFPN(nn.Module):
    """将双分支骨干网络与FPN结合"""
    
    def __init__(self, 
                 backbone: DualBranchBackbone,
                 return_layers: Dict[str, str] = None,
                 in_channels_list: List[int] = None,
                 out_channels: int = 256):
        """
        Args:
            backbone: 双分支骨干网络
            return_layers: 返回的层
            in_channels_list: 各层输入通道数
            out_channels: FPN输出通道数
        """
        super().__init__()
        
        self.backbone = backbone
        
        if return_layers is None:
            return_layers = {'res2': '0', 'res3': '1', 'res4': '2', 'res5': '3'}
        
        if in_channels_list is None:
            in_channels_list = [256, 512, 1024, 2048]
        
        self.return_layers = return_layers
        
        # 创建FPN
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )
        
        self.out_channels = out_channels
    
    def forward(self, rgb_images, swir_images):
        """
        Args:
            rgb_images: RGB影像
            swir_images: SWIR影像
        Returns:
            FPN特征字典
        """
        # 双分支骨干网络
        features = self.backbone(rgb_images, swir_images)
        
        # 重新组织特征字典以匹配FPN输入
        fpn_input = {}
        for name, new_name in self.return_layers.items():
            fpn_input[new_name] = features[name]
        
        # FPN
        fpn_output = self.fpn(fpn_input)
        
        return fpn_output


class EnhancedMaskRCNN(nn.Module):
    """增强版Mask R-CNN"""
    
    def __init__(self,
                 num_classes: int,
                 anchor_sizes: List[Tuple[int, ...]] = None,
                 anchor_aspect_ratios: List[Tuple[float, ...]] = None,
                 backbone_pretrained: bool = True,
                 fusion_method: str = 'add',
                 use_cross_attention: bool = True,
                 min_size: int = 800,
                 max_size: int = 1333):
        """
        Args:
            num_classes: 类别数（包括背景）
            anchor_sizes: 锚框大小
            anchor_aspect_ratios: 锚框宽高比
            backbone_pretrained: 骨干网络是否使用预训练权重
            fusion_method: 特征融合方式
            use_cross_attention: 是否使用交叉注意力
            min_size: 输入图像最小尺寸
            max_size: 输入图像最大尺寸
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_cross_attention = use_cross_attention
        
        # 默认锚框配置（4个尺度匹配FPN的4个特征层）
        if anchor_sizes is None:
            anchor_sizes = ((64,), (128,), (256,), (512,))
        if anchor_aspect_ratios is None:
            anchor_aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        # 1. 双分支骨干网络 + FPN
        backbone = DualBranchBackbone(
            rgb_channels=3,
            swir_channels=1,  # SWIR单通道
            pretrained=backbone_pretrained,
            fusion_method=fusion_method
        )
        
        self.backbone_with_fpn = BackboneWithFPN(
            backbone=backbone,
            return_layers={'res2': '0', 'res3': '1', 'res4': '2', 'res5': '3'},
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # 2. RPN
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_aspect_ratios
        )
        
        rpn_head = RPNHead(
            in_channels=256,
            num_anchors=anchor_generator.num_anchors_per_location()[0]
        )
        
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
        )
        
        # 3. RoI Heads
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2
        )
        
        # 边界框预测头
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            in_channels=256 * 7 * 7,
            representation_size=1024
        )
        
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_channels=1024,
            num_classes=num_classes
        )
        
        # 增强掩码预测头
        mask_head = EnhancedMaskRCNNHead(
            in_channels=256,
            num_classes=num_classes,
            roi_size=14,
            nir_channels=1,  # NIR单通道
            use_cross_attention=use_cross_attention
        )
        
        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=None  # 已集成在mask_head中
        )
        
        # 4. Transform
        self.transform = GeneralizedRCNNTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
        
    def forward(self, 
                rgb_images: torch.Tensor,
                swir_images: torch.Tensor,
                nir_images: torch.Tensor,
                targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        Args:
            rgb_images: RGB影像 [B, 3, H, W]
            swir_images: SWIR影像 [B, 1, H, W]
            nir_images: NIR影像 [B, 1, H, W]
            targets: 训练时的标注信息
        Returns:
            训练时返回损失字典，推理时返回预测结果
        """
        original_image_sizes = []
        for img in rgb_images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        
        # 使用transform处理RGB图像
        rgb_images_list = [img for img in rgb_images]
        images, targets = self.transform(rgb_images_list, targets)
        
        # 提取特征
        features = self.backbone_with_fpn(images.tensors, swir_images)
        
        # RPN
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        # RoI Heads
        # 注意：暂时简化实现，不传递NIR图像
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        
        # 后处理
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        else:
            return detections


def build_enhanced_mask_rcnn(num_classes: int,
                             anchor_config_path: str = None,
                             backbone_pretrained: bool = True,
                             fusion_method: str = 'add',
                             use_cross_attention: bool = True):
    """
    构建增强版Mask R-CNN模型
    
    Args:
        num_classes: 类别数（包括背景）
        anchor_config_path: KMeans生成的锚框配置文件路径
        backbone_pretrained: 是否使用预训练骨干网络
        fusion_method: 特征融合方式
        use_cross_attention: 是否使用交叉注意力
    """
    # 加载锚框配置
    anchor_sizes = None
    anchor_aspect_ratios = None
    
    if anchor_config_path:
        import json
        with open(anchor_config_path, 'r') as f:
            anchor_config = json.load(f)
        
        # 从KMeans结果生成锚框配置
        anchor_wh = anchor_config['anchor_sizes']
        
        # 将锚框按尺度分组（匹配FPN的4个特征层）
        num_scales = 4
        anchors_per_scale = len(anchor_wh) // num_scales
        
        anchor_sizes = []
        anchor_aspect_ratios = []
        
        for i in range(num_scales):
            start_idx = i * anchors_per_scale
            end_idx = start_idx + anchors_per_scale if i < num_scales - 1 else len(anchor_wh)
            
            scale_anchors = anchor_wh[start_idx:end_idx]
            
            # 计算平均尺度
            avg_size = sum([w * h for w, h in scale_anchors]) / len(scale_anchors)
            avg_size = int(avg_size ** 0.5)
            
            anchor_sizes.append((avg_size,))
            
            # 计算宽高比
            ratios = [w / h for w, h in scale_anchors]
            anchor_aspect_ratios.append(tuple(ratios))
        
        print(f"从配置文件加载锚框: {anchor_sizes}")
    
    # 构建模型
    model = EnhancedMaskRCNN(
        num_classes=num_classes,
        anchor_sizes=anchor_sizes,
        anchor_aspect_ratios=anchor_aspect_ratios,
        backbone_pretrained=backbone_pretrained,
        fusion_method=fusion_method,
        use_cross_attention=use_cross_attention
    )
    
    return model


def test_enhanced_mask_rcnn():
    """测试增强版Mask R-CNN"""
    print("测试增强版Mask R-CNN...")
    
    # 创建模型
    model = build_enhanced_mask_rcnn(
        num_classes=81,  # 80类 + 背景
        backbone_pretrained=False,
        use_cross_attention=True
    )
    
    # 测试输入
    batch_size = 2
    rgb_images = torch.randn(batch_size, 3, 800, 800)
    swir_images = torch.randn(batch_size, 1, 800, 800)  # SWIR单通道
    nir_images = torch.randn(batch_size, 1, 800, 800)  # NIR单通道
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(rgb_images, swir_images, nir_images)
    
    print(f"\n输出数量: {len(outputs)}")
    print(f"第一个样本的检测框数量: {len(outputs[0]['boxes'])}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n测试通过！")


if __name__ == '__main__':
    test_enhanced_mask_rcnn()
