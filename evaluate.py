"""
评估脚本
"""
import os
import torch
from torch.utils.data import DataLoader
import argparse
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path

from modules.dataset import MultiModalRemoteSensingDataset, get_transforms, collate_fn
from models.enhanced_mask_rcnn import build_enhanced_mask_rcnn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Evaluator:
    """评估器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 数据集
        self.test_dataset = MultiModalRemoteSensingDataset(
            root_dir=config['data_root'],
            annotation_file=config['annotation_file'],
            split='val',
            transforms=get_transforms(train=False),
            train_ratio=config['train_ratio']
        )
        
        # DataLoader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,  # 评估时batch_size=1
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn
        )
        
        # 模型
        self.model = build_enhanced_mask_rcnn(
            num_classes=config['num_classes'],
            anchor_config_path=config.get('anchor_config_path'),
            backbone_pretrained=False,
            fusion_method=config.get('fusion_method', 'adaptive'),
            use_cross_attention=config.get('use_cross_attention', True)
        )
        self.model.to(self.device)
        
        # 加载权重
        self.load_checkpoint(config['checkpoint_path'])
        
        print(f"\n评估配置:")
        print(f"  设备: {self.device}")
        print(f"  测试集大小: {len(self.test_dataset)}")
        print(f"  检查点: {config['checkpoint_path']}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("权重加载完成")
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        
        results = []
        
        print("\n开始评估...")
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        for images, targets in pbar:
            # 移动到设备
            rgb_images = images['rgb'].to(self.device)
            swir_images = images['swir'].to(self.device)
            nir_images = images['nir'].to(self.device)
            
            # 推理（三模态并行：RGB + NIR + SWIR）
            outputs = self.model(rgb_images, nir_images, swir_images)
            
            # 收集结果
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks = output['masks'].cpu().numpy()
                
                for j in range(len(boxes)):
                    result = {
                        'image_id': image_id,
                        'category_id': int(labels[j]),
                        'bbox': boxes[j].tolist(),
                        'score': float(scores[j]),
                        'segmentation': self._mask_to_rle(masks[j, 0])
                    }
                    results.append(result)
        
        return results
    
    def _mask_to_rle(self, mask):
        """将掩码转换为RLE格式（简化版本）"""
        # 这里返回简化的格式，实际应用中需要使用pycocotools的encode
        return {'size': mask.shape, 'counts': []}
    
    def compute_metrics(self, results):
        """计算评估指标"""
        if len(results) == 0:
            print("没有检测结果")
            return
        
        # 保存结果
        results_file = Path(self.config['output_dir']) / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        print(f"\n检测结果已保存到: {results_file}")
        print(f"总检测数量: {len(results)}")
        
        # 计算基本统计
        scores = [r['score'] for r in results]
        print(f"\n置信度统计:")
        print(f"  平均: {np.mean(scores):.4f}")
        print(f"  最大: {np.max(scores):.4f}")
        print(f"  最小: {np.min(scores):.4f}")
        
        # 如果有COCO API，可以计算mAP等指标
        try:
            coco_gt = COCO(self.config['annotation_file'])
            coco_dt = coco_gt.loadRes(results_file)
            
            # 边界框评估
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # 分割评估
            coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        except Exception as e:
            print(f"\n无法计算COCO指标: {e}")


def main():
    parser = argparse.ArgumentParser(description='评估增强版Mask R-CNN')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_root', type=str, default='./instance_segmentation_dataset', help='数据集根目录')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            'data_root': args.data_root,
            'annotation_file': os.path.join(args.data_root, 'annotations/instances.json'),
            'train_ratio': 0.8,
            'num_classes': 81,
            'anchor_config_path': './outputs/anchors/anchor_config.json',
            'fusion_method': 'adaptive',
            'use_cross_attention': True,
            'num_workers': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'output_dir': args.output_dir
        }
    
    config['checkpoint_path'] = args.checkpoint
    
    # 创建输出目录
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # 评估
    evaluator = Evaluator(config)
    results = evaluator.evaluate()
    evaluator.compute_metrics(results)


if __name__ == '__main__':
    main()
