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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import time
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2

from modules.dataset import MultiModalRemoteSensingDataset, get_transforms, collate_fn
from models.enhanced_mask_rcnn import build_enhanced_mask_rcnn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


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
        inference_times = []
        
        print("\n开始评估...")
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        for images, targets in pbar:
            # 移动到设备
            rgb_images = images['rgb'].to(self.device)
            swir_images = images['swir'].to(self.device)
            nir_images = images['nir'].to(self.device)
            
            # 记录推理时间
            start_time = time.time()
            
            # 推理（三模态并行：RGB + NIR + SWIR）
            outputs = self.model(rgb_images, nir_images, swir_images)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
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
        
        # 保存推理时间统计
        self.inference_times = inference_times
        
        return results
    
    def _mask_to_rle(self, mask):
        """将掩码转换为RLE格式"""
        # 将掩码转换为uint8格式
        binary_mask = (mask > 0.5).astype(np.uint8)
        # 使用pycocotools编码
        rle = maskUtils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    
    def compute_metrics(self, results):
        """计算评估指标"""
        if len(results) == 0:
            print("没有检测结果")
            return {}
        
        # 保存结果
        results_file = Path(self.config['output_dir']) / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        print(f"\n检测结果已保存到: {results_file}")
        print(f"总检测数量: {len(results)}")
        
        # 存储所有指标
        all_metrics = {}
        
        # 1. 基本统计指标
        basic_metrics = self._compute_basic_metrics(results)
        all_metrics.update(basic_metrics)
        
        # 2. 推理效率指标
        efficiency_metrics = self._compute_efficiency_metrics()
        all_metrics.update(efficiency_metrics)
        
        # 3. 类别分布统计
        category_metrics = self._compute_category_metrics(results)
        all_metrics.update(category_metrics)
        
        # 4. 尺度分析指标
        scale_metrics = self._compute_scale_metrics(results)
        all_metrics.update(scale_metrics)
        
        # 5. COCO官方指标
        try:
            coco_gt = COCO(self.config['annotation_file'])
            coco_dt = coco_gt.loadRes(results_file)
            
            print("\n" + "="*60)
            print("COCO 边界框评估指标:")
            print("="*60)
            
            # 边界框评估
            coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval_bbox.evaluate()
            coco_eval_bbox.accumulate()
            coco_eval_bbox.summarize()
            
            # 提取边界框指标
            bbox_metrics = self._extract_coco_metrics(coco_eval_bbox, 'bbox')
            all_metrics.update(bbox_metrics)
            
            print("\n" + "="*60)
            print("COCO 分割评估指标:")
            print("="*60)
            
            # 分割评估
            coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
            coco_eval_segm.evaluate()
            coco_eval_segm.accumulate()
            coco_eval_segm.summarize()
            
            # 提取分割指标
            segm_metrics = self._extract_coco_metrics(coco_eval_segm, 'segm')
            all_metrics.update(segm_metrics)
            
            # 6. 每类别详细指标
            per_class_metrics = self._compute_per_class_metrics(coco_gt, coco_eval_bbox, coco_eval_segm)
            all_metrics.update(per_class_metrics)
            
        except Exception as e:
            print(f"\n无法计算COCO指标: {e}")
        
        # 7. 生成可视化报告
        self._generate_visualization_report(all_metrics, results)
        
        # 8. 保存完整指标报告
        self._save_metrics_report(all_metrics)
        
        return all_metrics
    
    def _compute_basic_metrics(self, results):
        """计算基本统计指标"""
        scores = [r['score'] for r in results]
        bboxes = [r['bbox'] for r in results]
        
        # 边界框面积统计
        areas = []
        aspect_ratios = []
        for bbox in bboxes:
            x, y, w, h = bbox
            area = w * h
            areas.append(area)
            aspect_ratio = w / h if h > 0 else 0
            aspect_ratios.append(aspect_ratio)
        
        metrics = {
            'basic/total_detections': len(results),
            'basic/score_mean': np.mean(scores),
            'basic/score_std': np.std(scores),
            'basic/score_median': np.median(scores),
            'basic/score_min': np.min(scores),
            'basic/score_max': np.max(scores),
            'basic/area_mean': np.mean(areas),
            'basic/area_std': np.std(areas),
            'basic/area_median': np.median(areas),
            'basic/aspect_ratio_mean': np.mean(aspect_ratios),
            'basic/aspect_ratio_std': np.std(aspect_ratios)
        }
        
        print(f"\n基本统计指标:")
        print(f"  总检测数量: {metrics['basic/total_detections']}")
        print(f"  置信度 - 均值: {metrics['basic/score_mean']:.4f}, 标准差: {metrics['basic/score_std']:.4f}")
        print(f"  边界框面积 - 均值: {metrics['basic/area_mean']:.2f}, 标准差: {metrics['basic/area_std']:.2f}")
        print(f"  长宽比 - 均值: {metrics['basic/aspect_ratio_mean']:.2f}, 标准差: {metrics['basic/aspect_ratio_std']:.2f}")
        
        return metrics
    
    def _compute_efficiency_metrics(self):
        """计算推理效率指标"""
        if not hasattr(self, 'inference_times') or not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        fps = 1.0 / times
        
        metrics = {
            'efficiency/inference_time_mean': np.mean(times),
            'efficiency/inference_time_std': np.std(times),
            'efficiency/inference_time_median': np.median(times),
            'efficiency/fps_mean': np.mean(fps),
            'efficiency/fps_min': np.min(fps),
            'efficiency/fps_max': np.max(fps),
            'efficiency/total_images': len(times),
            'efficiency/total_time': np.sum(times)
        }
        
        print(f"\n推理效率指标:")
        print(f"  平均推理时间: {metrics['efficiency/inference_time_mean']:.4f}s")
        print(f"  平均FPS: {metrics['efficiency/fps_mean']:.2f}")
        print(f"  总处理图像数: {metrics['efficiency/total_images']}")
        print(f"  总推理时间: {metrics['efficiency/total_time']:.2f}s")
        
        return metrics
    
    def _compute_category_metrics(self, results):
        """计算类别分布统计"""
        categories = [r['category_id'] for r in results]
        category_counts = Counter(categories)
        
        metrics = {
            'category/num_categories': len(category_counts),
            'category/detections_per_category_mean': np.mean(list(category_counts.values())),
            'category/detections_per_category_std': np.std(list(category_counts.values())),
            'category/most_frequent_category': max(category_counts, key=category_counts.get) if category_counts else 0,
            'category/least_frequent_category': min(category_counts, key=category_counts.get) if category_counts else 0
        }
        
        # 添加每个类别的检测数量
        for cat_id, count in category_counts.items():
            metrics[f'category/count_class_{cat_id}'] = count
        
        print(f"\n类别分布统计:")
        print(f"  检测到的类别数: {metrics['category/num_categories']}")
        print(f"  每类别平均检测数: {metrics['category/detections_per_category_mean']:.2f}")
        print(f"  最频繁类别: {metrics['category/most_frequent_category']} ({category_counts.get(metrics['category/most_frequent_category'], 0)} 次)")
        
        return metrics
    
    def _compute_scale_metrics(self, results):
        """计算不同尺度的统计"""
        areas = []
        for r in results:
            bbox = r['bbox']
            area = bbox[2] * bbox[3]  # width * height
            areas.append(area)
        
        if not areas:
            return {}
        
        areas = np.array(areas)
        
        # COCO尺度定义
        small_mask = areas < 32**2
        medium_mask = (areas >= 32**2) & (areas < 96**2)
        large_mask = areas >= 96**2
        
        metrics = {
            'scale/small_objects': np.sum(small_mask),
            'scale/medium_objects': np.sum(medium_mask),
            'scale/large_objects': np.sum(large_mask),
            'scale/small_ratio': np.mean(small_mask),
            'scale/medium_ratio': np.mean(medium_mask),
            'scale/large_ratio': np.mean(large_mask)
        }
        
        print(f"\n尺度分布统计:")
        print(f"  小目标 (< 32²): {metrics['scale/small_objects']} ({metrics['scale/small_ratio']:.2%})")
        print(f"  中等目标 (32²-96²): {metrics['scale/medium_objects']} ({metrics['scale/medium_ratio']:.2%})")
        print(f"  大目标 (> 96²): {metrics['scale/large_objects']} ({metrics['scale/large_ratio']:.2%})")
        
        return metrics
    
    def _extract_coco_metrics(self, coco_eval, eval_type):
        """提取COCO评估指标"""
        stats = coco_eval.stats
        
        metrics = {
            f'{eval_type}/AP': stats[0],           # AP @ IoU=0.50:0.95
            f'{eval_type}/AP_50': stats[1],        # AP @ IoU=0.50
            f'{eval_type}/AP_75': stats[2],        # AP @ IoU=0.75
            f'{eval_type}/AP_small': stats[3],     # AP for small objects
            f'{eval_type}/AP_medium': stats[4],    # AP for medium objects
            f'{eval_type}/AP_large': stats[5],     # AP for large objects
            f'{eval_type}/AR_1': stats[6],         # AR @ maxDets=1
            f'{eval_type}/AR_10': stats[7],        # AR @ maxDets=10
            f'{eval_type}/AR_100': stats[8],       # AR @ maxDets=100
            f'{eval_type}/AR_small': stats[9],     # AR for small objects
            f'{eval_type}/AR_medium': stats[10],   # AR for medium objects
            f'{eval_type}/AR_large': stats[11]     # AR for large objects
        }
        
        return metrics
    
    def _compute_per_class_metrics(self, coco_gt, coco_eval_bbox, coco_eval_segm):
        """计算每个类别的详细指标"""
        metrics = {}
        
        # 获取类别信息
        cats = coco_gt.loadCats(coco_gt.getCatIds())
        
        for i, cat in enumerate(cats):
            cat_id = cat['id']
            cat_name = cat['name']
            
            # 边界框指标
            if hasattr(coco_eval_bbox, 'eval') and coco_eval_bbox.eval:
                precision_bbox = coco_eval_bbox.eval['precision'][:, :, i, 0, -1]
                valid_precision_bbox = precision_bbox[precision_bbox > -1]
                if len(valid_precision_bbox) > 0:
                    metrics[f'per_class_bbox/AP_class_{cat_id}_{cat_name}'] = np.mean(valid_precision_bbox)
            
            # 分割指标
            if hasattr(coco_eval_segm, 'eval') and coco_eval_segm.eval:
                precision_segm = coco_eval_segm.eval['precision'][:, :, i, 0, -1]
                valid_precision_segm = precision_segm[precision_segm > -1]
                if len(valid_precision_segm) > 0:
                    metrics[f'per_class_segm/AP_class_{cat_id}_{cat_name}'] = np.mean(valid_precision_segm)
        
        return metrics
    
    def _generate_visualization_report(self, metrics, results):
        """生成可视化报告"""
        try:
            # 创建可视化目录
            vis_dir = Path(self.config['output_dir']) / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            # 1. 置信度分布图
            self._plot_score_distribution(results, vis_dir)
            
            # 2. 类别分布图
            self._plot_category_distribution(results, vis_dir)
            
            # 3. 尺度分布图
            self._plot_scale_distribution(results, vis_dir)
            
            # 4. 长宽比分布图
            self._plot_aspect_ratio_distribution(results, vis_dir)
            
            # 5. 推理时间分布图
            if hasattr(self, 'inference_times'):
                self._plot_inference_time_distribution(vis_dir)
            
            print(f"\n可视化报告已保存到: {vis_dir}")
            
        except Exception as e:
            print(f"生成可视化报告时出错: {e}")
    
    def _plot_score_distribution(self, results, vis_dir):
        """绘制置信度分布图"""
        scores = [r['score'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('置信度')
        plt.ylabel('频次')
        plt.title('检测结果置信度分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(vis_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_distribution(self, results, vis_dir):
        """绘制类别分布图"""
        categories = [r['category_id'] for r in results]
        category_counts = Counter(categories)
        
        plt.figure(figsize=(12, 6))
        cats = list(category_counts.keys())
        counts = list(category_counts.values())
        
        plt.bar(range(len(cats)), counts)
        plt.xlabel('类别ID')
        plt.ylabel('检测数量')
        plt.title('各类别检测数量分布')
        plt.xticks(range(len(cats)), cats)
        plt.grid(True, alpha=0.3)
        plt.savefig(vis_dir / 'category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scale_distribution(self, results, vis_dir):
        """绘制尺度分布图"""
        areas = []
        for r in results:
            bbox = r['bbox']
            area = bbox[2] * bbox[3]
            areas.append(area)
        
        plt.figure(figsize=(10, 6))
        plt.hist(areas, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('边界框面积 (像素²)')
        plt.ylabel('频次')
        plt.title('目标尺度分布')
        plt.axvline(x=32**2, color='red', linestyle='--', label='小/中等边界 (32²)')
        plt.axvline(x=96**2, color='orange', linestyle='--', label='中等/大边界 (96²)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(vis_dir / 'scale_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_aspect_ratio_distribution(self, results, vis_dir):
        """绘制长宽比分布图"""
        aspect_ratios = []
        for r in results:
            bbox = r['bbox']
            w, h = bbox[2], bbox[3]
            if h > 0:
                aspect_ratios.append(w / h)
        
        plt.figure(figsize=(10, 6))
        plt.hist(aspect_ratios, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('长宽比 (宽/高)')
        plt.ylabel('频次')
        plt.title('目标长宽比分布')
        plt.axvline(x=1.0, color='red', linestyle='--', label='正方形 (1:1)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(vis_dir / 'aspect_ratio_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inference_time_distribution(self, vis_dir):
        """绘制推理时间分布图"""
        times = self.inference_times
        
        plt.figure(figsize=(10, 6))
        plt.hist(times, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('推理时间 (秒)')
        plt.ylabel('频次')
        plt.title('推理时间分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(vis_dir / 'inference_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_metrics_report(self, metrics):
        """保存完整的指标报告"""
        # 保存为JSON
        metrics_file = Path(self.config['output_dir']) / 'metrics_report.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存为可读的文本报告
        report_file = Path(self.config['output_dir']) / 'metrics_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("实例分割模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 按类别组织指标
            categories = ['basic', 'efficiency', 'category', 'scale', 'bbox', 'segm']
            
            for cat in categories:
                cat_metrics = {k: v for k, v in metrics.items() if k.startswith(f'{cat}/')}
                if cat_metrics:
                    f.write(f"{cat.upper()} 指标:\n")
                    f.write("-" * 30 + "\n")
                    for key, value in cat_metrics.items():
                        metric_name = key.split('/', 1)[1]
                        if isinstance(value, float):
                            f.write(f"  {metric_name}: {value:.4f}\n")
                        else:
                            f.write(f"  {metric_name}: {value}\n")
                    f.write("\n")
        
        print(f"\n完整指标报告已保存到:")
        print(f"  JSON格式: {metrics_file}")
        print(f"  文本格式: {report_file}")


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
