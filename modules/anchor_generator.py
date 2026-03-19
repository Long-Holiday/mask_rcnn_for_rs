"""
KMeans锚框生成模块
独立运行此模块以计算最优锚框大小
"""
import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple


class AnchorGenerator:
    """基于KMeans聚类生成最优锚框"""
    
    def __init__(self, annotation_path: str, n_clusters: int = 9):
        """
        Args:
            annotation_path: COCO格式标注文件路径
            n_clusters: 聚类数量，默认9个锚框
        """
        self.annotation_path = annotation_path
        self.n_clusters = n_clusters
        self.anchors = None
        
    def load_annotations(self) -> List[Tuple[float, float]]:
        """加载标注文件并提取所有边界框的宽高"""
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)
        
        boxes = []
        for ann in data['annotations']:
            bbox = ann['bbox']  # [x, y, width, height]
            width, height = bbox[2], bbox[3]
            if width > 0 and height > 0:
                boxes.append((width, height))
        
        print(f"加载了 {len(boxes)} 个有效边界框")
        return boxes
    
    def compute_anchors(self, boxes: List[Tuple[float, float]]) -> np.ndarray:
        """使用KMeans计算最优锚框大小"""
        boxes_array = np.array(boxes)
        
        # 归一化到0-1范围
        max_w = boxes_array[:, 0].max()
        max_h = boxes_array[:, 1].max()
        boxes_normalized = boxes_array / np.array([max_w, max_h])
        
        # KMeans聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(boxes_normalized)
        
        # 反归一化
        anchors = kmeans.cluster_centers_ * np.array([max_w, max_h])
        
        # 按面积排序
        areas = anchors[:, 0] * anchors[:, 1]
        sorted_indices = np.argsort(areas)
        anchors = anchors[sorted_indices]
        
        self.anchors = anchors
        return anchors
    
    def visualize_anchors(self, boxes: List[Tuple[float, float]], save_path: str = None):
        """可视化锚框分布"""
        boxes_array = np.array(boxes)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(boxes_array[:, 0], boxes_array[:, 1], alpha=0.3, s=1, label='Ground Truth Boxes')
        plt.scatter(self.anchors[:, 0], self.anchors[:, 1], c='red', s=100, marker='x', 
                   linewidths=3, label='Anchor Centers')
        
        # 绘制锚框矩形
        for i, (w, h) in enumerate(self.anchors):
            rect = plt.Rectangle((0, 0), w, h, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(w, h, f'{i+1}', fontsize=12, color='red')
        
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('KMeans Anchor Boxes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        plt.close()
    
    def save_anchors(self, save_path: str):
        """保存锚框配置"""
        if self.anchors is None:
            raise ValueError("请先运行compute_anchors()计算锚框")
        
        anchor_config = {
            'num_anchors': self.n_clusters,
            'anchor_sizes': self.anchors.tolist(),
            'aspect_ratios': (self.anchors[:, 0] / self.anchors[:, 1]).tolist()
        }
        
        with open(save_path, 'w') as f:
            json.dump(anchor_config, f, indent=2)
        
        print(f"锚框配置已保存到: {save_path}")
        return anchor_config
    
    def run(self, output_dir: str = './outputs'):
        """完整运行流程"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("开始生成最优锚框...")
        print("=" * 60)
        
        # 加载数据
        boxes = self.load_annotations()
        
        # 计算锚框
        anchors = self.compute_anchors(boxes)
        
        # 打印结果
        print("\n生成的锚框大小 (width, height):")
        print("-" * 60)
        for i, (w, h) in enumerate(anchors):
            aspect_ratio = w / h
            area = w * h
            print(f"Anchor {i+1}: ({w:.2f}, {h:.2f}) | 宽高比: {aspect_ratio:.3f} | 面积: {area:.2f}")
        
        # 可视化
        viz_path = output_path / 'anchor_visualization.png'
        self.visualize_anchors(boxes, str(viz_path))
        
        # 保存配置
        config_path = output_path / 'anchor_config.json'
        config = self.save_anchors(str(config_path))
        
        print("\n" + "=" * 60)
        print("锚框生成完成！")
        print("=" * 60)
        
        return config


def main():
    """主函数：独立运行此模块生成锚框"""
    # 配置参数
    annotation_path = './instance_segmentation_dataset/annotations/instances.json'
    output_dir = './outputs/anchors'
    n_clusters = 9  # 生成9个锚框（3个尺度 × 3个宽高比）
    
    # 创建生成器并运行
    generator = AnchorGenerator(annotation_path, n_clusters)
    config = generator.run(output_dir)
    
    print("\n生成的锚框配置:")
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
