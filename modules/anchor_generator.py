"""
KMeans锚框生成模块 - 针对FPN不同层级分别聚类
使用IOU距离和二坐标系聚类
"""
import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict


def iou_distance(boxes, clusters):
    """
    计算边界框与聚类中心之间的IOU距离
    
    Args:
        boxes: [N, 2] 边界框宽高数组
        clusters: [K, 2] 聚类中心宽高数组
    Returns:
        distances: [N, K] 距离矩阵 (1 - IOU)
    """
    N = boxes.shape[0]
    K = clusters.shape[0]
    distances = np.zeros((N, K))
    
    for i in range(N):
        for j in range(K):
            # 计算交集面积 (假设边界框中心对齐)
            w_inter = np.minimum(boxes[i, 0], clusters[j, 0])
            h_inter = np.minimum(boxes[i, 1], clusters[j, 1])
            inter_area = w_inter * h_inter
            
            # 计算并集面积
            box_area = boxes[i, 0] * boxes[i, 1]
            cluster_area = clusters[j, 0] * clusters[j, 1]
            union_area = box_area + cluster_area - inter_area
            
            # IOU距离 = 1 - IOU
            iou = inter_area / (union_area + 1e-6)
            distances[i, j] = 1 - iou
    
    return distances


class IOUKMeans:
    """基于IOU距离的KMeans聚类"""
    
    def __init__(self, n_clusters: int, max_iter: int = 300, random_state: int = 42):
        """
        Args:
            n_clusters: 聚类数量
            max_iter: 最大迭代次数
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        
    def fit(self, boxes: np.ndarray):
        """
        使用IOU距离进行KMeans聚类
        
        Args:
            boxes: [N, 2] 边界框宽高数组
        """
        np.random.seed(self.random_state)
        N = boxes.shape[0]
        
        # 随机初始化聚类中心
        indices = np.random.choice(N, self.n_clusters, replace=False)
        self.cluster_centers_ = boxes[indices].copy()
        
        for iteration in range(self.max_iter):
            # 计算每个边界框到各聚类中心的IOU距离
            distances = iou_distance(boxes, self.cluster_centers_)
            
            # 分配到最近的聚类中心
            labels = np.argmin(distances, axis=1)
            
            # 更新聚类中心
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                cluster_boxes = boxes[labels == k]
                if len(cluster_boxes) > 0:
                    # 使用中位数作为新的聚类中心
                    new_centers[k] = np.median(cluster_boxes, axis=0)
                else:
                    # 如果某个聚类为空，保持原中心
                    new_centers[k] = self.cluster_centers_[k]
            
            # 检查收敛
            if np.allclose(self.cluster_centers_, new_centers):
                print(f"  IOU-KMeans在第{iteration+1}次迭代后收敛")
                break
            
            self.cluster_centers_ = new_centers
        
        return self


class AnchorGenerator:
    """基于IOU-KMeans聚类生成FPN层级自适应锚框"""
    
    def __init__(self, annotation_path: str, 
                 fpn_levels: int = 4,
                 anchors_per_level: int = 3):
        """
        Args:
            annotation_path: COCO格式标注文件路径
            fpn_levels: FPN层级数量，默认4层
            anchors_per_level: 每层锚框数量，默认3个
        """
        self.annotation_path = annotation_path
        self.fpn_levels = fpn_levels
        self.anchors_per_level = anchors_per_level
        self.level_anchors = {}  # 存储每层的锚框
        self.fpn_strides = [8, 16, 32, 64]  # FPN各层的步长
        
    def load_annotations(self) -> np.ndarray:
        """加载标注文件并提取所有边界框的宽高"""
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)
        
        boxes = []
        for ann in data['annotations']:
            bbox = ann['bbox']  # [x, y, width, height]
            width, height = bbox[2], bbox[3]
            if width > 0 and height > 0:
                boxes.append([width, height])
        
        boxes_array = np.array(boxes)
        print(f"加载了 {len(boxes)} 个有效边界框")
        print(f"  尺寸范围: 宽度[{boxes_array[:, 0].min():.1f}, {boxes_array[:, 0].max():.1f}], "
              f"高度[{boxes_array[:, 1].min():.1f}, {boxes_array[:, 1].max():.1f}]")
        return boxes_array
    
    def assign_boxes_to_levels(self, boxes: np.ndarray) -> Dict[int, np.ndarray]:
        """
        根据边界框尺寸将其分配到不同的FPN层级
        使用目标尺寸的平方根来决定最适合的FPN层级
        
        Args:
            boxes: [N, 2] 边界框宽高数组
        Returns:
            level_boxes: 字典，键为层级索引，值为该层级的边界框
        """
        # 计算每个边界框的尺度 (使用面积的平方根)
        scales = np.sqrt(boxes[:, 0] * boxes[:, 1])
        
        level_boxes = {i: [] for i in range(self.fpn_levels)}
        
        # 根据尺度分配到不同层级
        # FPN层级对应的尺度范围 (基于stride)
        for i, box in enumerate(boxes):
            scale = scales[i]
            
            # 根据尺度分配层级
            # P3(stride=8): 小目标 (scale < 64)
            # P4(stride=16): 中小目标 (64 <= scale < 128)
            # P5(stride=32): 中大目标 (128 <= scale < 256)
            # P6(stride=64): 大目标 (scale >= 256)
            if scale < 64:
                level = 0
            elif scale < 128:
                level = 1
            elif scale < 256:
                level = 2
            else:
                level = 3
            
            level_boxes[level].append(box)
        
        # 转换为numpy数组
        for level in range(self.fpn_levels):
            if len(level_boxes[level]) > 0:
                level_boxes[level] = np.array(level_boxes[level])
                print(f"  FPN层级 P{level+3} (stride={self.fpn_strides[level]}): "
                      f"{len(level_boxes[level])} 个边界框")
            else:
                # 如果某层没有分配到边界框，使用全部边界框的子集
                print(f"  警告: FPN层级 P{level+3} 没有分配到边界框，使用全局采样")
                level_boxes[level] = boxes[::max(1, len(boxes)//10)]  # 采样10%
        
        return level_boxes
    
    def compute_anchors_per_level(self, level_boxes: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        为每个FPN层级使用IOU-KMeans计算最优锚框
        
        Args:
            level_boxes: 每个层级的边界框字典
        Returns:
            level_anchors: 每个层级的锚框字典
        """
        level_anchors = {}
        
        for level in range(self.fpn_levels):
            print(f"\n为FPN层级 P{level+3} (stride={self.fpn_strides[level]}) 聚类锚框...")
            
            boxes = level_boxes[level]
            
            # 使用IOU-KMeans聚类
            kmeans = IOUKMeans(
                n_clusters=self.anchors_per_level,
                max_iter=300,
                random_state=42
            )
            kmeans.fit(boxes)
            
            # 获取聚类中心并按面积排序
            anchors = kmeans.cluster_centers_
            areas = anchors[:, 0] * anchors[:, 1]
            sorted_indices = np.argsort(areas)
            anchors = anchors[sorted_indices]
            
            level_anchors[level] = anchors
            
            # 打印该层级的锚框
            print(f"  生成的锚框:")
            for i, (w, h) in enumerate(anchors):
                aspect_ratio = w / h
                area = w * h
                print(f"    锚框 {i+1}: ({w:.2f}, {h:.2f}) | "
                      f"宽高比: {aspect_ratio:.3f} | 面积: {area:.2f}")
        
        self.level_anchors = level_anchors
        return level_anchors
    
    def visualize_anchors(self, all_boxes: np.ndarray, level_boxes: Dict[int, np.ndarray], 
                         save_path: str = None):
        """可视化FPN各层级的锚框分布"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'purple']
        
        for level in range(self.fpn_levels):
            ax = axes[level]
            
            # 绘制该层级的ground truth boxes
            boxes = level_boxes[level]
            ax.scatter(boxes[:, 0], boxes[:, 1], alpha=0.3, s=1, 
                      c=colors[level], label='Ground Truth Boxes')
            
            # 绘制该层级的锚框
            anchors = self.level_anchors[level]
            ax.scatter(anchors[:, 0], anchors[:, 1], c='red', s=200, marker='x', 
                      linewidths=3, label='Anchor Centers')
            
            # 绘制锚框矩形
            for i, (w, h) in enumerate(anchors):
                rect = plt.Rectangle((0, 0), w, h, fill=False, 
                                    edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(w, h, f'{i+1}', fontsize=10, color='red', fontweight='bold')
            
            ax.set_xlabel('Width', fontsize=12)
            ax.set_ylabel('Height', fontsize=12)
            ax.set_title(f'FPN Level P{level+3} (stride={self.fpn_strides[level]})', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n可视化结果已保存到: {save_path}")
        plt.close()
    
    def save_anchors(self, save_path: str):
        """保存FPN层级锚框配置"""
        if not self.level_anchors:
            raise ValueError("请先运行compute_anchors_per_level()计算锚框")
        
        # 构建配置字典
        anchor_config = {
            'fpn_levels': self.fpn_levels,
            'anchors_per_level': self.anchors_per_level,
            'fpn_strides': self.fpn_strides,
            'level_anchors': {}
        }
        
        # 为每个FPN层级保存锚框配置
        for level in range(self.fpn_levels):
            anchors = self.level_anchors[level]
            anchor_config['level_anchors'][f'P{level+3}'] = {
                'stride': self.fpn_strides[level],
                'anchor_sizes': anchors.tolist(),
                'aspect_ratios': (anchors[:, 0] / anchors[:, 1]).tolist(),
                'num_anchors': len(anchors)
            }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(anchor_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n锚框配置已保存到: {save_path}")
        return anchor_config
    
    def run(self, output_dir: str = './outputs'):
        """完整运行流程 - FPN层级自适应锚框生成"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("开始生成FPN层级自适应锚框 (使用IOU距离和二坐标系聚类)...")
        print("=" * 80)
        
        # 1. 加载数据
        print("\n步骤1: 加载标注数据")
        boxes = self.load_annotations()
        
        # 2. 将边界框分配到不同FPN层级
        print("\n步骤2: 将边界框分配到FPN层级")
        level_boxes = self.assign_boxes_to_levels(boxes)
        
        # 3. 为每个层级计算锚框
        print("\n步骤3: 为每个FPN层级使用IOU-KMeans聚类锚框")
        level_anchors = self.compute_anchors_per_level(level_boxes)
        
        # 4. 打印汇总结果
        print("\n" + "=" * 80)
        print("生成的FPN层级锚框汇总:")
        print("=" * 80)
        for level in range(self.fpn_levels):
            anchors = level_anchors[level]
            print(f"\nFPN层级 P{level+3} (stride={self.fpn_strides[level]}):")
            print("-" * 60)
            for i, (w, h) in enumerate(anchors):
                aspect_ratio = w / h
                area = w * h
                print(f"  锚框 {i+1}: ({w:.2f}, {h:.2f}) | "
                      f"宽高比: {aspect_ratio:.3f} | 面积: {area:.2f}")
        
        # 5. 可视化
        print("\n步骤4: 生成可视化")
        viz_path = output_path / 'anchor_visualization_fpn.png'
        self.visualize_anchors(boxes, level_boxes, str(viz_path))
        
        # 6. 保存配置
        print("\n步骤5: 保存配置文件")
        config_path = output_path / 'anchor_config.json'
        config = self.save_anchors(str(config_path))
        
        print("\n" + "=" * 80)
        print("FPN层级自适应锚框生成完成！")
        print("=" * 80)
        
        return config


def main():
    """主函数：独立运行此模块生成FPN层级自适应锚框"""
    # 配置参数
    annotation_path = './instance_segmentation_dataset/annotations/instances.json'
    output_dir = './outputs/anchors'
    fpn_levels = 4  # FPN层级数量 (P3, P4, P5, P6)
    anchors_per_level = 3  # 每层3个锚框
    
    # 创建生成器并运行
    generator = AnchorGenerator(
        annotation_path=annotation_path,
        fpn_levels=fpn_levels,
        anchors_per_level=anchors_per_level
    )
    config = generator.run(output_dir)
    
    print("\n生成的FPN层级锚框配置:")
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
