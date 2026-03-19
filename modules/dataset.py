"""
多模态遥感影像数据集加载器
支持RGB、SWIR、NIR三种影像的同步加载
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T


class MultiModalRemoteSensingDataset(Dataset):
    """多模态遥感影像数据集"""
    
    def __init__(self,
                 root_dir: str,
                 annotation_file: str,
                 split: str = 'train',
                 transforms=None,
                 train_ratio: float = 0.8):
        """
        Args:
            root_dir: 数据集根目录
            annotation_file: COCO格式标注文件路径
            split: 'train' 或 'val'
            transforms: 数据增强
            train_ratio: 训练集比例
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        # 影像目录
        self.rgb_dir = os.path.join(root_dir, 'images')
        self.swir_dir = os.path.join(root_dir, 'swir_images')
        self.nir_dir = os.path.join(root_dir, 'nir_images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # 加载标注
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # 划分训练集和验证集
        total_images = len(self.coco_data['images'])
        train_size = int(total_images * train_ratio)
        
        if split == 'train':
            self.image_list = self.coco_data['images'][:train_size]
        else:
            self.image_list = self.coco_data['images'][train_size:]
        
        # 构建图像ID到标注的映射
        self.img_to_anns = self._build_img_to_anns()
        
        # 类别信息
        self.categories = self.coco_data['categories']
        self.num_classes = len(self.categories)
        
        print(f"加载 {split} 集: {len(self.image_list)} 张影像")
        
    def _build_img_to_anns(self) -> Dict:
        """构建图像ID到标注的映射"""
        img_to_anns = {}
        valid_img_ids = {img['id'] for img in self.image_list}
        
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in valid_img_ids:
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)
        
        return img_to_anns
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def _load_image(self, path: str, mode: str = 'RGB') -> Image.Image:
        """
        加载影像
        
        Args:
            path: 影像路径
            mode: 'RGB' 或 'L' (灰度图)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"影像文件不存在: {path}")
        return Image.open(path).convert(mode)
    
    def _process_annotations(self, anns: List[Dict], img_width: int, img_height: int) -> Dict:
        """处理标注信息"""
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # 边界框 [x, y, width, height] -> [x1, y1, x2, y2]
            bbox = ann['bbox']
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            # 边界检查
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # 过滤无效框
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
            
            # 处理掩码（简化版本：使用边界框生成掩码）
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            mask[int(y1):int(y2), int(x1):int(x2)] = 1
            masks.append(mask)
        
        # 转换为tensor
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(np.array(masks), dtype=torch.uint8),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        
        return target
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Returns:
            images: 字典 {'rgb': tensor, 'swir': tensor, 'nir': tensor}
            target: 标注信息字典
        """
        # 获取图像信息
        img_info = self.image_list[idx]
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 加载三种模态影像
        rgb_path = os.path.join(self.rgb_dir, file_name)
        swir_path = os.path.join(self.swir_dir, file_name)
        nir_path = os.path.join(self.nir_dir, file_name)
        
        rgb_img = self._load_image(rgb_path, mode='RGB')  # RGB三通道
        swir_img = self._load_image(swir_path, mode='L')  # SWIR单通道灰度图
        nir_img = self._load_image(nir_path, mode='L')  # NIR单通道灰度图
        
        # 获取标注
        anns = self.img_to_anns.get(img_id, [])
        target = self._process_annotations(anns, img_width, img_height)
        target['image_id'] = torch.tensor([img_id])
        
        # 转换为tensor
        if self.transforms is not None:
            # RGB使用完整的transforms
            rgb_img, target = self.transforms(rgb_img, target)
            
            # SWIR和NIR只使用ToTensor和翻转，不使用RGB归一化
            swir_transform = get_single_channel_transforms(train=(self.split == 'train'))
            nir_transform = get_single_channel_transforms(train=(self.split == 'train'))
            
            swir_img, _ = swir_transform(swir_img, target.copy())
            nir_img, _ = nir_transform(nir_img, target.copy())
        else:
            # 默认转换
            to_tensor = T.ToTensor()
            rgb_img = to_tensor(rgb_img)
            swir_img = to_tensor(swir_img)
            nir_img = to_tensor(nir_img)
        
        # 组合多模态影像
        images = {
            'rgb': rgb_img,
            'swir': swir_img,
            'nir': nir_img
        }
        
        return images, target
    
    def get_class_names(self) -> List[str]:
        """获取类别名称"""
        return [cat['name'] for cat in self.categories]


class Compose:
    """组合多个数据增强操作"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """转换为Tensor"""
    
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target


class Normalize:
    """归一化"""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = T.functional.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = T.functional.hflip(image)
            
            if 'boxes' in target:
                boxes = target['boxes']
                width = image.shape[-1]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes
            
            if 'masks' in target:
                target['masks'] = target['masks'].flip(-1)
        
        return image, target


def get_transforms(train: bool = True):
    """获取RGB数据增强pipeline"""
    transforms_list = []
    
    transforms_list.append(ToTensor())
    
    if train:
        transforms_list.append(RandomHorizontalFlip(0.5))
    
    # ImageNet归一化（仅用于RGB）
    transforms_list.append(Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return Compose(transforms_list)


def get_single_channel_transforms(train: bool = True):
    """获取单通道图像（SWIR/NIR）的数据增强pipeline"""
    transforms_list = []
    
    transforms_list.append(ToTensor())
    
    if train:
        transforms_list.append(RandomHorizontalFlip(0.5))
    
    # 单通道归一化（使用简单的0-1范围，或者可以根据实际数据统计调整）
    # 不使用ImageNet的三通道归一化
    
    return Compose(transforms_list)


def collate_fn(batch):
    """自定义batch整理函数"""
    images_rgb = []
    images_swir = []
    images_nir = []
    targets = []
    
    for images, target in batch:
        images_rgb.append(images['rgb'])
        images_swir.append(images['swir'])
        images_nir.append(images['nir'])
        targets.append(target)
    
    # 堆叠图像
    images_rgb = torch.stack(images_rgb, dim=0)
    images_swir = torch.stack(images_swir, dim=0)
    images_nir = torch.stack(images_nir, dim=0)
    
    images_batch = {
        'rgb': images_rgb,
        'swir': images_swir,
        'nir': images_nir
    }
    
    return images_batch, targets


def test_dataset():
    """测试数据集加载"""
    print("测试数据集加载...")
    
    # 创建数据集
    dataset = MultiModalRemoteSensingDataset(
        root_dir='./instance_segmentation_dataset',
        annotation_file='./instance_segmentation_dataset/annotations/instances.json',
        split='train',
        transforms=get_transforms(train=True),
        train_ratio=0.8
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"类别数量: {dataset.num_classes}")
    print(f"类别名称: {dataset.get_class_names()}")
    
    # 测试加载一个样本
    images, target = dataset[0]
    
    print(f"\nRGB影像: {images['rgb'].shape}")
    print(f"SWIR影像: {images['swir'].shape}")
    print(f"NIR影像: {images['nir'].shape}")
    print(f"边界框数量: {len(target['boxes'])}")
    print(f"标签: {target['labels']}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print("\n测试DataLoader...")
    for images_batch, targets_batch in dataloader:
        print(f"Batch RGB: {images_batch['rgb'].shape}")
        print(f"Batch SWIR: {images_batch['swir'].shape}")
        print(f"Batch NIR: {images_batch['nir'].shape}")
        print(f"Batch targets: {len(targets_batch)}")
        break
    
    print("\n测试通过！")


if __name__ == '__main__':
    test_dataset()
