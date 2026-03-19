import json
import shutil
import os
from pathlib import Path
from collections import defaultdict
import random


def create_test_dataset(
    source_annotation_path='instance_segmentation_dataset/annotations/instances.json',
    source_images_dir='instance_segmentation_dataset/images',
    source_masks_dir='instance_segmentation_dataset/masks',
    source_nir_dir='instance_segmentation_dataset/nir_images',
    source_swir_dir='instance_segmentation_dataset/swir_images',
    output_dir='test_dataset',
    samples_per_class=5,
    seed=42
):
    """
    从数据集中每个类别挑选指定数量的样本创建测试数据集
    
    Args:
        source_annotation_path: 源标注文件路径
        source_images_dir: 源图像目录
        source_masks_dir: 源mask目录
        source_nir_dir: 源NIR图像目录
        source_swir_dir: 源SWIR图像目录
        output_dir: 输出目录
        samples_per_class: 每个类别挑选的样本数量
        seed: 随机种子
    """
    random.seed(seed)
    
    # 读取源标注文件
    print(f"读取标注文件: {source_annotation_path}")
    with open(source_annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    # 按类别组织annotations
    category_to_annotations = defaultdict(list)
    annotation_to_image = {}
    
    for ann in coco_data['annotations']:
        category_id = ann['category_id']
        category_to_annotations[category_id].append(ann)
        annotation_to_image[ann['id']] = ann['image_id']
    
    # 创建image_id到image信息的映射
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 打印类别信息
    print("\n数据集类别信息:")
    for cat in coco_data['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        count = len(category_to_annotations[cat_id])
        print(f"  {cat_name} (ID: {cat_id}): {count} 个样本")
    
    # 从每个类别中随机选择样本
    selected_annotations = []
    selected_image_ids = set()
    
    print(f"\n从每个类别中选择 {samples_per_class} 个样本:")
    for cat in coco_data['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        
        # 获取该类别的所有annotations
        cat_annotations = category_to_annotations[cat_id]
        
        # 随机选择
        if len(cat_annotations) < samples_per_class:
            print(f"  警告: {cat_name} 只有 {len(cat_annotations)} 个样本,少于要求的 {samples_per_class} 个")
            selected = cat_annotations
        else:
            selected = random.sample(cat_annotations, samples_per_class)
        
        selected_annotations.extend(selected)
        
        # 记录选中的图像ID
        for ann in selected:
            selected_image_ids.add(ann['image_id'])
        
        print(f"  {cat_name}: 选择了 {len(selected)} 个样本")
    
    # 获取选中的图像信息
    selected_images = [image_id_to_info[img_id] for img_id in selected_image_ids]
    
    print(f"\n总共选择了 {len(selected_images)} 张图像, {len(selected_annotations)} 个标注")
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    output_images_dir = output_path / 'images'
    output_masks_dir = output_path / 'masks'
    output_nir_dir = output_path / 'nir_images'
    output_swir_dir = output_path / 'swir_images'
    output_annotations_dir = output_path / 'annotations'
    
    for dir_path in [output_images_dir, output_masks_dir, output_nir_dir, 
                     output_swir_dir, output_annotations_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n创建输出目录: {output_dir}")
    
    # 复制图像和mask文件
    print("\n复制文件:")
    copied_count = 0
    for img_info in selected_images:
        file_name = img_info['file_name']
        base_name = Path(file_name).stem
        
        # 复制RGB图像
        src_img = Path(source_images_dir) / file_name
        dst_img = output_images_dir / file_name
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            copied_count += 1
        else:
            print(f"  警告: 找不到图像文件 {src_img}")
        
        # 复制mask (mask文件命名格式为mask_*.png)
        image_id = img_info['id']
        mask_name = f"mask_{image_id}.png"
        src_mask = Path(source_masks_dir) / mask_name
        dst_mask = output_masks_dir / mask_name
        if src_mask.exists():
            shutil.copy2(src_mask, dst_mask)
        else:
            print(f"  警告: 找不到mask文件 {src_mask}")
        
        # 复制NIR图像
        src_nir = Path(source_nir_dir) / file_name
        dst_nir = output_nir_dir / file_name
        if src_nir.exists():
            shutil.copy2(src_nir, dst_nir)
        
        # 复制SWIR图像
        src_swir = Path(source_swir_dir) / file_name
        dst_swir = output_swir_dir / file_name
        if src_swir.exists():
            shutil.copy2(src_swir, dst_swir)
    
    print(f"  复制了 {copied_count} 张图像及相关文件")
    
    # 创建新的标注文件
    new_coco_data = {
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': coco_data['categories']
    }
    
    output_annotation_path = output_annotations_dir / 'instances.json'
    with open(output_annotation_path, 'w') as f:
        json.dump(new_coco_data, f, indent=2)
    
    print(f"\n保存标注文件: {output_annotation_path}")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("测试数据集创建完成!")
    print("="*50)
    print(f"输出目录: {output_dir}")
    print(f"图像数量: {len(selected_images)}")
    print(f"标注数量: {len(selected_annotations)}")
    print("\n按类别统计:")
    for cat in coco_data['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        count = sum(1 for ann in selected_annotations if ann['category_id'] == cat_id)
        print(f"  {cat_name}: {count} 个样本")
    print("="*50)


if __name__ == '__main__':
    # 可以通过命令行参数自定义配置
    import argparse
    
    parser = argparse.ArgumentParser(description='从数据集中每个类别挑选样本创建测试数据集')
    parser.add_argument('--source_annotation', type=str, 
                       default='instance_segmentation_dataset/annotations/instances.json',
                       help='源标注文件路径')
    parser.add_argument('--source_images', type=str,
                       default='instance_segmentation_dataset/images',
                       help='源图像目录')
    parser.add_argument('--source_masks', type=str,
                       default='instance_segmentation_dataset/masks',
                       help='源mask目录')
    parser.add_argument('--source_nir', type=str,
                       default='instance_segmentation_dataset/nir_images',
                       help='源NIR图像目录')
    parser.add_argument('--source_swir', type=str,
                       default='instance_segmentation_dataset/swir_images',
                       help='源SWIR图像目录')
    parser.add_argument('--output_dir', type=str,
                       default='test_dataset',
                       help='输出目录')
    parser.add_argument('--samples_per_class', type=int,
                       default=5,
                       help='每个类别挑选的样本数量')
    parser.add_argument('--seed', type=int,
                       default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    create_test_dataset(
        source_annotation_path=args.source_annotation,
        source_images_dir=args.source_images,
        source_masks_dir=args.source_masks,
        source_nir_dir=args.source_nir,
        source_swir_dir=args.source_swir,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )
