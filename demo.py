"""
快速测试脚本 - 使用少量影像验证模型流程完整性
仅用于测试模型的前向传播、训练和推理流程是否正常工作
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import json

from modules.dataset import MultiModalRemoteSensingDataset, get_transforms, collate_fn
from models.enhanced_mask_rcnn import build_enhanced_mask_rcnn


def test_data_loading(data_root, annotation_file, num_samples=3):
    """测试数据加载"""
    print("\n" + "=" * 80)
    print("步骤1: 测试数据加载")
    print("=" * 80)
    
    # 创建数据集
    dataset = MultiModalRemoteSensingDataset(
        root_dir=data_root,
        annotation_file=annotation_file,
        split='train',
        transforms=get_transforms(train=True),
        train_ratio=0.8
    )
    
    print(f"数据集总大小: {len(dataset)}")
    print(f"类别数量: {dataset.num_classes}")
    print(f"类别名称: {dataset.get_class_names()}")
    
    # 仅使用前几个样本
    subset = Subset(dataset, range(min(num_samples, len(dataset))))
    print(f"\n使用样本数量: {len(subset)}")
    
    # 测试加载一个样本
    images, target = dataset[0]
    print(f"\n样本信息:")
    print(f"  RGB影像形状: {images['rgb'].shape}")
    print(f"  SWIR影像形状: {images['swir'].shape}")
    print(f"  NIR影像形状: {images['nir'].shape}")
    print(f"  边界框数量: {len(target['boxes'])}")
    print(f"  标签: {target['labels']}")
    
    # 创建DataLoader
    dataloader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"\n✓ 数据加载测试通过")
    return dataset, dataloader


def test_model_creation(num_classes):
    """测试模型创建"""
    print("\n" + "=" * 80)
    print("步骤2: 测试模型创建")
    print("=" * 80)
    
    # 创建模型
    model = build_enhanced_mask_rcnn(
        num_classes=num_classes,
        backbone_pretrained=False,  # 快速测试不使用预训练权重
        fusion_method='add',
        use_cross_attention=True
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  使用交叉注意力: True")
    print(f"  特征融合方式: add")
    
    # 测试骨干网络特征提取
    print(f"\n[调试] 测试骨干网络特征维度...")
    test_rgb = torch.randn(1, 3, 1024, 1024)
    test_swir = torch.randn(1, 1, 1024, 1024)
    
    model.eval()
    with torch.no_grad():
        # 测试RGB分支
        rgb_x = model.backbone_with_fpn.backbone.rgb_stem(test_rgb)
        print(f"  RGB Stem输出: {rgb_x.shape}")
        
        rgb_res2 = model.backbone_with_fpn.backbone.rgb_res2(rgb_x)
        print(f"  RGB Res2输出: {rgb_res2.shape}")
        
        rgb_res3 = model.backbone_with_fpn.backbone.rgb_res3(rgb_res2)
        print(f"  RGB Res3输出: {rgb_res3.shape}")
        
        rgb_res4 = model.backbone_with_fpn.backbone.rgb_res4(rgb_res3)
        print(f"  RGB Res4输出: {rgb_res4.shape}")
        
        rgb_res5 = model.backbone_with_fpn.backbone.rgb_res5(rgb_res4)
        print(f"  RGB Res5输出: {rgb_res5.shape}")
        
        # 测试SWIR分支
        swir_features = model.backbone_with_fpn.backbone.swir_branch(test_swir)
        print(f"\n  SWIR分支输出:")
        for name, feat in swir_features.items():
            print(f"    {name}: {feat.shape}")
        
        # 测试融合
        print(f"\n  特征融合测试:")
        fused_features = model.backbone_with_fpn.backbone(test_rgb, test_swir)
        for name, feat in fused_features.items():
            print(f"    {name}: {feat.shape}")
    
    print(f"\n✓ 模型创建测试通过")
    return model


def test_forward_pass(model, dataloader, device):
    """测试前向传播"""
    print("\n" + "=" * 80)
    print("步骤3: 测试前向传播（训练模式）")
    print("=" * 80)
    
    model.to(device)
    model.train()
    
    # 获取一个batch
    images_batch, targets_batch = next(iter(dataloader))
    
    # 移动到设备
    rgb_images = images_batch['rgb'].to(device)
    swir_images = images_batch['swir'].to(device)
    nir_images = images_batch['nir'].to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]
    
    print(f"输入形状:")
    print(f"  RGB: {rgb_images.shape}")
    print(f"  SWIR: {swir_images.shape}")
    print(f"  NIR: {nir_images.shape}")
    print(f"  Batch大小: {len(targets)}")
    
    # 前向传播
    try:
        print("\n[调试] 开始前向传播...")
        
        # 添加钩子函数来追踪特征尺寸
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    print(f"[调试] {name} 输出形状: {output.shape}")
                elif isinstance(output, dict):
                    print(f"[调试] {name} 输出字典:")
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: {v.shape}")
            return hook
        
        # 注册钩子
        hooks = []
        hooks.append(model.backbone_with_fpn.backbone.rgb_stem.register_forward_hook(hook_fn("RGB Stem")))
        hooks.append(model.backbone_with_fpn.backbone.rgb_res2.register_forward_hook(hook_fn("RGB Res2")))
        hooks.append(model.backbone_with_fpn.backbone.rgb_res3.register_forward_hook(hook_fn("RGB Res3")))
        hooks.append(model.backbone_with_fpn.backbone.rgb_res4.register_forward_hook(hook_fn("RGB Res4")))
        hooks.append(model.backbone_with_fpn.backbone.rgb_res5.register_forward_hook(hook_fn("RGB Res5")))
        hooks.append(model.backbone_with_fpn.backbone.swir_branch.register_forward_hook(hook_fn("SWIR Branch")))
        hooks.append(model.backbone_with_fpn.backbone.register_forward_hook(hook_fn("Backbone Fusion")))
        hooks.append(model.backbone_with_fpn.fpn.register_forward_hook(hook_fn("FPN")))
        
        loss_dict = model(rgb_images, swir_images, nir_images, targets)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        total_loss = sum(loss for loss in loss_dict.values())
        
        print(f"\n损失值:")
        for k, v in loss_dict.items():
            print(f"  {k}: {v.item():.4f}")
        print(f"  总损失: {total_loss.item():.4f}")
        
        print(f"\n✓ 前向传播测试通过")
        return True
    except Exception as e:
        print(f"\n✗ 前向传播失败: {str(e)}")
        import traceback
        print("\n完整错误堆栈:")
        traceback.print_exc()
        return False


def test_backward_pass(model, dataloader, device):
    """测试反向传播"""
    print("\n" + "=" * 80)
    print("步骤4: 测试反向传播")
    print("=" * 80)
    
    model.to(device)
    model.train()
    
    # 创建优化器
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # 获取一个batch
    images_batch, targets_batch = next(iter(dataloader))
    
    # 移动到设备
    rgb_images = images_batch['rgb'].to(device)
    swir_images = images_batch['swir'].to(device)
    nir_images = images_batch['nir'].to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]
    
    try:
        print("\n[调试] 开始反向传播测试...")
        
        # 前向传播
        loss_dict = model(rgb_images, swir_images, nir_images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        
        print(f"训练前损失: {total_loss.item():.4f}")
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        
        if has_grad:
            print(f"✓ 梯度计算成功")
        else:
            print(f"✗ 未检测到梯度")
            return False
        
        # 更新参数
        optimizer.step()
        print(f"✓ 参数更新成功")
        
        print(f"\n✓ 反向传播测试通过")
        return True
    except Exception as e:
        print(f"\n✗ 反向传播失败: {str(e)}")
        import traceback
        print("\n完整错误堆栈:")
        traceback.print_exc()
        return False


def test_inference(model, dataloader, device):
    """测试推理"""
    print("\n" + "=" * 80)
    print("步骤5: 测试推理（评估模式）")
    print("=" * 80)
    
    model.to(device)
    model.eval()
    
    # 获取一个batch
    images_batch, _ = next(iter(dataloader))
    
    # 移动到设备
    rgb_images = images_batch['rgb'].to(device)
    swir_images = images_batch['swir'].to(device)
    nir_images = images_batch['nir'].to(device)
    
    try:
        print("\n[调试] 开始推理测试...")
        
        with torch.no_grad():
            outputs = model(rgb_images, swir_images, nir_images)
        
        print(f"推理结果:")
        print(f"  输出数量: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            num_detections = len(output['boxes'])
            print(f"  样本{i+1}: 检测到 {num_detections} 个目标")
            
            if num_detections > 0:
                print(f"    - 边界框形状: {output['boxes'].shape}")
                print(f"    - 分数形状: {output['scores'].shape}")
                print(f"    - 标签形状: {output['labels'].shape}")
                if 'masks' in output:
                    print(f"    - 掩码形状: {output['masks'].shape}")
        
        print(f"\n✓ 推理测试通过")
        return True
    except Exception as e:
        print(f"\n✗ 推理失败: {str(e)}")
        import traceback
        print("\n完整错误堆栈:")
        traceback.print_exc()
        return False


def test_mini_training(model, dataloader, device, num_iterations=3):
    """测试迷你训练循环"""
    print("\n" + "=" * 80)
    print(f"步骤6: 测试迷你训练循环（{num_iterations}次迭代）")
    print("=" * 80)
    
    model.to(device)
    model.train()
    
    # 创建优化器
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    try:
        print("\n[调试] 开始迷你训练循环...")
        
        for iteration in range(num_iterations):
            # 获取一个batch
            images_batch, targets_batch = next(iter(dataloader))
            
            # 移动到设备
            rgb_images = images_batch['rgb'].to(device)
            swir_images = images_batch['swir'].to(device)
            nir_images = images_batch['nir'].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]
            
            print(f"\n[调试] 迭代 {iteration+1} - 输入形状: RGB={rgb_images.shape}, SWIR={swir_images.shape}, NIR={nir_images.shape}")
            
            # 前向传播
            loss_dict = model(rgb_images, swir_images, nir_images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            print(f"迭代 {iteration+1}/{num_iterations}: 损失 = {total_loss.item():.4f}")
        
        print(f"\n✓ 迷你训练循环测试通过")
        return True
    except Exception as e:
        print(f"\n✗ 迷你训练循环失败: {str(e)}")
        import traceback
        print("\n完整错误堆栈:")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("增强版Mask R-CNN - 快速流程测试")
    print("=" * 80)
    print("\n此脚本将使用少量影像测试模型流程的完整性")
    print("包括: 数据加载、模型创建、前向传播、反向传播、推理")
    
    # 配置
    data_root = './instance_segmentation_dataset'
    annotation_file = './instance_segmentation_dataset/annotations/instances.json'
    num_test_samples = 1015  # 仅使用3个样本
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n测试配置:")
    print(f"  数据集路径: {data_root}")
    print(f"  测试样本数: {num_test_samples}")
    print(f"  设备: {device}")
    
    # 检查数据集是否存在
    if not Path(data_root).exists():
        print(f"\n✗ 错误: 数据集目录不存在: {data_root}")
        print("请确保数据集位于正确的位置")
        return
    
    # 测试流程
    results = {}
    
    try:
        # 1. 测试数据加载
        dataset, dataloader = test_data_loading(data_root, annotation_file, num_test_samples)
        results['数据加载'] = True
        
        # 2. 测试模型创建
        model = test_model_creation(dataset.num_classes)
        results['模型创建'] = True
        
        # 3. 测试前向传播
        results['前向传播'] = test_forward_pass(model, dataloader, device)
        
        # 4. 测试反向传播
        results['反向传播'] = test_backward_pass(model, dataloader, device)
        
        # 5. 测试推理
        results['推理'] = test_inference(model, dataloader, device)
        
        # 6. 测试迷你训练循环
        results['迷你训练'] = test_mini_training(model, dataloader, device, num_iterations=3)
        
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 打印测试结果总结
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有测试通过！模型流程完整性验证成功")
        print("\n后续步骤:")
        print("  1. 运行完整训练: python train.py")
        print("  2. 查看快速开始: python quick_start.py")
        print("  3. 运行消融实验: python ablation_study.py")
    else:
        print("✗ 部分测试失败，请检查上述错误信息")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
