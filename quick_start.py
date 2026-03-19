"""
快速开始脚本
按顺序执行完整流程：锚框生成 -> 训练 -> 评估
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并打印输出"""
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)
    print(f"执行命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n错误: {description} 失败")
        sys.exit(1)
    
    print(f"\n{description} 完成！")


def main():
    print("\n" + "=" * 80)
    print("增强版Mask R-CNN - 快速开始")
    print("=" * 80)
    
    # 检查数据集
    data_root = Path('./instance_segmentation_dataset')
    if not data_root.exists():
        print(f"\n错误: 数据集目录不存在: {data_root}")
        print("请确保数据集位于正确的位置")
        sys.exit(1)
    
    print("\n数据集检查:")
    print(f"  RGB影像: {len(list((data_root / 'images').glob('*.png')))} 张")
    print(f"  SWIR影像: {len(list((data_root / 'swir_images').glob('*.png')))} 张")
    print(f"  NIR影像: {len(list((data_root / 'nir_images').glob('*.png')))} 张")
    
    # 步骤1: 生成锚框
    anchor_config = Path('./outputs/anchors/anchor_config.json')
    if not anchor_config.exists():
        run_command(
            "python modules/anchor_generator.py",
            "步骤1: 生成最优锚框"
        )
    else:
        print("\n锚框配置已存在，跳过生成步骤")
    
    # 步骤2: 训练模型（小规模测试）
    print("\n是否开始训练？这将需要较长时间...")
    print("  1. 快速测试（5个epoch）")
    print("  2. 完整训练（50个epoch）")
    print("  3. 跳过训练")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        run_command(
            "python train.py --num_epochs 5 --batch_size 2",
            "步骤2: 快速测试训练"
        )
    elif choice == '2':
        run_command(
            "python train.py --num_epochs 50 --batch_size 2",
            "步骤2: 完整训练"
        )
    else:
        print("\n跳过训练步骤")
    
    # 步骤3: 评估模型
    checkpoint = Path('./outputs/training/checkpoint_best.pth')
    if checkpoint.exists():
        print("\n是否评估模型？(y/n): ", end='')
        if input().strip().lower() == 'y':
            run_command(
                f"python evaluate.py --checkpoint {checkpoint}",
                "步骤3: 评估模型"
            )
    else:
        print("\n未找到训练好的模型，跳过评估")
    
    # 完成
    print("\n" + "=" * 80)
    print("快速开始流程完成！")
    print("=" * 80)
    
    print("\n后续步骤:")
    print("  1. 查看TensorBoard: tensorboard --logdir ./outputs/training/logs")
    print("  2. 运行消融实验: python ablation_study.py")
    print("  3. 查看使用说明: 使用说明.md")


if __name__ == '__main__':
    main()
