"""
消融实验脚本
用于测试不同组件的贡献
"""
import os
import torch
import json
from pathlib import Path
import argparse

from train import Trainer, get_default_config


def run_ablation_experiment(experiment_name, config_modifications):
    """
    运行单个消融实验
    
    Args:
        experiment_name: 实验名称
        config_modifications: 配置修改字典
    """
    print("\n" + "=" * 80)
    print(f"消融实验: {experiment_name}")
    print("=" * 80)
    
    # 获取基础配置
    config = get_default_config()
    
    # 应用修改
    config.update(config_modifications)
    
    # 设置输出目录
    config['output_dir'] = f"./outputs/ablation/{experiment_name}"
    
    # 保存配置
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n实验配置:")
    for key, value in config_modifications.items():
        print(f"  {key}: {value}")
    
    # 训练
    trainer = Trainer(config)
    trainer.train()
    
    print(f"\n实验 {experiment_name} 完成！")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    
    return trainer.best_val_loss


def main():
    parser = argparse.ArgumentParser(description='消融实验')
    parser.add_argument('--experiments', type=str, nargs='+', 
                       choices=['baseline', 'no_cross_attention', 'no_swir', 'fusion_concat', 'fusion_weighted', 'all'],
                       default=['all'],
                       help='要运行的实验')
    parser.add_argument('--num_epochs', type=int, default=30, help='每个实验的训练轮数')
    
    args = parser.parse_args()
    
    # 定义消融实验
    experiments = {
        'baseline': {
            'description': '完整模型（所有组件）',
            'config': {
                'use_cross_attention': True,
                'fusion_method': 'add',
                'num_epochs': args.num_epochs
            }
        },
        'no_cross_attention': {
            'description': '不使用交叉注意力',
            'config': {
                'use_cross_attention': False,
                'fusion_method': 'add',
                'num_epochs': args.num_epochs
            }
        },
        'fusion_concat': {
            'description': '使用拼接融合',
            'config': {
                'use_cross_attention': True,
                'fusion_method': 'concat',
                'num_epochs': args.num_epochs
            }
        },
        'fusion_weighted': {
            'description': '使用加权融合',
            'config': {
                'use_cross_attention': True,
                'fusion_method': 'weighted',
                'num_epochs': args.num_epochs
            }
        }
    }
    
    # 选择要运行的实验
    if 'all' in args.experiments:
        experiments_to_run = experiments.keys()
    else:
        experiments_to_run = args.experiments
    
    # 运行实验
    results = {}
    
    for exp_name in experiments_to_run:
        if exp_name not in experiments:
            print(f"警告: 未知实验 {exp_name}")
            continue
        
        exp_config = experiments[exp_name]
        print(f"\n开始实验: {exp_name}")
        print(f"描述: {exp_config['description']}")
        
        try:
            best_loss = run_ablation_experiment(exp_name, exp_config['config'])
            results[exp_name] = {
                'description': exp_config['description'],
                'best_val_loss': best_loss
            }
        except Exception as e:
            print(f"实验 {exp_name} 失败: {e}")
            results[exp_name] = {
                'description': exp_config['description'],
                'error': str(e)
            }
    
    # 保存总结
    summary_path = Path('./outputs/ablation/summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("消融实验总结")
    print("=" * 80)
    
    for exp_name, result in results.items():
        print(f"\n{exp_name}:")
        print(f"  描述: {result['description']}")
        if 'best_val_loss' in result:
            print(f"  最佳验证损失: {result['best_val_loss']:.4f}")
        else:
            print(f"  错误: {result.get('error', 'Unknown')}")
    
    print(f"\n总结已保存到: {summary_path}")


if __name__ == '__main__':
    main()
