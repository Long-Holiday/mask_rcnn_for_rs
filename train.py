"""
训练脚本
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import json
from pathlib import Path

from modules.dataset import MultiModalRemoteSensingDataset, get_transforms, collate_fn
from models.enhanced_mask_rcnn import build_enhanced_mask_rcnn


class Trainer:
    """训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # 数据集
        self.train_dataset = MultiModalRemoteSensingDataset(
            root_dir=config['data_root'],
            annotation_file=config['annotation_file'],
            split='train',
            transforms=get_transforms(train=True),
            train_ratio=config['train_ratio']
        )
        
        self.val_dataset = MultiModalRemoteSensingDataset(
            root_dir=config['data_root'],
            annotation_file=config['annotation_file'],
            split='val',
            transforms=get_transforms(train=False),
            train_ratio=config['train_ratio']
        )
        
        # DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # 模型
        self.model = build_enhanced_mask_rcnn(
            num_classes=config['num_classes'],
            anchor_config_path=config.get('anchor_config_path'),
            backbone_pretrained=config['backbone_pretrained'],
            fusion_method=config['fusion_method'],
            use_cross_attention=config['use_cross_attention']
        )
        self.model.to(self.device)
        
        # 优化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            params,
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config['lr_milestones'],
            gamma=config['lr_gamma']
        )
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # 加载检查点
        if config.get('resume_from'):
            self.load_checkpoint(config['resume_from'])
        
        print(f"\n训练配置:")
        print(f"  设备: {self.device}")
        print(f"  训练集大小: {len(self.train_dataset)}")
        print(f"  验证集大小: {len(self.val_dataset)}")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  总轮数: {config['num_epochs']}")
        print(f"  学习率: {config['learning_rate']}")
        print(f"  使用交叉注意力: {config['use_cross_attention']}")
        print(f"  特征融合方式: {config['fusion_method']}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        loss_dict_total = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # 移动到设备
            rgb_images = images['rgb'].to(self.device)
            swir_images = images['swir'].to(self.device)
            nir_images = images['nir'].to(self.device)
            
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # 前向传播
            loss_dict = self.model(rgb_images, swir_images, nir_images, targets)
            
            # 计算总损失
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            self.optimizer.zero_grad()
            losses.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += losses.item()
            
            for k, v in loss_dict.items():
                if k not in loss_dict_total:
                    loss_dict_total[k] = 0
                loss_dict_total[k] += v.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/loss_total', losses.item(), global_step)
            for k, v in loss_dict.items():
                self.writer.add_scalar(f'Train/{k}', v.item(), global_step)
        
        # 平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_dict = {k: v / len(self.train_loader) for k, v in loss_dict_total.items()}
        
        return avg_loss, avg_loss_dict
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.train()  # Mask R-CNN在验证时也需要train模式来计算损失
        
        total_loss = 0
        loss_dict_total = {}
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for images, targets in pbar:
            # 移动到设备
            rgb_images = images['rgb'].to(self.device)
            swir_images = images['swir'].to(self.device)
            nir_images = images['nir'].to(self.device)
            
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # 前向传播
            loss_dict = self.model(rgb_images, swir_images, nir_images, targets)
            
            # 计算总损失
            losses = sum(loss for loss in loss_dict.values())
            
            # 统计
            total_loss += losses.item()
            
            for k, v in loss_dict.items():
                if k not in loss_dict_total:
                    loss_dict_total[k] = 0
                loss_dict_total[k] += v.item()
            
            pbar.set_postfix({'loss': f"{losses.item():.4f}"})
        
        # 平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_loss_dict = {k: v / len(self.val_loader) for k, v in loss_dict_total.items()}
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/loss_total', avg_loss, epoch)
        for k, v in avg_loss_dict.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)
        
        return avg_loss, avg_loss_dict
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")
        
        # 定期保存
        if (epoch + 1) % self.config['save_interval'] == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"从 {checkpoint_path} 加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"从epoch {self.start_epoch} 继续训练")
    
    def train(self):
        """完整训练流程"""
        print("\n开始训练...")
        print("=" * 80)
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # 训练
            train_loss, train_loss_dict = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_loss_dict = self.validate(epoch)
            
            # 学习率调度
            self.lr_scheduler.step()
            
            # 打印信息
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
            print("-" * 80)
        
        print("\n训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        self.writer.close()


def get_default_config():
    """获取默认配置"""
    return {
        # 数据
        'data_root': './instance_segmentation_dataset',
        'annotation_file': './instance_segmentation_dataset/annotations/instances.json',
        'train_ratio': 0.8,
        'num_classes': 81,  # 根据实际数据集调整
        
        # 模型
        'anchor_config_path': './outputs/anchors/anchor_config.json',
        'backbone_pretrained': True,
        'fusion_method': 'add',  # 'add', 'concat', 'weighted'
        'use_cross_attention': True,
        
        # 训练
        'batch_size': 2,
        'num_epochs': 50,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_milestones': [30, 40],
        'lr_gamma': 0.1,
        
        # 其他
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': './outputs/training',
        'save_interval': 5,
        'resume_from': None
    }


def main():
    parser = argparse.ArgumentParser(description='训练增强版Mask R-CNN')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_root', type=str, help='数据集根目录')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--no_cross_attention', action='store_true', help='不使用交叉注意力（消融实验）')
    parser.add_argument('--fusion_method', type=str, choices=['add', 'concat', 'weighted'], help='特征融合方式')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # 命令行参数覆盖
    if args.data_root:
        config['data_root'] = args.data_root
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.resume:
        config['resume_from'] = args.resume
    if args.no_cross_attention:
        config['use_cross_attention'] = False
    if args.fusion_method:
        config['fusion_method'] = args.fusion_method
    
    # 保存配置
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
