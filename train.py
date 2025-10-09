import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from configs.config import cfg
from datasets.voc import VOCDataset, VOC_CLASSES
from models.faster_rcnn import FasterRCNN, FasterRCNNTrainer
from utils.logger import setup_logger


class TrainingManager:
    """
    训练管理器
    
    负责：
    1. 数据加载
    2. 模型训练
    3. 检查点保存
    4. 日志记录
    5. 可视化
    """
    
    def __init__(self, config=None):
        self.config = config or cfg
        self.logger = setup_logger('training')
        
        # 创建输出目录
        self.checkpoint_dir = os.path.join(self.config.checkpoint_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.log_dir = os.path.join(self.config.log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化模型和训练器
        self.model = FasterRCNN()
        self.trainer = FasterRCNNTrainer(self.model, learning_rate=self.config.learning_rate)
        
        # 检查点管理器
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.trainer.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=5
        )
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_rpn_cls_loss': [],
            'train_rpn_bbox_loss': [],
            'train_det_cls_loss': [],
            'train_det_bbox_loss': [],
            'val_accuracy': []
        }
        
        self.logger.info(f"训练管理器初始化完成")
        self.logger.info(f"检查点目录: {self.checkpoint_dir}")
        self.logger.info(f"日志目录: {self.log_dir}")
    
    def load_data(self):
        """加载训练和验证数据"""
        self.logger.info("加载数据集...")
        
        # 训练数据集
        self.train_dataset = VOCDataset(split='train').get_dataset(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # 验证数据集
        self.val_dataset = VOCDataset(split='test').get_dataset(
            batch_size=1,  # 验证时使用batch_size=1
            shuffle=False
        )
        
        self.logger.info(f"训练数据集加载完成")
        self.logger.info(f"验证数据集加载完成")
    
    def train(self, num_epochs=10, save_freq=5, val_freq=2):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_freq: 保存频率（每N个epoch保存一次）
            val_freq: 验证频率（每N个epoch验证一次）
        """
        self.logger.info(f"开始训练，共{num_epochs}个epoch")
        
        # 加载预训练权重
        self.logger.info("加载预训练权重...")
        self.model.load_pretrained_weights()
        
        # 训练循环
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_metrics = self.trainer.train_epoch(self.train_dataset)
            
            # 记录训练指标
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['train_loss'].numpy())
            self.train_history['train_rpn_cls_loss'].append(train_metrics['train_rpn_cls_loss'].numpy())
            self.train_history['train_rpn_bbox_loss'].append(train_metrics['train_rpn_bbox_loss'].numpy())
            self.train_history['train_det_cls_loss'].append(train_metrics['train_det_cls_loss'].numpy())
            self.train_history['train_det_bbox_loss'].append(train_metrics['train_det_bbox_loss'].numpy())
            
            # 打印训练指标
            self.logger.info(f"训练损失: {train_metrics['train_loss']:.4f}")
            self.logger.info(f"RPN分类损失: {train_metrics['train_rpn_cls_loss']:.4f}")
            self.logger.info(f"RPN回归损失: {train_metrics['train_rpn_bbox_loss']:.4f}")
            self.logger.info(f"检测分类损失: {train_metrics['train_det_cls_loss']:.4f}")
            self.logger.info(f"检测回归损失: {train_metrics['train_det_bbox_loss']:.4f}")
            
            # 验证
            if (epoch + 1) % val_freq == 0:
                self.logger.info("开始验证...")
                val_metrics = self.trainer.evaluate(self.val_dataset)
                self.train_history['val_accuracy'].append(val_metrics['accuracy'])
                self.logger.info(f"验证准确率: {val_metrics['accuracy']:.4f}")
            
            # 保存检查点
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1)
                self.logger.info(f"检查点已保存: epoch {epoch + 1}")
            
            # 绘制训练曲线
            self.plot_training_curves()
        
        self.logger.info("训练完成！")
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint_path = self.checkpoint_manager.save(checkpoint_number=epoch)
        self.logger.info(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.checkpoint.restore(checkpoint_path)
        self.logger.info(f"检查点已加载: {checkpoint_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.train_history['epoch']) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Faster R-CNN Training Curves', fontsize=16)
        
        # 总损失
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # RPN分类损失
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['train_rpn_cls_loss'], 'r-', label='RPN Cls Loss')
        axes[0, 1].set_title('RPN Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RPN回归损失
        axes[0, 2].plot(self.train_history['epoch'], self.train_history['train_rpn_bbox_loss'], 'g-', label='RPN Bbox Loss')
        axes[0, 2].set_title('RPN Regression Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 检测分类损失
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['train_det_cls_loss'], 'm-', label='Det Cls Loss')
        axes[1, 0].set_title('Detection Classification Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 检测回归损失
        axes[1, 1].plot(self.train_history['epoch'], self.train_history['train_det_bbox_loss'], 'c-', label='Det Bbox Loss')
        axes[1, 1].set_title('Detection Regression Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 验证准确率
        if len(self.train_history['val_accuracy']) > 0:
            val_epochs = self.train_history['epoch'][::2]  # 假设每2个epoch验证一次
            axes[1, 2].plot(val_epochs[:len(self.train_history['val_accuracy'])], 
                          self.train_history['val_accuracy'], 'k-', label='Val Accuracy')
            axes[1, 2].set_title('Validation Accuracy')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, score_thresh=0.05, nms_thresh=0.3):
        """
        评估模型性能
        
        Args:
            score_thresh: 得分阈值
            nms_thresh: NMS阈值
            
        Returns:
            metrics: 评估指标
        """
        self.logger.info("开始模型评估...")
        
        # 这里可以实现更详细的评估指标，如mAP
        # 暂时使用简单的准确率
        val_metrics = self.trainer.evaluate(self.val_dataset, score_thresh, nms_thresh)
        
        self.logger.info(f"评估完成，准确率: {val_metrics['accuracy']:.4f}")
        
        return val_metrics
    
    def demo_inference(self, image_path=None, num_samples=3):
        """
        演示推理结果
        
        Args:
            image_path: 图像路径（如果为None，则使用验证集样本）
            num_samples: 演示样本数量
        """
        self.logger.info("开始演示推理...")
        
        if image_path:
            # 单张图像推理
            # 这里可以实现单张图像的推理逻辑
            pass
        else:
            # 使用验证集样本
            for i, batch in enumerate(self.val_dataset.take(num_samples)):
                images, gt_boxes, gt_labels = batch
                
                # 预测
                detections = self.model.predict(images)
                
                # 可视化结果
                self.visualize_detections(images[0], detections[0], gt_boxes[0], gt_labels[0], i)
    
    def visualize_detections(self, image, detections, gt_boxes, gt_labels, sample_idx):
        """
        可视化检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果 [num_detections, 6]
            gt_boxes: 真实边界框
            gt_labels: 真实标签
            sample_idx: 样本索引
        """
        # 恢复图像（加回均值并转换为RGB）
        pixel_means = np.array([102.9801, 115.9465, 122.7717])  # BGR格式
        image_np = image.numpy() + pixel_means
        image_np = image_np[:, :, [2, 1, 0]]  # BGR转RGB
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        # 创建图像
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_np)
        
        # 绘制真实边界框（绿色）
        for box, label in zip(gt_boxes.numpy(), gt_labels.numpy()):
            if label > 0:  # 非背景
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='g', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1, f'GT: {VOC_CLASSES[label]}', color='g', fontsize=10)
        
        # 绘制检测结果（红色）
        for detection in detections.numpy():
            if len(detection) >= 6:
                x1, y1, x2, y2, score, cls = detection
                if score > 0.1:  # 只显示高得分的检测
                    rect = plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1, f'Pred: {VOC_CLASSES[int(cls)]} ({score:.2f})', 
                           color='r', fontsize=10)
        
        ax.set_title(f'Detection Results - Sample {sample_idx}')
        ax.axis('off')
        
        # 保存图像
        demo_path = os.path.join(self.log_dir, f'demo_sample_{sample_idx}.png')
        plt.savefig(demo_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"演示图像已保存: {demo_path}")


def main():
    """主训练函数"""
    # 设置TensorFlow
    tf.config.run_functions_eagerly(True)
    
    # 创建训练管理器
    trainer = TrainingManager()
    
    # 加载数据
    trainer.load_data()
    
    # 训练模型
    trainer.train(num_epochs=20, save_freq=5, val_freq=2)
    
    # 评估模型
    trainer.evaluate_model()
    
    # 演示推理
    trainer.demo_inference(num_samples=5)
    
    print("训练完成！")


if __name__ == "__main__":
    main()
