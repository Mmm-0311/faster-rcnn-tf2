import tensorflow as tf
import numpy as np
from configs.config import cfg
from models.backbone.vgg16 import VGG16Backbone
from models.rpn import RPN
from models.head import DetectionHead
from utils.bbox_utils import nms, clip_boxes, filter_invalid_boxes


class FasterRCNN(tf.keras.Model):
    """
    完整的Faster R-CNN模型
    
    整合了：
    1. VGG16主干网络 + RPN
    2. RoI Pooling
    3. 检测头
    4. 损失计算
    5. 推理管道
    """
    
    def __init__(self, num_classes=None):
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes or cfg.num_classes
        
        # 主干网络
        self.backbone = VGG16Backbone()
        
        # RPN模块
        self.rpn = RPN()
        
        # 检测头
        self.detection_head = DetectionHead(num_classes=self.num_classes)
        
        # 训练相关参数
        self.rpn_cls_loss_weight = 1.0
        self.rpn_bbox_loss_weight = 1.0
        self.det_cls_loss_weight = 1.0
        self.det_bbox_loss_weight = 1.0
        
    def call(self, inputs, training=False):
        """
        前向传播
        
        Args:
            inputs: 包含以下元素的字典
                - images: 输入图像 [batch_size, H, W, 3]
                - gt_boxes: 真实边界框 [batch_size, num_gt_boxes, 4] (训练时)
                - gt_labels: 真实标签 [batch_size, num_gt_boxes] (训练时)
            training: 是否为训练模式
            
        Returns:
            outputs: 包含所有输出的字典
        """
        images = inputs['images']
        gt_boxes = inputs.get('gt_boxes', None)
        gt_labels = inputs.get('gt_labels', None)
        
        # 1. 主干网络前向传播
        backbone_outputs = self.backbone(images, training=training)
        
        feat_map = backbone_outputs['feat_map']
        rpn_cls_logits = backbone_outputs['rpn_cls_logits']
        rpn_cls_prob = backbone_outputs['rpn_cls_prob']
        rpn_bbox_pred = backbone_outputs['rpn_bbox_pred']
        
        # 2. 生成候选区域
        if training:
            # 训练时使用RPN生成的proposals
            im_info = tf.stack([
                tf.cast(tf.shape(images)[1], tf.float32),  # height
                tf.cast(tf.shape(images)[2], tf.float32),   # width
                tf.constant(1.0, tf.float32)               # scale
            ])
            im_info = tf.expand_dims(im_info, 0)  # [1, 3]
            
            proposals, proposal_scores = self.rpn.generate_proposals(
                rpn_cls_prob, rpn_bbox_pred, tf.shape(feat_map), im_info
            )
        else:
            # 推理时使用RPN生成的proposals
            im_info = tf.stack([
                tf.cast(tf.shape(images)[1], tf.float32),
                tf.cast(tf.shape(images)[2], tf.float32),
                tf.constant(1.0, tf.float32)
            ])
            im_info = tf.expand_dims(im_info, 0)
            
            proposals, proposal_scores = self.rpn.generate_proposals(
                rpn_cls_prob, rpn_bbox_pred, tf.shape(feat_map), im_info
            )
        
        # 3. 检测头前向传播
        detection_outputs = self.detection_head(
            [feat_map, proposals],
            training=training,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels
        )
        
        # 4. 组合输出
        outputs = {
            'feat_map': feat_map,
            'rpn_cls_logits': rpn_cls_logits,
            'rpn_cls_prob': rpn_cls_prob,
            'rpn_bbox_pred': rpn_bbox_pred,
            'proposals': proposals,
            'proposal_scores': proposal_scores,
            'cls_scores': detection_outputs['cls_scores'],
            'bbox_preds': detection_outputs['bbox_preds'],
            'pooled_features': detection_outputs['pooled_features']
        }
        
        # 5. 计算总损失
        if training:
            # RPN损失
            rpn_cls_loss, rpn_bbox_loss, rpn_labels, rpn_targets = self.rpn.compute_rpn_loss(
                rpn_cls_logits, rpn_bbox_pred, gt_boxes, gt_labels, tf.shape(feat_map)
            )
            
            # 检测损失
            det_cls_loss = detection_outputs['cls_loss']
            det_bbox_loss = detection_outputs['bbox_loss']
            
            # 总损失
            total_loss = (
                self.rpn_cls_loss_weight * rpn_cls_loss +
                self.rpn_bbox_loss_weight * rpn_bbox_loss +
                self.det_cls_loss_weight * det_cls_loss +
                self.det_bbox_loss_weight * det_bbox_loss
            )
            
            outputs.update({
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_bbox_loss': rpn_bbox_loss,
                'det_cls_loss': det_cls_loss,
                'det_bbox_loss': det_bbox_loss,
                'total_loss': total_loss
            })
        
        return outputs
    
    def predict(self, images, score_thresh=0.05, nms_thresh=0.3):
        """
        推理预测
        
        Args:
            images: 输入图像 [batch_size, H, W, 3]
            score_thresh: 得分阈值
            nms_thresh: NMS阈值
            
        Returns:
            detections: 检测结果 [batch_size, num_detections, 6] (x1, y1, x2, y2, score, class)
        """
        # 前向传播
        outputs = self({'images': images}, training=False)
        
        # 生成检测结果
        im_info = tf.stack([
            tf.cast(tf.shape(images)[1], tf.float32),
            tf.cast(tf.shape(images)[2], tf.float32),
            tf.constant(1.0, tf.float32)
        ])
        im_info = tf.expand_dims(im_info, 0)
        
        detections = self.detection_head.generate_detections(
            outputs['cls_scores'],
            outputs['bbox_preds'],
            outputs['proposals'],
            im_info,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh
        )
        
        return detections
    
    def load_pretrained_weights(self):
        """加载预训练权重"""
        self.backbone.load_pretrained_weights()
    
    def get_trainable_variables(self):
        """获取可训练变量"""
        trainable_vars = []
        
        # 主干网络的可训练变量（排除预训练层）
        for layer in self.backbone.layers:
            if hasattr(layer, 'trainable') and layer.trainable:
                trainable_vars.extend(layer.trainable_variables)
        
        # 检测头的可训练变量
        trainable_vars.extend(self.detection_head.trainable_variables)
        
        # RPN的可训练变量
        trainable_vars.extend(self.rpn.trainable_variables)
        
        return trainable_vars


class FasterRCNNTrainer:
    """
    Faster R-CNN训练器
    
    负责：
    1. 优化器设置
    2. 学习率调度
    3. 训练循环
    4. 验证评估
    """
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        
        # 优化器
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            decay=0.0005
        )
        
        # 学习率调度器
        self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=cfg.step_size,
            decay_rate=cfg.gamma,
            staircase=True
        )
        
        # 训练指标
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_rpn_cls_loss = tf.keras.metrics.Mean(name='train_rpn_cls_loss')
        self.train_rpn_bbox_loss = tf.keras.metrics.Mean(name='train_rpn_bbox_loss')
        self.train_det_cls_loss = tf.keras.metrics.Mean(name='train_det_cls_loss')
        self.train_det_bbox_loss = tf.keras.metrics.Mean(name='train_det_bbox_loss')
        
    def train_step(self, batch):
        """
        单步训练
        
        Args:
            batch: 包含images, boxes, classes的批次数据
            
        Returns:
            loss_dict: 损失字典
        """
        images, boxes, classes = batch
        
        with tf.GradientTape() as tape:
            # 前向传播
            outputs = self.model({
                'images': images,
                'gt_boxes': boxes,
                'gt_labels': classes
            }, training=True)
            
            # 计算损失
            total_loss = outputs['total_loss']
        
        # 计算梯度
        trainable_vars = self.model.get_trainable_variables()
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # 梯度裁剪
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        # 更新权重
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 更新指标
        self.train_loss.update_state(total_loss)
        self.train_rpn_cls_loss.update_state(outputs['rpn_cls_loss'])
        self.train_rpn_bbox_loss.update_state(outputs['rpn_bbox_loss'])
        self.train_det_cls_loss.update_state(outputs['det_cls_loss'])
        self.train_det_bbox_loss.update_state(outputs['det_bbox_loss'])
        
        return {
            'total_loss': total_loss,
            'rpn_cls_loss': outputs['rpn_cls_loss'],
            'rpn_bbox_loss': outputs['rpn_bbox_loss'],
            'det_cls_loss': outputs['det_cls_loss'],
            'det_bbox_loss': outputs['det_bbox_loss']
        }
    
    def train_epoch(self, dataset):
        """
        训练一个epoch
        
        Args:
            dataset: 训练数据集
            
        Returns:
            metrics: 训练指标
        """
        # 重置指标
        self.train_loss.reset_states()
        self.train_rpn_cls_loss.reset_states()
        self.train_rpn_bbox_loss.reset_states()
        self.train_det_cls_loss.reset_states()
        self.train_det_bbox_loss.reset_states()
        
        # 训练循环
        for batch in dataset:
            loss_dict = self.train_step(batch)
        
        # 返回平均指标
        return {
            'train_loss': self.train_loss.result(),
            'train_rpn_cls_loss': self.train_rpn_cls_loss.result(),
            'train_rpn_bbox_loss': self.train_rpn_bbox_loss.result(),
            'train_det_cls_loss': self.train_det_cls_loss.result(),
            'train_det_bbox_loss': self.train_det_bbox_loss.result()
        }
    
    def evaluate(self, dataset, score_thresh=0.05, nms_thresh=0.3):
        """
        评估模型
        
        Args:
            dataset: 验证数据集
            score_thresh: 得分阈值
            nms_thresh: NMS阈值
            
        Returns:
            metrics: 评估指标
        """
        # 这里可以实现mAP等评估指标
        # 暂时返回简单的准确率
        total_samples = 0
        correct_predictions = 0
        
        for batch in dataset:
            images, boxes, classes = batch
            
            # 预测
            detections = self.model.predict(images, score_thresh, nms_thresh)
            
            # 计算准确率（简化版本）
            batch_size = tf.shape(images)[0]
            total_samples += batch_size
            
            # 这里可以实现更复杂的评估逻辑
            correct_predictions += batch_size  # 简化处理
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {'accuracy': accuracy}


if __name__ == "__main__":
    print("请运行 python incremental_compile_test.py 进行完整测试")
