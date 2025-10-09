import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from configs.config import cfg
from models.roi_pooling import ROIPoolingV2


class DetectionHead(Model):
    """
    检测头模块
    
    功能：
    1. 接收RoI Pooling后的特征
    2. 进行分类和边界框回归
    3. 计算检测损失
    4. 生成最终检测结果
    """
    
    def __init__(self, num_classes=None):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes or cfg.num_classes
        self.roi_pooling_size = cfg.roi_pooling_size
        
        # RoI Pooling层
        self.roi_pooling = ROIPoolingV2(pool_size=self.roi_pooling_size)
        
        # 全连接层
        self.fc1 = layers.Dense(4096, activation='relu', name='fc1')
        self.fc2 = layers.Dense(4096, activation='relu', name='fc2')
        
        # 分类头
        self.cls_score = layers.Dense(self.num_classes, name='cls_score')
        
        # 回归头
        self.bbox_pred = layers.Dense(4 * self.num_classes, name='bbox_pred')
        
        # Dropout层
        self.dropout1 = layers.Dropout(0.5)
        self.dropout2 = layers.Dropout(0.5)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for layer in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.01)
                layer.bias_initializer = tf.keras.initializers.Constant(0.0)
    
    def call(self, inputs, training=False, gt_boxes=None, gt_labels=None):
        """
        前向传播
        
        Args:
            inputs: 包含以下元素的元组
                - feat_map: 特征图 [batch_size, height, width, channels]
                - rois: RoI坐标 [batch_size, num_rois, 4]
            training: 是否为训练模式
            gt_boxes: 真实边界框 [batch_size, num_gt_boxes, 4]
            gt_labels: 真实标签 [batch_size, num_gt_boxes]
            
        Returns:
            outputs: 包含检测结果的字典
        """
        feat_map, rois = inputs
        
        # 1. RoI Pooling
        pooled_features = self.roi_pooling([feat_map, rois])  # [batch_size, num_rois, pool_size, pool_size, channels]
        
        # 2. 展平特征
        batch_size = tf.shape(pooled_features)[0]
        num_rois = tf.shape(pooled_features)[1]
        flattened_features = tf.reshape(pooled_features, [batch_size * num_rois, self.roi_pooling_size * self.roi_pooling_size * 512])
        
        # 3. 全连接层
        fc1_out = self.fc1(flattened_features)
        fc1_out = self.dropout1(fc1_out, training=training)
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.dropout2(fc2_out, training=training)
        
        # 4. 分类和回归
        cls_scores = self.cls_score(fc2_out)  # [batch_size * num_rois, num_classes]
        bbox_preds = self.bbox_pred(fc2_out)  # [batch_size * num_rois, 4 * num_classes]
        
        # 5. 重塑输出
        cls_scores = tf.reshape(cls_scores, [batch_size, num_rois, self.num_classes])
        bbox_preds = tf.reshape(bbox_preds, [batch_size, num_rois, 4 * self.num_classes])
        
        outputs = {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'pooled_features': pooled_features
        }
        
        # 6. 如果是训练模式，计算检测损失
        if training and gt_boxes is not None and gt_labels is not None:
            cls_loss, bbox_loss = self.compute_detection_loss(
                cls_scores, bbox_preds, rois, gt_boxes, gt_labels
            )
            outputs.update({
                'cls_loss': cls_loss,
                'bbox_loss': bbox_loss
            })
        
        return outputs
    
    def compute_detection_loss(self, cls_scores, bbox_preds, rois, gt_boxes, gt_labels):
        """
        计算检测损失
        
        Args:
            cls_scores: 分类得分 [batch_size, num_rois, num_classes]
            bbox_preds: 回归预测 [batch_size, num_rois, 4 * num_classes]
            rois: RoI坐标 [batch_size, num_rois, 4]
            gt_boxes: 真实边界框 [batch_size, num_gt_boxes, 4]
            gt_labels: 真实标签 [batch_size, num_gt_boxes]
            
        Returns:
            cls_loss: 分类损失
            bbox_loss: 回归损失
        """
        batch_size = tf.shape(cls_scores)[0]
        num_rois = tf.shape(cls_scores)[1]
        
        cls_losses = []
        bbox_losses = []
        
        for b in range(batch_size):
            batch_cls_scores = cls_scores[b]  # [num_rois, num_classes]
            batch_bbox_preds = bbox_preds[b]  # [num_rois, 4 * num_classes]
            batch_rois = rois[b]  # [num_rois, 4]
            batch_gt_boxes = gt_boxes[b]  # [num_gt_boxes, 4]
            batch_gt_labels = gt_labels[b]  # [num_gt_boxes]
            
            # 过滤掉背景框
            valid_mask = batch_gt_labels > 0
            batch_gt_boxes = tf.boolean_mask(batch_gt_boxes, valid_mask)
            batch_gt_labels = tf.boolean_mask(batch_gt_labels, valid_mask)
            
            if tf.shape(batch_gt_boxes)[0] == 0:
                # 如果没有有效目标，损失为0
                cls_losses.append(tf.constant(0.0))
                bbox_losses.append(tf.constant(0.0))
                continue
            
            # 为RoI分配标签
            roi_labels, roi_targets = self._assign_roi_targets(
                batch_rois, batch_gt_boxes, batch_gt_labels
            )
            
            # 计算分类损失
            cls_loss = self._compute_cls_loss(batch_cls_scores, roi_labels)
            cls_losses.append(cls_loss)
            
            # 计算回归损失
            bbox_loss = self._compute_bbox_loss(batch_bbox_preds, roi_targets, roi_labels)
            bbox_losses.append(bbox_loss)
        
        # 平均损失
        total_cls_loss = tf.reduce_mean(tf.stack(cls_losses))
        total_bbox_loss = tf.reduce_mean(tf.stack(bbox_losses))
        
        return total_cls_loss, total_bbox_loss
    
    def _assign_roi_targets(self, rois, gt_boxes, gt_labels):
        """
        为RoI分配标签和回归目标
        
        Args:
            rois: [num_rois, 4]
            gt_boxes: [num_gt_boxes, 4]
            gt_labels: [num_gt_boxes]
            
        Returns:
            roi_labels: [num_rois]
            roi_targets: [num_rois, 4]
        """
        num_rois = tf.shape(rois)[0]
        num_gt_boxes = tf.shape(gt_boxes)[0]
        
        # 计算IoU矩阵
        ious = self._compute_iou_matrix(rois, gt_boxes)  # [num_rois, num_gt_boxes]
        
        # 找到每个RoI的最大IoU和对应的GT框
        max_ious = tf.reduce_max(ious, axis=1)  # [num_rois]
        argmax_ious = tf.argmax(ious, axis=1)  # [num_rois]
        
        # 分配标签
        roi_labels = tf.zeros(num_rois, dtype=tf.int32)  # 默认背景
        
        # 正样本：IoU > 0.5
        positive_mask = max_ious >= 0.5
        matched_gt_labels = tf.gather(gt_labels, argmax_ious)
        roi_labels = tf.where(positive_mask, matched_gt_labels, roi_labels)
        
        # 负样本：IoU < 0.5
        negative_mask = max_ious < 0.5
        roi_labels = tf.where(negative_mask, tf.zeros_like(roi_labels), roi_labels)
        
        # 计算回归目标
        matched_gt_boxes = tf.gather(gt_boxes, argmax_ious)
        roi_targets = self._compute_bbox_targets(rois, matched_gt_boxes)
        
        return roi_labels, roi_targets
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """计算IoU矩阵"""
        # 计算交集
        x1 = tf.maximum(boxes1[:, 0:1], tf.transpose(boxes2[:, 0:1]))
        y1 = tf.maximum(boxes1[:, 1:2], tf.transpose(boxes2[:, 1:2]))
        x2 = tf.minimum(boxes1[:, 2:3], tf.transpose(boxes2[:, 2:3]))
        y2 = tf.minimum(boxes1[:, 3:4], tf.transpose(boxes2[:, 3:4]))
        
        intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
        
        # 计算面积
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # 计算并集
        union = tf.expand_dims(area1, 1) + tf.expand_dims(area2, 0) - intersection
        
        # 计算IoU
        ious = intersection / (union + 1e-8)
        
        return ious
    
    def _compute_bbox_targets(self, rois, gt_boxes):
        """
        计算边界框回归目标
        
        Args:
            rois: [num_rois, 4] (x1, y1, x2, y2)
            gt_boxes: [num_rois, 4] (x1, y1, x2, y2)
            
        Returns:
            targets: [num_rois, 4] (dx, dy, dw, dh)
        """
        # 计算中心点和宽高
        roi_width = rois[:, 2] - rois[:, 0] + 1.0
        roi_height = rois[:, 3] - rois[:, 1] + 1.0
        roi_ctr_x = rois[:, 0] + 0.5 * roi_width
        roi_ctr_y = rois[:, 1] + 0.5 * roi_height
        
        gt_width = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
        gt_height = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
        gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_width
        gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_height
        
        # 计算回归目标
        dx = (gt_ctr_x - roi_ctr_x) / roi_width
        dy = (gt_ctr_y - roi_ctr_y) / roi_height
        dw = tf.math.log(gt_width / roi_width)
        dh = tf.math.log(gt_height / roi_height)
        
        targets = tf.stack([dx, dy, dw, dh], axis=1)
        
        return targets
    
    def _compute_cls_loss(self, cls_scores, roi_labels):
        """计算分类损失"""
        # 过滤掉忽略的样本
        valid_mask = roi_labels >= 0
        valid_scores = tf.boolean_mask(cls_scores, valid_mask)
        valid_labels = tf.boolean_mask(roi_labels, valid_mask)
        
        if tf.shape(valid_scores)[0] == 0:
            return tf.constant(0.0)
        
        # 计算交叉熵损失
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=valid_labels, logits=valid_scores
        )
        
        return tf.reduce_mean(cls_loss)
    
    def _compute_bbox_loss(self, bbox_preds, roi_targets, roi_labels):
        """计算回归损失"""
        # 只计算正样本的回归损失
        positive_mask = roi_labels > 0
        
        if tf.reduce_sum(tf.cast(positive_mask, tf.int32)) == 0:
            return tf.constant(0.0)
        
        # 重塑预测结果 [num_rois, 4 * num_classes] -> [num_rois, num_classes, 4]
        bbox_preds = tf.reshape(bbox_preds, [-1, self.num_classes, 4])
        
        # 获取正样本的预测和目标
        positive_preds = tf.boolean_mask(bbox_preds, positive_mask)
        positive_targets = tf.boolean_mask(roi_targets, positive_mask)
        positive_labels = tf.boolean_mask(roi_labels, positive_mask)
        
        # 选择对应类别的预测
        batch_indices = tf.range(tf.shape(positive_labels)[0])
        indices = tf.stack([batch_indices, positive_labels], axis=1)
        selected_preds = tf.gather_nd(positive_preds, indices)
        
        # 计算smooth L1损失
        diff = tf.abs(selected_preds - positive_targets)
        bbox_loss = tf.where(
            diff < 1.0,
            0.5 * diff ** 2,
            diff - 0.5
        )
        
        return tf.reduce_mean(tf.reduce_sum(bbox_loss, axis=1))
    
    def generate_detections(self, cls_scores, bbox_preds, rois, im_info, score_thresh=0.05, nms_thresh=0.3):
        """
        生成最终检测结果
        
        Args:
            cls_scores: 分类得分 [batch_size, num_rois, num_classes]
            bbox_preds: 回归预测 [batch_size, num_rois, 4 * num_classes]
            rois: RoI坐标 [batch_size, num_rois, 4]
            im_info: 图像信息 [batch_size, 3]
            score_thresh: 得分阈值
            nms_thresh: NMS阈值
            
        Returns:
            detections: 检测结果 [batch_size, num_detections, 6] (x1, y1, x2, y2, score, class)
        """
        batch_size = tf.shape(cls_scores)[0]
        
        detections_list = []
        
        for b in range(batch_size):
            batch_cls_scores = cls_scores[b]  # [num_rois, num_classes]
            batch_bbox_preds = bbox_preds[b]  # [num_rois, 4 * num_classes]
            batch_rois = rois[b]  # [num_rois, 4]
            
            # 计算softmax概率
            cls_probs = tf.nn.softmax(batch_cls_scores, axis=-1)
            
            # 获取最大概率和对应类别
            max_scores = tf.reduce_max(cls_probs, axis=-1)  # [num_rois]
            max_classes = tf.argmax(cls_probs, axis=-1)  # [num_rois]
            
            # 过滤低得分
            score_mask = max_scores >= score_thresh
            filtered_rois = tf.boolean_mask(batch_rois, score_mask)
            filtered_scores = tf.boolean_mask(max_scores, score_mask)
            filtered_classes = tf.boolean_mask(max_classes, score_mask)
            filtered_preds = tf.boolean_mask(batch_bbox_preds, score_mask)
            
            if tf.shape(filtered_rois)[0] == 0:
                detections_list.append(tf.zeros([0, 6], dtype=tf.float32))
                continue
            
            # 应用回归变换
            # 重塑预测结果 [num_rois, 4 * num_classes] -> [num_rois, num_classes, 4]
            filtered_preds = tf.reshape(filtered_preds, [-1, self.num_classes, 4])
            
            # 选择对应类别的预测
            batch_indices = tf.range(tf.shape(filtered_classes)[0])
            indices = tf.stack([batch_indices, filtered_classes], axis=1)
            selected_preds = tf.gather_nd(filtered_preds, indices)
            
            # 应用回归变换
            det_boxes = self._apply_bbox_transform(filtered_rois, selected_preds)
            
            # 裁剪到图像边界
            height, width = im_info[b, 0], im_info[b, 1]
            det_boxes = tf.stack([
                tf.clip_by_value(det_boxes[:, 0], 0, width - 1),
                tf.clip_by_value(det_boxes[:, 1], 0, height - 1),
                tf.clip_by_value(det_boxes[:, 2], 0, width - 1),
                tf.clip_by_value(det_boxes[:, 3], 0, height - 1)
            ], axis=1)
            
            # 应用NMS
            keep_indices = tf.image.non_max_suppression(
                det_boxes, filtered_scores,
                max_output_size=100,  # 最多保留100个检测结果
                iou_threshold=nms_thresh
            )
            
            final_boxes = tf.gather(det_boxes, keep_indices)
            final_scores = tf.gather(filtered_scores, keep_indices)
            final_classes = tf.gather(filtered_classes, keep_indices)
            
            # 组合检测结果
            detections = tf.stack([
                final_boxes[:, 0], final_boxes[:, 1], final_boxes[:, 2], final_boxes[:, 3],
                final_scores, tf.cast(final_classes, tf.float32)
            ], axis=1)
            
            detections_list.append(detections)
        
        # 堆叠结果
        detections = tf.stack(detections_list, axis=0)
        
        return detections
    
    def _apply_bbox_transform(self, rois, bbox_preds):
        """应用边界框回归变换"""
        # 计算RoI的中心点和宽高
        roi_width = rois[:, 2] - rois[:, 0] + 1.0
        roi_height = rois[:, 3] - rois[:, 1] + 1.0
        roi_ctr_x = rois[:, 0] + 0.5 * roi_width
        roi_ctr_y = rois[:, 1] + 0.5 * roi_height
        
        # 应用回归变换
        pred_ctr_x = roi_ctr_x + bbox_preds[:, 0] * roi_width
        pred_ctr_y = roi_ctr_y + bbox_preds[:, 1] * roi_height
        pred_width = roi_width * tf.exp(bbox_preds[:, 2])
        pred_height = roi_height * tf.exp(bbox_preds[:, 3])
        
        # 转换回边界框格式
        boxes = tf.stack([
            pred_ctr_x - 0.5 * pred_width,
            pred_ctr_y - 0.5 * pred_height,
            pred_ctr_x + 0.5 * pred_width - 1.0,
            pred_ctr_y + 0.5 * pred_height - 1.0
        ], axis=1)
        
        return boxes


if __name__ == "__main__":
    print("请运行 python incremental_compile_test.py 进行完整测试")
