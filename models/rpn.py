import tensorflow as tf
import numpy as np
from configs.config import cfg


class RPN(tf.keras.Model):
    """
    Region Proposal Network (RPN) 模块
    
    功能：
    1. 在特征图上生成锚框
    2. 计算RPN分类和回归损失
    3. 生成候选区域（proposals）
    4. 应用非极大值抑制（NMS）
    """
    
    def __init__(self):
        super(RPN, self).__init__()
        self.num_anchors = cfg.num_anchors
        self.feat_stride = cfg.feat_stride
        self.rpn_pre_nms_top_n = cfg.rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = cfg.rpn_post_nms_top_n
        self.rpn_nms_thresh = cfg.rpn_nms_thresh
        
        # 生成基础锚框
        self.base_anchors = self._generate_anchors()
        
    def generate_all_anchors(self, feat_map_shape):
        """
        在特征图上生成所有锚框
        
        Args:
            feat_map_shape: 特征图形状 [batch_size, height, width, channels]
            
        Returns:
            all_anchors: 所有锚框坐标 [num_anchors_total, 4]
        """
        # 这里代码存疑，没有考虑到批量
        batch_size = feat_map_shape[0]
        feat_h = feat_map_shape[1]
        feat_w = feat_map_shape[2]
        
        # 1. 生成特征图上的网格点
        shift_x = tf.cast(tf.range(0, feat_w) * self.feat_stride, tf.float32)  # [feat_w]
        shift_y = tf.cast(tf.range(0, feat_h) * self.feat_stride, tf.float32)  # [feat_h]
        
        # 2. 生成所有网格点组合
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y, indexing='ij')
        shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=-1)  # [feat_h, feat_w, 4]
        shifts = tf.reshape(shifts, [-1, 4])  # [feat_h*feat_w, 4]
        
        # 3. 将基础锚框与所有偏移量相加
        # base_anchors: [num_anchors, 4], shifts: [feat_h*feat_w, 4]
        all_anchors = tf.expand_dims(self.base_anchors, 0) + tf.expand_dims(shifts, 1)
        all_anchors = tf.reshape(all_anchors, [-1, 4])  # [feat_h*feat_w*num_anchors, 4]
        
        return all_anchors
    
    def compute_rpn_loss(self, rpn_cls_logits, rpn_bbox_pred, gt_boxes, gt_labels, feat_map_shape):
        """
        计算RPN损失
        
        Args:
            rpn_cls_logits: RPN分类logits [batch_size, num_anchors_total, 2]
            rpn_bbox_pred: RPN回归预测 [batch_size, feat_h, feat_w, 4*num_anchors]
            gt_boxes: 真实边界框 [batch_size, num_gt_boxes, 4]
            gt_labels: 真实标签 [batch_size, num_gt_boxes]
            feat_map_shape: 特征图形状
            
        Returns:
            rpn_cls_loss: 分类损失
            rpn_bbox_loss: 回归损失
            rpn_labels: 锚框标签 [batch_size, num_anchors_total]
            rpn_targets: 锚框回归目标 [batch_size, num_anchors_total, 4]
        """
        batch_size = tf.shape(rpn_cls_logits)[0]
        
        # 生成所有锚框
        all_anchors = self.generate_all_anchors(feat_map_shape)  # [num_anchors_total, 4]
        
        # 为每个batch计算标签和目标
        rpn_labels_list = []
        rpn_targets_list = []
        
        for i in range(batch_size):
            # 获取当前batch的真实框和标签
            batch_gt_boxes = gt_boxes[i]  # [num_gt_boxes, 4]
            batch_gt_labels = gt_labels[i]  # [num_gt_boxes]
            
            # 过滤掉背景框（标签为0）
            valid_mask = batch_gt_labels > 0
            batch_gt_boxes = tf.boolean_mask(batch_gt_boxes, valid_mask)
            
            if tf.shape(batch_gt_boxes)[0] == 0:
                # 如果没有有效目标，所有锚框都标记为背景
                batch_labels = tf.zeros(tf.shape(all_anchors)[0], dtype=tf.int32)
                batch_targets = tf.zeros(tf.shape(all_anchors), dtype=tf.float32)
            else:
                batch_labels, batch_targets = self._compute_anchor_targets(
                    all_anchors, batch_gt_boxes
                )
            
            rpn_labels_list.append(batch_labels)
            rpn_targets_list.append(batch_targets)
        
        # 堆叠所有batch的结果
        rpn_labels = tf.stack(rpn_labels_list, axis=0)  # [batch_size, num_anchors_total]
        rpn_targets = tf.stack(rpn_targets_list, axis=0)  # [batch_size, num_anchors_total, 4]
        
        # 计算分类损失
        rpn_cls_loss = self._compute_cls_loss(rpn_cls_logits, rpn_labels)
        
        # 计算回归损失
        rpn_bbox_loss = self._compute_bbox_loss(rpn_bbox_pred, rpn_targets, rpn_labels)
        
        return rpn_cls_loss, rpn_bbox_loss, rpn_labels, rpn_targets
    
    def _compute_anchor_targets(self, anchors, gt_boxes):
        """
        为锚框分配标签和回归目标
        
        Args:
            anchors: 锚框坐标 [num_anchors, 4]
            gt_boxes: 真实边界框 [num_gt_boxes, 4]
            
        Returns:
            labels: 锚框标签 [num_anchors]
            targets: 回归目标 [num_anchors, 4]
        """
        num_anchors = tf.shape(anchors)[0]
        num_gt_boxes = tf.shape(gt_boxes)[0]
        
        # 计算IoU矩阵
        ious = self._compute_iou_matrix(anchors, gt_boxes)  # [num_anchors, num_gt_boxes]
        
        # 找到每个锚框的最大IoU和对应的GT框
        max_ious = tf.reduce_max(ious, axis=1)  # [num_anchors]
        argmax_ious = tf.argmax(ious, axis=1)  # [num_anchors]
        
        # 分配标签
        labels = tf.zeros(num_anchors, dtype=tf.int32)  # 默认背景
        
        # 正样本：IoU > 0.7 或 最大IoU的锚框
        positive_mask = max_ious >= 0.7
        labels = tf.where(positive_mask, tf.ones_like(labels), labels)
        
        # 负样本：IoU < 0.3
        negative_mask = max_ious < 0.3
        labels = tf.where(negative_mask, tf.zeros_like(labels), labels)
        
        # 计算回归目标
        targets = self._compute_bbox_targets(anchors, gt_boxes, argmax_ious)
        
        return labels, targets
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """
        计算两组边界框之间的IoU矩阵
        
        Args:
            boxes1: [N, 4]
            boxes2: [M, 4]
            
        Returns:
            ious: [N, M]
        """
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
    
    def _compute_bbox_targets(self, anchors, gt_boxes, argmax_ious):
        """
        计算边界框回归目标
        
        Args:
            anchors: [num_anchors, 4]
            gt_boxes: [num_gt_boxes, 4]
            argmax_ious: [num_anchors] 每个锚框对应的最佳GT框索引
            
        Returns:
            targets: [num_anchors, 4]
        """
        # 获取每个锚框对应的GT框
        matched_gt_boxes = tf.gather(gt_boxes, argmax_ious)  # [num_anchors, 4]
        
        # 计算回归目标（中心点偏移和宽高缩放）
        targets = self._bbox_transform(anchors, matched_gt_boxes)
        
        return targets
    
    def _bbox_transform(self, boxes, gt_boxes):
        """
        计算边界框回归目标
        
        Args:
            boxes: [N, 4] (x1, y1, x2, y2)
            gt_boxes: [N, 4] (x1, y1, x2, y2)
            
        Returns:
            targets: [N, 4] (dx, dy, dw, dh)
        """
        # 计算中心点和宽高
        box_width = boxes[:, 2] - boxes[:, 0] + 1.0
        box_height = boxes[:, 3] - boxes[:, 1] + 1.0
        box_ctr_x = boxes[:, 0] + 0.5 * box_width
        box_ctr_y = boxes[:, 1] + 0.5 * box_height
        
        gt_width = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
        gt_height = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
        gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_width
        gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_height
        
        # 计算回归目标
        dx = (gt_ctr_x - box_ctr_x) / box_width
        dy = (gt_ctr_y - box_ctr_y) / box_height
        dw = tf.math.log(gt_width / box_width)
        dh = tf.math.log(gt_height / box_height)
        
        targets = tf.stack([dx, dy, dw, dh], axis=1)
        
        return targets
    
    def _compute_cls_loss(self, rpn_cls_logits, rpn_labels):
        """
        计算RPN分类损失
        
        Args:
            rpn_cls_logits: [batch_size, num_anchors_total, 2]
            rpn_labels: [batch_size, num_anchors_total]
            
        Returns:
            cls_loss: 标量损失
        """
        # 过滤掉忽略的样本（标签为-1）
        valid_mask = rpn_labels >= 0
        valid_logits = tf.boolean_mask(rpn_cls_logits, valid_mask)
        valid_labels = tf.boolean_mask(rpn_labels, valid_mask)
        
        if tf.shape(valid_logits)[0] == 0:
            return tf.constant(0.0)
        
        # 计算交叉熵损失
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=valid_labels, logits=valid_logits
        )
        
        return tf.reduce_mean(cls_loss)
    
    def _compute_bbox_loss(self, rpn_bbox_pred, rpn_targets, rpn_labels):
        """
        计算RPN回归损失
        
        Args:
            rpn_bbox_pred: [batch_size, feat_h, feat_w, 4*num_anchors]
            rpn_targets: [batch_size, num_anchors_total, 4]
            rpn_labels: [batch_size, num_anchors_total]
            
        Returns:
            bbox_loss: 标量损失
        """
        batch_size, feat_h, feat_w, _ = tf.unstack(tf.shape(rpn_bbox_pred))
        
        # 重塑预测结果
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [batch_size, feat_h * feat_w * self.num_anchors, 4])
        
        # 只计算正样本的回归损失
        positive_mask = rpn_labels == 1
        positive_pred = tf.boolean_mask(rpn_bbox_pred, positive_mask)
        positive_targets = tf.boolean_mask(rpn_targets, positive_mask)
        
        if tf.shape(positive_pred)[0] == 0:
            return tf.constant(0.0)
        
        # 计算smooth L1损失
        diff = tf.abs(positive_pred - positive_targets)
        bbox_loss = tf.where(
            diff < 1.0,
            0.5 * diff ** 2,
            diff - 0.5
        )
        
        return tf.reduce_mean(tf.reduce_sum(bbox_loss, axis=1))
    
    def generate_proposals(self, rpn_cls_prob, rpn_bbox_pred, feat_map_shape, im_info):
        """
        生成候选区域
        
        Args:
            rpn_cls_prob: RPN分类概率 [batch_size, num_anchors_total, 2]
            rpn_bbox_pred: RPN回归预测 [batch_size, feat_h, feat_w, 4*num_anchors]
            feat_map_shape: 特征图形状
            im_info: 图像信息 [batch_size, 3] (height, width, scale)
            
        Returns:
            proposals: 候选区域 [batch_size, num_proposals, 4]
            scores: 候选区域得分 [batch_size, num_proposals]
        """
        batch_size = tf.shape(rpn_cls_prob)[0]
        
        # 生成所有锚框
        all_anchors = self.generate_all_anchors(feat_map_shape)  # [num_anchors_total, 4]
        
        proposals_list = []
        scores_list = []
        
        for i in range(batch_size):
            # 获取前景概率
            fg_probs = rpn_cls_prob[i, :, 1]  # [num_anchors_total]
            
            # 重塑回归预测
            batch_bbox_pred = rpn_bbox_pred[i]  # [feat_h, feat_w, 4*num_anchors]
            feat_h, feat_w = tf.unstack(tf.shape(batch_bbox_pred)[:2])
            batch_bbox_pred = tf.reshape(batch_bbox_pred, [-1, 4])  # [num_anchors_total, 4]
            
            # 应用回归变换得到候选框
            proposals = self._apply_bbox_transform(all_anchors, batch_bbox_pred)
            
            # 裁剪到图像边界
            proposals = self._clip_boxes(proposals, im_info[i])
            
            # 过滤掉无效框
            valid_mask = self._filter_invalid_boxes(proposals)
            proposals = tf.boolean_mask(proposals, valid_mask)
            fg_probs = tf.boolean_mask(fg_probs, valid_mask)
            
            # 按得分排序
            sorted_indices = tf.argsort(fg_probs, direction='DESCENDING')
            proposals = tf.gather(proposals, sorted_indices)
            fg_probs = tf.gather(fg_probs, sorted_indices)
            
            # 应用NMS
            proposals, scores = self._apply_nms(proposals, fg_probs)
            
            proposals_list.append(proposals)
            scores_list.append(scores)
        
        # 堆叠结果
        proposals = tf.stack(proposals_list, axis=0)
        scores = tf.stack(scores_list, axis=0)
        
        return proposals, scores
    
    def _apply_bbox_transform(self, anchors, bbox_pred):
        """
        应用边界框回归变换
        
        Args:
            anchors: [N, 4] (x1, y1, x2, y2)
            bbox_pred: [N, 4] (dx, dy, dw, dh)
            
        Returns:
            proposals: [N, 4] (x1, y1, x2, y2)
        """
        # 计算锚框的中心点和宽高
        anchor_width = anchors[:, 2] - anchors[:, 0] + 1.0
        anchor_height = anchors[:, 3] - anchors[:, 1] + 1.0
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_width
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_height
        
        # 应用回归变换
        pred_ctr_x = anchor_ctr_x + bbox_pred[:, 0] * anchor_width
        pred_ctr_y = anchor_ctr_y + bbox_pred[:, 1] * anchor_height
        pred_width = anchor_width * tf.exp(bbox_pred[:, 2])
        pred_height = anchor_height * tf.exp(bbox_pred[:, 3])
        
        # 转换回边界框格式
        proposals = tf.stack([
            pred_ctr_x - 0.5 * pred_width,
            pred_ctr_y - 0.5 * pred_height,
            pred_ctr_x + 0.5 * pred_width - 1.0,
            pred_ctr_y + 0.5 * pred_height - 1.0
        ], axis=1)
        
        return proposals
    
    def _clip_boxes(self, boxes, im_info):
        """
        将边界框裁剪到图像边界内
        
        Args:
            boxes: [N, 4] (x1, y1, x2, y2)
            im_info: [3] (height, width, scale)
            
        Returns:
            clipped_boxes: [N, 4]
        """
        height, width = im_info[0], im_info[1]
        
        clipped_boxes = tf.stack([
            tf.clip_by_value(boxes[:, 0], 0, width - 1),
            tf.clip_by_value(boxes[:, 1], 0, height - 1),
            tf.clip_by_value(boxes[:, 2], 0, width - 1),
            tf.clip_by_value(boxes[:, 3], 0, height - 1)
        ], axis=1)
        
        return clipped_boxes
    
    def _filter_invalid_boxes(self, boxes):
        """
        过滤掉无效的边界框（宽或高小于阈值）
        
        Args:
            boxes: [N, 4] (x1, y1, x2, y2)
            
        Returns:
            valid_mask: [N] 布尔掩码
        """
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        
        min_size = 16.0  # 最小尺寸阈值
        valid_mask = (widths >= min_size) & (heights >= min_size)
        
        return valid_mask
    
    def _apply_nms(self, boxes, scores):
        """
        应用非极大值抑制
        
        Args:
            boxes: [N, 4] (x1, y1, x2, y2)
            scores: [N]
            
        Returns:
            keep_boxes: [M, 4]
            keep_scores: [M]
        """
        # 使用TensorFlow的NMS实现
        keep_indices = tf.image.non_max_suppression(
            boxes, scores, 
            max_output_size=self.rpn_post_nms_top_n,
            iou_threshold=self.rpn_nms_thresh
        )
        
        keep_boxes = tf.gather(boxes, keep_indices)
        keep_scores = tf.gather(scores, keep_indices)
        
        return keep_boxes, keep_scores
    
    def _generate_anchors(self, scales=cfg.anchor_scales, ratios=cfg.anchor_ratios):
        """生成基础锚框（兼容 tf.function 的向量化实现）"""
        scales = tf.constant(scales, dtype=tf.float32)  # 形状：[3]
        ratios = tf.constant(ratios, dtype=tf.float32)  # 形状：[3]

        # 基础锚框（中心点在(0,0)，面积为16x16）
        base_size = 16.0
        base_anchor = tf.constant([0, 0, base_size - 1, base_size - 1], dtype=tf.float32)  # [x1, y1, x2, y2]

        # 计算基础宽高和面积
        w = base_anchor[2] - base_anchor[0] + 1.0  # 16.0
        h = base_anchor[3] - base_anchor[1] + 1.0  # 16.0
        area = w * h  # 256.0

        # 1. 计算所有宽高比对应的宽和高（向量化操作，替代第一个循环）
        area_ratios = area / ratios  # 形状：[3]（每个ratio对应一个面积）
        ws = tf.sqrt(area_ratios)  # 形状：[3]（宽）
        hs = ws * ratios  # 形状：[3]（高）

        # 2. 生成 scales 和 ratios 的所有组合（3x3=9种）
        # 用 meshgrid 生成组合索引（替代嵌套循环）
        ratio_indices = tf.range(tf.size(ratios))  # [0,1,2]
        scale_indices = tf.range(tf.size(scales))  # [0,1,2]
        r_idx, s_idx = tf.meshgrid(ratio_indices, scale_indices, indexing='ij')  # 形状：[3,3]
        r_idx = tf.reshape(r_idx, [-1])  # [0,0,0,1,1,1,2,2,2]（9个元素）
        s_idx = tf.reshape(s_idx, [-1])  # [0,1,2,0,1,2,0,1,2]（9个元素）

        # 3. 按组合索引提取宽高并缩放（向量化）
        # tf.gather用于从输入 Tensor 中根据指定的索引选取元素
        ws_selected = tf.gather(ws, r_idx)  # 按ratio索引取宽，形状：[9]
        hs_selected = tf.gather(hs, r_idx)  # 按ratio索引取高，形状：[9]
        scales_selected = tf.gather(scales, s_idx)  # 按scale索引取尺度，形状：[9]

        # 缩放宽高（scale是放大倍数）
        ws_scaled = ws_selected * scales_selected  # 形状：[9]
        hs_scaled = hs_selected * scales_selected  # 形状：[9]

        # 4. 计算所有锚框的坐标（向量化）
        x_center = (base_anchor[0] + base_anchor[2]) / 2.0  # 7.5
        y_center = (base_anchor[1] + base_anchor[3]) / 2.0  # 7.5

        x1 = x_center - ws_scaled / 2.0
        y1 = y_center - hs_scaled / 2.0
        x2 = x_center + ws_scaled / 2.0 - 1.0  # -1确保包含边界像素
        y2 = y_center + hs_scaled / 2.0 - 1.0

        # 组合成锚框张量（形状：[9,4]）
        anchors = tf.stack([x1, y1, x2, y2], axis=1)

        return anchors
