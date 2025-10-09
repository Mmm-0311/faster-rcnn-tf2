import tensorflow as tf
import numpy as np


def compute_iou(boxes1, boxes2):
    """
    计算两组边界框之间的IoU
    
    Args:
        boxes1: [N, 4] (x1, y1, x2, y2)
        boxes2: [M, 4] (x1, y1, x2, y2)
        
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


def bbox_transform(boxes, gt_boxes):
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


def bbox_transform_inv(boxes, bbox_pred):
    """
    应用边界框回归变换
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        bbox_pred: [N, 4] (dx, dy, dw, dh)
        
    Returns:
        proposals: [N, 4] (x1, y1, x2, y2)
    """
    # 计算边界框的中心点和宽高
    box_width = boxes[:, 2] - boxes[:, 0] + 1.0
    box_height = boxes[:, 3] - boxes[:, 1] + 1.0
    box_ctr_x = boxes[:, 0] + 0.5 * box_width
    box_ctr_y = boxes[:, 1] + 0.5 * box_height
    
    # 应用回归变换
    pred_ctr_x = box_ctr_x + bbox_pred[:, 0] * box_width
    pred_ctr_y = box_ctr_y + bbox_pred[:, 1] * box_height
    pred_width = box_width * tf.exp(bbox_pred[:, 2])
    pred_height = box_height * tf.exp(bbox_pred[:, 3])
    
    # 转换回边界框格式
    proposals = tf.stack([
        pred_ctr_x - 0.5 * pred_width,
        pred_ctr_y - 0.5 * pred_height,
        pred_ctr_x + 0.5 * pred_width - 1.0,
        pred_ctr_y + 0.5 * pred_height - 1.0
    ], axis=1)
    
    return proposals


def clip_boxes(boxes, im_info):
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


def filter_invalid_boxes(boxes, min_size=16.0):
    """
    过滤掉无效的边界框（宽或高小于阈值）
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        min_size: 最小尺寸阈值
        
    Returns:
        valid_mask: [N] 布尔掩码
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    
    valid_mask = (widths >= min_size) & (heights >= min_size)
    
    return valid_mask


def nms(boxes, scores, max_output_size=100, iou_threshold=0.5):
    """
    非极大值抑制
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        scores: [N]
        max_output_size: 最大输出数量
        iou_threshold: IoU阈值
        
    Returns:
        keep_indices: 保留的索引
    """
    keep_indices = tf.image.non_max_suppression(
        boxes, scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold
    )
    
    return keep_indices


def generate_anchors(base_size=16, scales=[8, 16, 32], ratios=[0.5, 1, 2]):
    """
    生成基础锚框
    
    Args:
        base_size: 基础尺寸
        scales: 尺度列表
        ratios: 宽高比列表
        
    Returns:
        anchors: [num_anchors, 4] (x1, y1, x2, y2)
    """
    scales = tf.constant(scales, dtype=tf.float32)
    ratios = tf.constant(ratios, dtype=tf.float32)
    
    # 基础锚框（中心点在(0,0)，面积为base_size x base_size）
    base_anchor = tf.constant([0, 0, base_size - 1, base_size - 1], dtype=tf.float32)
    
    # 计算基础宽高和面积
    w = base_anchor[2] - base_anchor[0] + 1.0
    h = base_anchor[3] - base_anchor[1] + 1.0
    area = w * h
    
    # 计算所有宽高比对应的宽和高
    area_ratios = area / ratios
    ws = tf.sqrt(area_ratios)
    hs = ws * ratios
    
    # 生成 scales 和 ratios 的所有组合
    ratio_indices = tf.range(tf.size(ratios))
    scale_indices = tf.range(tf.size(scales))
    r_idx, s_idx = tf.meshgrid(ratio_indices, scale_indices, indexing='ij')
    r_idx = tf.reshape(r_idx, [-1])
    s_idx = tf.reshape(s_idx, [-1])
    
    # 按组合索引提取宽高并缩放
    ws_selected = tf.gather(ws, r_idx)
    hs_selected = tf.gather(hs, r_idx)
    scales_selected = tf.gather(scales, s_idx)
    
    # 缩放宽高
    ws_scaled = ws_selected * scales_selected
    hs_scaled = hs_selected * scales_selected
    
    # 计算所有锚框的坐标
    x_center = (base_anchor[0] + base_anchor[2]) / 2.0
    y_center = (base_anchor[1] + base_anchor[3]) / 2.0
    
    x1 = x_center - ws_scaled / 2.0
    y1 = y_center - hs_scaled / 2.0
    x2 = x_center + ws_scaled / 2.0 - 1.0
    y2 = y_center + hs_scaled / 2.0 - 1.0
    
    # 组合成锚框张量
    anchors = tf.stack([x1, y1, x2, y2], axis=1)
    
    return anchors


def generate_all_anchors(feat_map_shape, base_anchors, feat_stride=16):
    """
    在特征图上生成所有锚框
    
    Args:
        feat_map_shape: 特征图形状 [batch_size, height, width, channels]
        base_anchors: 基础锚框 [num_anchors, 4]
        feat_stride: 特征图步长
        
    Returns:
        all_anchors: 所有锚框坐标 [num_anchors_total, 4]
    """
    batch_size, feat_h, feat_w, _ = feat_map_shape
    
    # 生成特征图上的网格点
    shift_x = tf.range(0, feat_w) * feat_stride
    shift_y = tf.range(0, feat_h) * feat_stride
    
    # 生成所有网格点组合
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y, indexing='ij')
    shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=-1)
    shifts = tf.reshape(shifts, [-1, 4])
    
    # 将基础锚框与所有偏移量相加
    all_anchors = tf.expand_dims(base_anchors, 0) + tf.expand_dims(shifts, 1)
    all_anchors = tf.reshape(all_anchors, [-1, 4])
    
    return all_anchors


def smooth_l1_loss(pred, target, sigma=1.0):
    """
    计算smooth L1损失
    
    Args:
        pred: 预测值
        target: 目标值
        sigma: 平滑参数
        
    Returns:
        loss: smooth L1损失
    """
    diff = tf.abs(pred - target)
    loss = tf.where(
        diff < 1.0 / sigma,
        0.5 * sigma * diff ** 2,
        diff - 0.5 / sigma
    )
    
    return loss


def compute_bbox_area(boxes):
    """
    计算边界框面积
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        
    Returns:
        areas: [N]
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    areas = widths * heights
    
    return areas


def compute_bbox_center(boxes):
    """
    计算边界框中心点
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        
    Returns:
        centers: [N, 2] (cx, cy)
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    
    cx = boxes[:, 0] + 0.5 * widths
    cy = boxes[:, 1] + 0.5 * heights
    
    centers = tf.stack([cx, cy], axis=1)
    
    return centers


def compute_bbox_size(boxes):
    """
    计算边界框尺寸
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        
    Returns:
        sizes: [N, 2] (width, height)
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    
    sizes = tf.stack([widths, heights], axis=1)
    
    return sizes


def bbox_to_center_form(boxes):
    """
    将边界框从角点形式转换为中心点形式
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        
    Returns:
        center_boxes: [N, 4] (cx, cy, w, h)
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    
    cx = boxes[:, 0] + 0.5 * widths
    cy = boxes[:, 1] + 0.5 * heights
    
    center_boxes = tf.stack([cx, cy, widths, heights], axis=1)
    
    return center_boxes


def center_form_to_bbox(center_boxes):
    """
    将边界框从中心点形式转换为角点形式
    
    Args:
        center_boxes: [N, 4] (cx, cy, w, h)
        
    Returns:
        boxes: [N, 4] (x1, y1, x2, y2)
    """
    cx, cy, w, h = tf.unstack(center_boxes, axis=1)
    
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w - 1.0
    y2 = cy + 0.5 * h - 1.0
    
    boxes = tf.stack([x1, y1, x2, y2], axis=1)
    
    return boxes


if __name__ == "__main__":
    print("请运行 python incremental_compile_test.py 进行完整测试")
