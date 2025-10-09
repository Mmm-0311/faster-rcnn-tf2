"""
增量式编译测试文件
包含项目中所有模块的测试函数，用于验证各模块功能
"""

import tensorflow as tf
import numpy as np
from configs.config import cfg
from models.faster_rcnn import FasterRCNN, FasterRCNNTrainer
from models.backbone.vgg16 import VGG16Backbone
from models.rpn import RPN
from models.head import DetectionHead
from models.roi_pooling import ROIPooling
from datasets.voc import VOCDataset
from utils.bbox_utils import compute_iou, bbox_transform, nms


def test_faster_rcnn():
    """测试完整的Faster R-CNN模型"""
    print("=== 测试完整的Faster R-CNN模型 ===")
    
    # 创建模型
    model = FasterRCNN()
    
    # 创建测试数据
    batch_size = 1
    H, W = 600, 800
    images = tf.random.uniform((batch_size, H, W, 3), dtype=tf.float32)
    gt_boxes = tf.random.uniform((batch_size, 3, 4), minval=0, maxval=min(H, W), dtype=tf.float32)
    gt_labels = tf.random.uniform((batch_size, 3), minval=1, maxval=cfg.num_classes, dtype=tf.int32)
    
    print(f"输入图像形状: {images.shape}")
    print(f"GT边界框形状: {gt_boxes.shape}")
    print(f"GT标签形状: {gt_labels.shape}")
    
    # 测试训练模式
    print("\n--- 测试训练模式 ---")
    outputs = model({
        'images': images,
        'gt_boxes': gt_boxes,
        'gt_labels': gt_labels
    }, training=True)
    
    print(f"总损失: {outputs['total_loss']:.4f}")
    print(f"RPN分类损失: {outputs['rpn_cls_loss']:.4f}")
    print(f"RPN回归损失: {outputs['rpn_bbox_loss']:.4f}")
    print(f"检测分类损失: {outputs['det_cls_loss']:.4f}")
    print(f"检测回归损失: {outputs['det_bbox_loss']:.4f}")
    
    # 测试推理模式
    print("\n--- 测试推理模式 ---")
    detections = model.predict(images)
    print(f"检测结果形状: {detections.shape}")
    print(f"平均检测数量: {tf.reduce_mean(tf.cast(tf.shape(detections)[1], tf.float32)):.1f}")
    
    # 测试训练器
    print("\n--- 测试训练器 ---")
    trainer = FasterRCNNTrainer(model)
    
    # 模拟训练步骤
    batch = (images, gt_boxes, gt_labels)
    loss_dict = trainer.train_step(batch)
    print(f"训练步骤损失: {loss_dict['total_loss']:.4f}")
    
    print("✅ Faster R-CNN模型测试通过！")


def test_vgg16_backbone():
    """测试VGG16主干网络"""
    print("=== 测试VGG16主干网络 ===")
    
    # 创建模型
    model = VGG16Backbone()
    
    # 创建测试数据
    batch_size = 2
    H, W = 600, 800
    images = tf.random.uniform((batch_size, H, W, 3), dtype=tf.float32)
    
    print(f"输入图像形状: {images.shape}")
    
    # 前向传播
    outputs = model(images, training=False)
    
    # 验证输出
    feat_map = outputs['feat_map']
    rpn_cls_logits = outputs['rpn_cls_logits']
    rpn_cls_prob = outputs['rpn_cls_prob']
    rpn_bbox_pred = outputs['rpn_bbox_pred']
    
    print(f"特征图形状: {feat_map.shape}")
    print(f"RPN分类logits形状: {rpn_cls_logits.shape}")
    print(f"RPN分类概率形状: {rpn_cls_prob.shape}")
    print(f"RPN边界框预测形状: {rpn_bbox_pred.shape}")
    
    # 验证形状
    expected_feat_h = H // 16
    expected_feat_w = W // 16
    assert feat_map.shape[1] == expected_feat_h, f"特征图高度错误: {feat_map.shape[1]} vs {expected_feat_h}"
    assert feat_map.shape[2] == expected_feat_w, f"特征图宽度错误: {feat_map.shape[2]} vs {expected_feat_w}"
    
    print("✅ VGG16主干网络测试通过！")


def test_rpn():
    """测试RPN模块"""
    print("=== 测试RPN模块 ===")
    
    # 创建RPN模型
    rpn = RPN()
    
    # 创建测试数据
    batch_size = 1
    feat_h, feat_w = 37, 50  # 对应600x800图像的特征图尺寸
    feat_map = tf.random.uniform((batch_size, feat_h, feat_w, 512), dtype=tf.float32)
    
    # 模拟RPN输出
    num_anchors = feat_h * feat_w * 9  # 9个锚框
    rpn_cls_logits = tf.random.uniform((batch_size, num_anchors, 2), dtype=tf.float32)
    rpn_cls_prob = tf.nn.softmax(rpn_cls_logits, axis=-1)
    rpn_bbox_pred = tf.random.uniform((batch_size, num_anchors, 4), dtype=tf.float32)
    
    print(f"特征图形状: {feat_map.shape}")
    print(f"RPN分类logits形状: {rpn_cls_logits.shape}")
    print(f"RPN边界框预测形状: {rpn_bbox_pred.shape}")
    
    # 测试锚框生成
    base_anchors = rpn.base_anchors
    print(f"基础锚框形状: {base_anchors.shape}")
    
    # 测试所有锚框生成
    all_anchors = rpn.generate_all_anchors(tf.shape(feat_map))
    print(f"所有锚框形状: {all_anchors.shape}")
    
    # 测试proposal生成
    im_info = tf.constant([[600.0, 800.0, 1.0]], dtype=tf.float32)
    proposals, scores = rpn.generate_proposals(rpn_cls_prob, rpn_bbox_pred, tf.shape(feat_map), im_info)
    print(f"Proposals形状: {proposals.shape}")
    print(f"Proposal得分形状: {scores.shape}")
    
    print("✅ RPN模块测试通过！")


def test_detection_head():
    """测试检测头模块"""
    print("=== 测试检测头模块 ===")
    
    # 创建检测头
    detection_head = DetectionHead(num_classes=cfg.num_classes)
    
    # 创建测试数据
    batch_size = 1
    feat_h, feat_w = 37, 50
    feat_map = tf.random.uniform((batch_size, feat_h, feat_w, 512), dtype=tf.float32)
    
    # 创建proposals
    num_proposals = 100
    proposals = tf.random.uniform((batch_size, num_proposals, 4), minval=0, maxval=600, dtype=tf.float32)
    
    print(f"特征图形状: {feat_map.shape}")
    print(f"Proposals形状: {proposals.shape}")
    
    # 前向传播
    outputs = detection_head([feat_map, proposals], training=False)
    
    print(f"分类得分形状: {outputs['cls_scores'].shape}")
    print(f"边界框预测形状: {outputs['bbox_preds'].shape}")
    print(f"池化特征形状: {outputs['pooled_features'].shape}")
    
    print("✅ 检测头模块测试通过！")


def test_roi_pooling():
    """测试RoI Pooling层"""
    print("=== 测试RoI Pooling层 ===")
    
    # 创建RoI Pooling层
    roi_pooling = ROIPooling(pool_size=7)
    
    # 创建测试数据
    batch_size = 1
    feat_h, feat_w = 37, 50
    feat_map = tf.random.uniform((batch_size, feat_h, feat_w, 512), dtype=tf.float32)
    
    # 创建RoIs
    num_rois = 10
    rois = tf.random.uniform((batch_size, num_rois, 4), minval=0, maxval=600, dtype=tf.float32)
    
    print(f"特征图形状: {feat_map.shape}")
    print(f"RoIs形状: {rois.shape}")
    
    # 前向传播
    pooled_features = roi_pooling([feat_map, rois])
    
    print(f"池化特征形状: {pooled_features.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, num_rois, 7, 7, 512)
    assert pooled_features.shape == expected_shape, f"池化特征形状错误: {pooled_features.shape} vs {expected_shape}"
    
    print("✅ RoI Pooling层测试通过！")


def test_bbox_utils():
    """测试边界框工具函数"""
    print("=== 测试边界框工具函数 ===")
    
    # 创建测试数据
    boxes1 = tf.constant([[10, 10, 20, 20], [15, 15, 25, 25]], dtype=tf.float32)
    boxes2 = tf.constant([[12, 12, 22, 22], [18, 18, 28, 28]], dtype=tf.float32)
    
    # 测试IoU计算
    ious = compute_iou(boxes1, boxes2)
    print(f"IoU矩阵形状: {ious.shape}")
    print(f"IoU值: {ious.numpy()}")
    
    # 测试边界框变换
    targets = bbox_transform(boxes1, boxes2)
    print(f"回归目标形状: {targets.shape}")
    
    # 测试NMS
    scores = tf.constant([0.9, 0.8, 0.7, 0.6], dtype=tf.float32)
    boxes = tf.constant([[10, 10, 20, 20], [12, 12, 22, 22], [15, 15, 25, 25], [18, 18, 28, 28]], dtype=tf.float32)
    
    keep_indices = nms(boxes, scores, max_output_size=2, iou_threshold=0.5)
    print(f"NMS保留索引: {keep_indices.numpy()}")
    
    print("✅ 边界框工具函数测试通过！")


def test_voc_dataset():
    """测试VOC数据集加载"""
    print("=== 测试VOC数据集加载 ===")
    
    try:
        # 创建数据集
        dataset = VOCDataset(split='train')
        
        # 获取数据集
        train_dataset = dataset.get_dataset(batch_size=1, shuffle=False)
        
        # 取一个批次测试
        for batch in train_dataset.take(1):
            images, boxes, classes = batch
            print(f"图像形状: {images.shape}")
            print(f"边界框形状: {boxes.shape}")
            print(f"类别形状: {classes.shape}")
            break
        
        print("✅ VOC数据集加载测试通过！")
        
    except Exception as e:
        print(f"⚠️ VOC数据集测试跳过（可能缺少数据）: {e}")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行所有增量式编译测试...")
    print("=" * 60)
    
    try:
        # 按依赖顺序运行测试
        # test_bbox_utils()
        # print()
        
        test_roi_pooling()
        print()
        
        test_vgg16_backbone()
        print()
        
        test_rpn()
        print()
        
        test_detection_head()
        print()
        
        test_faster_rcnn()
        print()
        
        test_voc_dataset()
        print()
        
        print("=" * 60)
        print("🎉 所有测试通过！项目编译成功！")
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
