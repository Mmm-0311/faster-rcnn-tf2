import os
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # 全局强制 Eager 模式
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from configs.config import cfg
from datasets.voc import VOCDataset, VOC_CLASSES
import numpy as np
from models.backbone.vgg16 import VGG16Backbone, generate_anchors

def visualize_dataset_sample(dataset, num_samples=2):
    """可视化数据集中的样本"""
    for batch in dataset.take(num_samples):
        images, boxes, classes = batch

        for i in range(images.shape[0]):
            image = images[i].numpy()
            # 恢复图像（加回均值并转换为RGB）
            pixel_means = np.array([102.9801, 115.9465, 122.7717])  # BGR格式
            image += pixel_means
            image = image[:, :, [2, 1, 0]]  # BGR转RGB
            image = np.clip(image, 0, 255).astype(np.uint8)

            # 创建图像和轴
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(image)

            # 绘制边界框
            img_boxes = boxes[i].numpy()
            img_classes = classes[i].numpy()

            for box, cls in zip(img_boxes, img_classes):
                if cls == 0:  # 背景
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1, VOC_CLASSES[cls], color='r', fontsize=12)

            plt.show()


def test_backbone():
    # 1. 初始化模型
    model = VGG16Backbone()

    # 2. 生成测试输入（随机图像，模拟VOC图像尺寸）
    batch_size = 1
    H, W = 600, 800  # 典型VOC图像尺寸
    test_image = tf.random.uniform((batch_size, H, W, 3), dtype=tf.float32)  # 随机图像

    # 3. 前向传播测试
    outputs = model(test_image, training=False)
    model.load_pretrained_weights()

    # 4. 验证输出形状
    feat_map = outputs['feat_map']
    rpn_cls_prob = outputs['rpn_cls_prob']
    rpn_bbox_pred = outputs['rpn_bbox_pred']

    # 特征图尺寸应为输入的 1/16（VGG16 下采样 16 倍）
    assert feat_map.shape[1] == H // 16, f"特征图高度错误：{feat_map.shape[1]} vs {H // 16}"
    assert feat_map.shape[2] == W // 16, f"特征图宽度错误：{feat_map.shape[2]} vs {W // 16}"

    # RPN分类概率形状：[B, H_feat*W_feat*NUM_ANCHORS, 2]
    num_anchors_total = (H // 16) * (W // 16) * cfg.num_anchors
    assert rpn_cls_prob.shape == (batch_size, num_anchors_total, 2), f"RPN分类输出形状错误"

    # RPN边界框偏移形状：[B, H_feat, W_feat, 4*NUM_ANCHORS]
    assert rpn_bbox_pred.shape == (batch_size, H // 16, W // 16, 4 * cfg.num_anchors), f"RPN回归输出形状错误"

    # 5. 锚框生成测试
    anchors = generate_anchors()
    assert anchors.shape == (cfg.num_anchors, 4), f"锚框生成形状错误：{anchors.shape}"

    print("主干网络测试通过！")


def main():
    # 加载配置

    # 检查TensorFlow版本和GPU可用性
    # print(f"TensorFlow version: {tf.__version__}")
    # print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

    # 创建输出目录
    # os.makedirs(cfg.output.log_dir, exist_ok=True)
    # os.makedirs(cfg.output.checkpoint_dir, exist_ok=True)
    # os.makedirs(cfg.output.demo_dir, exist_ok=True)

    # 测试VOC数据集加载
    print("Loading VOC dataset...")
    train_dataset = VOCDataset(split='train').get_dataset(batch_size=2)

    # 测试parse_xml执行流程
    # init_dataset = VOCDataset(cfg, split='train')
    # boxes, classes = init_dataset.parse_xml("./data/VOCdevkit\\VOC2007", '000005')

    # 可视化样本
    visualize_dataset_sample(train_dataset)

    print("Dataset test completed successfully!")


if __name__ == "__main__":
    # main()
    test_backbone()
