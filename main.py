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
    main()
