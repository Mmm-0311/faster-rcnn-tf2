import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import cv2
import numpy as np
from configs.config import cfg

# VOC数据集类别
VOC_CLASSES = (
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


class VOCDataset:
    def __init__(self, split='train'):
        self.split = split
        # self.data_root = os.path.join(cfg.voc_root, 'VOCdevkit')
        self.data_root = cfg.voc_root
        self.image_sets = self._get_image_sets()
        self.num_classes = len(VOC_CLASSES)
        self.pixel_means = np.array(cfg.pixel_means, dtype=np.float32)  # BGR格式均值

    def _get_image_sets(self):
        """获取训练/测试数据集的图像ID列表"""
        image_sets = []
        if self.split == 'train':
            sets = cfg.train_sets
        else:
            sets = cfg.test_sets

        for dataset in sets:
            _, year, split = dataset.split('_')
            voc_path = os.path.join(self.data_root, f'VOC{year}')
            image_set_file = os.path.join(voc_path, 'ImageSets', 'Main', f'{split}.txt')
            with open(image_set_file, 'r') as f:
                for line in f:
                    image_id = line.strip()
                    image_sets.append((voc_path, image_id))
        return image_sets

    def _parse_xml(self, voc_path, image_id):
        """解析XML标注文件，返回边界框和类别"""
        xml_path = os.path.join(voc_path,'Annotations', f'{image_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        boxes = []
        classes = []

        for obj in root.iter('object'):
            # 解析类别
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in VOC_CLASSES:
                continue
            cls_idx = VOC_CLASSES.index(cls_name)

            # 解析边界框
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1  # 转换为0-based坐标
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            # 确保边界框有效
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            boxes.append([x1, y1, x2, y2])
            classes.append(cls_idx)

        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)

    def _load_image(self, voc_path, image_id):
        """加载图像并转换为BGR格式"""
        image_path = os.path.join(voc_path, 'JPEGImages', f'{image_id}.jpg')
        image = cv2.imread(image_path)  # OpenCV默认读取为BGR格式
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return image.astype(np.float32)

    def _preprocess(self, image, boxes):
        """图像预处理：减去均值"""
        image = image - self.pixel_means
        return image, boxes

    def _augment(self, image, boxes):
        """数据增强：随机水平翻转"""
        if tf.random.uniform(()) > 0.5:
            # 水平翻转图像
            image = tf.image.flip_left_right(image)

            # 调整边界框坐标
            width = tf.cast(tf.shape(image)[1], tf.float32) #(height, width, channels)
            x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
            new_x1 = width - x2
            new_x2 = width - x1
            boxes = tf.concat([new_x1, y1, new_x2, y2], axis=1)

        return image, boxes

    def get_dataset(self, batch_size=1, shuffle=True):
        """创建TF Dataset对象"""

        def generator():
            """
                一张图像的像素数据
                该图像中所有目标的边界框（BBox）
                每个目标对应的类别标签
            """
            for voc_path, image_id in self.image_sets:
                # 加载图像和标注
                image = self._load_image(voc_path, image_id)
                boxes, classes = self._parse_xml(voc_path, image_id)

                # 过滤无标注的图像
                if len(boxes) == 0:
                    continue

                yield image, boxes, classes

        # 创建数据集
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        # iterator = iter(dataset)
        # sample = next(iterator)  # 这一行会触发生成器执行，此时可单步进入生成器函数
        # 数据增强（仅训练集）
        if self.split == 'train':
            # 这里map后返回的是个新数据集，因为有部分数据可能以1/2概率被翻转了，用了"="将源数据集覆盖了
            dataset = dataset.map(
                # 为每一个数据应用lambda函数
                lambda img, boxes, cls: tf.py_function(
                    self._augment, # 指定函数
                    [img, boxes], # 指定参数
                    [tf.float32, tf.float32] # 指定数据类型
                ) + [cls],
                # 指定并行处理的数量：tf.data.AUTOTUNE 会让 TensorFlow 根据系统资源（如 CPU 核心数）自动调整并行度，加速数据预处理流程
                num_parallel_calls=tf.data.AUTOTUNE
            )
            # 扩充数据集可以使用dataset.concatenate()

        # 预处理
        dataset = dataset.map(
            lambda img, boxes, cls: (*self._preprocess(img, boxes), cls),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 打乱和批处理
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_sets))

        # 注意：由于图像和边界框尺寸可变，需要使用padded_batch
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [None, None, 3],  # 图像
                [None, 4],  # 边界框
                [None]  # 类别
            ),
            drop_remainder=True
        )

        return dataset.prefetch(tf.data.AUTOTUNE)