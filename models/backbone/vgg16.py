import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from configs.config import cfg
from tensorflow.keras.applications import VGG16


class VGG16Backbone(Model):
    def __init__(self):  # 21类（VOC2007：20类+背景）
        super(VGG16Backbone, self).__init__()
        self.num_classes = cfg.num_classes
        self.feat_stride = cfg.feat_stride

        # -------------------------- 1. 特征提取网络（VGG16 卷积层） --------------------------
        # conv1: 2层3x3卷积 + 2x2池化（步长2）
        self.conv1_1 = layers.Conv2D(64, (3, 3), padding='same', name='conv1_1')
        self.conv1_2 = layers.Conv2D(64, (3, 3), padding='same', name='conv1_2')
        self.pool1 = layers.MaxPooling2D((2, 2), strides=2, name='pool1')

        # conv2: 2层3x3卷积 + 2x2池化
        self.conv2_1 = layers.Conv2D(128, (3, 3), padding='same', name='conv2_1')
        self.conv2_2 = layers.Conv2D(128, (3, 3), padding='same', name='conv2_2')
        self.pool2 = layers.MaxPooling2D((2, 2), strides=2, name='pool2')

        # conv3: 3层3x3卷积 + 2x2池化
        self.conv3_1 = layers.Conv2D(256, (3, 3), padding='same', name='conv3_1')
        self.conv3_2 = layers.Conv2D(256, (3, 3), padding='same', name='conv3_2')
        self.conv3_3 = layers.Conv2D(256, (3, 3), padding='same', name='conv3_3')
        self.pool3 = layers.MaxPooling2D((2, 2), strides=2, name='pool3')

        # conv4: 3层3x3卷积 + 2x2池化
        self.conv4_1 = layers.Conv2D(512, (3, 3), padding='same', name='conv4_1')
        self.conv4_2 = layers.Conv2D(512, (3, 3), padding='same', name='conv4_2')
        self.conv4_3 = layers.Conv2D(512, (3, 3), padding='same', name='conv4_3')
        self.pool4 = layers.MaxPooling2D((2, 2), strides=2, name='pool4')

        # conv5: 3层3x3卷积（无池化，输出共享特征图）
        self.conv5_1 = layers.Conv2D(512, (3, 3), padding='same', name='conv5_1')
        self.conv5_2 = layers.Conv2D(512, (3, 3), padding='same', name='conv5_2')
        self.conv5_3 = layers.Conv2D(512, (3, 3), padding='same', name='conv5_3')

        # self.vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        # # 冻结 VGG16 的权重
        # self.vgg16.trainable = False

        # -------------------------- 2. RPN 网络 --------------------------
        # RPN 共享卷积层
        self.rpn_conv = layers.Conv2D(512, (3, 3), padding='same', name='rpn_conv')
        # RPN 分类分支（前景/背景）
        self.rpn_cls = layers.Conv2D(2 * cfg.num_anchors, (1, 1), name='rpn_cls')  # 2类
        # RPN 回归分支（边界框偏移）
        self.rpn_bbox = layers.Conv2D(4 * cfg.num_anchors, (1, 1), name='rpn_bbox')  # 4个坐标

        # 初始化权重（匹配原仓库的初始化方式）
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化卷积层权重（参考原仓库的截断正态分布）"""
        for layer in self.layers:
            if isinstance(layer, layers.Conv2D):
                layer.kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.01)
                layer.bias_initializer = tf.keras.initializers.Constant(0.0)
        # self.load_pretrained_weights()

    def call(self, inputs, training=False):
        """前向传播：输入图像 → 特征图 → RPN输出"""
        x = inputs  # 输入形状：[batch_size, H, W, 3]

        # 1. 特征提取（conv1-conv5）
        x = tf.nn.relu(self.conv1_1(x))
        x = tf.nn.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = tf.nn.relu(self.conv2_1(x))
        x = tf.nn.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = tf.nn.relu(self.conv3_1(x))
        x = tf.nn.relu(self.conv3_2(x))
        x = tf.nn.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = tf.nn.relu(self.conv4_1(x))
        x = tf.nn.relu(self.conv4_2(x))
        x = tf.nn.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = tf.nn.relu(self.conv5_1(x))
        x = tf.nn.relu(self.conv5_2(x))
        feat_map = tf.nn.relu(self.conv5_3(x))  # 共享特征图

        # 2. RPN 前向传播
        rpn_conv_out = tf.nn.relu(self.rpn_conv(feat_map))
        rpn_cls_logits = self.rpn_cls(rpn_conv_out)  # 分类logits：[B, H_feat, W_feat, 2*cfg.num_anchors]
        rpn_bbox_pred = self.rpn_bbox(rpn_conv_out)  # 边界框偏移：[B, H_feat, W_feat, 4*cfg.num_anchors]

        # 调整RPN分类输出形状（便于计算softmax）
        # 从 [B, H, W, 2*N] 转为 [B, H*W*N, 2]
        rpn_cls_logits = tf.reshape(rpn_cls_logits,
                                    [-1, tf.shape(rpn_cls_logits)[1] * tf.shape(rpn_cls_logits)[2] * cfg.num_anchors, 2])
        rpn_cls_prob = tf.nn.softmax(rpn_cls_logits, axis=-1)  # 前景概率

        return {
            'feat_map': feat_map,  # 共享特征图
            'rpn_cls_logits': rpn_cls_logits,  # RPN分类logits
            'rpn_cls_prob': rpn_cls_prob,  # RPN分类概率
            'rpn_bbox_pred': rpn_bbox_pred  # RPN边界框偏移
        }

    def load_pretrained_weights(self):
        # 加载预训练VGG16（仅保留卷积层，不包含顶部全连接层）
        pretrained_vgg = VGG16(weights='imagenet', include_top=False)
        # print("预训练权重形状：", [w.shape for w in pretrained_vgg])
        for layer in pretrained_vgg.layers:
            if layer.name == 'block1_conv1':  # 官方 VGG16 的第一个卷积层名称
                print("官方层权重形状：", [w.shape for w in layer.get_weights()])
        layer_mapping = {
            self.conv1_1: 'block1_conv1',
            self.conv1_2: 'block1_conv2',
            self.conv2_1: 'block2_conv1',
            self.conv2_2: 'block2_conv2',
            self.conv3_1: 'block3_conv1',
            self.conv3_2: 'block3_conv2',
            self.conv3_3: 'block3_conv3',
            self.conv4_1: 'block4_conv1',
            self.conv4_2: 'block4_conv2',
            self.conv4_3: 'block4_conv3',
            self.conv5_1: 'block5_conv1',
            self.conv5_2: 'block5_conv2',
            self.conv5_3: 'block5_conv3'
        }
        # 加载权重到当前模型
        for custom_layer, pretrained_layer_name in layer_mapping.items():
            # 从预训练模型获取对应层的权重
            pretrained_weights = pretrained_vgg.get_layer(pretrained_layer_name).get_weights()
            # 设置到自定义层
            custom_layer.set_weights(pretrained_weights)
            # 固定权重（不参与训练更新）
            custom_layer.trainable = False


# -------------------------- 辅助工具函数（锚框生成） --------------------------
@tf.function
def generate_anchors(scales=cfg.anchor_scales, ratios=cfg.anchor_ratios):
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