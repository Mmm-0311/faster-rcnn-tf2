import tensorflow as tf
import numpy as np
from configs.config import cfg


class ROIPooling(tf.keras.layers.Layer):
    """
    RoI Pooling层实现
    
    功能：
    1. 将不同尺寸的RoI区域池化为固定尺寸的特征图
    2. 支持批量处理多个RoI
    3. 使用最大池化操作
    """
    
    def __init__(self, pool_size=7, **kwargs):
        super(ROIPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        
    def call(self, inputs):
        """
        执行RoI Pooling操作
        
        Args:
            inputs: 包含以下元素的元组
                - feat_map: 特征图 [batch_size, height, width, channels]
                - rois: RoI坐标 [batch_size, num_rois, 4] (x1, y1, x2, y2)
                
        Returns:
            pooled_features: 池化后的特征 [batch_size, num_rois, pool_size, pool_size, channels]
        """
        feat_map, rois = inputs
        
        batch_size = tf.shape(feat_map)[0]
        num_rois = tf.shape(rois)[1]
        channels = tf.shape(feat_map)[3]
        
        # 为每个batch和每个RoI执行池化
        pooled_features_list = []
        
        for b in range(batch_size):
            batch_feat_map = feat_map[b]  # [height, width, channels]
            batch_rois = rois[b]  # [num_rois, 4]
            
            batch_pooled = self._pool_rois_batch(batch_feat_map, batch_rois)
            pooled_features_list.append(batch_pooled)
        
        # 堆叠所有batch的结果
        pooled_features = tf.stack(pooled_features_list, axis=0)
        
        return pooled_features
    
    def _pool_rois_batch(self, feat_map, rois):
        """
        对单个batch的RoI进行池化
        
        Args:
            feat_map: [height, width, channels]
            rois: [num_rois, 4] (x1, y1, x2, y2)
            
        Returns:
            pooled: [num_rois, pool_size, pool_size, channels]
        """
        num_rois = tf.shape(rois)[0]
        channels = tf.shape(feat_map)[2]
        
        pooled_list = []
        
        for i in range(num_rois):
            roi = rois[i]  # [4]
            roi_pooled = self._pool_single_roi(feat_map, roi, stride=16)
            pooled_list.append(roi_pooled)
        
        # 堆叠所有RoI的结果
        pooled = tf.stack(pooled_list, axis=0)
        
        return pooled
    
    def _pool_single_roi(self, feat_map, roi, stride=16):
        """
        对单个RoI进行池化
        
        Args:
            feat_map: [height, width, channels]
            roi: [4] (x1, y1, x2, y2) - 原始图像坐标
            stride: 下采样倍数，默认16
            
        Returns:
            pooled: [pool_size, pool_size, channels]
        """
        # 提取RoI坐标
        x1, y1, x2, y2 = tf.unstack(roi)
        
        # 将原始图像坐标映射到特征图坐标
        feat_x1 = x1 / stride
        feat_y1 = y1 / stride
        feat_x2 = x2 / stride
        feat_y2 = y2 / stride
        
        # 确保坐标在有效范围内
        feat_height = tf.cast(tf.shape(feat_map)[0], tf.float32)
        feat_width = tf.cast(tf.shape(feat_map)[1], tf.float32)
        channels = tf.shape(feat_map)[2]
        
        # 裁剪坐标到特征图边界
        feat_x1 = tf.clip_by_value(feat_x1, 0, feat_width - 1)
        feat_y1 = tf.clip_by_value(feat_y1, 0, feat_height - 1)
        feat_x2 = tf.clip_by_value(feat_x2, feat_x1 + 1, feat_width)
        feat_y2 = tf.clip_by_value(feat_y2, feat_y1 + 1, feat_height)
        
        # 计算RoI的宽高
        roi_width = feat_x2 - feat_x1
        roi_height = feat_y2 - feat_y1
        
        # 计算每个池化窗口的尺寸
        bin_width = roi_width / self.pool_size
        bin_height = roi_height / self.pool_size
        
        # 执行池化操作
        pooled_regions = []
        
        for ph in range(self.pool_size):
            for pw in range(self.pool_size):
                # 计算当前池化窗口的坐标
                start_x = feat_x1 + pw * bin_width
                start_y = feat_y1 + ph * bin_height
                end_x = feat_x1 + (pw + 1) * bin_width
                end_y = feat_y1 + (ph + 1) * bin_height
                
                # 转换为整数坐标
                start_x = tf.cast(tf.floor(start_x), tf.int32)
                start_y = tf.cast(tf.floor(start_y), tf.int32)
                end_x = tf.cast(tf.math.ceil(end_x), tf.int32)
                end_y = tf.cast(tf.math.ceil(end_y), tf.int32)
                
                # 确保坐标不超出边界
                start_x = tf.maximum(0, start_x)
                start_y = tf.maximum(0, start_y)
                end_x = tf.minimum(tf.cast(feat_width, tf.int32), end_x)
                end_y = tf.minimum(tf.cast(feat_height, tf.int32), end_y)
                
                # 提取区域并执行最大池化
                if end_x > start_x and end_y > start_y:
                    region = feat_map[start_y:end_y, start_x:end_x, :]
                    pooled_value = tf.reduce_max(region, axis=[0, 1])  # [channels]
                else:
                    # 如果区域为空，使用零填充
                    pooled_value = tf.zeros(tf.shape(feat_map)[2], dtype=feat_map.dtype)
                
                pooled_regions.append(pooled_value)
        
        # 重塑为 [pool_size, pool_size, channels]
        pooled = tf.stack(pooled_regions, axis=0)
        pooled = tf.reshape(pooled, [self.pool_size, self.pool_size, channels])
        
        return pooled


class ROIPoolingV2(tf.keras.layers.Layer):
    """
    优化版本的RoI Pooling层
    
    使用TensorFlow的crop_and_resize操作实现更高效的RoI Pooling
    """
    
    def __init__(self, pool_size=7, **kwargs):
        super(ROIPoolingV2, self).__init__(**kwargs)
        self.pool_size = pool_size
        
    def call(self, inputs):
        """
        使用crop_and_resize实现RoI Pooling
        
        Args:
            inputs: 包含以下元素的元组
                - feat_map: 特征图 [batch_size, height, width, channels]
                - rois: RoI坐标 [batch_size, num_rois, 4] (x1, y1, x2, y2)
                
        Returns:
            pooled_features: 池化后的特征 [batch_size, num_rois, pool_size, pool_size, channels]
        """
        feat_map, rois = inputs
        
        batch_size = tf.shape(feat_map)[0]
        num_rois = tf.shape(rois)[1]
        channels = tf.shape(feat_map)[3]
        
        # 将RoI坐标归一化到[0, 1]范围
        feat_height = tf.cast(tf.shape(feat_map)[1], tf.float32)
        feat_width = tf.cast(tf.shape(feat_map)[2], tf.float32)
        
        # 归一化RoI坐标
        normalized_rois = tf.stack([
            rois[:, :, 1] / feat_height,  # y1
            rois[:, :, 0] / feat_width,   # x1
            rois[:, :, 3] / feat_height,  # y2
            rois[:, :, 2] / feat_width    # x2
        ], axis=-1)  # [batch_size, num_rois, 4]
        
        # 为每个batch执行crop_and_resize
        pooled_features_list = []
        
        for b in range(batch_size):
            batch_feat_map = feat_map[b]  # [height, width, channels]
            batch_rois = normalized_rois[b]  # [num_rois, 4]
            
            # 使用crop_and_resize进行RoI Pooling
            batch_pooled = tf.image.crop_and_resize(
                tf.expand_dims(batch_feat_map, 0),  # [1, height, width, channels]
                batch_rois,  # [num_rois, 4]
                tf.zeros(num_rois, dtype=tf.int32),  # 所有RoI都属于batch 0
                [self.pool_size, self.pool_size]  # 输出尺寸
            )  # [num_rois, pool_size, pool_size, channels]
            
            pooled_features_list.append(batch_pooled)
        
        # 堆叠所有batch的结果
        pooled_features = tf.stack(pooled_features_list, axis=0)
        
        return pooled_features


if __name__ == "__main__":
    print("请运行 python incremental_compile_test.py 进行完整测试")
