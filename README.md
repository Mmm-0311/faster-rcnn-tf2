# Faster R-CNN TensorFlow 2.6 实现

这是一个从零重写的 Faster R-CNN 项目，基于 TensorFlow 2.6.0 和 Python 3.9，参考了 [dBeker/Faster-RCNN-TensorFlow-Python3](https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3) 项目。

## 🚀 项目特点

- **完全基于 TensorFlow 2.6.0**：使用 Eager Execution 和 tf.function 优化
- **模块化设计**：清晰的代码结构，易于理解和扩展
- **增量式实现**：逐步完成各个模块，便于学习和调试
- **VOC数据集支持**：完整支持 VOC2007/VOC2012 数据集
- **预训练权重**：支持 VGG16 预训练权重加载
- **完整训练管道**：包含训练、验证、检查点保存等功能
- **推理演示**：支持单张图像和批量推理

## 📁 项目结构

```
faster-rcnn-tf2/
├── configs/                 # 配置文件
│   ├── config.py           # 配置管理类
│   └── default.yaml        # 默认配置
├── datasets/               # 数据集处理
│   └── voc.py             # VOC数据集加载器
├── models/                 # 模型定义
│   ├── backbone/          # 主干网络
│   │   └── vgg16.py       # VGG16 + RPN
│   ├── roi_pooling.py     # RoI Pooling层
│   ├── head.py            # 检测头
│   ├── rpn.py             # RPN模块
│   └── faster_rcnn.py     # 完整Faster R-CNN模型
├── utils/                  # 工具函数
│   ├── bbox_utils.py      # 边界框工具
│   └── logger.py          # 日志工具
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── main.py               # 主程序
└── requirements.txt      # 依赖包
```

## 🛠️ 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装TensorFlow GPU版本（推荐）
pip install tensorflow-gpu==2.6.0
```

## 📊 数据集准备

1. 下载 VOC2007 数据集：
```bash
# 创建数据目录
mkdir -p data/VOCdevkit

# 下载VOC2007数据集
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# 解压
tar -xvf VOCtrainval_06-Nov-2007.tar -C data/VOCdevkit/
tar -xvf VOCtest_06-Nov-2007.tar -C data/VOCdevkit/
```

2. 数据集结构：
```
data/VOCdevkit/VOC2007/
├── Annotations/          # XML标注文件
├── ImageSets/Main/       # 训练/测试集划分
├── JPEGImages/          # 图像文件
└── ...
```

## 🏃‍♂️ 快速开始

### 1. 测试数据集加载
```bash
python main.py
```

### 2. 运行增量式编译测试
```bash
# 运行所有模块的测试（推荐）
python incremental_compile_test.py

# 或者单独测试某个模块
python -c "from incremental_compile_test import test_faster_rcnn; test_faster_rcnn()"
```

### 3. 训练模型
```bash
python train.py
```

### 4. 推理预测
```bash
# 单张图像推理
python inference.py --image path/to/image.jpg --output_dir results/

# 批量推理
python inference.py --image_dir path/to/images/ --output_dir results/
```

## 🔧 配置说明

主要配置参数在 `configs/default.yaml` 中：

```yaml
# 训练配置
train:
  batch_size: 1
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  max_iter: 70000

# 模型配置
model:
  backbone: "vgg16"
  num_classes: 21  # VOC数据集20类+背景
  rpn_pre_nms_top_n: 12000
  rpn_post_nms_top_n: 2000
  rpn_nms_thresh: 0.7

# 锚框配置
anchor:
  feat_stride: 16
  scales: [8, 16, 32]
  ratios: [0.5, 1, 2]
```

## 🏗️ 架构详解

### 1. 主干网络 (VGG16 + RPN)
- **特征提取**：VGG16 卷积层提取共享特征
- **RPN网络**：生成候选区域和RPN损失
- **预训练权重**：支持ImageNet预训练权重加载

### 2. RoI Pooling
- **固定尺寸输出**：将不同尺寸的RoI池化为7×7特征图
- **两种实现**：基础版本和优化版本（使用crop_and_resize）

### 3. 检测头
- **分类分支**：预测目标类别
- **回归分支**：精化边界框位置
- **损失计算**：分类损失 + 回归损失

### 4. 训练管道
- **优化器**：SGD with momentum
- **学习率调度**：指数衰减
- **梯度裁剪**：防止梯度爆炸
- **检查点保存**：自动保存和恢复

## 📈 训练监控

训练过程中会自动生成：
- **训练曲线**：损失变化趋势
- **检查点**：模型权重保存
- **演示结果**：检测结果可视化

## 🎯 性能优化

### TensorFlow 2.6 特性
- **Eager Execution**：便于调试和开发
- **tf.function**：自动图优化
- **混合精度**：可选的FP16训练
- **XLA编译**：加速计算

### 内存优化
- **梯度累积**：支持大batch训练
- **动态batch**：根据GPU内存调整
- **数据预取**：tf.data优化

## 🔍 调试技巧

### 1. 模块测试
```python
# 测试RPN模块
python test_rpn.py

# 测试边界框工具
python -c "from utils.bbox_utils import test_bbox_utils; test_bbox_utils()"
```

### 2. 可视化调试
```python
# 可视化数据集样本
from main import visualize_dataset_sample
visualize_dataset_sample(dataset, num_samples=2)
```

### 3. 梯度检查
```python
# 检查梯度
with tf.GradientTape() as tape:
    outputs = model(inputs, training=True)
    loss = outputs['total_loss']

gradients = tape.gradient(loss, model.trainable_variables)
for grad in gradients:
    print(f"梯度形状: {grad.shape}, 是否包含NaN: {tf.reduce_any(tf.math.is_nan(grad))}")
```

## 🐛 常见问题

### 1. 内存不足
- 减小batch_size
- 使用梯度累积
- 启用混合精度训练

### 2. 训练不收敛
- 检查学习率设置
- 验证数据预处理
- 确认损失函数计算

### 3. 推理速度慢
- 使用tf.function装饰器
- 启用XLA编译
- 优化NMS阈值

## 📚 参考资料

- [Faster R-CNN论文](https://arxiv.org/abs/1506.01497)
- [TensorFlow 2.6文档](https://www.tensorflow.org/)
- [VOC数据集](http://host.robots.ox.ac.uk/pascal/VOC/)
- [原项目参考](https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

感谢以下开源项目的启发：
- [dBeker/Faster-RCNN-TensorFlow-Python3](https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3)
- [TensorFlow官方实现](https://github.com/tensorflow/models)
- [PyTorch实现](https://github.com/pytorch/vision)

---

**注意**：这是一个教学和研究项目，主要用于学习Faster R-CNN的实现原理和TensorFlow 2的使用。在生产环境中使用前，请进行充分的测试和优化。

Faster R-CNN 迁移至 TensorFlow 2.6.0 版本