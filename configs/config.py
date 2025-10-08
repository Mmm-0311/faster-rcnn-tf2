# config.py
import yaml
import os


class Config:
    def __init__(self, config_path: str = "configs/default.yaml"):
        """初始化配置加载器，解析yaml配置文件"""
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在：{os.path.abspath(config_path)}")

        # 加载yaml配置
        with open(config_path, "r", encoding="utf-8") as f:
            self._raw_config = yaml.safe_load(f)  # 原始配置字典

        # 解析各部分配置
        self.parse_train_config()
        self.parse_model_config()
        self.parse_data_config()
        self.parse_output_config()
        self.parse_anchor_config()

    def parse_train_config(self):
        """解析训练相关配置"""
        train_cfg = self._raw_config.get("train", {})

        # 训练批次大小
        self.batch_size = train_cfg.get("batch_size", 1)
        # 初始学习率
        self.learning_rate = train_cfg.get("learning_rate", 0.001)
        # 动量（用于SGD优化器）
        self.momentum = train_cfg.get("momentum", 0.9)
        # 权重衰减（L2正则化）
        self.weight_decay = train_cfg.get("weight_decay", 0.0005)
        # 最大迭代次数
        self.max_iter = train_cfg.get("max_iter", 70000)
        # 学习率衰减步数
        self.step_size = train_cfg.get("step_size", 30000)
        # 学习率衰减因子
        self.gamma = train_cfg.get("gamma", 0.1)

    def parse_model_config(self):
        """解析模型相关配置"""
        model_cfg = self._raw_config.get("model", {})

        # 主干网络类型（如vgg16、resnet50等）
        self.backbone = model_cfg.get("backbone", "vgg16")
        # 类别数（含背景）
        self.num_classes = model_cfg.get("num_classes", 21)
        # RPN非极大值抑制前保留的候选框数量（训练阶段）
        self.rpn_pre_nms_top_n = model_cfg.get("rpn_pre_nms_top_n", 12000)
        # RPN非极大值抑制后保留的候选框数量（训练阶段）
        self.rpn_post_nms_top_n = model_cfg.get("rpn_post_nms_top_n", 2000)
        # RPN非极大值抑制阈值
        self.rpn_nms_thresh = model_cfg.get("rpn_nms_thresh", 0.7)
        # RoI池化输出尺寸（固定为7x7）
        self.roi_pooling_size = model_cfg.get("roi_pooling_size", 7)

    def parse_data_config(self):
        """解析数据相关配置"""
        data_cfg = self._raw_config.get("data", {})

        # VOC数据集根目录
        self.voc_root = data_cfg.get("voc_root", "./data/VOCdevkit")
        # 图像像素均值（BGR格式，用于归一化）
        self.pixel_means = data_cfg.get("pixel_means", [102.9801, 115.9465, 122.7717])
        # 训练集名称列表
        self.train_sets = data_cfg.get("train_sets", ["VOC_2007_trainval"])
        # 测试集名称列表
        self.test_sets = data_cfg.get("test_sets", ["VOC_2007_test"])

    def parse_output_config(self):
        """解析输出相关配置"""
        output_cfg = self._raw_config.get("output", {})

        # 日志保存目录
        self.log_dir = output_cfg.get("log_dir", "./logs")
        # 模型检查点保存目录
        self.checkpoint_dir = output_cfg.get("checkpoint_dir", "./checkpoints")
        # 演示结果保存目录
        self.demo_dir = output_cfg.get("demo_dir", "./demo")

        # 确保输出目录存在
        for dir_path in [self.log_dir, self.checkpoint_dir, self.demo_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def parse_anchor_config(self):
        """解析锚框相关配置"""
        anchor_cfg = self._raw_config.get("anchor", {})

        # 特征图相对于输入图像的步长（下采样倍数）
        self.feat_stride = anchor_cfg.get("feat_stride", 16)
        # 锚框尺度（基础尺寸的倍数）
        self.anchor_scales = anchor_cfg.get("scales", [8, 16, 32])
        # 锚框宽高比
        self.anchor_ratios = anchor_cfg.get("ratios", [0.5, 1, 2])
        # 每个特征图位置的锚框数量（尺度数 × 宽高比数）
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

    def __repr__(self) -> str:
        """打印配置信息（便于调试）"""
        return f"Config(\n" \
               f"  train: batch_size={self.batch_size}, lr={self.learning_rate}, max_iter={self.max_iter}\n" \
               f"  model: backbone={self.backbone}, num_classes={self.num_classes}\n" \
               f"  data: voc_root={self.voc_root}, train_sets={self.train_sets}\n" \
               f"  anchor: feat_stride={self.feat_stride}, num_anchors={self.num_anchors}\n" \
               f")"


# 全局配置实例（项目中其他模块直接导入此实例使用）
cfg = Config()