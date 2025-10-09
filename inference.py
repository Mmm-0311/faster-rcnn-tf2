import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from configs.config import cfg
from datasets.voc import VOC_CLASSES
from models.faster_rcnn import FasterRCNN
from utils.logger import setup_logger


class InferenceManager:
    """
    推理管理器
    
    负责：
    1. 模型加载
    2. 图像预处理
    3. 推理预测
    4. 结果后处理
    5. 可视化输出
    """
    
    def __init__(self, checkpoint_path=None):
        self.logger = setup_logger('inference')
        
        # 初始化模型
        self.model = FasterRCNN()
        
        # 加载检查点
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            # 加载预训练权重
            self.model.load_pretrained_weights()
            self.logger.info("使用预训练权重")
        
        # 图像预处理参数
        self.pixel_means = np.array(cfg.pixel_means, dtype=np.float32)  # BGR格式
        
        # 推理参数
        self.score_thresh = 0.05
        self.nms_thresh = 0.3
        self.max_detections = 100
        
        self.logger.info("推理管理器初始化完成")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(checkpoint_path)
        self.logger.info(f"检查点已加载: {checkpoint_path}")
    
    def preprocess_image(self, image_path):
        """
        图像预处理
        
        Args:
            image_path: 图像路径
            
        Returns:
            processed_image: 预处理后的图像
            original_image: 原始图像
            im_info: 图像信息
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_image = image.copy()
        
        # 转换为float32
        image = image.astype(np.float32)
        
        # 减去均值
        image = image - self.pixel_means
        
        # 转换为TensorFlow张量
        processed_image = tf.constant(image, dtype=tf.float32)
        processed_image = tf.expand_dims(processed_image, 0)  # 添加batch维度
        
        # 图像信息
        height, width = original_image.shape[:2]
        im_info = tf.constant([[height, width, 1.0]], dtype=tf.float32)
        
        return processed_image, original_image, im_info
    
    def predict(self, image_path):
        """
        预测单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            detections: 检测结果
            original_image: 原始图像
        """
        self.logger.info(f"预测图像: {image_path}")
        
        # 预处理图像
        processed_image, original_image, im_info = self.preprocess_image(image_path)
        
        # 推理
        detections = self.model.predict(
            processed_image,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh
        )
        
        # 移除batch维度
        detections = detections[0]  # [num_detections, 6]
        
        self.logger.info(f"检测到 {len(detections)} 个目标")
        
        return detections, original_image
    
    def visualize_detections(self, image, detections, save_path=None, show=True):
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果 [num_detections, 6]
            save_path: 保存路径
            show: 是否显示图像
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图像
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # 绘制检测结果
        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, score, cls = detection
                
                if score > self.score_thresh:
                    # 绘制边界框
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 添加标签
                    label = f'{VOC_CLASSES[int(cls)]}: {score:.2f}'
                    ax.text(x1, y1 - 5, label, color='red', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('Faster R-CNN Detection Results')
        ax.axis('off')
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"结果图像已保存: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()
    
    def batch_predict(self, image_dir, output_dir=None):
        """
        批量预测
        
        Args:
            image_dir: 图像目录
            output_dir: 输出目录
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))
        
        self.logger.info(f"找到 {len(image_files)} 张图像")
        
        # 批量预测
        for i, image_path in enumerate(image_files):
            try:
                # 预测
                detections, original_image = self.predict(image_path)
                
                # 可视化
                if output_dir:
                    filename = os.path.basename(image_path)
                    name, ext = os.path.splitext(filename)
                    save_path = os.path.join(output_dir, f'{name}_detection{ext}')
                    
                    self.visualize_detections(
                        original_image, detections, 
                        save_path=save_path, show=False
                    )
                
                self.logger.info(f"处理完成: {i+1}/{len(image_files)}")
                
            except Exception as e:
                self.logger.error(f"处理图像失败 {image_path}: {str(e)}")
    
    def evaluate_on_dataset(self, dataset_path, output_dir=None):
        """
        在数据集上评估
        
        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 这里可以实现更详细的评估指标
        # 暂时使用简单的准确率计算
        
        # 获取测试图像列表
        test_images = []
        for file in os.listdir(dataset_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(dataset_path, file))
        
        self.logger.info(f"评估 {len(test_images)} 张测试图像")
        
        total_detections = 0
        for image_path in test_images:
            try:
                detections, _ = self.predict(image_path)
                total_detections += len(detections)
            except Exception as e:
                self.logger.error(f"评估图像失败 {image_path}: {str(e)}")
        
        avg_detections = total_detections / len(test_images) if test_images else 0
        self.logger.info(f"平均每张图像检测到 {avg_detections:.2f} 个目标")
        
        return {'avg_detections': avg_detections}
    
    def set_inference_params(self, score_thresh=None, nms_thresh=None, max_detections=None):
        """
        设置推理参数
        
        Args:
            score_thresh: 得分阈值
            nms_thresh: NMS阈值
            max_detections: 最大检测数量
        """
        if score_thresh is not None:
            self.score_thresh = score_thresh
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        if max_detections is not None:
            self.max_detections = max_detections
        
        self.logger.info(f"推理参数更新: score_thresh={self.score_thresh}, "
                        f"nms_thresh={self.nms_thresh}, max_detections={self.max_detections}")


def demo_inference():
    """演示推理功能"""
    print("=== Faster R-CNN 推理演示 ===")
    
    # 创建推理管理器
    inferencer = InferenceManager()
    
    # 设置推理参数
    inferencer.set_inference_params(score_thresh=0.1, nms_thresh=0.3)
    
    # 创建测试图像（如果没有真实图像）
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        # 创建随机测试图像
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, test_image)
        print(f"创建测试图像: {test_image_path}")
    
    try:
        # 预测
        detections, original_image = inferencer.predict(test_image_path)
        
        # 可视化结果
        inferencer.visualize_detections(original_image, detections, show=False)
        
        print(f"检测到 {len(detections)} 个目标")
        for i, detection in enumerate(detections):
            if len(detection) >= 6:
                x1, y1, x2, y2, score, cls = detection
                print(f"目标 {i+1}: {VOC_CLASSES[int(cls)]} (得分: {score:.3f}) "
                      f"位置: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        print("✅ 推理演示完成！")
        
    except Exception as e:
        print(f"推理演示失败: {str(e)}")
    
    finally:
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)


def main():
    """主推理函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Faster R-CNN Inference')
    parser.add_argument('--checkpoint', type=str, help='检查点路径')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录路径')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--score_thresh', type=float, default=0.05, help='得分阈值')
    parser.add_argument('--nms_thresh', type=float, default=0.3, help='NMS阈值')
    
    args = parser.parse_args()
    
    # 创建推理管理器
    inferencer = InferenceManager(checkpoint_path=args.checkpoint)
    
    # 设置推理参数
    inferencer.set_inference_params(
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh
    )
    
    if args.image:
        # 单张图像推理
        detections, original_image = inferencer.predict(args.image)
        inferencer.visualize_detections(
            original_image, detections, 
            save_path=args.output_dir, show=True
        )
    
    elif args.image_dir:
        # 批量推理
        inferencer.batch_predict(args.image_dir, args.output_dir)
    
    else:
        # 演示推理
        demo_inference()


if __name__ == "__main__":
    # 设置TensorFlow
    tf.config.run_functions_eagerly(True)
    
    # 运行演示
    demo_inference()
