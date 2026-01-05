#!/home/u/ZHC/FusionReID-master/.venv/bin/python3.8
#coding: utf-8

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from config import cfg
from modeling import make_model
from torchvision import transforms as T
import time

'''
    #
    聚类内存管理类，用于管理重识别特征的聚类。

    属性:
        max_clusters (int): 最大簇数
        eps (float): 聚类半径 (1 - 相似度阈值)
        min_samples (int): 最小样本数
        update_rate (float): 簇更新率
        max_inactive_seconds (float): 最大不活动秒数
        fps (float): 视频帧率
'''
class ClusteredMemoryManager:
    def __init__(self, max_clusters=100, eps=0.10, min_samples=2,
        update_rate=0.2, max_inactive_seconds=3.0, fps=15):
        # 簇存储
        self.clusters = []  # 簇中心 [numpy array]
        self.cluster_ids = []  # 簇ID [int]
        self.cluster_sizes = []  # 簇大小 [int]
        self.last_active_time = []  # 最后活动时间戳 [float]

        # 超参数
        self.max_clusters = max_clusters
        self.eps = eps  # 聚类半径 (1 - 相似度阈值)
        self.min_samples = min_samples
        self.update_rate = update_rate
        self.max_inactive_seconds = max_inactive_seconds
        self.fps = fps

        # 统计信息
        self.total_features = 0
        self.total_clusters_created = 0
        self.memory_cleanup_count = 0

    def assign_feature(self, feature, timestamp=None):
        """分配特征到簇，返回簇ID"""
        if timestamp is None:
            timestamp = time.time()

        self.total_features += 1

        # 1. 计算与现有簇的相似度
        if self.clusters:
            similarities = self._calculate_similarities(feature)
            max_sim = max(similarities)
            max_idx = similarities.index(max_sim)

            # 2. 判断是否匹配现有簇
            if max_sim >= (1 - self.eps):
                # 更新现有簇
                self._update_cluster(max_idx, feature, timestamp)
                return self.cluster_ids[max_idx]

        # 3. 创建新簇
        return self._create_new_cluster(feature, timestamp)

    def _calculate_similarities(self, feature):
        """计算特征与所有簇中心的余弦相似度"""
        similarities = []
        for center in self.clusters:
            # 余弦相似度
            dot_product = np.dot(feature, center)
            norm_product = np.linalg.norm(feature) * np.linalg.norm(center)
            if norm_product == 0:
                similarity = 0
            else:
                similarity = dot_product / norm_product
            similarities.append(similarity)
        return similarities

    def _update_cluster(self, cluster_idx, new_feature, timestamp):
        """更新现有簇"""
        # 指数移动平均更新簇中心
        alpha = self.update_rate
        self.clusters[cluster_idx] = (
                alpha * new_feature + (1 - alpha) * self.clusters[cluster_idx]
        )

        # 更新统计信息
        self.cluster_sizes[cluster_idx] += 1
        self.last_active_time[cluster_idx] = timestamp

    def _create_new_cluster(self, feature, timestamp):
        """创建新簇"""
        self.total_clusters_created += 1

        # 1. 清理过期簇
        self._cleanup_memory(timestamp)

        # 2. 检查内存上限
        if len(self.clusters) >= self.max_clusters:
            # 合并最相似的两个簇
            self._merge_similar_clusters()

        # 3. 创建新簇
        new_id = max(self.cluster_ids, default=-1) + 1
        self.clusters.append(feature.copy())
        self.cluster_ids.append(new_id)
        self.cluster_sizes.append(1)
        self.last_active_time.append(timestamp)

        return new_id

    def _cleanup_memory(self, current_time):
        """清理过期簇"""
        if not self.clusters:
            return

        # 1. 计算簇置信度
        confidences = [self._calculate_confidence(i) for i in range(len(self.clusters))]

        # 2. 确定保留的簇
        active_indices = []
        for i in range(len(self.clusters)):
            time_inactive = current_time - self.last_active_time[i]
            confidence = confidences[i]

            # 保留条件：高置信度或最近活动
            if (confidence >= 0.4) or (time_inactive <= self.max_inactive_seconds):
                active_indices.append(i)

        # 3. 更新内存
        if len(active_indices) < len(self.clusters):
            self.memory_cleanup_count += 1
            self.clusters = [self.clusters[i] for i in active_indices]
            self.cluster_ids = [self.cluster_ids[i] for i in active_indices]
            self.cluster_sizes = [self.cluster_sizes[i] for i in active_indices]
            self.last_active_time = [self.last_active_time[i] for i in active_indices]

    def _calculate_confidence(self, cluster_idx):
        """计算簇置信度"""
        size = self.cluster_sizes[cluster_idx]
        time_inactive = time.time() - self.last_active_time[cluster_idx]

        # 基于大小和活动性的置信度
        size_confidence = size / (size + 5)  # 饱和函数
        time_confidence = np.exp(-0.5 * time_inactive)  # 指数衰减

        return 0.7 * size_confidence + 0.3 * time_confidence

    def _merge_similar_clusters(self):
        """合并最相似的两个簇"""
        if len(self.clusters) < 2:
            return

        # 1. 计算所有簇对的相似度
        max_similarity = -1
        merge_pair = (0, 1)

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                similarity = np.dot(self.clusters[i], self.clusters[j]) / (
                        np.linalg.norm(self.clusters[i]) * np.linalg.norm(self.clusters[j])
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    merge_pair = (i, j)

        # 2. 合并两个最相似的簇
        i, j = merge_pair
        if i > j:
            i, j = j, i

        # 加权平均合并
        weight_i = self.cluster_sizes[i]
        weight_j = self.cluster_sizes[j]
        new_center = (weight_i * self.clusters[i] + weight_j * self.clusters[j]) / (weight_i + weight_j)

        # 保留置信度更高的簇ID
        new_id = self.cluster_ids[i] if self._calculate_confidence(i) > self._calculate_confidence(j) else \
        self.cluster_ids[j]

        # 3. 更新簇列表
        self.clusters[i] = new_center
        self.cluster_sizes[i] = weight_i + weight_j
        self.cluster_ids[i] = new_id
        self.last_active_time[i] = max(self.last_active_time[i], self.last_active_time[j])

        # 4. 删除被合并的簇
        for lst in [self.clusters, self.cluster_ids, self.cluster_sizes, self.last_active_time]:
            lst.pop(j)

    def get_memory_stats(self):
        """获取内存统计信息"""
        return {
            'current_clusters': len(self.clusters),
            'total_features': self.total_features,
            'total_clusters_created': self.total_clusters_created,
            'memory_cleanup_count': self.memory_cleanup_count,
            'avg_cluster_size': np.mean(self.cluster_sizes) if self.cluster_sizes else 0,
            'memory_usage_kb': len(self.clusters) * self.clusters[0].nbytes / 1024 if self.clusters else 0
        }

class reid_node:
    def __init__(self):
        rospy.init_node('reid_node', anonymous=True)

        # 获取参数
        self.camera_topic = rospy.get_param('~camera_topic', '/feynman_camera/M1CF118G24070005/rgb/image_rect_color')
        
        self.video_topic = rospy.get_param('~video_topic', '/image_raw')
        
        self.output_image_topic = rospy.get_param('~output_image_topic', '/reid_node/result_image')
        
        self.result_topic = rospy.get_param('~result_topic', '/reid_node/result')

        self.bridge = CvBridge()

        # 加载模型
        self.load_models()
        rospy.loginfo("ReID Node started")

        # 初始化聚类内存管理器
        self.memory_manager = ClusteredMemoryManager(
            max_clusters=150,
            eps=0.20,
            update_rate=0.1,
            max_inactive_seconds=50.0,
            fps=30  # 默认30fps，实际会根据图像获取频率调整
        )

        # 订阅图像话题
        self.image_sub = rospy.Subscriber(self.video_topic, Image, self.image_callback)

        # 发布处理结果话题
        self.image_pub = rospy.Publisher(self.output_image_topic, Image, queue_size=10)
        self.result_pub = rospy.Publisher(self.result_topic, String, queue_size=10)

        # 初始化重识别结果列表
        self.reid_results = []

        rospy.loginfo("ReID Node initialized")

    def load_models(self):
        """加载YOLO和ReID模型"""
        # Load YOLO model
        self.yolo_model = YOLO("/home/u/catkin_ws/src/reid_ros_pkg/yolo11n.pt")  # Load an official Detect model

        # Load ReID model
        cfg.merge_from_file("/home/u/catkin_ws/src/reid_ros_pkg/configs/MSMT17/msmt_vitb12_res50_layer2.yml")  # Load your config file
        self.reid_model = make_model(cfg, num_class=1, camera_num=15, view_num=0)  # Initialize model with dummy values
        self.reid_model.load_param_ignore_classifier("/home/u/catkin_ws/src/reid_ros_pkg/FusionReID_180.pth")  # Load trained weights
        self.reid_model.eval()  # Set to evaluation mode

        # 确定设备并移动模型到相应设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reid_model.to(self.device)  # Move to GPU if available
        
        rospy.loginfo("Models loaded successfully")

    def image_callback(self, data):
        """处理接收到的图像数据"""
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # 显示接收到的原始图像
            # cv2.imshow("Received Image", cv_image)
            # cv2.waitKey(1)  # 处理GUI事件，等待1毫秒

            # 执行重识别处理
            annotated_image = self.process_reid(cv_image)

            # 发布处理后的图像
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_msg.header = data.header  # 保持原始消息头
            self.image_pub.publish(annotated_msg)

            # 发布结果信息
            result_msg = String()
            result_msg.data = f"Processed image with {len([r for r in getattr(self, 'reid_results', []) if r.get('reid_id') is not None])} re-identified objects"
            self.result_pub.publish(result_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def process_reid(self, frame):
        """执行重识别处理"""
        current_time = time.time()

        # 运行跟踪推理
        results = self.yolo_model.track(frame, show=False, classes=[0], persist=True, verbose=False)  # 不显示结果，但保留跟踪ID

        # 为每个检测框分配重识别ID
        self.reid_results = []
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # 裁剪图像
                cropped_img = frame[y1:y2, x1:x2]

                # 提取ReID特征
                feature = self.extract_reid_features_batch([cropped_img], self.reid_model, cfg, self.device)[0]
                
                # 使用聚类内存管理分配ID
                reid_id = self.memory_manager.assign_feature(feature, current_time)
                
                # 保存结果
                self.reid_results.append({
                    'bbox': (x1, y1, x2, y2),
                    'reid_id': reid_id,
                    'confidence': box.conf[0].cpu().numpy()
                })

        # 在帧上绘制重识别ID而不是跟踪ID
        annotated_frame = frame.copy()
        for result in self.reid_results:
            x1, y1, x2, y2 = result['bbox']
            reid_id = result['reid_id']
            confidence = result['confidence']

            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 在边界框上方显示重识别ID
            if reid_id is not None:
                label = f"ID: {reid_id}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated_frame
    
    def extract_reid_features_batch(self, images, reid_model, cfg, device):
        """批量提取重识别特征，使用与项目一致的处理方式"""
        # 预处理图像
        val_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(cfg.INPUT.SIZE_TEST),  # 调整图像大小
            T.ToTensor(),  # 转换为Tensor
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)  # 标准化
        ])

        # 批量处理图像
        batch_tensors = []
        for img in images:
            img_tensor = val_transforms(img)
            batch_tensors.append(img_tensor)

        # 堆叠成批次张量
        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            # 使用与inference_processor.py和test_net.py一致的特征提取方式
            batch_size = batch_tensor.size(0)
            cam_label = torch.zeros(batch_size, dtype=torch.long).to(device)
            feats = reid_model(batch_tensor, cam_label=cam_label, view_label=None)
            feats_np = feats.cpu().numpy()

        # 返回特征列表
        return [feat.flatten() for feat in feats_np]
    
'''
    # 
    聚类内存管理类，用于管理重识别特征的聚类。
    
    属性:
        max_clusters (int): 最大簇数
        eps (float): 聚类半径 (1 - 相似度阈值)
        min_samples (int): 最小样本数
        update_rate (float): 簇更新率
        max_inactive_seconds (float): 最大不活动秒数
        fps (float): 视频帧率
'''
class ClusteredMemoryManager:
    def __init__(self, max_clusters=100, eps=0.20, min_samples=2,
        update_rate=0.1, max_inactive_seconds=50.0, fps=15):
        # 簇存储
        self.clusters = []  # 簇中心 [numpy array]
        self.cluster_ids = []  # 簇ID [int]
        self.cluster_sizes = []  # 簇大小 [int]
        self.last_active_time = []  # 最后活动时间戳 [float]

        # 超参数
        self.max_clusters = max_clusters
        self.eps = eps  # 聚类半径 (1 - 相似度阈值)
        self.min_samples = min_samples
        self.update_rate = update_rate
        self.max_inactive_seconds = max_inactive_seconds
        self.fps = fps

        # 统计信息
        self.total_features = 0
        self.total_clusters_created = 0
        self.memory_cleanup_count = 0

    def assign_feature(self, feature, timestamp=None):
        """分配特征到簇，返回簇ID"""
        if timestamp is None:
            timestamp = time.time()

        self.total_features += 1

        # 1. 计算与现有簇的相似度
        if self.clusters:
            similarities = self._calculate_similarities(feature)
            max_sim = max(similarities)
            max_idx = similarities.index(max_sim)

            # 2. 判断是否匹配现有簇
            if max_sim >= (1 - self.eps):
                # 更新现有簇
                self._update_cluster(max_idx, feature, timestamp)
                return self.cluster_ids[max_idx]

        # 3. 创建新簇
        return self._create_new_cluster(feature, timestamp)

    def _calculate_similarities(self, feature):
        """计算特征与所有簇中心的余弦相似度"""
        similarities = []
        for center in self.clusters:
            # 余弦相似度
            dot_product = np.dot(feature, center)
            norm_product = np.linalg.norm(feature) * np.linalg.norm(center)
            if norm_product == 0:
                similarity = 0
            else:
                similarity = dot_product / norm_product
            similarities.append(similarity)
        return similarities

    def _update_cluster(self, cluster_idx, new_feature, timestamp):
        """更新现有簇"""
        # 指数移动平均更新簇中心
        alpha = self.update_rate
        self.clusters[cluster_idx] = (
                alpha * new_feature + (1 - alpha) * self.clusters[cluster_idx]
        )

        # 更新统计信息
        self.cluster_sizes[cluster_idx] += 1
        self.last_active_time[cluster_idx] = timestamp

    def _create_new_cluster(self, feature, timestamp):
        """创建新簇"""
        self.total_clusters_created += 1

        # 1. 清理过期簇
        self._cleanup_memory(timestamp)

        # 2. 检查内存上限
        if len(self.clusters) >= self.max_clusters:
            # 合并最相似的两个簇
            self._merge_similar_clusters()

        # 3. 创建新簇
        new_id = max(self.cluster_ids, default=-1) + 1
        self.clusters.append(feature.copy())
        self.cluster_ids.append(new_id)
        self.cluster_sizes.append(1)
        self.last_active_time.append(timestamp)

        return new_id

    def _cleanup_memory(self, current_time):
        """清理过期簇"""
        if not self.clusters:
            return

        # 1. 计算簇置信度
        confidences = [self._calculate_confidence(i) for i in range(len(self.clusters))]

        # 2. 确定保留的簇
        active_indices = []
        for i in range(len(self.clusters)):
            time_inactive = current_time - self.last_active_time[i]
            confidence = confidences[i]

            # 保留条件：高置信度或最近活动
            if (confidence >= 0.4) or (time_inactive <= self.max_inactive_seconds):
                active_indices.append(i)

        # 3. 更新内存
        if len(active_indices) < len(self.clusters):
            self.memory_cleanup_count += 1
            self.clusters = [self.clusters[i] for i in active_indices]
            self.cluster_ids = [self.cluster_ids[i] for i in active_indices]
            self.cluster_sizes = [self.cluster_sizes[i] for i in active_indices]
            self.last_active_time = [self.last_active_time[i] for i in active_indices]

    def _calculate_confidence(self, cluster_idx):
        """计算簇置信度"""
        size = self.cluster_sizes[cluster_idx]
        time_inactive = time.time() - self.last_active_time[cluster_idx]

        # 基于大小和活动性的置信度
        size_confidence = size / (size + 5)  # 饱和函数
        time_confidence = np.exp(-0.5 * time_inactive)  # 指数衰减

        return 0.7 * size_confidence + 0.3 * time_confidence

    def _merge_similar_clusters(self):
        """合并最相似的两个簇"""
        if len(self.clusters) < 2:
            return

        # 1. 计算所有簇对的相似度
        max_similarity = -1
        merge_pair = (0, 1)

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                similarity = np.dot(self.clusters[i], self.clusters[j]) / (
                        np.linalg.norm(self.clusters[i]) * np.linalg.norm(self.clusters[j])
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    merge_pair = (i, j)

        # 2. 合并两个最相似的簇
        i, j = merge_pair
        if i > j:
            i, j = j, i

        # 加权平均合并
        weight_i = self.cluster_sizes[i]
        weight_j = self.cluster_sizes[j]
        new_center = (weight_i * self.clusters[i] + weight_j * self.clusters[j]) / (weight_i + weight_j)

        # 保留置信度更高的簇ID
        new_id = self.cluster_ids[i] if self._calculate_confidence(i) > self._calculate_confidence(j) else \
        self.cluster_ids[j]

        # 3. 更新簇列表
        self.clusters[i] = new_center
        self.cluster_sizes[i] = weight_i + weight_j
        self.cluster_ids[i] = new_id
        self.last_active_time[i] = max(self.last_active_time[i], self.last_active_time[j])

        # 4. 删除被合并的簇
        for lst in [self.clusters, self.cluster_ids, self.cluster_sizes, self.last_active_time]:
            lst.pop(j)

    def get_memory_stats(self):
        """获取内存统计信息"""
        return {
            'current_clusters': len(self.clusters),
            'total_features': self.total_features,
            'total_clusters_created': self.total_clusters_created,
            'memory_cleanup_count': self.memory_cleanup_count,
            'avg_cluster_size': np.mean(self.cluster_sizes) if self.cluster_sizes else 0,
            'memory_usage_kb': len(self.clusters) * self.clusters[0].nbytes / 1024 if self.clusters else 0
        }

def assign_id_to_feature(feature, known_features, known_ids, threshold=0.9):
    """
    根据特征向量相似度为新特征分配ID，使用与inference_processor.py一致的逻辑
    """
    if len(known_features) == 0:
        return None

    # 计算与所有已知特征的余弦相似度，与inference_processor.py保持一致
    similarities = []
    for known_feat in known_features:
        # 计算余弦相似度
        dot_product = np.dot(feature, known_feat)
        norm_product = np.linalg.norm(feature) * np.linalg.norm(known_feat)
        if norm_product == 0:
            similarity = 0
        else:
            similarity = dot_product / norm_product
        similarities.append(similarity)

    # 找到最大相似度
    max_similarity = max(similarities)
    max_index = similarities.index(max_similarity)

    # 如果最大相似度超过阈值，则认为是同一个ID
    if max_similarity >= threshold:
        return known_ids[max_index]
    else:
        return None


if __name__ == '__main__':
    reid_node = reid_node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("ReID Node shutting down")
    finally:
        cv2.destroyAllWindows()