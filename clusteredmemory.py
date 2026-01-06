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
import json



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
        update_rate=0, max_inactive_seconds=50.0, fps=15):
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

    def add_feature_as_cluster(self, feature, cluster_id=None):
        """将特征作为新簇添加，指定簇ID"""
        timestamp = time.time()

        self.total_clusters_created += 1
        self.clusters.append(feature.copy())
        self.cluster_ids.append(cluster_id)
        self.cluster_sizes.append(1)
        self.last_active_time.append(timestamp)

        return cluster_id

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