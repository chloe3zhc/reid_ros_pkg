#!/home/u/ZHC/FusionReID-master/.venv/bin/python3.8
#coding: utf-8

'''
reid_ros_11.py文件
设置节点 reid_node_0 
订阅话题
/feynman_camera/M1CF118G24070001/rgb/image_rect_color
/video_pub_node/image_raw_0
发布话题
/reid_node_0/result_image
/reid_node_0/result
/reid_node_0/cluster_data
'''

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
from clusteredmemory import ClusteredMemoryManager

class reid_node_0:
    def __init__(self):
        rospy.init_node('reid_node_0', anonymous=True)

        # 获取参数
        # 相机图像话题
        self.camera_topic = rospy.get_param('~camera_topic', '/feynman_camera/M1CF118G24070001/rgb/image_rect_color')
        # 本地视频话题
        self.video_topic = rospy.get_param('~video_topic', '/video_pub_node/image_raw_0')
        # 重识别输出图像话题
        self.output_image_topic = rospy.get_param('~output_image_topic', '/reid_node_0/result_image')
        # ByteTrack跟踪输出图像话题
        self.bytetrack_output_image_topic = rospy.get_param('~bytetrack_output_image_topic', '/reid_node_0/bytetrack_result_image')
        # 重识别结果话题
        self.result_topic = rospy.get_param('~result_topic', '/reid_node_0/result')
        # 聚类数据话题
        self.cluster_data_topic = rospy.get_param('~cluster_data_topic', '/reid_node_0/cluster_data')

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
        self.camera_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.image_sub = rospy.Subscriber(self.video_topic, Image, self.image_callback)

        # 发布处理结果话题
        self.image_pub = rospy.Publisher(self.output_image_topic, Image, queue_size=10)
        self.bytetrack_image_pub = rospy.Publisher(self.bytetrack_output_image_topic, Image, queue_size=10)
        self.result_pub = rospy.Publisher(self.result_topic, String, queue_size=10)
        self.cluster_data_pub = rospy.Publisher(self.cluster_data_topic, String, queue_size=10)

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

            # 执行重识别处理
            reid_annotated_image = self.process_reid(cv_image)

            # 执行ByteTrack跟踪处理
            bytetrack_annotated_image = self.process_bytetrack(cv_image)

            # 发布重识别处理后的图像
            reid_annotated_msg = self.bridge.cv2_to_imgmsg(reid_annotated_image, "bgr8")
            reid_annotated_msg.header = data.header  # 保持原始消息头
            self.image_pub.publish(reid_annotated_msg)

            # 发布ByteTrack跟踪处理后的图像
            bytetrack_annotated_msg = self.bridge.cv2_to_imgmsg(bytetrack_annotated_image, "bgr8")
            bytetrack_annotated_msg.header = data.header  # 保持原始消息头
            self.bytetrack_image_pub.publish(bytetrack_annotated_msg)

            # 发布结果信息
            result_msg = String()
            result_msg.data = f"Processed image with {len([r for r in getattr(self, 'reid_results', []) if r.get('reid_id') is not None])} re-identified objects"
            self.result_pub.publish(result_msg)

            # 发布聚类数据
            self.publish_cluster_data()

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    # 添加发布聚类数据的方法：
    def publish_cluster_data(self):
        """发布聚类数据"""
        try:
            # 获取聚类管理器中的数据
            cluster_data = {
                'clusters': [cluster.tolist() for cluster in self.memory_manager.clusters],
                'cluster_ids': self.memory_manager.cluster_ids,
                'current_time': time.time()
            }
            
            # 将聚类数据转换为JSON字符串并发布
            cluster_data_json = json.dumps(cluster_data, ensure_ascii=False)
            cluster_msg = String()
            cluster_msg.data = cluster_data_json
            self.cluster_data_pub.publish(cluster_msg)
            
        except Exception as e:
            rospy.logerr(f"Error publishing cluster data: {e}")

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

    def process_bytetrack(self, frame):
        """执行ByteTrack目标跟踪处理"""
        # 运行ByteTrack推理 - 使用bytetrack配置文件
        results = self.yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", show=False, classes=[0], verbose=False)

        # 在帧上绘制跟踪结果
        annotated_frame = frame.copy()
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # 获取跟踪ID（如果存在）
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else 0
                confidence = float(box.conf[0].cpu().numpy())

                # 绘制边界框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # 在边界框上方显示跟踪ID
                label = f"BT: {track_id}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return annotated_frame


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
    reid_node_0()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("ReID Node shutting down")
    finally:
        cv2.destroyAllWindows()