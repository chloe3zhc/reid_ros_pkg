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
import os
import yaml
from clusteredmemory import ClusteredMemoryManager

class ReIDProcessor:
    def __init__(self, config, bridge):
        self.config = config
        self.bridge = bridge
        # 加载模型
        self.load_models()
        # 初始化聚类内存管理器
        self.memory_manager = ClusteredMemoryManager(
            max_clusters=self.config['max_clusters'],
            eps=self.config['eps'],
            min_samples=self.config['min_samples'],
            update_rate=self.config['update_rate'],
            max_inactive_seconds=self.config['max_inactive_seconds'],
            fps=self.config['fps']
        )
        # 初始化重识别结果列表
        self.reid_results = []
        print("ReIDProcessor initialized")
    
    """加载YOLO和ReID模型"""
    def load_models(self):
        # Load YOLO model
        self.yolo_model = YOLO(self.config['yolo_model_path'])  # Load an official Detect model
        # Load ReID model
        cfg.merge_from_file(self.config['reid_config_path'])  # Load your config file
        self.reid_model = make_model(cfg, num_class=1, camera_num=15, view_num=0)  # Initialize model with dummy values
        self.reid_model.load_param_ignore_classifier(self.config['reid_model_path'])  # Load trained weights
        self.reid_model.eval()  # Set to evaluation mode
        # 确定设备并移动模型到相应设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reid_model.to(self.device)  # Move to GPU if available
        
        print("Models loaded successfully")
    """执行重识别处理"""
    def process_reid(self, frame):
        current_time = time.time()
        # 运行跟踪推理
        results = self.yolo_model.track(frame, show=False, classes=[0], persist=True, verbose=False)  # 不显示结果，但保留跟踪ID
        # 为每个检测框分配重识别ID
        self.reid_results = []
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # 确保检测到的是人，避免衣服等也被检测
                if box.conf[0].cpu().numpy() > self.config['bbox_conf_threshold']:
                    # 确保坐标在图像范围内
                    x1 = max(0, x1)  # 确保左上角x坐标不小于0
                    y1 = max(0, y1)  # 确保左上角y坐标不小于0
                    x2 = min(frame.shape[1], x2)  # 确保右下角x坐标不超过图像宽度
                    y2 = min(frame.shape[0], y2)  # 确保右下角y坐标不超过图像高度
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
    """批量提取重识别特征，使用与项目一致的处理方式"""
    def extract_reid_features_batch(self, images, reid_model, cfg, device):
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
            batch_size = batch_tensor.size(0)
            cam_label = torch.zeros(batch_size, dtype=torch.long).to(device)
            feats = reid_model(batch_tensor, cam_label=cam_label, view_label=None)
            feats_np = feats.cpu().numpy()
        # 返回特征列表
        return [feat.flatten() for feat in feats_np]