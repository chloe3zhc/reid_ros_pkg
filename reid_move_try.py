#!/home/u/ZHC/FusionReID-master/.venv/bin/python3.8
#coding: utf-8


# reid_move_try.py文件
# 目的：构建外观-运动双分支MOT框架
# 设置节点 feature_check_node 
# 订阅话题
# /feynman_camera/M1CF118G24070001/rgb/image_rect_color
# /video_pub_node/image_raw_0
# 发布话题
# /reid_node_0/result_image
# /reid_node_0/result
# /reid_node_0/cluster_data


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
import os
import yaml
from clusteredmemory import ClusteredMemoryManager
from ReIDProcessor import ReIDProcessor


class feature_check_node:
    def __init__(self):
        rospy.init_node('reid_move_node', anonymous=True)

        # 获取当前脚本所在目录，获取配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.yaml')

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # 创建CvBridge实例
        self.bridge = CvBridge()

        self.processor = ReIDProcessor(self.config, self.bridge)

        # 获取参数
        # 相机图像话题
        self.camera_topic = rospy.get_param('~camera_topic', '/feynman_camera/M1CF118G24070001/rgb/image_rect_color')
        # 本地视频话题
        self.video_topic = rospy.get_param('~video_topic', '/video_pub_node/image_raw_0')
        # 重识别输出图像话题
        self.output_image_topic = rospy.get_param('~output_image_topic', '/reid_move_node/result_image')

        # 订阅图像话题
        self.camera_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.image_sub = rospy.Subscriber(self.video_topic, Image, self.image_callback)

        # 发布处理结果话题
        self.image_pub = rospy.Publisher(self.output_image_topic, Image, queue_size=10)

        rospy.loginfo("ReID Node initialized")

    def image_callback(self, data):
        """处理接收到的图像数据"""
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # 执行重识别处理
            annotated_image = self.processor.process_reid(cv_image)

            # 发布处理后的图像
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_msg.header = data.header  # 保持原始消息头
            self.image_pub.publish(annotated_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    feature_check_node()
    rospy.spin()