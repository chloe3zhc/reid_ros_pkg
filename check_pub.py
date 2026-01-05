#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import time
from std_msgs.msg import String, Int32, Float32, Bool, Header
from sensor_msgs.msg import LaserScan, PointCloud2
import pickle
import json

class TopicSubscriberSaver:
    def __init__(self):
        rospy.init_node('topic_subscriber_saver', anonymous=True)
        
        # 保存路径配置
        self.save_directory = rospy.get_param('~save_directory', os.path.join(os.getcwd(), 'subscribed_data'))
        self.topic_name = rospy.get_param('~topic_name', '/reid_node_0/cluster_data')
        self.message_type = rospy.get_param('~message_type', 'String')
        self.save_format = rospy.get_param('~save_format', 'json')  # 可选: pickle, json
        
        # 创建保存目录
        os.makedirs(self.save_directory, exist_ok=True)
        
        # 计数器，用于给保存的文件编号
        self.counter = 0
        
        # 根据消息类型订阅话题
        self.subscribe_to_topic()
        
        rospy.loginfo(f"开始订阅话题: {self.topic_name}")
        rospy.loginfo(f"保存路径: {self.save_directory}")
        rospy.loginfo(f"消息类型: {self.message_type}")
        rospy.loginfo(f"保存格式: {self.save_format}")
    
    def subscribe_to_topic(self):
        """根据指定的消息类型订阅话题"""
        if self.message_type == 'String':
            rospy.Subscriber(self.topic_name, String, self.string_callback)
        elif self.message_type == 'Int32':
            rospy.Subscriber(self.topic_name, Int32, self.int32_callback)
        elif self.message_type == 'Float32':
            rospy.Subscriber(self.topic_name, Float32, self.float32_callback)
        elif self.message_type == 'Bool':
            rospy.Subscriber(self.topic_name, Bool, self.bool_callback)
        elif self.message_type == 'LaserScan':
            rospy.Subscriber(self.topic_name, LaserScan, self.laserscan_callback)
        elif self.message_type == 'PointCloud2':
            rospy.Subscriber(self.topic_name, PointCloud2, self.pointcloud2_callback)
        elif self.message_type == 'Header':
            rospy.Subscriber(self.topic_name, Header, self.header_callback)
        else:
            rospy.logwarn(f"不支持的消息类型: {self.message_type}，使用String类型作为默认")
            rospy.Subscriber(self.topic_name, String, self.string_callback)
    
    def string_callback(self, msg):
        """处理String类型消息"""
        rospy.loginfo(f"收到String消息: {msg.data}")
        
        filename = os.path.join(self.save_directory, f"string_msg_{self.counter:04d}.{self.save_format}")
        
        if self.save_format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({"data": msg.data, "timestamp": rospy.get_time()}, f, ensure_ascii=False, indent=2)
        else:  # 默认使用pickle
            with open(filename, 'wb') as f:
                pickle.dump({"data": msg.data, "timestamp": rospy.get_time()}, f)
        
        rospy.loginfo(f"消息已保存到: {filename}")
        self.counter += 1
    
    def int32_callback(self, msg):
        """处理Int32类型消息"""
        rospy.loginfo(f"收到Int32消息: {msg.data}")
        
        filename = os.path.join(self.save_directory, f"int32_msg_{self.counter:04d}.pickle")
        with open(filename, 'wb') as f:
            pickle.dump({"data": msg.data, "timestamp": rospy.get_time()}, f)
        
        rospy.loginfo(f"消息已保存到: {filename}")
        self.counter += 1
    
    def float32_callback(self, msg):
        """处理Float32类型消息"""
        rospy.loginfo(f"收到Float32消息: {msg.data}")
        
        filename = os.path.join(self.save_directory, f"float32_msg_{self.counter:04d}.pickle")
        with open(filename, 'wb') as f:
            pickle.dump({"data": msg.data, "timestamp": rospy.get_time()}, f)
        
        rospy.loginfo(f"消息已保存到: {filename}")
        self.counter += 1
    
    def bool_callback(self, msg):
        """处理Bool类型消息"""
        rospy.loginfo(f"收到Bool消息: {msg.data}")
        
        filename = os.path.join(self.save_directory, f"bool_msg_{self.counter:04d}.pickle")
        with open(filename, 'wb') as f:
            pickle.dump({"data": msg.data, "timestamp": rospy.get_time()}, f)
        
        rospy.loginfo(f"消息已保存到: {filename}")
        self.counter += 1
    
    def laserscan_callback(self, msg):
        """处理LaserScan类型消息"""
        rospy.loginfo(f"收到LaserScan消息，范围数量: {len(msg.ranges)}")
        
        filename = os.path.join(self.save_directory, f"laserscan_msg_{self.counter:04d}.pickle")
        with open(filename, 'wb') as f:
            pickle.dump({
                "ranges": list(msg.ranges),
                "intensities": list(msg.intensities),
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                "time_increment": msg.time_increment,
                "scan_time": msg.scan_time,
                "range_min": msg.range_min,
                "range_max": msg.range_max,
                "timestamp": rospy.get_time()
            }, f)
        
        rospy.loginfo(f"LaserScan消息已保存到: {filename}")
        self.counter += 1
    
    def pointcloud2_callback(self, msg):
        """处理PointCloud2类型消息"""
        rospy.loginfo(f"收到PointCloud2消息，宽度: {msg.width}，高度: {msg.height}")
        
        filename = os.path.join(self.save_directory, f"pointcloud2_msg_{self.counter:04d}.pickle")
        with open(filename, 'wb') as f:
            pickle.dump({
                "header": {
                    "seq": msg.header.seq,
                    "stamp": {
                        "secs": msg.header.stamp.secs,
                        "nsecs": msg.header.stamp.nsecs
                    },
                    "frame_id": msg.header.frame_id
                },
                "height": msg.height,
                "width": msg.width,
                "fields": [{"name": f.name, "offset": f.offset, "datatype": f.datatype, "count": f.count} for f in msg.fields],
                "is_bigendian": msg.is_bigendian,
                "point_step": msg.point_step,
                "row_step": msg.row_step,
                "data": bytes(msg.data),
                "is_dense": msg.is_dense,
                "timestamp": rospy.get_time()
            }, f)
        
        rospy.loginfo(f"PointCloud2消息已保存到: {filename}")
        self.counter += 1
    
    def header_callback(self, msg):
        """处理Header类型消息"""
        rospy.loginfo(f"收到Header消息，frame_id: {msg.frame_id}")
        
        filename = os.path.join(self.save_directory, f"header_msg_{self.counter:04d}.pickle")
        with open(filename, 'wb') as f:
            pickle.dump({
                "seq": msg.seq,
                "stamp": {
                    "secs": msg.stamp.secs,
                    "nsecs": msg.stamp.nsecs
                },
                "frame_id": msg.frame_id,
                "timestamp": rospy.get_time()
            }, f)
        
        rospy.loginfo(f"Header消息已保存到: {filename}")
        self.counter += 1
    
    def run(self):
        """运行节点"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("节点被用户中断")
        finally:
            rospy.loginfo(f"总共保存了 {self.counter} 条消息")

if __name__ == '__main__':
    try:
        subscriber = TopicSubscriberSaver()
        subscriber.run()
    except rospy.ROSInterruptException:
        pass