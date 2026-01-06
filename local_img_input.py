#!/home/u/ZHC/FusionReID-master/.venv/bin/python3.8
#coding: utf-8

"""
ROS节点，用于读取本地视频文件并将帧作为图像消息发布，
模拟相机节点的输出。
"""

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os


class VideoPublisher:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('video_pub_node', anonymous=True)

        # 从ROS参数获取视频文件路径或使用默认值
        self.video_path_0 = rospy.get_param('~video_path_0', '/home/u/Videos/dancetrack_test1.mp4')

        self.video_topic_0 = rospy.get_param('~video_topic_0', '/video_pub_node/image_raw_0')

        # 创建图像话题的发布者
        self.image_pub = rospy.Publisher(self.video_topic_0, Image, queue_size=1)

        # 创建CvBridge用于将OpenCV图像转换为ROS消息
        self.bridge = CvBridge()

        # 打开视频文件
        self.cap = cv2.VideoCapture(self.video_path_0)

        # 获取视频属性
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps <= 0:
            rospy.logwarn("无法从视频获取FPS，默认使用30 FPS")
            self.fps = 15.0

        rospy.loginfo(f"视频已加载: {self.video_path_0}")
        rospy.loginfo(f"帧率: {self.fps}, 总帧数: {self.frame_count}")

        # 计算帧之间的延迟（秒）
        self.frame_delay = 1.0 / self.fps

        # 初始化帧计数器
        self.current_frame = 0

        # 定时器控制发布速率
        self.timer = rospy.Timer(rospy.Duration(self.frame_delay), self.publish_frame)

        rospy.loginfo("视频发布器已初始化。正在向'/video_pub_node/image_raw_0'话题发布帧。")

    def publish_frame(self, event):
        """发布视频的单帧"""
        ret, frame = self.cap.read()

        if ret:
            # 将OpenCV图像转换为ROS图像消息
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                # 设置时间戳为当前ROS时间
                ros_image.header.stamp = rospy.Time.now()
                # 发布图像
                self.image_pub.publish(ros_image)

                self.current_frame += 1
                if self.current_frame % 30 == 0:  # 每30帧记录一次
                    rospy.logdebug(f"已发布帧 {self.current_frame}/{self.frame_count}")

            except Exception as e:
                rospy.logerr(f"图像转换错误: {e}")
        else:
            # 视频结束
            rospy.loginfo("视频结束。重新开始播放...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
            self.current_frame = 0

    def run(self):
        """运行视频发布器"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("正在关闭视频发布器...")
        finally:
            # 释放视频捕获
            if hasattr(self, 'cap'):
                self.cap.release()


def main():
    try:
        video_publisher = VideoPublisher()
        if hasattr(video_publisher, 'cap'):
            video_publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断。")
    except Exception as e:
        rospy.logerr(f"视频发布器错误: {e}")


if __name__ == '__main__':
    main()