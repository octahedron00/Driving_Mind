#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division


import rospy
import cv2
import numpy as np
import time
import math

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from math import *
from collections import deque


from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_square_pos, Line
from _mode import StartMode, Stanley2GreenMode, Turn2VoidMode, Turn2RoadMode


true_green_confidence = 100
true_green_dist_from_road = 20 #mm



frame_ignore_level = 2


class bot_mind:

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        pub = self.pub

        self.mode_list = [
            StartMode(pub),
            Stanley2GreenMode(pub),
            Turn2VoidMode(pub, is_left=True),
            Turn2RoadMode(pub, is_left=True),
            Stanley2GreenMode(pub),
            Turn2VoidMode(pub, is_left=True),
            Turn2RoadMode(pub, is_left=True),
        ]
        
        self.mode = StartMode(pub)
        self.stage = 0

        self.count_frame = 1

    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.count_frame += 1
        if self.count_frame % frame_ignore_level == 0:
            self.action()


    def action(self):
        
        if self.mode.end:
            self.stage += 1
            self.mode = self.mode_list[self.stage]
        frame = self.image

        self.mode.set_frame_and_move(frame, showoff = True)
        print(self.mode.log)
        cv2.waitKey(1)


if __name__ == "__main__":

    if not rospy.is_shutdown():
        bot_mind()
        rospy.spin()
