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
import datetime


from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_square_pos, Line
from _mode import StartMode, Stanley2GreenMode, Stanley2CrossMode, Turn2VoidMode, Turn2RoadMode, EndMode



frame_ignore_level = 1


class bot_mind:

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        pub = self.pub
        now = datetime.datetime.now().strftime("%H%M")
        self.logwriter = cv2.VideoWriter("log_" + now + ".avi", cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (640, 480))
        self.logtxt = open("log_" + now + ".txt", 'w')

        self.mode_list = [
            StartMode(pub),

            Stanley2CrossMode(pub, 1),
            Turn2RoadMode(pub, 2, is_left=False, is_curve=True),
            Stanley2GreenMode(pub, 3, left_offset = -10),
            Turn2VoidMode(pub, 4, is_left=True),
            Turn2RoadMode(pub, 5, is_left=True),
            Stanley2CrossMode(pub, 6),
            Turn2RoadMode(pub, 7, is_left=False, is_curve=True),
            Stanley2GreenMode(pub, 8),

            Stanley2GreenMode(pub, 9, from_it=True, left_offset = -10),
            Turn2VoidMode(pub, 10, is_left=True),
            Turn2RoadMode(pub, 11, is_left=False),

            Stanley2CrossMode(pub, 12, left_way=False, from_it=True),
            Turn2RoadMode(pub, 13, is_left=False, left_way=False, is_curve=True),
            Stanley2GreenMode(pub, 14, left_offset = -10),
            Turn2VoidMode(pub, 15, is_left=True),
            Turn2RoadMode(pub, 16, is_left=False),
            Stanley2GreenMode(pub, 17, from_it=True, left_offset = -10),
            Turn2VoidMode(pub, 30, is_left=True),
            Turn2RoadMode(pub, 31, is_left=False),
            Stanley2CrossMode(pub, 32, right_way=False, from_it=True),
            Turn2RoadMode(pub, 33, is_left=True, right_way=False, is_curve=True),
            Stanley2GreenMode(pub, 100),
            EndMode(pub, 1000),

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

        time_start = time.time()
        self.mode.set_frame_and_move(frame, showoff = True)
        if len(self.mode.log)>0:
            self.logwriter.write(frame)
            print("time spent:", round(time.time()-time_start, 3), end="   |  ")
            print(self.mode.log)
            self.logtxt.write(self.mode.log + "\n")
        else:
            self.logwriter.release()
            a = input("Was it good?")
        cv2.waitKey(1)


if __name__ == "__main__":

    if not rospy.is_shutdown():
        bot_mind()
        rospy.spin()
