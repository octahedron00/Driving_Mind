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

from ultralytics import YOLO

from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_square_pos, Line
from _mode import StartMode, EventMode, Stanley2GreenMode, Stanley2CrossMode, Turn2VoidMode, Turn2RoadMode, EndMode



frame_ignore_level = 1

log_vid = False
log_txt = True


yolo_pt = "best.pt"

class bot_mind:

    def __init__(self):

        self.model = YOLO(yolo_pt) 
        null_predict_to_turn_on_yolo = self.model.predict(np.zeros((480, 640, 3)))


        now = datetime.datetime.now().strftime("%H%M")
        if log_vid:
            self.logwriter = cv2.VideoWriter("log_" + now + ".avi", cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (640, 480))
        if log_txt:
            self.logtxt = open("log_" + now + ".txt", 'w')

        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        pub = self.pub

        self.mode = StartMode(pub)
        self.stage = 0

        self.count_frame = 1

        self.mode_list = [
            StartMode(pub),
            # EndMode(pub, 1000),
            Stanley2CrossMode(pub, 1),
            Turn2RoadMode(pub, 2, is_left=False, is_curve=True),
            Stanley2GreenMode(pub, 3, left_offset = -10),
            Turn2VoidMode(pub, 4, is_left=False),

            # EventMode(pub, self.model, 10, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 11, is_left=False, min_turn_sec=1.5),
            Stanley2CrossMode(pub, 12),
            Turn2RoadMode(pub, 13, is_left=False, is_curve=True),
            Stanley2GreenMode(pub, 14),
            # Stanley2GreenMode(pub, 14.5, from_it=True),
            Turn2VoidMode(pub, 15, is_left=True),

            # EventMode(pub, self.model, 20, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 21, is_left=False, min_turn_sec=1.5),
            Stanley2CrossMode(pub, 22, left_way=False, from_it=True, left_offset=0),
            Turn2RoadMode(pub, 23, is_left=False, left_way=False, is_curve=True),
            Stanley2GreenMode(pub, 24, left_offset = -10),
            Turn2VoidMode(pub, 25, is_left=True),

            # EventMode(pub, self.model, 30, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 31, is_left=False, min_turn_sec=1.5),
            Stanley2GreenMode(pub, 32, from_it=True, left_offset = -10),
            Turn2VoidMode(pub, 33, is_left=True),


            # EventMode(pub, self.model, 40, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 41, is_left=False, min_turn_sec=1.5),
            Stanley2CrossMode(pub, 42, right_way=False),
            Turn2RoadMode(pub, 43, is_left=True, right_way=False, is_curve=True),
            Stanley2GreenMode(pub, 44),
            EndMode(pub, 100),

        ]

        

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

        if self.mode.running:
            self.mode.log_add("time: ", time.time() - time_start)
            print(self.mode.log)
            if log_vid:
                self.logwriter.write(frame)
            if log_txt:
                self.logtxt.write(self.mode.log + "\n")
        else:
            a = input("Was it good?")
        cv2.waitKey(1)


if __name__ == "__main__":

    if not rospy.is_shutdown():
        bot_mind()
        rospy.spin()
