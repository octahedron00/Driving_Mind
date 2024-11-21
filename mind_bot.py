#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division


import rospy
import cv2
import numpy as np
import time
import math

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from math import *
from collections import deque
import datetime

# from ultralytics import YOLO

from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_square_pos, Line
from _mode import StartMode, EventMode, Stanley2GreenMode, Stanley2CrossMode, Turn2VoidMode, Turn2RoadMode, EndMode



FRAME_IGNORE_LEVEL = 1

IS_LOG = True
IS_LOG_VID = False


FILE_YOLO = "best.pt"

class bot_mind:

    def __init__(self):

        self.model = YOLO(FILE_YOLO) 
        null_predict_to_turn_on_yolo = self.model.predict(np.zeros((480, 640, 3)))


        now = datetime.datetime.now().strftime("%H%M")
        if IS_LOG_VID:
            self.logwriter = cv2.VideoWriter("log_" + now + ".avi", cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (640, 480))
        if IS_LOG:
            self.logtxt = open("log_" + now + ".txt", 'w')

        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw', Image, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        pub = self.pub

        self.mode = StartMode(pub)
        self.mode_pos = 0

        self.count_frame = 1

        self.mode_list = [
            StartMode(pub),

            Stanley2GreenMode(pub, -1, left_offset = -20, from_it = True),
            Stanley2GreenMode(pub, -2, left_offset = -20, from_it = True),
            Stanley2GreenMode(pub, -3, left_offset = -20, from_it = True),
            Turn2RoadMode(pub, -4, is_left=True, min_turn_sec=1),

            Stanley2GreenMode(pub, -11, left_offset = 20, from_it = True),
            Stanley2GreenMode(pub, -12, left_offset = 20, from_it = True),
            Stanley2GreenMode(pub, -13, left_offset = 20, from_it = True),
            Turn2RoadMode(pub, -14, is_left=True, min_turn_sec=1),
            # EndMode(pub, 1000),


            Stanley2CrossMode(pub, 1, use_green = True),
            Turn2RoadMode(pub, 2, is_left=False, is_curve=True),
            # Stanley2GreenMode(pub, 1, left_offset = 0),
            # Turn2RoadMode(pub, 2, is_left=False, min_turn_sec = 1.),
            Stanley2GreenMode(pub, 3, left_offset = -10),
            Turn2VoidMode(pub, 4, is_left=True, other_turn_sec=0),

            # EventMode(pub, self.model, 10, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 11, is_left=True, min_turn_sec=1),
            Stanley2CrossMode(pub, 12),
            Turn2RoadMode(pub, 13, is_left=False, is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 14, from_it = True),
            # Stanley2GreenMode(pub, 14.5, from_it=True),
            Turn2VoidMode(pub, 15, is_left=True, other_turn_sec=0),

            # EventMode(pub, self.model, 20, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 21, is_left=False, min_turn_sec=1.),
            Stanley2CrossMode(pub, 22, left_way=False, from_it=True, left_offset=0),
            Turn2RoadMode(pub, 23, is_left=False, left_way=False, is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 24, left_offset = -10),
            Turn2VoidMode(pub, 25, is_left=True, other_turn_sec=0),

            # EventMode(pub, self.model, 30, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 31, is_left=False, min_turn_sec=1.),
            Stanley2GreenMode(pub, 32, from_it=True, left_offset = -10),
            Turn2VoidMode(pub, 33, is_left=True, other_turn_sec=0),


            # EventMode(pub, self.model, 40, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 41, is_left=False, min_turn_sec=1.),
            Stanley2CrossMode(pub, 42, right_way=False),
            Turn2RoadMode(pub, 43, is_left=True, right_way=False, min_turn_sec=1., is_curve=True),
            Stanley2GreenMode(pub, 44),

            # EndMode(pub, 100),

            Turn2RoadMode(pub, 101, is_left=False),
            Stanley2CrossMode(pub, 102),
            Turn2RoadMode(pub, 103, is_left=False, min_turn_sec=1., is_curve=True),
            Stanley2GreenMode(pub, 104),
            Stanley2CrossMode(pub, 105),
            Turn2RoadMode(pub, 106, is_left=True, min_turn_sec=1., is_curve=True),
            Stanley2CrossMode(pub, 107),
            Turn2RoadMode(pub, 108, is_left=True, is_curve=True),
            Stanley2GreenMode(pub, 109, left_offset = -10),
            Turn2VoidMode(pub, 110, is_left=True, other_turn_sec=0),

            # EventMode(pub, self.model, 10, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 111, is_left=True, min_turn_sec=1),
            Stanley2CrossMode(pub, 112),
            Turn2RoadMode(pub, 113, is_left=True, is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 114, from_it=True),
            Turn2RoadMode(pub, 115, is_left=True),


            Stanley2GreenMode(pub, 1, left_offset = 0),
            Turn2RoadMode(pub, 2, is_left=False, min_turn_sec = 1.),
            Stanley2GreenMode(pub, 3, left_offset = -10),
            Turn2VoidMode(pub, 4, is_left=True, other_turn_sec=0),

            # EventMode(pub, self.model, 10, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 11, is_left=True, min_turn_sec=1),
            Stanley2CrossMode(pub, 12),
            Turn2RoadMode(pub, 13, is_left=False, is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 14),
            # Stanley2GreenMode(pub, 14.5, from_it=True),
            Turn2VoidMode(pub, 15, is_left=True, other_turn_sec=0),

            # EventMode(pub, self.model, 20, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 21, is_left=False, min_turn_sec=1.),
            Stanley2CrossMode(pub, 22, left_way=False, from_it=True, left_offset=0),
            Turn2RoadMode(pub, 23, is_left=False, left_way=False, is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 24, left_offset = -10),
            Turn2VoidMode(pub, 25, is_left=True, other_turn_sec=0),

            # EventMode(pub, self.model, 30, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 31, is_left=False, min_turn_sec=1.),
            Stanley2GreenMode(pub, 32, from_it=True, left_offset = -10),
            Turn2VoidMode(pub, 33, is_left=True, other_turn_sec=0),


            # EventMode(pub, self.model, 40, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 41, is_left=False, min_turn_sec=1.),
            Stanley2CrossMode(pub, 42, right_way=False),
            Turn2RoadMode(pub, 43, is_left=True, right_way=False, min_turn_sec=1., is_curve=True),
            Stanley2GreenMode(pub, 44),

            # EndMode(pub, 100),

            Turn2RoadMode(pub, 101, is_left=False),
            Stanley2CrossMode(pub, 102),
            Turn2RoadMode(pub, 103, is_left=False, min_turn_sec=1., is_curve=True),
            Stanley2GreenMode(pub, 104),
            Stanley2CrossMode(pub, 105),
            Turn2RoadMode(pub, 106, is_left=True, min_turn_sec=1., is_curve=True),
            Stanley2CrossMode(pub, 107),
            Turn2RoadMode(pub, 108, is_left=True, is_curve=True),
            Stanley2GreenMode(pub, 109, left_offset = -10),
            Turn2VoidMode(pub, 110, is_left=True, other_turn_sec=0),

            # EventMode(pub, self.model, 10, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 111, is_left=True, min_turn_sec=1),
            Stanley2CrossMode(pub, 112),
            Turn2RoadMode(pub, 113, is_left=True, is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 114, from_it=True),
            Turn2RoadMode(pub, 115, is_left=True),

            EndMode(pub, 200),

        ]

        

    def camera_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.count_frame += 1
        if self.count_frame % FRAME_IGNORE_LEVEL == 0:
            self.action()


    def action(self):
        
        if self.mode.end:
            capsule = self.mode.capsule
            self.mode_pos += 1
            self.mode = self.mode_list[self.mode_pos]
            self.mode.capsule = capsule
        frame = self.image

        time_start = time.time()
        self.mode.set_frame_and_move(frame, showoff = True)

        if self.mode.running:
            self.mode.log_add("time: ", time.time() - time_start)
            print(self.mode.log)
            if IS_LOG_VID:
                self.logwriter.write(frame)
            if IS_LOG:
                self.logtxt.write(self.mode.log + "\n")
        else:
            a = input("Was it good?")
        cv2.waitKey(1)


if __name__ == "__main__":

    if not rospy.is_shutdown():
        bot_mind()
        rospy.spin()
