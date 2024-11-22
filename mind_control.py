



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
from collections import deque
import datetime

# from ultralytics import YOLO

from _mode import move_robot, showing_off

FRAME_IGNORE_LEVEL = 1

IS_LOG_VID = True

class control_mind:

    def __init__(self):

        now = datetime.datetime.now().strftime("%H%M")
        if IS_LOG_VID:
            self.logwriter = cv2.VideoWriter("log_control_" + now + ".avi", cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (640, 480))

        self.pub = TikiMini()
        pub = self.pub

        self.count_frame = 1

        self.image_name = "image_" + now + "_"
        self.image_count = 1

        cap = cv2.VideoCapture(1)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.count_frame += 1
            if self.count_frame % FRAME_IGNORE_LEVEL == 0:
                self.action(frame)



    def action(self, frame):
        pub = self.pub

        showing_off([frame])
        if IS_LOG_VID:
            self.logwriter.write(frame)
        
        k = cv2.waitKey(1)
        print(k)
        if k == ord('w'):
            move_robot(pub, 0.2, 0)
        elif k == ord('s'):
            move_robot(pub, -0.2, 0)
        elif k == ord('a'):
            move_robot(pub, 0, 0.5)
        elif k == ord('d'):
            move_robot(pub, 0, -0.5)
        elif k == ord('c'):
            cv2.imwrite(self.image_name + f"{self.image_count:04d}.jpg", frame)
            self.image_count += 1
        elif k == ord('n'):
            self.image_name = input("new dataset name:")
            self.image_count = 1
        else:
            move_robot(pub)



if __name__ == "__main__":

    if not rospy.is_shutdown():
        control_mind()
        rospy.spin()
