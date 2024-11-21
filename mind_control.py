



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

from _mode import move_robot, showing_off

frame_ignore_level = 1

log_vid = True

class bot_mind:

    def __init__(self):

        now = datetime.datetime.now().strftime("%H%M")
        if log_vid:
            self.logwriter = cv2.VideoWriter("log_control_" + now + ".avi", cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (640, 480))

        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw', Image, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        pub = self.pub

        self.stage = 0

        self.count_frame = 1

        self.speed_x = 0
        self.speed_z = 0


    def camera_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.count_frame += 1
        if self.count_frame % frame_ignore_level == 0:
            self.action()


    def action(self):
        pub = self.pub
        frame = self.image

        showing_off([frame])
        if log_vid:
            self.logwriter.write(frame)
        
        k = cv2.waitKey(25)

        if k == ord('w'):
            move_robot(pub, 0.5, 0)
        elif k == ord('s'):
            move_robot(pub, -0.5, 0)
        elif k == ord('a'):
            move_robot(pub, 0, 0.5)
        elif k == ord('d'):
            move_robot(pub, 0, -0.5)
        else:
            move_robot(pub)



if __name__ == "__main__":

    if not rospy.is_shutdown():
        bot_mind()
        rospy.spin()
