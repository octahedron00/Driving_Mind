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


from src.lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur, get_square_pos, Line




class Mode:

    end = False
    pub = None

    def __init__(self, pub):
        self.end = False
        self.pub = pub
    
    def set_frame_and_move(self, frame, showoff=True):
        
        if showoff:
            pass
        cv2.waitKey(1)
        pass



true_green_confidence = 100
true_green_dist_from_road = 20 #mm


class Stanley2GreenMode(Mode):

    def __init__(self, pub):
        super(Stanley2GreenMode, self).__init__(pub)

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.green_encounter = 0


    def set_frame_and_move(self, frame, showoff=True):

        bev, _ = get_bev(frame)

        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 25)
        road_sw_bev, x_list, y_list = get_sliding_window_result(filter_frame, self.init_pos_for_sliding_windows)

        if len(x_list) > 2:
            self.init_pos_for_sliding_windows = x_list[1]
            self.line_road = Line(x_list, y_list)

            # print("x = ", self.line_road.var_1, " y + ", self.line_road.var_0)
            # print("Dist from origin: ", self.line_road.get_distance(bot_from_bev_x,bot_from_bev_y), ", angle: ", self.line_road.get_angle())
        else:
            self.init_pos_for_sliding_windows = -1
        
        if self.line_road == None:
            print("What, No Road? You Real? BRUHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            return

            # Must find the line here, First!

        # stanley
        offset_mm = self.line_road.get_offset(bot_from_bev_x,bot_from_bev_y)
        angle_deg = self.line_road.get_angle()

        kp= 0.001
        k = 0.01
        x = 0.15

        speed = Twist()
        speed.linear.x = x
        speed.angular.z = -(angle_deg + atan(kp*offset_mm)) * k / x
        self.pub.publish(speed)
        print(offset_mm, angle_deg speed.angular.z)

        # green event!
        green_bev = get_green(bev)
        green_blur_bev, green_pos, green_max = get_square_pos(green_bev, 41)

        if green_max > true_green_confidence and self.line_road.get_distance(green_pos[1], green_pos[0]) < true_green_dist_from_road:
            print("What, the true Green!!!", green_max, self.line_road.get_distance(green_pos[1], green_pos[0]))
            self.green_encounter += 1
        else:
            self.green_encounter -= 1
        
        
        if self.green_encounter >= 5:
            self.end = True

            speed = Twist()
            speed.linear.x = 0
            speed.angular.z = 0
            self.pub.publish(speed)
        
        

class Turn2RoadMode(Mode):

    def __init__(self, pub, is_left = True):
        super(Stanley2GreenMode, self).__init__(pub)

        self.is_left = is_left
        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.green_encounter = 0

        if way == "e"



