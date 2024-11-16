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


from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur, get_square_pos, Line


bot_from_bev_x = 100
bot_from_bev_y = 400



def showing_off(image_list):

    px = [0, 600, 1200, 0, 600, 1200, 0, 600, 1200]
    py = [0, 0, 0, 400, 400, 400, 800, 800, 800]

    for i, frame in enumerate(image_list):

        if i > 8:
            break

        cv2.namedWindow(f"win_{i+1}")
        cv2.moveWindow(f"win_{i+1}", px[i], py[i])
        cv2.imshow(f"win_{i+1}", frame)



class Mode:

    end = False
    pub = None

    def __init__(self, pub):
        self.end = False
        self.pub = pub
    
    def set_frame_and_move(self, frame, showoff=True):
        pass


class StartMode:

    def __init__(self, pub):
        self.end = True
        self.pub = pub
    
    def set_frame_and_move(self, frame, showoff=True):
        
        if showoff:
            pass
        cv2.waitKey(1)
        pass



true_green_confidence = 100
true_green_dist_from_road = 20 #mm


class Stanley2GreenMode(Mode):

    def __init__(self, pub, index=0):
        
        self.end = False
        self.pub = pub

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.green_encounter = 0

        self.index = index
        self.log = f"{self.index}_Stanley2Green_"


    def set_frame_and_move(self, frame, showoff=True):

        bev, _ = get_bev(frame)

        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 25)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

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
        self.log = f"{self.index}_Stanley2Green offset: {offset_mm:.03f} mm, angle: {angle_deg:.03f} deg, z_ang_speed: {speed.angular.z:.03f}"
        # print("offset:", offset_mm, "angle:", angle_deg, "z_angular_value:", speed.angular.z)

        # green event!
        green_bev = get_green(bev)
        green_blur_bev, green_pos, green_max = get_square_pos(green_bev, 41)

        if green_max > true_green_confidence and self.line_road.get_distance(green_pos[1], green_pos[0]) < true_green_dist_from_road:
            print("What, the true Green!!!", green_max, self.line_road.get_distance(green_pos[1], green_pos[0]))
            self.green_encounter += 1
        else:
            self.green_encounter -= 1
            self.green_encounter = max(0, self.green_encounter)
        
        
        if self.green_encounter >= 5:
            self.end = True

            speed = Twist()
            speed.linear.x = 0
            speed.angular.z = 0
            self.pub.publish(speed)
        
        if showoff:
            cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
            showing_off([frame, road_bev, road_sw_bev, bev, green_bev, green_blur_bev])
        
        cv2.waitKey(1)

        

class SetFromGreenMode(Mode):

    pass





class Turn2VoidMode(Mode):

    def __init__(self, pub, index=0, is_left = True, other_turn_sec = 0.1):
        self.end = False
        self.pub = pub

        self.is_left = is_left
        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.other_turn_sec = other_turn_sec

        self.stage = -1
        self.time_since_stage = 0
        self.angle_list = []
        self.time_list = []
        self.est_time = 0
        
        self.index = index
        self.log = f"{self.index}_Turn2Void_"


    def set_frame_and_move(self, frame, showoff=True):

        bev, _ = get_bev(frame)

        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 25)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

        # starting
        if self.stage == -1:
            self.stage = 0
            self.time_since_stage = time.time()

        # slightly turning to other side: to get max data
        if self.stage == 0:
            self.log = f"{self.index}_Turn2Void_stage{self.stage}"
            speed = Twist()
            speed.linear.x = 0
            if self.is_left:
                speed.angular.z = -0.5 # move other side a little bit
            else:
                speed.angular.z = 0.5
            self.pub.publish(speed)
            if time.time() - self.time_since_stage > self.other_turn_sec:
                self.stage = 1
                self.time_since_stage = time.time()
                speed = Twist()
                speed.linear.x = 0
                speed.angular.z = 0
                self.pub.publish(speed)
        
        # turning while the line is shown: to estimate time to be exact 90 degrees
        if self.stage == 1:
            speed = Twist()
            speed.linear.x = 0
            if self.is_left:
                speed.angular.z = 0.5 # move other side a little bit
            else:
                speed.angular.z = -0.5
            self.pub.publish(speed)

            if len(x_list) > 2:
                self.init_pos_for_sliding_windows = x_list[1]
                line_road = Line(x_list, y_list)
                self.line_road = line_road

                self.log = f"{self.index}_Turn2Void_stage{self.stage}_angle:{line_road.get_angle()}"
                self.time_list.append(time.time() - self.time_since_stage)
                self.angle_list.append(line_road.get_angle())
                cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
            else:
                self.stage = 2
                calc = np.poly1d(np.polyfit(self.angle_list, self.time_list, deg=1))
                if is_left:
                    self.est_time = calc(-90)
                else:  
                    self.est_time = calc(90)


        if self.stage == 2:
            self.log = f"{self.index}_Turn2Void_stage{self.stage}_time/est:{time.time() - self.time_since_stage:.03f}/{est_time:.03f}"
            if time.time() < self.time_since_stage + self.est_time:
                speed = Twist()
                speed.linear.x = 0
                if self.is_left:
                    speed.angular.z = 0.5 # move other side a little bit
                else:
                    speed.angular.z = -0.5
                self.pub.publish(speed)
            else:
                speed = Twist()
                self.pub.publish(speed)
                self.end = True

        if showoff:
            showing_off([frame, road_bev, road_sw_bev, bev])



class Turn2RoadMode(Mode):

    def __init__(self, pub, index = 0, is_left = True, min_turn_sec = 2.0):
        self.end = False
        self.pub = pub

        self.is_left = is_left
        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.min_turn_sec = min_turn_sec

        self.stage = -1
        self.time_since_stage = 0
        self.est_time = 0

        self.index = index
        self.log = f"{self.index}_Turn2Road_"


    def set_frame_and_move(self, frame, showoff=True):

        bev, _ = get_bev(frame)

        # starting
        if self.stage == -1:
            self.stage = 0
            self.time_since_stage = time.time()

        # turning at least certain amount: to ignore post-road
        if self.stage == 0:
            self.log = f"{self.index}_Turn2Void_stage{self.stage}"
            speed = Twist()
            speed.linear.x = 0
            if self.is_left:
                speed.angular.z = 0.5 # move other side a little bit
            else:
                speed.angular.z = -0.5
            self.pub.publish(speed)
            if time.time() - self.time_since_stage > self.min_turn_sec:
                self.stage = 1
                self.time_since_stage = time.time()
                speed = Twist()
                self.pub.publish(speed)
        
        # turning while the line is shown: to estimate time to be exact 90 degrees
        if self.stage == 1:
            self.log = f"{self.index}_Turn2Void_stage{self.stage}_lineNotShown"
            speed = Twist()
            speed.linear.x = 0
            if self.is_left:
                speed.angular.z = 0.5 # move other side a little bit
            else:
                speed.angular.z = -0.5
            self.pub.publish(speed)

            # slidingwindow
            road_bev = get_road(bev)
            road_blur_bev = get_rect_blur(road_bev, 25)
            road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

            if len(x_list) > 2:
                self.init_pos_for_sliding_windows = x_list[1]
                line_road = Line(x_list, y_list)
                self.line_road = line_road

                self.log = f"{self.index}_Turn2Void_stage{self.stage}_line_angle: {line_road.get_angle()}"

                cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)

                if len(x_list) > 8 or abs(line_road.get_angle()) < 10:
                    self.stage == 2

        if self.stage == 2:        
            speed = Twist()
            self.pub.publish(speed)
            self.end = True

        if showoff:
            showing_off([frame, road_bev, road_sw_bev, bev])




