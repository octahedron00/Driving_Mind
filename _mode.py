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


from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur, get_cm_px_from_mm, get_square_pos, get_road_edges, get_road_edge_angle, Line


bot_from_bev_x = 100
bot_from_bev_y = 400



def showing_off(image_list):

    px = [0, 600, 1200, 0, 600, 1200, 0, 600, 1200]
    py = [0, 0, 0, 400, 400, 400, 800, 800, 800]

    for i, frame in enumerate(image_list):

        if i > 8:
            break

        cv2.namedWindow("win_" + str(i+1))
        cv2.moveWindow("win_" + str(i+1), px[i], py[i])
        cv2.imshow("win_" + str(i+1), frame)


def move_robot(pub, vel_x=0, rot_z=0, is_left=True):

    speed = Twist()
    speed.linear.x = vel_x
    speed.angular.z = rot_z
    if not is_left:
        speed.angular.z = -rot_z
    pub.publish(speed)


def move_stanley(pub, offset_mm, angle_deg)

    kp= 0.001
    k = 0.2
    x = 0.2

    z = -(angle_deg + atan(kp*offset_mm)) * k * x
    
    move_robot(pub, x, z)

    return z



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
        self.log = str(self.index) + "_Stanley2Green_"


    def set_frame_and_move(self, frame, showoff=True):

        bev, _ = get_bev(frame)

        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

        if len(x_list) > 2:
            self.init_pos_for_sliding_windows = x_list[1]
            self.line_road = Line(x_list, y_list)

        else:
            self.init_pos_for_sliding_windows = -1
        
        if self.line_road == None:
            # Must find the line here, First!
            # print("What, No Road? You Real? BRUHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            showing_off([frame, road_bev, road_sw_bev, bev])
            move_robot(self.pub)
            return


        # stanley
        offset_mm = self.line_road.get_offset(bot_from_bev_x,bot_from_bev_y)
        angle_deg = self.line_road.get_angle()

        z = move_stanley(pub, offset_mm, angle_deg)
        self.log = str(self.index) + "_Stanley2Green_offset " + str(offset_mm) + " mm / angle " + str(angle_deg) + "deg / z speed " + str(z)


        # green event!
        green_bev = get_green(bev)
        green_bev_cm = get_cm_px_from_mm(green_bev)
        green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 5)
        green_pos = [pos*10 for pos in green_pos_cm]

        if green_max > true_green_confidence and self.line_road.get_distance(green_pos[1], green_pos[0]) < true_green_dist_from_road:
            print("What, the true Green!!!", green_max, self.line_road.get_distance(green_pos[1], green_pos[0]))
            self.green_encounter += 1
        else:
            self.green_encounter -= 1
            self.green_encounter = max(0, self.green_encounter)
        
        if self.green_encounter >= 3:
            self.end = True
            move_robot(self.pub)
        

        # showoff now
        if showoff:
            cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
            showing_off([frame, road_bev, road_sw_bev, bev, green_bev_cm, green_blur_bev])


        

class SetFromGreenMode(Mode):

    pass




z_ang_speed = 0.8
default_time = 2.2 / z_ang_speed

class Turn2VoidMode(Mode):

    def __init__(self, pub, index=0, is_left = True, other_turn_sec = 0.2):
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
        self.log = str(self.index) + "_Turn2Void_"


    def set_frame_and_move(self, frame, showoff=True):

        bev, _ = get_bev(frame)

        # road edge detection
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        road_edge_bev, angle = get_road_edge_angle(road_bev, self.is_left)


        # starting
        if self.stage == -1:
            self.stage = 0
            self.time_since_stage = time.time()


        # slightly turning to other side: to get max data
        if self.stage == 0:
            self.log = str(self.index) + "_Turn2Void_stage" + str(self.stage)
            move_robot(self.pub, 0, -z_ang_speed, self.is_left)

            if time.time() - self.time_since_stage > self.other_turn_sec:
                self.stage = 1
                self.time_since_stage = time.time()            
                move_robot(self.pub)
        

        # turning while the line is shown: to estimate time to be exact 90 degrees
        if self.stage == 1:
            move_robot(self.pub, 0, z_ang_speed, self.is_left)

            if abs(angle) < 45:
                self.time_list.append(time.time() - self.time_since_stage)
                self.angle_list.append(angle)
                
                # cv2.imwrite("frame_edge_" + str(len(self.time_list)) + ".jpg", road_edge_bev)
                # cv2.imwrite("frame_original_" + str(len(self.time_list)) + ".jpg", road_bev)
                self.log = str(self.index) + "_Turn2Void_stage" + str(self.stage) + " " + str(angle)
            else:
                self.stage = 2

                if len(self.time_list) > 6:
                    len_ignore = int(len(self.time_list)/2.5)
                    self.time_list = self.time_list[len_ignore:-2]
                    self.angle_list = self.angle_list[len_ignore:-2]
                print(self.time_list, self.angle_list)

                if len(self.time_list) < 2:
                    self.est_time = default_time
                    return
                calc = np.poly1d(np.polyfit(self.angle_list, self.time_list, deg=1))
                if self.is_left:
                    self.est_time = calc(90)
                else:  
                    self.est_time = calc(-90)


        if self.stage == 2:
            self.log = str(self.index) + "_Turn2Void_stage" + str(self.stage) + " " + str(self.est_time)

            if time.time() < self.time_since_stage + self.est_time:
                move_robot(self.pub, 0, z_ang_speed, self.is_left)
            else:
                move_robot(self.pub)
                self.end = True

        if showoff:
            showing_off([frame, road_bev, road_edge_bev, bev])



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
        self.log = str(self.index) + "_Turn2Road_"


    def set_frame_and_move(self, frame, showoff=True):

        bev, _ = get_bev(frame)
        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

        # starting
        if self.stage == -1:
            self.stage = 0
            self.time_since_stage = time.time()

        # turning at least certain amount: to ignore post-road
        if self.stage == 0:
            self.log = str(self.index) + "_Turn2Road_"
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
            self.log = str(self.index) + "_Turn2Road_stage1_linenotshown"
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

                self.log = str(self.index) + "_Turn2Road_stage1_line_on_angle_" + str(line_road.get_angle())

                cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)

                if len(x_list) > 8 or abs(line_road.get_angle()) < 10:
                    self.stage = 2

        if self.stage == 2:        
            speed = Twist()
            self.pub.publish(speed)
            self.end = True

        if showoff:
            showing_off([frame, road_bev, road_sw_bev, bev])




