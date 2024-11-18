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


from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur
from _lane_detect import get_cm_px_from_mm, get_square_pos, get_road_edge_angle, get_road_and_cross_pos, Line


bot_from_bev_x = 100
bot_from_bev_y = 400

speed_x = 0.4


def showing_off(image_list):
#    return

    px = [0, 600, 1200, 0, 600, 1200, 0, 600, 1200]
    py = [0, 0, 0, 400, 400, 400, 800, 800, 800]

    for i, frame in enumerate(image_list[1:]):

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


def move_stanley(pub, offset_mm, angle_deg):

    kp= 0.03
    ka= 0.2
    k = 0.7
    x = speed_x

    z = -(angle_deg*ka - atan(kp*offset_mm)) * x * k
    
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

class CamMode:

    def __init__(self, pub):
        self.end = True
        self.pub = pub
    
    def set_frame_and_move(self, frame, showoff=True):
        
        if showoff:
            pass
        cv2.waitKey(1)
        pass


class EndMode(Mode):

    def __init__(self, pub, index=0):
        self.end = False
        self.pub = pub
        move_robot(pub)
        self.log = ""
    
    def set_frame_and_move(self, frame, showoff=True):
        
        if showoff:
            pass
        cv2.waitKey(1)
        pass



true_green_confidence = 100
true_green_dist_from_road = 30 #mm


class Stanley2GreenMode(Mode):

    def __init__(self, pub, index=0, from_it=False, left_offset = 0, debug=False):
        self.end = False
        self.pub = pub

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.green_encounter = -2
        if from_it:
            self.green_encounter = -100
        self.left_offset = left_offset

        self.index = index
        self.log = str(self.index) + "_Stanley2Green_"
        self.debug = debug


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
        offset_mm = self.line_road.get_offset(bot_from_bev_x+self.left_offset,bot_from_bev_y)
        angle_deg = self.line_road.get_angle()

        z = move_stanley(self.pub, offset_mm, angle_deg)
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
            self.green_encounter = max(int(self.green_encounter/2.1), self.green_encounter)
        
        if self.green_encounter >= 3:
            self.end = True
            move_robot(self.pub)
        

        # showoff now
        if showoff:
            cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
            showing_off([frame, road_bev, road_sw_bev, bev, green_bev_cm, green_blur_bev])
        if self.debug:
            cv2.imwrite("S2G_" + str(self.index) + "_debug_green" + str(self.green_encounter) + ".jpg", green_bev)
            cv2.imwrite("S2G_" + str(self.index) + "_debug_bev" + str(self.green_encounter) + ".jpg", bev)


true_cross_confidence = 200


class Stanley2CrossMode(Mode):

    def __init__(self, pub, index=0, left_way = True, right_way = True, from_it=False, left_offset = 0):
        self.end = False
        self.pub = pub

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.cross_encounter = -2
        if from_it:
            self.cross_encounter = -1000
        self.left_way = left_way
        self.right_way = right_way
        self.left_offset = left_offset

        self.index = index
        self.log = str(self.index) + "_Stanley2Cross_"


    def set_frame_and_move(self, frame, showoff=True):

        bev, _ = get_bev(frame)

        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        # cross event!
        cross_find_view, x_list, y_list, is_cross, positions = get_road_and_cross_pos(road_blur_bev, 5, self.left_way, self.right_way, self.init_pos_for_sliding_windows)
        
        
        road_sw_bev = cross_find_view

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
        offset_mm = self.line_road.get_offset(bot_from_bev_x+self.left_offset,bot_from_bev_y)
        angle_deg = self.line_road.get_angle()

        z = move_stanley(self.pub, offset_mm, angle_deg)
        self.log = str(self.index) + "_Stanley2Cross_offset " + str(offset_mm) + " mm / angle " + str(angle_deg) + "deg / z speed " + str(z)


        # print(cross_pos, cross_max)
        self.log = self.log + " CROSS::" + str(positions)

        if is_cross:
            print("OK, Now We At the Cross!!!", positions)
            self.cross_encounter += 1
        else:
            self.cross_encounter -= 1
            self.cross_encounter = max(0, self.cross_encounter)
        
        if self.cross_encounter >= 3:
            self.end = True
            move_robot(self.pub)
            self.memory = np.mean(positions)
        

        # showoff now
        if showoff:
            cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
            showing_off([frame, road_bev, road_sw_bev, bev, cross_find_view])


z_ang_speed = 1
default_time = 2.7 / z_ang_speed



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

        self.waiting_for_next_frame = 2
        
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
                self.waiting_for_next_frame = 2
                self.time_list.append(time.time() - self.time_since_stage)
                self.angle_list.append(angle)
                
                #cv2.imwrite(str(self.index) + "_frame_edge_" + str(len(self.time_list)) + ".jpg", road_edge_bev)
                #cv2.imwrite(str(self.index) + "_frame_original_" + str(len(self.time_list)) + ".jpg", road_bev)
                self.log = str(self.index) + "_Turn2Void_stage" + str(self.stage) + " " + str(angle)
            elif self.waiting_for_next_frame > 0:
                self.waiting_for_next_frame -= 1
            else:
                self.stage = 2
                self.log += str(self.angle_list) + str(self.time_list)
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
                
                if self.est_time < default_time - 1 or default_time + 1.2 < self.est_time:
                    self.est_time = default_time


        if self.stage == 2:
            self.log = str(self.index) + "_Turn2Void_stage" + str(self.stage) + " " + str(self.est_time)

            if time.time() < self.time_since_stage + self.est_time:
                move_robot(self.pub, 0, z_ang_speed, self.is_left)
            else:
                move_robot(self.pub)
                self.end = True

        if showoff:
            showing_off([frame, road_bev, road_edge_bev, bev])



radius_vx_vz_coefficient = 900  # edit this


class Turn2RoadMode(Mode):

    def __init__(self, pub, index = 0, is_left = True, min_turn_sec = 1.5, is_curve = False, left_way = True, right_way = True):
        self.end = False
        self.pub = pub

        self.is_left = is_left
        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.min_turn_sec = min_turn_sec
        self.is_curve = is_curve

        self.road_angle = -1000
        self.dist_from_cross = -1

        self.stage = -1
        self.time_since_stage = 0
        self.est_time = 0

        self.rot_z = z_ang_speed
        self.speed_x = speed_x
        self.left_way = left_way
        self.right_way = right_way

        self.index = index
        self.log = str(self.index) + "_Turn2Road_"


    def set_frame_and_move(self, frame, showoff=True):


        bev, _ = get_bev(frame)
        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        

        if self.stage == -1 and self.is_curve:
            road_sw_bev, x_list, y_list, is_cross, positions = get_road_and_cross_pos(road_blur_bev, 5, self.left_way, self.right_way, self.init_pos_for_sliding_windows)
            dist_from_cross = bot_from_bev_y - np.mean(positions)

            not_right = True
            if self.is_left:
                not_right = False
            road_edge_bev, angle = get_road_edge_angle(road_bev, not_right)
            if abs(self.road_angle) > 10:
                self.road_angle = 0

            if self.is_left:
                self.road_angle = -self.road_angle

            radius = dist_from_cross / (1 + math.sin(self.road_angle*math.pi/180))

            self.speed_x = radius * self.rot_z / radius_vx_vz_coefficient
            self.log = self.log + " | positions " + str(positions) + " | radius " + str(radius) + " | rot_z " + str(self.rot_z)
            cv2.imwrite(str(self.index) + "_curve_dist.jpg", road_sw_bev)
            cv2.imwrite(str(self.index) + "_curve_angle.jpg", road_edge_bev)

        else:
            road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)


        # starting
        if self.stage == -1:
            self.stage = 0
            self.time_since_stage = time.time()
       

        # turning at least certain amount: to ignore post-road
        if self.stage == 0:
            x = 0
            if self.is_curve:
                x = self.speed_x
            move_robot(self.pub, x, self.rot_z, self.is_left)

            if time.time() - self.time_since_stage > self.min_turn_sec:
                self.stage = 1
                self.time_since_stage = time.time()
                move_robot(self.pub)
        
        # turning while the line is shown: to estimate time to be exact 90 degrees
        if self.stage == 1:
            self.log = str(self.index) + "_Turn2Road_stage1_linenotshown"
            x = 0
            if self.is_curve:
                x = self.speed_x
            move_robot(self.pub, x, self.rot_z, self.is_left)

            if len(x_list) > 2:
                self.init_pos_for_sliding_windows = x_list[1]
                line_road = Line(x_list, y_list)
                self.line_road = line_road

                self.log = str(self.index) + "_Turn2Road_stage1_line_on_angle_" + str(line_road.get_angle())

                cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)

                if len(x_list) > 8 or abs(line_road.get_angle()) < 10:
                    self.stage = 2

        if self.stage == 2:        
            move_robot(self.pub)
            self.end = True

        if showoff:
            showing_off([frame, road_bev, road_sw_bev, bev])




