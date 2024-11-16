#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

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


from src.lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_square_pos, Line


true_green_confidence = 100
true_green_dist_from_road = 20 #mm

bot_from_bev_x = 100
bot_from_bev_y = 400

deg_90_sec = 3.0


class bot_mind:

    GO_STRAIGHT = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    MISSION = 4
    # GO_CURVE = 5
    WELL_DONE = -1

    LIST = [0, 1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 4, 3, 1, -1]

    stage = 1
    state = 1
    green_encounter = 0
    line_road = None
    init_pos_for_sliding_windows = -1

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        self.stage = 1
        self.state = 1
        self.green_encounter = 0
        self.line_road = None

    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.action()


    def action(self):

        frame = self.image

        bev_frame, Minv = get_bev(frame)

        filter_frame = get_road(bev_frame)

        green_frame = get_green(bev_frame)

        window_frame, x_list, y_list = get_sliding_window_result(filter_frame, self.init_pos_for_sliding_windows)

        if len(x_list) > 3:
            self.init_pos_for_sliding_windows = x_list[0]
            self.line_road = Line(x_list, y_list)

            print("x = ", self.line_road.var_1, " y + ", self.line_road.var_0)
            print("Dist from origin: ", self.line_road.get_distance(bot_from_bev_x,bot_from_bev_y), ", angle: ", self.line_road.get_angle())
        else:
            self.init_pos_for_sliding_windows = -1
        
        if self.line_road == None:
            return

        green_position_frame, green_pos, green_max = get_square_pos(green_frame)

        if green_max > true_green_confidence and self.line_road.get_distance(green_pos[1], green_pos[0]) < true_green_dist_from_road:
            print("What, the true Green!!!", green_max, self.line_road.get_distance(green_pos[1], green_pos[0]))
            self.green_encounter += 1
        else:
            self.green_encounter -= 1
            self.green_encounter = max(int(self.green_encounter*0.8), self.green_encounter)


        if self.green_encounter >= 5:
            self.green_encounter = -10

            self.stage += 1
            self.state = self.LIST[self.stage]


        if self.state == self.GO_STRAIGHT:
            self.go_stanley()
        elif self.state == self.TURN_LEFT:
            self.turn_left()
            self.stage += 1
            self.state = self.LIST[self.stage]
        elif self.state == self.TURN_RIGHT:
            self.turn_right()
            self.stage += 1
            self.state = self.LIST[self.stage]
        elif self.state == self.MISSION:
            print("Try a mission?")
            self.stage += 1
            self.state = self.LIST[self.stage]
        else:
            print("Mission Completed!")
            

        

        cv2.namedWindow('ori')
        cv2.moveWindow('ori', 0, 0)
        cv2.imshow('ori', frame)

        cv2.namedWindow('bev')
        cv2.moveWindow('bev', 700, 0)
        cv2.imshow('bev', bev_frame)

        cv2.namedWindow('filt')
        cv2.moveWindow('filt', 1400, 0)
        cv2.imshow('filt', filter_frame)

        cv2.namedWindow('green')
        cv2.moveWindow('green', 1400, 300)
        cv2.imshow('green', green_frame)
        
        cv2.namedWindow('green_blur')
        cv2.moveWindow('green_blur', 1400, 600)
        cv2.imshow('green_blur', green_position_frame)

        cv2.line(window_frame, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(window_frame)[0])), np.shape(window_frame)[0]), (0, 0, 255), 5)
        cv2.namedWindow('window')
        cv2.moveWindow('window', 0, 600)
        cv2.imshow('window', window_frame)
        
        cv2.waitKey(1)


    def go_stanley(self):
        offset_mm = self.line_road.get_offset(bot_from_bev_x,bot_from_bev_y)
        angle_deg = self.line_road.get_angle()

        kp= 0.001
        k = 0.01
        x = 0.15

        print(offset_mm, angle_deg)
        speed = Twist()
        speed.linear.x = x
        speed.angular.z = (angle_deg + atan(kp*offset_mm)) * k
        self.pub.publish(speed)
        print(speed.angular.z)


    def turn_left(self):
        print("turning left...")

        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 0
        self.pub.publish(speed)
        time.sleep(3)

        speed = Twist()
        speed.linear.x = 1.0
        speed.angular.z = 0
        self.pub.publish(speed)
        time.sleep(0.3)
        speed = Twist()
        speed.linear.x = -1.0
        speed.angular.z = 0
        self.pub.publish(speed)
        time.sleep(0.3)

        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = -1
        self.pub.publish(speed)
        time.sleep(deg_90_sec)
        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 0
        self.pub.publish(speed)


    def turn_right(self):
        print("turning right...")

        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 1
        self.pub.publish(speed)
        time.sleep(deg_90_sec)
        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 0
        self.pub.publish(speed)

    def finished(self):
        while True:
            speed = Twist()
            speed.linear.x = 0
            speed.angular.z = 0.5
            self.pub.publish(speed)
            time.sleep(deg_90_sec)
            speed = Twist()
            speed.linear.x = 0
            speed.angular.z = -0.5
            self.pub.publish(speed)
            time.sleep(deg_90_sec)



if __name__ == "__main__":

    if not rospy.is_shutdown():
        bot_mind()
        rospy.spin()
