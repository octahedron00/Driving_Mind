
#! /usr/bin/env python
# -*- coding: utf-8 -*-

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



if __name__ == "__main__":

    cap = cv2.VideoCapture("omo2.mp4")
    print(cap)
    # ret, frame = cap.read()
    # print(ret, frame)

    init_pos_for_sliding_windows = -1

    stage = 0
    state = 0
    green_encounter = 0
    line_road = None

    true_green_confidence = 100
    true_green_dist_from_road = 20 #mm



    GO_STRAIGHT = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    MISSION = 4
    # GO_CURVE = 5
    WELL_DONE = -1

    LIST = [0, 1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 4, 3, 1, -1]


    while cap.isOpened():
        
        state = LIST[stage]

        ret, frame = cap.read()

        if not ret:
            print("End of Video?")
            break

        bev_frame, Minv = get_bev(frame)

        filter_frame = get_road(bev_frame)

        green_frame = get_green(bev_frame)

        window_frame, x_list, y_list = get_sliding_window_result(filter_frame, init_pos_for_sliding_windows)

        if len(x_list) > 3:
            init_pos_for_sliding_windows = x_list[0]
            line_road = Line(x_list, y_list)

            print("x = ", line_road.var_1, " y + ", line_road.var_0)
            print("Dist from origin: ", line_road.get_offset(100,400), ", angle: ", line_road.get_angle())
        else:
            init_pos_for_sliding_windows = -1
        
        if line_road == None:
            continue

        green_position_frame, green_pos, green_max = get_square_pos(green_frame)

        if green_max > true_green_confidence and line_road.get_distance(green_pos[1], green_pos[0]) < true_green_dist_from_road:
            print("What, the true Green!!!", green_max, line_road.get_distance(green_pos[1], green_pos[0]))
            green_encounter += 1
        else:
            green_encounter -= 1
            green_encounter = max(int(green_encounter*0.8), green_encounter)


        if green_encounter >= 5:
            a = input("")
            green_encounter = 0


        # green_position

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

        cv2.line(window_frame, (int(line_road.calc(0)), 0), (int(line_road.calc(np.shape(window_frame)[0])), np.shape(window_frame)[0]), (0, 0, 255), 5)
        cv2.namedWindow('window')
        cv2.moveWindow('window', 0, 600)
        cv2.imshow('window', window_frame)


        # a = input("")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()