
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

from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur
from _lane_detect import get_cm_px_from_mm, get_square_pos, get_road_edge_angle, get_road_and_cross_pos, Line


video_file = "log_1457.avi"


def on_click(event, x, y, flags, images):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:    
        image, ori = images[0], images[1]
        if np.shape(image)[0] != np.shape(ori)[0]:
            ori = image
        hls = cv2.cvtColor(ori, cv2.COLOR_BGR2HLS)
        print("x =", x, " | y =", y ," | value =", image[y, x], " | original =",ori[y,x], "| hls_value =", hls[y, x])
 


def show(frame, frame_before, stop=False):

    bev, Minv = get_bev(frame)
    road_bev = get_road(bev)
    green_bev = get_green(bev)
    green_bev_cm = get_cm_px_from_mm(green_bev)
    green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 7)
    edge_bev, _ = get_road_edge_angle(road_bev)
    cross_bev, _, _, _, _ = get_road_and_cross_pos(road_bev)
    

    frame_list = [
        frame,
        bev,
        road_bev,
        green_bev,
        green_blur_bev,
        edge_bev,
        cross_bev,]
    name_list = [
        'ori',
        'bev',
        'road',
        'green',
        'green_pos',
        'road_edge',
        'cross',]
    for i, f in enumerate(frame_list):

        if i > 8:
            break
        cv2.namedWindow(name_list[i])
        cv2.moveWindow(name_list[i], px[i], py[i])
        cv2.imshow(name_list[i], f)
        cv2.setMouseCallback(name_list[i], on_click, (f, bev))

    if stop:
        a = cv2.waitKey(0)
        if a in [ord('a'), ord('b')]:
            if len(frame_before) > 0:
                k = show(frame_before[-1], frame_before[:-1], stop=True)
                if k > 0:
                    k = show(frame, frame_before, stop=True)
                    return k
            else:
                print("No More Data Available")
        elif a in [ord('d'), ord('f')]:
            return 1
        return 0
    a = cv2.waitKey(50)
    print(a)
    if 120 > a > 30:
        a = cv2.waitKey(0)
        if a in [ord('a'), ord('b')]:
            if len(frame_before) > 0:
                show(frame_before[-1], frame_before[:-1], stop=True)
            else:
                print("No More Data Available")

if __name__=="__main__":
 
    print("Start of Video?")
    cap = cv2.VideoCapture(video_file)
    print(cap)

    px = [0, 600, 1200, 0, 600, 1200, 0, 600, 1200]
    py = [0, 0, 0, 400, 400, 400, 800, 800, 800]

    frame_before = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of Video?")
            break

        show(frame, frame_before)
 
        frame_before.append(frame)
        if len(frame_before) > 5:
            frame_before = frame_before[1:]
 
 
    # close the window
    cv2.destroyAllWindows()





