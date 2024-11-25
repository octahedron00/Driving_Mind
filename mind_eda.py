
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import math
import sys

from math import *

from src._lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur, get_mm_px_from_cm
from src._lane_detect import get_cm_px_from_mm, get_square_pos, get_road_edge_angle, get_sliding_window_and_cross_result, Line
from src._lane_detect import LT_BEV, LD_BEV, RD_BEV, RT_BEV, H_BEV, W_BEV



def on_click(event, x, y, flags, images):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:    
        image, ori = images[0], images[1]
        if np.shape(image)[0] != np.shape(ori)[0]:
            ori = image
        hls = cv2.cvtColor(ori, cv2.COLOR_BGR2HLS)
        print("x =", x, " | y =", y ," | value =", image[y, x], " | original =",ori[y,x], "| hls_value =", hls[y, x])
 


def show(frame, frame_before, stop=False):

    bev = get_bev(frame)
    road_bev = get_road(bev)
    green_bev = get_green(bev)
    green_bev_cm = get_cm_px_from_mm(green_bev)
    green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 7)
    edge_bev, _ = get_road_edge_angle(road_bev)
    cross_bev, _, _, _, _ = get_sliding_window_and_cross_result(road_bev)
    
    polypoint = np.array([LT_BEV, LD_BEV, RD_BEV, RT_BEV], dtype=np.int32)
    # print(polypoint)

    pos_x = [0, 460, 660, 860, 1060, 1260, 1460, 0, 0]
    pos_y = [0, 0, 0, 0, 0, 0, 0, 0, 0] 

    cv2.polylines(frame, [polypoint], isClosed=True, color=(0, 255, 0), thickness=2)

    frame_list = [
        frame,
        bev,
        road_bev,
        green_bev,
        edge_bev,
        cross_bev,
        get_mm_px_from_cm(green_blur_bev),
        ]
    name_list = [
        'ori',
        'bev',
        'road',
        'green',
        'edge',
        'cross',
        'green_blur',
        ]
    for i, f in enumerate(frame_list):

        if i > 8:
            break
        cv2.namedWindow(name_list[i])
        cv2.moveWindow(name_list[i], pos_x[i], pos_y[i])
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
#    print(a)
    if 120 > a > 30:
        a = cv2.waitKey(0)
        if a in [ord('a'), ord('b')]:
            if len(frame_before) > 0:
                show(frame_before[-1], frame_before[:-1], stop=True)
            else:
                print("No More Data Available")




if __name__=="__main__":
    
    video_file = sys.argv[1]
     
    print("Start of Video?")
    cap = cv2.VideoCapture(video_file)
    print(cap)

    frame_before = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of Video?")
            break

        show(frame, frame_before)
 
        frame_before.append(frame)
        if len(frame_before) > 20:
            frame_before = frame_before[1:]
 
 
    # close the window
    cv2.destroyAllWindows()





