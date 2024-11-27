
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
 


# 급조한 재귀함수입니다
def show(frame, frame_before, stop=False):

    bev = get_bev(frame)
    road_bev = get_road(bev)
    green_bev = get_green(bev)
    green_bev_cm = get_cm_px_from_mm(green_bev)
    green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 7)
    edge_bev, _ = get_road_edge_angle(road_bev)
    cross_bev, _, _, _, _ = get_sliding_window_and_cross_result(road_bev)
    
    # BEV만드는 ROI는 여기서 만듭니다!
    polypoint = np.array([LT_BEV, LD_BEV, RD_BEV, RT_BEV], dtype=np.int32)
    cv2.polylines(frame, [polypoint], isClosed=True, color=(0, 255, 0), thickness=2)
    # print(polypoint)

    # 각 창의 position 구성. 그냥 맨 윗줄에 싹 모아두기. 제일 큰 화면이 맨 뒤로 감
    pos_x = [0, 460, 660, 860, 1060, 1260, 1460, 0, 0]
    pos_y = [0, 0, 0, 0, 0, 0, 0, 0, 0] 


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


    # 프레임 앞뒤로 움직이기 가능하게 함, 
    # 일단 멈추고 싶다면 아무 키나 누르기,
    # 뒤로 가고 싶다면 A, B 키를 누르기: 약 20프레임 정도 저장됨(아래에서 수정)
    # 앞으로 다시 돌아가려면 D, F 누르기:
    # 다시 재생하려면 다른 아무 키나 누르기.
    # 재귀 함수 구현이라, 자세한 설명을 적으면 너무 길어집니다. 일단 잘 작동함!
    # 
    # 유일한 문제 상황은, RAM이 작을 때. 가능하면 local에서만 돌릴 것!
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
                # 한 프레임 전으로
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

        frame = np.array(frame[::-1])
        if not ret:
            print("End of Video?")
            break

        show(frame, frame_before)
 
        frame_before.append(frame)
        if len(frame_before) > 20:
            frame_before = frame_before[1:]
 
 
    # close the window
    cv2.destroyAllWindows()





