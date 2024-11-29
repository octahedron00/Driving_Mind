#! /usr/bin/env python
# -*- coding: utf-8 -*-
# %matplotlib inline


import datetime
import os
import cv2
import numpy as np
import time
import math

from ultralytics import YOLO, RTDETR

from src._lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur, get_mm_px_from_cm 
from src._lane_detect import get_cm_px_from_mm, get_square_pos, get_road_edge_angle, get_sliding_window_and_cross_result
from src._lane_detect import get_resize_image_4_model, get_pos_before_xy, Line

#for fun
from src._sing import sing




mtx = np.array([[624.021794, 0, 705.539195],
                [0, 624.719173, 398.307132],
                [0, 0, 1]])
dist = np.array([[-0.318379, 0.108202, -0.000758, 0.000421, -0.016728]])

roi = (0, 0, 1279, 959)

def calibrate(img):
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, mtx) #newcameramtx==mtx
    # crop the image
    x, y, w, h = roi
    dst = dst[y+250:y+h, x:x+w]

    return dst




if __name__ == "__main__":

    cap = cv2.VideoCapture("vlog_1039.avi")

    logwriter = cv2.VideoWriter(f"vlog_control.avi", cv2.VideoWriter_fourcc(*"MJPG"), 16 + 0.0, (640, 640))


    while cap.isOpened():


        ret, frame = cap.read()

        if not ret:
            break
        frame_2 = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_LINEAR)


        calib_frame = calibrate(frame_2)

        print(np.shape(calib_frame))

        # cv2.imshow(calib_frame)
        # cv2.waitKey(0)
        calib_frame = cv2.resize(calib_frame, (640, 640), interpolation=cv2.INTER_LINEAR)

        logwriter.write(calib_frame)
    
    logwriter.release()
        

