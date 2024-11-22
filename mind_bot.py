#! /usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import time
import math
import datetime

from ultralytics import YOLO, RTDETR

from fake_tiki import TikiMini

from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_square_pos, Line
from _mode import StartMode, EventMode, Stanley2GreenMode, Stanley2CrossMode, Turn2VoidMode, Turn2RoadMode, EndMode



FRAME_IGNORE_LEVEL = 1
VID_CONNECT_CMD = "log_2125.avi"


IS_LOG = True
IS_LOG_VID = False


FILE_EACH = "best.pt"
FILE_ALL = "best.pt"

class bot_mind:

    def __init__(self):

        self.model_each = YOLO(FILE_EACH)
        self.model_all = YOLO(FILE_ALL) 
        # self.model_all = RTDETR(FILE_ALL) 

        # self.model_each.to('cuda') 
        # self.model_each.to('cuda')
        null_predict_to_turn_on = self.model_each.predict(np.zeros((480, 640, 3)))
        null_predict_to_turn_on = self.model_all.predict(np.zeros((480, 640, 3)))

        now = datetime.datetime.now().strftime("%H%M")
        if IS_LOG_VID:
            self.logwriter = cv2.VideoWriter("log_" + now + ".avi", cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (640, 480))
        if IS_LOG:
            self.logtxt = open("log_" + now + ".txt", 'w')

        self.pub = TikiMini()
        pub = self.pub
        pub.set_motor_mode(pub.MOTOR_MODE_PID)

        self.mode = StartMode(pub)
        self.mode_pos = 0

        self.count_frame = 1

        self.mode_list = [
            StartMode(pub),

            Stanley2CrossMode(pub, 1,   use_green = True),
            Turn2RoadMode(pub, 2,       is_left=False,  is_curve=True),
            Stanley2GreenMode(pub, 3,   left_offset = -10),
            Turn2VoidMode(pub, 4,       is_left=True,   other_turn_sec=0),

            EventMode(pub, self.model_each, 10, n_frame = 5, wait_sec = 0),
            Turn2RoadMode(pub, 11,      is_left=True,   min_turn_sec=1),
            Stanley2CrossMode(pub, 12),
            Turn2RoadMode(pub, 13,      is_left=False,  is_curve=True,  min_turn_sec=1.),
            Stanley2GreenMode(pub, 14,  from_it = True),
            Turn2VoidMode(pub, 15,      is_left=True,   other_turn_sec=0),

            EventMode(pub, self.model_each, 20, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 21,      is_left=False,  min_turn_sec=1.),
            Stanley2CrossMode(pub, 22,  left_way=False, from_it=True, left_offset=0),
            Turn2RoadMode(pub, 23,      is_left=False,  is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 24,  left_offset = -10),
            Turn2VoidMode(pub, 25,      is_left=True,   other_turn_sec=0),

            EventMode(pub, self.model_each, 30, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 31,      is_left=False, min_turn_sec=1.),
            Stanley2GreenMode(pub, 32,  from_it=True, left_offset = -10),
            Turn2VoidMode(pub, 33,      is_left=True, other_turn_sec=0),


            EventMode(pub, self.model_each, 40, n_frame = 5, wait_sec = 2.0),
            Turn2RoadMode(pub, 41,      is_left=False,  min_turn_sec=1.),
            Stanley2CrossMode(pub, 42,  right_way=False),
            Turn2RoadMode(pub, 43,      is_left=True,   min_turn_sec=1., is_curve=True),
            # Stanley2GreenMode(pub, 44),

            EndMode(pub, self.model_all, 100),


            Turn2RoadMode(pub, 101, is_left=False),
            Stanley2CrossMode(pub, 102),
            Turn2RoadMode(pub, 103, is_left=False, min_turn_sec=1., is_curve=True),
            Stanley2GreenMode(pub, 104),
            Stanley2CrossMode(pub, 105),
            Turn2RoadMode(pub, 106, is_left=True, min_turn_sec=1., is_curve=True),
            Stanley2CrossMode(pub, 107),
            Turn2RoadMode(pub, 108, is_left=True, is_curve=True),
            Stanley2GreenMode(pub, 109, left_offset = -10),
            Turn2VoidMode(pub, 110, is_left=True, other_turn_sec=0),

            Turn2RoadMode(pub, 111, is_left=True, min_turn_sec=1),
            Stanley2CrossMode(pub, 112),
            Turn2RoadMode(pub, 113, is_left=True, is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 114, from_it=True),
            Turn2RoadMode(pub, 115, is_left=True),


            EndMode(pub, 200),

        ]

        cap = cv2.VideoCapture(VID_CONNECT_CMD)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.count_frame += 1
            if self.count_frame % FRAME_IGNORE_LEVEL == 0:
                self.action(frame)
        print("cap gone")
        

    def action(self, frame):
        
        if self.mode.end:
            capsule = self.mode.capsule
            self.mode_pos += 1
            self.mode = self.mode_list[self.mode_pos]
            self.mode.capsule = capsule
            if IS_LOG:
                self.logtxt.write(f">>> -------   -------\n>>> Capsule Passed: {capsule.keys()}\n>>> -------   -------\n")

        time_start = time.time()
        self.mode.set_frame_and_move(frame, showoff = True)

        if self.mode.running:
            self.mode.log_add("time: ", time.time() - time_start)
            print(self.mode.log)
            if IS_LOG_VID:
                self.logwriter.write(frame)
            if IS_LOG:
                self.logtxt.write(self.mode.log + "\n")
        else:
            a = input("Was it good?")
        cv2.waitKey(50)


if __name__ == "__main__":

    bot_mind()
