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
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
CAM_FRAMERATE = 20
VID_CONNECT_CMD = (
    f'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){CAM_WIDTH}, height=(int){CAM_HEIGHT}, format=(string)NV12, framerate=(fraction){CAM_FRAMERATE}/1 '
    f'! nvvidconv flip-method=0 ! video/x-raw, width=(int){CAM_WIDTH}, height=(int){CAM_HEIGHT}, format=(string)BGRx '
    f'! videoconvert ! video/x-raw, format=(string)BGR '
    f'! appsink'
)
VID_CONNECT_CMD = "log_2125.avi"


IS_LOG = True
IS_LOG_VID = True
IS_LOG_SIGHT = True

IS_SHOW = True

FILE_EACH = "best.pt"
FILE_ALL = "best.pt"



def showing_off(image_list, log="", get_image = False):

    canvas = np.zeros((1000, 1200, 3), dtype=np.uint8) + 255

    pos_x = [0, 740, 0, 300, 600, 900, 0, 0, 0]
    pos_y = [0, 0, 580, 580, 580, 580, 0, 0, 0] 

    for i, image in enumerate(image_list):
        if len(np.shape(image)) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        y, x = np.shape(image)[:2]

        y_i = pos_y[i] + 50
        y_f = y_i + y

        x_i = pos_x[i] + 50
        x_f = x_i + x

        canvas[y_i:y_f, x_i:x_f] = image

    cv2.putText(canvas, log, (20, 960), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(0, 0, 0), thickness=1)

    if not get_image:
        cv2.imshow("canvas", canvas)
    return canvas


class Bot_Mind:

    def __init__(self, show_function = showing_off, go=True):

        self.model_each = YOLO(FILE_EACH)
        self.model_all = YOLO(FILE_ALL) 
        # self.model_all = RTDETR(FILE_ALL) 

        # self.model_each.to('cuda') 
        # self.model_each.to('cuda')
        null_predict_to_turn_on = self.model_each.predict(np.zeros((480, 640, 3)))
        null_predict_to_turn_on = self.model_all.predict(np.zeros((480, 640, 3)))

        now = datetime.datetime.now().strftime("%H%M")
        if IS_LOG_SIGHT:
            self.log_sight_writer = cv2.VideoWriter(f"vlog_sight_{now}.avi", cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (1200, 1000))
        if IS_LOG_VID:
            self.logwriter = cv2.VideoWriter(f"vlog_{now}.avi", cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (640, 480))
        if IS_LOG:
            self.logtxt = open("log_" + now + ".txt", 'w')

        self.pub = TikiMini()
        pub = self.pub
        pub.set_motor_mode(pub.MOTOR_MODE_PID)

        self.mode = StartMode(pub)
        self.mode_pos = 0

        self.count_frame = 1

        self.show_function = show_function

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
            Stanley2GreenMode(pub, 14,  from_it = True, speed_weight=1.3),
            Turn2VoidMode(pub, 15,      is_left=True,   other_turn_sec=0),

            EventMode(pub, self.model_each, 20, n_frame = 5, wait_sec = 1.0),
            Turn2RoadMode(pub, 21,      is_left=False,  min_turn_sec=1.),
            Stanley2CrossMode(pub, 22,  left_way=False, from_it=True, left_offset=0),
            Turn2RoadMode(pub, 23,      is_left=False,  is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 24,  left_offset = -10),
            Turn2VoidMode(pub, 25,      is_left=True,   other_turn_sec=0),

            EventMode(pub, self.model_each, 30, n_frame = 5, wait_sec = 1.0),
            Turn2RoadMode(pub, 31,      is_left=False, min_turn_sec=1.),
            Stanley2GreenMode(pub, 32,  from_it=True, left_offset = -10),
            Turn2VoidMode(pub, 33,      is_left=True, other_turn_sec=0),


            EventMode(pub, self.model_each, 40, n_frame = 5, wait_sec = 1.0),
            Turn2RoadMode(pub, 41,      is_left=False,  min_turn_sec=1.),
            Stanley2CrossMode(pub, 42,  right_way=False),
            Turn2RoadMode(pub, 43,      is_left=True,   min_turn_sec=1., is_curve=True),
            # Stanley2GreenMode(pub, 44, speed_weight = 1.5),

            EndMode(pub, self.model_all, 100),


        ]

        if not go:
            _ = input("Ready?")


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
                self.logtxt.write(f"   / -------   -------\n  *  Capsule Passed: {capsule.keys()}\n   \\ -------   -------\n")

        time_start = time.time()
        self.mode.set_frame_and_move(frame, showoff = IS_SHOW)

        self.mode.log = f"{self.count_frame:04d} : {self.mode.log}"


        if IS_SHOW:
            image_list = self.mode.show_list
            canvas = self.show_function(image_list, self.mode.log)
            if IS_LOG_SIGHT:
                self.log_sight_writer.write(canvas)


        if self.mode.running:
            self.mode.log_add("time: ", time.time() - time_start)
            print(self.mode.log)
            if IS_LOG_VID:
                
                cv2.putText(frame, f"{self.count_frame:04d}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(0, 0, 0), thickness=1)
                self.logwriter.write(frame)
            if IS_LOG:
                self.logtxt.write(self.mode.log + "\n")
        else:
            a = input("Was it good?")
        cv2.waitKey(1)


        if len(VID_CONNECT_CMD) < 30:
            time.sleep(max(0, 0.1 - time.time() + time_start))

if __name__ == "__main__":

    Bot_Mind()
