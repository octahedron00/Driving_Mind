# -*- coding: utf-8 -*-


import cv2
import numpy as np
import time
import math
import datetime
from multiprocessing import Process, Manager

from ultralytics import YOLO, RTDETR

from src.fake_tiki import TikiMini

from src._lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_square_pos, Line
from src._mode import StartMode, EventMode, Stanley2GreenMode, Stanley2CrossMode, Turn2VoidMode, Turn2RoadMode, EndMode
from src._model_second import run_model_second


DO_DETR = False
DO_SECOND = True
DO_SECOND_DETR = True
FILE_EACH = "best.pt"
FILE_SECOND = "rtdetr-l.pt"

IS_LOG = True
IS_LOG_VID = False
IS_LOG_SIGHT = False

IS_SHOW = False


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


def showing_off(image_list, log="", get_image = False):

    pos_x = [0, 700, 900, 1100, 1300, 1500, 0, 0, 0]
    pos_y = [0, 0, 0, 0, 0, 0, 0, 0, 0] 

    if not get_image:
            
        for i, image in enumerate(image_list):
            if i > 8:
                break
            cv2.namedWindow("win_" + str(i+1))
            cv2.moveWindow("win_" + str(i+1), pos_x[i], pos_y[i])
            cv2.imshow("win_" + str(i+1), image)
        return
    

    canvas = np.zeros((1920, 1080, 3), dtype=np.uint8) + 255
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

    return canvas




class Bot_Mind:

    def __init__(self, show_function = showing_off, go=True):



        now = datetime.datetime.now().strftime("%H%M")
        if IS_LOG_SIGHT:
            self.log_sight_writer = cv2.VideoWriter(f"vlog_sight_{now}.avi", cv2.VideoWriter_fourcc(*'MP4V'), CAM_FRAMERATE + 0.0, (CAM_WIDTH, CAM_HEIGHT))
        if IS_LOG_VID:
            self.logwriter = cv2.VideoWriter(f"vlog_{now}.avi", cv2.VideoWriter_fourcc(*'MP4V'), CAM_FRAMERATE + 0.0, (CAM_WIDTH, CAM_HEIGHT))
        if IS_LOG:
            self.logtxt = open("log_" + now + ".txt", 'w')


        # turn on tiki / 맨 첫 줄에는 시작 시간의 V/mA 값이 나옴
        self.pub = TikiMini()
        pub = self.pub
        pub.set_motor_mode(pub.MOTOR_MODE_PID)
        pub.log(f" AI-FORCE  >v< {pub.get_battery_voltage()}V {pub.get_current()}mA")
        

        # second thread 준비, init까지 진행
        manager = Manager()
        self.shared_list = manager.list([None] * 10)
        if DO_SECOND:
            self.thread_model_second = Process(target=run_model_second, args=(pub, FILE_SECOND, self.shared_list, DO_SECOND_DETR))
            self.thread_model_second.start()
        self.shared_list[0] = np.zeros((480, 640, 3))


        # first thread: 모델 하나 준비, RT인지 YOLO인지 확인까지.
        '''단, 이때 반드시 cuda 켜서 진행할 것! 아래 주석 되어있는 건 cpu 상 테스트였기 때문...'''
        if DO_DETR:
            self.model_each = RTDETR(FILE_EACH)
        else:
            self.model_each = YOLO(FILE_EACH)
        # self.model_each.to('cuda')
        null_predict_to_turn_on = self.model_each.predict(np.zeros((480, 640, 3)))


        # 기타 기본 셋팅
        self.mode = StartMode(pub)
        self.mode_pos = 0
        self.count_frame = 1
        self.show_function = show_function

        self.mode_list = [
            StartMode(pub),

            Stanley2CrossMode(pub, 1,   use_green = True),
            Turn2RoadMode(pub, 2,       is_left=False,  is_curve=True),
            Stanley2GreenMode(pub, 3,   left_offset = -10),
            Turn2VoidMode(pub, 4,       is_left=True),

            EventMode(pub, self.model_each, self.shared_list, 10, n_frame = 5, wait_sec = 1.0, show_log= not DO_SECOND),
            Turn2RoadMode(pub, 11,      is_left=True,   min_turn_sec=1),
            Stanley2CrossMode(pub, 12),
            Turn2RoadMode(pub, 13,      is_left=False,  is_curve=True,  min_turn_sec=1.),
            Stanley2GreenMode(pub, 14,  from_it = True, speed_weight=1.3),
            Turn2VoidMode(pub, 15,      is_left=True),

            EventMode(pub, self.model_each, self.shared_list, 20, n_frame = 5, wait_sec = 1.0, show_log= not DO_SECOND),
            Turn2RoadMode(pub, 21,      is_left=False,  min_turn_sec=1.),
            Stanley2CrossMode(pub, 22,  left_way=False, from_it=True, left_offset=0),
            Turn2RoadMode(pub, 23,      is_left=False,  is_curve=True, min_turn_sec=1.),
            Stanley2GreenMode(pub, 24,  left_offset = -10),
            Turn2VoidMode(pub, 25,      is_left=True),

            EventMode(pub, self.model_each, self.shared_list, 30, n_frame = 5, wait_sec = 1.0, show_log= not DO_SECOND),
            Turn2RoadMode(pub, 31,      is_left=False, min_turn_sec=1.),
            Stanley2GreenMode(pub, 32,  from_it=True, left_offset = -10),
            Turn2VoidMode(pub, 33,      is_left=True),

            EventMode(pub, self.model_each, self.shared_list, 40, n_frame = 5, wait_sec = 1.0, show_log= not DO_SECOND),
            Turn2RoadMode(pub, 41,      is_left=False,  min_turn_sec=1.),
            Stanley2CrossMode(pub, 42,  right_way=False),
            Turn2RoadMode(pub, 43,      is_left=True,   min_turn_sec=1., is_curve=True),
            # Stanley2GreenMode(pub, 44, speed_weight = 1.5),

            EndMode(pub, None, 100, predict_all=False),
        ]

        # 모델 부르기 등이 오래 걸릴 수 있으니, 출발 자체는 엔터 한 번으로 진행
        # input 하나 받는 걸로 준비해두기.
        if not go:
            _ = input("Ready?")


        '''Running Part Here'''
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
        
        # 끝났다고 하면, 다음 모드 켜고 그 사이 capsule 운반.
        # capsule 안에는 기존 촬영 이미지나 사거리부터의 거리 등, 
        # Mode 사이에서 전달이 필요한 변수들이 들어간다
        if self.mode.end:
            capsule = self.mode.capsule
            self.mode_pos += 1
            self.mode = self.mode_list[self.mode_pos]
            self.mode.capsule = capsule
            if IS_LOG:
                self.logtxt.write(f"   / -------   -------\n  *  Capsule Passed: {capsule.keys()}\n   \\ -------   -------\n")


        # 시간 재고 바로 실행
        time_start = time.time()
        self.mode.set_frame_and_move(frame, showoff = IS_SHOW)
        self.mode.log = f"{self.count_frame:04d} : {self.mode.log}"


        # 필요 시 출력까지
        if IS_SHOW:
            image_list = self.mode.show_list
            canvas = self.show_function(image_list, self.mode.log, IS_LOG_SIGHT)
            if IS_LOG_SIGHT:
                self.log_sight_writer.write(canvas)


        # 로그 출력까지. 
        if self.mode.running:
            self.mode.log_add("time: ", time.time() - time_start)
            self.mode.log_add(f"[ battery: {int((self.pub.get_battery_voltage() - 9.5)*100/(12.6-9.5)):02d}%,",
                              f"{self.pub.get_battery_voltage()}V / {self.pub.get_current()}mA ]")
            print(self.mode.log)
            if IS_LOG_VID:
                
                cv2.putText(frame, f"{self.count_frame:04d}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(0, 0, 0), thickness=1)
                self.logwriter.write(frame)
            if IS_LOG:
                self.logtxt.write(self.mode.log + "\n")
        else:
            if DO_SECOND:
                self.thread_model_second.join()
            _ = input("Was it good?")
            return
        cv2.waitKey(1)


        if len(VID_CONNECT_CMD) < 30:
            time.sleep(max(0, 0.1 - time.time() + time_start))

if __name__ == "__main__":

    Bot_Mind()
