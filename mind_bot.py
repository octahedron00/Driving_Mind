# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import time
import math
import datetime
from multiprocessing import Process, Manager

from ultralytics import YOLO, RTDETR

from tiki.mini import TikiMini

from src._lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_square_pos, Line
from src._mode import StartMode, EventMode, Stanley2GreenMode, Stanley2CrossMode, Turn2VoidMode, Turn2RoadMode, EndMode, _SheepMode
from src._model_second import run_model_second


DO_MODEL = True

DO_DETR = False
DO_SECOND = False
DO_SECOND_DETR = False
FILE_EACH = "best.pt"


DO_ENSEMBLE = False

MODEL_FIRST_LIST = ["yolo11s.pt"]
MODEL_SECOND_LIST = ["best.pt", 'yolo11s.pt']

INF_FRAMES = 3
STEP_FOR_CAM = 0


IS_LOG = True
IS_LOG_VID = True
IS_LOG_SIGHT = False

IS_SHOW = True


FRAME_IGNORE_LEVEL = 1
CAM_WIDTH = 1280
CAM_HEIGHT = 960
CAM_FRAMERATE = 16
VID_CONNECT_CMD = (
    f'nvarguscamerasrc awbmode=manual aeLock=true exposuretimerange=\"5000000 5000000\" ! video/x-raw(memory:NVMM), width={CAM_WIDTH}, height={CAM_HEIGHT}, format=(string)NV12, framerate=(fraction){CAM_FRAMERATE}/1 '
    f'! nvvidconv flip-method=2 ! video/x-raw, width=(int){CAM_WIDTH}, height=(int){CAM_HEIGHT}, format=(string)BGRx '
    f'! videoconvert ! video/x-raw, format=(string)BGR '
    f'! appsink max-buffers=1 drop=True'
)

# gamma add
VID_CONNECT_CMD = (
    f'nvarguscamerasrc awbmode=manual gammavalue=1.0 aeLock=true exposuretimerange=\"5000000 5000000\" ! video/x-raw(memory:NVMM), width={CAM_WIDTH}, height={CAM_HEIGHT}, format=(string)NV12, framerate=(fraction){CAM_FRAMERATE}/1 '
    f'! nvvidconv flip-method=2 ! video/x-raw, width=(int){CAM_WIDTH}, height=(int){CAM_HEIGHT}, format=(string)BGRx '
    f'! videoconvert ! video/x-raw, format=(string)BGR '
    f'! appsink max-buffers=1 drop=True'
)

SLEEP_SEC = 0.2


def showing_off(image_list, log="", get_image = False):
    # 다른 곳에서 이미지 받아가는 게 아니면, 창 여러 개 출력
    # 다른 곳에서 이미지 받아가는 게 맞다면, 그냥 큰 이미지에 짬뽕해서 되돌려주기

    pos_x = [0, 900, 1100, 1300, 1500, 1700, 0, 0]
    pos_y = [0, 0, 0, 0, 0, 0, 0, 0, 0] 

    # 1920 1080은 화면 전체를 채우니까,
    # 여기서 resize 한 번 하고 보여주기로 합시다.
    # 게다가 inference한 것도 사이즈 맞게 변할 것: 640 640이 다시 펴지는 방향으로

    # 다른 곳에서 이미지 받아가는 게 아니면, 창 여러 개 출력
    if not get_image:
        return
        for i, image in enumerate(image_list):
            if i > 8:
                break
            if i == 0:
                image = cv2.resize(image.copy(), (960, 540))
            cv2.namedWindow("win_" + str(i+1))
            cv2.moveWindow("win_" + str(i+1), pos_x[i], pos_y[i])
            cv2.imshow("win_" + str(i+1), image)
        return
    

    # 다른 곳에서 이미지 받아가는 게 맞다면, 그냥 큰 이미지에 짬뽕해서 되돌려주기
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8) + 255
    for i, image in enumerate(image_list):
        if len(np.shape(image)) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        y, x = np.shape(image)[:2]
        y_i = pos_y[i]
        y_f = y_i + y
        x_i = pos_x[i]
        x_f = x_i + x
        canvas[y_i:y_f, x_i:x_f] = image
    cv2.putText(canvas, log, (20, 960), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0), thickness=1)

    return canvas




class Bot_Mind:

    def __init__(self, show_function = showing_off, go=False):


        # 로그파일 만들기
        now = datetime.datetime.now().strftime("%H%M")
        if IS_LOG_SIGHT:
            self.log_sight_writer = cv2.VideoWriter(os.path.join("log", f"vlog_sight_{now}.avi"), cv2.VideoWriter_fourcc(*'MJPG'), CAM_FRAMERATE + 0.0, (1920, 1080))
        if IS_LOG_VID:
            self.logwriter = cv2.VideoWriter(os.path.join("log", f"vlog_{now}.avi"), cv2.VideoWriter_fourcc(*'MJPG'), CAM_FRAMERATE + 0.0, (int(CAM_WIDTH/2), int(CAM_HEIGHT/2)))
        if IS_LOG:
            self.logtxt = open(os.path.join("log", f"log_{now}.txt"), 'w')


        # turn on tiki / 맨 첫 줄에는 시작 시간의 V/mA 값이 나옴
        self.pub = TikiMini()
        pub = self.pub
        pub.set_motor_mode(pub.MOTOR_MODE_PID)
        pub.set_motor_power(pub.MOTOR_LEFT, 0)
        pub.set_motor_power(pub.MOTOR_RIGHT, 0)
        pub.stop_buzzer()
        pub.log_clear()
        for i in range(8):
            pub.set_led_color(i, 50, 50, 50)
        

        # second thread 준비, init까지 진행
        manager = Manager()
        self.shared_list = manager.list([None] * 10)
        if DO_SECOND:
            self.thread_model_second = Process(target=run_model_second, args=(pub, MODEL_SECOND_LIST, self.shared_list, DO_SECOND_DETR))
            self.thread_model_second.start()
            self.shared_list[0] = ([np.zeros((640, 640, 3))], [])


        self.models = []
        if DO_MODEL:
            # first thread: 모델 하나 준비, RT인지 YOLO인지 확인까지.
            '''단, 이때 반드시 cuda 켜서 진행할 것! 아래 주석 되어있는 건 cpu 상 테스트였기 때문...'''
            if DO_DETR:
                self.models = [RTDETR(FILE_EACH)]
            elif DO_ENSEMBLE:
                for model_address in MODEL_FIRST_LIST:
                    self.models.append(YOLO(model_address))
            else:
                self.models = [YOLO(FILE_EACH)]
            
            for model in self.models:
                model.to('cuda')
                null_predict_to_turn_on = model.predict(np.zeros((640, 640, 3)), device=0)


        # 기타 기본 셋팅
        self.mode = StartMode(pub)
        self.mode_pos = 0
        self.count_frame = 1
        self.show_function = show_function

        self.mode_list = [
            StartMode(pub),

            # Turn2VoidMode(pub, 4,       is_left=True),
            # Turn2RoadMode(pub, 13,      is_left=False),

            Stanley2GreenMode(pub, 1, speeding_time=0.0),
            Turn2RoadMode(pub, 2,       is_left=True,  is_curve=True),
            Stanley2GreenMode(pub, 3,   left_offset = 0, prefer_dist=220),
            _SheepMode(pub, 3.5, sleep_sec = SLEEP_SEC),
            Turn2VoidMode(pub, 4,       is_left=True),

            EventMode(pub, self.models, self.shared_list, 10, n_frame = INF_FRAMES, wait_sec = 0.3, show_log= not DO_SECOND, step_for_cam=STEP_FOR_CAM),
            Turn2RoadMode(pub, 11,      is_left=True),
            Stanley2GreenMode(pub, 12),
            Turn2RoadMode(pub, 13,      is_left=True,  is_curve=True),
            Stanley2GreenMode(pub, 14,  from_it = True, speed_weight=1, prefer_dist=250, speeding_time=0.0),
            _SheepMode(pub, 0, sleep_sec = SLEEP_SEC),
            Turn2VoidMode(pub, 15,      is_left=True),

            EventMode(pub, self.models, self.shared_list, 20, n_frame = INF_FRAMES, wait_sec = 0.3, show_log= not DO_SECOND, step_for_cam=STEP_FOR_CAM),
            Turn2RoadMode(pub, 21,      is_left=False),
            Stanley2GreenMode(pub, 22,  from_it=True),
            Turn2RoadMode(pub, 23,      is_left=False,  is_curve=True),
            Stanley2GreenMode(pub, 24,  prefer_dist=260, speeding_time=0.0),
            _SheepMode(pub, 24.5,   sleep_sec = SLEEP_SEC),
            Turn2VoidMode(pub, 25,      is_left=True),

            EventMode(pub, self.models, self.shared_list, 30, n_frame = INF_FRAMES, wait_sec = 0.3, show_log= not DO_SECOND, step_for_cam=STEP_FOR_CAM),
            Turn2RoadMode(pub, 31,      is_left=False),
            Stanley2GreenMode(pub, 32,  from_it=True, prefer_dist=230, speeding_time=0.0),
            _SheepMode(pub, 32.5, sleep_sec = SLEEP_SEC),
            Turn2VoidMode(pub, 33,      is_left=True),

            EventMode(pub, self.models, self.shared_list, 40, n_frame = INF_FRAMES, wait_sec = 0.3, show_log= not DO_SECOND, step_for_cam=STEP_FOR_CAM),
            Turn2RoadMode(pub, 41,      is_left=False),
            Stanley2GreenMode(pub, 42, from_it=True),
            Turn2RoadMode(pub, 43,      is_left=True, is_curve=True),
            Stanley2GreenMode(pub, 44, speed_weight = 1, prefer_dist=200),

            EndMode(pub, None, 100, predict_all=False),

            # Turn2RoadMode(pub, 100,       is_left=False, min_turn_sec=2.6),            

            # Stanley2CrossMode(pub, 101, speeding_time=0.0, prefer_dist=220, front_way=False),
            # Turn2RoadMode(pub, 102,       is_left=False,  is_curve=True, z_ratio=0.3),
            # Stanley2CrossMode(pub, 103, speeding_time=0.0, left_way=False, prefer_dist=220, from_it=True),
            # Stanley2CrossMode(pub, 104, speeding_time=0.0, prefer_dist=220, front_way=False, from_it=True),
            # Turn2RoadMode(pub, 105,       is_left=True,  is_curve=True, z_ratio=0.3),
            # 
            # Stanley2GreenMode(pub, 106, prefer_dist=400),
            # Stanley2CrossMode(pub, 107, speeding_time=0.0, prefer_dist=220, from_it=True),
            # Stanley2CrossMode(pub, 108, speeding_time=0.0, prefer_dist=220, from_it=True),
            # Turn2RoadMode(pub, 109,       is_left=False, min_turn_sec=2.6),
            # _SheepMode(pub, 110, sleep_sec = 1.5, x_speed = -0.12),

            # EndMode(pub, None, 0),


        ]

        # 모델 부르기 등이 오래 걸릴 수 있으니, 출발 자체는 엔터 한 번으로 진행
        # input 하나 받는 걸로 준비해두기.
        if not go:
            _ = input("Ready?")


        self.time_taken_by_mode = time.time()

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
                self.logtxt.write(f"   / -------   -------\n  *  Time Passed: {time.time() - self.time_taken_by_mode}\n   \\ -------   -------\n")
                self.logtxt.write(f"   / -------   -------\n  *  Capsule Passed: {capsule.keys()}\n   \\ -------   -------\n")
            self.time_taken_by_mode = time.time()

        # 시간 재고 바로 실행
        time_start = time.time()
        self.mode.set_frame_and_move(frame, showoff = IS_SHOW)
        self.mode.log = f"{self.count_frame:04d} : {self.mode.log}"


        # 화면 보이기, 필요 시 그대로 record까지 진행.
        # 이때, is_log_sight 켜진 상태로 진행하면 조금 느려진다! 감안해야 함... 
        if IS_SHOW:
            image_list = self.mode.show_list
            canvas = self.show_function(image_list, self.mode.log, IS_LOG_SIGHT)
            if IS_LOG_SIGHT:
                self.log_sight_writer.write(canvas)


        # 로그 출력
        if self.mode.running:
            self.mode.log_add("time: ", time.time() - time_start)
            # self.mode.log_add(f"[ battery: {self.pub.get_battery_voltage()}V / {self.pub.get_current()}mA ]")
            print(self.mode.log)
            if IS_LOG:
                self.logtxt.write(self.mode.log + "\n")
            
            # vlog 만들 때, 번호를 그려주는 방식으로.
            if IS_LOG_VID:
                frame_write = cv2.resize(frame, dsize=(int(CAM_WIDTH/2), int(CAM_HEIGHT/2)))
                cv2.putText(frame_write, f"{self.count_frame:04d}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(255,255,255), thickness=1)
                cv2.putText(frame_write, self.mode.log, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color=(255,255,255), thickness=1)
                self.logwriter.write(frame_write)
        else:
            if DO_SECOND:
                self.thread_model_second.join()
            _ = input("Was it good?")
            return
        cv2.waitKey(1)


if __name__ == "__main__":

    Bot_Mind()
