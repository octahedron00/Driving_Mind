# -*- coding: utf-8 -*-

import os
import time

import cv2
import datetime

import keyboard

from tiki.mini import TikiMini

from src._mode import move_robot
from mind_bot import showing_off, VID_CONNECT_CMD, FRAME_IGNORE_LEVEL, CAM_WIDTH, CAM_HEIGHT, CAM_FRAMERATE


class Control_Mind:

    def __init__(self, show_function=showing_off, keyboard_listener=keyboard):

        now = datetime.datetime.now().strftime("%H%M")
        self.logwriter = None

        self.pub = TikiMini()
        self.pub.set_motor_mode(self.pub.MOTOR_MODE_PID)

        self.count_frame = 1

        self.image_name = f"image_{now}"
        try:
            os.makedirs(os.path.join("img", self.image_name))
        except:
            pass
        self.image_count = 1

        self.show_function = show_function
        self.keyboard_listener = keyboard_listener

        self.speed_x = 0
        self.speed_z = 0

        # 조금 길게 눌렀다는 이유로 2번 인식되지 않도록 추가함: 각각 촬영 / 녹화온오프 버튼에 대함
        self.clicked_c = False
        self.clicked_v = False

        cap = cv2.VideoCapture(VID_CONNECT_CMD)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.count_frame += 1
            if self.count_frame % FRAME_IGNORE_LEVEL == 0:
                self.action(frame)
        
        cv2.destroyAllWindows()

    def action(self, frame):
        pub = self.pub

        self.show_function([frame])

        if self.logwriter != None:
            self.logwriter.write(frame)

        # 앞뒤좌우 단순하게
        if self.keyboard_listener.is_pressed("w"):
            self.speed_x = 0.3
        elif self.keyboard_listener.is_pressed("s"):
            self.speed_x = -0.3
        else:
            self.speed_x = 0

        if self.keyboard_listener.is_pressed("a"):
            self.speed_z = 0.5
        elif keyboard.is_pressed("d"):
            self.speed_z = -0.5
        else:
            self.speed_z = 0

        # c: 캡쳐, 사진 만들기
        if self.keyboard_listener.is_pressed("c"):
            if not self.clicked_c:
                cv2.imwrite(os.path.join("img", self.image_name, f"{self.image_name}_{self.image_count:04d}.jpg"), frame)
                print(f"{self.image_name}_{self.image_count:04d}.jpg is saved")
                self.image_count += 1
            self.clicked_c = True
        else:
            self.clicked_c = False


        # n: 캡쳐 경로 바꾸기
        # img/이름/이름_번호.jpg로 저장됨, 번호는 그 안 파일 갯수 다음으로 지정됨
        if self.keyboard_listener.is_pressed("n"):
            
            self.image_name = input("new dataset name:")
            if len(self.image_name) < 1:
                now = datetime.datetime.now().strftime("%H%M")
                self.image_name = f"image_{now}"
            try:
                os.makedirs(os.path.join("img", self.image_name))
            except:
                pass
            self.image_count = len(os.listdir(os.path.join("img", self.image_name))) + 1




        if self.keyboard_listener.is_pressed("v"):
            if not self.clicked_v:
                self.clicked_v = True
                if self.logwriter == None:
                    now = datetime.datetime.now().strftime("%H%M")
                    self.logwriter = cv2.VideoWriter(f"vlog_control_{now}.avi", cv2.VideoWriter_fourcc(*"MJPG"), CAM_FRAMERATE + 0.0, (CAM_WIDTH, CAM_HEIGHT))
                    print(f"Recording Start : vlog_control_{now}.avi")
                else:
                    print("Recording finished")
                    self.logwriter.release()
                    self.logwriter = None
        else:
            self.clicked_v = False

        move_robot(pub, self.speed_x, self.speed_z)

        cv2.waitKey(1)
        # time.sleep(0.05)


if __name__ == "__main__":

    Control_Mind()
