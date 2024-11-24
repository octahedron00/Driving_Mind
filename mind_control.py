# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import time
import math
import datetime

import keyboard

from fake_tiki import TikiMini

from _mode import move_robot
from mind_bot import showing_off, VID_CONNECT_CMD, FRAME_IGNORE_LEVEL


IS_LOG_VID = True


class Control_Mind:

    def __init__(self, show_function=showing_off, keyboard_listener=keyboard):

        now = datetime.datetime.now().strftime("%H%M")
        if IS_LOG_VID:
            self.logwriter = cv2.VideoWriter(
                f"vlog_control_{now}.avi",
                cv2.VideoWriter_fourcc(*"MP4V"),
                10.0,
                (640, 480),
            )

        self.pub = TikiMini()
        self.pub.set_motor_mode(self.pub.MOTOR_MODE_PID)

        self.count_frame = 1

        self.image_name = f"image_{now}"
        self.image_count = 1

        self.show_function = show_function
        self.keyboard_listener = keyboard_listener

        cap = cv2.VideoCapture(VID_CONNECT_CMD)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.count_frame += 1
            if self.count_frame % FRAME_IGNORE_LEVEL == 0:
                self.action(frame)

    def action(self, frame):
        pub = self.pub

        self.show_function([frame])

        if IS_LOG_VID:
            self.logwriter.write(frame)

        cv2.waitKey(1)

        if self.keyboard_listener.is_pressed("w"):
            move_robot(pub, 0.2, 0)
        elif self.keyboard_listener.is_pressed("s"):
            move_robot(pub, -0.2, 0)
        elif self.keyboard_listener.is_pressed("a"):
            move_robot(pub, 0, 0.5)
        elif keyboard.is_pressed("d"):
            move_robot(pub, 0, -0.5)
        elif self.keyboard_listener.is_pressed("c"):
            cv2.imwrite(
                os.path.join("img", f"{self.image_name}_{self.image_count:04d}.jpg"),
                frame,
            )
            self.image_count += 1
        elif self.keyboard_listener.is_pressed("n"):
            self.image_name = input("new dataset name:")
            if len(self.image_name) < 1:
                now = datetime.datetime.now().strftime("%H%M")
                self.image_name = f"image_{now}"

            self.image_count = 1
        else:
            move_robot(pub)


if __name__ == "__main__":

    Control_Mind()
