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
