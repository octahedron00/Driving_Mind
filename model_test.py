# -*- coding: utf-8 -*-


import numpy as np
import time

from ultralytics import YOLO, RTDETR




# 마음껏 수정하며 진행할 것.

DO_SECOND = False


FILE_1 = "rtdetr-l.pt"
FILE_1_TRT = "rtdetr-l.onnx"
FILE_2 = "yolo11l.pt"
FILE_2_TRT = "yolo11l.engine"

IMG_SIZE = (640, 640, 3)
IMG_TEST = np.zeros(IMG_SIZE)

print("All import finished")


start_time = time.time()
model_1 = RTDETR(FILE_1)
model_1.to('cpu')
# model_1.to('cuda')
print(f"Model 1 Starting: {time.time()-start_time:.03f} seconds")

# model_1.export(format="onnx")

# model_1 = RTDETR(FILE_1_TRT)

if DO_SECOND:
    start_time = time.time()
    model_2 = YOLO(FILE_2)
    # model_2.to('cuda')
    print(f"Model 2 Starting: {time.time()-start_time:.03f} seconds")



start_time = time.time()
model_1.predict(IMG_TEST)
print(f"Model 1 First inference: {time.time()-start_time:.03f} seconds")

if DO_SECOND:
    start_time = time.time()
    model_2.predict(IMG_TEST)
    print(f"Model 2 First inference: {time.time()-start_time:.03f} seconds")



for i in range(5):

    start_time = time.time()
    model_1.predict(IMG_TEST)
    print(f"Model 1 RealTime inference: {time.time()-start_time:.03f} seconds")

    if DO_SECOND:
        start_time = time.time()
        model_2.predict(IMG_TEST)
        print(f"Model 2 Second inference: {time.time()-start_time:.03f} seconds")
    print()
