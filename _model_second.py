# -*- coding: utf-8 -*-


import cv2
import numpy as np
import time


from ultralytics import YOLO, RTDETR

from fake_tiki import TikiMini

from _mode import get_vote_count_result, IOU_THRESHOLD, CONF_THRESHOLD


def run_model_second(tiki: TikiMini, model_address, shared_list):

    pos = 0

    model = YOLO(model_address)
    # model = RTDETR(model_address)
    # model.to('cuda')

    while pos < 5:

        while shared_list[pos] is None:
            time.sleep(1)
        
        image_list = shared_list[pos]

        result_list = model.predict(image_list, show=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        
        count_map_list = []
        for k, result in enumerate(result_list):
            count_map = dict()
            predict_frame = result.plot()

            for i in range(len(result.boxes)):
                res = result.boxes[i]
                class_id = result.names[res.cls[0].item()]

                count_map[class_id] = 1 + count_map.get(class_id, 0)

            cv2.imwrite(f"predict_{pos}_troops_{k*10+10}.jpg", predict_frame)
            count_map_list.append(count_map)
        count_result = get_vote_count_result(count_map_list=count_map_list)

        print(count_result)
        tiki.log(str(count_result))
        shared_list[pos] = count_result
        pos += 1






