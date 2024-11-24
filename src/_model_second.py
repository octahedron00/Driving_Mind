# -*- coding: utf-8 -*-


import cv2
import numpy as np
import time

import os

from ultralytics import YOLO, RTDETR

from src.fake_tiki import TikiMini

from src._mode import get_vote_count_result, IOU_THRESHOLD, CONF_THRESHOLD, KEY_PREDICT


AREA_NAME = "0ABCD"


def run_model_second(tiki: TikiMini, model_address, shared_list):
    """
        Model Second: 2번째 thread를 이용하는, multithreading:
        결과가 바로 나오지 않지만, 끝날 때에는 반드시 나온다!
        주행과 동시에 연산하기, 커다란 모델도 걱정 없이 사용 가능.

        그나마 걱정은 상태가 안 좋을까봐... 멀티쓰레딩은 조금 무섭다

        Bot_Mind init에서 Process로 만들어냄, manager: shared_list로 다음 데이터 받기.
        다음 데이터 오기 전까지 기다리다가, 데이터 오면 연산 시작하는 방식으로 진행.

        log: 바로바로 로봇에 출력하도록. log는 얘만 쓸 수 있게 하면 된다!
    """


    pos = 0

    model = YOLO(model_address)
    # model = RTDETR(model_address)
    # model.to('cuda')

    while pos < 5:

        # 다음 데이터 들어올 때까지 기다리기.
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

            cv2.imwrite(os.path.join("predict", f"predict_{pos*10}_troops_{k}.jpg"), predict_frame)
            count_map_list.append(count_map)
        count_result = get_vote_count_result(count_map_list=count_map_list)

        print(count_result)
        if pos > 0:
            tiki.log(f" {AREA_NAME[pos]} AREA: Ally {count_result[KEY_PREDICT[0]]} / Enem {count_result[KEY_PREDICT[1]]}")
        shared_list[pos] = count_result
        pos += 1






