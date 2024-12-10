#! /usr/bin/env python
# -*- coding: utf-8 -*-
# %matplotlib inline


import datetime
import os
import cv2
import numpy as np
import time
import math

from ultralytics import YOLO, RTDETR

from src._lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur, get_mm_px_from_cm 
from src._lane_detect import get_cm_px_from_mm, get_square_pos, get_road_edge_angle, get_sliding_window_and_cross_result
from src._lane_detect import get_resize_image_4_model, get_cross_pos_by_filter, get_pos_before_xy, Line

#for fun
from src._sing import sing



BOT_FROM_BEV_X = 100  # edit this
BOT_FROM_BEV_Y = 500  # edit this

SPEED_X = 0.25
SPEED_Z = 0.4
TIME_90DEG = 0.53 / SPEED_Z
TIME_SET_STANLEY = 1
RATIO_SPEEDING = 3

# 반경 줄일 거면 값을 높이기: x_speed 감소함
RADIUS_VZ_OVER_VX_CONST = 215  #EDA

TRUE_GREEN_CONF = 100
TRUE_GREEN_DIST_FROM_ROAD = 30  # mm

TRUE_CROSS_CONF = 190


BEV_SHAPE = (300, 200)

# for event:

TIME_MOVE_BACK = 0.04 / SPEED_X

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.6
WAIT_FRAME_4_MODEL = 1 # 0.5 second: in 16fps, will be enough for jetson nano computing smaller yolo


KEY_PREDICT = ("ally", "enem", "ally_tank", "enem_tank")
AREA_NAME = "0ABCDXXXX"*5 # 1번도 A, 10번도 A, 2번도 B, 20번도 B, 를 만드는 괜찮은 방법.


PREFER_ERR_DEG = 5

PREFER_DIST = 380
PREFER_ERR_RATIO = 0.15


def move_robot(pub, vel_x=0, rot_z=0, is_left=True):
    '''
        # move the robot: with x speed and z rotation value
        # 간단함: x만큼 둘다 더하고, z만큼 한쪽 더하고 한쪽 빼줌.

        # 최댓값은 각각 140, 40 rpm으로 되어있음: 대략 30~40cm/s, 6 seconds for rotation
        # 높일 수 있지만, 일단 rotation은 더 과감하게 하기 어렵다.

        문제는, 그냥 느려야 한다는 거.
        미끄러진다. 아주 잘. Stanley가 안 될 정도로.
    '''
    x_max = 140
    z_max = 40

    x_real = x_max * vel_x
    z_real = z_max * rot_z

    if not is_left:
        z_real *= -1

    speed_left = x_real - z_real
    speed_right = x_real + z_real

    speed_max = max(speed_left, speed_right)
    if speed_max > 180:
        speed_left *= 180 / speed_max
        speed_right *= 180 / speed_max

    # print("SPEED: ", speed_left, speed_right)

    speed_left, speed_right = int(speed_left), int(speed_right)

    pub.set_motor_power(pub.MOTOR_LEFT, speed_left)
    pub.set_motor_power(pub.MOTOR_RIGHT, speed_right)


def move_stanley(pub, offset_mm, angle_deg, x_ratio=1):

    kp = 0.03 # 0.05
    ka = 0.11 # 0.10
    k = 2.0
    x = SPEED_X * x_ratio

    z = -(angle_deg * ka - math.atan(kp * offset_mm)) * k * x * x 

    move_robot(pub, x, z)

    return z


def get_vote_count_result(count_map_list):
    """
    getting the most shown value for each key
    return: dict(class(str) -> #(int))
    """

    result = dict()
    for key in KEY_PREDICT:
        list_count = []
        for count_map in count_map_list:
            list_count.append(count_map.get(key, 0))
        print(key, list_count, count_map_list)

        max_n = 1
        max_i = 0
        for i in range(len(list_count)):
            count_n = 0
            for j in range(len(list_count)):
                if list_count[i] == list_count[j]:
                    count_n += 1
            if count_n > max_n:
                max_n, max_i = count_n, i
        result[key] = list_count[max_i]

    return result


def is_concensus(count_map_list):

    for key in KEY_PREDICT:
        value = count_map_list[0].get(key, 0)
        for count_map in count_map_list:
            if value != count_map.get(key, 0):
                return False
    return True


def get_2_point_dist(p1, p2):
    # PYTHAGORAS ROX
    dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist



mtx = np.array([[624.021794, 0, 705.539195],
                [0, 624.719173, 398.307132],
                [0, 0, 1]])
dist = np.array([[-0.318379, 0.108202, -0.000758, 0.000421, -0.016728]])

roi = (0, 0, 1279, 959)

def calibrate(img):
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, mtx) #newcameramtx==mtx
    # crop the image
    x, y, w, h = roi
    dst = dst[y+250:y+h, x:x+w]

    return dst





class Mode:
    '''
        로봇의 Query list 상에 존재할 mode의 기본 구조.
        일단 init을 각각에 맞게 받고,
        해당 mode 차례가 오면 set_frame_and_move를 매 frame마다 받는다.

    '''
    end = False
    pub = None
    log = ""
    running = True
    phase = 0
    capsule = dict()
    capsule["angle_from_road"] = 0
    index = 0
    show_list = []

    def __init__(self, pub):
        self.end = False
        self.pub = pub

    def set_frame_and_move(self, frame, showoff=True):
        pass

    def log_add(self, a, b=""):
        try:
            self.log += f"  | {a:.03f}"
        except:
            self.log += f"  | {a}"
        try:
            self.log += f" {b:.03f}"
        except:
            self.log += f" {b}"

    def log_set(self, id, name):
        self.log = str(id) + " " + str(name)




class StartMode(Mode):
    '''
        시작하자마자 바로 end가 나오기 때문에
        바로 다음으로 넘어가는 mode
    '''

    def __init__(self, pub):
        self.end = True
        self.pub = pub

    def set_frame_and_move(self, frame, showoff=True):

        if showoff:
            pass
        cv2.waitKey(1)
        pass



#End
class EndMode(Mode):

    def __init__(self, pub, model_all, index=0, predict_all=False):
        self.end = False
        self.pub = pub
        self.model = model_all
        self.running = True
        self.predict_all = predict_all

        move_robot(pub)

    def set_frame_and_move(self, frame, showoff=True):

        if self.running and self.predict_all:
            self.pub.log_clear()
            '''
                일단 잠가뒀지만, 만약 쓸 일 생기면 바로 풀어서 넣으면 됨.
            '''


            # ### Prediction works for each Event Zone(A, B, C, D: part no. 10, 20, 30, 40, each.)
            # for index in range(10, 41, 10):
            #     event_image_list = self.capsule.get(f"event_{index}_frame_list", [])


            #     ### Get all result for each Event Zone
            #     result_list = self.model.predict(
            #         event_image_list, show=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD
            #     )


            #     ### Get all count result
            #     count_map_list = []
            #     for k, result in enumerate(result_list):
            #         count_map = dict()
            #         predict_frame = result.plot()

            #         for i in range(len(result.boxes)):
            #             res = result.boxes[i]
            #             class_id = result.names[res.cls[0].item()]

            #             count_map[class_id] = 1 + count_map.get(class_id, 0)

            #         self.log_add("count: ", str(count_map))

            #         cv2.imwrite(os.path.join("predict", f"predict_{index}_final_{k+1}.jpg"), predict_frame)
            #         count_map_list.append(count_map)
            #     count_result = get_vote_count_result(count_map_list=count_map_list)

            #     ### Show result in OLED monitor, add one line (4 times). 
            #     self.pub.log(f"Ally: {count_result[KEY_PREDICT[0]]} / Enem: {count_result[KEY_PREDICT[1]]}")

        if self.running:
            '''
                sing sing sing sing 
            '''
            # sing(self.pub)
            self.running = False


EVE_CONCENSUS_LIMIT = 3


#Eve
class EventMode(Mode):

    def __init__(self, pub, models, shared_list_model_second, index, n_frame=5, wait_sec=1.0, show_log = False, step_for_cam = 0):
        '''
            model: 여기서 바로 사용할 model의 뭐시기를 그대로 가져옴
            shared_list: model_second를 위한 것
            index: 여기서는 아주 중요함! A 구역이면 10, B 구역은 20, 등.
            n_frame: 몇 장 찍어서 최빈값 찾을 것인가: 많으면 좋지만 시간이 오래 걸림
            wait_sec: 맨 마지막에 기다릴 시간, frame 보충에 적당한 시간이 필요.

        '''

        self.end = False
        self.index = index
        self.pub = pub
        self.models = models

        self.shared_list = shared_list_model_second

        self.phase = 0
        self.n_frame = n_frame
        self.n_frame_done = 0

        # WAIT_FRAME_4_MODEL: inference에 걸리는 시간만큼 frame 보내는 역할.
        # 20fps 기준, 대략 10 frame이면 충분할 것!
        # 처음에는 2배 기다림.
        self.wait_frame_4_predict = 8

        self.wait_sec = wait_sec

        self.count_map_list = []

        # enem_tank 있는지 확인하고 위치 기록, 다 나오면 그 중 중앙값 이용.
        self.enem_tank_x_list = []
        self.enem_tank_y_list = []

        # enem_tank 위치 잡고, 각도 잡고, 돌아야 하는 시간과 속도 구함.
        self.rot_time = 0
        self.rot_speed = 0
        self.rot_total_angle = 0

        self.time_start = time.time()
        self.show_log = show_log

        self.step_for_cam = step_for_cam


    def set_frame_and_move(self, frame, showoff=True):
        """
        phase 1: getting prediction for n_frame for every WAIT_FRAME_4_MODEL
        phase 2: /end: vote by each prediction! + get enem_tank position and rotational angle, time start
        phase 3: rotate to the tank position /end: shot! time start
        phase 4: rotate back to the original angle
        phase 5: wait for time ends: filling frames at once

        """
        predict_frame = frame
        self.log_set(self.index, "Eve")
        self.log_set("p", self.phase)

        if self.phase == 0:
            self.time_start = time.time()
            self.phase = 1
            move_robot(self.pub)


        ### phase 1: 일단 뒤로 조금 가볼까? -> 90도 맞추기에도 도움이 될 것으로 보임...      
        if self.phase == 1:

            # self.log_add("mode ", self.n_frame)
            if self.step_for_cam > 0:
                move_robot(self.pub, SPEED_X)
            elif self.step_for_cam < 0:
                move_robot(self.pub, -SPEED_X)

            if (TIME_MOVE_BACK * abs(self.step_for_cam)) < time.time() - self.time_start:
                move_robot(self.pub)
                self.phase = 2
    

        ### phase 2: getting frames and run prediction for each. 
        if self.phase == 2:
            self.log_add("mode ", self.n_frame)

            if self.wait_frame_4_predict > 0:
                move_robot(self.pub)
                if self.wait_frame_4_predict == 3:
                    move_robot(self.pub, 0, 0.2 * ((-1)**self.n_frame))
                self.wait_frame_4_predict -= 1
                return
            self.wait_frame_4_predict = WAIT_FRAME_4_MODEL
            self.n_frame_done += 1

            move_robot(self.pub)
            frame = calibrate(frame)

            # 모델에 맞는 이미지로 변환, 넣기.
            resize_image_4_model = get_resize_image_4_model(frame)

            cv2.imwrite(os.path.join("predict", f"predict_{datetime.datetime.now().strftime('%H%M')}_{self.index}_{self.n_frame_done}.jpg"), resize_image_4_model)
            ### saving frame in capsule for final prediction / End나 Model Second에서 쓰임
            ### 저장할 때부터 이미 이미지 resize 된 걸 저장하기로 함
            self.capsule[f"event_{self.index}_frame_list"] = self.capsule.get(f"event_{self.index}_frame_list", list()) + [resize_image_4_model]


            ### prediction works here
            # 촬영 부저 음
            for n_model, model in enumerate(self.models):
                count_map = dict()
                self.pub.play_buzzer(440)
                result_list = model.predict(resize_image_4_model, show=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=0)
                self.pub.stop_buzzer()
            
            # 결과 풀어서 정리
                for result in result_list:
                    predict_frame = result.plot()

                    for i in range(len(result.boxes)):
                        res = result.boxes[i]
                        cords = res.xyxy[0].tolist()
                        cords = [round(x) for x in cords]
                        class_id = result.names[res.cls[0].item()]

                        count_map[class_id] = 1 + count_map.get(class_id, 0)

                        if class_id == KEY_PREDICT[-1]: # enem_tank
                            # 원래 이미지에서의 위치를 계산하고 받아내서 이용!
                            pos_x, pos_y = get_pos_before_xy(frame, resize_image_4_model, ((cords[0]+cords[2])/2, (cords[1]+cords[3])/2))
                            self.enem_tank_x_list.append(pos_x)
                            self.enem_tank_y_list.append(pos_y)

                # 만든 값 출력
                self.log_add("count: ", str(count_map))
                cv2.imwrite(os.path.join("predict", f"predict_{datetime.datetime.now().strftime('%H%M')}_{self.index}_{self.n_frame_done}_model{n_model}.jpg"), predict_frame)
                self.count_map_list.append(count_map)


            # 만약 3장 이상에서 합의 봤으면 그냥 넘어가기.
            if self.n_frame_done == EVE_CONCENSUS_LIMIT and is_concensus(self.count_map_list):
                self.phase = 3


            # 프레임 다 썼다면? 다음으로 넘어가기.
            if self.n_frame <= self.n_frame_done:
                self.phase = 3


        ### phase 3: Get consensus of count result! 
        elif self.phase == 3:
            self.phase = 4
            self.time_start = time.time()

            count_result_map = get_vote_count_result(self.count_map_list)

            if count_result_map[KEY_PREDICT[-1]] > 0: # enem_tank
                # n_xy = len(self.enem_tank_x_list)
                # enem_tank_xy = (sorted(self.enem_tank_x_list)[int((n_xy - 0.5) / 2)], sorted(self.enem_tank_y_list)[int((n_xy - 0.5) / 2)])

                self.pub.play_buzzer(880)
                self.pub.fire_cannon()
                time.sleep(0.05)
                self.pub.fire_cannon()
                self.pub.stop_buzzer()

            # if enem tank is not found here: skip adjusting / no cannon firing
            # else:
            self.phase = 4

            self.log_add("prdctn_rslt: ", str(count_result_map))

            # 만약 second model이 아니라 여기서 결과를 내야 한다면, 이렇게 진행.
            if self.show_log:
                self.pub.log(f" {AREA_NAME[self.index]}: Ally {count_result_map[KEY_PREDICT[0]]} / Enem {count_result_map[KEY_PREDICT[1]]}")

            # model_second를 위해서 자료 제공
            self.shared_list[int(self.index/10)] = (self.capsule[f"event_{self.index}_frame_list"], self.count_map_list)
            
            self.time_start = time.time()

        elif self.phase == 4:
            if self.step_for_cam > 0:
                move_robot(self.pub, -SPEED_X)
            elif self.step_for_cam < 0:
                move_robot(self.pub, SPEED_X)

            if (TIME_MOVE_BACK * abs(self.step_for_cam)) < time.time() - self.time_start:
                move_robot(self.pub)
                self.phase = 5



        # ### phase 3: For enem_tank: rotate the exact angle one time and fire cannon
        # elif self.phase == 3:
        #     '''
        #         만약 매번 adjust해야 한다면, 여기를 수정하면 되는 일.
        #         그러나 그 가능성이 매우 낮다고 가정하여, 여기서는 수행하지 않음.
        #     '''
        #     if self.rot_time > time.time() - self.time_start:
        #         self.log_add("speed: ", self.rot_speed)
        #         self.log_add("rotating: ", self.wait_sec)
        #         self.log_add("until: ", time.time() - self.time_start)
        #         move_robot(self.pub, 0, self.rot_speed)
        #     else:
        #         self.phase = 4
        #         self.time_start = time.time()
        #         move_robot(self.pub)
        #         self.pub.play_buzzer(880)
        #         self.pub.fire_cannon()
        #         self.pub.stop_buzzer()
        #         # BANG!!!


        # ### phase 4: Rotate back the exact same angle (preparing next move)
        # elif self.phase == 4:
        #     if self.rot_time > time.time() - self.time_start:
        #         self.log_add("speed: ", -self.rot_speed)
        #         self.log_add("rotating_back: ", self.wait_sec)
        #         self.log_add("until: ", time.time() - self.time_start)
        #         move_robot(self.pub, 0, -self.rot_speed)
        #     else:
        #         self.phase = 5
        #         self.time_start = time.time()
        #         move_robot(self.pub)


        ### phase 5: Waiting a little bit: to fill the frames again...
        elif time.time() - self.time_start < self.wait_sec:
            self.log_add("hold: ", self.wait_sec)
            self.log_add("\'til: ", time.time() - self.time_start)
        else:
            self.end = True

        if showoff:
            self.show_list = [predict_frame]



class _SheepMode(Mode):

    def __init__(self, pub, index=0, sleep_sec = 1.0):
        '''
            pub = tiki
            index = 번호, 로그에 남기기 위함
            from_it: 시작 첫 10프레임에 보이는 green을 무시할 것인지의 여부
            left_offset: 길보다 왼쪽으로 몇 mm 틀어서 갈 것인가
            speed_weight: 기본 stanley에서 속도를 조금 더 빠르게 할 수 있음: max 1.5, 그 이상은 의미 X
        '''
        self.end = False
        self.pub = pub

        self.phase = 1

        self.index = index

        self.sleep_sec = sleep_sec
        self.time_start = time.time()


    def set_frame_and_move(self, frame, showoff=True):
        """
            phase 1: stanley until find the green point on road
            phase 2: adjusting distance from green: slowing down / backing
        """


        self.log_set(self.index, "Shp")
        bev = get_bev(frame)
        self.log_add("phase ", self.phase)

        # slidingwindow
        road_bev = get_road(bev)



        if self.phase == 1:
            self.time_start = time.time()
            self.phase = 2
            move_robot(self.pub)


        elif self.phase == 2:
            self.log_add("sleep ", time.time() - self.time_start)
            self.log_add("until ", self.sleep_sec)
            if time.time() - self.time_start < self.sleep_sec:
                pass
            else:
                self.end = True



        # showoff now
        if showoff:
            self.show_list = [frame, bev, road_bev]

S2X_LIMIT_Z = 0.7


#S2G
class Stanley2GreenMode(Mode):

    def __init__(self, pub, index=0, from_it=False, left_offset=0, speed_weight=1.0, prefer_dist = PREFER_DIST, speeding_time=0.0):
        '''
            pub = tiki
            index = 번호, 로그에 남기기 위함
            from_it: 시작 첫 10프레임에 보이는 green을 무시할 것인지의 여부
            left_offset: 길보다 왼쪽으로 몇 mm 틀어서 갈 것인가
            speed_weight: 기본 stanley에서 속도를 조금 더 빠르게 할 수 있음: max 1.5, 그 이상은 의미 X
        '''
        self.end = False
        self.pub = pub

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.green_encounter = -2
        self.left_offset = left_offset

        self.phase = 1
        if speeding_time > 0.1:
            self.phase = -2
        self.speeding_time = speeding_time
        self.time_start = time.time()

        self.index = index

        self.speed_weight = speed_weight

        self.frame_without_line = 5
        self.prefer_dist = prefer_dist

        self.frame_from_start_sensing = 0
        if from_it:
            # 20프레임동안은 무시
            self.frame_from_start_sensing = -20

    def set_frame_and_move(self, frame, showoff=True):
        """
            phase -2 : setting time
            phase -1 : normal stanley
            phase  0 : speeding!!
            phase 1: stanley until find the green point on road
            phase 2: adjusting distance from green: slowing down / backing
        """
        self.frame_from_start_sensing += 1

        self.log_set(self.index, "S2G")
        bev = get_bev(frame)
        self.log_add("phase ", self.phase)

        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

        
        # x, y list = sliding window midpoints -> can be used to find the exact line
        if len(x_list) > 2:
            self.init_pos_for_sliding_windows = x_list[1]
            self.line_road = Line(x_list, y_list)
            self.frame_without_line = 0
            # if no line was found next time, the Line will be used as a real one for 5 frames
        elif self.frame_without_line < 5:
            self.frame_without_line += 1
            self.init_pos_for_sliding_windows = -1
        else:
            self.line_road = None


        # 시작 시 길 없거나 / 길 잃어버리면 여기서 다시 시작: 조금 뒤로 가면, 길은 보이게 되어있다.
        if self.line_road == None:
            # Must find the line here, First!
            self.log_add("no line here... backing")
            self.show_list = [frame, bev, road_bev, road_sw_bev]
            self.frame_from_start_sensing -= 2
            move_robot(self.pub, -SPEED_X)
            return


        # for green event!
        green_bev = get_green(bev)
        green_bev_cm = get_cm_px_from_mm(green_bev)
        green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 5)
        green_pos = [pos * 10 for pos in green_pos_cm]

        # for stanley!
        offset_mm = self.line_road.get_offset(BOT_FROM_BEV_X + self.left_offset, BOT_FROM_BEV_Y)
        angle_deg = self.line_road.get_angle()
        self.log_add("offset", offset_mm)
        self.log_add("angle", angle_deg)


        if self.phase == -2:
            self.time_start = time.time()
            self.phase = -1
        
        if self.phase == -1:
            self.log_add("Speeding ready")
            if time.time() - self.time_start < TIME_SET_STANLEY:
                z = move_stanley(self.pub, offset_mm, angle_deg, x_ratio=self.speed_weight)
                self.log_add("z speed ", z)
            else:
                self.time_start = time.time()
                self.phase = 0

        if self.phase == 0:
            self.pub.play_buzzer(660)
            if time.time() - self.time_start < self.speeding_time:
                self.log_add("Speeding!!!")
                z = move_stanley(self.pub, offset_mm, angle_deg, x_ratio=RATIO_SPEEDING)
                if abs(z) > S2X_LIMIT_Z:
                    self.time_start = time.time()
                    self.phase = 1
                self.log_add("z speed ", z)
            else:
                self.time_start = time.time()
                self.phase = 1



        if self.phase == 1:
            self.pub.stop_buzzer()
            
            # do stanley
            z = move_stanley(self.pub, offset_mm, angle_deg, x_ratio=self.speed_weight)


            self.log_add("z speed ", z)

            if green_max > TRUE_GREEN_CONF and self.line_road.get_distance(green_pos[1], green_pos[0]) < TRUE_GREEN_DIST_FROM_ROAD:
                self.log_add("true green?", green_max)
                self.log_add("true green at", green_pos)
                self.log_add("true green from line", self.line_road.get_distance(green_pos[1], green_pos[0]))
                if self.frame_from_start_sensing > 0:
                    self.green_encounter += 1
            else:
                self.green_encounter = max(0, self.green_encounter-1)

            if self.green_encounter >= 3:
                self.phase = 2
                # move_robot(self.pub)


        ### 녹색 보고 phase 2: 속도 자체를 녹색 표식과의 위치 관계에 따라 바꿈, PREFER_ERR_RATIO 안으로 들어오면 끝. 
        elif self.phase == 2:
            self.capsule["dist_from_cross"] = 0
            self.capsule["angle_from_road"] = angle_deg
            # 만약 Green이 사라진다면? 일단 거기서 끝내기. 그럴 일은 없겠지만, 거기서 멈추기로 함.
            if green_max < TRUE_GREEN_CONF:
                self.end = True
                self.log_add("Green is Gone! ", green_max)
                return

            # 가까우면 음수, 멀면 양수, -0.5 ~ 0.25까지 나올 수 있음: BEV 따라서..
            dist_ratio = (get_2_point_dist((green_pos[1], green_pos[0]), (BOT_FROM_BEV_X, BOT_FROM_BEV_Y)) / self.prefer_dist) - 1
            self.log_add("Dist ratio ", dist_ratio)

            if abs(dist_ratio) > PREFER_ERR_RATIO:
                z = move_stanley(self.pub, offset_mm, angle_deg, dist_ratio*2)  # slow down a lot
            else:
                z = move_robot(self.pub)  # stop
                self.end = True


        # showoff now
        if showoff:
            cv2.line(road_sw_bev,
                     (int(self.line_road.calc(0)), 0),(int(self.line_road.calc(np.shape(road_sw_bev)[0])),np.shape(road_sw_bev)[0],),
                     (0, 0, 255),5)
            self.show_list = [frame, bev, road_bev, road_sw_bev, get_mm_px_from_cm(green_bev_cm), get_mm_px_from_cm(green_blur_bev),]




#S2C
class Stanley2CrossMode(Mode):

    def __init__(self, pub, index=0, left_way=True, right_way=True, from_it=False, left_offset=0, use_green=False, speed_weight=1.0, speeding_time=0.0, prefer_dist = 220):
        '''
            pub = tiki
            index = 번호, 로그에 남기기 위함
            left / right_way = cross에 각 방향의 길이 있는가? 십자는 둘 다 true / 둘 다 false일 수 없음.
            from_it: 시작 첫 10프레임에 보이는 green/cross을 무시할 것인지의 여부
            left_offset: 길보다 왼쪽으로 몇 mm 틀어서 갈 것인가
            speed_weight: 기본 stanley에서 속도를 조금 더 빠르게 할 수 있음: max 1.5, 그 이상은 의미 X
        '''
        self.end = False
        self.pub = pub

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.green_encounter = -2
        self.cross_encounter = -2
        self.left_way = left_way
        self.right_way = right_way
        self.left_offset = left_offset
        self.prefer_dist = prefer_dist

        self.phase = 1        
        if speeding_time > 0.1:
            self.phase = -2
        self.speeding_time = speeding_time
        self.time_start = time.time()

        self.index = index

        self.capsule = dict()
        self.use_green = use_green

        self.speed_weight = speed_weight
        self.frame_without_line = 5

        self.frame_from_start_sensing = 0
        if from_it:
            self.frame_from_start_sensing = -20


    def set_frame_and_move(self, frame, showoff=True):
        """
            phase -2 : setting time
            phase -1 : normal stanley
            phase  0 : speeding!!

            Phase는 녹색을 쓸 때만: S2G와 동일 / 그 외에는 그냥, 진행시킴.
        """
        
        """
            phase -2 : setting time
            phase -1 : normal stanley
            phase  0 : speeding!!
            phase 1: stanley until find the green point on road
            phase 2: adjusting distance from green: slowing down / backing
        """
        self.frame_from_start_sensing += 1

        self.log_set(self.index, "S2G")
        bev = get_bev(frame)
        self.log_add("phase ", self.phase)

        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

        
        # x, y list = sliding window midpoints -> can be used to find the exact line
        if len(x_list) > 2:
            self.init_pos_for_sliding_windows = x_list[1]
            self.line_road = Line(x_list, y_list)
            self.frame_without_line = 0
            # if no line was found next time, the Line will be used as a real one for 5 frames
        elif self.frame_without_line < 5:
            self.frame_without_line += 1
            self.init_pos_for_sliding_windows = -1
        else:
            self.line_road = None


        # 시작 시 길 없거나 / 길 잃어버리면 여기서 다시 시작: 조금 뒤로 가면, 길은 보이게 되어있다.
        if self.line_road == None:
            # Must find the line here, First!
            self.log_add("no line here... backing")
            self.show_list = [frame, bev, road_bev, road_sw_bev]
            self.frame_from_start_sensing -= 2
            move_robot(self.pub, -SPEED_X)
            return


        # for cross event!
        road_bev_cm = get_cm_px_from_mm(road_bev)
        road_blur_bev, cross_pos_cm, cross_max = get_cross_pos_by_filter(road_bev_cm, 5, left_way=self.left_way, right_way=self.right_way)
        cross_pos = [pos * 10 for pos in cross_pos_cm]

        # for stanley!
        offset_mm = self.line_road.get_offset(BOT_FROM_BEV_X + self.left_offset, BOT_FROM_BEV_Y)
        angle_deg = self.line_road.get_angle()
        self.log_add("offset", offset_mm)
        self.log_add("angle", angle_deg)


        if self.phase == -2:
            self.time_start = time.time()
            self.phase = -1
        
        if self.phase == -1:
            self.log_add("Speeding ready")
            if time.time() - self.time_start < TIME_SET_STANLEY:
                z = move_stanley(self.pub, offset_mm, angle_deg, x_ratio=self.speed_weight)
                self.log_add("z speed ", z)
            else:
                self.time_start = time.time()
                self.phase = 0

        if self.phase == 0:
            self.pub.play_buzzer(660)
            if time.time() - self.time_start < self.speeding_time:
                self.log_add("Speeding!!!")
                z = move_stanley(self.pub, offset_mm, angle_deg, x_ratio=RATIO_SPEEDING)
                if abs(z) > S2X_LIMIT_Z:
                    self.time_start = time.time()
                    self.phase = 1
                self.log_add("z speed ", z)
            else:
                self.time_start = time.time()
                self.phase = 1



        if self.phase == 1:
            self.pub.stop_buzzer()
            
            # do stanley
            z = move_stanley(self.pub, offset_mm, angle_deg, x_ratio=self.speed_weight)


            self.log_add("z speed ", z)

            if cross_max > TRUE_CROSS_CONF and self.line_road.get_distance(cross_pos[1], cross_pos[0]) < TRUE_GREEN_DIST_FROM_ROAD:
                self.log_add("true cross?", cross_max)
                self.log_add("true cross at", cross_pos)
                self.log_add("true cross from line", self.line_road.get_distance(cross_pos[1], cross_pos[0]))
                if self.frame_from_start_sensing > 0:
                    self.green_encounter += 1
            else:
                self.green_encounter = max(0, self.green_encounter-1)

            if self.green_encounter >= 3:
                self.phase = 2
                # move_robot(self.pub)


        ### 녹색 보고 phase 2: 속도 자체를 녹색 표식과의 위치 관계에 따라 바꿈, PREFER_ERR_RATIO 안으로 들어오면 끝. 
        elif self.phase == 2:
            self.capsule["dist_from_cross"] = get_2_point_dist((cross_pos[1], cross_pos[0]), (BOT_FROM_BEV_X, BOT_FROM_BEV_Y))
            self.capsule["angle_from_road"] = angle_deg
            # 만약 Green이 사라진다면? 일단 거기서 끝내기. 그럴 일은 없겠지만, 거기서 멈추기로 함.
            if cross_max < TRUE_CROSS_CONF:
                self.end = True            
                self.capsule["dist_from_cross"] = BOT_FROM_BEV_Y - BEV_SHAPE[0]
                self.log_add("Cross is Gone! ", cross_max)
                return

            # 가까우면 음수, 멀면 양수, -0.5 ~ 0.25까지 나올 수 있음: BEV 따라서..
            dist_ratio = (get_2_point_dist((cross_pos[1], cross_pos[0]), (BOT_FROM_BEV_X, BOT_FROM_BEV_Y)) / self.prefer_dist) - 1
            self.log_add("Dist ratio ", dist_ratio)

            if abs(dist_ratio) > PREFER_ERR_RATIO:
                z = move_stanley(self.pub, offset_mm, angle_deg, dist_ratio*2)  # slow down a lot
            else:
                z = move_robot(self.pub)  # stop
                self.end = True


        # showoff now
        if showoff:
            cv2.line(road_sw_bev,
                     (int(self.line_road.calc(0)), 0),(int(self.line_road.calc(np.shape(road_sw_bev)[0])),np.shape(road_sw_bev)[0],),
                     (0, 0, 255),5)
            self.show_list = [frame, bev, road_bev, road_sw_bev, get_mm_px_from_cm(road_bev_cm), get_mm_px_from_cm(road_blur_bev),]




class Turn2RoadMode(Mode):

    def __init__(self, pub, index=0, is_left=True, min_turn_sec=TIME_90DEG*0.9, is_curve=False, z_ratio = 1):
        """
            is_left: 왼쪽으로 돌 때 true / 오른쪽으로 돌 거면 false
            min_turn_sec: 길을 무시하고 돌아갈 시간, 다른 오브젝트나 기물이 길처럼 보일 수 있음: 예상 시간의 80% 정도로 잡기.
            is_curve: True면 반경 받아와서 그걸로 직선속도 구한 뒤 회전 / 아니면 제자리 회전.
        """
        self.end = False
        self.pub = pub

        self.is_left = is_left
        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.min_turn_sec = min_turn_sec/z_ratio
        self.is_curve = is_curve

        self.road_encounter = 0

        self.road_angle = -1000
        self.dist_from_cross = -1

        self.phase = 0
        self.time_since_phase = 0
        self.est_time = 0

        self.rot_z = SPEED_Z * z_ratio
        self.speed_x = 0

        self.index = index

        self.capsule = {"dist_from_cross": BOT_FROM_BEV_Y - BEV_SHAPE[0]}

    def set_frame_and_move(self, frame, showoff=True):
        """
        phase 0: / time starting + getting angle and est_time, SPEED_X, rot_z
        phase 1: rotating at least min_turn_sec
        phase 2: rotate until you see the line
        """

        self.log_set(self.index, "T2R")
        self.log_add("p", self.phase)

        bev = get_bev(frame)
        road_bev = get_road(bev, with_green=False)
        road_blur_bev = get_rect_blur(road_bev, 5)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

        # 시작, 커브일 경우 반경과 속도를 구해야 함!
        if self.phase == 0 and self.is_curve:
            dist_from_cross = self.capsule["dist_from_cross"]

            road_edge_bev, angle = get_road_edge_angle(road_bev, self.is_left)
            self.road_angle = angle
            if abs(self.road_angle) > 20:
                self.road_angle = 0
            if self.is_left:
                self.road_angle = -self.road_angle

            # cross까지의 거리, 길에 대해 기울어진 각도를 통해 최적의 코너 반경을 구함: 공식은 이상하지 않을걸요?
            radius = dist_from_cross / (1 + math.sin(self.road_angle * math.pi / 180))

            # 반경 * 회전각속도 / 직선속도는 항상 일정한 상수가 됨, 회전각속도 고정하고 직선속도 변화 (반경 300에서, 예상 속도 1.3)
            self.speed_x = radius * self.rot_z / RADIUS_VZ_OVER_VX_CONST
            self.log_add("radius", radius)
            self.log_add("rot_z", self.rot_z)
            self.log_add("SPEED_X", self.speed_x)


        # starting: 0을 1로, 시간 측정 시작
        if self.phase == 0:
            self.phase = 1
            self.time_since_phase = time.time()


        # Phase 1. turning at least certain amount: to ignore post-road
        if self.phase == 1:

            # 아예 처음에 엄청 빨리 돌아서, 시간을 좀 절약하자.
            # z_slowing_down = 1*(0.6-(time.time()-self.time_since_phase))

            # if self.rot_z < z_slowing_down:
            #     self.pub.play_buzzer(660)
            # else:
            #     self.pub.stop_buzzer()


            move_robot(self.pub, self.speed_x, max(self.rot_z, self.rot_z), self.is_left)

            if time.time() - self.time_since_phase > self.min_turn_sec:
                self.phase = 2
                self.time_since_phase = time.time()
                # move_robot(self.pub)

        # Phase 2. turning while the line is shown: to estimate time to be exact 90 degrees
        if self.phase == 2:
            self.log_add("no_line")

            if self.road_encounter <= 0:
                move_robot(self.pub, self.speed_x, self.rot_z, self.is_left)

            if len(x_list) > 2:
                self.init_pos_for_sliding_windows = x_list[1]
                line_road = Line(x_list, y_list)
                self.line_road = line_road

                self.log_add("line_on_angle", line_road.get_angle())

                cv2.line(road_sw_bev,
                         (int(self.line_road.calc(0)), 0),  (int(self.line_road.calc(np.shape(road_sw_bev)[0])),np.shape(road_sw_bev)[0]),
                         (0, 0, 255), 5)

                if len(x_list) > 4 or abs(line_road.get_angle()) < 10:
                    self.road_encounter += 1
                    move_robot(self.pub, 0, -self.rot_z, self.is_left)
                # needs 2 time for road_encounter
                if self.road_encounter >= 4:
                    move_robot(self.pub)
                    self.end = True
            else:
                self.road_encounter = max(0, self.road_encounter-1)

        if showoff:
            self.show_list = [frame, bev, road_bev, road_sw_bev]


class Turn2VoidMode(Mode):

    def __init__(self, pub, index=0, is_left=True, other_turn_sec=0):
        """
            기본 방식: 지정된 시간 * 돌아야 하는 각도(현재 각도에서 계산)만큼 돌기.
            is_left: 왼쪽으로 돌 것인지 확인
            other_turn_sec: 이전에 쓰던 방식, 지금은 쓰지 않기로.

            이전 방식: 한 번 돌면서, 길이 보이는 동안 시간에 따른 길의 각도를 측정함
            추세선을 그리고 외삽하여, 예상 시간을 구하는 방식
            문제점: 각도 측정에 오차가 잘 생기는 편... 연산량 줄인 HoughLinesP의 오차가 추세선 상에 매번 누적됨
        """

        self.end = False
        self.pub = pub

        self.is_left = is_left
        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.other_turn_sec = other_turn_sec

        self.phase = 0
        self.time_since_phase = 0
        self.angle_at_start = -90
        self.angle_list = []
        self.time_list = []
        self.est_time = 0
        self.est_time_regression = 0
        self.est_time_angle_calc = 0

        self.waiting_for_next_frame = 2

        self.index = index

    def set_frame_and_move(self, frame, showoff=True):
        """
        phase 0: time starting
        phase 1: rotating to other side a little bit / end: get angle and est_time
        phase 2: getting all angle data while rotating : Now ignored
        phase 3: waiting / end: stop
        """
        self.log_set(self.index, "T2V")
        self.log_add("p", self.phase)


        # road edge angle detection
        bev = get_bev(frame)
        road_bev = get_road(bev)
        road_edge_bev, angle = get_road_edge_angle(road_bev, self.is_left)


        # phase 0: time starting
        if self.phase == 0:
            self.phase = 1
            self.time_since_phase = time.time()


        # phase 1: rotating to other side a little bit: to get max data and right angle.
        if self.phase == 1:
            # move_robot(self.pub, 0, -SPEED_Z, self.is_left)

            # 지금은 other turn sec = 0이라, 아래 코드가 바로 작동된다. 하지만 가서는 쓰게 될수도...
            if time.time() - self.time_since_phase > self.other_turn_sec:
                self.phase = 2
                self.time_since_phase = time.time()
                move_robot(self.pub)

                angle_at_start = angle

                if self.is_left:
                    self.est_time_angle_calc = (TIME_90DEG / 90) * (90 - angle_at_start)
                else:
                    self.est_time_angle_calc = (TIME_90DEG / 90) * (90 + angle_at_start)
                self.est_time = self.est_time_angle_calc
                if abs(angle) > 10:
                    self.est_time = TIME_90DEG

                self.log_add("angle", angle)
                self.log_add("time", self.est_time)

        # turning while the line is shown: to estimate time to be exact 90 degrees
        if self.phase == 2:
            self.time_since_phase = time.time()
            move_robot(self.pub, 0, SPEED_Z, self.is_left)

            # to skip the image using phase...
            self.phase = 3

            """
            if abs(angle) < 45:
                self.waiting_for_next_frame = 2
                self.time_list.append(time.time() - self.time_since_phase)
                self.angle_list.append(angle)
                
                # cv2.imwrite(str(self.index) + "_frame_edge_" + str(len(self.time_list)) + ".jpg", road_edge_bev)
                # cv2.imwrite(str(self.index) + "_frame_original_" + str(len(self.time_list)) + ".jpg", road_bev)
            elif self.waiting_for_next_frame > 0:
                self.waiting_for_next_frame -= 1
            else:
                self.phase = 3
                self.log += str(self.angle_list) + str(self.time_list)
                if len(self.time_list) > 6:
                    len_ignore = int(len(self.time_list)/2.5)
                    self.time_list = self.time_list[len_ignore:-2]
                    self.angle_list = self.angle_list[len_ignore:-2]
                print(self.time_list, self.angle_list)

                if len(self.time_list) < 2:
                    return
                calc = np.poly1d(np.polyfit(self.angle_list, self.time_list, deg=1))
                if self.is_left:
                    self.est_time_regression = calc(90)
                else:  
                    self.est_time_regression = calc(-90)
                
                if self.est_time - 0.1 < self.est_time_regression < self.est_time + 0.1 :
                    self.est_time = self.est_time_regression
            """

        # Phase 3: 그냥 그 시간만큼 turn하면 됨
        if self.phase == 3:

            if time.time() < self.time_since_phase + self.est_time:
                move_robot(self.pub, 0, SPEED_Z, self.is_left)
            else:
                move_robot(self.pub)
                self.end = True

        if showoff:
            self.show_list = [frame, bev, road_bev, road_edge_bev]
