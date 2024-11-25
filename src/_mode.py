#! /usr/bin/env python
# -*- coding: utf-8 -*-
# %matplotlib inline


import os
import cv2
import numpy as np
import time
import math

from ultralytics import YOLO, RTDETR

from src._lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur, get_mm_px_from_cm 
from src._lane_detect import get_cm_px_from_mm, get_square_pos, get_road_edge_angle, get_sliding_window_and_cross_result
from src._lane_detect import get_resize_image_4_model, get_pos_before_xy, Line

BOT_FROM_BEV_X = 100  # edit this
BOT_FROM_BEV_Y = 500  # edit this

SPEED_X = 1.0
SPEED_Z = 1.0
TIME_90DEG = 1.5 / SPEED_Z

RADIUS_VZ_OVER_VX_CONST = 240  # edit this

TRUE_GREEN_CONF = 100
TRUE_GREEN_DIST_FROM_ROAD = 30  # mm


BEV_SHAPE = (300, 200)

# for event:
CONF_THRESHOLD = 0.6
IOU_THRESHOLD = 0.6
WAIT_FRAME_4_MODEL = 10 # 0.5 second: will be enough for jetson nano computing smaller yolo
WAIT_FRAME_4_MODEL = 0

KEY_PREDICT = ("ally", "enem", "ally_tank", "enem_tank")

PREFER_ERR_DEG = 5

PREFER_DIST = 400
PREFER_ERR_RATIO = 0.1


def move_robot(pub, vel_x=0, rot_z=0, is_left=True):

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

    speed_left, speed_right = int(speed_left), int(speed_right)

    pub.set_motor_power(pub.MOTOR_LEFT, speed_left)
    pub.set_motor_power(pub.MOTOR_RIGHT, speed_right)


def move_stanley(pub, offset_mm, angle_deg, x_ratio=1):

    kp = 0.035
    ka = 0.10
    k = 0.5
    x = SPEED_X * x_ratio

    z = -(angle_deg * ka - math.atan(kp * offset_mm)) * x * k

    move_robot(pub, x, z)

    return z


def get_vote_count_result(count_map_list):
    """
    getting the most shown value for each key
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


def get_2_point_dist(p1, p2):

    dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist


class Mode:

    end = False
    pub = None
    log = ""
    running = True
    phase = 0
    capsule = dict()
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

    def __init__(self, pub, model_all, index=0, predict_all=True):
        self.end = False
        self.pub = pub
        self.model = model_all
        self.running = True
        self.predict_all = predict_all

        move_robot(pub)

    def set_frame_and_move(self, frame, showoff=True):

        if self.running and self.predict_all:
            self.pub.log_clear()

            ### Prediction works for each Event Zone(A, B, C, D: part no. 10, 20, 30, 40, each.)
            for index in range(10, 41, 10):
                event_image_list = self.capsule.get(f"event_{index}_frame_list", [])


                ### Get all result for each Event Zone
                result_list = self.model.predict(
                    event_image_list, show=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD
                )


                ### Get all count result
                count_map_list = []
                for k, result in enumerate(result_list):
                    count_map = dict()
                    predict_frame = result.plot()

                    for i in range(len(result.boxes)):
                        res = result.boxes[i]
                        class_id = result.names[res.cls[0].item()]

                        count_map[class_id] = 1 + count_map.get(class_id, 0)

                    self.log_add("count: ", str(count_map))

                    cv2.imwrite(os.path.join("predict", f"predict_{index}_final_{k+1}.jpg"), predict_frame)
                    count_map_list.append(count_map)
                count_result = get_vote_count_result(count_map_list=count_map_list)

                ### Show result in OLED monitor, add one line (4 times). 
                self.pub.log(f"Ally: {count_result[KEY_PREDICT[0]]} / Enem: {count_result[KEY_PREDICT[1]]}")

        if self.running:
            self.running = False



#Eve
class EventMode(Mode):

    def __init__(self, pub, model, shared_list_model_second, index=0, n_frame=5, wait_sec=2.0):

        self.end = False
        self.index = index
        self.pub = pub
        self.model = model

        self.shared_list = shared_list_model_second

        self.phase = 1
        self.n_frame = n_frame
        self.n_frame_done = 0
        self.wait_frame_4_predict = WAIT_FRAME_4_MODEL * 2

        self.wait_sec = wait_sec
        self.enem_tank_x_list = []
        self.enem_tank_y_list = []

        self.rot_time = 0
        self.rot_speed = 0
        self.rot_total_angle = 0

        self.time_start = time.time()

        self.count_map_list = []


    def set_frame_and_move(self, frame, showoff=True):
        """
        phase 1: getting prediction for n_frame for every WAIT_FRAME_4_MODEL
        phase 2: /end: vote by each prediction! + get enem_tank position and rotational angle, time start
        phase 3: rotate to the tank position /end: shot! time start
        phase 4: rotate back to the original angle
        phase 5: wait for time ends: filling frames at once

        """
        # frame = cv2.imread("2356.jpg")

        predict_frame = frame
        self.log_set(self.index, "Event")
        self.log_set("phase", self.phase)


        ### phase 1: getting frames and run prediction for each. 
        if self.phase == 1:
            self.log_add("mode ", self.n_frame)
            move_robot(self.pub)

            if self.wait_frame_4_predict > 0:
                self.wait_frame_4_predict -= 1
                return
            self.wait_frame_4_predict = WAIT_FRAME_4_MODEL
            self.n_frame_done += 1

            # 모델에 맞는 이미지로 변환, 넣기.
            resize_image_4_model = get_resize_image_4_model(frame)


            ### saving frame in capsule for final prediction / End나 Model Second에서 쓰임
            ### 저장할 때부터 이미 이미지 resize 된 걸 저장하기로 함
            self.capsule[f"event_{self.index}_frame_list"] = self.capsule.get(f"event_{self.index}_frame_list", list()) + [resize_image_4_model]


            ### prediction works here
            self.pub.play_buzzer(440)
            result_list = self.model.predict(resize_image_4_model, show=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            self.pub.stop_buzzer()
            count_map = dict()
            for result in result_list:
                predict_frame = result.plot()

                for i in range(len(result.boxes)):
                    res = result.boxes[i]
                    cords = res.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    class_id = result.names[res.cls[0].item()]

                    count_map[class_id] = 1 + count_map.get(class_id, 0)

                    if class_id == KEY_PREDICT[-1]:
                        # 원래 이미지에서의 위치를 계산하고 받아내서 이용!
                        pos_x, pos_y = get_pos_before_xy(frame, resize_image_4_model, ((cords[0]+cords[2])/2, (cords[1]+cords[3])/2))
                        self.enem_tank_x_list.append(pos_x)
                        self.enem_tank_y_list.append(pos_y)

            self.log_add("count: ", str(count_map))

            cv2.imwrite(os.path.join("predict", f"predict_{self.index}_{self.n_frame_done}.jpg"), predict_frame)
            self.count_map_list.append(count_map)

            if self.n_frame - self.n_frame_done < 1:
                self.phase = 2


        ### phase 2: Get consensus of count result! 
        elif self.phase == 2:
            self.phase = 3
            self.time_start = time.time()

            count_result_map = get_vote_count_result(self.count_map_list)

            if count_result_map[KEY_PREDICT[-1]] > 0: # enem_tank
                n_xy = len(self.enem_tank_x_list)
                enem_tank_xy = (
                    sorted(self.enem_tank_x_list)[int((n_xy - 0.5) / 2)],
                    sorted(self.enem_tank_y_list)[int((n_xy - 0.5) / 2)],
                )
                angle_frame = np.zeros_like(frame)
                cv2.line(angle_frame, enem_tank_xy, (int(np.shape(angle_frame)[1] / 2), np.shape(angle_frame)[0]), 255, 2)
                angle_bev = get_bev(angle_frame)
                _bev, angle = get_road_edge_angle(angle_bev, ignore_canny=True)

                k = SPEED_Z
                if angle > 0:
                    k = -k
                angle = abs(angle)

                self.rot_time = TIME_90DEG * angle / 90
                self.rot_speed = k

                self.log_add("Enem tank angle", angle)
                self.log_add("rot speed", k)
                self.log_add("rot time", self.rot_time)

            # if enem tank is not found here: skip adjusting / cannon firing
            else:
                self.phase = 5

            self.log_add("prediction result: ", str(count_result_map))


            # model_second를 위해서 자료 제공
            self.shared_list[int(self.index/10)] = self.capsule[f"event_{self.index}_frame_list"]


        ### phase 3: For enem_tank: rotate the exact angle one time and fire cannon
        elif self.phase == 3:
            if self.rot_time > time.time() - self.time_start:
                self.log_add("speed: ", self.rot_speed)
                self.log_add("rotating: ", self.wait_sec)
                self.log_add("until: ", time.time() - self.time_start)
                move_robot(self.pub, 0, self.rot_speed)
            else:
                self.phase = 4
                self.time_start = time.time()
                move_robot(self.pub)
                self.pub.play_buzzer(880)
                self.pub.fire_cannon()
                self.pub.stop_buzzer()
                # BANG!!!


        ### phase 4: Rotate back the exact same angle (preparing next move)
        elif self.phase == 4:
            if self.rot_time > time.time() - self.time_start:
                self.log_add("speed: ", -self.rot_speed)
                self.log_add("rotating_back: ", self.wait_sec)
                self.log_add("until: ", time.time() - self.time_start)
                move_robot(self.pub, 0, -self.rot_speed)
            else:
                self.phase = 5
                self.time_start = time.time()
                move_robot(self.pub)


        ### phase 5: Waiting a little bit: to fill the frames again...
        elif time.time() - self.time_start < self.wait_sec:
            self.log_add("holding: ", self.wait_sec)
            self.log_add("until: ", time.time() - self.time_start)
        else:
            self.end = True

        if showoff:
            self.show_list = [predict_frame]




#S2G
class Stanley2GreenMode(Mode):

    def __init__(self, pub, index=0, from_it=False, left_offset=0, speed_weight=1.0):
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

        self.index = index

        self.speed_weight = speed_weight

        self.frame_without_line = 5

        self.frame_from_start_sensing = 0
        if from_it:
            self.frame_from_start_sensing = -20

    def set_frame_and_move(self, frame, showoff=True):
        """
            phase 1: stanley until find the green point on road
            phase 2: adjusting distance from green: slowing down / backing
        """
        self.frame_from_start_sensing += 1

        self.log_set(self.index, "Stanley2Green")
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
            move_robot(self.pub, -0.25)
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


        if self.phase == 1:
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
            # 만약 Green이 사라진다면? 일단 거기서 끝내기. 그럴 일은 없겠지만, 거기서 멈추기로 함.
            if green_max < TRUE_GREEN_CONF:
                self.end = True
                self.log_add("Green is Gone! ", green_max)
                return

            # 가까우면 음수, 멀면 양수, -0.5 ~ 0.25까지 나올 수 있음: BEV 따라서..
            dist_ratio = (get_2_point_dist((green_pos[1], green_pos[0]), (BOT_FROM_BEV_X, BOT_FROM_BEV_Y)) / PREFER_DIST) - 1
            self.log_add("Dist ratio ", dist_ratio)

            if abs(dist_ratio) > PREFER_ERR_RATIO:
                z = move_stanley(self.pub, offset_mm, angle_deg, dist_ratio)  # slow down a lot
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

    def __init__(self, pub, index=0, left_way=True, right_way=True, from_it=False, left_offset=0, use_green=False, speed_weight=1.0):
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
        self.phase = 1
        self.phase = 1

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
            Phase는 녹색을 쓸 때만: S2G와 동일 / 그 외에는 그냥, 진행시킴.
        """
        
        self.frame_from_start_sensing += 1

        self.log_set(self.index, "Stanley2Cross")
        bev = get_bev(frame)


        # slidingwindow & cross position!
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        cross_find_view, x_list, y_list, is_cross, positions = get_sliding_window_and_cross_result(road_blur_bev, self.left_way, self.right_way, self.init_pos_for_sliding_windows)
        road_sw_bev = cross_find_view

        # green event!
        green_bev = get_green(bev)
        green_bev_cm = get_cm_px_from_mm(green_bev)
        green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 5)
        green_pos = [pos * 10 for pos in green_pos_cm]



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
            move_robot(self.pub, -0.25)
            return


        # stanley
        offset_mm = self.line_road.get_offset(BOT_FROM_BEV_X + self.left_offset, BOT_FROM_BEV_Y)
        angle_deg = self.line_road.get_angle()


        # Phase 2는 녹색을 쓰는 경우만 / 그때는 거리에 맞춰서 속도 줄이고 할 예정.
        if self.phase == 1:
            z = move_stanley(self.pub, offset_mm, angle_deg, x_ratio=self.speed_weight)

        self.log_add("offset", offset_mm)
        self.log_add("angle", angle_deg)
        self.log_add("speed_z", z)

        self.log_add("Cross position", positions)

        if is_cross and self.frame_from_start_sensing > 0:
            self.cross_encounter += 1
            self.log_add("cross?", self.cross_encounter)
        else:
            self.cross_encounter = max(0, self.cross_encounter-1)

        if self.cross_encounter >= 3:
            self.end = True
            move_robot(self.pub)
            self.capsule["dist_from_cross"] = BOT_FROM_BEV_Y - np.mean(positions)


        # Green 보고 처리할 때의 코드: phase랑 use_green 확인, green 위치랑 거리 확인, 감속까지.
        if self.use_green and self.phase == 1:
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

        elif self.use_green and self.phase == 2:
            # 만약 Green이 사라진다면? 일단 거기서 끝내기. 그럴 일은 없겠지만, 거기서 멈추기로 함.
            if green_max < TRUE_GREEN_CONF:
                self.end = True
                self.log_add("Green is Gone! ", green_max)
                return

            # 가까우면 음수, 멀면 양수, -0.5 ~ 0.25까지 나올 수 있음: BEV 따라서..
            dist_ratio = (get_2_point_dist((green_pos[1], green_pos[0]), (BOT_FROM_BEV_X, BOT_FROM_BEV_Y)) / PREFER_DIST) - 1
            self.log_add("Dist ratio ", dist_ratio)

            if abs(dist_ratio) > PREFER_ERR_RATIO:
                z = move_stanley(self.pub, offset_mm, angle_deg, dist_ratio)  # slow down a lot
            else:
                z = move_robot(self.pub)  # stop
                self.end = True

        # showoff now
        if showoff:
            cv2.line(road_sw_bev,
                     (int(self.line_road.calc(0)), 0),(int(self.line_road.calc(np.shape(road_sw_bev)[0])),np.shape(road_sw_bev)[0],),
                     (0, 0, 255),5)
            self.show_list = [frame, bev, road_bev, road_sw_bev, get_mm_px_from_cm(green_bev_cm), get_mm_px_from_cm(green_blur_bev)]




class Turn2RoadMode(Mode):

    def __init__(self, pub, index=0, is_left=True, min_turn_sec=1.2, is_curve=False):
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
        self.min_turn_sec = min_turn_sec
        self.is_curve = is_curve

        self.road_encounter = 0

        self.road_angle = -1000
        self.dist_from_cross = -1

        self.phase = 0
        self.time_since_phase = 0
        self.est_time = 0

        self.rot_z = SPEED_Z
        self.speed_x = 0

        self.index = index

        self.capsule = {"dist_from_cross": BOT_FROM_BEV_Y - BEV_SHAPE[0]}

    def set_frame_and_move(self, frame, showoff=True):
        """
        phase 0: / time starting + getting angle and est_time, SPEED_X, rot_z
        phase 1: rotating at least min_turn_sec
        phase 2: rotate until you see the line
        """

        self.log_set(self.index, "Turn2Road")
        self.log_add("phase", self.phase)

        bev = get_bev(frame)
        road_bev = get_road(bev)
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
            move_robot(self.pub, self.speed_x, self.rot_z, self.is_left)

            if time.time() - self.time_since_phase > self.min_turn_sec:
                self.phase = 2
                self.time_since_phase = time.time()
                # move_robot(self.pub)

        # Phase 2. turning while the line is shown: to estimate time to be exact 90 degrees
        if self.phase == 2:
            self.log_add("no_line")

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
                
                # needs 2 time for road_encounter
                if self.road_encounter >= 2:
                    self.end = True

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
        self.log_set(self.index, "Turn2Void")
        self.log_add("phase", self.phase)


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
                if abs(angle) > 45:
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
