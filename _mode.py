#! /usr/bin/env python
# -*- coding: utf-8 -*-
# %matplotlib inline

from __future__ import print_function, division


import rospy
import cv2
import numpy as np
import time
import math

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from math import *
from collections import deque
from std_msgs.msg import String


# from ultralytics import YOLO


from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur
from _lane_detect import get_cm_px_from_mm, get_square_pos, get_road_edge_angle, get_sliding_window_and_cross_result, Line


BOT_FROM_BEV_X = 100 # edit this
BOT_FROM_BEV_Y = 400 # edit this

SPEED_X = 0.4
SPEED_Z = 0.9
TIME_90DEG = 1.5 / SPEED_Z

RADIUS_VZ_OVER_VX_CONST = 1200  # edit this

TRUE_GREEN_CONF = 100
TRUE_GREEN_DIST_FROM_ROAD = 30 #mm


BEV_SHAPE = (200, 200)

# for event:
CONF_THRESHOLD = 0.6
IOU_THRESHOLD = 0.6
WAIT_FRAME_4_YOLO = 5

KEY_PREDICT = ("ally", "enem", "ally_tank", "enem_tank")

PREFER_ERR_DEG = 5

PREFER_DIST = 300
PREFER_ERR_RATIO = 0.1




def showing_off(image_list):
#    return

    px = [0, 600, 1200, 0, 600, 1200, 0, 600, 1200]
    py = [0, 0, 0, 400, 400, 400, 800, 800, 800]

    for i, frame in enumerate(image_list):

        if i > 8:
            break
        cv2.namedWindow("win_" + str(i+1))
        cv2.moveWindow("win_" + str(i+1), px[i], py[i])
        cv2.imshow("win_" + str(i+1), frame)




def move_robot(pub, vel_x=0, rot_z=0, is_left=True):

    speed = Twist()
    speed.linear.x = vel_x
    speed.angular.z = rot_z
    if not is_left:
        speed.angular.z = -rot_z
    pub.publish(speed)
    '''
    pub.set_motor_power(motor, value)
    

    '''

def move_stanley(pub, offset_mm, angle_deg, SPEED_X_ratio = 1):

    kp= 0.04
    ka= 0.10
    k = 1.5
    x = SPEED_X * SPEED_X_ratio

    z = -(angle_deg*ka - atan(kp*offset_mm)) * x * k
    
    move_robot(pub, x, z)

    return z



def get_vote_count_result(count_map_list, KEY_PREDICT):
    '''
        getting the most shown value for each key
    '''


    result = dict()
    for key in KEY_PREDICT:
        list_count = []
        for count_map in count_map_list:
            list_count.append(count_map.get(key, 0))

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

    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


class Mode:

    end = False
    pub = None
    log = ""
    running = True
    phase = 0
    capsule = dict()
    capsule = dict()
    index = 0

    def __init__(self, pub):
        self.end = False
        self.pub = pub
    
    def set_frame_and_move(self, frame, showoff=True):
        pass

    def log_add(self, a, b=""):
        self.log += "  | " + str(a) + " " + str(b)

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


class EndMode(Mode):

    def __init__(self, pub, index=0):
        self.end = False
        self.pub = pub
        move_robot(pub)
        self.running = False
    
    def set_frame_and_move(self, frame, showoff=True):
        
        if showoff:
            pass
        cv2.waitKey(1)
        pass



class EventMode(Mode):

    def __init__(self, pub, model, index = 0, n_frame=5, wait_sec = 2.0):
        self.lcd_pub_1 = rospy.Publisher("/lcd_str_1", String, queue_size=1)
        self.str_msg_1 = String()
        
        self.end = False
        self.pub = pub
        
        self.phase = 1
        self.n_frame = n_frame
        self.WAIT_FRAME_4_YOLO = WAIT_FRAME_4_YOLO * 2

        self.wait_sec = wait_sec
        self.index = index
        self.model = model
        self.enem_tank_x_list = []
        self.enem_tank_y_list = []

        
        self.rot_time = 0
        self.rot_speed = 0
        self.rot_total_angle = 0

        self.time_start = time.time()

        self.count_map_list = []
    
    def set_frame_and_move(self, frame, showoff=True):
        '''
            phase 1: getting prediction for n_frame for every WAIT_FRAME_4_YOLO
            phase 2: /end: vote by each prediction! + get enem_tank position and rotational angle, time start
            phase 3: rotate to the tank position /end: shot! time start
            phase 4: rotate back to the original angle
            phase 5: wait for time ends: filling frames at once
            
        '''
        # frame = cv2.imread("2356.jpg")

        predict_frame = frame
        self.log_set(self.index, "Event") 


        if self.phase == 1:
            self.log_add("mode ", self.n_frame)
            if self.WAIT_FRAME_4_YOLO > 0:
                self.WAIT_FRAME_4_YOLO -= 1
                return
            self.n_frame -= 1
            self.WAIT_FRAME_4_YOLO = WAIT_FRAME_4_YOLO

            result_list = self.model.predict(frame, device = 'cpu', show=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            # result_list = self.model.predict(frame, device = 'cuda', show=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            
            count_map = dict()
            # print(result_list)
            for result in result_list:
                predict_frame = result.plot()
                if len(result.boxes) < 1:
                    continue   

                for i in range(len(result.boxes)):
                    res = result.boxes[i] 
                    cords = res.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    class_id = result.names[res.cls[0].item()]

                    count_map[class_id] = 1 + count_map.get(class_id, 0)
            
                    if class_id == 'enem_tank': # KEY_PREDICT[-1]
                        self.enem_tank_x_list.append(int((cords[0]+cords[2])/2))
                        self.enem_tank_y_list.append(int((cords[1]+cords[3])/2))
        
            self.log_add("count: ", str(count_map))

            cv2.imwrite("predict_" + str(self.index) + "_" + str(self.n_frame) + ".jpg", predict_frame)
            self.count_map_list.append(count_map)

            if self.n_frame < 1:
                self.phase = 2
        
        elif self.phase == 2:
            self.phase = 3
            self.time_start = time.time()

            count_result_map = get_vote_count_result(self.count_map_list, KEY_PREDICT)
            
            if len(self.enem_tank_x_list) > 0:
                n_xy = len(self.enem_tank_x_list)
                enem_tank_xy = sorted(self.enem_tank_x_list)[int((n_xy-0.5)/2)], sorted(self.enem_tank_y_list)[int((n_xy-0.5)/2)],                  
                angle_frame = np.zeros_like(frame)
                cv2.line(angle_frame, enem_tank_xy, (int(np.shape(angle_frame)[1]/2), np.shape(angle_frame)[0]), 255, 2)

                angle_bev = get_bev(angle_frame)
                angle_show_bev, angle = get_road_edge_angle(angle_bev, ignore_canny=True)

                k = SPEED_Z
                if angle > 0:
                    k = -k
                angle = abs(angle)

                self.rot_time = TIME_90DEG * angle / 90
                self.rot_speed = k

                self.log_add("Enem tank angle", angle)
                self.log_add("rot speed", k)
                self.log_add("rot time", self.rot_time)

            # self.str_msg_1.data = f"alli: {count_result_map['ally']}/{count_result_map['ally_tank']},enem: {count_result_map['enem']}/{count_result_map['enem_tank']}"
            # self.lcd_pub_1.publish(self.str_msg_1)
            self.log_add("prediction result: ", str(count_result_map))  

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
                # BANG!!!

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
                # BANG!!!

        # phase 5  
        elif time.time() - self.time_start < self.wait_sec:
            self.log_add("holding: ", self.wait_sec)
            self.log_add("until: ", time.time() - self.time_start)
        else:
            self.end = True

        if showoff:
            showing_off([frame, predict_frame])


class Stanley2GreenMode(Mode):

    def __init__(self, pub, index=0, from_it=False, left_offset = 0, debug=False):
        self.end = False
        self.pub = pub

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.green_encounter = -2
        if from_it:
            self.green_encounter = -100
        self.left_offset = left_offset

        self.phase = 1

        self.index = index
        self.debug = debug


    def set_frame_and_move(self, frame, showoff=True):

        self.log_set(self.index, "Stanley2Green") 
        bev = get_bev(frame)
        self.log_add("phase ", self.phase)


        # slidingwindow
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

        if len(x_list) > 2:
            self.init_pos_for_sliding_windows = x_list[1]
            self.line_road = Line(x_list, y_list)
        else:
            self.init_pos_for_sliding_windows = -1
        

        if self.line_road == None:
            # Must find the line here, First!
            # print("What, No Road? You Real? BRUHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            showing_off([frame, road_bev, road_sw_bev, bev])
            move_robot(self.pub)
            return

        # green event!
        green_bev = get_green(bev)
        green_bev_cm = get_cm_px_from_mm(green_bev)
        green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 7)
        green_pos = [pos*10 for pos in green_pos_cm]

        offset_mm = self.line_road.get_offset(BOT_FROM_BEV_X+self.left_offset,BOT_FROM_BEV_Y)
        angle_deg = self.line_road.get_angle()
        self.log_add("offset", offset_mm)
        self.log_add("angle", angle_deg)

        if self.phase == 1:
            # stanley

            z = move_stanley(self.pub, offset_mm, angle_deg)
            self.log_add("z speed ", str(z))

            if green_max > TRUE_GREEN_CONF and self.line_road.get_distance(green_pos[1], green_pos[0]) < TRUE_GREEN_DIST_FROM_ROAD:
                self.log_add("true green?", green_max)
                self.log_add("true green at", green_pos)
                self.log_add("true green from line", self.line_road.get_distance(green_pos[1], green_pos[0]))
                self.green_encounter += 1
            else:
                self.green_encounter -= 1
                self.green_encounter = max(int(self.green_encounter/2.1), self.green_encounter)
            
            if self.green_encounter >= 3:
                self.phase = 2
                # move_robot(self.pub)
        
        elif self.phase == 2:
                
            if green_max < TRUE_GREEN_CONF:
                self.end = True
                self.log_add("Green is Gone! ", green_max)
                return

            dist_ratio = (get_2_point_dist((green_pos[1], green_pos[0]), (BOT_FROM_BEV_X, BOT_FROM_BEV_Y)) / PREFER_DIST) - 1
            self.log_add("Dist ratio ", dist_ratio)
            
            if dist_ratio > PREFER_ERR_RATIO:
                z = move_stanley(self.pub, offset_mm, angle_deg, dist_ratio) # slow down a lot
            else:
                z = move_robot(self.pub) # stop
                self.end = True


        # showoff now
        if showoff:
            cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
            showing_off([frame, road_bev, road_sw_bev, bev, green_bev_cm, green_blur_bev])
        if self.debug:
            cv2.imwrite("S2G_" + str(self.index) + "_debug_green" + str(self.green_encounter) + ".jpg", green_bev)
            cv2.imwrite("S2G_" + str(self.index) + "_debug_bev" + str(self.green_encounter) + ".jpg", bev)


class Stanley2CrossMode(Mode):

    def __init__(self, pub, index=0, left_way = True, right_way = True, from_it=False, left_offset = 0, use_green = False):
        self.end = False
        self.pub = pub

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.green_encounter = -2
        self.green_encounter = -2
        self.cross_encounter = -2
        if from_it:
            self.cross_encounter = -1000
        self.left_way = left_way
        self.right_way = right_way
        self.left_offset = left_offset
        self.phase = 1
        self.phase = 1

        self.index = index
        
        self.capsule = dict()
        self.use_green = use_green


    def set_frame_and_move(self, frame, showoff=True):

        self.log_set(self.index, "Stanley2Cross") 
        bev = get_bev(frame)

        # slidingwindow & cross position!
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        cross_find_view, x_list, y_list, is_cross, positions = get_sliding_window_and_cross_result(road_blur_bev, 5, self.left_way, self.right_way, self.init_pos_for_sliding_windows)
        road_sw_bev = cross_find_view

        # green event!
        green_bev = get_green(bev)
        green_bev_cm = get_cm_px_from_mm(green_bev)
        green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 7)
        green_pos = [pos*10 for pos in green_pos_cm]
        # green event!
        green_bev = get_green(bev)
        green_bev_cm = get_cm_px_from_mm(green_bev)
        green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 7)
        green_pos = [pos*10 for pos in green_pos_cm]

        if len(x_list) > 2:
            self.init_pos_for_sliding_windows = x_list[1]
            self.line_road = Line(x_list, y_list)
        else:
            self.init_pos_for_sliding_windows = -1
        
        if self.line_road == None:
            # Must find the line here, First!
            # print("What, No Road? You Real? BRUHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            showing_off([frame, road_bev, road_sw_bev, bev])
            # move_robot(self.pub)
            return


        # stanley
        offset_mm = self.line_road.get_offset(BOT_FROM_BEV_X+self.left_offset,BOT_FROM_BEV_Y)
        angle_deg = self.line_road.get_angle()

        z = move_stanley(self.pub, offset_mm, angle_deg)

        self.log_add("offset", offset_mm)
        self.log_add("angle", angle_deg)
        self.log_add("SPEED_Z", z)


        # print(cross_pos, cross_max)
        self.log_add("Cross position", positions)

        if is_cross:
            self.cross_encounter += 1
        else:
            self.cross_encounter -= 1
            self.cross_encounter = max(0, self.cross_encounter)
        
        if self.cross_encounter >= 3:
            self.end = True
            move_robot(self.pub)
            self.capsule["dist_from_cross"] = BOT_FROM_BEV_Y - np.mean(positions)

        
        if self.use_green and self.phase == 1:


            if green_max > TRUE_GREEN_CONF and self.line_road.get_distance(green_pos[1], green_pos[0]) < TRUE_GREEN_DIST_FROM_ROAD:
                self.log_add("true green?", green_max)
                self.log_add("true green at", green_pos)
                self.log_add("true green from line", self.line_road.get_distance(green_pos[1], green_pos[0]))
                self.green_encounter += 1
            else:
                self.green_encounter -= 1
                self.green_encounter = max(int(self.green_encounter/2.1), self.green_encounter)
            
            if self.green_encounter >= 3:
                self.phase = 2
        
        elif self.use_green and self.phase == 2:
                
            if green_max < TRUE_GREEN_CONF:
                self.end = True
                self.log_add("Green is Gone! ", green_max)
                return

            dist_ratio = 1 - (get_2_point_dist((green_pos[1], green_pos[0]), (BOT_FROM_BEV_X, BOT_FROM_BEV_Y)) / PREFER_DIST)
            self.log_add("Dist ratio ", dist_ratio)
            
            if dist_ratio > PREFER_ERR_RATIO:
                z = move_stanley(self.pub, offset_mm, angle_deg, dist_ratio) # slow down a lot
            else:
                z = move_robot(self.pub) # stop
                self.capsule["dist_from_cross"] = 0
                self.end = True



        # showoff now
        if showoff:
            cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
            showing_off([frame, road_bev, road_sw_bev, bev, cross_find_view])




class Turn2VoidMode(Mode):

    def __init__(self, pub, index=0, is_left = True, other_turn_sec = 0.2):
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
        '''
            phase 0: time starting
            phase 1: rotating to other side a little bit / end: get angle and est_time
            phase 2: getting all angle data while rotating : Now ignored
            phase 3: waiting / end: stop
        '''
        self.log_set(self.index, "Turn2Void") 
        self.log_add("phase", self.phase)

        bev = get_bev(frame)

        # road edge angle detection
        road_bev = get_road(bev)
        road_edge_bev, angle = get_road_edge_angle(road_bev, self.is_left)


        # phase 0: time starting
        if self.phase == 0:
            self.phase = 1
            self.time_since_phase = time.time()


        # phase 1: rotating to other side a little bit: to get max data and right angle.
        if self.phase == 1:
            move_robot(self.pub, 0, -SPEED_Z, self.is_left)

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
            move_robot(self.pub, 0, SPEED_Z, self.is_left)

            # to skip the image using phase...
            self.phase = 3

            '''
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
            '''

                    
        if self.phase == 3:

            if time.time() < self.time_since_phase + self.est_time:
                move_robot(self.pub, 0, SPEED_Z, self.is_left)
            else:
                move_robot(self.pub)
                self.end = True

        if showoff:
            showing_off([frame, road_bev, road_edge_bev, bev])


class Turn2RoadMode(Mode):

    def __init__(self, pub, index = 0, is_left = True, min_turn_sec = 1.2, is_curve = False, left_way = True, right_way = True):
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
        self.SPEED_X = SPEED_X
        self.left_way = left_way
        self.right_way = right_way

        self.index = index

        self.capsule = {"dist_from_cross": BOT_FROM_BEV_Y - BEV_SHAPE[0]}


    def set_frame_and_move(self, frame, showoff=True):
        '''
            phase 0: / time starting + getting angle and est_time, SPEED_X, rot_z
            phase 1: rotating at least min_turn_sec 
            phase 2: rotate until you see the line
        '''
        self.log_set(self.index, "Turn2Road") 
        self.log_add("phase", self.phase)

        bev = get_bev(frame)
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)

        road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)


        if self.phase == 0 and self.is_curve:
            dist_from_cross = self.capsule["dist_from_cross"]
            self.log_add(self.capsule)


            road_edge_bev, angle = get_road_edge_angle(road_bev, self.is_left)
            self.road_angle = angle
            if abs(self.road_angle) > 20:
                self.road_angle = 0

            if self.is_left:
                self.road_angle = -self.road_angle

            radius = dist_from_cross / (1 + math.sin(self.road_angle*math.pi/180))

            self.SPEED_X = radius * self.rot_z / RADIUS_VZ_OVER_VX_CONST
            self.log_add("radius", radius)
            self.log_add("rot_z", self.rot_z)
            self.log_add("SPEED_X", self.SPEED_X)
            # cv2.imwrite(str(self.index) + "_curve_dist.jpg", road_sw_bev)
            # cv2.imwrite(str(self.index) + "_curve_angle.jpg", road_edge_bev)
        elif self.phase == 0 and not self.is_curve:
            self.SPEED_X = 0
        
        

        # starting
        if self.phase == 0:
            self.phase = 1
            self.time_since_phase = time.time()
       

        # turning at least certain amount: to ignore post-road
        if self.phase == 1:
            move_robot(self.pub, self.SPEED_X, self.rot_z, self.is_left)

            if time.time() - self.time_since_phase > self.min_turn_sec:
                self.phase = 2
                self.time_since_phase = time.time()
                # move_robot(self.pub)
        
        # turning while the line is shown: to estimate time to be exact 90 degrees
        if self.phase == 2:
            self.log_add("line_not_shown")

            move_robot(self.pub, self.SPEED_X, self.rot_z, self.is_left)

            if len(x_list) > 2:
                self.init_pos_for_sliding_windows = x_list[1]
                line_road = Line(x_list, y_list)
                self.line_road = line_road

                self.log_add("line_on_angle", line_road.get_angle())

                cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
                
                if len(x_list) > 4 or abs(line_road.get_angle()) < 10:
                    # move_robot(self.pub)
                    self.road_encounter += 1
                if self.road_encounter > 1:
                    self.end = True
            # cv2.imwrite(str(self.index) + "_T2R_" + str(round((time.time() - self.time_since_phase)*1000, 0)) + ".jpg", road_sw_bev)
 

        if showoff:
            showing_off([frame, road_bev, road_sw_bev, bev])




