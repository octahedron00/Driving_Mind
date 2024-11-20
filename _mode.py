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


from ultralytics import YOLO


from _lane_detect import get_bev, get_road, get_sliding_window_result, get_green, get_rect_blur
from _lane_detect import get_cm_px_from_mm, get_square_pos, get_road_edge_angle, get_sliding_window_and_cross_result, Line


bot_from_bev_x = 100 # edit this
bot_from_bev_y = 400 # edit this

speed_x = 0.4
z_ang_speed = 1
default_time_90deg = 2.5 / z_ang_speed

radius_vx_vz_coefficient = 900  # edit this

true_green_confidence = 140
true_green_dist_from_road = 30 #mm


# for event:
conf_threshold = 0.6
iou_threshold = 0.6
sleep_frame = 5

key_predict = ("ally", "enem", "ally_tank", "enem_tank")



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

def move_stanley(pub, offset_mm, angle_deg):

    kp= 0.03
    ka= 0.15
    k = 1
    x = speed_x

    z = -(angle_deg*ka - atan(kp*offset_mm)) * x * k
    
    move_robot(pub, x, z)

    return z



def get_vote_count_result(count_map_list, key_predict):
    '''
        getting the most shown value for each key
    '''


    result = dict()
    for key in key_predict:
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

                



class Mode:

    end = False
    pub = None
    log = ""
    running = True
    stage = 0

    def __init__(self, pub):
        self.end = False
        self.pub = pub
        self.log = ""
    
    def set_frame_and_move(self, frame, showoff=True):
        pass

    def log_add(self, a, b=""):
        self.log += "  | " + str(a) + " " + str(b)


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
        self.log = ""
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
        
        self.stage = 1
        self.n_frame = n_frame
        self.sleep_frame = sleep_frame * 2

        self.wait_sec = wait_sec
        self.index = index
        self.log = str(self.index) + "_Event_"
        self.model = model
        self.enem_tank_x_list = []
        self.enem_tank_y_list = []

        
        self.rot_time = 0
        self.rot_speed = 0

        self.time_start = time.time()

        self.count_map_list = []
    
    def set_frame_and_move(self, frame, showoff=True):
        '''
            stage 1: getting prediction for n_frame for every sleep_frame
            stage 2: /end: vote by each prediction! + get enem_tank position and rotational angle, time start
            stage 3: rotate to the tank position /end: shot! time start
            stage 4: rotate back to the original angle
            stage 5: wait for time ends: filling frames at once
            
        '''
        # frame = cv2.imread("2356.jpg")

        predict_frame = frame
        self.log = str(self.index) + "_Event_"

        if self.stage == 1:
            self.log_add("mode ", self.n_frame)
            if self.sleep_frame > 0:
                self.sleep_frame -= 1
                return
            self.n_frame -= 1
            self.sleep_frame = sleep_frame

            result_list = self.model.predict(frame, device = 'cpu', show=False, conf=conf_threshold, iou=iou_threshold)
            # result_list = self.model.predict(frame, device = 'cuda', show=False, conf=conf_threshold, iou=iou_threshold)
            
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
            
                    if class_id == 'enem_tank': # key_predict[-1]
                        self.enem_tank_x_list.append(int((cords[0]+cords[2])/2))
                        self.enem_tank_y_list.append(int((cords[1]+cords[3])/2))
        
            self.log_add("count: ", str(count_map))

            cv2.imwrite("predict_" + str(self.index) + "_" + str(self.n_frame) + ".jpg", predict_frame)
            self.count_map_list.append(count_map)

            if self.n_frame < 1:
                self.stage = 2
        
        elif self.stage == 2:
            self.stage = 3
            self.time_start = time.time()

            count_result_map = get_vote_count_result(self.count_map_list, key_predict)
            
            if self.enem_tank_xy[0] >= 0:
                angle_frame = np.zeros_like(frame)
                cv2.line(angle_frame, self.enem_tank_xy, (int(np.shape(angle_frame)[1]/2), np.shape(angle_frame)[0]), 255, 2)

                angle_bev = get_bev(angle_frame)
                angle_show_bev, angle = get_road_edge_angle(angle_bev, ignore_canny=True)

                k = z_ang_speed
                if angle > 0:
                    k = -k
                angle = abs(angle)

                self.rot_time = default_time_90deg * angle / 90
                self.rot_speed = k

                self.log_add("Enem tank angle", angle)
                self.log_add("rot speed", k)
                self.log_add("rot time", self.rot_time)

            # self.str_msg_1.data = f"alli: {count_result_map['ally']}/{count_result_map['ally_tank']},enem: {count_result_map['enem']}/{count_result_map['enem_tank']}"
            # self.lcd_pub_1.publish(self.str_msg_1)
            self.log_add("prediction result: ", str(count_result_map))  

        elif self.stage == 3:
            if self.rot_time > time.time() - self.time_start:
                self.log_add("speed: ", self.rot_speed)
                self.log_add("rotating: ", self.wait_sec)
                self.log_add("until: ", time.time() - self.time_start)
                move_robot(self.pub, 0, self.rot_speed)
            else:
                self.stage = 4
                self.time_start = time.time()
                move_robot(self.pub)
                # BANG!!!

        elif self.stage == 4:
            if self.rot_time > time.time() - self.time_start:
                self.log_add("speed: ", -self.rot_speed)
                self.log_add("rotating_back: ", self.wait_sec)
                self.log_add("until: ", time.time() - self.time_start)
                move_robot(self.pub, 0, -self.rot_speed)
            else:
                self.stage = 5
                self.time_start = time.time()
                move_robot(self.pub)
                # BANG!!!

        # stage 5  
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

        self.index = index
        self.log = str(self.index) + "_Stanley2Green_"
        self.debug = debug


    def set_frame_and_move(self, frame, showoff=True):

        self.log = str(self.index) + "_Stanley2Green_"
        bev = get_bev(frame)

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


        # stanley
        offset_mm = self.line_road.get_offset(bot_from_bev_x+self.left_offset,bot_from_bev_y)
        angle_deg = self.line_road.get_angle()

        z = move_stanley(self.pub, offset_mm, angle_deg)
        self.log = str(self.index) + "_Stanley2Green_offset " + str(offset_mm) + " mm / angle " + str(angle_deg) + "deg / z speed " + str(z)


        # green event!
        green_bev = get_green(bev)
        green_bev_cm = get_cm_px_from_mm(green_bev)
        green_blur_bev, green_pos_cm, green_max = get_square_pos(green_bev_cm, 7)
        green_pos = [pos*10 for pos in green_pos_cm]

        if green_max > true_green_confidence and self.line_road.get_distance(green_pos[1], green_pos[0]) < true_green_dist_from_road:
            print("What, the true Green!!!", green_max, self.line_road.get_distance(green_pos[1], green_pos[0]))
            self.green_encounter += 1
        else:
            self.green_encounter -= 1
            self.green_encounter = max(int(self.green_encounter/2.1), self.green_encounter)
        
        if self.green_encounter >= 3:
            self.end = True
            # move_robot(self.pub)
        

        # showoff now
        if showoff:
            cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
            showing_off([frame, road_bev, road_sw_bev, bev, green_bev_cm, green_blur_bev])
        if self.debug:
            cv2.imwrite("S2G_" + str(self.index) + "_debug_green" + str(self.green_encounter) + ".jpg", green_bev)
            cv2.imwrite("S2G_" + str(self.index) + "_debug_bev" + str(self.green_encounter) + ".jpg", bev)


class Stanley2CrossMode(Mode):

    def __init__(self, pub, index=0, left_way = True, right_way = True, from_it=False, left_offset = 0):
        self.end = False
        self.pub = pub

        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.cross_encounter = -2
        if from_it:
            self.cross_encounter = -1000
        self.left_way = left_way
        self.right_way = right_way
        self.left_offset = left_offset

        self.index = index
        self.log = str(self.index) + "_Stanley2Cross_"


    def set_frame_and_move(self, frame, showoff=True):

        self.log = str(self.index) + "_Stanley2Cross_"
        bev = get_bev(frame)

        # slidingwindow & cross position!
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        cross_find_view, x_list, y_list, is_cross, positions = get_sliding_window_and_cross_result(road_blur_bev, 5, self.left_way, self.right_way, self.init_pos_for_sliding_windows)
        road_sw_bev = cross_find_view


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
        offset_mm = self.line_road.get_offset(bot_from_bev_x+self.left_offset,bot_from_bev_y)
        angle_deg = self.line_road.get_angle()

        z = move_stanley(self.pub, offset_mm, angle_deg)
        self.log = str(self.index) + "_Stanley2Cross_offset " + str(offset_mm) + " mm / angle " + str(angle_deg) + "deg / z speed " + str(z)


        # print(cross_pos, cross_max)
        self.log = self.log + " CROSS::" + str(positions)

        if is_cross:
            print("OK, Now We At the Cross!!!", positions)
            self.cross_encounter += 1
        else:
            self.cross_encounter -= 1
            self.cross_encounter = max(0, self.cross_encounter)
        
        if self.cross_encounter >= 3:
            self.end = True
            # move_robot(self.pub)
            self.memory = np.mean(positions)
        

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

        self.stage = 0
        self.time_since_stage = 0
        self.angle_at_start = -90
        self.angle_list = []
        self.time_list = []
        self.est_time = 0
        self.est_time_regression = 0
        self.est_time_angle_calc = 0

        self.waiting_for_next_frame = 2
        
        self.index = index
        self.log = str(self.index) + "_Turn2Void_"


    def set_frame_and_move(self, frame, showoff=True):
        '''
            stage 0: time starting
            stage 1: rotating to other side a little bit / end: get angle and est_time
            stage 2: getting all angle data while rotating : Now ignored
            stage 3: waiting / end: stop
        '''
        self.log = str(self.index) + "_Turn2Void_"
        self.log_add("stage", self.stage)

        bev = get_bev(frame)

        # road edge angle detection
        road_bev = get_road(bev)
        road_edge_bev, angle = get_road_edge_angle(road_bev, self.is_left)


        # stage 0: time starting
        if self.stage == 0:
            self.stage = 1
            self.time_since_stage = time.time()


        # stage 1: rotating to other side a little bit: to get max data and right angle.
        if self.stage == 1:
            move_robot(self.pub, 0, -z_ang_speed, self.is_left)

            if time.time() - self.time_since_stage > self.other_turn_sec:
                self.stage = 2  
                self.time_since_stage = time.time()  
                move_robot(self.pub)

                angle_at_start = angle

                if self.is_left:  
                    self.est_time_angle_calc = (default_time_90deg / 90) * (90 - angle_at_start)
                else:
                    self.est_time_angle_calc = (default_time_90deg / 90) * (90 + angle_at_start)
                self.est_time = self.est_time_angle_calc
                if abs(angle) > 45:
                    self.est_time = default_time_90deg

        

        # turning while the line is shown: to estimate time to be exact 90 degrees
        if self.stage == 2:
            move_robot(self.pub, 0, z_ang_speed, self.is_left)

            # to skip the image using stage...
            self.stage = 3

            if abs(angle) < 45:
                self.waiting_for_next_frame = 2
                self.time_list.append(time.time() - self.time_since_stage)
                self.angle_list.append(angle)
                
                # cv2.imwrite(str(self.index) + "_frame_edge_" + str(len(self.time_list)) + ".jpg", road_edge_bev)
                # cv2.imwrite(str(self.index) + "_frame_original_" + str(len(self.time_list)) + ".jpg", road_bev)
                self.log = str(self.index) + "_Turn2Void_stage" + str(self.stage) + " " + str(angle)
            elif self.waiting_for_next_frame > 0:
                self.waiting_for_next_frame -= 1
            else:
                self.stage = 3
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


                    
        if self.stage == 3:
            self.log = str(self.index) + "_Turn2Void_stage" + str(self.stage) + " " + str(self.est_time)

            if time.time() < self.time_since_stage + self.est_time:
                move_robot(self.pub, 0, z_ang_speed, self.is_left)
            else:
                # move_robot(self.pub)
                self.end = True

        if showoff:
            showing_off([frame, road_bev, road_edge_bev, bev])


class Turn2RoadMode(Mode):

    def __init__(self, pub, index = 0, is_left = True, min_turn_sec = 1.5, is_curve = False, left_way = True, right_way = True):
        self.end = False
        self.pub = pub

        self.is_left = is_left
        self.line_road = None
        self.init_pos_for_sliding_windows = -1
        self.min_turn_sec = min_turn_sec
        self.is_curve = is_curve

        self.road_angle = -1000
        self.dist_from_cross = -1

        self.stage = 0
        self.time_since_stage = 0
        self.est_time = 0

        self.rot_z = z_ang_speed
        self.speed_x = speed_x
        self.left_way = left_way
        self.right_way = right_way

        self.index = index
        self.log = str(self.index) + "_Turn2Road_"


    def set_frame_and_move(self, frame, showoff=True):
        '''
            stage 0: / time starting + getting angle and est_time, speed_x, rot_z
            stage 1: rotating at least min_turn_sec 
            stage 2: rotate until you see the line
        '''

        self.log = str(self.index) + "_Turn2Road_"
        self.log_add("stage", self.stage)

        bev = get_bev(frame)
        road_bev = get_road(bev)
        road_blur_bev = get_rect_blur(road_bev, 5)
        

        if self.stage == 0 and self.is_curve:
            road_sw_bev, x_list, y_list, is_cross, positions = get_sliding_window_and_cross_result(road_blur_bev, 5, self.left_way, self.right_way, self.init_pos_for_sliding_windows)
            if len(positions) > 0:
                dist_from_cross = bot_from_bev_y - np.mean(positions)
            else:
                dist_from_cross = bot_from_bev_y - np.shape(road_sw_bev)[0]

            not_right = True
            if self.is_left:
                not_right = False
            road_edge_bev, angle = get_road_edge_angle(road_bev, not_right)
            self.road_angle = angle
            if abs(self.road_angle) > 20:
                self.road_angle = 0

            if self.is_left:
                self.road_angle = -self.road_angle

            radius = dist_from_cross / (1 + math.sin(self.road_angle*math.pi/180))

            self.speed_x = radius * self.rot_z / radius_vx_vz_coefficient
            self.log_add("positions", positions)
            self.log_add("radius", radius)
            self.log_add("rot_z", self.rot_z)
            self.log_add("speed_x", self.speed_x)
            # cv2.imwrite(str(self.index) + "_curve_dist.jpg", road_sw_bev)
            # cv2.imwrite(str(self.index) + "_curve_angle.jpg", road_edge_bev)
        elif self.stage == 0 and not self.is_curve:
            self.speed_x = 0
            road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)
        else:   
            road_sw_bev, x_list, y_list = get_sliding_window_result(road_blur_bev, self.init_pos_for_sliding_windows)


        # starting
        if self.stage == 0:
            self.stage = 1
            self.time_since_stage = time.time()
       

        # turning at least certain amount: to ignore post-road
        if self.stage == 1:
            move_robot(self.pub, self.speed_x, self.rot_z, self.is_left)

            if time.time() - self.time_since_stage > self.min_turn_sec:
                self.stage = 2
                self.time_since_stage = time.time()
                # move_robot(self.pub)
        
        # turning while the line is shown: to estimate time to be exact 90 degrees
        if self.stage == 2:
            self.log_add("line_not_shown")

            move_robot(self.pub, self.speed_x, self.rot_z, self.is_left)

            if len(x_list) > 2:
                self.init_pos_for_sliding_windows = x_list[1]
                line_road = Line(x_list, y_list)
                self.line_road = line_road

                self.log = str(self.index) + "_Turn2Road_stage1_line_on_angle_" + str(line_road.get_angle())

                cv2.line(road_sw_bev, (int(self.line_road.calc(0)), 0), (int(self.line_road.calc(np.shape(road_sw_bev)[0])), np.shape(road_sw_bev)[0]), (0, 0, 255), 5)
                
                if len(x_list) > 4 or abs(line_road.get_angle()) < 10:
                    # move_robot(self.pub)
                    self.end = True
 

        if showoff:
            showing_off([frame, road_bev, road_sw_bev, bev])




