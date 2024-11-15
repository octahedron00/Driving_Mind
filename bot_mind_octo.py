#! /usr/bin/env python
# -*- coding: utf-8 -*-

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



true_green_confidence = 100
true_green_dist_from_road = 20 #mm

bot_from_bev_x = 100
bot_from_bev_y = 400



class Line:
    var_1 = 0
    var_0 = 0
    calc = None
    from_y_to_x = True

    def __init__(self, x_list, y_list, from_y_to_x=True):

        if len(x_list) == 0:
            pass

        variables = np.polyfit(y_list, x_list, deg=1)

        self.var_1 = variables[0]
        self.var_0 = variables[1]

        self.calc = np.poly1d(variables)
        self.from_y_to_x = from_y_to_x
    

    def get_distance(self, x, y):
        
        dist = ((self.var_1 * y) - x + self.var_0) / math.sqrt(1 + (self.var_1*self.var_1))
        dist = abs(dist)

        return dist

    def get_offset(self, x, y):
        
        dist = -((self.var_1 * y) - x + self.var_0) / math.sqrt(1 + (self.var_1*self.var_1))

        return dist

    def get_angle(self, is_deg=True):

        angle = math.atan(self.var_1)

        if is_deg:
            return -(angle / math.pi) * 180
        return angle



def get_bev(image):
    """
        Image, 그리고 position 4개 > height와 width, mm 단위로 나타내기. 
        rt: right top, lt: left top, rd: right down, ld: left down


        Return
        1) _image : BEV result image
        2) minv : inverse matrix of BEV conversion matrix
    """
    lt = (147, 104)
    ld = (0, 343)
    rt = (487, 113)
    rd = (636, 355)

    h = 200
    w = 200

    source = np.float32([[lt], [rt], [ld], [rd]])
    destination = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    M = cv2.getPerspectiveTransform(source, destination)
    Minv = cv2.getPerspectiveTransform(destination, source)
    
    
    warp_image = cv2.warpPerspective(image, M, (h, w), flags=cv2.INTER_LINEAR)

    return warp_image, Minv



def get_road(image):
    """
    returning black and green 색상 부분만 뽑아 이미지로 만들어 줌.
    다만, 주변의 녹색을 잘라내야 하니까... 음... 

    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rev = 255 - image
    
    lower = np.array([150, 150, 150])
    upper = np.array([255, 255, 255])
    black_mask = cv2.inRange(rev, lower, upper)
    black = cv2.bitwise_and(rev, rev, mask = black_mask)

    # plt.imshow(black)
    green_mask = cv2.inRange(hsv, (50, 150, 0), (80, 255, 255))
    green = cv2.bitwise_and(image, image, mask = green_mask)
    
    # plt.imshow(green)

    masked = cv2.addWeighted(black, 1, green, 1, 0)

    # plt.imshow(masked)

    
    bev_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # ret, bev_binary = cv2.threshold(bev_gray, 100, 255, cv2.THRESH_BINARY)
    
    
    return bev_gray



def get_green(image):
    """
    green 색상 부분만 뽑아 이미지로 만들어 줌.
    주변의 녹색이 문제가 되지만, 일단 filter 해놓고 생각하기.

    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(image, (0, 80, 0), (80, 255, 80))
    green = cv2.bitwise_and(image, image, mask = green_mask)
    
    
    green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    ret, bev_binary = cv2.threshold(green_gray, 30, 255, cv2.THRESH_BINARY)

    return bev_binary



def get_sliding_window_result(image, init=-1):
    """
    Sliding window를 자동으로 돌려주기.
    기존 이미지에 대한 비율로 설정: 높이, 너비, 개수(맨 아래부터)

    얼마나의 픽셀이 채워졌는지에 따라 계산에 넣을지 안 넣을지를 생각함.

    """
    h, w = np.shape(image)

    win_h = 0.05
    win_w = 0.5
    win_n = 10
    fill_min = 0.05
    fill_max = 0.6

    if init > 0:
        lane_point = init
    else:
        lane_point = int(w * 0.5)
    window_frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x_list, y_list = [], []

    for i in range(win_n):
        b = int(h*(1 - (i*win_h)))
        t = int(h*(1 - ((i+1)*win_h)))
        l = int(lane_point - (w*win_w/2))
        r = int(lane_point + (w*win_w/2))

        if l < 0:
            r -= l
            l = 0
        if r > w:
            l -= (r-w)
            r = w

        cv2.rectangle(window_frame, (l, t), (r, b), (0, 130, 0), 2)

        roi = image[t:b, l:r]

        x_hist = np.sum(roi, axis=0)

        x_hist_calib = np.zeros(np.shape(x_hist), dtype=np.int32)
        for i in range(np.shape(x_hist)[0]):
            x_hist_calib[i] = x_hist[i] * (2 - (abs((np.shape(x_hist)[0]/2)-i) / (np.shape(x_hist)[0]/4)) )

        x_hist_around_max = np.zeros(np.shape(x_hist), dtype=np.int32)

        max_pos = 1
        for i in range(np.shape(x_hist)[0]):
            if x_hist[max_pos] < x_hist[i]:
                max_pos = i
        
        try:
            x_hist_around_max[max_pos] = x_hist[max_pos]
        except:
            print(l, r, t, b, roi, x_hist_calib, max_pos)
        # print(max_pos)

        for i in range(max_pos+1, np.shape(x_hist)[0]):
            # print(x_hist_around_max, x_hist)
            x_hist_around_max[i] = min(x_hist_around_max[i-1], x_hist[i])

        for i in range(max_pos-1, -1, -1):
            x_hist_around_max[i] = min(x_hist_around_max[i+1], x_hist[i])

        # print(x_hist_around_max)
        x_weigh_sum = 0
        for i in range(np.shape(x_hist)[0]):
            x_weigh_sum += i * x_hist_around_max[i]

        real_sum = sum(x_hist_around_max)

        # print(i, "pos", lane_point, "max", max_pos, "filled area", real_sum, real_sum/(np.shape(roi)[0]*np.shape(roi)[1]*255))
        if fill_min < real_sum/(np.shape(roi)[0]*np.shape(roi)[1]*255) < fill_max:
            lane_point = (x_weigh_sum/real_sum) + l
            x_p = lane_point
            y_p = (t + b) / 2
            x_list.append(x_p)
            y_list.append(y_p)
            cv2.rectangle(window_frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.rectangle(window_frame, (int(x_p), int(y_p)), (int(x_p), int(y_p)), (255, 0, 0), 5)


    return window_frame, x_list, y_list



def get_square_pos(green_frame):

    size_square_mm = 31 # odd

    filter = np.ones((size_square_mm, size_square_mm), dtype=np.float64) / (size_square_mm*size_square_mm)

    blurred_frame = cv2.filter2D(green_frame, -1, filter)

    color_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_GRAY2BGR)

    max_pos = (0, 0)
    max_pos_xy = (0, 0)

    for i in range(np.shape(blurred_frame)[0]):
        for j in range(np.shape(blurred_frame)[1]):
            
            if blurred_frame[max_pos] < blurred_frame[i, j]:
                max_pos = (i, j)
                max_pos_xy = (j, i)

    cv2.rectangle(color_frame, max_pos_xy, max_pos_xy, (0, int(blurred_frame[max_pos]), 0), 5)

    return color_frame, max_pos, blurred_frame[max_pos]



class bot_mind:

    GO_STRAIGHT = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    MISSION = 4
    # GO_CURVE = 5
    WELL_DONE = -1

    LIST = [1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 4, 3, 1, -1]

    stage = 0
    state = 0
    green_encounter = 0
    line_road = None
    init_pos_for_sliding_windows = -1

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('lane_detection_node', anonymous=False)
        rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        self.stage = 0
        self.state = 0
        self.green_encounter = 0
        self.line_road = None

    def camera_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.action()


    def action(self):

        frame = self.image

        bev_frame, Minv = get_bev(frame)

        filter_frame = get_road(bev_frame)

        green_frame = get_green(bev_frame)

        window_frame, x_list, y_list = get_sliding_window_result(filter_frame, self.init_pos_for_sliding_windows)

        if len(x_list) > 3:
            self.init_pos_for_sliding_windows = x_list[0]
            self.line_road = Line(x_list, y_list)

            print("x = ", self.line_road.var_1, " y + ", self.line_road.var_0)
            print("Dist from origin: ", self.line_road.get_distance(bot_from_bev_x,bot_from_bev_y), ", angle: ", self.line_road.get_angle())
        else:
            self.init_pos_for_sliding_windows = -1
        
        if self.line_road == None:
            return

        green_position_frame, green_pos, green_max = get_square_pos(green_frame)

        if green_max > true_green_confidence and self.line_road.get_distance(green_pos[1], green_pos[0]) < true_green_dist_from_road:
            print("What, the true Green!!!", green_max, self.line_road.get_distance(green_pos[1], green_pos[0]))
            self.green_encounter += 1
        else:
            self.green_encounter -= 1
            self.green_encounter = max(int(self.green_encounter*0.8), self.green_encounter)


        if green_encounter >= 5:
            green_encounter = -10

            self.stage += 1
            self.state = self.LIST[self.stage]


        if self.state == self.GO_STRAIGHT:
            self.go_stanley()
        elif self.state == self.TURN_LEFT:
            self.turn_left()
            self.stage += 1
            self.state = self.LIST[self.stage]
        elif self.state == self.TURN_RIGHT:
            self.turn_right()
            self.stage += 1
            self.state = self.LIST[self.stage]
        elif self.state == self.MISSION:
            print("Try a mission?")
            self.stage += 1
            self.state = self.LIST[self.stage]

        

        cv2.namedWindow('ori')
        cv2.moveWindow('ori', 0, 0)
        cv2.imshow('ori', frame)

        cv2.namedWindow('bev')
        cv2.moveWindow('bev', 700, 0)
        cv2.imshow('bev', bev_frame)

        cv2.namedWindow('filt')
        cv2.moveWindow('filt', 1400, 0)
        cv2.imshow('filt', filter_frame)

        cv2.namedWindow('green')
        cv2.moveWindow('green', 1400, 300)
        cv2.imshow('green', green_frame)
        
        cv2.namedWindow('green_blur')
        cv2.moveWindow('green_blur', 1400, 600)
        cv2.imshow('green_blur', green_position_frame)

        cv2.line(window_frame, (int(line_road.calc(0)), 0), (int(line_road.calc(np.shape(window_frame)[0])), np.shape(window_frame)[0]), (0, 0, 255), 5)
        cv2.namedWindow('window')
        cv2.moveWindow('window', 0, 600)
        cv2.imshow('window', window_frame)
        
        cv2.waitKey(1)


    def go_stanley(self):
        offset_mm = self.line_road.get_offset(bot_from_bev_x,bot_from_bev_y)
        angle_deg = self.line_road.get_angle()

        kp= 0.001
        k = 0.01
        x = 0.15

        print(offset_mm, angle_deg)
        speed = Twist()
        speed.linear.x = x
        speed.angular.z = (angle_deg + atan(kp*offset_mm)) * k
        self.pub.publish(speed)
        print(speed.angular.z)


    def turn_left(self):
        print("turning left...")

        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 0
        self.pub.publish(speed)
        time.sleep(3)

        speed = Twist()
        speed.linear.x = 1.0
        speed.angular.z = 0
        self.pub.publish(speed)
        time.sleep(0.3)
        speed = Twist()
        speed.linear.x = -1.0
        speed.angular.z = 0
        self.pub.publish(speed)
        time.sleep(0.3)

        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = -1
        self.pub.publish(speed)
        time.sleep(3)


    def turn_right(self):
        print("turning right...")

        speed = Twist()
        speed.linear.x = 0
        speed.angular.z = 1
        self.pub.publish(speed)
        time.sleep(3)



if __name__ == "__main__":

    if not rospy.is_shutdown():
        bot_mind()
        rospy.spin()
