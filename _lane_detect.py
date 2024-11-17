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
    returning black and green 

    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rev = 255 - image
    
    lower = np.array([150, 150, 150])
    upper = np.array([255, 255, 255])
    black_mask = cv2.inRange(rev, lower, upper)
    black = cv2.bitwise_and(rev, rev, mask = black_mask)

    # plt.imshow(black)
    green_mask = cv2.inRange(image, (0, 80, 0), (100, 255, 100))
    green = cv2.bitwise_and(image, image, mask = green_mask)
    
    # plt.imshow(green)

    masked = cv2.addWeighted(black, 1, green, 2, 0)

    # plt.imshow(masked)

    
    bev_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # ret, bev_binary = cv2.threshold(bev_gray, 100, 255, cv2.THRESH_BINARY)
    
    
    return bev_gray


def get_sliding_window_result(image, init=-1):
    """
    Sliding window

    """
    h, w = np.shape(image)

    win_h = 0.05
    win_w = 0.5
    win_n = 10
    fill_min = 0.3
    fill_max = 0.7

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
        if fill_min < float(real_sum)/(np.shape(roi)[0]*np.shape(roi)[1]*255) < fill_max:
            lane_point = (x_weigh_sum/real_sum) + l
            x_p = lane_point
            y_p = (t + b) / 2
            x_list.append(x_p)
            y_list.append(y_p)
            cv2.rectangle(window_frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.rectangle(window_frame, (int(x_p), int(y_p)), (int(x_p), int(y_p)), (255, 0, 0), 5)


    return window_frame, x_list, y_list


def get_road_edge_angle(frame, is_left = True):

    segment_count = 10
    minimum = 2

    matrix = np.zeros((5, 5))
    if is_left:
        matrix[1:4, 3:] = 0.2
        matrix[1:4, :2] = -0.2
    else:
        matrix[1:4, 3:] = -0.2
        matrix[1:4, :2] = 0.2
    
    edge_frame = cv2.filter2D(frame, -1, matrix)
    ret, binary_edge = cv2.threshold(edge_frame, 100, 255, cv2.THRESH_BINARY)
    color_frame = cv2.cvtColor(binary_edge, cv2.COLOR_GRAY2BGR)

    angle_list = []
    roi_height = int(np.shape(binary_edge)[0] / segment_count)
    for s in range(segment_count):
        roi = binary_edge[s*roi_height: (s+1)*roi_height]

        x_list, y_list = [], []
        for i in range(np.shape(roi)[0]):
            for j in range(np.shape(roi)[1]):
                if roi[i, j]>0:
                    x_list.append(j)
                    y_list.append(i)
        if len(x_list) > roi_height * 0.5: # depends on the size of the filter...
            line = Line(x_list, y_list)
            angle_list.append(line.get_angle())
            cv2.rectangle(color_frame, (0, s*roi_height), (np.shape(color_frame)[1], (s+1)*roi_height), (0, 255, 0), 1)
        else:
            cv2.rectangle(color_frame, (0, s*roi_height), (np.shape(color_frame)[1], (s+1)*roi_height), (50, 50, 50), 1)

    angle_list = sorted(angle_list)
    print("angle list", angle_list)

    if len(angle_list) > minimum:
        angle_median = angle_list[int((len(angle_list)-1)/2)]
    else:
        angle_median = 180

    x_midpoint = int(np.shape(color_frame)[1] / 2)
    y_height = np.shape(color_frame)[0]
    cv2.line(color_frame, (x_midpoint, y_height), (int(x_midpoint + (math.tan((angle_median/180)*math.pi)*y_height)), 0), (0, 0, 255), 3)
    
    return color_frame, angle_median


def get_rect_blur(frame, size_square_mm = 5):

    filter = np.ones((size_square_mm, size_square_mm), dtype=np.float64) / (size_square_mm*size_square_mm)

    blurred_frame = cv2.filter2D(frame, -1, filter)

    return blurred_frame



def get_green(image):
    """

    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(image, (0, 80, 0), (80, 255, 80))
    green = cv2.bitwise_and(image, image, mask = green_mask)
    
    
    green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    ret, bev_binary = cv2.threshold(green_gray, 30, 255, cv2.THRESH_BINARY)

    return bev_binary



def get_square_pos(green_frame, size_square = 5):
    blurred_frame = get_rect_blur(green_frame, size_square)
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


def get_cross_pos(frame_cm, width_road = 5, left_way = True, right_way = True):

    a = 6
    b = -2
    if left_way and right_way:
        a = 9
        b = -3
    matrix = np.ones((width_road*3, width_road*3))/(a*width_road*width_road)
    matrix[:, :width_road] = 3 / (a*width_road*width_road)
    matrix[:, width_road*2:] = 3 / (a*width_road*width_road)
    if not left_way:
        matrix[:, :width_road] = b / (a*width_road*width_road)
    if not right_way:
        matrix[:, width_road*2:] = b / (a*width_road*width_road)

    matrix[:width_road, :width_road] = b / (a*width_road*width_road)
    matrix[width_road*2:, :width_road] = b / (a*width_road*width_road)
    matrix[:width_road, width_road*2:] = b / (a*width_road*width_road)
    matrix[width_road*2:, width_road*2:] = b / (a*width_road*width_road)
    
    cross_frame = cv2.filter2D(frame_cm, -1, matrix)

    color_frame = cv2.cvtColor(cross_frame, cv2.COLOR_GRAY2BGR)

    max_pos = (0, 0)
    max_pos_xy = (0, 0)

    for i in range(np.shape(cross_frame)[0]):
        for j in range(np.shape(cross_frame)[1]):
            
            if cross_frame[max_pos] < cross_frame[i, j]:
                max_pos = (i, j)
                max_pos_xy = (j, i)

    cv2.rectangle(color_frame, max_pos_xy, max_pos_xy, (0, int(cross_frame[max_pos]), 0), 1)

    return color_frame, max_pos, cross_frame[max_pos]


def get_cm_px_from_mm(frame_mm):
    """
        make image size / 10
    """
    mm_0 = np.shape(frame_mm)[0]
    mm_1 = np.shape(frame_mm)[1]
    cm_0, cm_1 = int(mm_0/10), int(mm_1/10)

    frame_cm = cv2.resize(frame_mm, dsize=(cm_0, cm_1), interpolation=cv2.INTER_LINEAR)
    return frame_cm