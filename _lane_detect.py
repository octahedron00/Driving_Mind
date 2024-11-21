import rospy
import cv2
import numpy as np
import time
import math

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from collections import deque

BOT_FROM_BEV_X = 100 # edit this
BOT_FROM_BEV_Y = 400 # edit this







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
        """
            offset, left side: + / right side: -
        """
        dist = -((self.var_1 * y) - x + self.var_0) / math.sqrt(1 + (self.var_1*self.var_1))

        return dist

    def get_angle(self, is_deg=True):
        """
            angle, left side: - / right side: +
        """
        angle = math.atan(self.var_1)

        if is_deg:
            return -(angle / math.pi) * 180
        return angle





def get_bev(image):
    """
        get image and return bev frame only.
        rt: right top, lt: left top, rd: right down, ld: left down
        Return
        1) _image : BEV result image
        2) minv : inverse matrix of BEV conversion matrix
    """
    lt = (170, 175)
    ld = (0, 412)
    rt = (501, 188)
    rd = (638, 435)

    h = 200
    w = 200

    source = np.float32([[lt], [rt], [ld], [rd]])
    destination = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    M = cv2.getPerspectiveTransform(source, destination)
    Minv = cv2.getPerspectiveTransform(destination, source)
    
    
    warp_image = cv2.warpPerspective(image, M, (h, w), flags=cv2.INTER_LINEAR)

    return warp_image



def get_road(image):
    """
    returning black and green in binary

    """
    # rev = 255 - image

    black_max = 105
    black_min = 0

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    black = cv2.inRange(image_gray, black_min, black_max)
    green = get_green(image)
    
    road_bin = cv2.add(black, green)

    return road_bin



def get_sliding_window_result(image, init=-1):
    """
    Sliding window

    """
    h, w = np.shape(image)

    # pixel means mm here
    win_h = 10
    win_w = 120
    win_n = 10
    fill_min = 0.2
    fill_max = 0.8

    if init > 0:
        lane_point = init
    else:
        lane_point = int(w * 0.5)
    window_frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x_list, y_list = [], []

    for i in range(win_n):
        b = int(h - (i*win_h))
        t = int(h - ((i+1)*win_h))
        l = int(lane_point - (win_w/2))
        r = int(lane_point + (win_w/2))

        if l < 0:
            r -= l
            l = 0
        if r > w:
            l -= (r-w)
            r = w

        cv2.rectangle(window_frame, (l, t), (r, b), (0, 130, 0), 2)

        roi = image[t:b, l:r]

        x_hist = np.sum(roi, axis=0)

        max_pos = np.argmax(x_hist)
        
        x_left = 0
        x_right = 0
        for i in range(max_pos, 0, -1):
            if x_hist[i] < 256:
                x_left = i
                break
        for i in range(max_pos, len(x_hist)):
            if x_hist[i] < 256:
                x_right = i
                break
        

        # print(i, "pos", lane_point, "max", max_pos, "filled area", real_sum, real_sum/(np.shape(roi)[0]*np.shape(roi)[1]*255))
        if fill_min < float(x_right-x_left)/len(x_hist) < fill_max:
            x_p = int((x_left + x_right)/2) + l
            lane_point = x_p
            y_p = (t + b) / 2
            x_list.append(x_p)
            y_list.append(y_p)
            cv2.rectangle(window_frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.rectangle(window_frame, (int(x_p), int(y_p)), (int(x_p), int(y_p)), (255, 0, 0), 5)


    return window_frame, x_list, y_list


def get_road_edge_angle(frame, is_left = True):

    segment_count = 8
    minimum = 2

    matrix = np.zeros((3, 3))
    if is_left:
        matrix[:, 2:] = 0.3
        matrix[:, :1] = -0.3
    else:
        matrix[:, 2:] = -0.3
        matrix[:, :1] = 0.3
    
    edge_frame = cv2.filter2D(frame, -1, matrix)
    ret, binary_edge = cv2.threshold(edge_frame, 100, 255, cv2.THRESH_BINARY)
    color_frame = cv2.cvtColor(binary_edge, cv2.COLOR_GRAY2BGR)


    lines = cv2.HoughLinesP(binary_edge,1,np.pi/180,20,minLineLength=50,maxLineGap=4)

    if lines is None:
        
        angle_median = 180
    else:
        for l in lines:
            xyxy = l[0]
            line = Line([xyxy[0],xyxy[2]], [xyxy[1],xyxy[3]])
            cv2.line(color_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (50, 200, 50), 1)
        
        xyxy = lines[0][0]
        line = Line([xyxy[0],xyxy[2]], [xyxy[1],xyxy[3]])
        angle_median = line.get_angle()
        cv2.line(color_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

    

        x_midpoint = int(np.shape(color_frame)[1] / 2)
        y_height = np.shape(color_frame)[0]
        print(angle_median)
        print(line.get_offset(BOT_FROM_BEV_X, BOT_FROM_BEV_Y))
        cv2.line(color_frame, (x_midpoint, y_height), (int(x_midpoint + (math.tan((angle_median/180)*math.pi)*y_height)), 0), (0, 0, 255), 3)
    
    return color_frame, angle_median



def get_rect_blur(frame, size_square_mm = 5):

    filter = np.ones((size_square_mm, size_square_mm), dtype=np.float64) / (size_square_mm*size_square_mm)

    blurred_frame = cv2.filter2D(frame, -1, filter)

    return blurred_frame



def get_green(image):
    """
        getting green(255) with using hls value filter.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    green = cv2.inRange(hls, (45, 40, 45), (95, 200, 255))
    return green



def get_square_pos(green_frame, size_square = 7):
    """
        getting blurred frame, position of max value by using filter, max_value
    """
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



def get_sliding_window_and_cross_result(image, width_road = 5, left_way = True, right_way = True, init = -1):

    """
    Sliding window

    """
    h, w = np.shape(image)

    win_h = 10
    win_w = 180
    win_n = 8
    fill_min = 0.1
    fill_max = 0.5

    if init > 0:
        lane_point = init
    else:
        lane_point = int(w * 0.5)
    window_frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x_list, y_list = [], []
    x_mid = int(w/2)
    max_position = []
  
    for i in range(win_n):
        b = int(h - (i*win_h))
        t = int(h - ((i+1)*win_h))
        l = int(lane_point - (win_w/2))
        r = int(lane_point + (win_w/2))

        if l < 0:
            r -= l
            l = 0
        if r > w:
            l -= (r-w)
            r = w

        cv2.rectangle(window_frame, (l, t), (r, b), (0, 130, 0), 2)

        roi = image[t:b, l:r]

        x_hist = np.sum(roi, axis=0)

        max_pos = np.argmax(x_hist)
        
        x_left = 0
        x_right = 0
        for i in range(max_pos, 0, -1):
            if x_hist[i] > 256:
                x_left = i
            else:
                break
        for i in range(max_pos, len(x_hist)):
            if x_hist[i] > 256:
                x_right = i
            else:
                break

        # print(i, "pos", lane_point, "max", max_pos, "filled area", real_sum, real_sum/(np.shape(roi)[0]*np.shape(roi)[1]*255))
        if fill_min < float(x_right-x_left)/len(x_hist) < fill_max:
            x_mid = int((x_left + x_right)/2)
            x_p = x_mid + l
            lane_point = x_p
            y_p = (t + b) / 2
            x_list.append(x_p)
            y_list.append(y_p)
            cv2.rectangle(window_frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.rectangle(window_frame, (int(x_p), int(y_p)), (int(x_p), int(y_p)), (255, 0, 0), 3)


        sum_left = np.sum(x_hist[:x_mid])
        sum_right = np.sum(x_hist[x_mid:])
        print(sum_left, sum_right, (np.shape(roi)[0]*np.shape(roi)[1]*255/2))        

        # sum from actual mid of road / sum must be bigger than total half * fill_max
        if left_way and fill_max > float(sum_left) / (np.shape(roi)[0]*np.shape(roi)[1]*255/2):
            continue
        if right_way and fill_max > float(sum_right) / (np.shape(roi)[0]*np.shape(roi)[1]*255/2):
            continue
        cv2.rectangle(window_frame, (l, t), (r, b), (0, 255, 0), 2)
        max_position.append(int((t+b)/2))


    if 1 <= len(max_position) <= 10:
        return window_frame, x_list, y_list, True, max_position
    return window_frame, x_list, y_list, False, max_position




def get_cm_px_from_mm(frame_mm):
    """
        make image size / 10
    """
    mm_0 = np.shape(frame_mm)[0]
    mm_1 = np.shape(frame_mm)[1]
    cm_0, cm_1 = int(mm_0/10), int(mm_1/10)

    frame_cm = cv2.resize(frame_mm, dsize=(cm_0, cm_1), interpolation=cv2.INTER_LINEAR)
    return frame_cm



def get_mm_px_from_cm(frame_cm):
    """
        make image size / 10
    """
    cm_0 = np.shape(frame_mm)[0]
    cm_1 = np.shape(frame_mm)[1]
    mm_0, mm_1 = cm_0*10, cm_1*10

    frame_mm = cv2.resize(frame_cm, dsize=(mm_0, mm_1), interpolation=cv2.INTER_LINEAR)
    return frame_mm


# Not using Code:
# def get_cross_pos_by_filter(frame_cm, width_road = 5, left_way = True, right_way = True):

#     a = 6
#     b = -2
#     if left_way and right_way:
#         a = 9
#         b = -3
#     matrix = np.ones((width_road*3, width_road*3))/float(a*width_road*width_road)
#     matrix[:, :width_road] = 3 / float(a*width_road*width_road)
#     matrix[:, width_road*2:] = 3 / float(a*width_road*width_road)
#     if not left_way:
#         matrix[:, :width_road-1] = b / float(a*width_road*width_road)
#     if not right_way:
#         matrix[:, width_road*2+1:] = b / float(a*width_road*width_road)

#     matrix[:width_road-1, :width_road-1] = b / float(a*width_road*width_road)
#     matrix[width_road*2+1:, :width_road-1] = b / float(a*width_road*width_road)
#     matrix[:width_road-1, width_road*2+1:] = b / float(a*width_road*width_road)
#     matrix[width_road*2+1:, width_road*2+1:] = b / float(a*width_road*width_road)
    
#     cross_frame = cv2.filter2D(frame_cm, -1, matrix)

#     color_frame = cv2.cvtColor(cross_frame, cv2.COLOR_GRAY2BGR)

#     max_pos = (0, 0)
#     max_pos_xy = (0, 0)

#     for i in range(np.shape(cross_frame)[0]):
#         for j in range(np.shape(cross_frame)[1]):
            
#             if cross_frame[max_pos] < cross_frame[i, j]:
#                 max_pos = (i, j)
#                 max_pos_xy = (j, i)

#     cv2.rectangle(color_frame, max_pos_xy, max_pos_xy, (0, int(cross_frame[max_pos]), 0), 1)

#     return color_frame, max_pos, cross_frame[max_pos]

