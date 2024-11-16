#!/usr/bin/env python3

import rosbag
import cv2
from cv_bridge import CvBridge
import argparse

def bag_to_mp4(bag_file, output_file, image_topic, fps=30):
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    
    first_image_msg = None
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        first_image_msg = msg
        break
    
    if first_image_msg is None:
        print(f"No messages found on topic '{image_topic}' in the bag file.")
        return
    
    cv_image = bridge.compressed_imgmsg_to_cv2(first_image_msg, desired_encoding="bgr8") if hasattr(first_image_msg, 'format') \
               else bridge.imgmsg_to_cv2(first_image_msg, "bgr8")
    height, width, _ = cv_image.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        if hasattr(msg, 'format'):  
            cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        else:  
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        video_writer.write(cv_image)

    bag.close()
    video_writer.release()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    """
    python3 bag_to_mp4.py <bagfile 이름> <output mp4파일 이름> /main_camera/image_raw/compressed 
    """
    parser = argparse.ArgumentParser(description="Convert a ROS bag file to an MP4 video file.")
    parser.add_argument("bag_file", help="Path to the ROS bag file")
    parser.add_argument("output_file", help="Path to the output MP4 file")
    parser.add_argument("image_topic", help="Image topic in the bag file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    
    args = parser.parse_args()
    bag_to_mp4(args.bag_file, args.output_file, args.image_topic, args.fps)