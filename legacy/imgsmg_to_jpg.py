#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver:
    def __init__(self):
        rospy.init_node("image_saver", anonymous=True)

        self.image_topic = "/main_camera/image_raw/compressed"
        self.save_directory = "./images" # check save img directory
        
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.bridge = CvBridge()
        self.image_count = 0

        rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback)
        rospy.loginfo(f"Subscribed to {self.image_topic}, saving images to {self.save_directory}")

    def image_callback(self, img_msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        
        filename = os.path.join(self.save_directory, f"image_{self.image_count:04d}.jpg")
        
        cv2.imwrite(filename, cv_image)
        rospy.loginfo(f"Saved {filename}")
        cv2.imshow("currImage",cv_image)
        cv2.waitKey(1)

        self.image_count += 1

if __name__ == "__main__":
    try:
        ImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
