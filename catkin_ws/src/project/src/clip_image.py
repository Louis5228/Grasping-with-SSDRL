#!/usr/bin/env python3

import rospy
import cv2
import message_filters
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

class clip_image:
    def __init__(self):
        self.cv_bridge = CvBridge()

        ## Subscriber
        image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)

        ## Publisher for color images
        self.color_pub_r = rospy.Publisher("/clip_image/color/right", Image, queue_size=10)
        self.color_pub_l = rospy.Publisher("/clip_image/color/left", Image, queue_size=10)
        ## Publisher for depth images
        self.depth_pub_r = rospy.Publisher("/clip_image/depth/right", Image, queue_size=10)
        self.depth_pub_l = rospy.Publisher("/clip_image/depth/left", Image, queue_size=10)

    def rgb_callback(self, data):
        cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")

        ## clip rgb image
        color_right_image = cv_image[90:410, :320, :]
        color_left_image = cv_image[90:410, 320:, :]

        self.color_pub_l.publish(self.cv_bridge.cv2_to_imgmsg(color_left_image, "bgr8"))
        self.color_pub_r.publish(self.cv_bridge.cv2_to_imgmsg(color_right_image, "bgr8"))

    def depth_callback(self, data):
        cv_depth = self.cv_bridge.imgmsg_to_cv2(data, "16UC1")

        ## clip depth image
        d_left_image = cv_depth[90:410, :320]
        d_right_image = cv_depth[90:410, 320:]

        self.depth_pub_r.publish(self.cv_bridge.cv2_to_imgmsg(d_left_image, "16UC1"))
        self.depth_pub_l.publish(self.cv_bridge.cv2_to_imgmsg(d_right_image, "16UC1"))

if __name__ == '__main__':
    rospy.init_node('clip_image', anonymous=True)
    ic = clip_image()
    rospy.spin()