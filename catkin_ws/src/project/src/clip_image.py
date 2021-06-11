#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class clip_image:
    def __init__(self):
        self.bridge = CvBridge()
        ## Subscriber
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        ## Publisher
        self.image_pub_r = rospy.Publisher("/clip_image/right", Image, queue_size=10)
        self.image_pub_l = rospy.Publisher("/clip_image/left", Image, queue_size=10)

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        ## clip image
        left_image = cv_image[:, :320, :]
        right_image = cv_image[:, 320:, :]

        self.image_pub_l.publish(self.bridge.cv2_to_imgmsg(left_image, "bgr8"))
        self.image_pub_r.publish(self.bridge.cv2_to_imgmsg(right_image, "bgr8"))

        ## Visualize images
        # cv2.imshow("left_Image window", left_image)
        # cv2.imshow("right_Image window", right_image)
        # cv2.waitKey(3)

if __name__ == '__main__':
    rospy.init_node('clip_image', anonymous=True)
    ic = clip_image()
    rospy.spin()