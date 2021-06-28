#!/usr/bin/env python3

import rospy
import cv2
import tf
import math
import numpy as np
import tf.transformations as tfm
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from project.srv import *

class uv2xyz():
    def __init__(self):
        rospy.loginfo("Wait for Camera Info")
        info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        self.fx = info.P[0]
        self.fy = info.P[5]
        self.cx = info.P[2]
        self.cy = info.P[6]
        print("camera_info: ", self.fx, self.fy, self.cx, self.cy)

        ## ROS service
        rospy.Service("/uv2xyz", uvTransform, self.uvTransform)
        ## ROS client
        self.mani_ee_srv = '/ee_target_pose'
        self.mani_move_srv = rospy.ServiceProxy(self.mani_ee_srv, ee_move)
        self.mani_req = ee_moveRequest()

        ## for opencv
        self.cv_bridge = CvBridge()

        self.listener = tf.TransformListener()

    def uvTransform(self, req):
        rgb_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
        depth_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)

        cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        zc = cv_depth[req.v, req.u]
        zc = float(zc)/1000. # 1000. for D435
        rx, ry, rz = self.getXYZ(req.u/1.0 , req.v/1.0, zc/1.0)

        q = tf.transformations.quaternion_from_euler( -90.0 * math.pi/180, 90.0 * math.pi/180, req.angle * math.pi/180, axes="rxzx")

        t = [rx, ry, rz]
        # print(t)
        pose = self.transform_pose_to_base_link(t, q)
        # print(pose)

        self.mani_req.target_pose.position.x = pose[0]
        self.mani_req.target_pose.position.y = pose[1]
        self.mani_req.target_pose.position.z = pose[2] + 0.2
        self.mani_req.target_pose.orientation.x = q[0]
        self.mani_req.target_pose.orientation.y = q[1]
        self.mani_req.target_pose.orientation.z = q[2]
        self.mani_req.target_pose.orientation.w = q[3]

        res = uvTransformResponse()

        try:
            mani_resp = self.mani_move_srv(self.mani_req)
            rospy.sleep(0.1)
            self.mani_req.target_pose.position.z = pose[2] - 0.05
            mani_resp = self.mani_move_srv(self.mani_req)
            res.result = "success"
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.result = "fail"
            print("Service call failed: %s"%e)
            
        return res

    def getXYZ(self, x, y, zc):
        x = float(x)
        y = float(y)
        zc = float(zc)
        inv_fx = 1.0 / self.fx
        inv_fy = 1.0 / self.fy
        x = (x - self.cx) * zc * inv_fx
        y = (y - self.cy) * zc * inv_fy
        return x, y, zc

    def transform_pose_to_base_link(self, t, q):
        euler_pose = tfm.euler_from_quaternion(q)
        tf_cam_col_opt_fram = tfm.compose_matrix(translate=t, angles=euler_pose)
        trans, quat = self.listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
        euler = tfm.euler_from_quaternion(quat)
        tf = tfm.compose_matrix(translate=trans, angles=euler)
        t_pose = np.dot(tf, tf_cam_col_opt_fram)
        t_ba_li = t_pose[0:3, 3]
        return t_ba_li

    def angle2radius(self, angele):
        radius = angle * math.pi/180.0
        return radius

if __name__ == '__main__':
    rospy.init_node('uv2xyz', anonymous=True)
    ic = uv2xyz()
    rospy.spin()