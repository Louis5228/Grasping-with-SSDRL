#!/usr/bin/env python3

import rospy
from arm_operation.srv import *
from arm_operation.msg import *
from arm_bringup.srv import *
from std_srvs.srv import Trigger, TriggerResponse

class joint_move():
    def __init__(self):
        rospy.Service("/joint_move/home", Trigger, self.ur5_home)
        rospy.Service("/joint_move/right_tote", Trigger, self.ur5_right_tote)
        rospy.Service("/joint_move/left_tote", Trigger, self.ur5_left_tote)
        self.mani_joint_srv = '/ur5_control_server/ur_control/goto_joint_pose'
        self.mani_move_srv = rospy.ServiceProxy(self.mani_joint_srv, joint_pose)
        self.mani_req = joint_poseRequest()
        self.p = joint_value()

    def ur5_home(self, req):
        self.p.joint_value = [-3.125488344823019, -0.7493508497821253, -2.38697320619692, -1.5348437468158167, 1.5634725093841553, -1.5657637755023401]
        self.mani_req.joints.append(self.p)
        res = TriggerResponse()

        try:
            rospy.wait_for_service(self.mani_joint_srv)
            mani_resp = self.mani_move_srv(self.mani_req)
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

    def ur5_right_tote(self, req):
        self.p.joint_value = [-3.31625205675234, -1.6230791250811976, -1.8041918913470667, -1.243310276662008, 1.5713356733322144, -1.7564342657672327]
        self.mani_req.joints.append(self.p)
        res = TriggerResponse()

        try:
            rospy.wait_for_service(self.mani_joint_srv)
            mani_resp = self.mani_move_srv(self.mani_req)
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)

        return res

    def ur5_left_tote(self, req):
        self.p.joint_value = [-2.5716171900378626, -1.6171773115741175, -1.8077481428729456, -1.2562854925738733, 1.5428555011749268, -1.012155834828512]
        self.mani_req.joints.append(self.p)
        res = TriggerResponse()

        try:
            rospy.wait_for_service(self.mani_joint_srv)
            mani_resp = self.mani_move_srv(self.mani_req)
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)

        return res

class endeffector_move():
    def __init__(self):
        rospy.Service("/ee_target_pose", ee_move, self.goto_pose)
        self.mani_ee_srv = '/ur5_control_server/ur_control/goto_pose'
        self.mani_move_srv = rospy.ServiceProxy(self.mani_ee_srv, target_pose)
        self.mani_req = target_poseRequest()
        self.factor = 0.75

    def goto_pose(self, req):
        self.mani_req.target_pose.position.x = req.target_pose.position.x
        self.mani_req.target_pose.position.y = req.target_pose.position.y
        self.mani_req.target_pose.position.z = req.target_pose.position.z
        self.mani_req.target_pose.orientation.x = req.target_pose.orientation.x
        self.mani_req.target_pose.orientation.y = req.target_pose.orientation.y
        self.mani_req.target_pose.orientation.z = req.target_pose.orientation.z
        self.mani_req.target_pose.orientation.w = req.target_pose.orientation.w
        self.mani_req.factor = self.factor

        res = ee_moveResponse()

        try:
            rospy.wait_for_service(self.mani_ee_srv)
            mani_resp = self.mani_move_srv(self.mani_req)
            res.result = "success"
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.result = "fail"
            print("Service call failed: %s"%e)

        return res

if __name__ == "__main__":
    rospy.init_node('ur5_specific_position', anonymous=True)
    move = joint_move()
    ee = endeffector_move()
    rospy.spin()