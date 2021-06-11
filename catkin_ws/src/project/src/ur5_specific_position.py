#!/usr/bin/env python3
import rospy
from arm_operation.srv import *
from arm_operation.msg import *

class joint_move():
    def __init__(self):
        self.mani_joint_srv = '/ur5_control_server/ur_control/goto_joint_pose'
        self.mani_move_srv = rospy.ServiceProxy(self.mani_joint_srv, joint_pose)
        self.mani_req = joint_poseRequest()
        self.p = joint_value()

    def ur5_home(self):
        self.p.joint_value = [-3.112878386174337, -0.9103191534625452, -2.285224262868063, -1.4964826742755335, 1.5014410018920898, -1.3952811400042933]
        self.mani_req.joints.append(self.p)

        try:
            rospy.wait_for_service(self.mani_joint_srv)
            mani_resp = self.mani_move_srv(self.mani_req)
        except (rospy.ServiceException, rospy.ROSException) as e:
            print("Service call failed: %s"%e)

    def ur5_right_tote(self):
        self.p.joint_value = [-3.309857432042257, -1.765533749257223, -2.0725868383990687, -0.8151291052447718, 1.5616744756698608, -1.6369159857379358]
        self.mani_req.joints.append(self.p)

        try:
            rospy.wait_for_service(self.mani_joint_srv)
            mani_resp = self.mani_move_srv(self.mani_req)
        except (rospy.ServiceException, rospy.ROSException) as e:
            print("Service call failed: %s"%e)

    def ur5_left_tote(self):
        self.p.joint_value = [-2.580636803303854, -1.7984502951251429, -2.023400608693258, -0.8525250593768519, 1.5245158672332764, -0.9081133047686976]
        self.mani_req.joints.append(self.p)

        try:
            rospy.wait_for_service(self.mani_joint_srv)
            mani_resp = self.mani_move_srv(self.mani_req)
        except (rospy.ServiceException, rospy.ROSException) as e:
            print("Service call failed: %s"%e)


if __name__ == "__main__":
    move = joint_move()
    move.ur5_home()
    move.ur5_right_tote()
    move.ur5_left_tote()
    move.ur5_home()