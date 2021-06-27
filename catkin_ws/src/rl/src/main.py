#!/usr/bin/env python3

from option import Option
from trainer import Trainer
from utils.prioritized_memory import Memory
from utils.logger import Logger
import utils.utils as utils

import time
import os
import sys
import cv2
import time
import numpy as np
import torch
import rospy
import rospkg
import itertools
import wandb
from cv_bridge import CvBridge
from torchvision import transforms
from collections import namedtuple

# srv and msg
from sensor_msgs.msg import Image
from arm_operation.srv import *
from arm_operation.msg import *
from project.srv import *
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest, Empty

# Define transition tuple
Transition = namedtuple('Transition', ['color', 'depth', 'pixel_idx', 'reward', 'next_color', 'next_depth', 'is_empty'])

class setup():
    def __init__(self):

        # rosservice for ur5 endeffector
        self.go_home = rospy.ServiceProxy("/joint_move/home", Trigger)
        self.go_right_tote = rospy.ServiceProxy("/joint_move/right_tote", Trigger)
        self.go_left_tote = rospy.ServiceProxy("/joint_move/left_tote", Trigger)
        self.uvtrans = rospy.ServiceProxy("/uv2xyz", uvTransform)

        # rosservice for gripper
        self.close = rospy.ServiceProxy("/robotiq_finger_control_node/close_gripper", Empty)
        self.open = rospy.ServiceProxy("/robotiq_finger_control_node/open_gripper", Empty)
        self.is_grasped = rospy.ServiceProxy("/robotiq_finger_control_node/get_grasp_state", Trigger)

        self.initial()

        # self.grasp_object(200, 200, 45, True)

        # if self.check_grasped():
        #     print("grasped object")
        # else:
        #     print("trash")

        # self.place_object(True)

    def initial(self):

        req = TriggerRequest()
        _ = self.go_home(req)

        rospy.sleep(0.2)

        self.open()

        rospy.sleep(0.2)

        rospy.loginfo("Already to use")
    
    def grasp_object(self, u, v, angle, is_right):

        req_trans = uvTransformRequest()
        if is_right:
            req_trans.u = u + 60
        else :
            req_trans.u = u + 340
        req_trans.v = v + 128    
        req_trans.angle = angle

        _ = self.uvtrans(req_trans)

        rospy.sleep(0.2)

        self.close()

        rospy.sleep(0.2)

        req = TriggerRequest()
        _ = self.go_right_tote(req) if is_right else self.go_left_tote(req)

        rospy.sleep(0.2)

        _ = self.go_home(req)

        rospy.loginfo("Finish grasp")
        
    def check_grasped(self):

        req = TriggerRequest()
        res = self.is_grasped(req)
        return res.success

    def place_object(self, is_right):

        req = TriggerRequest()
        _ = self.go_left_tote(req) if is_right else self.go_right_tote(req)

        rospy.sleep(0.2)

        self.open()

        rospy.sleep(0.2)

        req = TriggerRequest()
        _ = self.go_home(req)

        rospy.loginfo("Finish placeing")

def shutdown_process(path, gri_mem, regular=True):

    gri_mem.save_memory(path, "gripper_memory.pkl")
    if regular: print("Regular shutdown")
    else: print("Shutdown since user interrupt")
    sys.exit(0)

def sample_data(memory, batch_size):
    done = False
    mini_batch = []; idxs = []; is_weight = []
    while not done:
        success = True
        mini_batch, idxs, is_weight = memory.sample(batch_size)
        for transition in mini_batch:
            success = success and isinstance(transition, Transition)
        if success: done = True
    return mini_batch, idxs, is_weight

if __name__ == '__main__':

    rospy.init_node('rl_train_node', anonymous=True)

    cv_bridge = CvBridge()

    # setup ros
    Setup = setup()

    # paramters
    args = Option().create()

    run = wandb.init(project="DLP_final_project", entity="kuolunwang")
    config = wandb.config

    # trainer
    trainer = Trainer(args, run)

    config.learning_rate = args.learning_rate
    config.epsilon = args.epsilon
    config.buffer_size = args.buffer_size
    config.learning_freq = args.learning_freq
    config.updating_freq = args.updating_freq
    config.mini_batch_size = args.mini_batch_size
    config.densenet_lr = args.densenet_lr
    config.save_every = args.save_every
    config.gripper_memory = args.gripper_memory

    gripper_memory_buffer = Memory(args.buffer_size)

    program_ts = time.time()
    program_time = 0.0

    if args.gripper_memory!="":
        gripper_memory_buffer.load_memory(args.gripper_memory)

    cv2.namedWindow("prediction")
    episode = args.episode
    is_right = True
    sufficient_exp = 0
    learned_times = 0
    loss_list = []

    while True:
        objects = 3
        episode += 1
        program_time += time.time()-program_ts
        cmd = input("\033[1;34m[%f] Reset environment, if ready, press 's' to start. 'e' to exit: \033[0m" %(program_time))
        program_ts = time.time()

        # Logger
        log = Logger(episode)
        log_path, color_path, depth_path, weight_path = log.get_path()

        '''
        workspace in right and left tote
        '''
        if is_right:
            workspace = [120, 220, 100, 350]
        else:
            workspace = [420, 520, 100, 350]

        if cmd == 'E' or cmd == 'e': # End
            shutdown_process(log_path, gripper_memory_buffer, True)
            # cv2.destroyWindow("prediction")

        elif cmd == 'S' or cmd == 's': # Start

            iteration = 0
            result = 0.0
            is_empty = False

            while is_empty is not True:
                print("\033[0;32m[%f] Iteration: %d\033[0m" %(program_time+time.time()-program_ts, iteration))
                if not args.test: 
                    epsilon_ = max(args.epsilon * np.power(0.998, iteration), 0.1) # half after 350 steps

                ts = time.time()

                '''
                need input color and depth
                '''
                if is_right:
                    color = rospy.wait_for_message("/clip_image/color/right", Image)
                    depth = rospy.wait_for_message("/clip_image/depth/right", Image)
                else:
                    color = rospy.wait_for_message("/clip_image/color/left", Image)
                    depth = rospy.wait_for_message("/clip_image/depth/left", Image)

                # size -> (224*224)
                color = cv_bridge.imgmsg_to_cv2(color, "bgr8")
                depth = cv_bridge.imgmsg_to_cv2(depth, "16UC1")

                depth = np.array(depth) / 1000.0
                # cv2.imshow("depth_img.png", depth_array)
                # cv2.imshow("color.png",color)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                grasp_prediction = trainer.forward(color, depth, is_volatile=True)
                print("Forward past: {} seconds".format(time.time()-ts))

                # SELECT ACTION
                if not args.test: # Train
                    action, pixel_index, angle, explore = utils.epsilon_greedy_policy(epsilon_, grasp_prediction)
                else: # Testing
                    action, pixel_index, angle = utils.greedy_policy(grasp_prediction)
                    explore = False
                
                log.write_csv("action_primitive", action)
                log.write_csv("angle", angle)
                log.write_csv("pixel", (pixel_index[1], pixel_index[2]))

                heatmaps, mixed_imgs = log.save_heatmap_and_mixed(grasp_prediction[pixel_index[0]], color, iteration)
                
                del grasp_prediction
                '''
                transform real x,y,z with u,v
                '''
                #real_x, real_y, real_z 

                # utils.print_action(action_str, pixel_index, points[pixel_index[1], pixel_index[2]])
                # print("Take action primitive {} with angle {%d} at ({%d}, {%d}) -> ({%d}, {%d}, {%d})".format(action, angle, pixel_index[1], pixel_index[2], real_x, real_y, real_z ))

                # Save (color heightmap + prediction heatmap + motion primitive and corresponding position), then show it
                visual_img = log.draw_image(mixed_imgs, pixel_index, iteration, explore)
                # cv2.imshow("prediction", cv2.resize(visual_img, None, fx=2, fy=2))
                # cv2.waitKey(0)

                # Check if action valid (is NAN?)
                is_valid = utils.check_if_valid(pixel_index[1:3], workspace, is_right)

                # Visualize in RViz
                # _viz(points[pixel_index[1], pixel_index[2]], action, angle, is_valid)

                # will_collide = None
                if is_valid: # Only take action if valid

                    '''
                    take action
                    '''
                    Setup.grasp_object(pixel_index[2], pixel_index[1], (int)(angle / 3.14 * 180.0) + 90, is_right)

                    # '''
                    # make sure action success
                    # '''
                    action_success = Setup.check_grasped()

                else: # invalid

                    action_success = False
                        
                if action_success: 
                    '''
                    go to next tote and go home
                    '''
                    Setup.place_object(is_right)
                    objects -= 1

                else: 

                    Setup.initial()

                rospy.sleep(0.2) 
               
                # Get next images, and check if workspace is empty

                '''
                need input color and depth
                '''
                if is_right:
                    next_color = rospy.wait_for_message("/clip_image/color/right", Image)
                    next_depth = rospy.wait_for_message("/clip_image/depth/right", Image)
                else:
                    next_color = rospy.wait_for_message("/clip_image/color/left", Image)
                    next_depth = rospy.wait_for_message("/clip_image/depth/left", Image)

                # size -> (320 * 320)
                next_color = cv_bridge.imgmsg_to_cv2(next_color, "bgr8")
                next_depth = cv_bridge.imgmsg_to_cv2(next_depth, "16UC1")

                next_depth = np.array(next_depth) / 1000.0
                
                # check the tote empty?
                if(objects == 0):
                    is_empty == True
                else:
                    is_empty == False

                current_reward = utils.reward_judgement(5, is_valid, action_success)

                log.write_csv("reward", current_reward)
                log.write_csv("valid", is_valid)
                log.write_csv("success", action_success)

                result += current_reward * np.power(0.5, iteration)

                print("\033[1;33mCurrent reward: {} \t Return: {}\033[0m".format(current_reward, result))
                # Store transition to experience buffer
                color_name, depth_name, next_color_name, next_depth_name = utils.wrap_strings(color_path, depth_path, iteration)
                # save color and depth
                cv2.imwrite(color_name, color)
                cv2.imwrite(next_color_name, next_color)
                np.save(depth_name, depth)
                np.save(next_depth_name, next_depth)
                transition = Transition(color_name, depth_name, pixel_index, current_reward, next_color_name, next_depth_name, is_empty)

                gripper_memory_buffer.add(transition)
                print("Gripper_Buffer: {}".format(gripper_memory_buffer.length))
                iteration += 1

                if(is_empty == True):
                    is_right = False
                    wandb.log({"reward": result})
                    wandb.log({"loss mean": np.mean(loss_list)})

                if iteration == args.size_lim:
                    is_empty = True
                    shutdown_process(log_path, gripper_memory_buffer, True)

                if not args.record:
                    ################################TRAIN################################
                    # Start training after buffer has sufficient experiences
                    if gripper_memory_buffer.length   > args.mini_batch_size:
                        sufficient_exp+=1
                        if (sufficient_exp-1) % args.learning_freq == 0:
                            back_ts = time.time()
                            learned_times += 1
                            mini_batch = []
                            idxs = []
                            is_weight = []
                            old_q = []
                            td_target_list = []
                            
                            _mini_batch, _idxs, _is_weight = sample_data(gripper_memory_buffer, args.mini_batch_size)
                            mini_batch += _mini_batch
                            idxs += _idxs
                            is_weight += list(_is_weight)
                            
                            for i in range(len(mini_batch)):
                                color = cv2.imread(mini_batch[i].color)
                                depth = np.load(mini_batch[i].depth)
                                pixel_index = mini_batch[i].pixel_idx
                                next_color = cv2.imread(mini_batch[i].next_color)
                                next_depth = np.load(mini_batch[i].next_depth)

                                rotate_idx = pixel_index[0]
                                old_q.append(trainer.forward(color, depth, False, rotate_idx, clear_grad=True)[0, pixel_index[1], pixel_index[2]])
                                td_target = trainer.get_label_value(mini_batch[i].reward, next_color, next_depth, mini_batch[i].is_empty)
                                td_target_list.append(td_target)
                                loss_ = trainer.backprop(color, depth, pixel_index, td_target, is_weight[i], args.mini_batch_size, i==0, i==len(mini_batch)-1)
                                loss_list.append(loss_)

                            # After parameter updated, update prioirites tree
                            for i in range(len(mini_batch)):

                                color = cv2.imread(mini_batch[i].color)
                                depth = np.load(mini_batch[i].depth)
                                pixel_index = mini_batch[i].pixel_idx
                                next_color = cv2.imread(mini_batch[i].next_color)
                                next_depth = np.load(mini_batch[i].next_depth)
                                td_target = trainer.get_label_value(mini_batch[i].reward, next_color, next_depth, mini_batch[i].is_empty)
                                rotate_idx = pixel_index[0]
                                old_value = trainer.forward(color, depth, False, rotate_idx, clear_grad=True)[0, pixel_index[1], pixel_index[2]]

                                print("New Q value: {:03f} -> {:03f} | TD Target: {:03f}".format(old_q[i], old_value, td_target_list[i]))
                                print("========================================================================================")

                                gripper_memory_buffer.update(idxs[i], td_target-old_value)

                            back_t = time.time()-back_ts

                            print("Backpropagation& Updating: {} seconds \t|\t Avg. {} seconds".format(back_t, back_t/(args.mini_batch_size)))

                            if learned_times % args.updating_freq == 0:
                                print("[%f] Replace target network to behavior network" %(program_time+time.time()-program_ts))
                                trainer.target_net.load_state_dict(trainer.behavior_net.state_dict())
                            if learned_times % args.save_every == 0:
                                model_name = weight_path + "behavior_e{}_i{}.pth".format(episode, iteration)
                                torch.save(trainer.behavior_net.state_dict(), model_name)
                                print("[%f] Model: %s saved" %(program_time+time.time()-program_ts, model_name))