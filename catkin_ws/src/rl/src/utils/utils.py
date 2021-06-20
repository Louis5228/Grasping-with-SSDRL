#!/usr/bin/env python3

import numpy as np
import cv2

def wrap_strings(image_path, depth_path, iteration):
	color_name = image_path + "color_{:06}.jpg".format(iteration)
	next_color_name = image_path + "next_color_{:06}.jpg".format(iteration)
	depth_name = depth_path + "depth_data_{:06}.npy".format(iteration)
	next_depth_name = depth_path + "next_depth_data_{:06}.npy".format(iteration) # for saving space of hard disk
	return color_name, depth_name, next_color_name, next_depth_name

def check_if_valid(position, workspace):
    if (position[0] > workspace[0] and position[0] < workspace[1]) and \
       (position[1] > workspace[2] and position[1] < workspace[3]) and \
       (position[2] > workspace[4] and position[2] < workspace[5]):
       return True # valid
    else: 
        return False # invalid

def reward_judgement(reward, is_valid, action_success):
    if not action_valid:
        return -3*reward_unit # Invalid
    if action_success:
        return reward_unit # Valid and success
    else:
        return -reward_unit # Valid and failed

# Choose action using epsilon-greedy policy
def epsilon_greedy_policy(epsilon, grasp_prediction):
    explore = np.random.uniform() < epsilon # explore with probability epsilon
    action = 0
    angle = 0
    pixel_index = [] # primitive index, y, x
    out_str = ""
    # print(np.max(grasp_prediction[0]), np.max(grasp_prediction[1]), np.max(grasp_prediction[2]), np.max(grasp_prediction[3]))

    if not explore: # Choose max Q
        out_str += "|Exploit| "

        primitives_max = np.max(grasp_prediction)
        max_q_index = np.where(primitives_max==np.max(primitives_max))[0][0]
       
        tmp = np.where(grasp_prediction == np.max(grasp_prediction))
        pixel_index = [tmp[0][0], tmp[1][0], tmp[2][0]]
        angle = -90.0+45.0*tmp[0][0]
        angle = np.radians(angle)
        action = tmp[0][0]
        out_str += "Select grasp with angle {} at ({}, {}) with Q value {:.3f}\n".format(angle, pixel_index[1], pixel_index[2], grasp_prediction[action, pixel_index[1], pixel_index[2]])
    
    else: # Random 
        out_str += "|Explore| "

        w, h = grasp_prediction[0].shape[1:]

        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        primitive = np.random.randint(0, 4) # grasp_-90, grasp_-45, grasp_0, grasp_45
 
        angle = np.radians(-90.0 + 45.0 * primitive)
        angle_deg = -90.0 + 45.0 * primitive

        action = primitive
        out_str += "Select grasp with angle {} at ({}, {}) with Q value {:.3f}\n".format(angle, x, y, grasp_prediction[primitive, x, y])

        pixel_index = [primitive, x, y]
    print(out_str)
    return action, pixel_index, angle

# Choose action using greedy policy
def greedy_policy(grasp_prediction):

    # print np.max(suck_1_prediction)*suck_1_scale, np.max(suck_2_prediction)*suck_2_scale, np.max(grasp_prediction[0])*grasp_scale, np.max(grasp_prediction[1])*grasp_scale, np.max(grasp_prediction[2])*grasp_scale, np.max(grasp_prediction[3])*grasp_scale
    action = 0
    angle = 0
    pixel_index = [] # rotate_idx, y, x

    primitives_max = np.max(grasp_prediction)
    max_q_index = np.where(primitives_max==np.max(primitives_max))[0][0]
   
    tmp = np.where(grasp_prediction == np.max(grasp_prediction))
    pixel_index = [tmp[0][0], tmp[1][0], tmp[2][0]]
    angle = -90.0+45.0*tmp[0][0]
    angle = np.radians(angle)
    action = tmp[0][0]

    return action, pixel_index, angle