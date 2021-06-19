#!/usr/bin/env python3

from option import Option
from trainer import Trainer
from utils.prioritized_memory import Memory
from utils.logger import Logger
import utils

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

class setup():
    def __init__(self):

        # rosservice and rostopic

        # go home

def shutdown_process(gri_mem, regular=True):

    gri_mem.save_memory(path, "gripper_memory.pkl")
    if regular: print "Regular shutdown"
    else: print "Shutdown since user interrupt"
    sys.exit(0)

if __name__ == '__main__':

    # setup ros
    Setup = setup()

    # paramters
    args = Option().create()

    # Logger
    log = Logger()
    log_path, color_path, depth_path = log.get_path()

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

    # Define transition tuple
    Transition = namedtuple('Transition', ['color', 'depth', 'pixel_idx', 'reward', 'next_color', 'next_depth', 'is_empty'])

    gripper_memory_buffer = Memory(arg.buffer_size)

    program_ts = time.time()
    program_time = 0.0

    if gripper_memory!="":
        gripper_memory_buffer.load_memory(gripper_memory)

    cv2.nemedwindow("prediction")
    is_right = True
    episode = 0
    sufficient_exp = 0
    learned_times = 0
    loss_list = []

    while True:
        objects = 5
        episode += 1
        program_time += time.time()-program_ts
        cmd = raw_input("\033[1;34m[%f] Reset environment, if ready, press 's' to start. 'e' to exit: \033[0m" %(program_time))
        program_ts = time.time()

        '''
        workspace in right and left tote
        '''
        # if is_right:
        #     workspace = 
        # else
        #     workspace = 

        if cmd == 'E' or cmd == 'e': # End
            shutdown_process(package_path, gripper_memory_buffer, True)
            cv2.destroyWindow("prediction")

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
                # if(is_right == True):
                #     color = 
                #     depth = 
                # else:
                #     color = 
                #     depth = 

                grasp_prediction = trainer.forward(color, depth, is_volatile=True)
                print("Forward past: {} seconds".format(time.time()-ts))

                heatmaps, mixed_imgs = log.save_heatmap_and_mixed(grasp_prediction, color, iteration, episode)

                # SELECT ACTION
                if not args.test: # Train
                    action, pixel_index, angle = utils.epsilon_greedy_policy(epsilon_, grasp_prediction)
                else: # Testing
                    action, pixel_index, angle = utils.greedy_policy(grasp_prediction)
                
                del grasp_prediction

                log.write_csv("action_primitive", "action_primitive", action)
                log.write_csv("angle", "angle", angle)
                log.write_csv("pixel", "pixel", (pixel_index[1], pixel_index[2]))

                '''
                transform real x,y,z with u,v
                '''
                #real_x, real_y, real_z 

                # utils.print_action(action_str, pixel_index, points[pixel_index[1], pixel_index[2]])
                print("Take action primitive {} with angle {%d} at ({%d}, {%d}) -> ({%d}, {%d}, {%d})".format(action, angle, pixel_index[1], pixel_index[2], real_x, real_y, real_z ))

                # Save (color heightmap + prediction heatmap + motion primitive and corresponding position), then show it
                visual_img = log.draw_image(mixed_imgs[pixel_index[0]], pixel_index, episode, iteration)
                cv2.imshow("prediction", cv2.resize(visual_img, None, fx=2, fy=2))
                cv2.waitKey(33)

                # Check if action valid (is NAN?)
                is_valid = utils.check_if_valid(points[pixel_index[1], pixel_index[2]], workspace)

                # Visualize in RViz
                # _viz(points[pixel_index[1], pixel_index[2]], action, angle, is_valid)

                # will_collide = None
                if is_valid: # Only take action if valid

                    '''
                    take action
                    '''

                    '''
                    make sure action success
                    '''

                else: # invalid

                    action_success = False
                        
                if action_success: 
                    '''
                    go to next tote and go home
                    '''
                    if(is_right == True):
                        # go to left tote
                    else:
                        # go to rigth tote
                    go_home()
                    objects -= 1

                else: 
    
                    go_home()

                time.sleep(1.0) 
               
                # Get next images, and check if workspace is empty

                '''
                need input color and depth
                '''
                if(is_right == True):
                    next_color = 
                    next_depth = 
                else:
                    next_color = 
                    next_depth = 
                
                # check the tote empty?
                # is_empty = _check_if_empty(next_pc.pc)
                if(objects == 0):
                    is_empty == True
                else:
                    is_empty == False

                current_reward = utils.reward_judgement(reward, is_valid, action_success)

                log.write_csv("reward", "reward", current_reward)
                log.write_csv("valid", "valid", is_valid)
                log.write_csv("success", "success", action_success)

                result += current_reward * np.power(0.5, iteration)

                print("\033[1;33mCurrent reward: {} \t Return: {}\033[0m".format(current_reward, result))
                # Store transition to experience buffer
                color_name, depth_name, next_color_name, next_depth_name = utils.wrap_strings(color_path, depth_path, iteration)
                transition = Transition(color_name, depth_name, pixel_index, current_reward, next_color_name, next_depth_name, is_empty)

                gripper_memory_buffer.add(transition)
                print("Gripper_Buffer: {}".format(gripper_memory_buffer.length))
                iteration += 1

                if(is_empty == True):
                    is_right = False
                    wandb.log({"reward": result})
                    wandb.log({"loss mean": np.mean(loss_list)})


                ################################TRAIN################################
                # Start training after buffer has sufficient experiences
                if gripper_memory_buffer.length   > args.mini_batch_size:
                    sufficient_exp+=1
                    if (sufficient_exp-1) % args.learning_freq == 0:
                        back_ts = time.time(); 
                        learned_times += 1
                        mini_batch = []
                        idxs = []
                        is_weight = []
                        old_q = []
                        td_target_list = []
                        
                        _mini_batch, _idxs, _is_weight = utils.sample_data(gripper_memory_buffer, args.mini_batch_size)
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
                            td_target = trainer.get_label_value(mini_batch[i].reward, next_color, next_depth, mini_batch[i].is_empty, pixel_index[0])
                            td_target_list.append(td_target)
                            loss_ = trainer.backprop(color, depth, pixel_index, td_target, is_weight[i], mini_batch_size, i==0, i==len(mini_batch)-1)
                            loss_list.append(loss_)
                            

                        # After parameter updated, update prioirites tree
                        for i in range(len(mini_batch)):

                            color = cv2.imread(mini_batch[i].color)
                            depth = np.load(mini_batch[i].depth)
                            pixel_index = mini_batch[i].pixel_idx
                            next_color = cv2.imread(mini_batch[i].next_color)
                            next_depth = np.load(mini_batch[i].next_depth)
                            td_target = trainer.get_label_value(mini_batch[i].reward, next_color, next_depth, mini_batch[i].is_empty, pixel_index[0])
                            rotate_idx = pixel_index[0]
                            old_value = trainer.forward(color, depth, False, rotate_idx, clear_grad=True)[0, pixel_index[1], pixel_index[2]]

                            print("New Q value: {:03f} -> {:03f} | TD Target: {:03f}".format(old_q[i], old_value, td_target_list[i]))
                            print("========================================================================================")

                            gripper_memory_buffer.update(idxs[i], td_target-old_value)

                        back_t = time.time()-back_ts


                        print("Backpropagation& Updating: {} seconds \t|\t Avg. {} seconds".format(back_t, back_t/(mini_batch_size)))

                        if learned_times % args.updating_freq == 0:
                            print("[%f] Replace target network to behavior network" %(program_time+time.time()-program_ts))
                            trainer.target_net.load_state_dict(trainer.behavior_net.state_dict())
                        if learned_times % args.save_every == 0:
                            model_name = model_path + "e{}_i{}.pth".format(episode, iteration)
                            torch.save(trainer.behavior_net.state_dict(), model_name)
                            print("[%f] Model: %s saved" %(program_time+time.time()-program_ts, model_name))

