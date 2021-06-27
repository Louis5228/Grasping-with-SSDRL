#!/usr/bin/env python3

import os
import numpy as np
import cv2
import torch
import argparse
import wandb
from collections import namedtuple

from trainer import Trainer
from utils.prioritized_memory import Memory

# Define transition tuple
Transition = namedtuple('Transition', ['color', 'depth', 'pixel_idx', 'reward', 'next_color', 'next_depth', 'is_empty'])

class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(prog="DLP final project", description='This program for offline learning')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate for the trainer, default is 2.5e-4")
        parser.add_argument("--densenet_lr", type=float, default=5e-5, help="Learning rate for the densenet block, default is 5e-5")
        parser.add_argument("--mini_batch_size", type=int, default=10, help="How many transitions should used for learning, default is 10") # K
        parser.add_argument("--save_freq", type=int, default=5, help="Every how many update should save the model, default is 5")
        parser.add_argument("--updating_freq", type=int, default=6, help="Frequency for updating target network, default is 6") # C
        parser.add_argument("--iteration", type=int, default=30, help="The train iteration, default is 30") # M
        parser.add_argument("memory_size", type=int, default=None, help="The memory size, default is None")
        parser.add_argument("gripper_memory", type=str, default=None, help="The pkl file for save experience")

        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save model in save folder, default is current path")
        parser.add_argument("--load_model",  type=str, default=None, help="load model from wandb, ex. 'kuolunwang/DLP_final_project/model:v0', default is None")

        # cuda
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training, default is False')
        
        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args 

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

class Offline_training():
    def __init__(self, args):
        
        self.gripper_memory = Memory(args.memory_size)
        self.gripper_memory.load_memory(args.gripper_memory)

        run = wandb.init(project="DLP_final_project", entity="kuolunwang")
        config = wandb.config

        config.learning_rate = args.learning_rate
        config.iteration = args.iteration
        config.memory_size = args.memory_size
        config.updating_freq = args.updating_freq
        config.mini_batch_size = args.mini_batch_size
        config.densenet_lr = args.densenet_lr
        config.save_freq = args.save_freq
        config.gripper_memory = args.gripper_memory

        self.trainer = Trainer(args, run)

        #crate folder
        self.weight_path = os.path.join(args.save_folder,"weight")
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)

        self.training(args)

    def training(self, args):
        for i in range(args.iteration):
            mini_batch = []
            idxs = []
            is_weight = []
            old_q = []
            loss_list = []

            _mini_batch, _idxs, _is_weight = sample_data(self.gripper_memory, args.mini_batch_size)
            mini_batch += _mini_batch
            idxs += _idxs
            is_weight += list(_is_weight)

            for j in range(len(mini_batch)):
                color = cv2.imread(mini_batch[j].color)
                depth = np.load(mini_batch[j].depth)
                pixel_index = mini_batch[j].pixel_idx
                next_color = cv2.imread(mini_batch[j].next_color)
                next_depth = np.load(mini_batch[j].next_depth)

                rotate_idx = pixel_index[0]
                old_q.append(self.trainer.forward(color, depth, False, rotate_idx, clear_grad=True)[0, pixel_index[1], pixel_index[2]])
                td_target = self.trainer.get_label_value(mini_batch[j].reward, next_color, next_depth, mini_batch[j].is_empty)
                loss_ = trainer.backprop(color, depth, pixel_index, td_target, is_weight[j], args.mini_batch_size, j==0, j==len(mini_batch)-1)
                loss_list.append(loss_)

            # Update priority
            for j in range(len(mini_batch)):
                color = cv2.imread(mini_batch[j].color)
                depth = np.load(mini_batch[j].depth)
                pixel_index = mini_batch[j].pixel_idx
                next_color = cv2.imread(mini_batch[j].next_color)
                next_depth = np.load(mini_batch[j].next_depth)

                td_target = trainer.get_label_value(mini_batch[j].reward, next_color, next_depth, mini_batch[j].is_empty)
                rotate_idx = pixel_index[0]
                new_value = trainer.forward(color, depth, False, rotate_idx, clear_grad=True)[0, pixel_index[1], pixel_index[2]]
                self.gripper_memory.update(idxs[j], td_target-new_value)

            if (i+1) % args.save_freq == 0:

                torch.save(self.trainer.behavior_net.state_dict(), os.path.join(self.weight, "behavior_{}.pth".format(i+1)))

            if (i+1) % args.updating_freq == 0:
                self.trainer.target_net.load_state_dict(trainer.behavior_net.state_dict())

            if (i+1) == args.iteration:
                artifact = wandb.Artifact('model', type='model')
                artifact.add_file(os.path.join(os.path.join(self.weight, "behavior_{}.pth".format(i+1))))
                self.run.log_artifact(artifact)
                self.run.join()

            wandb.log({"loss mean": np.mean(loss_list)})

if __name__ == "__main__":

    args = Option().create()

    offline_learning = Offline_training(args)
