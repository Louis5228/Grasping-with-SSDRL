#!/usr/bin/env python3

import argparse
import torch
import os 

class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(prog="DLP final project", description='This project will implement grasp with self supervised deep reinforcement learining')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the trainer, default is 5e-4")
        parser.add_argument("--epsilon", type=float, default=0.5, help="Probability to choose random action, default is 0.5")
        parser.add_argument("--buffer_size", type=int, default=1000, help="Experience buffer size, default is 1000") # N
        parser.add_argument("--learning_freq", type=int, default=5, help="Frequency for updating behavior network, default is 5") # M
        parser.add_argument("--updating_freq", type=int, default=10, help="Frequency for updating target network, default is 10") # C
        parser.add_argument("--mini_batch_size", type=int, default=4, help="How many transitions should used for learning, default is 4") # K
        parser.add_argument("--densenet_lr", type=float, default=1e-4, help="Learning rate for the densenet block, default is 1e-4")
        parser.add_argument("--save_every", type=int, default=5, help="Every how many update should save the model, default is 5")
        parser.add_argument("--gripper_memory", type=str, default="")
        parser.add_argument("--record", action="store_true", default=False, help="collect data for replay buffer")
        parser.add_argument("--size_lim", type=int, default=20, help="The times of collect data")
        parser.add_argument("--episode", type=int, default=0, help="From the episode start training, default is 0")

        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save model in save folder, default is current path")
        parser.add_argument("--load_model",  type=str, default=None, help="load model from wandb, ex. 'kuolunwang/DLP_final_project/model:v0', default is None")
        parser.add_argument("--test", action="store_true", default=False, help="True is test model, False is keep train, default is False")

        # cuda
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training, default is False')
        
        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args 










