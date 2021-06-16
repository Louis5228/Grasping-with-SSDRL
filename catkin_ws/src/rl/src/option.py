#!/usr/bin/env python3

import argparse
import torch
import os 

class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(prog="DLP homework 7", description='This lab will implement cgan')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for the trainer, default is 0.0002")
        parser.add_argument("--epochs", type=int, default=250, help="The number of the training, default is 250")
        parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for training, default is 32")
        parser.add_argument("--hidden_size", type=int, default=64, help="The hidden size for model, default is 64")

        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save model in save folder, default is current path")
        parser.add_argument("--load_model",  type=str, default=None, help="load model from wandb, ex. 'kuolunwang/DLP_homework7/model:v0', default is None")
        parser.add_argument("--dataset",  type=str, default="test", help="evaluate dataset(new_test and test), default is test.json")
        parser.add_argument("--test", action="store_true", default=False, help="True is test model, False is keep train, default is False")

        # cuda
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training, default is False')
        
        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args 