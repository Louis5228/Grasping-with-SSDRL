#!/usr/bin/env python3

import os
import csv
import cv2
import numpy as np

class Logger():
    def __init__(self):

        cur_path = os.getcwd()

        self.log_path = os.path.join(cur_path, "Log")
        self.pic_path = os.path.join(cur_path, "Result")
        self.color = os.path.join(self.pic_path, "Color")
        self.depth = os.path.join(self.pic_path, "Depth")

        # create folder
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if not os.path.exists(self.pic_path):
            os.makedirs(self.pic_path)

        if not os.path.exists(self.color):
            os.makedirs(self.color)

        if not os.path.exists(self.depth):
            os.makedirs(self.depth)

    def get_path(self):
        return self.pic_path, self.color, self.depth

    def vis_affordance(self, prediction):
        tmp = np.copy(prediction)
        # View the value as probability
        tmp[tmp<0] = 0
        tmp /= 5
        tmp[tmp>1] = 1
        tmp = (tmp*255).astype(np.uint8)
        tmp.shape = (tmp.shape[0], tmp.shape[1], 1)
        heatmap = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
        return heatmap

    def draw_image(image, pixel_index, episode, iteration):
        '''
        pixel_index[0] == 0: grasp, -90
        pixel_index[0] == 1: grasp, -45
        pixel_index[0] == 2: grasp,   0
        pixel_index[0] == 3: grasp,  45
        '''
        center = (pixel_index[2], pixel_index[1])


        rotate_idx = pixel_index[0] - 2
        theta = np.radians(-90.0+45.0*rotate_idx)
        X = 20
        Y = 7
        x_unit = [ np.cos(theta), np.sin(theta)]
        y_unit = [-np.sin(theta), np.cos(theta)]
        p1  = (center[0]+np.ceil(x_unit[0]*X), center[1]+np.ceil(x_unit[1]*X))
        p2  = (center[0]-np.ceil(x_unit[0]*X), center[1]-np.ceil(x_unit[1]*X))
        p11 = (p1[0]-np.ceil(y_unit[0]*Y), p1[1]-np.ceil(y_unit[1]*Y))
        p12 = (p1[0]+np.ceil(y_unit[0]*Y), p1[1]+np.ceil(y_unit[1]*Y))
        p21 = (p2[0]-np.ceil(y_unit[0]*Y), p2[1]-np.ceil(y_unit[1]*Y))
        p22 = (p2[0]+np.ceil(y_unit[0]*Y), p2[1]+np.ceil(y_unit[1]*Y))
        p11 = (int(p11[0]), int(p11[1]))
        p12 = (int(p12[0]), int(p12[1]))
        p21 = (int(p21[0]), int(p21[1]))
        p22 = (int(p22[0]), int(p22[1]))
        result = cv2.circle(image, center, 3, color, 2)
        result = cv2.line(result, p11, p12, color, 2)
        result = cv2.line(result, p21, p22, color, 2)

        img_name = os.path.join(self.pic_path, "action_e{0:02}_i{1:04}.jpg".format(episode, iteration))

        cv2.imwrite(image_name, result)
        return result

    def save_heatmap_and_mixed(self, prediction, color, iteration, episode):
        heatmaps   = []
        mixed_imgs = []
        for grasp_prediction in prediction:
            heatmaps.append(self.vis_affordance(grasp_prediction))
        for heatmap_idx in range(len(heatmaps)):
            img_name = os.path.join(self.pic_path, "heatmap_e{0:02}_i{1:04}.jpg".format(episode, iteration))
            cv2.imwrite(img_name, heatmaps[heatmap_idx])
            img_name = os.path.join(self.pic_path, "mix_e{0:02}_i{1:04}.jpg".format(episode, iteration))
            mixed = cv2.addWeighted(color, 1.0, heatmaps[heatmap_idx], 0.4, 0)
            mixed_imgs.append(mixed)
            cv2.imwrite(img_name, mixed)
        return heatmaps, mixed_imgs

    def write_csv(self, file_name, title, data):

        with open(os.path.join(self.log_path, file_name + '.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([title])
            writer.writerow([data])