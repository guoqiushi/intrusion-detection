import os
import glob
import torch
import cv2
import json
from shapely.geometry import Polygon,Point
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from segment_anything import SamPredictor,sam_model_registry

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
class GenTool:
    def __init__(self):
        self.person_list = glob.glob('imgs/PennFudanPed/PNGImages/*.png')
        self.base_list = glob.glob('imgs/base_imgs/*/*/*.jpg')
        self.base_mask = 'mask.png'
        self.raft = RaftTool()

    def _gen_point_inside_poly(self, mode='normal'):
        with open('mask.json') as f:
            data = json.load(f)
        poly = Polygon(data['shapes'][0]['points'])
        coor_far = [(578, 75), (614, 150), (698, 142), (639, 70), (578, 75)]
        poly_far = Polygon(coor_far)
        while True:
            if mode == 'far':
                point = Point((random.randint(570, 700), random.randint(0, 150)))
                if poly_far.contains(point):
                    return (int(point.x), int(point.y))

            elif mode == 'normal':
                point = Point((random.randint(330, 1100), random.randint(90, 500)))
                poly = Polygon(data['shapes'][0]['points'])
                if poly.contains(point):
                    return (int(point.x), int(point.y))

    def _get_ratio(self, y, raw_h):
        h = 0.85 * y + 60
        return h / raw_h

    def _get_person_df(self, img_path):
        res = yolo_model(img_path)
        res_df = res.pandas().xyxy[0]
        return res_df[res_df['name'] == 'person']

    def _get_person_img_mask(self, img_path):
        perspn_img = cv2.imread(img_path)
        mask_name = os.path.basename(img_path).split('.')[0] + '_mask.png'
        mask = cv2.imread(os.path.join('imgs/PennFudanPed/PedMasks', mask_name))
        person_df = self._get_person_df(img_path)
        if len(person_df) == 0:
            return mask[50:150, 50:80]
        index = random.randint(0, len(person_df) - 1)
        x1, y1, x2, y2 = [int(x) for x in person_df.iloc[index, :4].tolist()]
        return perspn_img[y1:y2, x1:x2], mask[y1:y2, x1:x2], [x1, y1, x2, y2]

    def gen_syn_person_with_raft(self):
        person_img_path = random.choice(self.person_list)
        base_image_path = random.choice(self.base_list)
        base_image = cv2.imread(base_image_path)
        next_image = base_image.copy()
        base_mask = cv2.imread(self.base_mask)
        upper_left = self._gen_point_inside_poly()
        upper_left_2 = [upper_left[0] + 50, upper_left[1] + 50]
        # person_img , person_mask,bbox = self._get_person_img_mask(person_img_path)
        person_img, person_mask, bbox = self._get_person_img_mask('imgs/PennFudanPed/PNGImages/FudanPed00026.png')
        x1, y1, x2, y2 = bbox
        w_ori = bbox[2] - bbox[0]
        h_ori = bbox[3] - bbox[1]
        ratio = self._get_ratio(upper_left[1], h_ori)
        print(ratio)
        person_img_resized = cv2.resize(person_img, (int(ratio * w_ori), int(ratio * h_ori)))
        person_mask_resized = cv2.resize(person_mask, (int(ratio * w_ori), int(ratio * h_ori)))
        h_p_resized, w_p_resized, _ = person_img_resized.shape
        for i in range(h_p_resized):
            for j in range(w_p_resized):
                # print(person_mask[i,j].tolist())
                if person_mask_resized[i, j].tolist() != [0, 0, 0]:
                    base_image[upper_left[1] + i, upper_left[0] + j] = person_img_resized[i, j]
                    next_image[upper_left_2[1] + i, upper_left_2[0] + j] = person_img_resized[i, j]
                    base_mask[upper_left_2[1] + i, upper_left_2[0] + j] = person_mask_resized[i, j]
        raft_img = self.raft.infer_cv2(base_image, next_image)
        return base_image, next_image, base_mask, raft_img

    def gen_syn_person(self):
        person_img_path = random.choice(self.person_list)
        base_image_path = random.choice(self.base_list)
        base_image = cv2.imread(base_image_path)
        # base_image = cv2.imread('imgs/PennFudanPed/PNGImages/FudanPed00026.png')
        next_image = base_image.copy()
        base_mask = cv2.imread(self.base_mask)
        upper_left = self._gen_point_inside_poly()
        person_img, person_mask, bbox = self._get_person_img_mask(person_img_path)
        x1, y1, x2, y2 = bbox
        w_ori = bbox[2] - bbox[0]
        h_ori = bbox[3] - bbox[1]
        ratio = self._get_ratio(upper_left[1], h_ori)
        print(ratio)
        person_img_resized = cv2.resize(person_img, (int(ratio * w_ori), int(ratio * h_ori)))
        person_mask_resized = cv2.resize(person_mask, (int(ratio * w_ori), int(ratio * h_ori)))
        h_p_resized, w_p_resized, _ = person_img_resized.shape
        for i in range(h_p_resized):
            for j in range(w_p_resized):
                if person_mask_resized[i, j].tolist() != [0, 0, 0]:
                    base_image[upper_left[1] + i, upper_left[0] + j] = person_img_resized[i, j]
                    base_mask[upper_left[1] + i, upper_left[0] + j] = person_mask_resized[i, j]
        return base_image, base_mask

if __name__ == '__main__':
    tool = GenTool()
    # print(tool._get_person_mask('imgs/PennFudanPed/PNGImages/FudanPed00001.png'))
    img = tool.gen_syn_person()
