from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.utils import flow_to_image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_small
from PIL import Image
import cv2

device = 'cuda'
model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
model = model.eval()


class RaftTool:
    def __init__(self):
        self.weights = Raft_Small_Weights.DEFAULT
        self.transforms = self.weights.transforms()

    def preprocess(self, img_1, img_2):
        img_1 = F.resize(img_1, size=[560, 960], antialias=False)
        img_2 = F.resize(img_2, size=[560, 960], antialias=False)
        return self.transforms(img_1, img_2)

    def infer(self, img1_path, img2_path):
        img_1 = Image.open(img1_path)
        img_2 = Image.open(img2_path)
        img_1, img_2 = self.preprocess(img_1, img_2)
        img_1 = torch.unsqueeze(img_1, 0)
        img_2 = torch.unsqueeze(img_2, 0)
        list_of_flows = model(img_1.to(device), img_2.to(device))
        predicted_flows = list_of_flows[-1]
        flow_imgs = flow_to_image(predicted_flows)
        predictions = F.to_pil_image(flow_imgs[0].to("cpu"))
        return predictions

    def infer_cv2(self, img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)
        img_1, img_2 = self.preprocess(pil_img1, pil_img2)
        img_1 = torch.unsqueeze(img_1, 0)
        img_2 = torch.unsqueeze(img_2, 0)
        list_of_flows = model(img_1.to(device), img_2.to(device))
        predicted_flows = list_of_flows[-1]
        flow_imgs = flow_to_image(predicted_flows)
        predictions = F.to_pil_image(flow_imgs[0].to("cpu"))
        return predictions