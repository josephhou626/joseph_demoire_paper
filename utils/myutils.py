import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def auto_create_path(FilePath):
    if os.path.exists(FilePath):   
            print(f"[Info]: {FilePath} exists!",flush = True)
    else:
            print(f"[Info]: {FilePath} not exists!",flush = True)
            os.makedirs(FilePath)  

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor2img(input_image):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

    return image_numpy


def aim_tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))*255

    return image_numpy


# vgg19 用的
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad