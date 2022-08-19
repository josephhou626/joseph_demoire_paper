import logging
import random
import os 
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


logger = logging.getLogger("base")


def default_loader(path1,path2):
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    return img1 ,img2


def default_loader_random_rotate(path1,path2):
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')

    t = random.randint(0,1)
    if t==0:
        img1=img1.transpose(Image.ROTATE_90)
        img2=img2.transpose(Image.ROTATE_90)

    return img1 ,img2


transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])


class ValDataset(data.Dataset):
    def __init__(self, opt):
        super(ValDataset, self).__init__()
        self.opt = opt


        self.HQ_root = opt["dataroot_HQ"] # gt root 
        self.LQ_root = opt["dataroot_LQ"] # moire root 
        
        self.img_path = os.listdir( self.LQ_root )
        self.img_path.sort()
        self.Gt_img_path = os.listdir( self.HQ_root )
        self.Gt_img_path.sort()
        self.transform = transform
        self.loader = default_loader

        self.N_frames = opt["N_frames"]
        self.data_type = self.opt["data_type"]


    def __getitem__(self, index):

        img_LQ_name = self.img_path[index]

        img_LQ , img_HQ = self.loader(os.path.join(self.LQ_root,self.img_path[index]),os.path.join(self.HQ_root,self.Gt_img_path[index]))


        if self.transform is not None:
            img_LQ = self.transform(img_LQ)
            img_HQ = self.transform(img_HQ)


        return {"LQ": img_LQ, "HQ": img_HQ}

    def __len__(self):
        return len(self.img_path)





class TIP18Dataset(data.Dataset):


    def __init__(self, opt):
        super(TIP18Dataset, self).__init__()
        self.opt = opt


        self.HQ_root = opt["dataroot_HQ"] # gt root 
        self.LQ_root = opt["dataroot_LQ"] # moire root 
        
        self.img_path = os.listdir( self.LQ_root )
        self.img_path.sort()
        self.Gt_img_path = os.listdir( self.HQ_root )
        self.Gt_img_path.sort()
        self.transform = transform
        self.loader = default_loader
      


        self.N_frames = opt["N_frames"]
        self.data_type = self.opt["data_type"]

    def __getitem__(self, index):
       
        img_LQ_name = self.img_path[index]

        img_LQ , img_HQ = self.loader(os.path.join(self.LQ_root,self.img_path[index]),os.path.join(self.HQ_root,self.Gt_img_path[index]))


        if self.transform is not None:
            img_LQ = self.transform(img_LQ)
            img_HQ = self.transform(img_HQ)


        return {"LQ": img_LQ, "HQ": img_HQ}

    def __len__(self):
        return len(self.img_path)