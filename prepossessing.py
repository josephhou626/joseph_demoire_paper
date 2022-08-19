import cv2
import numpy as np 
from PIL import Image
import os
import torch.utils.data as data
from utils.myutils import *
import argparse


def pil_crop_loader(path1,path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    [w,h]=img1.size
    img1 = img1.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6)))
    img2 = img2.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6)))
    img1 = img1.resize((256, 256),Image.BILINEAR)
    img2 = img2.resize((256, 256),Image.BILINEAR)
    return img1,img2

class commonDataset(data.Dataset):
  def __init__(self, img, Gt_img , transform=None, loader=pil_crop_loader ):


    self.img_folder = img
    self.Gt_img_folder = Gt_img
    
    self.img_path = os.listdir( self.img_folder )
    self.img_path.sort()
    
    self.Gt_img_path = os.listdir( self.Gt_img_folder )
    self.Gt_img_path.sort()

    
    self.transform = transform

    self.loader = loader


  def __getitem__(self, index):

    imgA_name = self.img_path[index]

    imgA , imgB = self.loader(os.path.join(self.img_folder,self.img_path[index]),os.path.join(self.Gt_img_folder,self.Gt_img_path[index]))

    return imgA, imgB , imgA_name

  def __len__(self):
    return len(self.img_path)



## setting parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_clear_path',default=r'D:\moire_data\moire_dataset\TIP2018\testData\target' ,type=str, required=False)
parser.add_argument('--data_moire_path', default=r'D:\moire_data\moire_dataset\TIP2018\testData\source' ,type=str,required=False)
parser.add_argument('--save_dir_name', default='TrainData',type=str,required=False)
opt = parser.parse_args()

if __name__ == "__main__":



  train_gt_path = opt.data_clear_path
  train_input_path = opt.data_moire_path
  save_dir_name = opt.save_dir_name


  save_gt_path = rf'datasets\TIP18_crop\{save_dir_name}\target'
  save_input_path = rf'datasets\TIP18_crop\{save_dir_name}\source'

  auto_create_path(save_input_path)
  auto_create_path(save_gt_path)

  mydataset = commonDataset(train_input_path,train_gt_path)

  count = 1 
  for batch_idx, (morie_image, clean_image, name) in enumerate(mydataset):
      print("Image : ", name)
      count = count + 1 
      name = name[:-3]
      name = name + 'bmp'
      morie_image.save(os.path.join(save_input_path, str('out_')+name))
      clean_image.save(os.path.join(save_gt_path, str('out_')+name))