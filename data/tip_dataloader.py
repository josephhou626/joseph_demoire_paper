import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import torch
from torchvision import transforms

# for train dataloader 
# def TIP_train_loader(path):
#   img = Image.open(path).convert('RGB')
#   [w,h]=img.size
#   img=img.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6)))
#   img = img.resize((256, 256),Image.BILINEAR)
#   return img

def TIP_test_loader(path):
  img = Image.open(path).convert('RGB')
  return img


# dataset 
class commonDataset(data.Dataset):
  def __init__(self, img, Gt_img , transform=None, loader=None ):


    self.img_folder = img
    self.Gt_img_folder = Gt_img
    

    self.img_path = os.listdir( self.img_folder )


    self.Gt_img_path = os.listdir( self.Gt_img_folder )

    self.transform = transform

    self.loader = loader


  def __getitem__(self, index):

    # moire image name 
    imgA_name = self.img_path[index]

    # moire image
    imgA = self.loader(os.path.join(self.img_folder,self.img_path[index]))

    # clear image
    imgB = self.loader(os.path.join(self.Gt_img_folder,self.Gt_img_path[index]))

    if self.transform is not None:
      imgA = self.transform(imgA)
      imgB = self.transform(imgB)

    return imgA, imgB , imgA_name

  def __len__(self):
    return len(self.img_path)


# get dataloader 
def getLoader(img , Gt_img, batchSize=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True):
  if split == 'train':
    dataset = commonDataset(
                            img = img ,
                            Gt_img = Gt_img , 
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                            ]),
                            loader=TIP_test_loader)

    print('[Info]: The number of images in the train dataset :' , dataset.__len__() )
  else:
    dataset = commonDataset(
                            img = img ,
                            Gt_img = Gt_img , 
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                             ]),
                             loader=TIP_test_loader)

    print('[Info]: The number of images in the Test dataset :' , dataset.__len__() )

  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batchSize, 
                                           shuffle=shuffle)
  return dataloader
