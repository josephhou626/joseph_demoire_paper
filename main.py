import os
import argparse
from test import *
from train import *
from utils.myutils import *
from models.DMSFN_plus import *
import torch
import torch.nn as nn

## setting parser
parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='train' ,help="choose train or test mode" ,type=str , required=False)
parser.add_argument('--data_clear_path',default=r'D:\moire_data\moire_dataset\TIP_2018_pil_bmp\testData\target' ,help="set clean image path" ,type=str, required=False)
parser.add_argument('--data_moire_path', default=r'D:\moire_data\moire_dataset\TIP_2018_pil_bmp\testData\source',help="set moire image path" ,type=str,required=False)
parser.add_argument('--max_epochs', default=50, help="total training epoch" ,type=int,required=False) # we set 50 epoch
parser.add_argument('--batch_size', default=4,type=int,required=False)
parser.add_argument('--device',help='cpu or gpu')
parser.add_argument('--resume', default=False,help="use to pretrain model",type=bool,required=False)
parser.add_argument('--load_model_path', default=r'checkpoints\DMSFN_plus\TIP_DMSFN_plus',type=str,required=False)
parser.add_argument('--load_vgg19_path', default='vgg_models/vgg19-dcbb9e9d.pth',type=str,required=False)
parser.add_argument('--save_model_path', default=r"checkpoints\DMSFN_plus",help="save model path",type=str,required=False)
parser.add_argument('--save_results_name', default='DMSFN_plus',type=str,required=False)
opt = parser.parse_args()


if __name__ == "__main__":

    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {opt.device} now!")


    model = DMSFN_plus()
    model = nn.DataParallel(model)
    print(f"[Info]: Finish creating model!",flush = True)
    print ('[Info]: Number of params: %d' % count_parameters( model ))
    
    if opt.mode == 'train':
        if opt.resume == True and opt.load_model_path!= None :
            pth = torch.load(opt.load_model_path) 
            model.load_state_dict(pth) 
            print(f"[Info]: Finish load pretrained weight!",flush = True)

        auto_create_path(opt.save_model_path)
        train(opt,model) # start training

    else:
        pth = torch.load(opt.load_model_path)
        model.load_state_dict(pth)
        test(opt,model) # start testing



    