import torch
import numpy as np
import os
from models.DMSFN_plus import *
from tqdm import tqdm
from data.tip_dataloader import * 
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from utils.myutils import *
from vgg_models.vgg import VGG19
import argparse
from utils.myutils import *

def train(opt,model):
    
    # tadm log
    pbar = tqdm(total= opt.max_epochs, ncols=0, desc="Train", unit=" step")

    # tensorboard log 
    writer = SummaryWriter(comment='DMSFN_plus')

    # set model
    model.train()
    model.to(opt.device)

    # set dataset and dataloader
    dataloader = getLoader(opt.data_moire_path,
                            opt.data_clear_path,
                        batchSize = opt.batch_size,
                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                        split='train',
                        shuffle=True)

    print(f"[Info]: Finish loading data!",flush = True)

    # optimizer
    optimizer_model = torch.optim.Adam(model.parameters(), lr = 0.0001, betas=(0.5, 0.999))
    
    # l1 loss 
    L1_loss = torch.nn.L1Loss()

    # perceptual loss
    vgg19 = VGG19()
    vgg19.load_model(opt.load_vgg19_path)
    vgg19.to(opt.device)
    set_requires_grad(vgg19, False)
    vgg19.eval()
    vgg_layers = {'conv1_2': 1.0, 'conv2_2': 1.0, 'conv3_2':0.5}
    mes_loss = torch.nn.MSELoss()

    print(f"[Info]: Start training data!",flush = True)
    for epoch in range(opt.max_epochs):

        tot_s_loss = list() # record batch loss

        for batch_idx , (morie_image , clean_image , name) in enumerate(dataloader):
            
            optimizer_model.zero_grad()

            morie_image = morie_image.to(opt.device)
            clean_image = clean_image.to(opt.device)
            
            clean_image2 = F.interpolate(clean_image, scale_factor=0.5, mode='bilinear')
            clean_image4 = F.interpolate(clean_image, scale_factor=0.25, mode='bilinear')

  
            DMSFN_outputs = model( morie_image )

            #perceptual loss : branch1 
            loss_per_branch3 = 0.0
            vgg_outputs = vgg19(DMSFN_outputs[2])
            vgg_clear = vgg19(clean_image)


            for l, w in vgg_layers.items():
                loss_per_branch3 += w * mes_loss(vgg_outputs[l], vgg_clear[l])


            #perceptual loss:  branch2
            loss_per_branch2 = 0.0
            vgg_outputs = vgg19(DMSFN_outputs[1])
            vgg_clear = vgg19(clean_image2)


            for l, w in vgg_layers.items():
                loss_per_branch2 += w * mes_loss(vgg_outputs[l], vgg_clear[l])


            #perceptual loss : branch3
            loss_per_branch1 = 0.0
            vgg_outputs = vgg19(DMSFN_outputs[0])
            vgg_clear = vgg19(clean_image4)

            for l, w in vgg_layers.items():
                loss_per_branch1 += w * mes_loss(vgg_outputs[l], vgg_clear[l])

            branch3_loss = L1_loss(clean_image , DMSFN_outputs[2] )
            branch1_loss = L1_loss(clean_image4 , DMSFN_outputs[0] )
            branch2_loss = L1_loss(clean_image2 , DMSFN_outputs[1] )


            batch_total_loss = branch1_loss + branch2_loss + branch3_loss + loss_per_branch1  + loss_per_branch2 + loss_per_branch3

            batch_total_loss.backward(retain_graph = True)

            tot_s_loss.append(batch_total_loss.item())


            optimizer_model.step()

        mean_s_loss = np.mean(tot_s_loss)
        pbar.update() 

        pbar.set_postfix({'Epoch Loss' : mean_s_loss  ,'epoch':epoch+1})
        writer.add_scalar('DMSFN_plus Epoch Loss', mean_s_loss,epoch+1)  
        
        # save model 
        if (epoch+1) % 1 == 0 :
            torch.save(model.state_dict(), os.path.join(opt.save_model_path,rf'DNSFN_{epoch}'))

    pbar.close()
    writer.close()

    torch.save(model.state_dict(), os.path.join(opt.save_model_path,rf'DNSFN_latest'))
    print(f"[Info]: Training over!",flush = True)


## setting parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_clear_path',default=r'D:\moire_data\moire_dataset\TIP_2018_pil_bmp\testData\target' ,help="set clean image path" ,type=str, required=False)
parser.add_argument('--data_moire_path', default=r'D:\moire_data\moire_dataset\TIP_2018_pil_bmp\testData\source',help="set moire image path" ,type=str,required=False)
parser.add_argument('--max_epochs', default=50, help="total training epoch" ,type=int,required=False) # we set 50 epoch
parser.add_argument('--batch_size', default=4,type=int,required=False)
parser.add_argument('--device',help='cpu or gpu')
parser.add_argument('--resume', default=False,help="use to pretrain model",type=bool,required=False)
parser.add_argument('--load_model_path', default=r'checkpoints\TIP18\DMSFN_plus\TIP_DMSFN_plus',type=str,required=False)
parser.add_argument('--load_vgg19_path', default='vgg_models/vgg19-dcbb9e9d.pth',type=str,required=False)
parser.add_argument('--save_model_path', default=r"checkpoints\TIP18\DMSFN_plus",help="save model path",type=str,required=False)
opt = parser.parse_args()


if __name__ == "__main__":
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {opt.device} now!")


    model = DMSFN_plus()
    model = nn.DataParallel(model)
    print(f"[Info]: Finish creating model!",flush = True)
    
    if opt.resume == True and opt.load_model_path!= None :
        pth = torch.load(opt.load_model_path) 
        model.load_state_dict(pth) 
        print(f"[Info]: Finish load pretrained weight!",flush = True)

    auto_create_path(opt.save_model_path)
    train(opt,model) # start training