import torch
from models.DMSFN_plus import *
from data.tip_dataloader import * 
from utils.myutils import *
import cv2
import argparse


def test(opt,model):
    model.eval() 
    model.to(opt.device)


    save_dir_name = opt.save_results_name
    save_output_path = rf'result\TIP18\{save_dir_name}\output'
    # save_gt_path = rf'result\{save_dir_name}\gt'
    # save_input_path = rf'result\{save_dir_name}\input'


    auto_create_path(save_output_path)

    dataloader = getLoader(opt.data_moire_path,
                            opt.data_clear_path,
                            batchSize = 1,
                            mean = (0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                            split='test',
                            shuffle=False,
                            mode='TIP')

    print(f"[Info]: Finish loading data!",flush = True)

    for batch_idx , (morie_image , clean_image , name) in enumerate(dataloader):
        with torch.no_grad():

            morie_image = morie_image.to(opt.device)
            clean_image = clean_image.to(opt.device)
            
            DMSFN_outputs= model( morie_image )
            
            for j in range(DMSFN_outputs[0].shape[0]):
                output = DMSFN_outputs[2][j, :,:,: ]
                # gt_img = clean_image[j, :,:,: ]
                # ori = morie_image[j, :, :, :]
                png_name = name[0][:-4] + '.png'
                output = cv2.cvtColor(tensor2img(output), cv2.COLOR_BGR2RGB)
                # gt_img = cv2.cvtColor(tensor2img(gt_img), cv2.COLOR_BGR2RGB)
                # ori = cv2.cvtColor(tensor2img(ori), cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(save_output_path,str('out_')+png_name), output)
                # cv2.imwrite(os.path.join(save_gt_path,str('out_')+png_name), gt_img)
                # cv2.imwrite(os.path.join(save_input_path,str('out_')+png_name), ori)
    

    print(f"[Info]: Testing over!",flush = True)


## setting parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_clear_path',default=r'D:\moire_data\moire_dataset\TIP_2018_pil_bmp\testData\target' ,help="set clean image path" ,type=str, required=False)
parser.add_argument('--data_moire_path', default=r'D:\moire_data\moire_dataset\TIP_2018_pil_bmp\testData\source',help="set moire image path" ,type=str,required=False)
parser.add_argument('--device',help='cpu or gpu')
parser.add_argument('--load_model_path', default=r'checkpoints\TIP18\DMSFN_plus\TIP_DMSFN_plus',type=str,required=False)
parser.add_argument('--save_results_name', default='DMSFN_plus',type=str,required=False)
opt = parser.parse_args()



if __name__ == "__main__":
    
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {opt.device} now!")


    model = DMSFN_plus()
    model = nn.DataParallel(model)
    print(f"[Info]: Finish creating model!",flush = True)
    print ('[Info]: Number of params: %d' % count_parameters( model ))

    pth = torch.load(opt.load_model_path)
    model.load_state_dict(pth)
    test(opt,model) # start testing