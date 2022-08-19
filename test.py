import torch
from models.DMSFN_plus import *
from data.tip_dataloader import * 
from utils.myutils import *
import cv2

def test(opt,model):
    model.eval() 
    model.to(opt.device)


    save_dir_name = opt.save_results_name
    save_output_path = rf'result\{save_dir_name}\output'
    # save_gt_path = rf'result\{save_dir_name}\gt'
    # save_input_path = rf'result\{save_dir_name}\input'


    auto_create_path(save_output_path)

    dataloader = getLoader(opt.data_moire_path,
                            opt.data_clear_path,
                            batchSize = 1,
                            mean = (0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                            split='test',
                            shuffle=False)

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