import argparse
import logging
import os
import os.path as osp
import random
from torchvision import transforms
from PIL import Image
import cv2
import data.util as data_util
import lmdb
import numpy as np
import torch
import utils.util as util
import yaml
from models.kernel_encoding.kernel_wizard import KernelWizard


def default_loader(path):
  img = Image.open(path).convert('RGB')
  return img

transform1 = transforms.Compose([
        transforms.ToTensor(),
    ])
def main():
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(description="Kernel extractor testing")


    parser.add_argument("--source_H", action="store",
                        help="source image height", type=int, default=256)
    parser.add_argument("--source_W", action="store",
                        help="source image width", type=int, default=256)
    parser.add_argument("--target_H", action="store",
                        help="target image height", type=int, default=256)
    parser.add_argument("--target_W", action="store",
                        help="target image width", type=int, default=256)
    parser.add_argument(
        "--augmented_H", action="store", help="desired height of the augmented images", type=int, default=256
    )
    parser.add_argument(
        "--augmented_W", action="store", help="desired width of the augmented images", type=int, default=256
    )

    # source moire
    parser.add_argument(
        "--source_LQ_root", action="store", help="source low-quality dataroot", type=str, default='../datasets/TIP18_crop/TrainData/source'
    )
    
    # source GT
    parser.add_argument(
        "--source_HQ_root", action="store", help="source high-quality dataroot", type=str, default='../datasets/TIP18_crop/TrainData/target'
    )

    # target GT 
    parser.add_argument(
        "--target_HQ_root", action="store", help="target high-quality dataroot", type=str, default='../datasets/TIP18_crop/TrainData/target'
    )

    # save results dir 
    parser.add_argument("--save_path", action="store",
                        help="save path", type=str, default='results/AUG_TIP')

    parser.add_argument("--yml_path", action="store",
                        help="yml path", type=str, default='options/data_augmentation/test_tip.yml')

    parser.add_argument(
        "--num_images", action="store", help="number of desire augmented images", type=int, default=10)

    args = parser.parse_args()

    source_LQ_root = args.source_LQ_root
    source_HQ_root = args.source_HQ_root
    target_HQ_root = args.target_HQ_root

    save_path = args.save_path
    source_H, source_W = args.source_H, args.source_W
    target_H, target_W = args.target_H, args.target_W
    augmented_H, augmented_W = args.augmented_H, args.augmented_W
    yml_path = args.yml_path
    num_images = args.num_images

    # Initializing logger
    logger = logging.getLogger("base")
    os.makedirs(save_path, exist_ok=True)
    util.setup_logger("base", save_path, "test",
                      level=logging.INFO, screen=True, tofile=True)
    logger.info("source LQ root: {}".format(source_LQ_root))
    logger.info("source HQ root: {}".format(source_HQ_root))
    logger.info("target HQ root: {}".format(target_HQ_root))
    logger.info("augmented height: {}".format(augmented_H))
    logger.info("augmented width: {}".format(augmented_W))
    logger.info("Number of augmented images: {}".format(num_images))

    # Initializing mode
    logger.info("Loading model...")
    with open(yml_path, "r") as f:
        print(yml_path)
        opt = yaml.load(f)["KernelWizard"]
    model_path = opt["pretrained"]
    model = KernelWizard(opt)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    logger.info("Done")
    

    source_LQ_files = os.listdir(source_LQ_root)
    source_LQ_files.sort()

    source_HQ_files = os.listdir(source_HQ_root)
    source_HQ_files.sort()

    target_HQ_files = os.listdir(target_HQ_root)
    target_HQ_files.sort()

    psnr_avg = 0

    for i in range(num_images):

        ran_num = random.randint(0, len(source_LQ_files)-1)
        source_HQ_file = source_HQ_files[ran_num]
        source_LQ_file = source_LQ_files[ran_num]

        target_file = np.random.choice(target_HQ_files)

    
        source_LQ = default_loader(os.path.join(source_LQ_root,source_LQ_file))
        source_HQ = default_loader(os.path.join(source_HQ_root,source_HQ_file))
        target_HQ = default_loader(os.path.join(target_HQ_root,target_file))

        source_LQ = transform1(source_LQ).unsqueeze(0)
        source_HQ = transform1(source_HQ).unsqueeze(0)
        target_HQ = transform1(target_HQ).unsqueeze(0)


        source_LQ= source_LQ.to(device)
        source_HQ = source_HQ.to(device)
        target_HQ = target_HQ.to(device)



        with torch.no_grad():

            kernel_mean, kernel_sigma = model(source_HQ, source_LQ)

            kernel = kernel_mean + kernel_sigma * torch.randn_like(kernel_mean)

            fake_source_LQ = model.adaptKernel(source_HQ, kernel)

            target_LQ = model.adaptKernel(target_HQ, kernel)

        LQ_img = util.tensor2img(source_LQ)
        fake_LQ_img = util.tensor2img(fake_source_LQ)
        target_LQ_img = util.tensor2img(target_LQ)
        target_HQ_img = util.tensor2img(target_HQ)
        source_HQ_img = util.tensor2img(source_HQ)


        target_HQ_dst = osp.join(
            save_path, "target_gt/aug_target_{:08d}.bmp".format(i % num_images))
        target_LQ_dst = osp.join(
            save_path, "target_moire/aug_source_{:08d}.bmp".format(i % num_images))

        source_LQ_dst = osp.join(
            save_path, "source_moire/{:08d}.bmp".format(i % num_images))

        fake_source_LQ_dst = osp.join(
            save_path, "fake_source_moire/{:08d}.bmp".format(i % num_images))

        source_HQ_dst = osp.join(
            save_path, "source_gt/{:08d}.bmp".format(i % num_images))

        os.makedirs(osp.dirname(fake_source_LQ_dst), exist_ok=True)
        os.makedirs(osp.dirname(target_HQ_dst), exist_ok=True)
        os.makedirs(osp.dirname(target_LQ_dst), exist_ok=True)
        os.makedirs(osp.dirname(source_LQ_dst), exist_ok=True)
        os.makedirs(osp.dirname(source_HQ_dst), exist_ok=True)

        cv2.imwrite(fake_source_LQ_dst, fake_LQ_img)
        cv2.imwrite(source_LQ_dst, LQ_img)  
        cv2.imwrite(target_HQ_dst, target_HQ_img)
        cv2.imwrite(target_LQ_dst, target_LQ_img)
        cv2.imwrite(source_HQ_dst, source_HQ_img)

        psnr = util.calculate_psnr(LQ_img, fake_LQ_img)

        logger.info(
            "Reconstruction PSNR of image #{:03d}/{:03d}: {:.2f}db".format(i, num_images, psnr))
        psnr_avg += psnr

    logger.info("Average reconstruction PSNR: {:.2f}db".format(
        psnr_avg / num_images))


main()
