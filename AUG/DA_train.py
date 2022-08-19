import argparse
import logging
import math
import os
import random
import numpy as np
import options.options as option
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from models import create_model
from utils import util


def init_dist(backend="nccl", **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YAML file.",
                        default='options/kernel_encoding/TIP18/train_tip.yml')
    parser.add_argument("--val_use" , type = bool , help="use valid set" , default= False)
    parser.add_argument("--traindis" ,type = int , default= 10)
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")


    # loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"], map_location=lambda storage, loc: storage.cuda(
                device_id)
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    if rank <= 0:
        if resume_state is None:
            # rename experiment folder if exists
            util.mkdir_and_rename(opt["path"]["experiments_root"])
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root" and "pretrain_model" not in key and "resume" not in key
                )
            )

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base", opt["path"]["log"], "train_" + opt["name"], level=logging.INFO, screen=True, tofile=True
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        if opt["use_tb_logger"]:
            from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="./tb_logger/" + opt["name"])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info("Random seed: {} ".format(seed))

    util.set_random_seed(seed)


    torch.backends.cudnn.benchmark = True


    # create train and val dataloader
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))

            train_sampler = None 

            train_loader = create_dataloader(
                train_set, dataset_opt, opt, train_sampler)


            logger.info("Number of train images: {:,d}, iters: {:,d}".format(
                    len(train_set), train_size))
            logger.info("Total epochs needed: {:d} for iters {:,d}".format(
                    total_epochs, total_iters))
    
    if args.val_use == True :
        for phase ,val_para in opt["validdataset"].items():
            if val_para["use"] == True : 
                val_set = create_dataset(val_para,val_para["use"])
                val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        logger.info("Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)))

 
    model = create_model(opt)


    model.cal_params()

    if resume_state:
        logger.info("Resuming training from epoch: {}, iter: {}.".format(
            resume_state["epoch"], resume_state["iter"]))

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0


    # training
    logger.info("Start training from epoch: {:d}, iter: {:d}".format(
        start_epoch, current_step))

    for epoch in range(start_epoch, total_epochs + 1):

        epoch_pix = list()
        epoch_total = list()
 
        id = 0 
        for _, train_data in enumerate(train_loader):
            current_step += 1
            id += 1 

            if current_step > total_iters:
                break
            # update learning rate 
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"])

            # training
            model.feed_data(train_data) 
            model.optimize_parameters(current_step) 

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log() 

                message = "[epoch:{:3d}, iter:{:8,d}, lr:(".format(
                    epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += "{:.3e},".format(v)
                message += ")] "
                for k, v in logs.items():
                    message += "{:s}: {:.5f} ".format(k, v)
                    # tensorboard logger
                    tb_logger.add_scalar(k, v, current_step)

                logger.info(message)

            cur_log = model.get_current_log()
            epoch_pix.append(cur_log["l_pix"])
            epoch_total.append(cur_log["l_total"])

        ##epoch loss 
        mean_total = np.mean(epoch_total)
        tb_logger.add_scalar('epoch pix loss ', mean_total, epoch)


        # save models and training states
        if epoch % 100 == 0:
            logger.info("Saving models and training states.")
            model.save(current_step) 
            model.save_training_state(epoch, current_step) 


    logger.info("Saving the final model.")
    model.save("latest") 
    logger.info("End of training .")
    tb_logger.close()


if __name__ == "__main__":
    main()
