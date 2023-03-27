#!/usr/bin/env python3

from net.Unet import Unet
from net.UnetData import UnetData
from utils.save_load import *
from utils.accuracy import *
from utils.read_arg import *

import os, cv2, json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
# from torch.utils.tensorboard import SummaryWriter


def train(args, cfg):

    # load argument -----------------------------------------------------
    file_name_ = args.pth
    epoch_ = cfg["epoch"]
    pth_path_ = cfg["pth_path_ubuntu"] if args.os.upper() == 'U' else cfg["pth_path_windows"]
    data_path_ = cfg["data_path_ubuntu"] if args.os.upper() == 'U' else cfg["data_path_windows"]
    infer_path_ = cfg["infer_path_ubuntu"] if args.os.upper() == 'U' else cfg["infer_path_windows"]
    prefix_name = cfg["prefix_name"]
    shuffle_ = cfg["shuffle"]
    data_rate_ = cfg["data_rate"]
    lr_ = cfg["learning_rate"]
    batch_size_ = cfg["batch_size"]
    num_workers_ = cfg["num_workers"]    
    depth_ = cfg["depth"]
    img_channel_ = cfg["img_channel"]

    # dataset load ------------------------------------------------------
    print(f"Data init  " + "="*60)
    train_data = UnetData(data_path_, mode='T')
    eval_data = UnetData(data_path_, mode='V')
    train_loader = DataLoader(train_data, batch_size=batch_size_, shuffle=shuffle_, num_workers=num_workers_)
    eval_loader = DataLoader(eval_data, batch_size=batch_size_, shuffle=shuffle_, num_workers=num_workers_)
    print(f"Data init complete  " + "="*51)

    # create network ----------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(class_num_=len(train_data.class_keys), depth_=depth_, image_ch_=img_channel_).to(device)
    print(f"Available Device = {device}")

    # loss_func = nn.BCEWithLogitsLoss().to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr_)
    acc = Accuracy(len(eval_data.class_keys))

    train_cnt_max = int(len(train_data) * data_rate_)
    eval_cnt_max = int(len(eval_data) * 0.05)
    # train_cnt_max = 1
    # eval_cnt_max = 5

    # initialize model --------------------------------------------------
    start_epoch = 0
    if args.load.upper() == 'T' and args.pth is not None: 
        print("Load pth file")
        model, optim, start_epoch = load_net(pth_path_, file_name_, prefix_name, model, optim)
    elif args.load.upper() == 'T':
        raise Exception("Put in pth filename")

    # start cycle -------------------------------------------------------
    for e in range(start_epoch+1, start_epoch + epoch_ + 1):
        model.train()
        loss_arr = []

        for idx, i in enumerate(train_loader):
            print(f"epoch = {e} // idx = {idx}", end="")
            train_input = i[0].to(device)
            train_label = i[1].to(device)

            optim.zero_grad()

            train_output = model(train_input)

            train_loss = loss_func(train_output, train_label)
            train_loss.backward()

            optim.step()
            loss_arr += [train_loss.item()]
            print(f" // loss mean = {np.mean(loss_arr)}")
            if idx >= train_cnt_max: break
        
        print("")
        
        with torch.no_grad():
            model.eval()
            loss_arr = []

            for idx, i in enumerate(eval_loader):
                print(f"epoch = {e} // eval = {idx}", end="")
                eval_input = i[0].to(device)
                eval_label = i[1].to(device)

                eval_output = model(eval_input)

                img_mIOU = acc.compute_mIOU(eval_output, eval_label)
                eval_loss = loss_func(eval_output, eval_label)

                loss_arr += [eval_loss.item()]
                print(f" // loss mean = {np.mean(loss_arr)}", end="")
                print(f"  //  mIOU = {img_mIOU}")

                if idx >= eval_cnt_max: break

            epoch_IOU, epoch_mIOU = acc.epoch_out_IOU()
        
        print("")

        save_net(pth_path_, model, optim, e, np.mean(loss_arr), eval_data.class_keys, epoch_IOU, epoch_mIOU)



if __name__ == "__main__":

    args = read_train_arg()
    with open("./Unet_config.json") as f:
        cfg = json.load(f)
    
    train(args=args, cfg=cfg)