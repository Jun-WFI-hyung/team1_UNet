#!/usr/bin/env python3

from net.Unet import Unet
from net.UnetData import UnetData
from utils.save_load import *
from utils.IOU import *
from utils.read_arg import *

import json, time
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms


def train(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Available Device = {device}")
    cudnn.enabled = True

    # load argument -----------------------------------------------------
    file_name_ = args.pth
    epoch_ = cfg["epoch"]
    pth_path_ = cfg["pth_path"] 
    data_path_ = cfg["data_path"]
    infer_path_ = cfg["infer_path"]
    prefix_name = cfg["prefix_name"]
    shuffle_ = cfg["shuffle"]
    data_rate_ = cfg["data_rate"]
    lr_ = cfg["learning_rate"]
    batch_size_ = cfg["batch_size"]
    num_workers_ = cfg["num_workers"]    
    depth_ = cfg["depth"]
    img_channel_ = cfg["img_channel"]
    target_channel_ = cfg["target_channel"]

    # dataset load ------------------------------------------------------
    print(f"Data init  " + "="*60)
    train_data = UnetData(data_path_, mode='T', depth_=depth_, target_ch=target_channel_)
    eval_data = UnetData(data_path_, mode='V', depth_=depth_, target_ch=target_channel_)
    train_loader = DataLoader(train_data, batch_size=batch_size_, shuffle=shuffle_, num_workers=num_workers_)
    eval_loader = DataLoader(eval_data, batch_size=batch_size_, shuffle=shuffle_, num_workers=num_workers_)
    class_num = len(train_data.class_keys) if target_channel_ is None else 1
    print(f"Data init complete  " + "="*51)

    # create network ----------------------------------------------------
    model = Unet(class_num_=class_num, depth_=depth_, image_ch_=img_channel_, target_ch_=target_channel_).to(device)

    loss_func = DiceLoss_BIN(class_num)
    sigmoid = nn.Sigmoid()
    optim = torch.optim.Adam(model.parameters(), lr=lr_)

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
        e_start = time.time()
        model.train()
        loss_arr = []

        for idx, i in enumerate(train_loader):
            start = time.time()
            print(f"epoch : {e} // idx : {idx}", end="")
            train_input = i[0].to(device)
            train_label = i[1].to(device)
            optim.zero_grad()

            train_output = model(train_input)
            train_output = sigmoid(train_output)
            train_loss = loss_func(train_output, train_label)
            train_loss.backward()

            optim.step()
            loss_arr += [train_loss.item()]
            end = time.time()            
            print(f" // Dice Loss : {100-np.mean(loss_arr)*100:.5f} % // {end-start:.5f} sec")
            if idx >= train_cnt_max: break

        train_loss = 100 - np.mean(loss_arr)*100
        print("")
        
        with torch.no_grad():
            model.eval()
            eval_loss_arr = []

            for idx, i in enumerate(eval_loader):
                start = time.time()
                print(f"epoch : {e} // eval : {idx}", end="")
                eval_input = i[0].to(device)
                eval_label = i[1].to(device)
                eval_output = model(eval_input)
                eval_loss = loss_func(eval_output, eval_label)
                eval_loss_arr += [eval_loss.item()]
                end = time.time()
                print(f" // Dice Loss : {100-np.mean(eval_loss_arr)*100:.5f} % // {end-start:.5f} sec")

                if idx >= eval_cnt_max: break
            eval_loss = 100 - np.mean(eval_loss_arr)*100
        
        print("")
        e_end = time.time()
        print(f"  - epoch elapsed time : {e_end-e_start:.5f} sec")
        save_net(pth_path_, model, optim, e, train_loss, eval_loss, e_end-e_start)



if __name__ == "__main__":

    args = read_train_arg()
    with open("./Unet_config.json") as f:
        cfg = json.load(f)
    
    train(args=args, cfg=cfg)