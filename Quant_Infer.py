#!/usr/bin/env python3

from net.Unet import Unet
from Quant_Unet2 import QuantizedUnet
from net.UnetData import UnetData
from utils.save_load import *
from utils.accuracy import *
from utils.read_arg import *

import os, cv2, json, time
import numpy as np
import torch.quantization as quantization

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
# from torch.utils.tensorboard import SummaryWriter

def save_log(infer_path, epoch, loss, cl_key, IOU, mIOU, F_time, E_time):
    log_path = os.path.join(infer_path, "log")
    log_file_name = "log_%04d.txt"%epoch
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, log_file_name), "w", encoding="utf-8") as f:
        f.write(f"loss mean = {np.round(loss, 6)}\n")
        f.write(f"FPS : {F_time:.5f} sec\n")
        f.write(f"Inference time : {np.mean(E_time):.5f} sec\n")
        f.write(f"mIOU = {np.round(mIOU, 6)}\n")
        for k, v in zip(cl_key, IOU):
            f.write(f"  - {k} : {np.round(v, 6) if not np.isnan(v) else 0}\n")

    print("\n **  saved log  **")
    print(f" - loss mean      : {np.round(loss, 6)}")
    print(f" - FPS            : {F_time:.5f} sec")
    print(f" - Inference time : {np.mean(E_time):.5f} sec")
    print(f" - mIOU           : {np.round(mIOU, 6) * 100:.2f} %")



def save_img(infer_path, epoch, idx, max_idx, cl_keys, classes, total):
    save_path = os.path.join(infer_path, "img")
    file_name = "epoch_%04d_%02d.png" % (epoch, idx)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img = np.zeros((max_idx.shape[0], max_idx.shape[1], 3), dtype=np.uint8)
    for cl_idx, cl in enumerate(cl_keys):
        img[(max_idx == cl_idx).cpu().numpy()] = np.array(classes[cl])
    
    print(f"save result ----- {idx+1} / {total}")
    cv2.imwrite(os.path.join(save_path, file_name), img)



def infer(args, cfg):
    # load argument -----------------------------------------------------
    file_name_ = args.pth
    pth_path_ = cfg["pth_path"]
    data_path_ = cfg["data_path"]
    infer_path_ = cfg["infer_path"]
    prefix_name = cfg["prefix_name"]
    batch_size_ = cfg["batch_size"]
    num_workers_ = cfg["num_workers"]
    depth_ = cfg["depth"]
    img_channel_ = cfg["img_channel"]
    
    # dataset load ------------------------------------------------------
    print(f"Data init  " + "="*60)
    infer_data = UnetData(data_path_, mode="I")
    infer_loader = DataLoader(infer_data, batch_size=batch_size_, num_workers=num_workers_, shuffle=False)
    class_num = len(infer_data.class_keys)
    print(f"Data init complete  " + "="*51)

    # create network ----------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Available Device = {device}")
    model = QuantizedUnet(Unet(class_num_=class_num, depth_=depth_, image_ch_=img_channel_)).to(device)
    acc = Accuracy(class_num)

    loss_func = nn.CrossEntropyLoss().to(device)
    # loss_func = nn.BCEWithLogitsLoss().to(device)

    # initialize model --------------------------------------------------
    model, epoch = load_net(pth_path_, file_name_, prefix_name, model)
    model = quantization.convert(model, inplace=False)
    print('---------------------------')
    print(model)
    print('---------------------------')
    # min_g = []
    # max_g = []
    # for param in model.parameters():
    #     print(param)
    #     print(param.data.min(),param.data.max())
    #     min_g.append(param.data.min().item())
    #     max_g.append(param.data.max().item())
    # print(min(min_g))
    # print(max(max_g))

    print("----------------------")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

    with torch.no_grad():
        model.eval()
        loss_arr = []
        elapsed_time = []
        cnt = 0

        fps_start = time.time()
        infer = []
        for idx, i in enumerate(infer_loader):
            cnt += 1
            infer_input = i[0].to(device)
            infer_label = i[1].to(device)

            infer_start = time.time()
            # infer_output = model(infer_input)
            with torch.autograd.profiler.profile() as prof:
                infer_output = model(infer_input)
            print(f"infer_output:{infer_output}")
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            infer_end = time.time()
            elapsed_time.append(infer_end - infer_start)
            print(infer_end - infer_start)
            infer.append(infer_end - infer_start)
            print(f"Inference output dtype: {infer_output.dtype}")

            img_mIOU = acc.compute_mIOU(infer_output, infer_label)
            
            infer_loss = loss_func(infer_output, infer_label)
            loss_arr += [infer_loss.item()]

            save_img(infer_path_, 
                     epoch, 
                     idx, 
                     acc.max_idx,
                     infer_data.class_keys,
                     infer_data.classes,
                     len(infer_data))
            
        fps_end = time.time()
        fps_time = (fps_end - fps_start) / cnt

        epoch_IOU, epoch_mIOU = acc.epoch_out_IOU()
        save_log(infer_path_, 
                 epoch, 
                 np.mean(loss_arr),
                 infer_data.class_keys,
                 epoch_IOU,
                 epoch_mIOU,
                 fps_time,
                 elapsed_time)
    print(f"quant_min_infer:{sum(infer)/10}")


if __name__ == "__main__":

    args = read_infer_arg()
    with open("./Unet_config.json") as f:
        cfg = json.load(f)

    infer(args=args, cfg=cfg)