#!/usr/bin/env python3

import torch, os
import numpy as np
from glob import glob



def save_net(pth_path, model, optim, epoch, loss, cl_key, IOU, mIOU):
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)

    cp_path = os.path.join(pth_path, "unet_epoch%03d.pth" % epoch)
    torch.save({"model" : model.state_dict(), "optim" : optim.state_dict()}, cp_path)
    
    log_path = os.path.join(pth_path, "log")
    log_file_name = "log_%04d.txt"%epoch
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, log_file_name), "w", encoding="utf-8") as f:
        f.write(f"epoch = {epoch} // loss mean = {np.round(loss, 6)} // mIOU = {np.round(mIOU, 6)}\n")
        for k, v in zip(cl_key, IOU):
            f.write(f"  - {k} : {np.round(v, 6) if not np.isnan(v) else 0}\n")

    print("save network")
    print("=" * 90 + "\n")



def load_net(pth_path, file_name, prefix_name, model, optim=None):
    print("start load")
    if not os.path.exists(pth_path):
        raise Exception("pth folder not found")

    data = torch.load(os.path.join(pth_path, file_name))
    epoch = int(file_name.lstrip(prefix_name).rstrip(".pth"))

    model.load_state_dict(data["model"])

    if optim is not None: 
        optim.load_state_dict(data["optim"])
        print(f"load network --------------  start {epoch+1}")
        return model, optim, epoch
    
    else:
        print(f"load network --------------  inference {file_name}")
        return model, epoch



# def load_net_infer(pth_path, file_name, prefix_name, model):
#     print("Find model")
#     if not os.path.exists(pth_path):
#         raise Exception("pth folder not found")

#     data = torch.load(os.path.join(pth_path, file_name))

#     model.load_state_dict(data["model"])
#     epoch = int(file_name.lstrip(prefix_name).rstrip(".pth"))
#     print(f"load network --------------  {file_name}")

#     return model, epoch