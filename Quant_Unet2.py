import torch
import torch.nn as nn
import torch.optim as optim
import json, copy
from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub, get_default_qat_qconfig, fuse_modules, prepare_qat, convert, default_qconfig, PerChannelMinMaxObserver, MinMaxObserver, default_qconfig, QConfig

from net.Unet import Unet
from net.UnetData import UnetData
from utils.save_load import *
from utils.accuracy import *
from utils.read_arg import *
from torchvision import transforms as transforms

class QuantizedUnet(nn.Module):
    def __init__(self, unet_model):
        super(QuantizedUnet, self).__init__()
        self.quant = QuantStub()
        self.unet = unet_model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.unet(x)
        x = self.dequant(x)
        return x
'''
    def __deepcopy__(self, memo):
        new_unet_model = copy.deepcopy(self.unet, memo)
        new_quantized_unet = QuantizedUnet(new_unet_model)
        new_quantized_unet.load_state_dict(copy.deepcopy(self.state_dict(), memo))
        return new_quantized_unet
'''
'''
def fuse_model(model):
    model.unet.encoder.conv1 = torch.quantization.fuse_modules(model.unet.encoder.conv1, ["0", "1"], inplace=True)
    for i in range(5):
        model.unet.encoder.blocks[i].conv1 = torch.quantization.fuse_modules(model.unet.encoder.blocks[i].conv1, ["0", "1"], inplace=True)
        model.unet.encoder.blocks[i].conv2 = torch.quantization.fuse_modules(model.unet.encoder.blocks[i].conv2, ["0", "1"], inplace=True)
        model.unet.decoder.blocks[i].conv1 = torch.quantization.fuse_modules(model.unet.decoder.blocks[i].conv1, ["0", "1"], inplace=True)
        model.unet.decoder.blocks[i].conv2 = torch.quantization.fuse_modules(model.unet.decoder.blocks[i].conv2, ["0", "1"], inplace=True)
    return model
'''
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

def copy_weights(original_model, quantized_model):
    for name, param in original_model.named_parameters():
        if 'weight_fake_quant' in name:
            name = name.replace('weight_fake_quant', 'weight')
        elif 'bias_fake_quant' in name:
            name = name.replace('bias_fake_quant', 'bias')
        else:
            continue
        quantized_model.state_dict()[name].data.copy_(param.data)

def training_loop_qat(args, cfg):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
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
    
    train_data = UnetData(data_path_, mode='T')
    eval_data = UnetData(data_path_, mode='V')
    train_loader = DataLoader(train_data, batch_size=batch_size_, shuffle=shuffle_, num_workers=num_workers_)
    eval_loader = DataLoader(eval_data, batch_size=batch_size_, shuffle=shuffle_, num_workers=num_workers_)
    
    print(f"Available Device = {device}")
    #pre_trained_unet
    pre_trained_unet = Unet(class_num_=len(train_data.class_keys), depth_=depth_, image_ch_=img_channel_).to(device)
    
    '''
    pre_trained_unet.to("cpu")
    fused_model = copy.deepcopy(pre_trained_unet)
    
    pre_trained_unet.train()
    fused_model.train()
    
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    print(pre_trained_unet)
    print(fused_model)    

    pre_trained_unet.eval()
    fused_model.eval()

    '''

    optim_pre_trained = torch.optim.Adam(pre_trained_unet.parameters(), lr=lr_)
    if args.load.upper() == 'T' and args.pth is not None: 
        print("Load pth file")
        pre_trained_unet, optim_pre_trained, start_epoch = load_net(pth_path_, file_name_, prefix_name, pre_trained_unet, optim_pre_trained)
    elif args.load.upper() == 'T':
        raise Exception("Put in pth filename")
    
    pre_trained_unet.eval()
    quantized_unet = QuantizedUnet(pre_trained_unet)

    ###fusion
    #quantized_unet = fuse_model(quantized_unet)        
    qconfig = get_default_qat_qconfig('fbgemm')

    qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8),
        weight=MinMaxObserver.with_args(dtype=torch.qint8)
    )
    quantized_unet.qconfig = qconfig
    print(quantized_unet.qconfig)
    
    print("quant?: ", quantized_unet.qconfig)

    quantized_unet_prepared = prepare_qat(quantized_unet, inplace=True)
    optim = torch.optim.Adam(quantized_unet_prepared.parameters(), lr=lr_)
    model = quantized_unet_prepared
    model.train()

    loss_func = nn.CrossEntropyLoss().to(device)
    
    acc = Accuracy(len(eval_data.class_keys))

    train_cnt_max = int(len(train_data) * data_rate_)
    eval_cnt_max = int(len(eval_data) * 0.05)
    
    start_epoch = 0
    

    for e in range(start_epoch+1, start_epoch + epoch_ + 1):
        
        loss_arr = []
        
        for idx, i in enumerate(train_loader):
            print(f"epoch = {e} // idx = {idx}", end="")
            train_input = i[0].to(device)
            train_input.requires_grad = True
            train_label = i[1].to(device)

            optim.zero_grad()

            train_output = model(train_input)

            train_loss = loss_func(train_output, train_label)
            train_loss.requires_grad_(True)
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
        
        print("@")
        #quantized_unet_prepared_trained#

        if e % 1 == 0:
            temp_model = copy.deepcopy(model)
            temp_model.to("cpu")
            convert_model = convert(temp_model, inplace=True)
            print(convert_model)
            model_temp = convert_model
            print("@@")
            save_net(pth_path_, model_temp, optim, e, np.mean(loss_arr), eval_data.class_keys, epoch_IOU, epoch_mIOU)
            
    temp_model = copy.deepcopy(model)
    temp_model.to("cpu")
    convert_model = convert(model, inplace=True)
    model_temp = convert_model
    save_net(pth_path_, model_temp, optim, e, np.mean(loss_arr), eval_data.class_keys, epoch_IOU, epoch_mIOU)

    return model_temp

if __name__ == "__main__":

    args = read_train_arg()
    with open("./Unet_config.json") as f:
        cfg = json.load(f)

    final_model = training_loop_qat(args=args, cfg=cfg)
    final_model.eval()
    print(final_model)