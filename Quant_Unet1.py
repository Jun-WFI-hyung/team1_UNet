import torch
import torch.nn as nn
import torch.optim as optim
import json, copy
from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub, get_default_qconfig, fuse_modules, prepare_qat, convert, default_qconfig, PerChannelMinMaxObserver, MinMaxObserver, default_qconfig, QConfig

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

    def __deepcopy__(self, memo):
        new_unet_model = copy.deepcopy(self.unet, memo)
        new_quantized_unet = QuantizedUnet(new_unet_model)
        new_quantized_unet.load_state_dict(copy.deepcopy(self.state_dict(), memo))
        return new_quantized_unet

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination._metadata[prefix + 'unet'] = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        local_metadata = local_metadata.setdefault(prefix + 'unet', {})
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

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

def training_loop_qat(args, cfg, model, criterion, optimizer, num_epochs):

    dataloader = DataLoader(train_data, batch_size=batch_size_, shuffle=shuffle_, num_workers=num_workers_)    

    model.train()
    
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    

    return model

if __name__ == "__main__":

    args = read_train_arg()
    with open("./Unet_config.json") as f:
        cfg = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    pre_trained_unet = Unet(class_num_=len(train_data.class_keys), depth_=depth_, image_ch_=img_channel_).to(device)
    pre_trained_unet.load_state_dict(torch.load('./carla_pth/unet_epoch007.pth')['model'])
    pre_trained_unet.eval()
    quantized_unet = QuantizedUnet(pre_trained_unet)

    my_qconfig = QConfig(
    activation=torch.quantization.default_observer,
    weight=torch.quantization.default_weight_observer
    )

    quantized_unet.qconfig = my_qconfig

    quantized_unet_prepared = prepare_qat(quantized_unet, inplace=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(quantized_unet_prepared.parameters(), lr=0.001)

    num_epochs = 1
    quantized_unet_prepared_trained = training_loop_qat(args, cfg, quantized_unet_prepared, criterion, optimizer, num_epochs)

    quantized_model = convert(quantized_unet_prepared_trained, inplace=True)
    copy_weights(quantized_unet_prepared_trained, quantized_model)
    quantized_model = quantized_model.cpu().eval()

    torch.save(quantized_model.state_dict(), 'trained_unet.pth')