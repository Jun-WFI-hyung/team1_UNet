import torch
import torch.nn as nn
import torch.optim as optim
import json
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
    #device = torch.device("cpu")
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
    # Load pre-trained U-Net model and weights
    pre_trained_unet = Unet(class_num_=len(train_data.class_keys), depth_=depth_, image_ch_=img_channel_).to(device)
    pre_trained_unet.load_state_dict(torch.load('./carla_pth/unet_epoch007.pth')['model'])
    #pre_trained_unet.fuse_model()
    pre_trained_unet.eval()

    # Create Quantized U-Net model
    quantized_unet = QuantizedUnet(pre_trained_unet)

    my_qconfig = QConfig(
    activation=torch.quantization.default_observer,
    weight=torch.quantization.default_weight_observer
    )

    quantized_unet.qconfig = my_qconfig

    quantized_unet_prepared = prepare_qat(quantized_unet, inplace=False)

    # Now, you can run the training loop with the prepared model
    # training_loop(quantized_unet_prepared)

    # Setup dataloader, criterion, and optimizer for training
    # Set up your dataloader for training data
    criterion = nn.CrossEntropyLoss()  # or another appropriate loss function
    optimizer = optim.Adam(quantized_unet_prepared.parameters(), lr=0.001)

    # Run the QAT training loop with the prepared model
    num_epochs = 1
    quantized_unet_prepared_trained = training_loop_qat(args, cfg, quantized_unet_prepared, criterion, optimizer, num_epochs)


    ###########training########

    # After training, convert the prepared model to a quantized model
    quantized_unet_prepared_trained.eval()
    quantized_unet_prepared_trained.cpu()
    # Convert the prepared model to a fully quantized model by removing the observers and fake quantization layers.
    #quantized_unet_final = convert(quantized_unet_prepared_trained, inplace=False)

    # Use the quantized model for inference
    # output = quantized_unet_final(input_tensor)

    # Run the quantized model with an input tensor to get the output.
