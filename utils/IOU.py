import torch
import torch.nn as nn

class IOU:
    def __init__(self, class_num):
        self.class_num = class_num
        self.mIOU_cnt = 0
        self.mIOU_sum = 0
        self.IOU = torch.zeros((1, class_num)).to(device="cuda:0")

    
    def IOU_out(self):
        return self.mIOU_sum / self.mIOU_cnt


    def IOU_bin(self, output:torch.Tensor, label:torch.Tensor):
        output = (output >= 1).to(torch.int8)
        px_hash,_ = torch.sort(label.view(-1).to(torch.int8) * 2 + output.view(-1))
        px_hash = torch.bincount(px_hash, minlength=self.class_num**2).view(-1, 2)

        tp = px_hash.diag()
        tp_fp_fn = px_hash.sum(dim=1) + px_hash.sum(dim=0) - tp
        
        IOU = (tp / tp_fp_fn)
        self.IOU += IOU
        self.mIOU_cnt += 1
        self.mIOU_sum += IOU[1]
        return IOU[1]
    
    
class DiceLoss_BIN(nn.Module):
    def __init__(self, class_num):
        super(DiceLoss_BIN,self).__init__()
        self.class_num = class_num
        if class_num != 1: raise Exception("Not binary class -- DiceLoss")
    
    def forward(self, output:torch.Tensor, label:torch.Tensor):
        smooth = 1e-5
        output = output.requires_grad_(True)
        label = label.requires_grad_(True)

        output = (output >= 1).to(torch.float32).view(-1)
        label = label.view(-1)

        inter = (label * output).sum()
        union = output.sum() + label.sum() - inter
        return 1 - (2*inter+smooth) / (union+smooth)


# a = torch.Tensor([[0,0,0],[0,0,0],[0,0,1]])
# b = torch.Tensor([[0,0,0],[0,0,0],[0,0,1]])

# print(IOU_bin(a,b))

