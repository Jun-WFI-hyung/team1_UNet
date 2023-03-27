import torch
import numpy as np

class Accuracy:
    def __init__(self, class_num):
        self.class_num = class_num
        
        self.freq = [0] * self.class_num
        self.sums = [0] * self.class_num

        self.mIOU_cnt = 0
        self.mIOU_sum = 0
        

    def calc_max_idx(self, out_):
        np_out = out_[0,:,:,:]
        np_out = np_out.unfold(0,np_out.shape[0],1)[0,:,:,:]
        _, max_idx = torch.max(np_out, dim=2)
        return max_idx
    
    def compute_mIOU(self, out_, label_):
        # iou = true_positive / (true_positive + false_positive + false_negative)
        label_ = label_[0,:,:,:]
        max_idx = self.calc_max_idx(out_)
        self.max_idx = max_idx
        IOU = []
        none_class = 0

        for i in range(label_.shape[0]):
            n = int(torch.count_nonzero(label_[i,:,:]).cpu())
            if n:
                pred = torch.zeros_like(max_idx)
                pred[max_idx == i] = 1
                pred_cnt = int(torch.count_nonzero(pred).cpu())

                fn = label_[i,:,:] - pred
                fn = torch.clip(fn, 0, 255)
                fn_cnt = int(torch.count_nonzero(fn).cpu())

                tp = label_[i,:,:] - fn
                tp = torch.clip(tp, 0, 255)
                tp_cnt = int(torch.count_nonzero(tp).cpu())

                div = pred_cnt + fn_cnt
                iou = 0 if not div else tp_cnt / div

                self.freq[i] += 1
                self.sums[i] += iou
                # print(f"tp = {tp_cnt} // pred_cnt = {pred_cnt} // fn_cnt = {fn_cnt} // mask_cnt = {n}")
            else: 
                # print("none")
                iou = 0
                none_class += 1

            IOU += [iou]
        
        mIOU = sum(IOU) / (self.class_num - none_class)
        self.mIOU_cnt += 1
        self.mIOU_sum += mIOU

        return mIOU
    
    def epoch_out_IOU(self):
        IOU = np.array(self.sums) / np.array(self.freq)
        mIOU = self.mIOU_sum / self.mIOU_cnt
        # print(IOU, mIOU)

        self.freq = [0] * self.class_num
        self.sums = [0] * self.class_num

        self.mIOU_cnt = 0
        self.mIOU_sum = 0
        self.max_idx = None
        return IOU, mIOU