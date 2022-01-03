import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import librosa
import librosa.display
import stempeg
import os

from torch.utils.data import TensorDataset
import torchvision.utils as vutils


def gen_dataloader(path,c):
    train_arr = np.load(path)
    print("Done loading array.")
    tensor_x = torch.Tensor(train_arr.T) # transform to torch tensor transpose is important to get dataset right
    tensor_b = tensor_x[0:100][:]
    tensor_b = torch.unsqueeze(tensor_b, 0)
    for i in range(1, 2048): #tensor_x.shape[0]//50 -2
        test_x = tensor_x[i*50:i*50+100][:]
        test_x =torch.unsqueeze(test_x, 0)
        tensor_b = torch.cat((tensor_b, test_x), 0)

    print("Done tensor processing.")
    # 增加一維度使能放入CNN
    tensor_b_n =torch.unsqueeze(tensor_b, 1)
    torch.save(tensor_b_n, "/musicData/logdata/data_array_vocal_2/tensor_{}.pt".format(c))




def main():
    c=0
    for root, dirs, files in os.walk("/musicData/logdata/data_array_vocal_2"):
        for file in files:
            if file.endswith(".npy"): #每個npy都做完全部epochs
                path = os.path.join(root, file)
                c+=1
                gen_dataloader(path,c)


if __name__=='__main__':
    main()