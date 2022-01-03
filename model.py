import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import stempeg
import os
import torch 
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.utils as vutils
import PIL.Image as Image

# Model structure
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1025, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
            
        )
    def forward(self, inputs):
        codes = self.encoder(inputs)
        return codes
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 1025),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder()
        # Decoder
        self.decoder = Decoder()
    def forward(self, inputs):
            codes = self.encoder(inputs)
            decoded = self.decoder(codes)
            return codes, decoded

print("finish")

#初始化模型參數

lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_ae = AutoEncoder().to(device)
# model_ae = torch.load('/content/drive/MyDrive/dataset/model_mus.pth').to(device)
optimizer = torch.optim.Adam(model_ae.parameters(), lr=lr)
loss_function = nn.MSELoss().to(device)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40], gamma=0.5)


torch.save(model_ae, './model_test.pth') # /drive/MyDrive/dataset