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

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", None,"epoch number")
flags.mark_flag_as_required("epoch")

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(          # in = (1,100,1025)
            nn.Conv2d(1, 8, 3, stride=2, padding=1), #out =(8,50,513)
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), #out =(16,25,257)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0), #out=(32,12,128)
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(12 * 128 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 12 * 128 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 12, 128))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),# out =(16,25,257)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, #out =(8,50,513)
            padding=1, output_padding=(1,0)),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,  #out =(8,100,1025)
            padding=1, output_padding=(1,0))
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x



def main(argv):


    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    d = 64

    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    encoder = Encoder(encoded_space_dim=d,fc2_input_dim=100*1025)
    decoder = Decoder(encoded_space_dim=d,fc2_input_dim=100*1025)


    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    epoch = FLAGS.epoch

    print(epoch)
    torch.save(encoder, './encoder_CNN_vocal_{}.pth'.format(epoch)) # /drive/MyDrive/datasets
    torch.save(decoder, './decoder_CNN_vocal_{}.pth'.format(epoch)) # /drive/MyDrive/datase


if __name__ == '__main__':
    app.run(main)