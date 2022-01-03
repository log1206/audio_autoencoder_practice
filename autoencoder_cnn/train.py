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

## my library
import model
from model import Encoder, Decoder

FLAGS = flags.FLAGS



# class Encoder(nn.Module):
    
#     def __init__(self, encoded_space_dim,fc2_input_dim):
#         super().__init__()
        
#         ### Convolutional section
#         self.encoder_cnn = nn.Sequential(          # in = (1,100,1025)
#             nn.Conv2d(1, 8, 3, stride=2, padding=1), #out =(8,50,513)
#             nn.ReLU(True),
#             nn.Conv2d(8, 16, 3, stride=2, padding=1), #out =(16,25,257)
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, 3, stride=2, padding=0), #out=(32,12,128)
#             nn.ReLU(True)
#         )
        
#         ### Flatten layer
#         self.flatten = nn.Flatten(start_dim=1)
# ### Linear section
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(12 * 128 * 32, 128),
#             nn.ReLU(True),
#             nn.Linear(128, encoded_space_dim)
#         )
        
#     def forward(self, x):
#         x = self.encoder_cnn(x)
#         x = self.flatten(x)
#         x = self.encoder_lin(x)
#         return x
# class Decoder(nn.Module):
    
#     def __init__(self, encoded_space_dim,fc2_input_dim):
#         super().__init__()
#         self.decoder_lin = nn.Sequential(
#             nn.Linear(encoded_space_dim, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 12 * 128 * 32),
#             nn.ReLU(True)
#         )

#         self.unflatten = nn.Unflatten(dim=1, 
#         unflattened_size=(32, 12, 128))

#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, 3, 
#             stride=2, output_padding=0),# out =(16,25,257)
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 3, stride=2, #out =(8,50,513)
#             padding=1, output_padding=(1,0)),
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 3, stride=2,  #out =(8,100,1025)
#             padding=1, output_padding=(1,0))
#         )
        
#     def forward(self, x):
#         x = self.decoder_lin(x)
#         x = self.unflatten(x)
#         x = self.decoder_conv(x)
#         x = torch.sigmoid(x)
#         return x

### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)



#將data做成dataloader
# print(train_arr.shape)
def gen_dataloader(path, batch_size):
    #   train_arr = np.load(path)
    #   print("Done loading array.")
    #   tensor_x = torch.Tensor(train_arr.T) # transform to torch tensor transpose is important to get dataset right
    #   tensor_b = tensor_x[0:100][:]
    #   tensor_b = torch.unsqueeze(tensor_b, 0)
    #   for i in range(1, 1024): #tensor_x.shape[0]//50 -2
    #     test_x = tensor_x[i*50:i*50+100][:]
    #     test_x =torch.unsqueeze(test_x, 0)
    #     tensor_b = torch.cat((tensor_b, test_x), 0)

    # print("Done tensor processing.")
    # # 增加一維度使能放入CNN
    # tensor_b_n =torch.unsqueeze(tensor_b, 1)
    
    #做成dataset
    tensor_b_n = torch.load(path)
    tensor_y = torch.zeros(()).new_empty((tensor_b_n.shape[0]))
    my_dataset = TensorDataset(tensor_b_n, tensor_y) # create your datset

    #做成dataloader
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size) # create your dataloader

    return my_dataloader


def main(argv):
    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Define an optimizer (both for the encoder and the decoder!)
    lr= 0.001

    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    d = 64

    num_epochs = FLAGS.epoch
    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    epath ="./encoder_CNN_vocal_{}.pth".format(num_epochs)
    dpath = './decoder_CNN_vocal_{}.pth'.format(num_epochs)
    encoder = torch.load(epath)
    decoder = torch.load(dpath)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)


    diz_loss = {'train_loss':[],'val_loss':[]}
    for epoch in range(num_epochs):
        total_train_loss =0
        fc=0
        for root, dirs, files in os.walk("/musicData/logdata/data_array_vocal_2"):
            for file in files:
                if file.endswith(".pt"): #每個npy都做完全部epochs
                    fc+=1
                    path = os.path.join(root, file)
                    
                    train_dataloader = gen_dataloader(path, 256)

                    
                    train_loss =train_epoch(encoder,decoder,device,
                    train_dataloader,loss_fn,optim)
                    total_train_loss += train_loss
                    
                    del train_dataloader
        #  val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
        total_train_loss /=fc
        print('\n EPOCH {}/{} train loss {}'.format(epoch + 1, num_epochs,total_train_loss))
        diz_loss['train_loss'].append(total_train_loss)
        #  diz_loss['val_loss'].append(val_loss)


    torch.save(encoder, './encoder_CNN_vocal_{}.pth' .format(num_epochs)) # /drive/MyDrive/datasets
    torch.save(decoder, './decoder_CNN_vocal_{}.pth' .format(num_epochs)) # /drive/MyDrive/datase

if __name__=='__main__':
    app.run(main)