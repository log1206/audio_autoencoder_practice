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

import model
from model import Encoder, Decoder


from absl import app
from absl import flags

FLAGS = flags.FLAGS

def test_dataloader(train_arr):

    print("Done loading array.")
    tensor_x = torch.Tensor(train_arr.T) # transform to torch tensor transpose is important to get dataset right
    tensor_b = tensor_x[0:100][:]
    tensor_b = torch.unsqueeze(tensor_b, 0)
    for i in range(1, tensor_x.shape[0]//100): #tensor_x.shape[0]//50 -2
        test_x = tensor_x[i*100:(i+1)*100][:]
        test_x =torch.unsqueeze(test_x, 0)
        tensor_b = torch.cat((tensor_b, test_x), 0)
    
    print("Done tensor processing.")
    # 增加一維度使能放入CNN
    tensor_b_n =torch.unsqueeze(tensor_b, 1)
    #做成dataset
    tensor_y = torch.zeros(()).new_empty((tensor_b.shape[0]))
    my_dataset = TensorDataset(tensor_b_n, tensor_y) # create your datset
    
    #做成dataloader
    my_dataloader = DataLoader(my_dataset, batch_size=tensor_x.shape[0]//100) # create your dataloader

    return my_dataloader


def test_one_song(encoder, decoder, path, loss_fn):

    # load song
    
    n_fft = 2048
    hop_length = 512
    S, rate = stempeg.read_stems(path)
    # load first 7s example song in musdb      
    test_w = (S[4].T)[0][:]
    test_w_2 =(S[4].T)[1][:]
    test_w += test_w_2
    test_w = test_w[::2] ## sample rate =14700 22050 44100
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(test_w ,n_fft = n_fft, hop_length = hop_length)), ref=np.max) #librosa.amplitude_to_db(, ref=np.max)
    min = np.min(D) ##這樣做會使每一筆的範圍都有點差異因為適用該筆的min和max
    D = D-min
    D = D/np.max(D)

    mytest_dataloader = test_dataloader(D)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decoder.eval()
    encoder.eval()

    # Test and show loss

    with torch.no_grad():
      # to tensor
      conc_out = []
      conc_label = []
      for image_batch, _ in mytest_dataloader:
          # Move tensor to the proper device
          image_batch = image_batch.to(device)
          # Encode data
          encoded_data = encoder(image_batch)
          # Decode data
          decoded_data = decoder(encoded_data)
          # Append the network output and the original image to the lists
          conc_out.append(decoded_data.cpu())
          conc_label.append(image_batch.cpu())
      # Create a single tensor with all the values in the lists
      conc_out = torch.cat(conc_out)
      conc_label = torch.cat(conc_label) 
      # Evaluate global loss
      val_loss = loss_fn(conc_out, conc_label)
      print(val_loss)
      
              
    return decoded_data, image_batch

def one_test():
    print("o can call it!!!")


def recover_song(origin_data):
    out_o = origin_data[0][0][:][:].numpy()
    print(out_o.shape)
    for i in range(1,origin_data.shape[0]):
        temp =origin_data[i][0][:][:].numpy()
        out_o = np.concatenate((out_o,temp), axis=0)

    print(out_o.shape)
    out_o = out_o[0:512][:]

    out_o = out_o.T*80 -80

    # rD = (myout.T * np.abs(mu))+ mu # normalize back and transport is important
    # rD = librosa.db_to_amplitude(rD)
    # rO = (omout.T * np.abs(mu))+ mu
    rO = librosa.db_to_amplitude(out_o)

    return rO



def main(argv):

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()
    epoch = FLAGS.epoch
    encoder = torch.load('./encoder_CNN_vocal_{}.pth'.format(epoch))
    decoder = torch.load('./decoder_CNN_vocal_{}.pth'.format(epoch))

    decode, origin_data =test_one_song(encoder, decoder, "./Secretariat - Borderline.stem.mp4", loss_fn)

    decode = decode.cpu()
    origin_data = origin_data.cpu()



      # test one song
    rD = recover_song(decode)
    rO = recover_song(origin_data)

    y_inv_1 = librosa.griffinlim(rD, win_length = 2048, hop_length = 512)
    y_inv_1_o = librosa.griffinlim(rO, win_length = 2048, hop_length = 512)
    y_inv_1_o = y_inv_1_o*60
    y_inv_1 = y_inv_1*160
    import scipy.io.wavfile
    rate = 22050
    scipy.io.wavfile.write('answer.wav',rate,y_inv_1_o)
    scipy.io.wavfile.write('test.wav',rate,y_inv_1)


if __name__ == '__main__':
    app.run(main)