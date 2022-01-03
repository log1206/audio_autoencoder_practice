import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
# import IPython.display as ipd
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

def gen_dataloader(train_arr, batch_size):

  tensor_x = torch.Tensor(train_arr.T) # transform to torch tensor transpose is important to get dataset right
  my_dataset = TensorDataset(tensor_x, tensor_x) # create your datset
  my_dataloader = DataLoader(my_dataset, batch_size=batch_size) # create your dataloader
  return my_dataloader

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
def main():
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") #use cuda:1
  print(device)
  lr = 0.001
  model_ae = torch.load('./model_test.pth').to(device)
  optimizer = torch.optim.Adam(model_ae.parameters(), lr=lr)
  loss_function = nn.MSELoss().to(device)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40], gamma=0.5)


  epochs = 300
  model_ae.train()  


  # Train
  log_loss=[]
  total_log_loss=[]
  print("Train start")
  # every epoch
  for epoch in range(epochs):
    # every segment file
    loader_c =0
    total_loss = 0
    for root, dirs, files in os.walk("/musicData/logdata/data_array_vocal_2"):
      for file in files:
        if file.endswith(".npy"):
          
          path = os.path.join(root, file)
          train_arr = np.load(path)
          my_dataloader = gen_dataloader(train_arr, 256)
          
          loader_c += len(my_dataloader)
          del train_arr

          
          # co = 0
          for data, _ in my_dataloader:
            # co +=1 
            inputs = data.view(-1, 1025).to(device) 
            model_ae.zero_grad()
            # Forward
            codes, decoded = model_ae(inputs)
            loss = loss_function(decoded, inputs)
            loss.backward()
            optimizer.step()
            total_loss+=loss
            log_loss.append(loss)
          
          del my_dataloader
    



    total_loss /= loader_c  #len(my_dataloader.dataset)
    total_log_loss.append(total_loss)

    # print("co = ",co)
    scheduler.step()
    if epoch % 5 ==0:
      print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss.item())
  print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss.item())

  torch.save(model_ae, './model_mus.pth') # /drive/MyDrive/dataset

  test_total_tensor =[]
  for i in range(len(total_log_loss)):
      test_total_tensor.append(torch.clone(total_log_loss[i].cpu()).detach().numpy())
  plt.savefig("loss.png")

if __name__ == '__main__':

  main()