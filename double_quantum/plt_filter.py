import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


inFilterDir="./coefs/filters/"
#this script plots filters





##############################################################
#plot Phi0 layer filters
Phi0_layer_outDir=inFilterDir+'/Phi0_filters/'
###plot W0
W0_filter_file=Phi0_layer_outDir+"/shared_conv_W0_weights.pth"
W0_filter_outPath=Phi0_layer_outDir+"/W0/"
Path(W0_filter_outPath).mkdir(parents=True, exist_ok=True)
W0=torch.load(W0_filter_file)
print(W0.shape)
C,_,W,H=W0.shape
W0_to_plt=W0[:,0,:,:].detach().numpy()




width=8
height=8

for j in range(0,C):
    plt.figure(figsize=(width, height))
    W0_channel_j=W0_to_plt[j,:,:]
    plt.imshow(W0_channel_j,cmap='seismic',interpolation='nearest')
    plt.colorbar()  # Add a colorbar
    plt.title(f"W0, channel {j}")
    plt.tight_layout()
    plt.savefig(f"{W0_filter_outPath}/W0_channel_{j}.svg")
    plt.close()

## plot W1
W1_filter_file = Phi0_layer_outDir + "/shared_conv_W1_weights.pth"
W1_filter_outPath = Phi0_layer_outDir + "/W1/"
Path(W1_filter_outPath).mkdir(parents=True, exist_ok=True)
W1 = torch.load(W1_filter_file)
print(W1.shape)
C, _, W, H = W1.shape
W1_to_plt = W1[:, 0, :, :].detach().numpy()

width = 8
height = 8

for j in range(0, C):
    plt.figure(figsize=(width, height))
    W1_channel_j = W1_to_plt[j, :, :]
    plt.imshow(W1_channel_j, cmap='seismic', interpolation='nearest')
    plt.colorbar()  # Add a colorbar
    plt.title(f"W1, channel {j}")
    plt.tight_layout()
    plt.savefig(f"{W1_filter_outPath}/W1_channel_{j}.svg")
    plt.close()

##############################################################

##############################################################
#plot T1 layer filters
W_in_T1_outDir="./coefs/filters/T_filters"
W_first_file=W_in_T1_outDir+"/shared_conv_W_first.pth"
W_second_file=W_in_T1_outDir+"/shared_conv_W_second.pth"

W_first_outPath=W_in_T1_outDir+"/W_first/"
W_second_outPath=W_in_T1_outDir+"/W_second/"
Path(W_first_outPath).mkdir(parents=True, exist_ok=True)
Path(W_second_outPath).mkdir(parents=True, exist_ok=True)
#plot W_first
W_first = torch.load(W_first_file)
print(f"W_first={W_first.shape}")
C,_,W,H = W_first.shape
W_first_to_plt=W_first[:, 0, :, :].detach().numpy()

for j in range(0, C):
    plt.figure(figsize=(width, height))
    W_first_channel_j = W_first_to_plt[j, :, :]
    plt.imshow(W_first_channel_j, cmap='seismic', interpolation='nearest')
    plt.colorbar()
    plt.title(f"W_first, channel {j}")
    plt.tight_layout()
    plt.savefig(f"{W_first_outPath}/W_first_channel_{j}.svg")
    plt.close()
#plot W_second
W_second = torch.load(W_second_file)
print(f"W_second={W_second.shape}")
C,_,W,H = W_second.shape
W_second_to_plt=W_second[:, 0, :, :].detach().numpy()

for j in range(0, C):
    plt.figure(figsize=(width, height))
    W_second_channel_j = W_second_to_plt[j, :, :]
    plt.imshow(W_second_channel_j, cmap='seismic', interpolation='nearest')
    plt.colorbar()
    plt.title(f"W_second, channel {j}")
    plt.tight_layout()
    plt.savefig(f"{W_second_outPath}/W_second_channel_{j}.svg")
    plt.close()
##############################################################


