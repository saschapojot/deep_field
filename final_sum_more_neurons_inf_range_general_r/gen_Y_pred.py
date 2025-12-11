import torch
import torch.nn as nn
import sys
import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model_dsnn_config import format_using_decimal,DSNN,CustomDataset,L,r,decrease_over
from model_dsnn_config import  decrease_rate,batch_size,learning_rate,weight_decay
from model_dsnn_config import  epoch_multiple,device,format_using_decimal
import glob

#this script generates Y_pred

num_layers=3
neuron_num=150
num_epochs=8000
K=455

decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
# for fl in glob.glob(f"{data_inDir}/*"):
#     print(fl)
fileNameTest=data_inDir+"/inf_range.test.pkl"
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}"
model_inDir=f"./out_model_L{L}_K{K}_r{r}/layer{num_layers}/neurons{neuron_num}/"

model_file=model_inDir+f"/DSNN_model{suffix_str}.pth"
with open(fileNameTest, 'rb') as f:
    X_test, Y_test = pickle.load(f)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float64).view(-1, 1).to(device)


print(f"Y_test.shape={Y_test.shape}")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# Set model to evaluation mode
model = DSNN(num_spins=L, num_layers=num_layers, num_neurons=neuron_num).double()

criterion = nn.MSELoss()
model.load_state_dict(torch.load(model_file))

# Move the model to the appropriate device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

print("Model successfully loaded and ready for evaluation.")
with torch.no_grad():
    # Forward pass to get predictions
    predictions = model(X_test)
    Y_pred = predictions.cpu().numpy().flatten().tolist()


save_data = [X_test, Y_pred]
out_value_file=model_inDir+f"/inference_data_L{L}_K{K}_r{r}_layer{num_layers}_neurons{neuron_num}.pkl"
with open(out_value_file,"wb") as fptr:
    pickle.dump(save_data,fptr)