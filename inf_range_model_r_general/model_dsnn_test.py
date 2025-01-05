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
from model_dsnn_config import  num_neurons,decrease_rate,batch_size,learning_rate,weight_decay
from model_dsnn_config import  epoch_multiple,device


num_layers=3

K=455

data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"

fileNameTest=data_inDir+"/inf_range.test.pkl"

model_inDir=f"./out_model_L{L}_K{K}_r{r}_layer{num_layers}/"

model_file=model_inDir+"/DSNN_model.pth"

with open(fileNameTest, 'rb') as f:
    X_test, Y_test = pickle.load(f)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float64).view(-1, 1).to(device)



# Set model to evaluation mode
model = DSNN(num_spins=L, num_layers=num_layers, num_neurons=num_neurons).double()

criterion = nn.MSELoss()
model.load_state_dict(torch.load(model_file))

# Move the model to the appropriate device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

print("Model successfully loaded and ready for evaluation.")

# Disable gradient computation for evaluation
errors = []
with torch.no_grad():
    # Forward pass to get predictions
    predictions = model(X_test)

    # Compute loss or other evaluation metrics
    test_loss = criterion(predictions, Y_test).item()
    batch_errors = (predictions - Y_test).cpu().numpy()  # Convert to NumPy for easier handling
    errors.extend(batch_errors.flatten())  # Flatten and add to the list

# print(errors)
print(f"Test Loss: {test_loss:.4f}")
# Convert errors to a NumPy array
errors = np.array(errors)

# Compute the variance of the errors
# error_variance = np.var(errors)
# print(f"Error Variance: {error_variance:.8f}")
std_loss=np.sqrt(test_loss)
outTxtFile=model_inDir+f"test_DSNN.txt"

out_content=f"MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}\n"
with open(outTxtFile,"w+") as fptr:
    fptr.write(out_content)
