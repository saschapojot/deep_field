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
from model_dsnn_config import  epoch_multiple,device





if (len(sys.argv) != 4):
    print("wrong number of arguments.")
    exit(21)


num_layers = int(sys.argv[1])
K=int(sys.argv[2])
num_neurons=int(sys.argv[3])
data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
fileNameTrain=data_inDir+"/inf_range.train.pkl"

with open(fileNameTrain,"rb") as fptr:
    X_train, Y_train = pickle.load(fptr)


num_sample,num_spins=X_train.shape
print(f"num_sample={num_sample}")
num_epochs =int(num_sample/batch_size*epoch_multiple)
print(f"num_epochs={num_epochs}")
X_train = torch.tensor(X_train, dtype=torch.float64).to(device)  # Move to device
Y_train = torch.tensor(Y_train, dtype=torch.float64).view(-1, 1).to(device)

# Create Dataset and DataLoader
dataset = CustomDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DSNN(num_spins=num_spins, num_layers=num_layers, num_neurons=num_neurons).double().to(device)


# Define loss function and optimizer with L2 regularization
# Optimizer and loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every step_size epochs

scheduler = StepLR(optimizer, step_size=decrease_over, gamma=decrease_rate)
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}"

out_model_Dir=f"./out_model_L{L}_K{K}_r{r}/layer{num_layers}/neurons{num_neurons}/"
Path(out_model_Dir).mkdir(exist_ok=True,parents=True)
loss_file_content=[]

print(f"num_spin={num_spins}")
print(f"num_neurons={num_neurons}")
# Training loop
tTrainStart = datetime.now()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    # dataCount=0
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move batch to device
        nRow,_=X_batch.shape
        # dataCount+=nRow
        # Forward pass
        predictions = model(X_batch)

        # Compute loss
        loss = criterion(predictions, Y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate batch loss
        epoch_loss += loss.item() * X_batch.size(0)

    # Average loss over total samples
    average_loss = epoch_loss / len(dataset)

    # Print and log epoch summary
    # print(f"dataCount={dataCount}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    loss_file_content.append(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.8f}\n")

    # Update the learning rate
    scheduler.step()

    # Optionally print the current learning rate
    current_lr = scheduler.get_last_lr()[0]
    print(f"Learning Rate after Epoch {epoch + 1}: {current_lr:.8e}")

# Save the loss log
with open(out_model_Dir + f"/DSNN_training_log{suffix_str}.txt", "w+") as fptr:
    fptr.writelines(loss_file_content)

# Save the model
torch.save(model.state_dict(), out_model_Dir + f"/DSNN_model{suffix_str}.pth")
tTrainEnd = datetime.now()

print("Training time:", tTrainEnd - tTrainStart)