import numpy as np
import re
import matplotlib.pyplot as plt
from model_dsnn_config import  num_neurons,decrease_rate,batch_size,learning_rate,weight_decay
from model_dsnn_config import L,r,decrease_over,K,format_using_decimal
# System Parameters
# L = 15  # Number of spins
# r = 2   # Number of spins in each interaction term
# num_layers = 8  # Number of DSNN layers
num_layers=3
num_epochs=8000
in_model_Dir=f"./out_model_L{L}_K{K}_r{r}_layer{num_layers}/"
# decrease_over=150
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}"
name="DSNN"
# log_fileName=in_model_Dir + f"/{name}_training_log.txt"
log_fileName=in_model_Dir+f"/{name}_training_log{suffix_str}.txt"
# Open the file and read each line
with open(log_fileName, 'r') as file:
    lines = file.readlines()

lines=[line.strip() for line in lines]
pattern_float=r'Loss:\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'

loss_vec=[]
# counter=0

for one_line in lines:
    # print(one_line)
    match_loss=re.search(pattern_float,one_line)
    if match_loss:
        loss_vec.append(float(match_loss.group(1)))
        # counter+=1
        # print(counter)
    else:
        print("format error")
        exit(12)

print(f"len(loss_vec)={len(loss_vec)}")
# Plotting the loss values
plt.figure(figsize=(10, 6))
epoch_vec=list(range(0,len(loss_vec)))
truncate_at=0
plt.figure()
plt.plot(epoch_vec[truncate_at:],loss_vec[truncate_at:], label="Loss", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
y_intersection=loss_vec[-1]
plt.axhline(y=y_intersection, color='r', linestyle='--', label=f'{y_intersection}')
plt.xscale("log")
plt.yscale("log")
plt.title(f"{name}: Training Loss Over Epochs", fontsize=14)
# plt.grid(True)
plt.legend(fontsize=12,loc="best")
# Add vertical dotted lines every step_size steps
# for step in range(0, len(loss_vec), decrease_over):
#     plt.axvline(x=step, color='pink', linestyle='dotted', linewidth=1.2)
plt.tight_layout()
plt.savefig(in_model_Dir+f"/{name}_loss.png")