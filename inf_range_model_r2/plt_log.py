import numpy as np
import re
import matplotlib.pyplot as plt
# from model_dsnn_train import L,r,num_layers,decrease_over
# System Parameters
L = 14  # Number of spins
r = 2   # Number of spins in each interaction term
num_layers = 4  # Number of DSNN layers
in_model_Dir=f"./out_model_L{L}_r{r}_layer{num_layers}/"
decrease_over=150
log_fileName=in_model_Dir + "/training_log.txt"

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
plt.plot(loss_vec, label="Loss", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.title("Training Loss Over Epochs", fontsize=14)
# plt.grid(True)
plt.legend(fontsize=12)
# Add vertical dotted lines every step_size steps
for step in range(0, len(loss_vec), decrease_over):
    plt.axvline(x=step, color='pink', linestyle='dotted', linewidth=1.2)
plt.tight_layout()
plt.savefig(in_model_Dir+"/loss.png")