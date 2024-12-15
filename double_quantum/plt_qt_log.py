
import re
import matplotlib.pyplot as plt
from model_qt_dsnn_config import *

#this script plots training loss

in_model_dir=f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"

log_fileName=in_model_dir +"/training_log.txt"


# Open the file and read each line
with open(log_fileName, 'r') as file:
    lines = file.readlines()

lines=[line.strip() for line in lines]
pattern_float=r'Loss:\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'

loss_vec=[]
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
plt.plot(epoch_vec[truncate_at:],loss_vec[truncate_at:], label="Loss", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.title(f"Training Loss Over Epochs", fontsize=14)
# plt.grid(True)
plt.legend(fontsize=12)
# Add vertical dotted lines every step_size steps
for step in range(0, len(loss_vec), decrease_over):
    plt.axvline(x=step, color='pink', linestyle='dotted', linewidth=1.2)
plt.tight_layout()
plt.savefig(in_model_dir+f"/loss.png")