
import re
import matplotlib.pyplot as plt
from model_qt_dsnn_config import *

#this script plots training loss
N=10
C=30
#layer
step_num_after_S1=2

decrease_over = 50

decrease_rate = 0.6

num_epochs = 1000
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
sampleNum=200000
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples{sampleNum}"
in_model_dir=f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"

log_fileName=in_model_dir +f"/training_log{suffix_str}.txt"


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

y_intersection=loss_vec[-1]
plt.axhline(y=y_intersection, color='r', linestyle='--', label=f'{y_intersection}')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.title(f"Training Loss Over Epochs, N={N}, C={C}, layer={step_num_after_S1}", fontsize=14)
# plt.grid(True)
plt.legend(fontsize=12)
# Add vertical dotted lines every step_size steps
for step in range(0, len(loss_vec), decrease_over):
    plt.axvline(x=step, color='pink', linestyle='dotted', linewidth=1.2)
plt.tight_layout()
plt.savefig(in_model_dir+f"/loss{suffix_str}.png")