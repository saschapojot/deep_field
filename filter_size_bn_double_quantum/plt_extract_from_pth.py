import glob
import re
import matplotlib.pyplot as plt
from model_qt_dsnn_config import *

#this script extracts information from checkpoint pth files and plots

N=10
C=30
#layer
step_num_after_S1=2

decrease_over = 50

decrease_rate = 0.9

# num_epochs = 1200
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
sampleNum=200000

pth_file_vec=[]

pth_dir=f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"
for file in glob.glob(pth_dir+"*.pth"):
    pth_file_vec.append(file)

pattern_epoch=r"epoch(\d+)"
def extract_epoch(fileName):
    match_epoch=re.search(pattern_epoch,fileName)
    if match_epoch:
        return int(match_epoch.group(1))
    else:
        print(f"format error: {fileName}")
        exit(2)

epoch_vec=[extract_epoch(file) for file in pth_file_vec]

ind_vec=np.argsort(epoch_vec)

sorted_epoch_vec=[epoch_vec[j] for j in ind_vec]
sorted_pth_file_vec=[pth_file_vec[j] for j in ind_vec]

def extract_loss(pth_file_name):
    checkpoint = torch.load(pth_file_name, map_location=device)
    return checkpoint['loss']

loss_vec=[extract_loss(file) for file in sorted_pth_file_vec]

plt.plot(sorted_epoch_vec,loss_vec, label="Loss", linewidth=2)
y_intersection=loss_vec[-1]
plt.axhline(y=y_intersection, color='r', linestyle='--', label=f'{y_intersection}')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.title(f"Training Loss Over Epochs, N={N}, C={C}, layer={step_num_after_S1}", fontsize=14)
plt.legend(fontsize=12)
# Add vertical dotted lines every step_size steps
for step in range(0, np.max(sorted_epoch_vec), decrease_over):
    plt.axvline(x=step, color='pink', linestyle='dotted', linewidth=1.2)
plt.tight_layout()
plt.savefig(pth_dir+f"/loss_over_{np.max(sorted_epoch_vec)}epochs.png")