import re
import matplotlib.pyplot as plt
import numpy as np
from sympy.codegen.ast import continue_

from model_qt_dsnn_config import *
import matplotlib as mpl


#this script compares test loss for the same N, same epoch, different layer numbers, different C values
#this script needs to manually input lin's mse, from slurm output on supercomputer

N=10
mpl.rcParams['axes.linewidth'] = 2.5  # Set for all plots
decrease_over = 50
decrease_rate = 0.9
step_num_after_S1_vec=[0,1,2]
C_vec=[10,15,20,25,30]


decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)

epoch_pattern=r"num_epochs=(\d+)"
std_pattern=r"std_loss=([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)"


def match_line_in_file(test_outFile,epoch_num):
    with open(test_outFile,'r') as fptr:
        contents=fptr.readlines()
    for line in contents:
        # print(line)
        match_epoch=re.search(epoch_pattern,line)
        if match_epoch:
            epoch_in_file=int(match_epoch.group(1))
            if epoch_in_file==epoch_num:
                match_std = re.search(std_pattern,line)
                if match_std:
                    # print(epoch_in_file)
                    # print(float(match_std.group(1)))
                    return epoch_in_file, float(match_std.group(1))
            else:
                continue

# oneFile="./out_model_data/N10/C15/layer1/test_over_epochs.txt"
# ep,std=match_line_in_file(oneFile,25)
# print(f"ep={ep}, std={std}")


def std_loss_all_one_epoch(epoch_num,layer,N,C_vec):
    ret_std_loss_vec=[]
    for C in C_vec:
        oneFile=f"./out_model_data/N{N}/C{C}/layer{layer}/test_over_epochs.txt"
        ep, stdTmp = match_line_in_file(oneFile, epoch_num)
        # print(f"ep={ep},C={C},layer={layer},sdTmp={stdTmp}")
        ret_std_loss_vec.append(stdTmp)
    return np.array(ret_std_loss_vec)



inDir=f"./train_test_data/N{N}/"
in_pkl_train_file=inDir+"/db.train_num_samples200000.pkl"
with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)
Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

Y_train_avg=np.mean(Y_train_array)

abs_Y_train_avg=np.abs(Y_train_avg)

print(f"Y_train_array.shape={Y_train_array.shape}")
set_epoch=300

layer0=step_num_after_S1_vec[0]
std_for_layer0=std_loss_all_one_epoch(set_epoch,layer0,N,C_vec)

layer1=step_num_after_S1_vec[1]
std_for_layer1=std_loss_all_one_epoch(set_epoch,layer1,N,C_vec)

layer2=step_num_after_S1_vec[2]
std_for_layer2=std_loss_all_one_epoch(set_epoch,layer2,N,C_vec)


relative_acc_layer0=std_for_layer0/abs_Y_train_avg
relative_acc_layer1=std_for_layer1/abs_Y_train_avg
relative_acc_layer2=std_for_layer2/abs_Y_train_avg
out_pic_dir="./compare/"
Path(out_pic_dir).mkdir(parents=True, exist_ok=True)
width=6
height=8
textSize=33
yTickSize=33
xTickSize=33
legend_fontsize=20
lineWidth1=3
marker_size1=100
plt.figure(figsize=(width, height))
plt.scatter(C_vec,relative_acc_layer0,color="blue",marker="o",s=marker_size1,label=f"EFNN, n={step_num_after_S1_vec[layer0]+1}")
plt.plot(C_vec,relative_acc_layer0,color="blue",linestyle="dashed",linewidth=lineWidth1)

plt.scatter(C_vec,relative_acc_layer1,color="magenta",marker="^",s=marker_size1,label=f"EFNN, n={step_num_after_S1_vec[layer1]+1}")
plt.plot(C_vec,relative_acc_layer1,color="magenta",linestyle="dashed",linewidth=lineWidth1)
plt.yscale("log")
plt.xticks(C_vec)

plt.scatter(C_vec,relative_acc_layer2,color="green",marker="s",s=marker_size1,label=f"EFNN, n={step_num_after_S1_vec[layer2]+1}")
plt.plot(C_vec,relative_acc_layer2,color="green",linestyle="dashed",linewidth=lineWidth1)


lin_mean_mse=0.9847254886017068
lin_mean_std=np.sqrt(lin_mean_mse)
lin_err_relative=lin_mean_std/abs_Y_train_avg

print(f"lin_err_relative={lin_err_relative}")
plt.axhline(y=lin_err_relative, color="black", linestyle="--", label=f"Effective model",linewidth=lineWidth1)
plt.xlabel("Channel number",fontsize=textSize)
plt.ylabel("Relative error",fontsize=textSize)
plt.gca().yaxis.set_label_position("right")  # Move label to the right
plt.legend(loc="upper right", bbox_to_anchor=(0.9, 0.8), fontsize=legend_fontsize)
plt.title(f"epoch={set_epoch}")
plt.tight_layout()
plt.savefig(out_pic_dir+f"epoch_{set_epoch}_N{N}.svg")