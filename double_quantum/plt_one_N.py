import re
import matplotlib.pyplot as plt
import numpy as np


from model_qt_dsnn_config import *

#this script compares test loss for the same N, different layer numbers, different C values

N=10

decrease_over = 50
decrease_rate = 0.6
num_epochs = 1000

#layer-1
step_num_after_S1_vec=[1,3,5]
C_vec=[10,20,30,40]

decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}"

def C_2_test_file(step_num_after_S1,C):
    in_model_dir = f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"
    test_fileName = in_model_dir + f"/test{suffix_str}.txt"
    return test_fileName




pattern_std=r'std_loss=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
def file_2_std(test_fileName):
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()

    match_std_loss=re.search(pattern_std,line)
    if match_std_loss:
        return float(match_std_loss.group(1))
    else:
        print("format error")
        exit(12)

inDir=f"./train_test_data/N{N}/"
in_pkl_train_file=inDir+"/db.train.pkl"
with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)

Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

Y_train_avg=np.mean(Y_train_array)

abs_Y_train_avg=np.abs(Y_train_avg)

std_vec_for_each_layer_num=[]
for layer in step_num_after_S1_vec:
    vecTmp=[]
    for C in C_vec:
        fileNameTmp=C_2_test_file(layer,C)
        vecTmp.append(file_2_std(fileNameTmp))
    std_vec_for_each_layer_num.append(vecTmp)

#row: same layer number, different C
relative_acc=np.array(std_vec_for_each_layer_num)/abs_Y_train_avg

out_pic_dir=f"./compare/N{N}/"

plt.figure()

ind0=0
plt.scatter(C_vec,relative_acc[ind0,:],color="blue",marker="o",label=f"n={step_num_after_S1_vec[ind0]+1}")
plt.plot(C_vec,relative_acc[ind0,:],color="blue",linestyle="dashed")

ind1=1
plt.scatter(C_vec,relative_acc[ind1,:],color="red",marker="^",label=f"n={step_num_after_S1_vec[ind1]+1}")
plt.plot(C_vec,relative_acc[ind1,:],color="red",linestyle="dashed")

ind1=2
plt.scatter(C_vec,relative_acc[ind1,:],color="green",marker="s",label=f"n={step_num_after_S1_vec[ind1]+1}")
plt.plot(C_vec,relative_acc[ind1,:],color="green",linestyle="dashed")

plt.xlabel("C value")
plt.ylabel("Relative error")
# Move Y-axis label to the right
plt.gca().yaxis.set_label_position("right")  # Move label to the right
plt.legend(loc="best")
plt.savefig(out_pic_dir+f"N{N}.png")