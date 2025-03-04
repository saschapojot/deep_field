import re
import matplotlib.pyplot as plt
import numpy as np


from model_qt_dsnn_config import *
import matplotlib as mpl

# Enable LaTeX rendering
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif', serif=['Computer Modern'])
#this script compares test loss for the same N, different layer numbers, different C values
#this script needs to manually input lin's mse, from slurm output on supercomputer
N=10

decrease_over = 50
decrease_rate = 0.6
num_epochs = 1000

#layer-1
step_num_after_S1_vec=[0,1,2]
C_vec=[10,15,20,25,30,35,40]

decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples200000"

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
Path(out_pic_dir).mkdir(parents=True, exist_ok=True)
plt.figure()

ind0=0
plt.scatter(C_vec,relative_acc[ind0,:],color="blue",marker="o",label=f"EFNN, n={step_num_after_S1_vec[ind0]+1}")
plt.plot(C_vec,relative_acc[ind0,:],color="blue",linestyle="dashed")

ind1=1
plt.scatter(C_vec,relative_acc[ind1,:],color="magenta",marker="^",label=f"EFNN, n={step_num_after_S1_vec[ind1]+1}")
plt.plot(C_vec,relative_acc[ind1,:],color="magenta",linestyle="dashed")
plt.yscale("log")
plt.xticks(C_vec)
ind1=2
plt.scatter(C_vec,relative_acc[ind1,:],color="green",marker="s",label=f"EFNN, n={step_num_after_S1_vec[ind1]+1}")
plt.plot(C_vec,relative_acc[ind1,:],color="green",linestyle="dashed")
print(relative_acc[ind1,:])
# ind1=3
# plt.scatter(C_vec,relative_acc[ind1,:],color="grey",marker="+",label=f"n={step_num_after_S1_vec[ind1]+1}")
# plt.plot(C_vec,relative_acc[ind1,:],color="grey",linestyle="dashed")
plt.yticks([0.007,0.01,0.02],labels=[r"0.007", "0.01", "0.02"])

lin_mean_mse=0.9847254886017068
lin_mean_std=np.sqrt(lin_mean_mse)
lin_err_relative=lin_mean_std/abs_Y_train_avg

print(f"lin_err_relative={lin_err_relative}")
plt.axhline(y=lin_err_relative, color="cyan", linestyle="--", linewidth=1, label=f"Effective model")
plt.xlabel("Channel number",fontsize=19)
plt.ylabel("Relative error",fontsize=20)
plt.title("EFNN vs effective model",fontsize=25)
# Move Y-axis label to the right
plt.gca().yaxis.set_label_position("right")  # Move label to the right
plt.legend(loc="best")
plt.savefig(out_pic_dir+f"N{N}.pdf")

# print(np.sqrt(0.09)/abs_Y_train_avg)