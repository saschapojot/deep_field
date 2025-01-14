import re
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib as mpl
from model_qt_dsnn_config import *
# plt.rcParams['font.family'] = 'DejaVu Sans'  # Or any desired font
# mpl.rc('text', usetex=True)
# mpl.rc('font', family='serif', serif=['Computer Modern'])
mpl.rcParams['axes.linewidth'] = 2.5  # Set for all plots
#this script plots the test performance of model trained on 10 by 10 lattice on larger lattices
#for all layers

layer_vec=np.array([0,1,2])
C=10
rate=0.9
N_vec=np.array([10,15,20,25,30,35,40])
num_suffix=40000
num_epochs = 25#optimal is 25

inDirRoot="./larger_lattice_test_performance/"
def N_2_test_file(N,layer_num):
    in_file_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file = in_file_dir + f"/custom_err_test_over50_rate{rate}_epoch{num_epochs}_num_samples{num_suffix}.txt"
    return test_txt_file


def N_2_test_data_pkl(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/"
    pkl_test_file=in_model_dir+f"/db.test_num_samples{num_suffix}.pkl"
    return pkl_test_file

pattern_std=r'std_loss=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
pattern_custom_err=r'custom_err\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'


def file_2_std(test_fileName):
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()
    # print(line)
    match_std_loss=re.search(pattern_std,line)
    match_custom_err=re.search(pattern_custom_err,line)
    if match_std_loss:
        std_loss= float(match_std_loss.group(1))
    else:
        print("format error")
        exit(12)
    if match_custom_err:
        custom_err=float(match_custom_err.group(1))
    return std_loss,custom_err


pkl_file_vec=[N_2_test_data_pkl(N) for N in N_vec]
abs_avg_Y_train_vec=[]
for j in range(0,len(pkl_file_vec)):
    file_pkl_tmp=pkl_file_vec[j]
    with open(file_pkl_tmp,"rb") as fptr:
        X_train_tmp,Y_train_tmp=pickle.load(fptr)
    Y_train_tmp=np.array(Y_train_tmp)
    absTmp=np.abs(np.mean(Y_train_tmp))
    # absTmp=np.std(Y_train_tmp)
    abs_avg_Y_train_vec.append(absTmp)

# print(abs_avg_Y_train_vec)
abs_avg_Y_train_vec=np.array(abs_avg_Y_train_vec)
#layer 0
layer0=layer_vec[0]
file_vec_layer0=[N_2_test_file(N,layer0) for N in N_vec]
std_loss_vec_layer0=[]
custom_err_vec_layer0=[]
for file in file_vec_layer0:
    std_loss,custom_err=file_2_std(file)
    std_loss_vec_layer0.append(std_loss)
    custom_err_vec_layer0.append(custom_err)

std_loss_vec_layer0=np.array(std_loss_vec_layer0)

custom_err_vec_layer0=np.array(custom_err_vec_layer0)
relative_error_layer0=(std_loss_vec_layer0/abs_avg_Y_train_vec)

# print(f"relative_error_layer0={relative_error_layer0}")
# print(f'custom_err_vec_layer0={custom_err_vec_layer0}')

# layer1
layer1=layer_vec[1]
file_vec_layer1=[N_2_test_file(N,layer1) for N in N_vec]
std_loss_vec_layer1=[]
custom_err_vec_layer1=[]
for file in file_vec_layer1:
    std_loss,custom_err=file_2_std(file)
    std_loss_vec_layer1.append(std_loss)
    custom_err_vec_layer1.append(custom_err)

std_loss_vec_layer1=np.array(std_loss_vec_layer1)

custom_err_vec_layer1=np.array(custom_err_vec_layer1)
relative_error_layer1=(std_loss_vec_layer1/abs_avg_Y_train_vec)

# print(relative_error_layer1)
# print(custom_err_vec_layer1)

# layer2
layer2=layer_vec[2]
file_vec_layer2=[N_2_test_file(N,layer2) for N in N_vec]
std_loss_vec_layer2=[]
custom_err_vec_layer2=[]
for file in file_vec_layer2:
    std_loss,custom_err=file_2_std(file)
    std_loss_vec_layer2.append(std_loss)
    custom_err_vec_layer2.append(custom_err)

std_loss_vec_layer2=np.array(std_loss_vec_layer2)

custom_err_vec_layer2=np.array(custom_err_vec_layer2)
relative_error_layer2=(std_loss_vec_layer2/abs_avg_Y_train_vec)

# print(relative_error_layer2)
# print(custom_err_vec_layer2)
width=6
height=8
textSize=33
yTickSize=33
xTickSize=33
legend_fontsize=24
lineWidth1=3
marker_size1=100
out_db_qt_dir="../fig_qt/"
Path(out_db_qt_dir).mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(width, height))

# Plot the data
#layer0
plt.scatter(N_vec,relative_error_layer0,color="blue",marker="o",s=marker_size1,label=f"EFNN, n={layer0+1}")
plt.plot(N_vec,relative_error_layer0,color="blue",linestyle="dashed",linewidth=lineWidth1)

#layer 1
plt.scatter(N_vec,relative_error_layer1,color="magenta",marker="^",s=marker_size1,label=f"EFNN, n={layer1+1}")
plt.plot(N_vec,relative_error_layer1,color="magenta",linestyle="dashed",linewidth=lineWidth1)

#layer 2
plt.scatter(N_vec,relative_error_layer2,color="green",marker="s",s=marker_size1,label=f"EFNN, n={layer2+1}")
plt.plot(N_vec,relative_error_layer2,color="green",linestyle="dashed",linewidth=lineWidth1)

plt.xlabel("$N$", fontsize=textSize)
plt.ylabel("Relative error",fontsize=textSize)
plt.yscale("log")
plt.xticks([10, 20, 30, 40], ["10", "20", "30", "40"], fontsize=xTickSize)
plt.yticks([1,0.1, 0.01], labels=["1", "0.1", "0.01"], fontsize=yTickSize)
# Adjust layout to fit y-label on the right and remove extra space
plt.tight_layout(rect=[0, 0, 1, 1])  # Prevent truncation and ensure the full figure fits
plt.legend(loc="upper right", bbox_to_anchor=(0.9, 0.8), fontsize=legend_fontsize)
plt.gca().yaxis.set_label_position("right")  # Move y-axis label to the right
# Save the figure
plt.savefig(out_db_qt_dir + "/custom_error.svg", bbox_inches="tight", dpi=300)  # bbox_inches ensures no truncation
plt.close()
