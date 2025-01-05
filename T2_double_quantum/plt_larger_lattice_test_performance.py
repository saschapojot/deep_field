import re
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from model_qt_dsnn_config import *


#this script plots the test performance of model trained on 10 by 10 lattice on larger lattices

layer_num=0
C=25

N_vec=[10,15,20]
step_num_after_S1=0

decrease_over = 50

decrease_rate = 0.95

num_data_train_test=200000
num_epochs = 1200
inDirRoot="./larger_lattice_test_performance/"
num_suffix=40000
def N_2_test_file(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file=in_model_dir+f"/test_over{decrease_over}_rate{decrease_rate}_epoch{num_epochs}_num_samples{num_suffix}.txt"
    return test_txt_file

def N_2_test_data_pkl(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    pkl_test_file=in_model_dir+f"/db.test_num_samples{num_suffix}.pkl"
    return pkl_test_file

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



file_vec=[N_2_test_file(N) for N in N_vec]

std_loss_vec=[file_2_std(test_fileName) for test_fileName in file_vec]

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

std_loss_vec=np.array(std_loss_vec)
abs_avg_Y_train=np.array(abs_avg_Y_train_vec)
N_2_vec=np.array(N_vec)**2
relative_error=std_loss_vec/N_2_vec
N_vec=np.array(N_vec)
# N_vec=N_vec.reshape(-1, 1)
# log_N = np.log(N_vec).reshape(-1, 1)
log_relative_error = np.log(relative_error)



plt.figure()
plt.scatter(N_vec,relative_error,color="blue",marker="o")
plt.plot(N_vec,relative_error,color="blue",linestyle="dashed")
plt.xlabel("Lattice size")
plt.ylabel("Relative error")
plt.xticks(N_vec)
plt.title(r"Performance of $10\times 10$ model on larger lattice")
plt.savefig(inDirRoot+"/larger_lattice_test_performance.png")


plt.close()


