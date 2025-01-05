import re
import matplotlib.pyplot as plt
import numpy as np


from model_qt_dsnn_config import *


#this script plots the test performance of model trained on 10 by 10 lattice on larger lattices

layer_num=2
C=30

N_vec=[10,15,20,25,30]

inDirRoot="./larger_lattice_test_performance/"

def N_2_test_file(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file=in_model_dir+f"/test_over50_rate0.6_epoch1000_num_samples40000.txt"
    return test_txt_file

def N_2_test_data_pkl(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    pkl_test_file=in_model_dir+"/db.test_num_samples40000.pkl"
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
    abs_avg_Y_train_vec.append(absTmp)

std_loss_vec=np.array(std_loss_vec)
abs_avg_Y_train=np.array(abs_avg_Y_train_vec)

relative_error=std_loss_vec/abs_avg_Y_train


plt.figure()
plt.scatter(N_vec,relative_error,color="blue",marker="o")
plt.plot(N_vec,relative_error,color="blue",linestyle="dashed")
plt.xlabel("Lattice size")
plt.ylabel("Relative error")
plt.xticks(N_vec)
plt.title(r"Performance of $10\times 10$ model on larger lattice")
plt.savefig(inDirRoot+"/larger_lattice_test_performance.png")


plt.close()