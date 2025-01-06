import re
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from model_qt_dsnn_config import *

#this script fits statistics of mean
layer_num=0
C=25
rate=0.9
N_vec=[10,15,20,25,30,35]
num_suffix=40000
num_epochs = 1000
inDirRoot="./larger_lattice_test_performance/"


def N_2_stats_file(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    text_file=in_model_dir+"/stats.txt"
    test_out_file=in_model_dir+f"/test_over50_rate{rate}_epoch{num_epochs}_num_samples{num_suffix}.txt"
    return text_file,test_out_file


pattern_N=r'N\s*=\s*(\d+)'
pattern_mean=r'mean\s*=\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
pattern_std=r'std\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'

def file_2_stats(text_file):
    #for stats.txt
    with open(text_file,"r") as fptr:
        line = fptr.readline()
    match_N=re.search(pattern_N,line)
    match_mean=re.search(pattern_mean,line)
    match_std=re.search(pattern_std,line)

    if match_N:
        N_val=int(match_N.group(1))
    if match_mean:
        mean=float(match_mean.group(1))
    if match_std:
        std=float(match_std.group(1))

    return N_val,mean,std

pattern_pred_mean=r'pred_mean\s*=\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
pattern_pred_std=r'pred_std\s*=\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
def extract_test_pred(test_file_name):
    with open(test_file_name, "r") as fptr:
        line = fptr.readline()
    match_pred_mean=re.search(pattern_pred_mean,line)
    match_pred_std=re.search(pattern_pred_std,line)
    if match_pred_mean:
        pred_mean=float(match_pred_mean.group(1))
    if match_pred_std:
        pred_std=float(match_pred_std.group(1))
    return pred_mean,pred_std

file_vec=[]
test_file_vec=[]
for N in N_vec:
    text_file, test_out_file=N_2_stats_file(N)
    file_vec.append(text_file)
    test_file_vec.append(test_out_file)

mean_vec=[]
std_vec=[]
for file in file_vec:
    N_val, mean, std=file_2_stats(file)
    mean_vec.append(mean)
    std_vec.append(std)
pred_mean_vec=[]
pred_std_vec=[]
for file in test_file_vec:
    pred_mean, pred_std = extract_test_pred(file)
    pred_mean_vec.append(pred_mean)
    pred_std_vec.append(pred_std)

N_vec=np.array(N_vec)
mean_vec=np.array(mean_vec)
std_vec=np.array(std_vec)
abs_mean_vec=np.abs(mean_vec)
abs_pred_mean_vec=np.abs(pred_mean_vec)

plt.figure()
plt.plot(N_vec,abs_mean_vec,color="green",linestyle="dashed",linewidth=2,label="data")
plt.scatter(N_vec,abs_mean_vec,color="green")
plt.plot(N_vec,abs_pred_mean_vec,color="red",linestyle="dashed",linewidth=2,label="pred")
plt.scatter(N_vec,abs_pred_mean_vec,color="red")
plt.title("|mean| vs $N$ in data and pred")
plt.xlabel("$N$")
plt.ylabel("|mean|")
plt.legend(loc="best")
plt.savefig(inDirRoot+"/mean_vs_N.png")
plt.close()