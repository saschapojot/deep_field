import re
import matplotlib.pyplot as plt
import numpy as np
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
abs_mean_vec=np.abs(mean_vec)#E_true
abs_pred_mean_vec=np.abs(pred_mean_vec)#E_pred
# print(abs_mean_vec)
# print(abs_pred_mean_vec)

N_vec_to_fit=N_vec

def model_E_vs_N(x, a, b, c):
    return a * x**b + c
#fit true
initial_guess = [1, 1, 1]
params_true,covariance_true=curve_fit(model_E_vs_N,N_vec_to_fit,abs_mean_vec,initial_guess)
alpha,beta,c_true=params_true
# e0Tmp=alpha*10**beta+c_true
# print(f"e0Tmp={e0Tmp}")
# print(f"covariance_true={covariance_true}")
print(f'alpha={alpha}, beta={beta}, c_true={c_true}')

#plot fitted curve for E_true
plt_fit_data_num=100
plt_true_N_vec_to_fit=np.linspace(np.min(N_vec_to_fit),np.max(N_vec_to_fit),plt_fit_data_num)

plt_fit_E_true=model_E_vs_N(plt_true_N_vec_to_fit,alpha,beta,c_true)

#fit pred
params_pred,covariance_pred=curve_fit(model_E_vs_N,N_vec_to_fit,abs_pred_mean_vec,initial_guess)
gamma,delta,c_pred=params_pred
# print(f"covariance_pred={covariance_pred}")
print(f'gamma={gamma}, delta={delta}, c_pred={c_pred}')
# e1Tmp=gamma*10**delta+c_pred
# print(f"e1Tmp={e1Tmp}")
print(f"delta-beta={delta-beta}")
#plot fitted curve for E_true
plt_pred_N_vec_to_fit=np.linspace(np.min(N_vec_to_fit),np.max(N_vec_to_fit),plt_fit_data_num)
plt_fit_E_pred=model_E_vs_N(plt_pred_N_vec_to_fit,gamma,delta,c_pred)

#E_true,E_pred, E_true_fit,E_pred_fit
plt.figure()
# plt.plot(N_vec,abs_mean_vec,color="green",linestyle="dashed",linewidth=2,label="data")
plt.scatter(N_vec,abs_mean_vec,color="green",label="E_true")#E_true
plt.plot(plt_true_N_vec_to_fit,plt_fit_E_true,color="lime",linestyle='--',label="E_true_fit",)#E_true fit
# plt.plot(N_vec,abs_pred_mean_vec,color="red",linestyle="dashed",linewidth=2,label="data")
plt.scatter(N_vec,abs_pred_mean_vec,color="red",label="E_pred")#E_pred
plt.plot(plt_pred_N_vec_to_fit,plt_fit_E_pred,color="magenta",linestyle="-.",label="E_pred_fit")#E_pred fit
plt.title("|mean| vs $N$ in data and pred")
plt.xlabel("$N$")
plt.ylabel("|mean|")
plt.legend(loc="best")
plt.savefig(inDirRoot+"/mean_vs_N.png")
plt.close()


#N^power
plt.figure()
sqrt_abs_mean_vec=(abs_mean_vec)**0.4
plt.plot(N_vec,sqrt_abs_mean_vec,color="red",linestyle="dashed",linewidth=2,label="data")
plt.scatter(N_vec,sqrt_abs_mean_vec,color="red")

plt.title("|mean| vs $N$ in data and pred")
plt.xlabel("$N$")
# plt.ylabel("|mean|")
plt.legend(loc="best")
plt.savefig(inDirRoot+"/mean_pow_vs_N.png")
plt.close()