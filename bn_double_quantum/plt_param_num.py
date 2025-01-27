import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

from model_qt_dsnn_config import *
import matplotlib as mpl

#this script extracts parameter number for all n, for all C

N=10
mpl.rcParams['axes.linewidth'] = 2.5  # Set for all plots
decrease_over = 50
decrease_rate = 0.9
step_num_after_S1_vec=[0,1,2]
step_num_after_S1_vec=np.array(step_num_after_S1_vec)
C_vec=[10,15,20,25,30]
epochNum=25
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)

total_params_pattern=r"total_params=(\d+)"
trainable_params_pattern=r"trainable_params=(\d+)"

def match_line_in_file(test_outFile):
    tot_num=-1
    trainable_num=-1
    with open(test_outFile,'r') as fptr:
        contents=fptr.readlines()
    for line in contents:
        match_total=re.search(total_params_pattern,line)
        if match_total:
            tot_num=int(match_total.group(1))
        match_trainable=re.search(trainable_params_pattern,line)
        if match_trainable:
            trainable_num=int(match_trainable.group(1))
    return tot_num, trainable_num

def param_nums_one_C_one_layer(epoch_num,N,C,layerNum):
    oneFile=f"./out_model_data/N{N}/C{C}/layer{layerNum}/test_over{decrease_over}_rate{decrease_rate}_epoch{epoch_num}_num_samples200000.txt"
    tot_num, trainable_num=match_line_in_file(oneFile)
    return tot_num, trainable_num


total_params_arr=np.zeros((len(C_vec),len(step_num_after_S1_vec)))
trainable_params_arr=np.zeros((len(C_vec),len(step_num_after_S1_vec)))

for i in range(0,len(C_vec)):
    for j in range(0,len(step_num_after_S1_vec)):
        C_tmp=C_vec[i]
        layer_tmp=step_num_after_S1_vec[j]
        tot_num_tmp, trainable_num_tmp=param_nums_one_C_one_layer(epochNum,N,C_tmp,layer_tmp)
        total_params_arr[i,j]=tot_num_tmp
        trainable_params_arr[i,j]=trainable_num_tmp


#we plot the total parameter number



outPath="../fig_params_num/"
Path(outPath).mkdir(exist_ok=True,parents=True)

#plot, for each C, all layer numbers
plt.figure()
for i in range(0,len(C_vec)):
    CTmp=C_vec[i]
    plt.plot(step_num_after_S1_vec+1,total_params_arr[i,:],label=f"C={CTmp}")
    plt.scatter(step_num_after_S1_vec+1,total_params_arr[i,:])

plt.title("Total param number vs layer number")
plt.ylabel("#")
plt.xlabel("layer number")
plt.legend(loc="best")
plt.savefig(outPath+"/each_C.png")
plt.close()
print(f"total_params_arr: {total_params_arr}")

###for each layer, for all C
def tot_param_num_vs_C(C,a,b,d):
    return a*C**2+b*C+d

plt.figure()
for j in range(0,len(step_num_after_S1_vec)):
    layer_tmp=step_num_after_S1_vec[j]+1
    plt.scatter(C_vec,total_params_arr[:,j],label=f"layer={layer_tmp+1}")
    params_tmp,cov_tmp=curve_fit(tot_param_num_vs_C,C_vec,total_params_arr[:,j],p0=[1,1,1])
    aTmp,bTmp,dTmp=params_tmp

    C_plt=np.linspace(np.min(C_vec),np.max(C_vec),100)
    y_plt=tot_param_num_vs_C(C_plt, aTmp, bTmp, dTmp)
    plt.plot(C_plt,y_plt)


plt.title("Total param number vs C")
plt.ylabel("#")
plt.xlabel("C")
plt.legend(loc="best")
plt.savefig(outPath+"/each_layer.png")
plt.close()