import re
import matplotlib.pyplot as plt
import numpy as np
from model_dsnn_config import *
import pandas as pd

#this script plots relative error for different layer number for DSNN, and DNN
# for generic r

layerNumVec=[1,2,3]

def layer_2_fileVec(layer):
    model_inDir = f"./out_model_L{L}_K{K}_r{r}_layer{layer}/"
    DNN_file_name = model_inDir + "/test_DNN.txt"
    DSNN_file_name = model_inDir + "/test_DSNN.txt"

    return DNN_file_name, DSNN_file_name


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

DNN_file_vec=[]
DSNN_file_vec=[]
for layer in layerNumVec:
    DNN_file_name,DSNN_file_name=layer_2_fileVec(layer)
    DNN_file_vec.append(DNN_file_name)
    DSNN_file_vec.append(DSNN_file_name)

DNN_std_loss_vec=[]

DSNN_std_loss_vec=[]

for file in DNN_file_vec:
    tmp=file_2_std(file)
    DNN_std_loss_vec.append(tmp)


for file in DSNN_file_vec:
    tmp=file_2_std(file)
    DSNN_std_loss_vec.append(tmp)

DNN_std_loss_vec=np.array(DNN_std_loss_vec)
DSNN_std_loss_vec=np.array(DSNN_std_loss_vec)

data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
fileNameTrain=data_inDir+"/inf_range.train.pkl"

with open(fileNameTrain,"rb") as fptr:
    X_train, Y_train = pickle.load(fptr)

Y_train_mean=np.abs(np.mean(Y_train))

relative_DNN_vec=DNN_std_loss_vec/np.abs(Y_train_mean)

relative_DSNN_vec=DSNN_std_loss_vec/np.abs(Y_train_mean)

plt.figure()
plt.scatter(layerNumVec,relative_DSNN_vec,color="blue",marker="o",label="EFNN")
plt.plot(layerNumVec,relative_DSNN_vec,color="blue",linestyle="dashed")

plt.scatter(layerNumVec,relative_DNN_vec,color="red",marker="s",label="DNN")
plt.plot(layerNumVec,relative_DNN_vec,color="red",linestyle="dashed")
plt.yscale("log")

custom_yticks = [0.06, 0.1, 1]
plt.yticks(custom_yticks, labels=["0.06", "0.1", "1"])
plt.xticks([1,2,3])
plt.legend(loc="best")
plt.xlabel("Number of FP layers")
plt.ylabel("Relative error",fontsize=14)
plt.title("EFNN vs DNN")
plt.savefig(f"./loss_relative_L{L}_K{K}_r{r}.pdf")


csvOutDir="../more_neurons_inf_range_general_r/compare_neuron_num/"

csv_fileName=csvOutDir+"/efnn_dnn_compare.csv"

out_df=pd.DataFrame({"layerNum":layerNumVec,"efnn":relative_DSNN_vec,"dnn":relative_DNN_vec})

out_df.to_csv(csv_fileName,index=False)