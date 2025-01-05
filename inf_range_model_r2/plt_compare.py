import re
import matplotlib.pyplot as plt
import numpy as np


#this script plots relative error for different layer number for DSNN, and DNN

from model_dsnn_config import *

layerNumVec=[1,2,3]

def layer_2_fileVec(layer):
    in_model_dir=f"./out_model_L{L}_r{r}_layer{layer}/"

    DNN_file_name=in_model_dir+"/test_DNN.txt"
    DSNN_file_name=in_model_dir+"/test_DSNN.txt"

    return DNN_file_name,DSNN_file_name


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

data_inDir=f"./data_inf_range_model_L{L}_r{r}/"
fileNameTrain=data_inDir+"/inf_range.train.pkl"

with open(fileNameTrain,"rb") as fptr:
    x_train,y_train=pickle.load(fptr)

y_train_mean=np.mean(y_train)

relative_DNN_vec=DNN_std_loss_vec/np.abs(y_train_mean)

relative_DSNN_vec=DSNN_std_loss_vec/np.abs(y_train_mean)

plt.figure()
plt.scatter(layerNumVec,relative_DSNN_vec,color="blue",marker="o",label="DSNN")
plt.plot(layerNumVec,relative_DSNN_vec,color="blue",linestyle="dashed")

plt.scatter(layerNumVec,relative_DNN_vec,color="red",marker="s",label="DNN")
plt.plot(layerNumVec,relative_DNN_vec,color="red",linestyle="dashed")
plt.savefig("./loss_relative.png")