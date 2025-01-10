import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_dsnn_config import *

# this script produces the efnn vs dnn csv

layerNumVec=np.array([1,2,3])
num_epochs=8000
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}"
def layer2file(layer):
    in_model_dir = f"./out_model_L{L}_K{K}_r{r}_layer{layer}/"
    DNN_file_name = in_model_dir + "/test_DNN.txt"
    DSNN_file_name = in_model_dir + f"/test_DSNN_over{decrease_overStr}_rate{decrease_rateStr}_epoch8000.txt"
    return DNN_file_name, DSNN_file_name



pattern_std=r'std_loss\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
def file_2_std(test_fileName):
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()
    # print(line)

    match_std_loss=re.search(pattern_std,line)
    if match_std_loss:
        return float(match_std_loss.group(1))
    else:
        print("format error")
        exit(12)

DNN_file_vec=[]
DSNN_file_vec=[]
for layer in layerNumVec:
    DNN_file_name,DSNN_file_name=layer2file(layer)
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
    x_train,y_train=pickle.load(fptr)

y_train_mean=np.mean(y_train)

relative_DNN_vec=DNN_std_loss_vec/np.abs(y_train_mean)

relative_DSNN_vec=DSNN_std_loss_vec/np.abs(y_train_mean)
print(f"relative_DSNN_vec={relative_DSNN_vec}")

outCsvDir="./efnn_vs_dnn_csv/"
Path(outCsvDir).mkdir(parents=True, exist_ok=True)

out_df=pd.DataFrame(
    {
        "layerNum":layerNumVec,
        "efnn":relative_DSNN_vec,
        "dnn":relative_DNN_vec,
    }
)

out_df.to_csv(outCsvDir+"efnn_vs_dnn.csv",index=False)