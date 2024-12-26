import numpy as np
import re
import matplotlib.pyplot as plt
from more_neurons_model_dsnn_config import  decrease_rate,batch_size,learning_rate,weight_decay
from more_neurons_model_dsnn_config import L,r,decrease_over,K

from pathlib import Path
import pandas as pd

#this script compares performance of dsnn with different neuron number, and extracts test results

neuron_num_vec=range(15,205,5)
layer_num=3
def neuron_num_2_file(neuron_num):
    in_model_dir = f"./out_model_L{L}_K{K}_r{r}_layer{layer_num}_neurons{neuron_num}/"
    testFileName=in_model_dir+f"/test_DSNN.txt"
    return testFileName

pattern_std=r'std_loss\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
def file_2_std(test_fileName):
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()
    # print(line)
    match_std_loss=re.search(pattern_std,line)
    if match_std_loss:
        return float(match_std_loss.group(1))
    else:
        print(f"{test_fileName}, format error")
        exit(12)



outPath="./compare_neuron_num/"
Path(outPath).mkdir(parents=True, exist_ok=True)

fileName_vec=[neuron_num_2_file(neuron_num) for neuron_num in neuron_num_vec]

std_loss_vec=[file_2_std(file ) for file in fileName_vec]



out_df=pd.DataFrame({'neuron_num':neuron_num_vec,'std_loss':std_loss_vec})

out_df.to_csv(outPath+"/std_loss.csv",index=False)

