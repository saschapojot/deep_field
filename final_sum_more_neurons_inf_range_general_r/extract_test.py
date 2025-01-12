import numpy as np
import re
import matplotlib.pyplot as plt
from model_dsnn_config import  decrease_rate,batch_size,learning_rate,weight_decay
from model_dsnn_config import L,r,decrease_over,K,format_using_decimal

from pathlib import Path
import pandas as pd

#this script compares performance of dsnn with different neuron number, and extracts test results

neuron_num_vec=range(15,165,15)
layer_num_vec=[1,2,3]
num_epochs=8000
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)

suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}"
def neuron_num_layer_2_file(neuron_num,layer):
    in_model_dir = f"./out_model_L{L}_K{K}_r{r}/layer{layer}/neurons{neuron_num}/"
    testFileName=in_model_dir+f"/test_DSNN{suffix_str}.txt"
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



outPath="./compare_layer_neuron_num/"
Path(outPath).mkdir(parents=True, exist_ok=True)

# fileName_vec=[neuron_num_2_file(neuron_num) for neuron_num in neuron_num_vec]
#
# std_loss_vec=[file_2_std(file ) for file in fileName_vec]
#
#
#
# out_df=pd.DataFrame({'neuron_num':neuron_num_vec,'std_loss':std_loss_vec})
#
# out_df.to_csv(outPath+"/std_loss.csv",index=False)

for layer in layer_num_vec:
    fileName_vec=[neuron_num_layer_2_file(neuron_num ,layer) for neuron_num in neuron_num_vec]
    std_loss_vec=[file_2_std(file ) for file in fileName_vec]
    out_df=pd.DataFrame({"neuron_num":neuron_num_vec,"std_loss":std_loss_vec})
    out_df.to_csv(outPath+f"{layer}_std_loss.csv",index=False)



