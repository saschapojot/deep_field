#this script generates bash files

from pathlib import Path

import numpy as np

neuron_vec=[15+15*n for n in range(0,10)]

def content(layer_num,neuron_num,fileName):
    lines=[
    "#!/bin/bash\n",
        "#SBATCH -n 1\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-10:00\n",
        "#SBATCH -p CLUSTER\n",
        "#SBATCH --mem=40GB\n",
        f"#SBATCH -o out_dsnnTrain_neuron{neuron_num}_layer{layer_num}.out\n",
        f"#SBATCH -e out_dsnnTrain_neuron{neuron_num}_layer{layer_num}.err\n",
        f"cd /home/cywanag/data/hpc/cywanag/liuxi/Document/pyCode/deep_field/more_neurons_inf_range_general_r\n",

        f"python3  -u  more_neurons_model_dsnn_train.py {layer_num} 455 {neuron_num}"
    ]
    with open(fileName,"w") as fptr:
        fptr.writelines(lines)
layer=3
outPath=f"./bashFiles/layer{layer}"
Path(outPath).mkdir(parents=True, exist_ok=True)

for neuron_num in neuron_vec:
    outFileName=outPath+f"/more_dsnn_train_neuron{neuron_num}_layer{layer}.sh"
    content(layer,neuron_num,outFileName)

