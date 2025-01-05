#this script generates bash files

from pathlib import Path

import numpy as np

neuron_vec=[65+5*n for n in range(0,28)]

def content(neuron_num,fileName):
    lines=[
    "#!/bin/bash\n",
        "#SBATCH -n 2\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-10:00\n",
        "#SBATCH -p CLUSTER\n",
        "#SBATCH --mem=40GB\n",
        f"#SBATCH -o out_dsnnTrain_{neuron_num}.out\n",
        f"#SBATCH -e out_dsnnTrain_{neuron_num}.err\n",
        f"cd /home/cywanag/data/hpc/cywanag/liuxi/Document/pyCode/deep_field/more_neurons_inf_range_general_r\n",

        f"python3  -u  more_neurons_model_dsnn_train.py 3 455 {neuron_num}"
    ]
    with open(fileName,"w") as fptr:
        fptr.writelines(lines)

outPath="./bashFiles/"
Path(outPath).mkdir(parents=True, exist_ok=True)

for neuron_num in neuron_vec:
    outFileName=outPath+f"/more_dsnn_train_neuron_{neuron_num}.sh"
    content(neuron_num,outFileName)

