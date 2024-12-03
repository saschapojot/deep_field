import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from scipy.special import binom

from decimal import Decimal, getcontext
from itertools import combinations

# this script generates data from 2-spin infinite-range model

#i.e., r=2

def format_using_decimal(value, precision=10):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


def all_r2_comb_2_E(spin_config,all_r2_comb,J_vec):
    """

    :param spin_config: spin values on L sites
    :param all_r2_comb: all 2-element combinations from {0,1,...,L-1}
    :param J_vec: coupling coefficients for each combination
    :return: energy
    """
    E_val=0
    for m in range(0,len(J_vec)):

        J_tmp=J_vec[m]
        v_tmp=all_r2_comb[m]
        E_val+=-J_tmp*spin_config[v_tmp[0]]*spin_config[v_tmp[1]]
    return E_val

A=1
# System Parameters
L = 15  # Number of spins
r = 2   # Number of spins in each interaction term

seed=17
np.random.seed(seed)
N_samples=15000
B = list(combinations(range(L), r))
K=len(B)
# Generate random spin configurations
spin_configurations_samples = np.random.choice([-1, 1], size=(N_samples, L))
J_vec=[np.random.normal(0,A) for _ in range(0,K)]


def generate_data(spin_configurations_samples,all_r2_comb,J_vec,train_ratio=0.8):
    """

    :param spin_configurations_samples:
    :param all_r2_comb:
    :param J_vec:
    :return:
    """
    # Compute energies for all configurations
    energies = []
    # counter = 0
    # tGenStart = datetime.now()
    for spin_config in spin_configurations_samples:
        energies.append(all_r2_comb_2_E(spin_config,all_r2_comb,J_vec))
        # if counter%100==0:
        #     # print("processed :"+str(counter))
        #     tGenEnd=datetime.now()
        #     # print("time: ",tGenEnd-tGenStart)
        # counter+=1

    # Split into training and testing datasets
    split_index = int(train_ratio * N_samples)
    X_train, Y_train = spin_configurations_samples[:split_index], energies[:split_index]
    X_test, Y_test = spin_configurations_samples[split_index:], energies[split_index:]

    return X_train, Y_train, X_test, Y_test



tStart=datetime.now()
X_train, Y_train, X_test, Y_test=generate_data(spin_configurations_samples,B,J_vec,0.8)


outDir=f"./data_inf_range_model_L{L}_r{r}/"
Path(outDir).mkdir(exist_ok=True,parents=True)
#save training data
fileNameTrain=outDir+"/inf_range.train.pkl"
with open(fileNameTrain,"wb") as fptr:
    pickle.dump((X_train,Y_train),fptr)


fileNameTest=outDir+"/inf_range.test.pkl"

with open(fileNameTest, "wb") as f:
    pickle.dump((X_test, Y_test), f)

tEnd=datetime.now()
print("time: ",tEnd-tStart)