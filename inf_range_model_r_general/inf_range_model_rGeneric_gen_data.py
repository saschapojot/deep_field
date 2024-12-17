import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from scipy.special import binom

from decimal import Decimal, getcontext
from itertools import combinations

# this script generates data from 2-spin infinite-range model

# for generic r

def format_using_decimal(value, precision=10):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


def all_rGeneric_comb_2_E(spin_config,selected_r_comb,J_vec):
    """

    :param spin_config: spin values on L sites
    :param selected_r_comb: selected r-element combinations from {0,1,...,L-1}
    :param J_vec: coupling coefficients for each combination
    :return: energy
    """
    E_val=0
    for m in range(0,len(J_vec)):

        J_tmp=J_vec[m]
        v_tmp=selected_r_comb[m]
        spin_vec_tmp=[spin_config[v_tmp_component] for v_tmp_component in v_tmp]
        E_val+=-J_tmp*np.prod(spin_vec_tmp)
    return E_val

A=2
# System Parameters
L = 15# Number of spins
r = 3 # Number of spins in each interaction term

seed=17
np.random.seed(seed)
N_samples=int(20000)
B = list(combinations(range(L), r))
# print(len(B))
# print(B[0])
# print(B[134])
K=len(B)
print(f"K={K}")

unique_integers = np.random.choice(range(0, len(B)), size=K, replace=False)
# print(unique_integers)
# print(unique_integers)
# Generate random spin configurations
spin_configurations_samples = np.random.choice([0, 1], size=(N_samples, L))
J_vec=[np.random.normal(1,A) for _ in range(0,K)]
split_ratio=0.8

B_selected=[B[ind] for ind in unique_integers]
# print(B_selected)
def generate_data(spin_configurations_samples,selected_r_comb,J_vec,train_ratio):
    """

    :param spin_configurations_samples:
    :param selected_r_comb:
    :param J_vec:
    :return:
    """
    # Compute energies for all configurations
    energies = []
    counter = 0
    tGenStart = datetime.now()
    for spin_config in spin_configurations_samples:
        energies.append(all_rGeneric_comb_2_E(spin_config,selected_r_comb,J_vec))
        if counter%5000==0:
            print("processed :"+str(counter))
            tGenEnd=datetime.now()
            print("time: ",tGenEnd-tGenStart)
        counter+=1

    # Split into training and testing datasets
    split_index = int(train_ratio * N_samples)
    X_train, Y_train = spin_configurations_samples[:split_index], energies[:split_index]
    X_test, Y_test = spin_configurations_samples[split_index:], energies[split_index:]

    return X_train, Y_train, X_test, Y_test



tStart=datetime.now()
X_train, Y_train, X_test, Y_test=generate_data(spin_configurations_samples,B_selected,J_vec,split_ratio)


outDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
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