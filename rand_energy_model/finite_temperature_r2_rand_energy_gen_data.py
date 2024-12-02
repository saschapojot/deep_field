import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from scipy.special import binom
import itertools
from decimal import Decimal, getcontext
from itertools import combinations
#this script generates data from random energy model under finite temperature

#this script is for r=2 case
def format_using_decimal(value, precision=10):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

def K_comb_2_exp_factor(A,T,spin_config,K_comb):
    """

    :param A:
    :param T: temperature
    :param spin_config:  spin values on L sites
    :param K_comb: K combinations of sites
    :return: exp factor
    """
    sum_tmp = 0
    for v in K_comb:
        sum_tmp+=spin_config[v[0]]**2*spin_config[v[1]]**2
    sum_tmp *= A ** 2 / (2 * T ** 2)
    return np.exp(sum_tmp)



def U_and_Z(A,T,spin_config,all_K_combinations):
    """

    :param A:
    :param T: temperature
    :param spin_config:  spin values on L sites
    :param all_K_combinations: all of K combinations of sites
    :return: U
    """
    Z = 0
    W = 0
    for one_K_comb in all_K_combinations:
        W_tmp = 0
        exp_factor = K_comb_2_exp_factor(A,T,spin_config,one_K_comb)
        Z += exp_factor
        for v in one_K_comb:
            W_tmp+=spin_config[v[0]]**2*spin_config[v[1]]**2
        W_tmp*=exp_factor
        W += W_tmp
    U = -A ** 2 / T * W / Z

    return U




def  generate_data(A,T,spin_configurations_samples,all_K_combinations, train_ratio=0.8):
    """

    :param A:
    :param T:
    :param spin_configurations_samples:
    :param all_K_combinations:
    :param train_ratio:
    :return:
    """
    # Compute energies for all configurations
    # energies = np.array([
    #     U_and_Z(A,T,spin_config,all_K_combinations) for spin_config in spin_configurations_samples
    # ])
    energies=[]
    counter=0
    tGenStart=datetime.now()
    for spin_config in spin_configurations_samples:
        energies.append(U_and_Z(A,T,spin_config,all_K_combinations))
        if counter%100==0:
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
A=1

T=1.5

# System Parameters
L = 14  # Number of spins
r = 2   # Number of spins in each interaction term
K = 3  # Number of interaction terms
N_samples=5000

# Generate random spin configurations
spin_configurations_samples = np.random.choice([0, 1], size=(N_samples, L))

B = list(combinations(range(L), 2))
C = list(combinations(B, K))
print(f"len(C)={len(C)}")
# Generate training and testing datasets
X_train, Y_train, X_test, Y_test = generate_data(A,T,spin_configurations_samples,C,0.8)

TStr=format_using_decimal(T)
outDir=f"./data_rand_energy_spin_T_{TStr}/"
Path(outDir).mkdir(exist_ok=True,parents=True)
#save training data
fileNameTrain=outDir+"/rand_energy_spin.train.pkl"
with open(fileNameTrain,"wb") as fptr:
    pickle.dump((X_train,Y_train),fptr)

fileNameTest=outDir+"/rand_energy_spin.test.pkl"

with open(fileNameTest, "wb") as f:
    pickle.dump((X_test, Y_test), f)

tEnd=datetime.now()
print("time: ",tEnd-tStart)



