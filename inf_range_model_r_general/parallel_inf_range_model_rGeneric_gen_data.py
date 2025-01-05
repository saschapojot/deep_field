import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from scipy.special import binom

from decimal import Decimal, getcontext
from itertools import combinations

from multiprocessing import Pool


# this script generates data from 2-spin infinite-range model

# for generic r
#using multiprocessing



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



A=1
# System Parameters
L = 50 # Number of spins
r = 5 # Number of spins in each interaction term

seed=17
np.random.seed(seed)
N_samples=int(1e7)
B = list(combinations(range(L), r))
K=40
print(f"K={K}")

tGenRandStart=datetime.now()
# Generate random spin configurations
spin_configurations_samples = np.random.choice([-1, 1], size=(N_samples, L))
J_vec=[np.random.normal(0,A) for _ in range(0,K)]
split_ratio=0.5

tGenRandEnd=datetime.now()
print("J_vec time:",tGenRandEnd-tGenRandStart)