import numpy as np
import sys
from model_qt_dsnn_config import *

#This script generates data for double exchange quantum model


def generate_uniform_matrix(N, low=0, high=2*np.pi, seed=None):
    """
    Generates an N x N matrix with elements uniformly distributed between low and high.

    Parameters:
    - N (int): Size of the matrix (number of rows and columns).
    - low (float): Lower bound of the uniform distribution (default is 0).
    - high (float): Upper bound of the uniform distribution (default is 2π).
    - seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - numpy.ndarray: An N x N matrix with uniformly distributed elements.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, size=(N, N))


def generate_Sigmax_Sigmay_Sigmaz(N,seed=None):
    """

    :param N: length of lattice
    :param seed:
    :return: components of classical spins over 2d lattice, 3 matrices are x, y , z components
    """
    Theta_mat=generate_uniform_matrix(N,0,np.pi,seed)
    Phi_mat=generate_uniform_matrix(N,0,2*np.pi,seed)

    sin_Theta_mat=np.sin(Theta_mat)

    sin_Phi_mat=np.sin(Phi_mat)

    cos_Theta_mat=np.cos(Theta_mat)

    cos_Phi_mat=np.cos(Phi_mat)

    Sigma_x=sin_Theta_mat*cos_Phi_mat
    Sigma_y=sin_Theta_mat*sin_Phi_mat

    Sigma_z=cos_Theta_mat

    Sigma_combined = np.stack([Sigma_x, Sigma_y, Sigma_z])

    return Sigma_combined


argErrCode=3
if (len(sys.argv)!=2):
    print("wrong number of arguments")
    print("example: python launch_one_run.py num_epochs decrease_over decrease_rate step_num_after_S1 C N")
    exit(argErrCode)
N=int(sys.argv[1])
#construct T0, T1 mat
T0_mat=np.zeros((N**2,N**2),dtype=float)

for r in range(0,N):
    for s in range(0,N):
        r_this=r
        r_next=(r+1)%N

        rowInd=r_this*N+s
        colInd=r_next*N+s

        T0_mat[rowInd,colInd]=1

        T0_mat[colInd,rowInd]=1


T1_mat=T0_mat


#construct T2, T3 mat

T2_mat=np.zeros((N**2,N**2),dtype=float)


for r in range(0,N):
    for s in range(0,N):
        s_this=s
        s_next=(s+1)%N

        rowInd=r*N+s_this

        colInd=r*N+s_next

        T2_mat[rowInd,colInd]=1

        T2_mat[colInd,rowInd]=1

T3_mat=T2_mat

def gen_Gamma_mat(T0_mat,T1_mat,T2_mat,T3_mat,Sigma_combined,t,J,mu,I):
    """

    :param T0_mat:
    :param T1_mat:
    :param T2_mat:
    :param T3_mat:
    :param Sigma_combined:
    :param t:
    :param J:
    :param mu:
    :param I:
    :return: Gamma(S)
    """
    # print(f"Sigma_combined.shape={Sigma_combined.shape}")
    Sigma_x=Sigma_combined[0,:,:]#N by N matrix

    Sigma_y=Sigma_combined[1,:,:]# N by N matrix

    Sigma_z=Sigma_combined[2,:,:]# N by N matrix

    Sx_vec=Sigma_x.flatten()# length N**2 vector

    Sy_vec=Sigma_y.flatten()# length N**2 vector

    Sz_vec=Sigma_z.flatten()# length N**2 vector

    Sx=np.diag(Sx_vec,k=0)# N**2 by N**2 matrix

    Sy=np.diag(Sy_vec,k=0)# N**2 by N**2 matrix

    Sz=np.diag(Sz_vec,k=0)# N**2 by N**2 matrix




    block00=-t*T0_mat-t*T2_mat-1/2*J*Sz-mu*I# N**2 by N**2 matrix

    block01=-1/2*J*Sx+1j*1/2*J*Sy# N**2 by N**2 matrix

    block10=-1/2*J*Sx-1j*1/2*J*Sy# N**2 by N**2 matrix

    block11=-t*T1_mat -t*T3_mat+1/2*J*Sz-mu*I# N**2 by N**2 matrix

    Gamma_mat=np.block(
        [[block00,block01],
        [block10,block11]]
    )# 2*N**2 by 2*N**2 matrix

    return Gamma_mat










def vec_2_E(eig_vals,T):
    """

    :param eig_vals:
    :param T:
    :return:
    """
    # print(f"eig_vals={eig_vals}")
    exp_vals=np.exp(-1/T*eig_vals)

    E=-T*np.sum(np.log(1+exp_vals))
    return E








def gen_dataset(num_samples,N,T0_mat,T1_mat,T2_mat,T3_mat,t,J,mu,I_N2,T,seed_vec=None):
    """

    :param num_samples:
    :param N:
    :param T0_mat:
    :param T1_mat:
    :param T2_mat:
    :param T3_mat:
    :param t:
    :param J:
    :param mu:
    :param I_N2:
    :param T:
    :param seedVal:
    :return:
    """
    data = []
    values = []

    for j in range(0,num_samples):
        Sigma_combined_tmp = generate_Sigmax_Sigmay_Sigmaz(N, seed_vec[j])
        gm_tmp = gen_Gamma_mat(T0_mat, T1_mat, T2_mat, T3_mat, Sigma_combined_tmp, t, J, mu, I_N2)

        eigValsTmp,eigVecsTmp=np.linalg.eigh(gm_tmp)

        E_tmp=vec_2_E(eigValsTmp,T)
        data.append(Sigma_combined_tmp)
        values.append(E_tmp)

    return data, values

tStart=datetime.now()


I_N2=np.eye(N**2)
num_samples=300000
seed_vec=range(100,100+num_samples)
data,values=gen_dataset(num_samples,N,T0_mat,T1_mat,T2_mat,T3_mat,t,J,mu,I_N2,T,seed_vec)


def generate_train_test(data,values,train_ratio):
    """

    :param data:
    :param values:
    :param train_ratio:
    :return:
    """
    N_samples=len(values)
    split_index = int(train_ratio * N_samples)

    X_train, Y_train = data[:split_index],values[:split_index]

    X_test, Y_test =data[split_index:],values[split_index:]

    return X_train, Y_train, X_test, Y_test



train_ratio=0





X_train, Y_train, X_test, Y_test=generate_train_test(data,values,train_ratio)


outDir=f"./train_test_data/N{N}/"
Path(outDir).mkdir(exist_ok=True,parents=True)

out_fileName_train=outDir+f"/db.train_num_samples{num_samples}.pkl"

with open(out_fileName_train,"wb") as fptr:
    pickle.dump((X_train,Y_train),fptr)

out_fileName_test=outDir+f"/db.test_num_samples{num_samples}.pkl"

with open(out_fileName_test,"wb") as fptr:
    pickle.dump((X_test, Y_test), fptr)

tEnd=datetime.now()
print("time: ",tEnd-tStart)