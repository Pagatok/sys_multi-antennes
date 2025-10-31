'''
Usage: python separe_signal.py [path/to/data.mat]

Ce script contient la chaine de traitement pour separerr les signaux dans le cadre de la partie On ne s'entend plus
'''

from scipy.io import loadmat
import numpy as np
from common import *
from music import music
import sys


# Ouverture du fichier data.mat et extraction de Y et de Fs
def open_file(filepath):
    data = loadmat(filepath)
    Fs = float(data['Fs'].squeeze())  # squeeze enlève les dimensions (1,1)
    N = len(data['data'][0])
    M = len(data['data'])
    Ys = np.zeros((M, N))
    for i in range(M):
        mat = data['data'][i]  # N = 226706
        Ys[i, :] = mat.ravel()
        
    return Ys, Fs, N, M


def zero_forcing(A, k, K):
    W = np.linalg.pinv(A)  # équivaut à (AᴴA)^(-1)Aᴴ de façon stable
    e_k = np.zeros((K, 1), dtype=complex)
    e_k[k, 0] = 1
    w_k = W.T @ e_k
    return w_k


# Calcul des vecteurs de direction
def get_A(angles_est, M, K):
    A = np.zeros((M, K), dtype=complex)
    for k in range(K):
        theta = angles_est[k]
        a = steering_vector(theta, M)
        A[:, k] = a 
        
    return A

def isolation_signaux(Y, A, K):
    s_est = []
    for k in range(K):
        w_k = zero_forcing(A, k, K)
        s_k_hat = np.conj(w_k).T @ Y
        s_est.append(np.real(s_k_hat).flatten())
    return s_est




if __name__ == "__main__":
    if len(sys.argv) < 2:
        filepath = "data.mat"
    else:
        filepath = sys.argv[1]
    
    # Ouverture du fichier .mat
    Y, Fs, N, M = open_file("data.mat")
    
    # Detection du nombre de sources K et de leurs angles d'arrivées
    angles_est = music(Y, M, K=-1, trace=True, seuil_ratio=5)
    print(angles_est)
    K = len(angles_est)
    A = get_A(angles_est, M, K)
    
    # Isolation des signaux selon leurs angles d'arrivee
    s_est = isolation_signaux(Y, A, K)
    
    # Pour jouer un des signaux detcets
    id_signal = 2
    play_signal(s_est[id_signal], Fs, 4.0)
    