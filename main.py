from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from common import *
from music import music
import sounddevice as sd


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

def play_signal(signal, Fs, gain=1.0):
    
    # Amplification
    signal_amp = signal * gain
    
    # Normalisation pour éviter la saturation
    max_val = np.max(np.abs(signal_amp))
    if max_val > 1.0:
        signal_amp = signal_amp / max_val
    
    sd.play(signal_amp, Fs)
    sd.wait()


Y, Fs, N, M = open_file("data.mat")
angles_est = music(Y, N, M, K=-1, trace=False, seuil_ratio=5)
print(angles_est)
K = len(angles_est)


# Calcul des vecteurs d'attenuation
A = np.zeros((M, K), dtype=complex)
for k in range(K):
    theta = angles_est[k]
    a = steering_vector(theta, M)
    A[:, k] = a 


s_est = []
for k in range(K):
    w_k = zero_forcing(A, k, K)
    s_k_hat = np.conj(w_k).T @ Y
    s_est.append(np.real(s_k_hat).flatten())


print(sd.query_devices())

play_signal(s_est[1], Fs, 4.0)