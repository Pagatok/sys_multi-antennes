from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from common import *
from music import music


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


# Convertir une liste de signaux recus en un seul par systeme antennes




Y, Fs, N, M = open_file("data.mat")
angles_est, _ = music(Y, N, M, K=-1, trace=True, seuil_ratio=5)






# Préparation pour methode MUSIC

