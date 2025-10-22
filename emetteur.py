import numpy as np
import matplotlib.pyplot as plt
from common import *


# Emetteur
sigma_v = 0.1               # Puissance du bruit
sigma_s1 = 1                # Puissance de la source 1
sigma_s2 = 1                # Puissance de la source 2
theta1 = 40                  # Elevation de la source 1
theta2 = 45                 # Elevation de la source 2
lambda_c = 0.3              # Longueur d'onde du signal



def old_build_Y(N, M, d):
    # Vecteurs de direction
    a1 = steering_vector(theta1, M, d)
    a2 = steering_vector(theta2, M, d)

    # Génération des signaux sources (complexes gaussiens)
    s1 = np.sqrt(sigma_s1/2) * (np.random.randn(N) + 1j*np.random.randn(N))
    s2 = np.sqrt(sigma_s2/2) * (np.random.randn(N) + 1j*np.random.randn(N))

    # Bruit blanc complexe
    v = np.sqrt(sigma_v/2) * (np.random.randn(M, N) + 1j*np.random.randn(M, N))

    # Signaux observés : superposition des sources + bruit
    Y = np.outer(a1, s1) + np.outer(a2, s2) + v
    
    return Y


def build_Y(N, M, d, thetas=[theta1, theta2], sigmas_s=[sigma_s1, sigma_s2], sigma_v=0.1):
    
    if len(thetas) != len(sigmas_s):
        print("Erreur: thetas et sigmas_s doivent etre de longeur egales")
        exit(1)
    
    # Construction des a [K]
    liste_a = []
    for theta in thetas:
        a = steering_vector(theta, M, d)
        liste_a.append(a)
        
    # Construction des s [N]
    liste_s = []
    for sigma in sigmas_s:
        s = np.sqrt(sigma/2) * (np.random.randn(N) + 1j*np.random.randn(N)) 
        liste_s.append(s)
        
    # Bruit blanc complexe
    v = np.sqrt(sigma_v/2) * (np.random.randn(M, N) + 1j*np.random.randn(M, N))
    
    # Assemblage des sources
    Y = v
    for i in range(len(thetas)):
        x = np.outer(liste_a[i], liste_s[i])
        Y = Y + x
        
    return Y


if __name__ == "__main__":
    N = 2000
    M = 25
    d = 1
    
    Y1 = old_build_Y(N, M, d)
    Y2 = build_Y(N, M, d, [theta1, theta2], [sigma_s1, sigma_s2], sigma_v)
