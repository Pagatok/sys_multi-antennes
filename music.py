import numpy as np
import matplotlib.pyplot as plt
from common import *
from emetteur import *
from scipy.signal import find_peaks



def estimate_R(Y):
    M_est, N = Y.shape
    somme = np.zeros((M_est, M_est))
    for n in range(N):
        yn = Y[:, n]
        prod = np.outer(yn, yn.conj())/N
        somme = somme + prod
    return somme


# Decomposition spectrale de R pour obtenir U et gamma
def get_U_gamma(R_est, trace=False):
    
    gamma, U = np.linalg.eigh(R_est) 
    idx = np.argsort(gamma)[::-1]
    gamma = gamma[idx]
    U = U[:, idx]
    
    
    if trace:
        # Index des valeurs propres
        indices = np.arange(1, len(gamma)+1)

        # Trace le graphique
        plt.figure(figsize=(6,4))
        plt.stem(indices, gamma)
        plt.xlabel("Index des valeurs propres")
        plt.ylabel("Valeur propre")
        plt.title(r"Valeurs propres de $\hat{R}$")
        plt.grid(True)
        plt.show()
    
    return U, gamma


# Estime le nombre de sources K à partir du spectre des valeurs propres.
def estimate_K(gamma, seuil_ratio=5):

    ratios = gamma[:-1] / gamma[1:]
    K = np.argmax(ratios > seuil_ratio) + 1  # premier saut significatif
    
    # Si aucun saut clair trouvé → bruit plat
    if K == 0 or K >= len(gamma):
        K = 1
    print(K)
    return K


def get_piT(U_est, K):
    
    # Sous-espace signal
    U_signal = U_est[:, :K]
    Pi = U_signal @ U_signal.conj().T

    # Sous-espace bruit
    U_bruit = U_est[:, K:]
    Pi_perp = U_bruit @ U_bruit.conj().T

    return Pi_perp
    


def d_est(theta, piT, M, d):
    a = steering_vector(theta, M, d)
    temp = np.dot(piT, a)
    return np.linalg.norm(temp, 2)**2


def trace_d_est(n_ech, piT, M, d, trace=True):

    angles = np.linspace(-90, 90, n_ech)  # vecteur d'angles
    d_calc = np.zeros_like(angles)

    for i in range(len(angles)):
        d_calc[i] = d_est(angles[i], piT, M, d)
        
    # Trace le graphique
    if trace:
        plt.figure(figsize=(12,4))
        plt.plot(angles, d_calc)
        plt.xlabel("Theta (deg)")
        plt.ylabel("d(Theta)")
        plt.title("Localisation Visuelle de sources par MUSIC")
        plt.grid(True)
        plt.show()
    
    return angles, d_calc


# Trouve les K minima locaux de d_est pour trouver les angles d'arrivée
def calc_angles(angles, d_calc, K):
    # Trouver les minima locaux de d_calc
    peaks, props = find_peaks(-d_calc, distance=5, prominence=0.0)

    if len(peaks) == 0:
        return np.array([]), np.array([])

    # Trier par profondeur décroissante (prominence)
    sorted_idx = np.argsort(props["prominences"])[::-1]

    # Sélectionner les K plus significatifs
    top_idx = sorted_idx[:K]

    # Récupérer les angles et valeurs correspondants
    angles_calc = angles[peaks[top_idx]]
    valeurs = d_calc[peaks[top_idx]]

    return angles_calc, valeurs


# Algorithme principal de MUSIC a utiliser
def music(Y, N, M, K=-1, trace=True, seuil_ratio=5):
    '''
    Cette fonction reproduit l'algorithme MUSIC pour l'estimation de nombre et la localisation des sources
    
    args:
        - Y (numpy array 1D)    Signal reçu
        - N (int)               Nombre d'observations
        - M (int)               Nombre de capeturs dans l'antenne
        - K (int) Default: -1   Nombre de sources estimées si deja connu (Laisser -1 pour que MUSIC l'estime lui-meme)
        - trace (bool) Def:True Indique si les courbs doivent etre tracées ou non
        - seuil_ratio Def:5     Ratio pour l'estimation de K
    '''

    print("======Localisation Sources par MUSIC======")
    
    print("Estimation de R...")
    R_est = estimate_R(Y)
    
    print("Estimation de U...")
    U_est, gamma_est = get_U_gamma(R_est, trace=trace)
    
    if K == -1:
        print("Estimation de K...")
        K = estimate_K(gamma_est, seuil_ratio=5)
        exit()
        
    print("Estimation de Pi_T...")
    pi_T = get_piT(U_est, K)
    
    print("Calcul de d...")
    angles, d_calc = trace_d_est(1000, pi_T, M, N, trace=trace)
    
    print("Calculs des angles d'arrivées...")
    angles_est, valeurs_est = calc_angles(angles, d_calc, K)
    
    return angles_est


if __name__ == "__main__":
    N_test = 2000
    M_test = 25
    K_test = 2
    d_capteurs_test = 1
    
    
    Y = build_Y(N_test, M_test, d_capteurs_test)
    print(music(Y, N_test, M_test, trace=False))