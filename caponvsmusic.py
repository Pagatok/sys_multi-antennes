'''
Versions abreges de capon et music pour effectuer des simulations en grand nombre dans le cadre de comparasion
de differentes methodes

Inclut aussi les fonctions effectuant differenetes comparaisons
'''

import numpy as np
from scipy.signal import find_peaks
from common import build_Y
from music import music
from tqdm import tqdm
import matplotlib.pyplot as plt


def capon_f(sigma_v=0.1, M=25):
    
    # --- Paramètres ---
    N = 1000
    theta1_deg, theta2_deg = 40, 45
    theta1, theta2 = np.deg2rad(theta1_deg), np.deg2rad(theta2_deg)
    sigma_s1 = 1
    sigma_s2 = 1

    # --- Vecteur directionnel a(theta) ---
    m = np.arange(M)
    a1 = np.exp(-1j * np.pi * m * np.sin(theta1))
    a2 = np.exp(-1j * np.pi * m * np.sin(theta2))
    A  = np.column_stack([a1, a2])

    # --- Signal source (gaussien complexe) ---
    s1 = np.sqrt(sigma_s1 / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
    s2 = np.sqrt(sigma_s2 / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
    S = np.vstack([s1, s2])

    # --- Bruit additif (gaussien complexe) ---
    v = np.sqrt(sigma_v / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))

    # --- Signal reçu par le réseau de capteurs ---
    y = A @ S + v

    # # --- Vérification : puissance moyenne reçue par capteur ---
    # received_power = np.mean(np.abs(y)**2, axis=1)

    # --- Matrice de covariance empirique ---
    R = (y @ y.conj().T) / N

    # --- Calcul du pseudo-spectre de Capon ---
    eps = 1e-6 * np.trace(R) / M  
    R_inv = np.linalg.inv(R + eps * np.eye(M))

    # Balayage angulaire
    theta_scan = np.linspace(-90, 90, 721)
    P_capon = []

    for theta_deg_scan in theta_scan:
        theta_rad_scan = np.deg2rad(theta_deg_scan)
        a_theta = np.exp(-1j * np.pi * m * np.sin(theta_rad_scan))
        denom = np.conj(a_theta).T @ R_inv @ a_theta
        P_capon.append(1 / np.real(denom))

    P_capon = np.array(P_capon)

    # Trouve les K minima locaux de d_est pour trouver les angles d'arrivée
    def calc_angles(angles, d_calc, K):
        # Trouver les minima locaux de d_calc
        peaks, props = find_peaks(d_calc, distance=5, prominence=0.0)

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

    angles, valeurs = calc_angles(theta_scan, P_capon, 2)

    return np.round(angles, 2)


def music_f(sigma_v=0.1, M=25):
    
    N = 2000
    d = 0.2
    K = 2

    sigma_s1 = 1                # Puissance de la source 1
    sigma_s2 = 1                # Puissance de la source 2
    theta1 = 40                  # Elevation de la source 1
    theta2 = 45                 # Elevation de la source 2
    lambda_c = 0.4              # Longueur d'onde du signal

    # Construction du signal (voir emetteur)
    Y = build_Y(N, M, d, thetas=[theta1, theta2], sigmas_s=[sigma_s1, sigma_s2], sigma_v=sigma_v)

    return music(Y, M, K=K, d=d, l_c=lambda_c, trace=False, prt=False)



def test_snr_compare(ff1, ff2, snr_values=np.logspace(-1, 2.5, 75), n_trials=200, tol=1.5):
    """
    Compare le pourcentage de détection correcte pour deux fonctions ff1 et ff2
    sur différentes valeurs de bruit.

    Args:
        ff1, ff2 : fonctions prenant le niveau de bruit sigma_v et retournant
                   une liste de deux angles estimés [theta1, theta2]
        snr_values : liste ou array de valeurs de sigma_v (bruit)
        n_trials : nombre d'essais par valeur de bruit
        tol : tolérance en degrés autour de 40 et 45 pour considérer la détection correcte
    """
    def compute_percentage(ff):
        detection_percentage = []
        for sigma_v in tqdm(snr_values, desc=ff.__name__):
            successes = 0
            for _ in range(n_trials):
                angles_est = ff(sigma_v)
                if len(angles_est) == 2:
                    diff1 = np.min(np.abs(np.array(angles_est) - 40))
                    diff2 = np.min(np.abs(np.array(angles_est) - 45))
                    if (diff1 <= tol and diff2 <= tol) or (diff2 <= tol and diff1 <= tol):
                        successes += 1
            detection_percentage.append((successes / n_trials) * 100)
        return detection_percentage

    perc1 = compute_percentage(ff1)
    perc2 = compute_percentage(ff2)

    # Tracé comparatif
    plt.figure(figsize=(8,4))
    plt.semilogx(snr_values, perc1, 'r', label='MUSIC', linewidth=2)
    plt.semilogx(snr_values, perc2, 'b', label='Capon', linewidth=2)
    plt.xlabel("Bruit sigma_v")
    plt.ylabel("Pourcentage de détection correcte (%)")
    plt.title(f"Comparaison de détection de sources")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    plt.show()

# Exemple d'utilisation
test_snr_compare(music_f, capon_f)