import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from music import calc_angles

# --- Paramètres ---
M = 20
N = 500
theta1_deg, theta2_deg = 40, 50
theta1, theta2 = np.deg2rad(theta1_deg), np.deg2rad(theta2_deg)   # conversion en radians
sigma_s1 = 1
sigma_s2 = 1
sigma_v = 0.1

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

# --- Vérification : puissance moyenne reçue par capteur ---
received_power = np.mean(np.abs(y)**2, axis=1)
print("Puissance moyenne par capteur :", np.round(received_power, 3))

# --- Matrice de covariance empirique ---
R = (y @ y.conj().T) / N
print("Matrice de covariance R̂_y :")
print(np.round(R, 3))

# --- Calcul du pseudo-spectre de Capon ---
# Inversion de la matrice de covariance (avec régularisation)
eps = 1e-6 * np.trace(R) / M  
R_inv = np.linalg.inv(R + eps * np.eye(M))

# Balayage angulaire
theta_scan = np.linspace(-90, 90, 721)  # de -90° à 90°, pas de 0.25°
P_capon = []

for theta_deg_scan in theta_scan:
    theta_rad_scan = np.deg2rad(theta_deg_scan)
    a_theta = np.exp(-1j * np.pi * m * np.sin(theta_rad_scan))  # vecteur directeur
    denom = np.conj(a_theta).T @ R_inv @ a_theta
    P_capon.append(1 / np.real(denom))

P_capon = np.array(P_capon)

angles, valeurs = calc_angles(theta_scan, P_capon, 2)
print("Angles détectés (°):", np.round(angles, 2))

# --- Visualisation de la partie réelle d'un capteur ---
plt.figure(figsize=(8,4))
plt.plot(np.real(y[0,:]))
plt.title(f"Signal reçu : source 1, θ₀ = {theta1_deg}° ; source 2, θ₀ = {theta2_deg}°")
plt.xlabel("n (échantillon)")
plt.ylabel("Re{y₁[n]}")
plt.grid(True)
plt.show()

# --- Visualisation du pseudo-spectre ---
plt.figure(figsize=(8,4))
plt.plot(theta_scan, 10 * np.log10(P_capon / np.max(P_capon)))
plt.title("Pseudo-spectre spatial de Capon (normalisé)")
plt.xlabel("Angle θ (degrés)")
plt.ylabel("P̂_Capon(θ) [dB]")
plt.grid(True)
plt.show()
