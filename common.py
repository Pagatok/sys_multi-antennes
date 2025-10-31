import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Emetteur
sigma_v = 0.1               # Puissance du bruit
sigma_s1 = 1                # Puissance de la source 1
sigma_s2 = 1                # Puissance de la source 2
theta1 = 40                  # Elevation de la source 1
theta2 = 45                 # Elevation de la source 2
lambda_c = 0.3              # Longueur d'onde du signal


def steering_vector(theta, M, d=1.0, l_c=2.0):

    liste_m = np.arange(M)  # indices des antennes 0, 1, ..., M-1
    phase = -1j * 2 * np.pi * (1/2) * liste_m * np.sin(np.deg2rad(theta))
    a = np.exp(phase)
    return a


def print_signal(Y, Fs=1):
    
    indices = np.arange(Y.size) / Fs
    
    plt.figure(figsize=(12,4))
    plt.plot(indices, Y)
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.title("Visualisation du signal transmi")
    plt.grid(True)
    plt.show()
    
    
def play_signal(signal, Fs, gain=1.0):
    
    # Amplification
    signal_amp = signal * gain
    
    # Normalisation pour éviter la saturation
    max_val = np.max(np.abs(signal_amp))
    if max_val > 1.0:
        signal_amp = signal_amp / max_val
    
    sd.play(signal_amp, Fs)
    sd.wait()
    

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
