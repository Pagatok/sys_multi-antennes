import numpy as np
import matplotlib.pyplot as plt

lambda_c = 0.3

def steering_vector(theta, M, d, lambda_c=lambda_c):

    m = np.arange(M)  # indices des antennes 0, 1, ..., M-1
    # phase = -1j * 2 * np.pi * d / lambda_c * m * np.sin(np.deg2rad(theta))
    phase = -1j * np.pi * m * np.sin(np.deg2rad(theta))
    a = np.exp(phase)
    return a