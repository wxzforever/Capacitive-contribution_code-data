# The code for generating CVs from pure EDLC condition with different scan rates

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# General parameters
T = 298.15  # Kelvin temperature
F = 96485.3329  # Faraday constant
R = 8.314  # Ideal gas constant

v_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # List of different scan rates, V/s
Ei = 0  # Initial potential, V
Ef = 1  # Final potential, V

# Variables for EDLC model
Rs = 100  # series resistance, ohm
Rp = 100000  # Parallel resistance
Cd = 0.004  # capacitance, F
AEDLCOX = 1  # Amplitude factor of positive EDLC
AEDLCRED = 1  # Amplitude factor of negative EDLC

def tcharge1(Eap, v, Ei):
    return (Eap - Ei) / v

def tcharge2(Ef, v, Ei):
    return (Ef - Ei) / v

def tdischarge(Eap, v, Ef):
    return (Ef - Eap) / v

def IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp):
    term1 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd)))*1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge1(Eap, v, Ei) - Rs * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd))))*1000

    term2 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd)))*1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge2(Ef, v, Ei) - Rs * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd))))*1000
    
    term3 = AEDLCRED * (-v) * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd)))*1000 + \
            AEDLCRED * (-v) * (1 / Rp) * (tdischarge(Eap, v, Ef) - Rs * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd))))*1000

    return term1, term2 + term3


def Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED):
    iedlc = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp)

    return iedlc

# I-E curve plot of combined models for different scan rates
plt.figure(figsize=(12,8))
base_color = np.array([128, 0, 128]) / 255.0  # Purple RGB value, normalized to the range [0, 1]
for i, v in enumerate(v_values):
    Eap_values = np.linspace(Ei, Ef, 100)
    Itotal_values = [Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED) for Eap in Eap_values]
    alpha = 0.1 + 0.9 * (v - min(v_values)) / (max(v_values) - min(v_values))  # Transparency from 0.1 to 1
    plt.plot(Eap_values, Itotal_values, color=np.append(base_color, alpha), linewidth=5, linestyle='--')

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Potential (V)', fontsize=24)
plt.ylabel("Current (mA)", fontsize=24)
plt.show()
