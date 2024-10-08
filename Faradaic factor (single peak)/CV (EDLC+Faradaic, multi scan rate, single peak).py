# The code for generating CVs from EDLC+Faradaic condition with different scan rates

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# General parameters
T = 298.15  # Kelvin temperature
F = 96485.3329  # Faraday constant
R = 8.314  # Ideal gas constant

v_values = [0.001,0.003,0.005,0.008,0.01]  # Select multiple scan rates, V/s
Ei = 0  # Initial potential, V
Ef = 1  # Final potential, V

# Variables for EDLC model
Rs = 80  # series resistance, ohm
Rp = 100000  # Parallel resistance
Cd = 0.04  # capacitance, F
AEDLCOX = 1  # Amplitude factor of positive EDLC
AEDLCRED = 1  # Amplitude factor of negative EDLC

# Variables for Faradaic model
n = 1  # Number of electron transfers
k = 0.01  # Standard rate constant
Gamma = 3.15e-5  # Surface concentration, mol·cm^-2
alpha = 0.5  # Charge transfer coefficient
S = 1

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

def IFarad(n, F, S, k, Gamma, alpha, R, T, v, Eap):
    I_forward = (n * F * S * k * Gamma *
                 np.exp(alpha * n * F / (R * T) * (Eap - 0.5)) /
                 np.exp((R * T) / (F * alpha * n) * k / v *
                        np.exp(alpha * n * F / (R * T) * (Eap - 0.5))))
    
    I_reverse = (-n * F * S * k * Gamma *
                 np.exp(alpha * n * F / (R * T) * -(Eap - 0.5)) /
                 np.exp((R * T) / (F * alpha * n) * k / v *
                        np.exp(alpha * n * F / (R * T) * -(Eap - 0.5))))
    
    return I_forward, I_reverse

def Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED, n, F, S, k, Gamma, alpha, R, T):
    iedlc1, iedlc2 = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp)
    ifarad_forward, ifarad_reverse = IFarad(n, F, S, k, Gamma, alpha, R, T, v, Eap)
    total_forward = iedlc1 + ifarad_forward
    total_reverse = iedlc2 + ifarad_reverse
    return total_forward, total_reverse

# I-E curve plot of combined models for multiple scan rates
plt.figure(figsize=(12, 8))

for v in v_values:
    Eap_values = np.linspace(Ei, Ef, 100)
    Itotal_values = [Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED, n, F, S, k, Gamma, alpha, R, T) for Eap in Eap_values]
    Itotal_values = np.array(Itotal_values)

    # Combine the two curves with a NaN separator
    combined_currents = np.concatenate([Itotal_values[:, 0], [np.nan], Itotal_values[:, 1]])
    combined_Eap = np.concatenate([Eap_values, [np.nan], Eap_values])

    # Plotting both the EDLC and Faradaic combined current as one line per scan rate
    plt.plot(combined_Eap, combined_currents, linewidth=5)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Potential (V)', fontsize=24)
plt.ylabel('Current (A)', fontsize=24)
plt.show()
