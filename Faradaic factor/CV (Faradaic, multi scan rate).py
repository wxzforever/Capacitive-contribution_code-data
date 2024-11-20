# The code for generating CVs from Faradaic condition with different scan rates

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Constants
n = 1  # Number of electron transfers
F = 96485  # Faraday constant, C·mol^-1
S = 1  # Electrode surface area, cm^-2
k = 0.01  # Standard rate constant
Gamma = 3.15e-5  # Surface concentration, mol·cm^-2
alpha = 0.5  # Charge transfer coefficient
R = 8.314  # Ideal gas constant, J·mol^-1·K^-1
T = 298  # Temperature, K

# Range of applied potentials
Eap = np.linspace(0.1, 0.9, 500)

# List of different scan rates
scan_rates = [0.001,0.003,0.005,0.008,0.01]  # Example scan rates in V/s

# Color map for distinguishing different scan rates
#colors = plt.cm.jet(np.linspace(0, 1, len(scan_rates)))

# Plotting
plt.figure(figsize=(12, 8))

for i, v in enumerate(scan_rates):
    # Function to calculate the Faradaic currents
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

    # Calculate currents
    I_forward, I_reverse = IFarad(n, F, S, k, Gamma, alpha, R, T, v, Eap)

    # Plot the combined currents for this scan rate
    plt.plot(Eap, I_forward, alpha=0.8, linewidth=5, color="black")
    plt.plot(Eap, I_reverse, alpha=0.8, linewidth=5, color="black")

plt.xlim(0, 1)
plt.ylim(-0.25, 0.25)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Potential (V)', fontsize=24)
plt.ylabel('Current (I)', fontsize=24)
plt.show()
