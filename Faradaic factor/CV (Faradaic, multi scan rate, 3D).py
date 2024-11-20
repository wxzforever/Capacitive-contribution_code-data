# The code for generating 3D CVs from Faradaic condition with different scan rates

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress

# General parameters
T = 298.15  # Kelvin temperature
F = 96485.3329  # Faraday constant
R = 8.314  # Ideal gas constant

v_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  # Multiple scan rates, V/s
Ei = 0  # Initial potential, V
Ef = 1  # Final potential, V

# Variables for EDLC model
Rs = 200 # Series resistance, ohm
Rp = 100000  # Parallel resistance, ohm
Cd = 0.04  # Capacitance, F
AEDLCOX = 1  # Amplitude factor of positive EDLC
AEDLCRED = 1  # Amplitude factor of negative EDLC

# Variables for Faradaic model
n = 1  # Number of electron transfers
k = 0.1  # Standard rate constant
Gamma = 3.15e-5  # Surface concentration, mol·cm^-2
alpha = 0.5  # Charge transfer coefficient
S = 1  # Electrode surface area

# Time functions
def tcharge1(Eap, v, Ei):
    return (Eap - Ei) / v

def tcharge2(Ef, v, Ei):
    return (Ef - Ei) / v

def tdischarge(Eap, v, Ef):
    return (Ef - Eap) / v

# EDLC current function
def IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp):
    term1 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd)))*1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge1(Eap, v, Ei) - Rs * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd))))*1000

    term2 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd)))*1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge2(Ef, v, Ei) - Rs * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd))))*1000
    
    term3 = AEDLCRED * (-v) * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd)))*1000 + \
            AEDLCRED * (-v) * (1 / Rp) * (tdischarge(Eap, v, Ef) - Rs * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd))))*1000

    return term1, term2 + term3

# Faradaic current function
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

# Total current function (EDLC + Faradaic)
def Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED, n, F, S, k, Gamma, alpha, R, T):
    iedlc1, iedlc2 = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp)
    ifarad_forward, ifarad_reverse = IFarad(n, F, S, k, Gamma, alpha, R, T, v, Eap)
    total_forward = iedlc1 + ifarad_forward
    total_reverse = iedlc2 + ifarad_reverse
    return total_forward, total_reverse

# Extract Eap, i1_total, i2_total, k1_1_v and k1_2_v data and calculate the results for each scan rate
Eap_values = np.linspace(0, 1, 100)
v_values = np.array(v_values)

# Initialize the matrix used to store surface data
Eap_mesh, v_mesh = np.meshgrid(Eap_values, v_values)

# Create a grid data to store i1_total, i2_total, k1_1_v and k1_2_v
i1_total_surface = np.zeros_like(Eap_mesh)
i2_total_surface = np.zeros_like(Eap_mesh)
k1_1_v_surface = np.zeros_like(Eap_mesh)
k1_2_v_surface = np.zeros_like(Eap_mesh)

# Fill the grid with the calculated results of each scan rate and Eap
for i, v in enumerate(v_values):
    for j, Eap in enumerate(Eap_values):
        term1_total, term2_total = Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED, n, F, S, k, Gamma, alpha, R, T)
        i1_total_surface[i, j] = term1_total
        i2_total_surface[i, j] = term2_total
        
        # Calculate k1_1*v and k1_2*v (using linear fitting here)
        i1_v05_total = term1_total / v**0.5
        i2_v05_total = term2_total / v**0.5
        v05 = v**0.5
        
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress([v05], [i1_v05_total])
        slope2, intercept2, r_value2, p_value2, std_err2 = linregress([v05], [i2_v05_total])
        
        k1_1_v_surface[i, j] = slope1 * v
        k1_2_v_surface[i, j] = slope2 * v

# Create a 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the surfaces of i1_total and i2_total
surf1 = ax.plot_surface(Eap_mesh, v_mesh, i1_total_surface, color='purple', edgecolor='none', alpha=0.7)
surf2 = ax.plot_surface(Eap_mesh, v_mesh, i2_total_surface, color='purple', edgecolor='none', alpha=0.7)

# Set axis labels
ax.set_xlabel('Potential (V)', fontsize=24, labelpad=20)  # Adjust labelpad to increase spacing
ax.set_ylabel('Scan Rate (V/s)', fontsize=24, labelpad=20)
ax.set_zlabel('Current (A)', fontsize=24, labelpad=20)

# Set the size of the ticks for each axis
ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick size
ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick size
ax.tick_params(axis='z', labelsize=24)  # Adjust z-axis tick size

# Display the graph
plt.tight_layout()
plt.show()