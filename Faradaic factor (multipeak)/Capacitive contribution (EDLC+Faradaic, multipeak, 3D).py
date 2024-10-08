import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress

# General parameters
T = 298.15  # Kelvin temperature
F = 96485.3329  # Faraday constant
R = 8.314  # Ideal gas constant

v_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  # Multiple scan rates, V/s
Ei = -0.8  # Initial potential, V
Ef = 0.8  # Final potential, V

# Variables for EDLC model
Rs = 200  # Series resistance, ohm
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

# E0 values for Faradaic calculation (multiple peaks)
E0_values = np.linspace(0.5, -0.5, 2)

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
def IFarad(n, F, S, k, Gamma, alpha, R, T, v, Eap, E0_values):
    I_forward = 0
    I_reverse = 0
    for E0 in E0_values:
        I_forward += (n * F * S * k * Gamma *
                      np.exp(alpha * n * F / (R * T) * (Eap - E0)) /
                      np.exp((R * T) / (F * alpha * n) * k / v *
                             np.exp(alpha * n * F / (R * T) * (Eap - E0))))
        
        I_reverse += (-n * F * S * k * Gamma *
                      np.exp(alpha * n * F / (R * T) * -(Eap - E0)) /
                      np.exp((R * T) / (F * alpha * n) * k / v *
                             np.exp(alpha * n * F / (R * T) * -(Eap - E0))))
    
    return I_forward, I_reverse

# Total current function (EDLC + Faradaic)
def Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED, n, F, S, k, Gamma, alpha, R, T, E0_values):
    iedlc1, iedlc2 = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp)
    ifarad_forward, ifarad_reverse = IFarad(n, F, S, k, Gamma, alpha, R, T, v, Eap, E0_values)
    total_forward = iedlc1 + ifarad_forward
    total_reverse = iedlc2 + ifarad_reverse
    return total_forward, total_reverse

# Generate all Eap values
Eap_values = np.linspace(Ei, Ef, 100)

# Initialize and store all Eap results
i1_total_surface = np.zeros((len(v_values), len(Eap_values)))
i2_total_surface = np.zeros((len(v_values), len(Eap_values)))
k1_1_v_surface = np.zeros((len(v_values), len(Eap_values)))
k1_2_v_surface = np.zeros((len(v_values), len(Eap_values)))

# Traverse all Eap values ​​and calculate i1 and i2 for total current (EDLC + Faradaic)
for j, Eap in enumerate(Eap_values):
    i1_total = []
    i2_total = []
    for i, v in enumerate(v_values):
        term1_total, term2_total = Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED, n, F, S, k, Gamma, alpha, R, T, E0_values)
        i1_total.append(term1_total)
        i2_total.append(term2_total)
    
    # Calculate i1/v^0.5 and i2/v^0.5 for total current
    i1_v05_total = [current / v**0.5 for current, v in zip(i1_total, v_values)]
    i2_v05_total = [current / v**0.5 for current, v in zip(i2_total, v_values)]
    
    # Calculate v^0.5
    v05 = [v**0.5 for v in v_values]
    
    # Linear fit for i1_v05 and v05 for total current
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(v05, i1_v05_total)
    k1_1 = slope1
    
    # Linear fit for i2_v05 and v05 for total current
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(v05, i2_v05_total)
    k1_2 = slope2
    
    # Calculate k1_1*v and k1_2*v for total current
    k1_1_v = [k1_1 * v for v in v_values]
    k1_2_v = [k1_2 * v for v in v_values]
    
    # Storing surface results
    i1_total_surface[:, j] = i1_total
    i2_total_surface[:, j] = i2_total
    k1_1_v_surface[:, j] = k1_1_v
    k1_2_v_surface[:, j] = k1_2_v

# Create mesh for scan rates and Eap
Eap_mesh, v_mesh = np.meshgrid(Eap_values, v_values)

# Plot the 3D results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot i1_total and i2_total surfaces
surf1 = ax.plot_surface(Eap_mesh, v_mesh, i1_total_surface, color='purple', edgecolor='none', alpha=0.7)
surf2 = ax.plot_surface(Eap_mesh, v_mesh, i2_total_surface, color='purple', edgecolor='none', alpha=0.7)

# Plot k1_1*v and k1_2*v surfaces
surf3 = ax.plot_surface(Eap_mesh, v_mesh, k1_1_v_surface, color='green', edgecolor='none', alpha=0.7)
surf4 = ax.plot_surface(Eap_mesh, v_mesh, k1_2_v_surface, color='green', edgecolor='none', alpha=0.7)

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