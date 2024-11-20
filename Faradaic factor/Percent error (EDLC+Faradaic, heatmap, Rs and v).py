# The code for generating a heatmap of percent error value from EDLC+Faradaic condition with Rs and v

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib as mpl

# General parameters
T = 298.15  # Kelvin temperature
F = 96485.3329  # Faraday constant
R = 8.314  # Ideal gas constant

v_values = np.linspace(0.01, 0.1, 10)     # Multiple scan rates, V/s
Ei = 0  # Initial potential, V
Ef = 1  # Final potential, V

# Variables for EDLC model
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

# Generate Rs values
Rs_values = np.linspace(0, 200, 100)  # Range of series resistance, ohms

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

# Generate all Eap values
Eap_values = np.linspace(0, 1, 100)

# Initialize storage for percent errors
percent_errors = np.zeros((len(Rs_values), len(v_values)))

# Main loop for calculating percent errors for different Rs and v
for j, v in tqdm(enumerate(v_values), total=len(v_values)):
    for i, Rs in enumerate(Rs_values):
        results = []
        for Eap in Eap_values:
            i1_total = []
            i2_total = []
            for v_current in v_values:
                # Calculate total current
                term1_total, term2_total = Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v_current, AEDLCOX, AEDLCRED, n, F, S, k, Gamma, alpha, R, T)
                i1_total.append(term1_total)
                i2_total.append(term2_total)

            # Calculate i1/v^0.5 and i2/v^0.5 for total current
            i1_v05_total = [current / v_current**0.5 for current, v_current in zip(i1_total, v_values)]
            i2_v05_total = [current / v_current**0.5 for current, v_current in zip(i2_total, v_values)]
            v05 = [v_current**0.5 for v_current in v_values]

            # Linear fit for total current
            slope1, intercept1, r_value1, p_value1, std_err1 = linregress(v05, i1_v05_total)
            k1_1 = slope1

            slope2, intercept2, r_value2, p_value2, std_err2 = linregress(v05, i2_v05_total)
            k1_2 = slope2

            # Calculate k1_1*v and k1_2*v for total current
            k1_1_v = k1_1 * v_values[0]
            k1_2_v = k1_2 * v_values[0]

            results.append({
                'Eap': Eap,
                'k1_1_v': k1_1_v,
                'k1_2_v': k1_2_v,
                'i1_total': i1_total[0],  # Store total current
                'i2_total': i2_total[0]   # Store total current
            })

        # Extract plotting data for total current
        i1_total_plot = [res['i1_total'] for res in results]
        i2_total_plot = [res['i2_total'] for res in results]
        Eap_plot = [res['Eap'] for res in results]  # Eap_plot should be defined here

        # Calculate the area between i1 and i2 for total current (use absolute value to ensure non-negative)
        delta_i_total = np.abs(np.array(i1_total_plot) - np.array(i2_total_plot))
        a1 = np.trapz(delta_i_total, Eap_plot)

        # Calculate the area between k1_1*v and k1_2*v for EDLC
        k1_1_v_plot = [res['k1_1_v'] for res in results]
        k1_2_v_plot = [res['k1_2_v'] for res in results]
        delta_k_EDLC = np.abs(np.array(k1_1_v_plot) - np.array(k1_2_v_plot))
        a2 = np.trapz(delta_k_EDLC, Eap_plot)

        # Calculate percent error
        percent_error = np.abs(a2 / a1 - 1) / 1
        percent_errors[i, j] = percent_error

# Plotting the heatmap
cmap0 = mpl.colors.LinearSegmentedColormap.from_list('red2green', ['green', 'orange', 'red'])
plt.figure(figsize=(12, 8))
plt.contourf(v_values, Rs_values, percent_errors, levels=100, cmap=cmap0)
plt.xlabel('Scan Rate (v, V/s)', fontsize=24)
plt.ylabel('Series Resistance (Rs, ohm)', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
cbar = plt.colorbar(label='Percent Error')
cbar.set_label('Percent Error', fontsize=24)
cbar.ax.tick_params(labelsize=24)
plt.show()