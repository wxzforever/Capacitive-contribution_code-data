# The code for generating a heatmap of percent error value from pure EDLC condition with Rs and Rp

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

v_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # List of different scan rates, V/s
Ei = 0  # Initial potential, V
Ef = 1  # Final potential, V

# Variables for EDLC model
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
    term1 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd))) + \
            AEDLCOX * v * (1 / Rp) * (tcharge1(Eap, v, Ei) - Rs * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd))))

    term2 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd))) + \
            AEDLCOX * v * (1 / Rp) * (tcharge2(Ef, v, Ei) - Rs * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd))))

    term3 = AEDLCRED * (-v) * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd))) + \
            AEDLCRED * (-v) * (1 / Rp) * (tdischarge(Eap, v, Ef) - Rs * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd))))

    return term1, term2 + term3

# Generate Eap value
Eap_values = np.linspace(0, 1, 100)

# Generate Rs value
Rs_values = np.linspace(0, 110, 100)

# Generate Rp value
Rp_values = np.linspace(0, 10000, 100)

# Initialize storage for capacitive contributions
percent_errors = np.zeros((len(Rs_values), len(Rp_values)))

# Calculate the capacitive contribution corresponding to each set of Rs and Rp values
for i, Rs in tqdm(enumerate(Rs_values), total=len(Rs_values)):
    for j, Rp in enumerate(Rp_values):
        results = []
        for Eap in Eap_values:
            i1 = []
            i2 = []
            for v in v_values:
                term1, term2_plus_term3 = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp)
                i1.append(term1)
                i2.append(term2_plus_term3)

            i1_v05 = [current / v**0.5 for current, v in zip(i1, v_values)]
            i2_v05 = [current / v**0.5 for current, v in zip(i2, v_values)]
            v05 = [v**0.5 for v in v_values]

            slope1, intercept1, r_value1, p_value1, std_err1 = linregress(v05, i1_v05)
            k1_1 = slope1

            slope2, intercept2, r_value2, p_value2, std_err2 = linregress(v05, i2_v05)
            k1_2 = slope2

            # Select a specific scan rate
            v_index = 0  # Select the index order here, starting from 0
            k1_1_v = k1_1 * v_values[v_index]
            k1_2_v = k1_2 * v_values[v_index]

            results.append({
                'Eap': Eap,
                'i1': i1[v_index],
                'i2': i2[v_index],
                'k1_1_v': k1_1_v,
                'k1_2_v': k1_2_v
            })

        Eap_plot = [res['Eap'] for res in results]
        i1_plot = [res['i1'] for res in results]
        i2_plot = [res['i2'] for res in results]
        k1_1_v_plot = [res['k1_1_v'] for res in results]
        k1_2_v_plot = [res['k1_2_v'] for res in results]

        # Calculate the area between i1 and i2 (use absolute value to ensure non-negative)
        delta_i = np.abs(np.array(i1_plot) - np.array(i2_plot))
        a1 = np.trapz(delta_i, Eap_plot)

        # Calculate the area between k1_1*v and k1_2*v (use absolute value to ensure non-negative)
        delta_k = np.abs(np.array(k1_1_v_plot) - np.array(k1_2_v_plot))
        a2 = np.trapz(delta_k, Eap_plot)

        # Calculate percent error
        percent_error = np.abs(a2 / a1 - 1)
        percent_errors[i, j] = percent_error

cmap0 = mpl.colors.LinearSegmentedColormap.from_list(
        'red2green', ['green', 'orange','red'])

# Plot a heat map
plt.figure(figsize=(12, 8))
plt.contourf(Rp_values, Rs_values, percent_errors, levels=100, cmap=cmap0)
cbar = plt.colorbar()
cbar.set_label('Percent error', fontsize=24)
cbar.ax.tick_params(labelsize=24)  # Set the scale font size
plt.xlabel('Rp (Parallel resistance, ohm)', fontsize=24)
plt.ylabel('Rs (Series resistance, ohm)', fontsize=24)
#plt.title('100±10 F/g (8 mF for approx 60 μg of active material)', fontsize=17)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.show()