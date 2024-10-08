# The code for generating capacitive contribution percentage from pure EDLC condition with one Rs and one Rp values

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# General parameters
T = 298.15  # Kelvin temperature
F = 96485.3329  # Faraday constant
R = 8.314  # Ideal gas constant

v_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # List of different scan rates, V/s
Ei = 0  # Initial potential, V
Ef = 1  # Final potential, V

# Variables for EDLC model
Rs = 60 # series resistance, ohm
Rp = 100000 # Parallel resistance
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

# Generate all Eap values
Eap_values = np.linspace(0, 1, 100)

# Initialize and store all Eap results
results = []

# Traverse all Eap values ​​and calculate i1 and i2
for Eap in Eap_values:
    i1 = []
    i2 = []
    for v in v_values:
        term1, term2_plus_term3 = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp)
        i1.append(term1)
        i2.append(term2_plus_term3)
    
    # Calculate i1/v^0.5 and i2/v^0.5
    i1_v05 = [current / v**0.5 for current, v in zip(i1, v_values)]
    i2_v05 = [current / v**0.5 for current, v in zip(i2, v_values)]
    
    # Calculate v^0.5
    v05 = [v**0.5 for v in v_values]
    
    # Linear fit for i1_v05 and v05
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(v05, i1_v05)
    k1_1 = slope1
    
    # Linear fit for i2_v05 and v05
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(v05, i2_v05)
    k1_2 = slope2
    
    # Calculate k1_1*v and k1_2*v
    k1_1_v = k1_1 * 1
    k1_2_v = k1_2 * 1
    
    # Select a specific scan rate (index starts at 0)
    v_index = 0
    k1_1_v = k1_1 * v_values[v_index]
    k1_2_v = k1_2 * v_values[v_index]
    
    # Storing results
    results.append({
        'Eap': Eap,
        'i1': i1[v_index],  # Store i1 at the selected scan rate
        'i2': i2[v_index],  # Store i2 at the selected scan rate
        'k1_1_v': k1_1_v,
        'k1_2_v': k1_2_v
    })

# Extract plotting data
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

# Calculate capacitive contribution
capacitive_contribution = a2 / a1

# Plotting
plt.figure(figsize=(12, 8))

# Plot i1 and i2
plt.plot(Eap_plot, i1_plot, color='purple',linewidth=5)
plt.plot(Eap_plot, i2_plot, color='purple',linewidth=5)

# Plot k1_1*v and k1_2*v
plt.plot(Eap_plot, k1_1_v_plot, color='green',linewidth=5)
plt.plot(Eap_plot, k1_2_v_plot, color='green',linewidth=5)

# Fill the area between k1_1*v and k1_2*v
plt.fill_between(Eap_plot, k1_1_v_plot, k1_2_v_plot, color='green', alpha=0.5, linewidth=5, label=f'Capacitive Contribution: {capacitive_contribution*100:.1f}%')

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Potential (V)', fontsize=24)
plt.ylabel('Current (mA)', fontsize=24)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.175), fontsize=32)
plt.show()