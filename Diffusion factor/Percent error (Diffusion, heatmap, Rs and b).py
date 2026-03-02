# Heatmap of percent error value from EDLC+Diffusion condition with Rs and b
# (Keep the original heatmap logic/structure; only replace Faradaic->Diffusion and v-axis->b-axis)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib as mpl

# General parameters (UNCHANGED style)
T = 298.15  # Kelvin temperature
F = 96485.3329  # Faraday constant
R = 8.314  # Ideal gas constant

# Dunn fitting still uses 10 scan rates (UNCHANGED)
v_values = np.linspace(0.01, 0.1, 10)     # Multiple scan rates, V/s
Ei = 0  # Initial potential, V
Ef = 1  # Final potential, V

# Variables for EDLC model (UNCHANGED)
Rp = 100000  # Parallel resistance, ohm
Cd = 0.04  # Capacitance, F
AEDLCOX = 1  # Amplitude factor of positive EDLC
AEDLCRED = 1  # Amplitude factor of negative EDLC

# Variables kept from original Faradaic block (we only use them to define A_diff scale, keep as-is)
n = 1  # Number of electron transfers
Gamma = 1e-6
S = 1  # Electrode surface area

# Diffusion parameters (fixed as previously confirmed)
E0 = 0.5  # peak center
# b values for heatmap axis (explicit, clear)
b_values = np.linspace(0.4, 1.0, 301)

# Generate Rs values (UNCHANGED)
Rs_values = np.linspace(0, 200, 500)  # Range of series resistance, ohms

# Time functions (UNCHANGED)
def tcharge1(Eap, v, Ei):
    return (Eap - Ei) / v

def tcharge2(Ef, v, Ei):
    return (Ef - Ei) / v

def tdischarge(Eap, v, Ef):
    return (Ef - Eap) / v

# EDLC current function (UNCHANGED; keep *1000 scaling)
def IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp):
    term1 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd))) * 1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge1(Eap, v, Ei) - Rs * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd)))) * 1000

    term2 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd))) * 1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge2(Ef, v, Ei) - Rs * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd)))) * 1000

    term3 = AEDLCRED * (-v) * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd))) * 1000 + \
            AEDLCRED * (-v) * (1 / Rp) * (tdischarge(Eap, v, Ef) - Rs * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd)))) * 1000

    return term1, term2 + term3

# Diffusion current function (REPLACES Faradaic)
def IDiffusion(A_diff, b, F, R, T, v, Eap, E0):
    x = (F / (R * T)) * (Eap - E0)
    g = np.exp(x) / (1.0 + np.exp(x))**2  # max=0.25 at E=E0
    amp = A_diff * (v ** b) * 1000
    I_forward = +amp * g
    I_reverse = -amp * g
    return I_forward, I_reverse

# Total current function (EDLC + Diffusion)
def Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v, AEDLCOX, AEDLCRED, A_diff, b, F, R, T, E0):
    iedlc1, iedlc2 = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp)
    idiff_forward, idiff_reverse = IDiffusion(A_diff, b, F, R, T, v, Eap, E0)
    total_forward = iedlc1 + idiff_forward
    total_reverse = iedlc2 + idiff_reverse
    return total_forward, total_reverse

# Amplitude scale (kept)
A_diff = n * F * S * Gamma

# Generate all Eap values (UNCHANGED)
Eap_values = np.linspace(0, 1, 100)

# Initialize storage for percent errors: rows=Rs, cols=b
percent_errors = np.zeros((len(Rs_values), len(b_values)))

# Reference capacitive contribution (fixed)
ref_cap = 0.44

# Main loop for calculating percent errors for different Rs and b
for j, b in tqdm(enumerate(b_values), total=len(b_values)):
    for i, Rs in enumerate(Rs_values):

        results = []

        for Eap in Eap_values:
            i1_total = []
            i2_total = []

            for v_current in v_values:
                term1_total, term2_total = Itotal(
                    Eap, Ei, Ef, Rs, Rp, Cd, v_current, AEDLCOX, AEDLCRED,
                    A_diff, b, F, R, T, E0
                )
                i1_total.append(term1_total)
                i2_total.append(term2_total)

            # Dunn fit on total current (UNCHANGED form)
            i1_v05_total = [current / v_current**0.5 for current, v_current in zip(i1_total, v_values)]
            i2_v05_total = [current / v_current**0.5 for current, v_current in zip(i2_total, v_values)]
            v05 = [v_current**0.5 for v_current in v_values]

            slope1, intercept1, r_value1, p_value1, std_err1 = linregress(v05, i1_v05_total)
            k1_1 = slope1

            slope2, intercept2, r_value2, p_value2, std_err2 = linregress(v05, i2_v05_total)
            k1_2 = slope2

            k1_1_v = k1_1 * v_values[0]
            k1_2_v = k1_2 * v_values[0]

            results.append({
                'Eap': Eap,
                'k1_1_v': k1_1_v,
                'k1_2_v': k1_2_v,
                'i1_total': i1_total[0],
                'i2_total': i2_total[0]
            })

        i1_total_plot = [res['i1_total'] for res in results]
        i2_total_plot = [res['i2_total'] for res in results]
        Eap_plot = [res['Eap'] for res in results]

        delta_i_total = np.abs(np.array(i1_total_plot) - np.array(i2_total_plot))
        a1 = np.trapz(delta_i_total, Eap_plot)

        k1_1_v_plot = [res['k1_1_v'] for res in results]
        k1_2_v_plot = [res['k1_2_v'] for res in results]
        delta_k_EDLC = np.abs(np.array(k1_1_v_plot) - np.array(k1_2_v_plot))
        a2 = np.trapz(delta_k_EDLC, Eap_plot)

        cap_ratio = a2 / (a1 + 1e-30)

        # Original error
        percent_error = np.abs(cap_ratio - ref_cap) / ref_cap

        # >>> ONLY NEW RULE YOU REQUESTED <<<
        # If capacitive contribution is larger than ref (=0.2), force it to be fully red.
        if cap_ratio > ref_cap:
            percent_error = 1.0

        percent_errors[i, j] = percent_error

# Plotting the heatmap (keep original style/cmap)
cmap0 = mpl.colors.LinearSegmentedColormap.from_list('red2green', ['green', 'orange', 'red'])
plt.figure(figsize=(12, 8))
plt.contourf(Rs_values, b_values, percent_errors.T, levels=100, cmap=cmap0, vmin=0, vmax=1)
plt.xlabel('Series Resistance (Rs, ohm)', fontsize=24)
plt.ylabel('b', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
cbar = plt.colorbar(label='Percent Error')
cbar.set_label('Percent Error', fontsize=24)
cbar.ax.tick_params(labelsize=24)
plt.savefig("heatmap3.png", dpi=300, bbox_inches="tight")