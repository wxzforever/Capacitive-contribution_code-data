import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =========================
# Constants
# =========================
T = 298.15
F = 96485.3329
R = 8.314

# =========================
# Dunn fitting uses 10 scan rates (same as your original logic)
# =========================
v_values = [0.01, 0.02, 0.03, 0.04, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.1]
v_index = 0  # only show one scan-rate curve (v=0.01)

Ei = 0
Ef = 1

# =========================
# EDLC parameters (single values, manual change)
# =========================
Rs = 200    # extremely small Rs, manual change
Rp = 100000
Cd = 0.04
AEDLCOX = 1
AEDLCRED = 1

# =========================
# Diffusion parameters (single values, manual change)
# =========================
E0 = 0.5
b = 0.7          # <-- YOU change b here (e.g., 0.5 or 1.0)

# =========================
# Amplitude scale (keep your original prefactor style)
# =========================
n = 1
Gamma = 1e-6
S = 1
A_diff = n * F * S * Gamma

# =========================
# Time functions (unchanged)
# =========================
def tcharge1(Eap, v, Ei):
    return (Eap - Ei) / v

def tcharge2(Ef, v, Ei):
    return (Ef - Ei) / v

def tdischarge(Eap, v, Ef):
    return (Ef - Eap) / v

# =========================
# EDLC current (unchanged; keep *1000 scaling)
# =========================
def IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp):

    term1 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd))) * 1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge1(Eap, v, Ei)
            - Rs * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd)))) * 1000

    term2 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd))) * 1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge2(Ef, v, Ei)
            - Rs * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd)))) * 1000

    term3 = AEDLCRED * (-v) * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd))) * 1000 + \
            AEDLCRED * (-v) * (1 / Rp) * (tdischarge(Eap, v, Ef)
            - Rs * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd)))) * 1000

    return term1, term2 + term3

# =========================
# Diffusion current (forward positive, reverse negative; keep *1000 scaling)
# =========================
def IDiffusion(A_diff, b, F, R, T, v, Eap, E0):

    x = (F / (R * T)) * (Eap - E0)
    g = np.exp(x) / (1.0 + np.exp(x))**2  # max 0.25 at E=E0

    amp = A_diff * (v ** b) * 1000

    I_forward = +amp * g
    I_reverse = -amp * g

    return I_forward, I_reverse

# =========================
# Total current (EDLC + Diffusion)
# =========================
def Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v,
           AEDLCOX, AEDLCRED,
           A_diff, b, F, R, T, E0):

    iedlc_f, iedlc_r = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp)
    idiff_f, idiff_r = IDiffusion(A_diff, b, F, R, T, v, Eap, E0)

    total_forward = iedlc_f + idiff_f
    total_reverse = iedlc_r + idiff_r

    return total_forward, total_reverse

# =========================
# Main calculation (single run)
# =========================
Eap_values = np.linspace(0, 1, 100)
results = []

for Eap in Eap_values:

    i1_total = []
    i2_total = []

    for v in v_values:
        term1_total, term2_total = Itotal(
            Eap, Ei, Ef, Rs, Rp, Cd, v,
            AEDLCOX, AEDLCRED,
            A_diff, b, F, R, T, E0
        )
        i1_total.append(term1_total)
        i2_total.append(term2_total)

    # Dunn fit on TOTAL current (same as your original approach)
    i1_v05 = [i / (v**0.5) for i, v in zip(i1_total, v_values)]
    i2_v05 = [i / (v**0.5) for i, v in zip(i2_total, v_values)]
    v05 = [v**0.5 for v in v_values]

    slope1, _, _, _, _ = linregress(v05, i1_v05)
    slope2, _, _, _, _ = linregress(v05, i2_v05)

    k1_1 = slope1
    k1_2 = slope2

    # only use v_index curve for plotting & area ratio (same logic)
    v_plot = v_values[v_index]
    k1_1_v = k1_1 * v_plot
    k1_2_v = k1_2 * v_plot

    results.append({
        'Eap': Eap,
        'i1_total': i1_total[v_index],
        'i2_total': i2_total[v_index],
        'k1_1_v': k1_1_v,
        'k1_2_v': k1_2_v
    })

# Extract plotting arrays
E_plot  = [r['Eap'] for r in results]
i1_plot = [r['i1_total'] for r in results]
i2_plot = [r['i2_total'] for r in results]
k1_plot = [r['k1_1_v'] for r in results]
k2_plot = [r['k1_2_v'] for r in results]

# Areas (ALLOW >1; do not clip)
a_total = np.trapz(np.abs(np.array(i1_plot) - np.array(i2_plot)), E_plot)
a_cap   = np.trapz(np.abs(np.array(k1_plot) - np.array(k2_plot)), E_plot)

cap_ratio = a_cap / (a_total + 1e-30)

# =========================
# Plot (keep style simple & stable)
# =========================
plt.figure(figsize=(12, 8))

plt.plot(E_plot, i1_plot, color='purple', linewidth=5)
plt.plot(E_plot, i2_plot, color='purple', linewidth=5)

plt.plot(E_plot, k1_plot, color='red', linewidth=5)
plt.plot(E_plot, k2_plot, color='red', linewidth=5)

plt.fill_between(
    E_plot, k1_plot, k2_plot,
    color='red', alpha=0.5, linewidth=5,
    label=f"Capacitive contribution: {cap_ratio*100:.1f}%"
)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Potential (V)', fontsize=24)
plt.ylabel('Current (A)', fontsize=24)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.175), fontsize=32)
plt.savefig("CC3.png", dpi=300, bbox_inches="tight")