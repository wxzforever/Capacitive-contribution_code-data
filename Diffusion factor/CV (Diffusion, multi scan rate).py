# The code for generating CVs from EDLC+Diffusion condition with different b-values (v fixed)
# Plot style and structure kept identical to original Faradaic CV script

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Physical constants
# =========================

T = 298              # K
F = 96485.3329       # C/mol
R = 8.314            # J/mol/K

# =========================
# Fixed scan condition
# =========================

v = 0.01             # V/s (FIXED, only b changes)

# Explicit b list (as requested)
b_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Ei = 0
Ef = 1
E0 = 0.5             # diffusion peak center

# =========================
# EDLC parameters (UNCHANGED)
# =========================

Rs = 80
Rp = 100000
Cd = 0.04

AEDLCOX = 1
AEDLCRED = 1

# =========================
# Amplitude reference (kept from Faradaic block)
# =========================

n = 1
Gamma = 1e-6
S = 1

# =========================
# Time definitions (UNCHANGED)
# =========================

def tcharge1(Eap, v, Ei):
    return (Eap - Ei) / v

def tcharge2(Ef, v, Ei):
    return (Ef - Ei) / v

def tdischarge(Eap, v, Ef):
    return (Ef - Eap) / v

# =========================
# EDLC current (UNCHANGED)
# =========================

def IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, Rp):

    term1 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd))) * 1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge1(Eap, v, Ei) -
            Rs * Cd * (1 - np.exp(-tcharge1(Eap, v, Ei) / (Rs * Cd)))) * 1000

    term2 = AEDLCOX * v * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd))) * 1000 + \
            AEDLCOX * v * (1 / Rp) * (tcharge2(Ef, v, Ei) -
            Rs * Cd * (1 - np.exp(-tcharge2(Ef, v, Ei) / (Rs * Cd)))) * 1000

    term3 = AEDLCRED * (-v) * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd))) * 1000 + \
            AEDLCRED * (-v) * (1 / Rp) * (tdischarge(Eap, v, Ef) -
            Rs * Cd * (1 - np.exp(-tdischarge(Eap, v, Ef) / (Rs * Cd)))) * 1000

    return term1 - 0.2, term2 + term3 - 0.2


# =========================
# Diffusion / pseudo current
# =========================

def IDiffusion(A_diff, b, F, R, T, v, Eap, E0):

    x = (F / (R * T)) * (Eap - E0)

    g = np.exp(x) / (1.0 + np.exp(x))**2   # peak shape

    # IMPORTANT: same scaling system as EDLC (mA → A)
    amp = A_diff * (v ** b) * 1000

    I_forward = +amp * g
    I_reverse = -amp * g

    return I_forward, I_reverse


# =========================
# Total current
# =========================

def Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v,
           AEDLCOX, AEDLCRED,
           A_diff, b, F, R, T, E0):

    iedlc_f, iedlc_r = IEDLC(Eap, Ei, Ef, Rs, Cd, v,
                             AEDLCOX, AEDLCRED, Rp)

    idiff_f, idiff_r = IDiffusion(A_diff, b, F, R, T, v, Eap, E0)

    return iedlc_f + idiff_f, iedlc_r + idiff_r


# =========================
# Diffusion amplitude scale
# =========================

A_diff = n * F * S * Gamma


# =========================
# Plot CV (style unchanged)
# =========================

plt.figure(figsize=(12, 8))

Eap_values = np.linspace(Ei, Ef, 100)

for b in b_values:

    Itotal_values = [Itotal(Eap, Ei, Ef, Rs, Rp, Cd, v,
                            AEDLCOX, AEDLCRED,
                            A_diff, b, F, R, T, E0)
                     for Eap in Eap_values]

    Itotal_values = np.array(Itotal_values)

    combined_I = np.concatenate([Itotal_values[:, 0],
                                 [np.nan],
                                 Itotal_values[:, 1]])

    combined_E = np.concatenate([Eap_values,
                                 [np.nan],
                                 Eap_values])

    plt.plot(combined_E, combined_I, linewidth=5, color='black')


plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Potential (V)', fontsize=24)
plt.ylabel('Current (A)', fontsize=24)

plt.show()
