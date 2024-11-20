import numpy as np
import matplotlib.pyplot as plt

# Constants
T = 298.15  # Temperature, K
F = 96485.3329  # Faraday constant, C/mol
R = 8.314  # Gas constant, J/(mol*K)

# Scan rate
v = 0.1  # Scan rate, V/s

# Initial and final potentials
Ei = -1  # Initial potential, V
Ef = 2  # Final potential, V

# Battery model parameters
n = 1  # Number of electron transfers
S = 1  # Electrode surface area, cm^-2
k = 0.1  # Standard rate constant
Gamma = 3.15e-4  # Surface concentration, mol·cm^-2
alpha = 0.5  # Charge transfer coefficient

# EDLC model parameters
EiEDLC = 0  # Initial EDLC potential, V
EfEDLC = 0  # Final EDLC potential, V
Rs = 20  # Ohmic resistance, ohm
Cd = 0.2 # Capacitance, F
AEDLCOX = 1  # Amplitude factor of positive EDLC
AEDLCRED = 1  # Amplitude factor of negative EDLC

# Pseudocapacitor model parameters
E0pseuOXi = 0.1  # Initial oxidation pseudo potential, V
E0pseuOXf = 1.2  # Final oxidation pseudo potential, V
E0pseuREDi = 0  # Initial reduction pseudo potential, V
E0pseuREDf = 1.1  # Final reduction pseudo potential, V
GpseuOX = 0.3  # Oxidation peak width, V
GpseuRED = 0.2  # Reduction peak width, V
ApseuOX = 4/100  # Amplitude factor of the oxidation current
ApseuRED = 3.3/100  # Amplitude factor of the reduction current
n1 = 30  # Number of oxidation centers
n2 = 30  # Number of reduction centers
epsilon1 = (E0pseuOXf - E0pseuOXi) / n1  # Hopping potential between oxidation centers, V
epsilon2 = (E0pseuREDf - E0pseuREDi) / n2  # Hopping potential between reduction centers, V

# Time functions for EDLC
def tcharge(Eap, v, Ei):
    return (Eap - Ei) / v

def tdis(Eap, v, Ef):
    return (Ef - Eap) / v

# EDLC current function
def IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED):
    t_charge = tcharge(Eap, v, Ei)
    t_discharge = tdis(Eap, v, Ef)
    
    ImaxEDLC = AEDLCOX * v * Cd * (1 - np.exp(-tcharge(Ef, v, Ei) / (Rs * Cd)))
    
    I_pos = AEDLCOX * v * Cd * (1 - np.exp(-t_charge / (Rs * Cd))) - ImaxEDLC / 2
    I_neg = (AEDLCOX * v * Cd * (1 - np.exp(-tcharge(Ef, v, Ei) / (Rs * Cd))) - ImaxEDLC / 2) + \
            AEDLCRED * (-v * Cd * (1 - np.exp(-t_discharge / (Rs * Cd))))
    
    return [I_pos, I_neg]

# Battery current function
def IFarad(n, F, S, k, Gamma, alpha, R, T, v, Eap):
    I_forward = (n * F * S * k * Gamma *
                 np.exp(alpha * n * F / (R * T) * (Eap - 0.2)) /
                 np.exp((R * T) / (F * alpha * n) * k / v *
                        np.exp(alpha * n * F / (R * T) * (Eap - 0.2))))
    
    
    I_reverse = (-n * F * S * k * Gamma *
                 np.exp(alpha * n * F / (R * T) * -(Eap - 0.5)) /
                 np.exp((R * T) / (F * alpha * n) * k / v *
                        np.exp(alpha * n * F / (R * T) * -(Eap - 0.5))))
    
    return I_forward, I_reverse

# Pseudocapacitor oxidation and reduction current functions
def PsiPseudoOX(Eap, GpseuOX, E0pseuOXi, E0pseuOXf, n1, epsilon1):
    sum_pseudo_ox = 0.0
    for n in range(n1 + 1):
        E_center = E0pseuOXi + n * epsilon1
        exponent = F / (R * T) * (Eap - E_center)
        term = (R * T) / (2 * F * GpseuOX) * np.exp(exponent) * (
            np.exp(F / (R * T) * GpseuOX) / (1 + np.exp(F / (R * T) * GpseuOX) * np.exp(F / (R * T) * (Eap - E_center)))
            - np.exp(-F / (R * T) * GpseuOX) / (1 + np.exp(-F / (R * T) * GpseuOX) * np.exp(F / (R * T) * (Eap - E_center)))
        )
        sum_pseudo_ox += term
    return sum_pseudo_ox

def PsiPseudoRED(Eap, GpseuRED, E0pseuREDi, E0pseuREDf, n2, epsilon2):
    sum_pseudo_red = 0.0
    for n in range(n2 + 1):
        E_center = E0pseuREDi + n * epsilon2
        exponent = F / (R * T) * (Eap - E_center)
        term = (-R * T) / (2 * F * GpseuRED) * np.exp(exponent) * (
            np.exp(F / (R * T) * GpseuRED) / (1 + np.exp(F / (R * T) * GpseuRED) * np.exp(F / (R * T) * (Eap - E_center)))
            - np.exp(-F / (R * T) * GpseuRED) / (1 + np.exp(-F / (R * T) * GpseuRED) * np.exp(F / (R * T) * (Eap - E_center)))
        )
        sum_pseudo_red += term
    return sum_pseudo_red

# Pseudocapacitor current function
def IPseudo(Eap, GpseuOX, GpseuRED, E0pseuOXi, E0pseuOXf, E0pseuREDi, E0pseuREDf, ApseuOX, ApseuRED, v, n1, n2, epsilon1, epsilon2):
    psi_ox = PsiPseudoOX(Eap, GpseuOX, E0pseuOXi, E0pseuOXf, n1, epsilon1)
    psi_red = PsiPseudoRED(Eap, GpseuRED, E0pseuREDi, E0pseuREDf, n2, epsilon2)
    
    I_pseudo_ox = ApseuOX * np.sqrt(v) * psi_ox
    I_pseudo_red = ApseuRED * np.sqrt(v) * psi_red
    
    return [I_pseudo_ox, I_pseudo_red]

# Total current function (combining EDLC, Battery, and Pseudocapacitor)
def Itotal(Eap, n, F, S, k, Gamma, alpha, R, T, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, GpseuOX, GpseuRED, E0pseuOXi, E0pseuOXf, E0pseuREDi, E0pseuREDf, ApseuOX, ApseuRED, n1, n2, epsilon1, epsilon2):
    I_batt = IFarad(n, F, S, k, Gamma, alpha, R, T, v, Eap)
    I_edlc = IEDLC(Eap, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED)
    I_pseudo = IPseudo(Eap, GpseuOX, GpseuRED, E0pseuOXi, E0pseuOXf, E0pseuREDi, E0pseuREDf, ApseuOX, ApseuRED, v, n1, n2, epsilon1, epsilon2)
    
    I_total_ox = 0*I_batt[0] + I_edlc[0] + 0.4*I_pseudo[0]
    I_total_red = 0*I_batt[1] + I_edlc[1] + 0.4*I_pseudo[1]
    
    return [I_total_ox*1000, I_total_red*1000]

# Generate and plot the total current
Eap_values = np.linspace(Ei, Ef, 100)  # Eap values from Ei to Ef
i1_total_plot = []
i2_total_plot = []

for Eap in Eap_values:
    I_total = Itotal(Eap, n, F, S, k, Gamma, alpha, R, T, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, GpseuOX, GpseuRED, E0pseuOXi, E0pseuOXf, E0pseuREDi, E0pseuREDf, ApseuOX, ApseuRED, n1, n2, epsilon1, epsilon2)
    i1_total_plot.append(I_total[0])
    i2_total_plot.append(I_total[1])

# Plotting I-E curve
plt.figure(figsize=(12, 8))
plt.plot(Eap_values, i1_total_plot, color='Purple', linewidth=5)
plt.plot(Eap_values, i2_total_plot, color='Purple', linewidth=5)
plt.xlabel('Potential (V)', fontsize=24)
plt.ylabel('Current (mA)', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()