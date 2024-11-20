import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Constants
T = 298.15  # Temperature, K
F = 96485.3329  # Faraday constant, C/mol
R = 8.314  # Gas constant, J/(mol*K)

# Scan rate
v_values = np.linspace(0.005, 0.05, 10)  # Scan rates, V/s

# Initial and final potentials
Ei = 0  # Initial potential, V
Ef = 1.2  # Final potential, V

# Battery model parameters
n = 1  # Number of electron transfers
F = 96485  # Faraday constant, C·mol^-1
S = 1  # Electrode surface area, cm^-2
k = 0.02  # Standard rate constant
Gamma = 3.15*10**-4  # Surface concentration, mol·cm^-2
alpha = 0.5  # Charge transfer coefficient
R = 8.314  # Ideal gas constant, J·mol^-1·K^-1
T = 298  # Temperature, K

# EDLC model parameters
Rs = 50  # Series resistance, ohm
Cd = 2.5  # Capacitance, F
AEDLCOX = 0.05  # Amplitude factor of positive EDLC
AEDLCRED = 0.05  # Amplitude factor of negative EDLC

# Pseudocapacitor model parameters
E0pseuOXi = 0.6  # Initial oxidation pseudo potential, V
E0pseuOXf = 1  # Final oxidation pseudo potential, V
E0pseuREDi = 1  # Initial reduction pseudo potential, V
E0pseuREDf = 0.1  # Final reduction pseudo potential, V
GpseuOX = 0.05  # Oxidation peak width, V
GpseuRED = 0.05  # Reduction peak width, V
ApseuOX = 0.008  # Amplitude factor of the oxidation current
ApseuRED = 0.0115  # Amplitude factor of the reduction current
n1 = 29  # Number of oxidation centers
n2 = 29  # Number of reduction centers
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
                 np.exp(alpha * n * F / (R * T) * (Eap - 0.6)) /
                 np.exp((R * T) / (F * alpha * n) * k / v *
                        np.exp(alpha * n * F / (R * T) * (Eap - 0.6))))
    
    I_reverse = (-n * F * S * k * Gamma *
                 np.exp(alpha * n * F / (R * T) * -(Eap - 0.6)) /
                 np.exp((R * T) / (F * alpha * n) * k / v *
                        np.exp(alpha * n * F / (R * T) * -(Eap - 0.6))))+0.2*(-n * F * S * k * Gamma *
                 np.exp(alpha * n * F / (R * T) * -(Eap - 0.9)) /
                 np.exp((R * T) / (F * alpha * n) * k / v *
                        np.exp(alpha * n * F / (R * T) * -(Eap - 0.9))))
    
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
    
    I_total_ox = I_batt[0] + I_edlc[0] + 500*I_pseudo[0]
    I_total_red = I_batt[1] + I_edlc[1] + 500*I_pseudo[1]
    
    return [I_total_ox, I_total_red]

# Traverse and store all Eap results
Eap_values = np.linspace(Ei, Ef, 100)
results = []

# Traverse all Eap values and calculate i1 and i2 for total current
for Eap in Eap_values:
    i1_total = []
    i2_total = []
    for v in v_values:
        # Calculate total current for each scan rate
        term1_total, term2_total = Itotal(Eap, n, F, S, k, Gamma, alpha, R, T, Ei, Ef, Rs, Cd, v, AEDLCOX, AEDLCRED, GpseuOX, GpseuRED, E0pseuOXi, E0pseuOXf, E0pseuREDi, E0pseuREDf, ApseuOX, ApseuRED, n1, n2, epsilon1, epsilon2)
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
    
    # Use the largest scan rate for plotting
    v_index = 0
    k1_1_v = k1_1 * v_values[v_index]
    k1_2_v = k1_2 * v_values[v_index]
    
    # Store the results
    results.append({
        'Eap': Eap,
        'i1_total': i1_total[v_index],
        'i2_total': i2_total[v_index],
        'k1_1_v': k1_1_v,
        'k1_2_v': k1_2_v
    })

# Extract plotting data
Eap_plot = [res['Eap'] for res in results]
i1_total_plot = [res['i1_total'] for res in results]
i2_total_plot = [res['i2_total'] for res in results]
k1_1_v_plot = [res['k1_1_v'] for res in results]
k1_2_v_plot = [res['k1_2_v'] for res in results]

# Calculate the area between i1 and i2 (use absolute value to ensure non-negative)
delta_i_total = np.abs(np.array(i1_total_plot) - np.array(i2_total_plot))
a1 = np.trapz(delta_i_total, Eap_plot)

# Calculate the area between k1_1*v and k1_2*v (use absolute value to ensure non-negative)
delta_k_total = np.abs(np.array(k1_1_v_plot) - np.array(k1_2_v_plot))
a2 = np.trapz(delta_k_total, Eap_plot)

# Calculate capacitive contribution
capacitive_contribution = np.abs(a2 / a1)

# Final plot including capacitive contribution
plt.figure(figsize=(12, 8))
plt.plot(Eap_plot, i1_total_plot, color='black', linewidth=15)
plt.plot(Eap_plot, i2_total_plot, color='black', linewidth=15)
plt.plot(Eap_plot, k1_1_v_plot, color='#8ccdbf', linewidth=15)
plt.plot(Eap_plot, k1_2_v_plot, color='#8ccdbf', linewidth=15)
plt.fill_between(Eap_plot, k1_1_v_plot, k1_2_v_plot, color='#8ccdbf', alpha=0.8, label=f'Capacitive contribution: {capacitive_contribution*100:.1f}%')

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Potential (V)', fontsize=24)
plt.ylabel('Current (mA)', fontsize=24)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.175), fontsize=32)
plt.show()