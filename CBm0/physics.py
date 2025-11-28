from __future__ import annotations
import math
import numpy as np
from typing import Tuple

# Universal gas constant
R_u = 8.314  # J/mol/K

# Species molecular weights [kg/mol]
M_CH4 = 16e-3
M_H2  = 2e-3
M_O2  = 32e-3
M_N2  = 28e-3
M_CO  = 28e-3
M_CO2 = 44e-3
M_H2O = 18e-3
M_NO  = 30e-3

# Westbrook–Dryer (SI) for CH4
WD_A1_SI = 5.01e11 * 1e-3
WD_Ea1   = 24200.0 * R_u
WD_a1_CH4 = 0.7
WD_a1_O2  = 0.8
WD_A2_SI = 2.24e12 * (10**(-4.5))
WD_Ea2   = 15100.0 * R_u
WD_a2_CO  = 1.0
WD_a2_O2  = 0.25
WD_a2_H2O = 0.5

# Shomate coefficients
SHOMATE = {
    'N2_low':  (28.98641, 1.853978, -9.647459, 16.63537, 0.000117, -8.671914, 226.4168),
    'N2_high': (19.50583, 19.88705, -8.598535, 1.369784, 0.527601, -4.935202, 212.39),
    'O2_low':  (31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471, 246.7945),
    'O2_high': (30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 236.1663),
    'CH4_low': (-0.703029, 108.4773, -42.52157, 5.862788, 0.678565, -76.84376, 158.7163),
    'CH4_high':(85.81217, 11.26467, -2.114146, 0.138190, -26.42221, -153.5327, 224.4143),
    'CO_low':  (25.56759, 6.096130, 4.054656, -2.671301, 0.131021, -118.0089, 227.3665),
    'CO_high': (35.15070,-1.300095,-0.205921, 0.013550,-3.282780, -127.8375, 231.7120),
    'CO2_low': (24.99735,55.18696,-33.69137, 7.948387,-0.136638, -403.6075, 228.2431),
    'CO2_high':(58.16639, 2.720074,-0.492289, 0.038844,-6.447293, -425.9186, 263.6125),
    'H2O_low': (30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967),
    'H2O_high':(41.96426, 8.622053,-1.499780, 0.098119,-11.15764,  -272.1797, 219.7809),

    # H2 (NIST, 298–1000 K / 1000–2500 K)
    'H2_low':  (33.066178, -11.363417, 11.432816, -2.772874,
                -0.158558, -9.980797, 172.707974),
    'H2_high': (18.563083,  12.257357, -2.859786, 0.268238,
                1.977990, -1.147438, 156.288133),

    # NO (NIST, 298–1200 K / 1200–6000 K)
    'NO_low':  (23.83491, 12.58878, -1.139011, -1.497459,
                0.214194, 83.35783, 237.1219),
    'NO_high': (35.99169, 0.957170, -0.148032, 0.009974,
               -3.004088, 73.10787, 246.1619),
}


def _shomate_pair(species: str, T: np.ndarray) -> Tuple[Tuple[float, ...], np.ndarray]:
    if species == 'N2':
        low, high, thr = SHOMATE['N2_low'], SHOMATE['N2_high'], 1000.0
    elif species == 'O2':
        low, high, thr = SHOMATE['O2_low'], SHOMATE['O2_high'], 1000.0
    elif species == 'CH4':
        low, high, thr = SHOMATE['CH4_low'], SHOMATE['CH4_high'], 1300.0
    elif species == 'CO':
        low, high, thr = SHOMATE['CO_low'], SHOMATE['CO_high'], 1300.0
    elif species == 'CO2':
        low, high, thr = SHOMATE['CO2_low'], SHOMATE['CO2_high'], 1200.0
    elif species == 'H2':
        low, high, thr = SHOMATE['H2_low'], SHOMATE['H2_high'], 1000.0
    elif species == 'NO':
        low, high, thr = SHOMATE['NO_low'], SHOMATE['NO_high'], 1200.0
    else:
        low, high, thr = SHOMATE['H2O_low'], SHOMATE['H2O_high'], 1700.0
    mask_low = (T <= thr)
    return (low, high), mask_low


def cp_molar_shomate_field(species: str, T: np.ndarray) -> np.ndarray:
    Tclip = np.clip(T, 200.0, 6000.0)
    t = Tclip / 1000.0
    (low, high), mask_low = _shomate_pair(species, Tclip)
    Al, Bl, Cl, Dl, El, Fl, Gl = low
    Ah, Bh, Ch, Dh, Eh, Fh, Gh = high
    Cp_low = Al + Bl * t + Cl * t**2 + Dl * t**3 + El / (t**2 + 1e-30)
    Cp_high = Ah + Bh * t + Ch * t**2 + Dh * t**3 + Eh / (t**2 + 1e-30)
    return np.where(mask_low, Cp_low, Cp_high)


def h_molar_shomate_field(species: str, T: np.ndarray) -> np.ndarray:
    Tclip = np.clip(T, 200.0, 6000.0)
    t = Tclip / 1000.0
    (low, high), mask_low = _shomate_pair(species, Tclip)
    Al, Bl, Cl, Dl, El, Fl, Gl = low
    Ah, Bh, Ch, Dh, Eh, Fh, Gh = high
    HkJ_low = (Al * t + Bl * t**2 / 2.0 + Cl * t**3 / 3.0 + Dl * t**4 / 4.0 - El / np.maximum(t, 1e-30) + Fl)
    HkJ_high = (Ah * t + Bh * t**2 / 2.0 + Ch * t**3 / 3.0 + Dh * t**4 / 4.0 - Eh / np.maximum(t, 1e-30) + Fh)
    return 1000.0 * np.where(mask_low, HkJ_low, HkJ_high)


def cp_species_mass(T):
    Cp_CH4 = cp_molar_shomate_field('CH4', T) / M_CH4
    Cp_O2  = cp_molar_shomate_field('O2',  T) / M_O2
    Cp_N2  = cp_molar_shomate_field('N2',  T) / M_N2
    Cp_CO  = cp_molar_shomate_field('CO',  T) / M_CO
    Cp_CO2 = cp_molar_shomate_field('CO2', T) / M_CO2
    Cp_H2O = cp_molar_shomate_field('H2O', T) / M_H2O
    return Cp_CH4, Cp_O2, Cp_N2, Cp_CO, Cp_CO2, Cp_H2O


def cp_mixture_mass(T, YF, YO, YC, YC2, YW, fuel_type: str = "CH4"):
    """
    Mixture cp on mass basis.
    YF: generic fuel mass fraction (CH4 or H2, selected by fuel_type).
    NO is ignored in cp and Rmix calculations (treated as a tracer scalar).
    """
    Cp_CH4, Cp_O2, Cp_N2, Cp_CO, Cp_CO2, Cp_H2O = cp_species_mass(T)

    if fuel_type.upper() == "H2":
        Cp_fuel = cp_molar_shomate_field('H2', T) / M_H2
    else:
        Cp_fuel = Cp_CH4

    YN2 = np.clip(1.0 - (YF + YO + YC + YC2 + YW), 0.0, 1.0)

    return (
        YF * Cp_fuel +
        YO * Cp_O2 +
        YC * Cp_CO +
        YC2 * Cp_CO2 +
        YW * Cp_H2O +
        YN2 * Cp_N2
    )


# Viscosity models
Tref = 300.0
SUTH = {
    'CH4': {'mu_ref': 1.10e-5, 'S': 154.0},
    'O2':  {'mu_ref': 2.07e-5, 'S': 127.0},
    'N2':  {'mu_ref': 1.663e-5, 'S': 111.0},
    'CO':  {'mu_ref': 1.78e-5, 'S': 118.0},
    'CO2': {'mu_ref': 1.48e-5, 'S': 240.0},
    'H2O': {'mu_ref': 9.0e-6,  'S': 120.0},
}


def mu_species_all(T, fuel_type: str = "CH4"):
    """
    If fuel_type == 'CH4', use existing CH4 viscosity, if 'H2', use H2 viscosity,
    return key as 'CH4' to maintain existing code compatibility.
    """
    out = {}

    if fuel_type.upper() == "H2":
        # Approximate H2 Sutherland parameters
        mu_r = 8.76e-6
        S = 72.0
    else:
        mu_r = SUTH['CH4']['mu_ref']
        S = SUTH['CH4']['S']
    out['CH4'] = mu_r * (T / Tref)**1.5 * (Tref + S) / (T + S)

    for sp in ('O2', 'N2', 'CO', 'CO2', 'H2O'):
        mu_r = SUTH[sp]['mu_ref']; S = SUTH[sp]['S']
        out[sp] = mu_r * (T / Tref)**1.5 * (Tref + S) / (T + S)
    return out


def mole_fractions_all(YF, YO, YC, YC2, YW, fuel_type: str = "CH4"):
    """
    Fuel is assumed to be either CH4 or H2. NO is ignored in mixture properties (assumed very sparse).
    Returns: x_fuel, x_O2, x_CO, x_CO2, x_H2O, x_N2
    """
    if fuel_type.upper() == "H2":
        M_fuel = M_H2
    else:
        M_fuel = M_CH4

    YN2 = np.clip(1.0 - (YF + YO + YC + YC2 + YW), 0.0, 1.0)

    denom = (
        YF / M_fuel +
        YO / M_O2 +
        YC / M_CO +
        YC2 / M_CO2 +
        YW / M_H2O +
        YN2 / M_N2 +
        1e-30
    )

    return (
        (YF / M_fuel) / denom,
        (YO / M_O2) / denom,
        (YC / M_CO) / denom,
        (YC2 / M_CO2) / denom,
        (YW / M_H2O) / denom,
        (YN2 / M_N2) / denom,
    )


def wilke_mixture_viscosity_general(mu_list, x_list, M_list):
    N = len(mu_list)
    mu = np.stack(mu_list)
    x = np.stack(x_list)
    M = np.array(M_list).reshape((N,) + (1,) * (mu.ndim - 1))
    phi = np.zeros((N, N) + mu_list[0].shape)
    for i in range(N):
        for j in range(N):
            if i == j:
                phi[i, j] = 1.0
            else:
                term = (mu[i] / (mu[j] + 1e-30))**0.5 * (M[j] / (M[i] + 1e-30))**0.25
                phi[i, j] = (1.0 + term)**2 / (math.sqrt(8.0) * (1.0 + (M[i] / (M[j] + 1e-30))))
    denom = np.sum(x[None, ...] * phi, axis=0)
    denom = np.where(denom <= 1e-30, 1e-30, denom)
    mu_mix = np.sum(x * mu / denom, axis=0)
    return mu_mix


def Rmix_and_rho(T, YF, YO, YC, YC2, YW, P0: float, fuel_type: str = "CH4"):
    """
    Ideal-gas mixture R, density.
    fuel_type: 'CH4' or 'H2'; NO is ignored (tracer scalar).
    """
    if fuel_type.upper() == "H2":
        M_fuel = M_H2
    else:
        M_fuel = M_CH4

    YN2 = np.clip(1.0 - (YF + YO + YC + YC2 + YW), 0.0, 1.0)
    denom = (
        YF / M_fuel +
        YO / M_O2 +
        YC / M_CO +
        YC2 / M_CO2 +
        YW / M_H2O +
        YN2 / M_N2 +
        1e-30
    )
    Mmix = 1.0 / denom
    Rmix = R_u / Mmix
    rho = P0 / (Rmix * np.maximum(T, 1.0))
    return Rmix, rho


def k_mixture(mu_mix, Cp_mix, Pr=0.7):
    return mu_mix * Cp_mix / max(Pr, 1e-12)


def D_species_T_all(T, fuel_type: str = "CH4"):
    """
    Fickian diffusion coefficients for species, temperature scaling ~ T^1.75.
    First component is fuel (CH4 or H2).
    """
    D_ref = {
        'CH4': 2.0e-5,
        'H2':  6.0e-5,   # H2 has roughly larger diffusion coefficient
        'O2':  1.8e-5,
        'N2':  1.9e-5,
        'CO':  1.9e-5,
        'CO2': 1.4e-5,
        'H2O': 2.5e-5,
    }
    scale = (T / 300.0)**1.75

    if fuel_type.upper() == "H2":
        D_fuel_ref = D_ref['H2']
    else:
        D_fuel_ref = D_ref['CH4']

    D_ch4_or_h2 = D_fuel_ref * scale
    D_o2 = D_ref['O2'] * scale
    D_co = D_ref['CO'] * scale
    D_co2 = D_ref['CO2'] * scale
    D_h2o = D_ref['H2O'] * scale
    D_n2 = D_ref['N2'] * scale

    return D_ch4_or_h2, D_o2, D_co, D_co2, D_h2o, D_n2

