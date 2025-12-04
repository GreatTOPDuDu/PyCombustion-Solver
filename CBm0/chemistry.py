from __future__ import annotations
import numpy as np
from .physics import (
    R_u,
    M_CH4, M_H2, M_O2, M_CO, M_CO2, M_H2O, M_N2, M_NO,
    WD_A1_SI, WD_Ea1, WD_a1_CH4, WD_a1_O2,
    WD_A2_SI, WD_Ea2, WD_a2_CO, WD_a2_O2, WD_a2_H2O,
    h_molar_shomate_field,
)


def compute_sources_wd2(
    T,
    rho,
    Cp_mix,
    YF,
    YO,
    YC,
    YC2,
    YW,
    A1_mult: float = 1.0,
    Ea1_mult: float = 1.0,
    A2_mult: float = 1.0,
    Ea2_mult: float = 1.0,
):
    """Two-step Westbrook–Dryer CH4 mechanism.

    The multipliers (A*_mult, Ea*_mult) act as global knobs on the pre-exponential
    factors and activation energies and are intended for sensitivity/teaching
    studies
    """
    Tclip = np.maximum(T, 1.0); rho_safe = np.maximum(rho, 1e-30)
    C_CH4 = np.maximum(0.0, rho * YF / (M_CH4 + 1e-30))
    C_O2  = np.maximum(0.0, rho * YO / (M_O2  + 1e-30))
    C_CO  = np.maximum(0.0, rho * YC / (M_CO  + 1e-30))
    C_H2O = np.maximum(0.0, rho * YW / (M_H2O + 1e-30))
    eps = 1e-30

    A1_eff = WD_A1_SI * A1_mult
    Ea1_eff = WD_Ea1 * Ea1_mult
    A2_eff = WD_A2_SI * A2_mult
    Ea2_eff = WD_Ea2 * Ea2_mult

    r1 = A1_eff * np.exp(-Ea1_eff / (R_u * Tclip)) * (C_CH4 + eps)**WD_a1_CH4 * (C_O2 + eps)**WD_a1_O2
    r2 = A2_eff * np.exp(-Ea2_eff / (R_u * Tclip)) * (C_CO  + eps)**WD_a2_CO  * (C_O2 + eps)**WD_a2_O2 * (C_H2O + eps)**WD_a2_H2O
    omega1_mf = (r1 * M_CH4) / rho_safe
    omega2_mf = (r2 * M_CO)  / rho_safe

    s1_O2_per_CH4 = 1.5 * (M_O2 / M_CH4)
    y_CO_per_CH4  = (M_CO / M_CH4)
    y_H2O_per_CH4 = 2.0 * (M_H2O / M_CH4)
    s2_O2_per_CO  = 0.5 * (M_O2 / M_CO)
    y_CO2_per_CO  = (M_CO2 / M_CO)

    h_CH4 = h_molar_shomate_field('CH4', Tclip)
    h_O2  = h_molar_shomate_field('O2',  Tclip)
    h_CO  = h_molar_shomate_field('CO',  Tclip)
    h_CO2 = h_molar_shomate_field('CO2', Tclip)
    h_H2O = h_molar_shomate_field('H2O', Tclip)
    dH1 = (h_CO + 2.0*h_H2O) - (h_CH4 + 1.5*h_O2)
    dH2 = (h_CO2) - (h_CO + 0.5*h_O2)

    HRR_chem = - (r1 * dH1 + r2 * dH2)
    S_T  = HRR_chem / np.maximum(Cp_mix * rho, 1e-12)
    S_YF = -omega1_mf
    S_YO = - (s1_O2_per_CH4 * omega1_mf) - (s2_O2_per_CO * omega2_mf)
    S_YC = + (y_CO_per_CH4  * omega1_mf) - omega2_mf
    S_YC2= + (y_CO2_per_CO  * omega2_mf)
    S_YW = + (y_H2O_per_CH4 * omega1_mf)

    return S_T, S_YF, S_YO, S_YC, S_YC2, S_YW, HRR_chem


def compute_sources_h2_global(
    T,
    rho,
    Cp_mix,
    YF,
    YO,
    YW,
    A_mult: float = 1.0,
    Ea_mult: float = 1.0,
):
    """Global one-step H2 + 0.5 O2 -> H2O reaction model.

    YF: H2, YO: O2, YW: H2O.
    The multipliers A_mult and Ea_mult act on the global pre-exponential
    factor and activation energy, mainly for sensitivity/educational use.
    """
    Tclip = np.maximum(T, 300.0)
    rho_safe = np.maximum(rho, 1e-30)

    # molar concentrations [mol/m^3]
    C_H2 = np.maximum(0.0, rho_safe * YF / (M_H2 + 1e-30))
    C_O2 = np.maximum(0.0, rho_safe * YO / (M_O2 + 1e-30))

    # Simple global rate: r = A exp(-Ea/T) [H2]^0.87 [O2]^1.1
    A = 1.0e11 * A_mult
    Ea_over_R = 6900.0 * Ea_mult
    r = A * np.exp(-Ea_over_R / np.maximum(Tclip, 1e-6)) * (C_H2**0.87) * (C_O2**1.10)

    # mass-fraction source terms [1/s]
    omega_H2  = - r * M_H2  / rho_safe
    omega_O2  = - 0.5 * r * M_O2 / rho_safe
    omega_H2O =   1.0 * r * M_H2O / rho_safe

    # reaction enthalpy (H2 + 0.5 O2 -> H2O)
    h_H2  = h_molar_shomate_field('H2',  Tclip)
    h_O2  = h_molar_shomate_field('O2',  Tclip)
    h_H2O = h_molar_shomate_field('H2O', Tclip)
    dH = h_H2O - h_H2 - 0.5 * h_O2   # [J/mol]

    HRR_chem = - r * dH              # [W/m^3]
    S_T = HRR_chem / np.maximum(Cp_mix * rho_safe, 1e-12)

    S_YF = omega_H2
    S_YO = omega_O2
    S_YC = np.zeros_like(T)
    S_YC2 = np.zeros_like(T)
    S_YW = omega_H2O

    return S_T, S_YF, S_YO, S_YC, S_YC2, S_YW, HRR_chem


def thermal_NO_source(T, rho, YF, YO, YC, YC2, YW):
    """
    Thermal (extended Zeldovich) NO generation global equation.
    Simply Novosselov type: d[NO]/dt = A [N2][O2]^0.5 T^-0.5 exp(-Ea/T)
    Returns: S_YNO (mass fraction equation source term, [1/s])
    """
    Tclip = np.maximum(T, 1000.0)
    rho_safe = np.maximum(rho, 1e-30)

    # N2 mass fraction (NO is considered very sparse and ignored)
    YN2 = np.clip(1.0 - (YF + YO + YC + YC2 + YW), 0.0, 1.0)

    # molar concentrations [mol/m^3]
    C_O2 = rho_safe * YO  / (M_O2 + 1e-30)
    C_N2 = rho_safe * YN2 / (M_N2 + 1e-30)

    A = 10.0**14.967
    Ea_over_R = 68899.0

    r_NO = A * C_N2 * np.sqrt(C_O2) * (Tclip**-0.5) * np.exp(-Ea_over_R / Tclip)

    S_YNO = r_NO * M_NO / rho_safe  # [1/s]

    return S_YNO

