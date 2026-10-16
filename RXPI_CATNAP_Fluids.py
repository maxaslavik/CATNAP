# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:28:24 2026

@author: Maximilian Slavik
"""



import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import math
from math import pi, tan
from scipy.optimize import brentq


################ NHNE Modeling Function ###############


##### Written by: Eli Macdonald 

##### Reviewed by: Maximilian Slavik, Andrew Ciambella
    
def clamp(x, lo, hi):
    return max(lo, min(hi, x))
  
def mdot_spi_hem_nhne(P1, T1, P2, Cd, Ainj, fluid='N2O', k_override=None,Q_in=0):
    # --- Upstream thermo (h1, s1)
    # h1 = CP.PropsSI("Hmass", "P", P1, "T", T1, fluid) #Modified to clamp liquid state
    h1 = CP.PropsSI("Hmass", "T", T1, "Q", Q_in, fluid)
    s1 = CP.PropsSI("Smass", "T", T1, "Q", Q_in, fluid)

    # --- Upstream liquid density for SPI (Evaluated at P and T for subcooled liquid!)
    rho1 = CP.PropsSI("Dmass", "Q", 0, "T", T1, fluid)

    # --- SPI model
    dP = max(P1 - P2, 0.0)
    mdot_spi = Cd * Ainj * math.sqrt(2.0 * rho1 * dP)

    # --- Check for Vapor Pressure (Flashing condition)
    Pv1 = CP.PropsSI("P", "T", T1, "Q", 0, fluid)  # Vapor pressure at inlet T
    
    # If vapor pressure is lower than downstream pressure, no flashing occurs.
    if Pv1 <= P2:
        return mdot_spi
        

    # --- HEM: isentropic to P2 (s2 = s1)
    # Below the triple point there is no liquid-vapour state to flash to (N2O: 87.8 kPa,
    # above 0.8 atm ambient at startup). The flow is choked long before this, so clamp.
    P2 = max(P2, 1.001*CP.PropsSI("ptriple", fluid))
    h2 =CP.PropsSI("Hmass", "P", P2, "Smass", s1, fluid)

    # Saturation props at P2 (for quality + mixture density)
    hf   = CP.PropsSI("Hmass", "P", P2, "Q", 0, fluid)
    hg   = CP.PropsSI("Hmass", "P", P2, "Q", 1, fluid)
    rhof = CP.PropsSI("Dmass", "P", P2, "Q", 0, fluid)
    rhog = CP.PropsSI("Dmass", "P", P2, "Q", 1, fluid)

    # Exit quality from enthalpy
    tiny = 1e-12
    x2 = (h2 - hf) / max(hg - hf, tiny)
    x2 = clamp(x2, 0.0, 1.0)

    # Homogeneous mixture density at exit
    if x2 > 0:
        rho2 = 1.0 / (x2 / rhog + (1.0 - x2) / rhof)
    else:
        rho2 = rhof

    # HEM mass flow
    dh = max(h1 - h2, 0.0)
    mdot_hem = Cd * Ainj * rho2 * math.sqrt(2.0 * dh)

    # --- Dyre/NHNE k-factor
    if k_override is not None:
        k = float(k_override)
    else:
        # Dyer parameter
        k = math.sqrt(max(P1 - P2, 0.0) / (Pv1 - P2))

    # Blend weights
    w = (1.0 / (1.0 + k))

    mdot_nhne = (1-w) * mdot_spi + w * mdot_hem
    
    return mdot_nhne

###########################

from RXPI_CATNAP_Config import CV_TO_CDA


class Valve_obj:
    '''
    Liquid valve pressure drops from Cv, dP_psi = SG (Q_gpm/Cv)^2. Kept for standalone use;
    the solver models valves as RXPI_CATNAP_Feed.Valve, which is the same relation for a
    liquid and the generalized Dyer model for a flashing propellant.
    '''

    def __init__(self,Cv_Ox,Cv_Fuel,Props_obj):

        self.CvOx = Cv_Ox
        self.CvFuel = Cv_Fuel
        self.ox = Props_obj.ox
        self.fuel = Props_obj.fuel

    @staticmethod
    def _dP(rho, mdot, Cv):
        # same as SG*(Q_gpm/Cv)**2 psi, in Pa
        return (mdot/(Cv*CV_TO_CDA))**2/(2*rho)

    def dPOx(self, P1, T1, mdotsat):
        rhosat = CP.PropsSI('D','T',T1,'Q',0,self.ox)   # saturated liquid; (T, P) is ambiguous at saturation
        return self._dP(rhosat, mdotsat, self.CvOx)

    def dPFuel(self, P1, T1, mdotsat):
        rhospi = CP.PropsSI('D','T',T1,'P',P1,self.fuel)
        return self._dP(rhospi, mdotsat, self.CvFuel)


############ Fast two-phase property path ############
# PropsSI rebuilds the fluid on every call (~0.5 ms for N2O). A cached AbstractState gives
# every saturation property at a pressure from one update (~7 us), and the feed solver
# evaluates the Dyer model tens of times per time step.

class _Fluid:
    def __init__(self, fluid):
        self.fluid = fluid
        self.AS = CP.AbstractState('HEOS', fluid)
        self.p_floor = 1.001*self.AS.trivial_keyed_output(CP.iP_triple)
        self.p_crit = self.AS.trivial_keyed_output(CP.iP_critical)
        self.T_crit = self.AS.trivial_keyed_output(CP.iT_critical)


_STATES = {}


def abstract_state(fluid):
    '''
    Cached HEOS AbstractState for fluid, with its triple and critical points.
    '''

    F = _STATES.get(fluid)
    if F is None:
        F = _STATES[fluid] = _Fluid(fluid)
    return F


def saturation(F, P):
    '''
    (T, hl, hg, sl, sg, vl, vg) of the saturated liquid and vapour at P.
    '''

    AS = F.AS
    AS.update(CP.PQ_INPUTS, P, 0)
    L = AS.saturated_liquid_keyed_output
    V = AS.saturated_vapor_keyed_output
    return (AS.T(), L(CP.iHmass), V(CP.iHmass), L(CP.iSmass), V(CP.iSmass),
            1/L(CP.iDmass), 1/V(CP.iDmass))


def n2o_viscosity(T, phase):
    '''
    Saturated N2O viscosity [Pa s], ESDU 91022 (CoolProp has no N2O transport model).
    Above Tcrit the vapour form is continued, which only feeds a friction factor.
    '''

    Tc = 309.57
    if phase == 'liquid':
        th = (Tc - 5.24)/(min(T, Tc - 0.1) - 5.24) - 1
        return 1e-3*0.0293423*math.exp(1.6089*th**(1/3) + 2.0439*th**(4/3))
    y = np.cbrt(Tc/T - 1)
    return 1e-6*math.exp(3.3281 - 1.18237*y - 0.055155*y**4)


def phase_viscosity(F, T, phase):
    if F.fluid.upper() in ('N2O', 'NITROUSOXIDE'):
        return n2o_viscosity(T, phase)
    return CP.PropsSI('viscosity', 'T', T, 'Q', 0 if phase == 'liquid' else 1, F.fluid)


class FlowState:
    '''
    Propellant state at a feed node, from (P, h). region is 'liquid' (subcooled or
    supercritical), 'mix' (saturated liquid up to saturated vapour, x = quality) or
    'vapor' (superheated). Pv is the vapour pressure at T, which drives Dyer's k.
    '''

    __slots__ = ('P', 'h', 'T', 'rho', 's', 'x', 'Pv', 'mu', 'gamma', 'region')

    def __repr__(self):
        return f'FlowState(P={self.P:.6g}, x={self.x:.4g}, rho={self.rho:.5g}, {self.region})'


def flow_state(F, P, h):
    st = FlowState()
    st.P, st.h, st.gamma = P, h, None

    if P < 0.999*F.p_crit:
        T, hl, hg, sl, sg, vl, vg = saturation(F, P)
        tol = 1e-9*abs(hl) + 1e-6
        if hl - tol <= h <= hg + tol:
            x = clamp((h - hl)/(hg - hl), 0.0, 1.0)
            st.T, st.x, st.Pv, st.region = T, x, P, 'mix'
            st.s = sl + x*(sg - sl)
            st.rho = 1/(vl + x*(vg - vl))
            mul = phase_viscosity(F, T, 'liquid')
            muv = phase_viscosity(F, T, 'vapor')
            st.mu = 1/(x/muv + (1 - x)/mul)          # McAdams homogeneous viscosity
            return st
        st.region = 'liquid' if h < hl else 'vapor'
    else:
        st.region = 'liquid'

    AS = F.AS
    AS.update(CP.HmassP_INPUTS, h, P)
    st.T, st.rho, st.s = AS.T(), AS.rhomass(), AS.smass()
    if st.region == 'vapor':
        st.x = 1.0
        st.gamma = AS.cpmass()/AS.cvmass()
        st.Pv = P
        st.mu = phase_viscosity(F, st.T, 'vapor')
    else:
        st.x = 0.0
        if st.T < F.T_crit:
            AS.update(CP.QT_INPUTS, 0, st.T)
            st.Pv = AS.p()
        else:
            st.Pv = 0.0                               # supercritical: nothing to flash to
        st.mu = phase_viscosity(F, st.T, 'liquid')
    return st


def saturated_state(F, P, Q):
    _, hl, hg, *_ = saturation(F, P)
    return flow_state(F, P, hl + Q*(hg - hl))


def dyer_mdot(F, st, P2, CdA, k_override=None):
    '''
    Generalized Dyer NHNE mass flow [kg/s] through a restriction of effective area CdA,
    from the inlet FlowState st to the back pressure P2.

    mdot = CdA (k G_SPI + G_HEM)/(1 + k),  k = sqrt((P1 - P2)/(Pv1 - P2))
      G_SPI = sqrt(2 rho1 (P1 - P2)), rho1 the homogeneous inlet density
      G_HEM = rho2 sqrt(2 (h1 - h2)), isentropic equilibrium expansion to P2

    A two-phase inlet (quality x1 > 0, e.g. flashed in an upstream valve or line) has
    Pv1 = P1 and so k = 1. A subcooled inlet has Pv1 < P1, k > 1, and recovers the original
    Dyer model; with no flashing (P2 >= Pv1) it is pure SPI. Superheated vapour goes
    through the isentropic compressible-orifice relation (choked below the critical ratio).
    '''

    P1 = st.P
    dP = P1 - P2
    if dP <= 0:
        return 0.0

    if st.region == 'vapor':
        g = st.gamma
        PR = max(P2/P1, (2/(g + 1))**(g/(g - 1)))
        G = math.sqrt(2*st.rho*P1*(g/(g - 1))*(PR**(2/g) - PR**((g + 1)/g)))
        return CdA*G

    G_spi = math.sqrt(2*st.rho*dP)
    if P2 >= st.Pv:
        return CdA*G_spi

    # Below the triple point there is no liquid-vapour state to flash to; the flow is choked
    # long before this, so clamp.
    P2 = max(P2, F.p_floor)
    _, hl, hg, sl, sg, vl, vg = saturation(F, P2)
    x2 = clamp((st.s - sl)/(sg - sl), 0.0, 1.0)
    h2 = hl + x2*(hg - hl)
    G_hem = math.sqrt(2*max(st.h - h2, 0.0))/(vl + x2*(vg - vl))

    k = float(k_override) if k_override is not None else math.sqrt(dP/(st.Pv - P2))

    return CdA*(k*G_spi + G_hem)/(1 + k)


def generalized_dyer(P1,Q1,P2,A_cross,Cd=1,fluid='N2O', k_override=None):
    '''
    Dyer NHNE mass flow [kg/s] from saturated propellant of quality Q1 at P1 to P2.
    Cd applies to both the SPI and HEM terms. See dyer_mdot.
    '''

    F = abstract_state(fluid)
    return dyer_mdot(F, saturated_state(F, P1, Q1), P2, Cd*A_cross, k_override)









class Injector_obj:
    def __init__(self,Cd_fuel,Cd_ox,Cd_film,numox,numfuel,numfilm,Dox,Dfuel,Dfilm,Props_obj):
        self.Cd_fuel = Cd_fuel
        self.Cd_ox = Cd_ox
        self.Cd_film = Cd_film
        self.numox = numox
        self.Aox = 0.25*numox*pi*(Dox**2)
        self.Afuel = 0.25*numfuel*pi*(Dfuel**2)
        self.Afilm = 0.25*numfilm*pi*(Dfilm**2)

        self.CdA_ox = Cd_ox*self.Aox
        self.CdA_fuel = Cd_fuel*self.Afuel
        self.CdA_film = Cd_film*self.Afilm

        self.fuel = Props_obj.fuel
        self.ox = Props_obj.ox




    def mdot_fuel(self,P1,T1,P2):
        
        rhof = CP.PropsSI('D','T',T1,'P',P1,self.fuel)

        mdot = self.CdA_fuel*(np.sqrt(2*rhof*(P1 - P2)))

        return mdot

    def mdot_film(self,P1,T1,P2):
        
        rhof = CP.PropsSI('D','T',T1,'P',P1,self.fuel)

        mdot = self.CdA_film*(np.sqrt(2*rhof*(P1 - P2)))

        return mdot
    
    def mdot_ox_nhne(self,P1,T1,P2):

        mdot = mdot_spi_hem_nhne(P1,T1,P2,self.Cd_ox,self.Aox,self.ox)

        return mdot
    
    def mdot_vapor_orifice(self,P1,T1,P2):

        Cd = self.Cd_ox

        N_holes = self.numox

        mdot = mdot_vapor_orifice(P1,T1,P2,Cd,N_holes,self.Aox)

        return mdot
        


def mdot_vapor_orifice(P1, T1, P2, Cd, N_holes, A_inj, fluid='N2O'):
    """
    Compressible vapor flow through orifice.
    
    Parameters
    ----------
    P1     : float - upstream pressure [Pa]
    T1     : float - upstream temperature [K]
    P2     : float - downstream pressure [Pa]
    Cd     : float - discharge coefficient [-]
    N_holes: int   - number of orifice holes
    d_m    : float - orifice diameter [m]
    fluid  : str   - CoolProp fluid name
    
    Returns
    -------
    mdot : float - mass flow rate [kg/s]
    """
    A = A_inj

    # Get gas properties at upstream state (superheated vapor)
    gamma  = CP.PropsSI('Cpmass', 'T|gas', T1, 'P', P1, fluid) / \
             CP.PropsSI('Cvmass', 'T|gas', T1, 'P', P1, fluid)
    R_spec = CP.PropsSI('gas_constant', fluid) / CP.PropsSI('molar_mass', fluid)
    rho1   = CP.PropsSI('D', 'T|gas', T1, 'P', P1, fluid)

    # Critical pressure ratio
    PR_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

    # Vandenkerckhove / choked flow factor
    Gamma = math.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))

    if P2 / P1 <= PR_crit:
        # Choked flow
        mdot = Cd * A * P1 * Gamma / math.sqrt(R_spec * T1)

    else:
        # Unchoked - isentropic subsonic
        PR = P2 / P1
        mdot = Cd * A * math.sqrt(
            2 * rho1 * P1 * (gamma / (gamma - 1)) *
            (PR**(2/gamma) - PR**((gamma+1)/gamma))
        )

    return mdot

def nozzle(mdot, gamma, R_spec, Tc, A_t, A_e, P_amb):
    """
    Solve for chamber pressure and thrust from mass flow rate.

    Parameters
    ----------
    mdot   : float - total mass flow rate [kg/s]
    gamma  : float - ratio of specific heats [-]
    R_spec : float - specific gas constant [J/kg·K]
    Tc     : float - chamber temperature [K]
    A_t    : float - throat area [m²]
    A_e    : float - exit area [m²]
    P_amb  : float - ambient pressure [Pa]

    Returns
    -------
    Pc : float - chamber pressure [Pa]
    F  : float - thrust [N]
    Pe : float - exit pressure [Pa]
    ve : float - exit velocity [m/s]
    """
    # c* and chamber pressure
    Gamma = np.sqrt(gamma) * (2/(gamma+1)) ** ((gamma+1) / (2*(gamma-1)))
    cstar = np.sqrt(R_spec * Tc) / Gamma
    Pc    = (mdot * cstar) / A_t

    # Exit Mach from area-Mach relation (supersonic root)
    AR = A_e / A_t
    Me = brentq(lambda Me: (1/Me) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * Me**2))
                **((gamma+1)/(2*(gamma-1))) - AR, 1.0, 50.0)

    # Isentropic exit conditions
    Pe = Pc * (1 + (gamma-1)/2 * Me**2) ** (-gamma/(gamma-1))
    Te = Tc * (1 + (gamma-1)/2 * Me**2) ** (-1)
    ve = Me * np.sqrt(gamma * R_spec * Te)

    # Thrust
    F = mdot * ve + (Pe - P_amb) * A_e

    return Pc, F, Pe, ve