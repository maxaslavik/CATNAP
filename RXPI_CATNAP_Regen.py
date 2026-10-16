# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:53:30 2026

@author: Maximilian Slavik
"""

import numpy as np
import scipy.interpolate
import CoolProp.CoolProp as CP
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import math
from math import pi, tan
from scipy.optimize import brentq
from scipy.optimize import root

from RXPI_CATNAP_Combustion import SolvePC, CombustionPerformance, ChamberTransport, TPRhoStag, Props_obj, Transport_obj






def RegenGeom(lfmin,wcmin,Rthroat,numchannels,twall,R):
    
    #NOTE: Constant rib height/thickness channels, throat is minimum width - size carefully for DFAM!
    
    # Will add support for variable height channels at a future time
    
    trib = (2*pi*(Rthroat + twall) - wcmin*numchannels)/numchannels
    
    def channelheight(z):
        return lfmin
    
    def channelwidth(z):
        widc = (2*pi*(R(z) + twall) - numchannels*trib)/numchannels
        
        return widc
    
    return channelheight,channelwidth, trib



    




RE_LAMINAR = 2300.0
RE_TURBULENT = 1.0e4


def shah_london_H1(aspect):
    '''
    Fully developed laminar Nu in a rectangular duct, H1 boundary condition (axially uniform
    heat flux, peripherally uniform wall temperature, i.e. a high-conductivity wall), Shah &
    London (1978). aspect = short/long side: 8.235 for parallel plates, 3.61 for a square.
    '''

    a = aspect
    return 8.235*(1 - 2.0421*a + 3.0853*a**2 - 2.4765*a**3 + 1.0578*a**4 - 0.1861*a**5)


def nusselt_laminar(Re, Pr, Dh, x, aspect):
    '''
    Local laminar Nu at distance x from the channel inlet, constant heat flux, thermally and
    hydrodynamically developing (VDI Heat Atlas G1, Gnielinski):

        Nu_x = [Nu_1^3 + 1 + (Nu_2 - 1)^3 + Nu_3^3]^(1/3)
        Nu_2 = 1.302 (Re Pr Dh/x)^(1/3)        thermal entry
        Nu_3 = 0.462 Pr^(1/3) (Re Dh/x)^(1/2)  simultaneous hydrodynamic entry

    The fully developed limit Nu_1 is Shah & London's rectangular-duct H1 value in place of
    the circular-tube 4.364.
    '''

    gz = Dh/max(x, 1e-3*Dh)
    Nu2 = 1.302*(Re*Pr*gz)**(1/3)
    Nu3 = 0.462*Pr**(1/3)*(Re*gz)**0.5
    return (shah_london_H1(aspect)**3 + 1 + (Nu2 - 1)**3 + Nu3**3)**(1/3)


def channel_nusselt(Re, Pr, Dh, x, aspect, nu_turbulent):
    '''
    Coolant Nu across laminar, transitional and turbulent flow (VDI Heat Atlas G1):

        Re <= 2300          laminar, nusselt_laminar(Re)
        2300 < Re < 1e4     (1 - g) Nu_lam(2300) + g Nu_turb(1e4),  g = (Re - 2300)/(1e4 - 2300)
        Re >= 1e4           nu_turbulent(Re)

    nu_turbulent is the channel's turbulent correlation (Dittus-Boelter or Gnielinski). VDI
    anchors the interpolation on Gnielinski at Re = 1e4; anchoring on the correlation used
    above 1e4 keeps h continuous there.
    '''

    if Re >= RE_TURBULENT:
        return nu_turbulent(Re)
    if Re <= RE_LAMINAR:
        return nusselt_laminar(Re, Pr, Dh, x, aspect)
    g = (Re - RE_LAMINAR)/(RE_TURBULENT - RE_LAMINAR)
    return (1 - g)*nusselt_laminar(RE_LAMINAR, Pr, Dh, x, aspect) + g*nu_turbulent(RE_TURBULENT)


def _coolant_state(z,mdotchannel,T1,P1,Regen_obj):
    '''
    Channel Dh, Re, Pr, k, aspect ratio and the distance from the coolant inlet (the nozzle
    exit, z = 0) along the channel to the middle of the cell at z.
    '''

    fluid = Regen_obj.coolant

    channelheight1 = Regen_obj.channelheight(z)

    channelwidth1 = Regen_obj.channelwidth(z)

    mu = CP.PropsSI('viscosity','T',T1,'P',P1,fluid)

    Dh = 4*(channelwidth1*channelheight1)/(2*channelwidth1 + 2*channelheight1)

    Reynolds = Dh*mdotchannel/(channelheight1*channelwidth1*mu)

    Prantl = CP.PropsSI('Prandtl','T',T1,'P',P1,fluid)

    k_fluid = CP.PropsSI('conductivity','T',T1,'P',P1,fluid)

    aspect = min(channelwidth1, channelheight1)/max(channelwidth1, channelheight1)

    x = z/np.cos(Regen_obj.genrad) + 0.5*Regen_obj.dz

    return Dh, Reynolds, Prantl, k_fluid, aspect, x


def DittusB(z,mdotchannel,T1,P1,Regen_obj):

    Dh, Reynolds, Prantl, k_fluid, aspect, x = _coolant_state(z, mdotchannel, T1, P1, Regen_obj)

    def turbulent(Re):
        return 0.023*(Re**0.8)*(Prantl**0.4)

    Nusselt = channel_nusselt(Reynolds, Prantl, Dh, x, aspect, turbulent)

    hcoolant = Nusselt*k_fluid/Dh

    return hcoolant


def Gneilinski(z,mdotchannel,Regen_obj,T1,P1):

    Dh, Reynolds, Prantl, k_fluid, aspect, x = _coolant_state(z, mdotchannel, T1, P1, Regen_obj)

    epsilon_rough = Regen_obj.epsilon_rough

    def turbulent(Re):
        f = 0.25*((np.log10(epsilon_rough/(3.7*Dh) + 5.74/(Re**0.9)))**(-2))
        return ((f/8)*(Re - 1000)*Prantl)/(1 + 12.7*((f/8)**0.5)*(Prantl**(2/3) - 1))

    Nusselt = channel_nusselt(Reynolds, Prantl, Dh, x, aspect, turbulent)

    hcoolant = Nusselt*k_fluid/Dh

    return hcoolant
    


def Etafin(z,hcoolant,Regen_obj):
    
    
    finthickness = Regen_obj.trib
    
    kcond_fin = Regen_obj.k_chamber
    
    channelheight = Regen_obj.channelheight
    channelheight1 = channelheight(z)
    
    msqr = 2*hcoolant/(kcond_fin*finthickness)
    
    m = np.sqrt(msqr)
    
    etafin = (np.tanh(m*channelheight1))/(m*channelheight1)

    
    return etafin


 
def bartz_hg(z, Mach, R, Dt, rc, Pc, cstar, mu, cp, Pr, gamma,Tcomb,TempsC):
    """
    Computes the hot-gas side heat transfer coefficient using the Bartz correlation.
    
    Parameters
    ----------
    z : float
        Axial position along the engine [m]
    Mach : callable
        Mach number as a function of z, Mach(z) [-]
    R : callable
        Local inner wall radius as a function of z, R(z) [m]
    Dt : float
        Throat diameter [m]
    rc : float
        Radius of curvature at the throat [m]
    Pc : float
        Chamber pressure [Pa]
    cstar : float
        Characteristic exhaust velocity [m/s]
    mu : callable
        Dynamic viscosity of combustion gas as a function of z, mu(z) [Pa·s]
    cp : callable
        Specific heat at constant pressure as a function of z, cp(z) [J/kg·K]
    Pr : callable
        Prandtl number of combustion gas as a function of z, Pr(z) [-]
    gamma : callable
        Ratio of specific heats as a function of z, gamma(z) [-]
    T0 : float
        Stagnation (chamber) temperature [K]
    Tw : float
        Wall temperature on the hot-gas side [K]
    
    Returns
    -------
    h_g : float
        Hot-gas side heat transfer coefficient [W/m²·K]
    
    """
    
    M   = Mach(z)
    g   = gamma(z)
    mu_ = mu(z)      
    cp_ = cp(z)  # J/kg-K
    Pr_ = Pr(z)

    T0 = Tcomb

    Tw, _ = TempsC(z,Mach)

    Rad = R(z)
    
    A_ = pi*(Rad**2)
    
    At = 0.25*pi*(Dt**2)
 
    mach_term = 1.0 + (g - 1.0) / 2.0 * (M**2)
 
    sigma = (0.5 * (Tw / T0) * mach_term + 0.5)**(-0.68) \
            * mach_term**(-0.12)
 
    h_g = (
        (0.026 / Dt**0.2)
        * (mu_**0.2 * cp_ / Pr_**0.6)
        * (Pc / cstar)**0.8
        * (Dt / rc)**0.1
        * (At / A_)**0.9
        * sigma
    )

    return h_g



def Resistances(z,hc,hg,R,Regen_obj):
    
    channelheight = Regen_obj.channelheight

    rad_inner = R(z)

    k_chamber = Regen_obj.k_chamber

    dz = Regen_obj.dz

    twall = Regen_obj.twall

    trib = Regen_obj.trib

    numchannels = Regen_obj.numchannels

    channelheight1 = channelheight(z)
    
    Rg = 1/(hg*2*pi*rad_inner*dz)
    
    Rwall = np.log((rad_inner + twall)/rad_inner)/(2*pi*k_chamber*dz)
    
    etafin = Etafin(z, hc, Regen_obj)
    
    Abase = (2*pi*(rad_inner + twall) - numchannels*trib)*dz
    
    Atotal = Abase + 2*numchannels*channelheight1*dz
    
    etanaught = (1 - ((2*numchannels*channelheight1*dz)/(Atotal))*(1 - etafin))
    
    Rc = 1/(etanaught*hc*Atotal)
    
    Rtotal = Rg + Rwall + Rc
    
    return Rtotal,Rg,Rwall,Rc



def DeltaP(z,mdotchannel,T1,P1,Regen_obj):

    channelheight = Regen_obj.channelheight

    channelwidth = Regen_obj.channelwidth

    fluid = Regen_obj.coolant

    epsilon_rough = Regen_obj.epsilon_rough

    dz = Regen_obj.dz
    
    channelheight1 = channelheight(z)
    channelheight2 = channelheight(z + dz)
    
    channelwidth1 = channelwidth(z)
    channelwidth2 = channelwidth(z + dz)
    

    mu = CP.PropsSI('viscosity','T',T1,'P',P1,fluid)
    
    Dh = 4*(channelwidth1*channelheight1)/(2*channelwidth1 + 2*channelheight1)
    
    Reynolds = Dh*mdotchannel/(channelheight1*channelwidth1*mu)
    
    if Reynolds < 2300:
        # Fully developed laminar flow in a rectangular duct, Shah & London (Darcy f*Re;
        # 64 for a circle, 56.9 for a square). Only the heat transfer correlations need
        # turbulent flow, so the pressure drop no longer stops the run here.
        a = min(channelwidth1, channelheight1)/max(channelwidth1, channelheight1)
        fRe = 96*(1 - 1.3553*a + 1.9467*a**2 - 1.7012*a**3 + 0.9564*a**4 - 0.2537*a**5)
        f = fRe/Reynolds
    else:
        f = 0.25*((np.log10(epsilon_rough/(3.7*Dh) + 5.74/(Reynolds**0.9)))**(-2))

    rho = CP.PropsSI('D','T',T1,'P',P1,fluid)
    
    u = mdotchannel/(rho*channelwidth1*channelheight1)
    
    u2 = mdotchannel/(rho*channelwidth2*channelheight2)
    
    darcydP = -(f/Dh)*(0.5*rho*(u**2))*dz
    
    areadP = 0.5*(rho)*(u**2 - u2**2)        
    
    DeltaP = darcydP + areadP
    
    return DeltaP





TBC_FLUX_REDUCTION_PER_PCT = 8.0   # % wall heat flux removed per 1 % PDMS fuel mass fraction (conservative)


def tbc_flux_scale(pdms_mass_pct):
    '''
    Wall heat flux multiplier for a PDMS-doped fuel (thermal barrier coating).
    Linear: each 1 % of fuel mass fraction removes TBC_FLUX_REDUCTION_PER_PCT % of the
    flux computed without the TBC. Floored at zero so it can never reverse the flux.
    Applied to the reported flux only, see SOLVE_REGEN.
    '''

    return max(0.0, 1.0 - TBC_FLUX_REDUCTION_PER_PCT*pdms_mass_pct/100.0)


def RectStress(z,twall,Q,Rwall):
    #TODO: implement later, use thermal stress eq from heister or something
    pass




class Regen_obj:
    def __init__(self, twall, lfmin, wcmin,R, Rthroat, throat_radcurv, numchannels, coolant, k_chamber, epsilon_rough,numpts_z,enginelength,genangle,tbc_scale=1.0):

        self.genrad = (genangle)*pi/180
        self.tbc_scale = tbc_scale   # wall heat flux multiplier, see tbc_flux_scale()
        
        self.twall = twall
        self.R = R #function R(z)
        self.throat_radcurv = throat_radcurv
        self.numchannels = numchannels
        self.Rthroat = Rthroat
        self.coolant = coolant
        self.k_chamber = k_chamber
        self.epsilon_rough = epsilon_rough
        self.numpts_z = numpts_z
        self.channelheight, self.channelwidth, self.trib = RegenGeom(lfmin,wcmin,Rthroat,numchannels,twall,R)
        self.z_array = np.linspace(0,enginelength,numpts_z)
        self.dz = (enginelength/numpts_z)/np.cos(self.genrad)

    


    def BalanceEnth(self,z,mdot_coolant,T1,P1,Transport_obj,Correlation_flag='Gneil'):

        MachArea = lambda z: Transport_obj.Mach(z, self.R)

        Cptransport, viscositytransport, thermalcondtransport, prantltransport, gammatransport = Transport_obj.Chambertransport()

        cstar = Transport_obj.getCstar()

        Pc = Transport_obj.Pc

        Tcomb = Transport_obj.Tcomb()

        TempsC,_,_,_ = Transport_obj.TPRhostag()

        mdotchannel = mdot_coolant/self.numchannels

        hg = bartz_hg(z,MachArea,self.R,2*self.Rthroat,self.throat_radcurv,Pc,cstar,viscositytransport,
                      Cptransport,prantltransport,gammatransport,Tcomb,TempsC)


        hcGneil = Gneilinski(z, mdotchannel, self, T1, P1)

        hcDB = DittusB(z, mdotchannel, T1, P1, self)
        
        Taw,_ = TempsC(z,MachArea)

        if Correlation_flag == 'DB':
        
            Rtotal,_,_,_ = Resistances(z,hcDB,hg,self.R,self)

        if Correlation_flag == 'Gneil':
        
            Rtotal,_,_,_ = Resistances(z,hcGneil,hg,self.R,self)

        Cpcool = CP.PropsSI('CPMASS','T',T1,'P',P1,Transport_obj.Props_obj.fuel)
        
        deltaTc = (Taw - T1)/(mdot_coolant*(Cpcool)*Rtotal)

        T2 = T1 + deltaTc

        return T2
    
    def DeltaPstep(self,z,T1,P1,mdot_coolant):

        mdot_channel = mdot_coolant/self.numchannels

        Deltap = DeltaP(z,mdot_channel,T1,P1,self)

        return Deltap

    '''
    def SOLVE_REGEN(self,mdot_coolant,Tcool_init,Pcool_init,Transport_obj):

        T1 = Tcool_init

        P1 = Pcool_init

        z1 = 0.0

        Tcool_array = np.zeros(self.numpts_z)
        Pcool_array = np.zeros(self.numpts_z)

        for i in range(self.numpts_z):
            T2 = self.BalanceEnth(z1,mdot_coolant,T1,P1,Transport_obj)

            P2 = P1 + self.DeltaPstep(z1,T1,P1,mdot_coolant)

            Tcool_array[i] = T2
            Pcool_array[i] = P2

            T1 = T2
            P1 = P2
            z1 = z1 + self.dz


        return Tcool_array, Pcool_array

        '''
    
    def dP_channel_Approx(self,Tc_init,Pcool_init,mdot_coolant):

        #NOTE: Do not use for normal regen circuit calculation, this loop approximates regen channel loss
        # without accounting for small changes in coolant properties due to temperature. Meant to be used
        # outside of snapshot loop. Properties at the inlet pressure (within 1e-4 of re-evaluating
        # them at the local pressure at every node).

        rho = CP.PropsSI('D','T',Tc_init,'P',Pcool_init,self.coolant)
        mu = CP.PropsSI('viscosity','T',Tc_init,'P',Pcool_init,self.coolant)

        return self.dP_channel_isothermal(mdot_coolant, rho, mu)

    def dP_channel_isothermal(self, mdot_coolant, rho, mu):
        '''
        dP_channel_Approx with the coolant density and viscosity held fixed, vectorized
        over the channel. The feed solver calls this every residual evaluation; a liquid's
        properties barely move over the channel pressure drop. Returns P_out - P_in (< 0).
        '''

        g = getattr(self, '_dp_geom', None)
        if g is None:
            z = self.z_array
            h1 = np.array([self.channelheight(zi) for zi in z], dtype=float)
            w1 = np.array([self.channelwidth(zi) for zi in z], dtype=float)
            h2 = np.array([self.channelheight(zi + self.dz) for zi in z], dtype=float)
            w2 = np.array([self.channelwidth(zi + self.dz) for zi in z], dtype=float)
            A1, A2 = w1*h1, w2*h2
            Dh = 4*A1/(2*w1 + 2*h1)
            a = np.minimum(w1, h1)/np.maximum(w1, h1)
            fRe = 96*(1 - 1.3553*a + 1.9467*a**2 - 1.7012*a**3 + 0.9564*a**4 - 0.2537*a**5)
            g = self._dp_geom = (A1, A2, Dh, fRe)

        A1, A2, Dh, fRe = g
        mdotchannel = mdot_coolant/self.numchannels
        Re = Dh*mdotchannel/(A1*mu)
        with np.errstate(divide='ignore'):
            f = np.where(Re < 2300, fRe/np.maximum(Re, 1e-12),
                         0.25*np.log10(self.epsilon_rough/(3.7*Dh) + 5.74/np.maximum(Re, 1e-12)**0.9)**-2)
        u1 = mdotchannel/(rho*A1)
        u2 = mdotchannel/(rho*A2)
        darcy = -(f/Dh)*(0.5*rho*u1**2)*self.dz
        area = 0.5*rho*(u1**2 - u2**2)

        return float(np.sum(darcy + area))


    def gas_side(self, Transport_obj):
        '''
        Hot-gas quantities along z_array. None of these depend on the coolant state,
        so they are evaluated once per snapshot and shared by the regen march and the
        film model.
        '''

        TempsC, Tstag, PressuresC, RhosC = Transport_obj.TPRhostag()
        cstar  = Transport_obj.getCstar()
        Pc     = Transport_obj.Pc
        Tcomb  = Transport_obj.Tcomb()
        Cptransport, viscositytransport, condtransport, prantltransport, gammatransport = Transport_obj.Chambertransport()
        MachFunc = Transport_obj.MachFunction(self.R)

        z = self.z_array

        gas = {'z': z, 'Tcomb': Tcomb}
        gas['M']     = np.array([MachFunc(z1) for z1 in z])
        gas['hg']    = np.array([bartz_hg(z1, MachFunc, self.R, 2*self.Rthroat, self.throat_radcurv,
                                          Pc, cstar, viscositytransport, Cptransport,
                                          prantltransport, gammatransport, Tcomb, TempsC) for z1 in z])
        temps        = np.array([TempsC(z1, MachFunc) for z1 in z])
        gas['Taw']   = temps[:, 0]
        gas['T0']    = temps[:, 1]
        gas['cp']    = np.array([float(Cptransport(z1)) for z1 in z])
        gas['mu']    = np.array([float(viscositytransport(z1)) for z1 in z])
        gas['k']     = np.array([float(condtransport(z1)) for z1 in z])
        gas['Pr']    = np.array([float(prantltransport(z1)) for z1 in z])
        gas['gamma'] = np.array([float(gammatransport(z1)) for z1 in z])
        gas['P']     = np.array([PressuresC(z1, MachFunc) for z1 in z])
        gas['rho']   = np.array([RhosC(z1, MachFunc) for z1 in z])
        gas['T_static'] = gas['T0']/(1 + 0.5*(gas['gamma'] - 1)*gas['M']**2)

        return gas


    def SOLVE_REGEN(self, mdot_coolant, Tcool_init, Pcool_init, Transport_obj,
                    gas=None, film=None, Film_obj=None, T_w_prev=None):
        '''
        Marches the coolant from the nozzle exit (z = 0) to the injector.

        gas : Optional precomputed gas_side() arrays.
        film : Optional wall-side solution from Film_obj.solve():
            - psi scales the convective coefficient for the gas layer touching the wall
              (film vapour / near-wall layer, Ievlev's S ratio),
            - q_rad is the gas radiation absorbed by the wall,
            - in cells wetted by the liquid film the wall only receives q_rad, weighted
              by the wetted fraction of the cell.
            The returned wall heat flux (Qflux_array) is multiplied by self.tbc_scale; the coolant
            march and returned temperatures are the no-TBC (conservative) state.
        T_w_prev : Hot-wall temperatures from the previous iteration. When given, the Bartz
            sigma factor is evaluated at the wall temperature instead of the recovery temperature.
        '''

        T1 = Tcool_init
        P1 = Pcool_init

        numz = self.numpts_z
        Tcool_array = np.zeros(numz)
        Pcool_array = np.zeros(numz)
        hg_array    = np.zeros(numz)
        Twall_array = np.zeros(numz)
        Qflux_array = np.zeros(numz)

        if gas is None:
            gas = self.gas_side(Transport_obj)

        from RXPI_CATNAP_Film import bartz_sigma

        wetted = film is not None and film['active']

        for i, z1 in enumerate(self.z_array):
            mdotchannel = mdot_coolant / self.numchannels

            hg = gas['hg'][i]

            Taw = gas['Taw'][i]

            if T_w_prev is not None:
                g = (gas['T0'][i], gas['M'][i], gas['gamma'][i])
                hg = hg*bartz_sigma(T_w_prev[i], *g)/bartz_sigma(Taw, *g)

            if film is not None:
                hg = hg*film['psi'][i]

            hcDB = DittusB(z1, mdotchannel, T1, P1, self)

            Rtotal, Rg, Rwall, Rc = Resistances(z1, hcDB, hg, self.R, self)

            Qr = film['q_rad'][i]*2*pi*self.R(z1)*self.dz if film is not None else 0.0

            # hot-wall balance: (Taw - Twg)/Rg + Qr = (Twg - T1)/(Rwall + Rc)
            Rw = Rwall + Rc
            Twg = (Taw/Rg + T1/Rw + Qr)/(1/Rg + 1/Rw)
            Q = (Twg - T1)/Rw

            if wetted and film['coverage'][i] > 0:
                phi = film['coverage'][i]
                Q = phi*film['q_wall'][i]*2*pi*self.R(z1)*self.dz + (1 - phi)*Q

            Twall = T1 + Q * (Rwall + Rc)

            Cpcool = CP.PropsSI('CPMASS', 'T', T1, 'P', P1, self.coolant)
            T2 = T1 + Q / (mdot_coolant * Cpcool)
            P2 = P1 + DeltaP(z1, mdotchannel, T1, P1, self)

            Tcool_array[i] = T2
            Pcool_array[i] = P2
            hg_array[i]    = hg
            Twall_array[i] = Twall
            # TBC: scales only the reported flux of the node (convection + radiation, film-blended).
            # Uncoupled and conservative: Twall, Tcool and the wall/film fixed point in
            # SOLVE_SNAPSHOT keep using the no-TBC Q, so the TBC gives no second-order credit.
            Qflux_array[i] = self.tbc_scale * Q / (2 * pi * self.R(z1) * self.dz)  # W/m² (this is approximate based on hotwall area)

            T1 = T2
            P1 = P2

        return Tcool_array, Pcool_array, hg_array, Twall_array, Qflux_array
    
    
        



        





# def transformdz(z,dz,gen_angles):
#     ### Only constant generatrix/helix angle channels supported for now, will add variability later
    
#     genanglelocal = gen_angles(z) #radians

#     dz = dz*(1/(np.cos(genanglelocal)))
    
    
    
    




        

    






















