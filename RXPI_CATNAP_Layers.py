# -*- coding: utf-8 -*-
"""
Near-wall layer chemistry for film and boundary layer cooling (RPA thermal module).

Ievlev's similarity relation (RPA, "Boundary Layer Cooling") scales the core-flow
convective flux by the properties of the gas layer touching the wall:

    q_layer / q_core = S_layer / S_core

    S = (I0 - I_w) T_e^0.425 mu_1000^0.15 / ( R_1500^0.425 (T_e + T_w)^0.595 (3 T_e + T_w)^0.15 )

  I0      stagnation enthalpy of the layer gas
  I_w     enthalpy at the wall temperature, composition frozen at 1500 K
          (at the stagnation temperature if that is below 1500 K)
  T_e     recovery temperature of the layer gas
  R_1500  gas constant of the products at 1500 K
  mu_1000 viscosity at 1000 K

The layer is a mix of combustion products and vaporized film coolant, so its
properties come from a chemical equilibrium calculation (Cantera, GRI-Mech 3.0 gas
plus graphite) at the layer's element composition and enthalpy. Propellant
formulas and enthalpies are read from the same rocketcea cards CEA uses, so the
core matches CEA (2772 vs 2771 K for N2O/ethanol at O/F 3.2).
"""

import re

import numpy as np
from scipy.optimize import brentq

import cantera as ct
import rocketcea.cea_obj as _cea


_CAL = 4.184
_ATOMIC = {'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007}


def propellant_card(name):
    '''
    Element counts, molar mass (kg/mol) and molar enthalpy (J/mol) of a RocketCEA propellant.
    '''

    for cards in (_cea.oxCards, _cea.fuelCards):
        if name in cards:
            text = ' '.join(cards[name])
            break
    else:
        raise ValueError(f'{name!r} is not a RocketCEA propellant')

    h = re.search(r'h,cal\s*=\s*([-+0-9.eE]+)', text)
    elems = dict((e, float(n)) for e, n in re.findall(r'\b([A-Z][a-z]?)\s+([0-9.]+)\b', text.split('wt%')[0]
                                                         .split('h,cal')[0]))
    elems = {e: n for e, n in elems.items() if e in _ATOMIC}

    if h is None or not elems:
        raise ValueError(f'RocketCEA card for {name!r} has no explicit formula/enthalpy; '
                         'near-wall layer chemistry is unavailable for it')

    mw = sum(_ATOMIC[e]*n for e, n in elems.items())/1000
    return elems, mw, float(h.group(1))*_CAL


class LayerChemistry:
    def __init__(self, oxname, fuelname):
        self.ox = propellant_card(oxname)
        self.fuel = propellant_card(fuelname)

        self.gas = ct.Solution('gri30.yaml')
        self.gas.transport_model = 'mixture-averaged'
        self.graphite = ct.Solution('graphite.yaml')
        self.mix = ct.Mixture([(self.gas, 1.0), (self.graphite, 0.0)])

        self.h_ox = self.ox[2]/self.ox[1]       # J/kg at the card temperature
        self.h_fuel = self.fuel[2]/self.fuel[1]


    def _elements(self, w_ox):
        '''
        Element moles per kg of a mixture with oxidizer mass fraction w_ox.
        '''

        n = {e: 0.0 for e in _ATOMIC}
        for (elems, mw, _), w in ((self.ox, w_ox), (self.fuel, 1 - w_ox)):
            for e, c in elems.items():
                n[e] += w*c/mw
        return n


    def _equilibrate_TP(self, T, P, n):
        g, mix = self.gas, self.mix
        g.TPX = T, P, {k: v for k, v in n.items() if v > 0}
        self.graphite.TP = T, P
        mix.species_moles = np.concatenate([g.X*sum(n.values()), [0.0]])
        mix.T, mix.P = T, P
        try:
            mix.equilibrate('TP', solver='gibbs', max_steps=3000, log_level=0)
        except ct.CanteraError:
            mix.equilibrate('TP', solver='vcs', max_steps=3000, log_level=0)
        ng, nc = mix.phase_moles(0), mix.phase_moles(1)
        return dict(X=g.X.copy(), ng=ng, nc=nc, m=ng*g.mean_molecular_weight + nc*12.011,
                    h=(ng*g.enthalpy_mole + nc*self.graphite.enthalpy_mole)
                      /(ng*g.mean_molecular_weight + nc*12.011))


    def frozen_h(self, st, T, P):
        '''
        Enthalpy (J/kg) of a frozen composition st at temperature T.
        '''

        self.gas.TPX = T, P, st['X']
        self.graphite.TP = T, P
        return (st['ng']*self.gas.enthalpy_mole + st['nc']*self.graphite.enthalpy_mole)/st['m']


    def layer(self, w_ox, H, P):
        '''
        Properties of a layer with oxidizer mass fraction w_ox and total enthalpy H (J/kg).
        '''

        n = self._elements(w_ox)
        T0 = brentq(lambda T: self._equilibrate_TP(T, P, n)['h'] - H, 300.0, 4500.0, xtol=0.05)

        Tf = min(T0, 1500.0)
        st = self._equilibrate_TP(Tf, P, n)

        self.gas.TPX = 1000.0, P, st['X']
        mu1000 = self.gas.viscosity

        return dict(T0=T0, H=H, frozen=st, R1500=ct.gas_constant*st['ng']/st['m'], mu1000=mu1000, P=P)


    def propellant_layer(self, OF, P):
        w_ox = OF/(1 + OF)
        return self.layer(w_ox, w_ox*self.h_ox + (1 - w_ox)*self.h_fuel, P)


    def S(self, L, T_e, T_w):
        '''
        Ievlev's similarity parameter for layer L at recovery temperature T_e and wall temperature T_w.
        '''

        dI = L['H'] - self.frozen_h(L['frozen'], T_w, L['P'])
        return (dI*T_e**0.425*L['mu1000']**0.15
                /(L['R1500']**0.425*(T_e + T_w)**0.595*(3*T_e + T_w)**0.15))


    def mixing_table(self, OF_s, h_coolant, P, n=31):
        '''
        Layers obtained by mixing near-wall gas (products at OF_s) with coolant vapour of
        enthalpy h_coolant, tabulated against the coolant mass fraction k_ff in [0, 1].
        '''

        w_s = OF_s/(1 + OF_s)
        H_s = w_s*self.h_ox + (1 - w_s)*self.h_fuel
        k = np.linspace(0, 1, n)
        return k, [self.layer((1 - kk)*w_s, (1 - kk)*H_s + kk*h_coolant, P) for kk in k]
