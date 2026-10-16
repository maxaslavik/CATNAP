# -*- coding: utf-8 -*-
"""

A. Ponomarenko, "RPA: Tool for Rocket Propulsion Analysis - Thermal Analysis of
Thrust Chambers", v.3 (2012), section "Film Cooling - Liquid Film" (after Ievlev).

"""

import numpy as np
import CoolProp.CoolProp as CP
from math import pi
from scipy.interpolate import PchipInterpolator


SIGMA_SB = 5.670374419e-8  # W/m^2-K^4



# Correlation constants c_ji from Leckner (1972) as tabulated in Modest, Radiative
# Heat Transfer. Row i multiplies log10(p_a L / 1 bar-cm)^i, column j multiplies (T/1000 K)^j.

_LECKNER_H2O = np.array([[-2.2118,  -1.1987,   0.035596],
                         [ 0.85667,  0.93048, -0.14391 ],
                         [-0.10838, -0.17156,  0.045915]])

_LECKNER_CO2 = np.array([[-3.9893,   2.7669,  -2.1081,   0.39163 ],
                         [ 1.2710,  -1.1090,   1.0195,  -0.21897 ],
                         [-0.23678,  0.19731, -0.19544,  0.044644]])


def _leckner_eps0(coef, t, paL):
    i = np.arange(coef.shape[0])[:, None]
    j = np.arange(coef.shape[1])[None, :]
    return np.exp(np.sum(coef * t**j * np.log10(paL)**i))


def _leckner_pressure_ratio(a, b, c, PE, paLm, paL):
    return 1 - (a - 1)*(1 - PE)/(a + b - 1 + PE)*np.exp(-c*np.log10(paLm/paL)**2)


def leckner_emissivity(T, P, x_h2o, x_co2, L):
    '''
    Total emissivity of an H2O/CO2 bearing gas.

    Parameters
    ----------
    T : Gas temperature (K).
    P : Total pressure (Pa).
    x_h2o, x_co2 : Mole fractions of water vapour and carbon dioxide.
    L : Mean beam length (m).

    Returns
    -------
    eps : Gas total emissivity (-).
    '''

    t = T/1000
    P_bar = P/1e5
    L_cm = L*100

    pw = x_h2o*P_bar
    pc = x_co2*P_bar

    eps_w = 0.0
    eps_c = 0.0

    if pw*L_cm > 1e-6:
        paL = pw*L_cm
        a = 2.144 if t < 0.75 else 1.88 - 2.053*np.log10(t)
        PE = P_bar + 2.56*pw/np.sqrt(t)
        eps_w = _leckner_eps0(_LECKNER_H2O, t, paL)*_leckner_pressure_ratio(
            a, 1.10/t**1.4, 0.5, PE, 13.2*t**2, paL)

    if pc*L_cm > 1e-6:
        paL = pc*L_cm
        paLm = 0.054/t**2 if t < 0.7 else 0.225*t**2
        PE = P_bar + 0.28*pc
        eps_c = _leckner_eps0(_LECKNER_CO2, t, paL)*_leckner_pressure_ratio(
            a=1 + 0.1/t**1.45, b=0.23, c=1.47, PE=PE, paLm=paLm, paL=paL)

    # H2O-CO2 band overlap correction
    deps = 0.0
    if pw > 0 and pc > 0:
        zeta = pw/(pw + pc)
        logpL = max(np.log10((pw + pc)*L_cm), 0.0)
        deps = (zeta/(10.7 + 101*zeta) - zeta**10.4/111.7)*logpL**2.76

    return max(eps_w + eps_c - deps, 0.0)




############ Film stability and gas-side flux ############

# RPA figure 2 ("Film stability coefficient", from Ievlev), digitised. Flat beyond the plotted range.
_ETA_RE  = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4800]
_ETA_VAL = [1.0, 0.90, 0.78, 0.69, 0.63, 0.585, 0.55, 0.52, 0.50, 0.47]
_eta_curve = PchipInterpolator(_ETA_RE, _ETA_VAL)


def film_stability(Re_f):
    '''
    RPA's liquid film stability coefficient eta(Re_f), Re_f = m_f / (2 pi R mu_f).
    '''

    return float(_eta_curve(min(max(Re_f, 0.0), _ETA_RE[-1])))


def bartz_sigma(Tw, T0, M, gamma):
    '''
    Bartz property correction factor sigma for a wall at Tw.
    '''

    m = 1 + 0.5*(gamma - 1)*M**2
    return (0.5*Tw/T0*m + 0.5)**(-0.68)*m**(-0.12)


############ Film model ############

class Film_obj:
    def __init__(self, enabled=True, wall_emissivity=0.8, gas_emissivity=None, gas_radiation=True,
                 mixing_coefficient=1e-3, near_wall_mass_fraction=0.1, near_wall_OF=None,
                 oxname=None, fuelname=None, beam_length_factor=0.95, substeps=20):
        '''
        Wall-side gas model: liquid film, mixing of its vapour into the near-wall layer
        (Ievlev boundary layer cooling) and gas radiation.

        enabled : Turns the liquid film on/off. Radiation and the near-wall layer still apply.
        wall_emissivity : Hot-wall emissivity eps_w (RPA table 1).
        gas_emissivity : None to use Leckner, or a fixed value.
        gas_radiation : Include gas radiation to the wall.
        mixing_coefficient : RPA's turbulent mixing intensity K_t, (0.05...0.20)e-2.
        near_wall_mass_fraction : Relative mass flow of the near-wall layer the film vapour mixes into.
        near_wall_OF : Mixture ratio of that layer; None = core mixture ratio.
        oxname, fuelname : RocketCEA propellant names, for the layer chemistry.
        beam_length_factor : Mean beam length / local diameter (0.95 for an infinite cylinder).
        substeps : Integration substeps per regen cell; the film is often only a few cells long.
        '''

        self.enabled = enabled
        self.eps_w = wall_emissivity
        self.gas_emissivity = gas_emissivity
        self.gas_radiation = gas_radiation
        self.K_t = mixing_coefficient
        self.m_s0 = near_wall_mass_fraction
        self.OF_s = near_wall_OF
        self.oxname, self.fuelname = oxname, fuelname
        self.beam_length_factor = beam_length_factor
        self.substeps = substeps
        self._chem = None
        self._tables = {}


    def chemistry(self):
        if self._chem is None:
            from RXPI_CATNAP_Layers import LayerChemistry
            self._chem = LayerChemistry(self.oxname, self.fuelname)
        return self._chem


    def liquid_props(self, coolant, Pc, T_inj):
        '''
        Vaporization temperature and heat at chamber pressure, and the mean liquid
        specific heat between injection and vaporization.
        '''

        T_vap = CP.PropsSI('T', 'P', Pc, 'Q', 0, coolant)
        Q_vap = CP.PropsSI('H', 'P', Pc, 'Q', 1, coolant) - CP.PropsSI('H', 'P', Pc, 'Q', 0, coolant)

        if T_inj < T_vap:
            c_f = (CP.PropsSI('H', 'P', Pc, 'Q', 0, coolant)
                   - CP.PropsSI('H', 'T', T_inj, 'P', Pc, coolant))/(T_vap - T_inj)
        else:
            c_f = CP.PropsSI('CPMASS', 'P', Pc, 'Q', 0, coolant)

        return dict(T_vap=T_vap, Q_vap=Q_vap, c_f=c_f)


    def mu_liquid(self, coolant, T, Pc, T_vap):
        if T < T_vap - 0.05:
            return CP.PropsSI('V', 'T', T, 'P', Pc, coolant)
        return CP.PropsSI('V', 'P', Pc, 'Q', 0, coolant)


    def q_conv(self, g, T_wall):
        '''
        Gas-side convective flux to a wall at T_wall, from the snapshot's Bartz h_g with
        sigma re-evaluated at T_wall (gas_side() evaluates it at the recovery temperature).
        '''

        hg = g['hg']*bartz_sigma(T_wall, g['T0'], g['M'], g['gamma'])/bartz_sigma(g['Taw'], g['T0'], g['M'], g['gamma'])
        return hg*(g['Taw'] - T_wall)


    def q_rad_wall(self, g, R, T_wall, x_h2o, x_co2):
        '''
        Radiative flux reaching the wall through the film, RPA:
        q_r = eps_e sigma (eps_g(T_gas) T_gas^4 - eps_g(T_wall) T_wall^4),
        eps_e = eps_w / [1 - (1 - eps_w)(1 - eps_g(T_gas))].
        '''

        T_gas = g['T_static']
        L = self.beam_length_factor*2*R
        if self.gas_emissivity is None:
            eg_gas = leckner_emissivity(T_gas, g['P'], x_h2o, x_co2, L)
            eg_wall = leckner_emissivity(T_wall, g['P'], x_h2o, x_co2, L)
        else:
            eg_gas = eg_wall = self.gas_emissivity

        eps_e = self.eps_w/(1 - (1 - self.eps_w)*(1 - eg_gas))
        return eps_e*SIGMA_SB*(eg_gas*T_gas**4 - eg_wall*T_wall**4), eg_gas


    def solve(self, z_array, R, gas, mdot_film, mdot_total, T_inj, Pc, OF_core, coolant,
              x_h2o, x_co2, T_w):
        '''
        Wall-side gas state at one snapshot. Returns the liquid film solution (see
        liquid()) plus, for every node:
          q_rad : radiative flux to the wall (W/m^2), at wall temperature T_w
          psi   : convective flux reduction S_layer/S_core from the gas layer touching the wall
          xi    : mixing extent of the film vapour with the near-wall layer
        T_w is the hot-wall temperature from the previous iteration (RPA step 2).
        '''

        n = len(z_array)
        Rz = np.array([R(z) for z in z_array])
        keys = ('hg', 'Taw', 'T0', 'M', 'gamma', 'T_static', 'P')
        nodes = [{k: gas[k][i] for k in keys} for i in range(n)]

        out = self.liquid(z_array, R, gas, mdot_film, T_inj, Pc, coolant, x_h2o, x_co2)

        out['q_rad'] = np.zeros(n)
        if self.gas_radiation:
            out['q_rad'] = np.array([self.q_rad_wall(nodes[i], Rz[i], T_w[i], x_h2o, x_co2)[0] for i in range(n)])
        out['q_wall'] = out['q_rad']           # the liquid film is transparent: radiation only

        out['psi'], out['xi'] = np.ones(n), np.zeros(n)

        has_vapour = self.enabled and mdot_film > 0 and Pc < CP.PropsSI('Pcrit', coolant)
        OF_s = OF_core if self.OF_s is None else self.OF_s
        if not has_vapour and OF_s == OF_core:
            return out

        try:
            chem = self.chemistry()
        except ValueError as e:
            out['warnings'].append(f'Near-wall layer chemistry unavailable ({e}); no film vapour protection.')
            return out

        key = (round(Pc, -3), round(OF_core, 3), round(OF_s, 3))
        if key not in self._tables:
            core = chem.propellant_layer(OF_core, Pc)
            # film vapour at its vaporization temperature, on the card's enthalpy reference
            h_cool = (chem.h_fuel + CP.PropsSI('H', 'P', Pc, 'Q', 1, coolant)
                      - CP.PropsSI('H', 'T', 298.15, 'P', Pc, coolant))
            self._tables[key] = (core, *chem.mixing_table(OF_s, h_cool, Pc))
        core, kgrid, layers = self._tables[key]

        # Mixing of the vapour with the near-wall layer, RPA "Gaseous Film":
        # xi = 1 - exp(-M x/H_s), M = K_t m_s/m_f, measured from where the liquid is gone
        m_f0 = mdot_film/mdot_total if has_vapour else 0.0
        H_s = Rz[-1]*(1 - np.sqrt(1 - self.m_s0))      # annulus carrying m_s0 at uniform mass flux
        s_nodes = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(z_array), np.diff(Rz))[::-1])])[::-1]
        s0 = out['Lc'] if out['active'] else 0.0

        for i in range(n):
            g = nodes[i]
            if m_f0 > 0:
                x = max(s_nodes[i] - s0, 0.0)
                xi = 1 - np.exp(-self.K_t*self.m_s0/m_f0*x/H_s)
                m_f = self.m_s0*xi/2 + m_f0*(1 - xi/2)
                k_ff = m_f0*(1 - xi/2)/m_f
            else:
                xi, k_ff = 1.0, 0.0

            rec = g['Taw']/g['T0']                        # recovery factor of the core at this station
            S_core = chem.S(core, g['Taw'], T_w[i])
            j = min(int(k_ff*(len(kgrid) - 1)), len(kgrid) - 2)
            w = (k_ff - kgrid[j])/(kgrid[j + 1] - kgrid[j])
            S_lay = ((1 - w)*chem.S(layers[j], rec*layers[j]['T0'], T_w[i])
                     + w*chem.S(layers[j + 1], rec*layers[j + 1]['T0'], T_w[i]))

            out['psi'][i] = max(S_lay/S_core, 0.0)
            out['xi'][i] = xi

        it = int(np.argmin(Rz))
        out.update(psi_throat=float(out['psi'][it]), xi_throat=float(out['xi'][it]),
                   q_rad_throat=float(out['q_rad'][it]))
        return out


    def liquid(self, z_array, R, gas, mdot_film, T_inj, Pc, coolant, x_h2o, x_co2):
        '''
        Marches the liquid film from the injector face (z = z_array[-1]) toward the
        nozzle exit (z = 0).

        Parameters
        ----------
        z_array : Axial nodes shared with the regen solver (m), 0 = nozzle exit.
        R : Contour radius function R(z) (m).
        gas : Gas-side arrays from Regen_obj.gas_side().
        mdot_film : Film coolant mass flow rate (kg/s).
        T_inj : Film injection temperature (K).
        Pc : Chamber pressure (Pa).
        coolant : CoolProp name of the film coolant.
        x_h2o, x_co2 : Chamber mole fractions, for gas radiation.

        Returns
        -------
        dict with the film-cooled length Lc (wall arc length from injection), the
        heating length L_heat, the dry-out axial position z_dry, node arrays of film
        temperature, liquid mass flow and fluxes, and the wetted fraction of each regen cell.
        '''

        n = len(z_array)
        out = dict(active=False, coverage=np.zeros(n), mdot_l=np.zeros(n), T_f=np.zeros(n),
                   q_conv=np.zeros(n), q_wall=np.zeros(n), Lc=0.0, L_heat=0.0,
                   z_dry=float(z_array[-1]), warnings=[], T_inj=T_inj)

        if not self.enabled or mdot_film <= 0:
            return out

        if Pc >= CP.PropsSI('Pcrit', coolant):
            out['warnings'].append('Chamber pressure is supercritical for the film coolant; '
                                   'there is no vaporization phase. Film ignored.')
            return out

        liq = self.liquid_props(coolant, Pc, T_inj)
        T_vap, Q_vap, c_f = liq['T_vap'], liq['Q_vap'], liq['c_f']

        # Coolant arriving above T_vap flashes part of itself on injection (energy balance)
        m_f, T_f, flash = mdot_film, T_inj, 0.0
        if T_inj > T_vap:
            flash = c_f*(T_inj - T_vap)/Q_vap
            if flash >= 1:
                out['warnings'].append(f'Film flashes completely on injection (T_inj = {T_inj:.1f} K, '
                                       f'T_vap = {T_vap:.1f} K): no liquid film.')
                return out
            out['warnings'].append(f'Film enters at {T_inj:.1f} K, above T_vap(Pc) = {T_vap:.1f} K: '
                                   f'{100*flash:.1f}% flashes on injection.')
            m_f, T_f = mdot_film*(1 - flash), T_vap

        Rz = np.array([R(z) for z in z_array])
        ds = np.hypot(np.diff(z_array), np.diff(Rz))
        keys = ('hg', 'Taw', 'T0', 'M', 'gamma', 'T_static', 'P')

        def local(j, s):
            # gas state a fraction s of the way from node j+1 to node j
            g = {k: (1 - s)*gas[k][j + 1] + s*gas[k][j] for k in keys}
            return g, (1 - s)*Rz[j + 1] + s*Rz[j]

        def record(j, g, Rj):
            out['T_f'][j] = T_f
            out['mdot_l'][j] = m_f
            out['q_conv'][j] = self.q_conv(g, T_f)
            out['q_wall'][j] = self.q_rad_wall(g, Rj, T_f, x_h2o, x_co2)[0]

        g_inj = {k: gas[k][-1] for k in keys}
        record(n - 1, g_inj, Rz[-1])

        mu0 = self.mu_liquid(coolant, T_f, Pc, T_vap)
        Re_f0 = m_f/(2*pi*Rz[-1]*mu0)
        q_rad0, eps_g0 = self.q_rad_wall(g_inj, Rz[-1], T_f, x_h2o, x_co2)

        s_pos = 0.0                                  # wall arc length from injection
        L_heat = 0.0 if T_f >= T_vap else None
        z_heat = float(z_array[-1]) if T_f >= T_vap else None
        dried = False
        N = self.substeps

        for j in range(n - 2, -1, -1):
            dx = ds[j]/N
            for k in range(N):
                g, Rm = local(j, (k + 0.5)/N)
                left = dx

                if T_f < T_vap:                                            # heating
                    mu = self.mu_liquid(coolant, T_f, Pc, T_vap)
                    eta = film_stability(m_f/(2*pi*Rm*mu))
                    dT = 2*pi*Rm*self.q_conv(g, T_f)*dx/(eta*m_f*c_f)
                    if T_f + dT < T_vap:
                        T_f += dT
                        left = 0.0
                    else:
                        used = (T_vap - T_f)/dT
                        T_f = T_vap
                        left = (1 - used)*dx
                        L_heat = s_pos + used*dx
                        z_heat = float(z_array[j + 1] - (k + used)/N*(z_array[j + 1] - z_array[j]))

                if left > 0:                                                # vaporization
                    dm = 2*pi*Rm*self.q_conv(g, T_vap)*left/Q_vap
                    if dm >= m_f:
                        frac = (dx - left + m_f/dm*left)/dx
                        s_pos += frac*dx
                        z_dry = float(z_array[j + 1] - (k + frac)/N*(z_array[j + 1] - z_array[j]))
                        m_f = 0.0
                        dried = True
                        break
                    m_f -= dm

                s_pos += dx

            if dried:
                break
            record(j, *local(j, 1.0))

        if not dried:
            z_dry = float(z_array[0])
            out['warnings'].append('Liquid film survives to the nozzle exit.')

        if L_heat is None:
            L_heat = s_pos
            out['warnings'].append('Film never reaches its vaporization temperature.')

        # wetted fraction of each regen cell [z_k - dz/2, z_k + dz/2] lying in [z_dry, z_end]
        dz = z_array[1] - z_array[0]
        lo = np.clip(z_array - 0.5*dz, z_array[0], z_array[-1])
        hi = np.clip(z_array + 0.5*dz, z_array[0], z_array[-1])
        out['coverage'] = np.clip((hi - np.maximum(lo, z_dry))/(hi - lo), 0.0, 1.0)

        # the cell holding the dry-out point has no node inside the film; give it the last wetted value
        wet = out['mdot_l'] > 0
        for k in np.where((out['coverage'] > 0) & ~wet)[0]:
            nxt = np.where(wet[k:])[0]
            if len(nxt):
                out['q_wall'][k] = out['q_wall'][k + nxt[0]]

        out.update(active=True, Lc=s_pos, L_heat=L_heat, z_dry=z_dry, z_heat=z_heat,
                   T_vap=T_vap, T_sat=T_vap, Q_vap=Q_vap, c_f=c_f, flash=flash,
                   Re_f_inj=Re_f0, eta_inj=film_stability(Re_f0),
                   q_conv_inj=self.q_conv(g_inj, out['T_f'][-1]), q_rad_inj=q_rad0, eps_g=eps_g0)

        return out
