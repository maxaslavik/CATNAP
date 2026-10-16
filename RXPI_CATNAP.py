# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:01:27 2026

@author: Maximilian Slavik


################################


RXPI CATNAP

Coupled Adiabatic Tank and Nozzle Analysis Program


⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡤⠶⠒⠛⠉⠙⠛⠒⠶⢤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠁⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠈⠙⢦⡀⠀⠀⠀
⠀⠀⠀⠀⣀⣠⠤⠤⡴⠻⢓⣶⠦⠤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀
⣀⡠⠴⠊⠁⠀⠀⠀⠀⠀⠒⠽⠀⠀⠀⠉⢙⠒⢢⡄⠀⠀⠀⠀⠀⠀⠀⠈⢷⡀
⠘⢆⠉⠑⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢆⠁⣠⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣇
⠀⠈⠓⠦⠁⢀⣀⠀⠀⠀⠀⠀⠀⣀⣀⢸⡊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻
⠀⠀⠀⡸⠀⠉⠀⠙⠆⠀⠀⠀⠏⠀⠈⠀⢇⠀⠀⠀⠀⠀⠀⠀⢀⠔⠀⠀⠀⣼
⠀⠀⠀⠹⣄⠀⠠⣤⡡⠪⡭⠃⡤⢤⡄⡰⠋⠀⠀⠀⣀⡠⠴⠊⠁⠀⠀⠀⢠⠇
⠀⠀⢠⢿⣠⠟⠓⠛⠉⠛⡟⠛⠛⠛⠛⠒⠒⠚⠉⠉⠁⠀⠀⠀⠀⠀⢀⡴⠋⠀
⠀⠀⠈⠛⢯⣀⣀⣀⡤⠤⠤⠤⢤⣤⣀⣀⣀⣀⣀⣀⣀⣤⠤⠴⠒⠋⠁⠀⠀⠀

#################################

"""

import json
import os
import webbrowser
from math import pi

import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from scipy.optimize import brentq

#### CATNAP functions
from RXPI_CATNAP_Combustion import SolvePC, CombustionPerformance, Transport_obj

import RXPI_CATNAP_Config as Config


_psi2Pa = 6894.75729

HERE = os.path.dirname(os.path.abspath(__file__))


class Cancelled(Exception):
    pass


def timestep(T1,mdotox,mdotf,x1,dt,m1,v1,Props_obj,dP_piston_psi):
    '''
    Function that evolves the thermodynamic state of an ADIABATIC tank over a time dt.
    Includes piston work, enthalpy loss by mass flow, and specific volume change.

    Parameters
    ----------
    T1 : Saturation temperature at time t (K).
    mdotox : Pressurant mass flow rate at time t (kg/s).
    mdotf : Fuel mass flow rate at time t (kg/s).
    x1 : Pressurant vapor mass fraction at time t.
    dt : Interval timestep.
    m1 : Total mass of pressurant within tank at time t (kg).
    v1 : Specific volume of pressurant at time t (m^3/kg).
    dP_piston_psi : Pressure drop across the piston (psi).

    Returns
    -------
    T2 : Saturation temperature at time t + dt (K).
    x2 : Pressurant vapor mass fraction at time t + dt.
    m2 : Total mass of pressurant within tank at time t + dt (kg).
    v2 : Specific volume of pressurant within tank at time t + dt (m^3/kg).

    phase : STRING. Denotes the phase of the pressurant within the tank.
    -phase = 'saturated' during phase transition, phase = 'vapor' after boiling completes.

    '''

    Pressurant = Props_obj.ox

    Fuel = Props_obj.fuel

    if 0.0 <= x1 and x1 <= 0.999:

        print('Phase: Saturated')

        phase = 'saturated'

        uf1 = CP.PropsSI('U','T',T1,'Q',0,Pressurant)
        ug1 = CP.PropsSI('U','T',T1,'Q',1,Pressurant)
        u1 = uf1 + x1*(ug1 - uf1)

        hf1 = CP.PropsSI('H','T',T1,'Q',0,Pressurant)

        vf1 = 1/(CP.PropsSI('D','T',T1,'Q',0,Pressurant))
        vg1 = 1/CP.PropsSI('D','T',T1,'Q',1,Pressurant)
        v1 = vf1 + x1*(vg1 - vf1)

        P1 = CP.PropsSI('P','T',T1,'Q',0,Pressurant)

        vfuel = 1/(CP.PropsSI('D','T',T1,'P',P1 - dP_piston_psi*_psi2Pa,Fuel))
        Wp = -P1*mdotf*vfuel*dt #work done by piston during dt

        u2 = (Wp - mdotox*dt*hf1 + m1*u1)/(m1 - mdotox*dt) # First law / control volume balance

        v2 = (v1*m1 + mdotf*vfuel*dt)/(m1 - mdotox*dt)

        T2 = brentq(rootT2, T1 - 20, CP.PropsSI('Tcrit',Pressurant) - 2, args=(v2, u2, Pressurant))

        vf2 = 1/(CP.PropsSI('D','T',T2,'Q',0,Pressurant))
        vg2 = 1/(CP.PropsSI('D','T',T2,'Q',1,Pressurant))

        x2 = (v2 - vf2)/(vg2 - vf2)

        m2 = m1 - mdotox*dt
        print('------')
        print()
        print('T2:',T2,'(K)')
        print('Tank Pressure:',P1/_psi2Pa,'psi')
        print('Vapor Mass Fraction',x2*100,'%')
        print()

    elif x1 >= 0.999:
        print('Phase: Vapor')

        phase = 'vapor'

        x2 = 1.0

        D1 = 1/v1

        u1 = CP.PropsSI('U','T',T1,'D',D1,Pressurant)

        hg1 = CP.PropsSI('H','T',T1,'D',D1,Pressurant)

        v1 = 1/(CP.PropsSI('D','T',T1,'D',D1,Pressurant))

        P1 = CP.PropsSI('P','T',T1,'D',D1,Pressurant)

        vfuel = 1/(CP.PropsSI('D','T',T1,'P',P1 - dP_piston_psi*_psi2Pa,Fuel))

        Wp = -P1*mdotf*vfuel*dt # work done by piston during dt

        u2 = (Wp - mdotox*dt*hg1 + m1*u1)/(m1 - mdotox*dt) # First law / control volume balance

        v2 = (v1*m1 + mdotf*vfuel*dt)/(m1 - mdotox*dt)

        m2 = m1 - mdotox*dt

        D2 = 1/v2

        T2 = CP.PropsSI('T','U',u2,'D',D2,Pressurant)

    return T2,x2,m2,v2,phase


def rootT2(T2,v2,u2,Pressurant):
    '''
    Residual to solve for specific internal energy within
    the tank after the thermodynamic process completes over
    interval dt

    '''

    uf = CP.PropsSI('U','T',T2,'Q',0,Pressurant)
    ug = CP.PropsSI('U','T',T2,'Q',1,Pressurant)
    vf = 1/(CP.PropsSI('D','T',T2,'Q',0,Pressurant))
    vg = 1/(CP.PropsSI('D','T',T2,'Q',1,Pressurant))

    x2 = (v2 - vf)/(vg - vf)

    return u2 - uf - x2*(ug - uf)


######################################


def SOLVE_SNAPSHOT(Regen_obj, Film_obj, Injector_obj, Props_obj, Trn,
                   mdot_coolant, mdotfilm, mdot_total, Tcool_init, Pcool_init,
                   T_film_guess, T_w_guess=None, tol=0.5, maxiter=15):
    '''
    Regen + wall-side gas solution at one snapshot, by fixed-point iteration on
      - the film injection temperature: the film is fed from the regen outlet manifold,
        so it is the coolant outlet temperature, which depends on the film protection;
      - the hot-wall temperature profile, which the Bartz sigma factor, gas radiation
        and Ievlev's S ratio depend on (RPA: wall temperatures from the previous iteration).
    '''

    gas = Regen_obj.gas_side(Trn)

    x_h2o, x_co2, _ = Trn.ChamberComposition()

    T_inj = T_film_guess
    T_w = np.full(Regen_obj.numpts_z, 600.0) if T_w_guess is None else np.asarray(T_w_guess, float)
    history = []

    for k in range(maxiter):
        film = Film_obj.solve(Regen_obj.z_array, Regen_obj.R, gas, mdotfilm, mdot_total, T_inj,
                              Trn.Pc, Trn.MR, Props_obj.fuel, x_h2o, x_co2, T_w)

        regen = Regen_obj.SOLVE_REGEN(mdot_coolant, Tcool_init, Pcool_init, Trn,
                                      gas=gas, film=film, T_w_prev=T_w)

        T_out = regen[0][-1]
        history.append((T_inj, float(np.max(T_w))))
        d_inj = abs(T_out - T_inj) if film['active'] else 0.0
        d_wall = float(np.max(np.abs(regen[3] - T_w)))

        if d_inj < tol and d_wall < tol:
            break

        T_inj = T_out
        T_w = regen[3]
    else:
        film['warnings'].append(f'Wall/film coupling did not converge in {maxiter} iterations '
                                f'(last change: film inlet {d_inj:.2f} K, wall {d_wall:.2f} K).')

    film['iterations'] = k + 1
    film['history'] = history

    return gas, film, regen


def tank_outlet(fluid, T, v, phase):
    '''
    State of the oxidizer leaving the tank: saturated liquid while the tank is two-phase,
    the tank vapour once the liquid is gone.
    '''

    if phase == 'saturated':
        return fluid.saturated_liquid(T)
    return fluid.tank_vapor(T, 1/v)


def CATNAP(Tinit, x1, dt, numsteps, m1,
           Regen_obj, Injector_obj, Props_obj, Feed, Film_obj,
           geom, eps, P_amb, dP_piston_psi, Pc_init,
           regen_times=None,
           Tcool_init=290,
           film_tol=0.5, film_maxiter=15,
           thermal=True,
           on_step=None, cancel=None):


    Pressurant = Props_obj.ox

    Fuel = Props_obj.fuel

    T1 = Tinit

    Rthroat = Regen_obj.Rthroat

    At = pi*(Rthroat**2)

    Feed_ox, Feed_fuel = Feed['ox'], Feed['fuel']

    # fuel and film orifices share the regen outlet manifold and split the flow by CdA
    film_share = Injector_obj.CdA_film/(Injector_obj.CdA_fuel + Injector_obj.CdA_film)

    i_regen = Feed_fuel.node_names().index('Regen channels outlet') - 1   # regen inlet node

    vginit = 1/(CP.PropsSI('D','Q',1,'T',T1,Pressurant))

    vfinit = 1/(CP.PropsSI('D','Q',0,'T',T1,Pressurant))

    v1 = vfinit + x1*(vginit - vfinit)

    st_ox = tank_outlet(Feed_ox.fluid, T1, v1, 'saturated' if x1 <= 0.999 else 'vapor')

    mdotox = Feed_ox.solve(st_ox, P_amb).mdot

    mdot_coolant = Feed_fuel.solve(Feed_fuel.fluid.set(T1, st_ox.P), P_amb).mdot

    mdotfilm = film_share*mdot_coolant

    mdotf = mdot_coolant - mdotfilm

    mdot_total = mdotox + mdotf + mdotfilm

    MR = mdotox/(mdotf + mdotfilm)

    Pc = SolvePC(mdot_total, MR, At, Pc_init, Props_obj)

    Pc_arr = []
    P2_arr = []
    T2_arr = []
    x2_arr = []
    F_arr  = []
    Isp_arr = []
    mdot_arr = []
    massratio_arr = []
    Pe_arr = []
    sepmargin_arr = []
    dPchannel_arr = []
    mdot_ox_arr = []
    mdot_fuel_arr = []
    mdot_film_arr = []
    P_ox_nodes_arr = []
    x_ox_nodes_arr = []
    P_fuel_nodes_arr = []


    regen_index_map = {}
    if regen_times is not None:
        for t in regen_times:
            idx = int(round(t / dt))
            if 0 <= idx < numsteps:
                regen_index_map[idx] = t

    numz     = Regen_obj.numpts_z
    n_snap   = len(regen_index_map)
    regen_snap_times = []
    Tcool_3d = np.zeros((n_snap, numz))
    Pcool_3d = np.zeros((n_snap, numz))
    hg_3d    = np.zeros((n_snap, numz))
    Twall_3d = np.zeros((n_snap, numz))
    Qflux_3d = np.zeros((n_snap, numz))
    filmcov_3d = np.zeros((n_snap, numz))
    filmmdot_3d = np.zeros((n_snap, numz))
    snap_count = 0

    transport_3d = []

    tempsC_3d = []

    film_snaps = []

    snapshot_errors = []

    T_film_guess = Tcool_init

    T_w_guess = None

    psi_3d = []

    qrad_3d = []

    for i in range(numsteps):

            if cancel is not None and cancel():
                raise Cancelled()

            time = i*dt
            print(f'Iteration Results (t = {time:.4f}):')
            print(f'Iteration # {i}')
            print()

            #######################

            # the piston displaces the whole fuel flow, film included
            T2,x2,m2,v2,phase = timestep(T1,mdotox,mdot_coolant,x1,dt,m1,v1,Props_obj,dP_piston_psi)

            # Feed networks between the tank and the previous step's chamber pressure
            st_ox = tank_outlet(Feed_ox.fluid, T2, v2, phase)

            P2 = st_ox.P   # tank pressure

            ox_sol = Feed_ox.solve(st_ox, Pc, mdot_guess=mdotox)

            mdotox = ox_sol.mdot

            fuel_sol = Feed_fuel.solve(Feed_fuel.fluid.set(T2, P2), Pc, mdot_guess=mdot_coolant)

            mdot_coolant = fuel_sol.mdot

            mdotfilm = film_share*mdot_coolant

            mdotf = mdot_coolant - mdotfilm

            P2f = fuel_sol.states[i_regen].P   # regen inlet (nozzle exit manifold)

            P2inj = fuel_sol.states[-1].P      # fuel injector manifold = regen outlet

            dPchannels = P2inj - P2f           # P_out - P_in (< 0)

            for sol in (ox_sol, fuel_sol):
                if not sol.converged:
                    print(f'Warning: {sol.names[-1]} feed solve did not converge at t = {time:.3f} s')
                if sol.choked:
                    print(f'Warning: flow choked upstream of the {sol.names[-1]} at t = {time:.3f} s')

            if P2inj < Pc*1.1:
                raise Warning('Injector Stiffness Below 10%! Instability Probable')

            mdot_total = mdotox + mdotf + mdotfilm

            massratio = mdotox/(mdotf + mdotfilm)

            Athroat = pi*(Rthroat**2)

            Pc = SolvePC(mdot_total, massratio, Athroat, Pc, Props_obj)

            Cf, F, Isp, cstar = CombustionPerformance(mdot_total,massratio,Athroat,
                                                    Pc,P_amb,eps,Props_obj)

            # Nozzle exit pressure and margin to the separation criterion P_exit < 0.4 P_amb
            Pe = Pc/Props_obj.C.get_PcOvPe(Pc=Pc, MR=massratio, eps=eps)

            sepmargin = 100*(Pe/(0.4*P_amb) - 1)

            snap = None
            snap_error = None
            gas = None

            if thermal and i in regen_index_map:
                Trn = Transport_obj(mdot_total, massratio, Athroat, Props_obj, geom, eps, Pc)

                Pcool_init = P2f

                try:
                    gas, film, (Tc_arr, Pc_snap, hg_arr, Tw_arr, Qf_arr) = SOLVE_SNAPSHOT(
                        Regen_obj, Film_obj, Injector_obj, Props_obj, Trn,
                        mdot_coolant, mdotfilm, mdot_total, Tcool_init, Pcool_init,
                        T_film_guess, T_w_guess=T_w_guess, tol=film_tol, maxiter=film_maxiter)
                except ValueError as e:
                    # e.g. laminar channel flow, where the heat transfer correlations don't apply.
                    # The snapshot is dropped; the time march doesn't depend on it.
                    print(f'[regen t={regen_index_map[i]:.2f} s] snapshot skipped: {e}')
                    snap_error = {'t': float(regen_index_map[i]), 'error': str(e)}
                    snapshot_errors.append(snap_error)
                    gas = None

            if gas is not None:

                trans_snap = np.column_stack([gas['cp'], gas['mu'], gas['k'],
                                              gas['Pr'], gas['gamma'], gas['M']])
                transport_3d.append(trans_snap.tolist())

                tempsC_3d.append(np.column_stack([gas['Taw'], gas['T0']]).tolist())

                Tcool_3d[snap_count] = Tc_arr
                Pcool_3d[snap_count] = Pc_snap
                hg_3d[snap_count]    = hg_arr
                Twall_3d[snap_count] = Tw_arr
                Qflux_3d[snap_count] = Qf_arr

                T_w_guess = Tw_arr
                psi_3d.append(film['psi'].tolist())
                qrad_3d.append(film['q_rad'].tolist())

                if film is not None:
                    filmcov_3d[snap_count] = film['coverage']
                    filmmdot_3d[snap_count] = film['mdot_l']
                    if film['active']:
                        T_film_guess = film['T_inj']
                    for w in film['warnings']:
                        print(f'[film t={regen_index_map[i]:.2f} s] {w}')

                snap = film_summary(regen_index_map[i], film, Tc_arr[-1], Tw_arr)
                film_snaps.append(snap)

                regen_snap_times.append(regen_index_map[i])
                snap_count += 1

            Pc_arr.append(Pc)
            P2_arr.append(P2)
            T2_arr.append(T2)
            x2_arr.append(x2)
            F_arr.append(F)
            Isp_arr.append(Isp)
            mdot_arr.append(mdot_total)
            massratio_arr.append(massratio)
            Pe_arr.append(Pe)
            sepmargin_arr.append(sepmargin)
            dPchannel_arr.append(-dPchannels)
            mdot_ox_arr.append(mdotox)
            mdot_fuel_arr.append(mdot_coolant)
            mdot_film_arr.append(mdotfilm)
            P_ox_nodes_arr.append(ox_sol.node_pressures())
            x_ox_nodes_arr.append(ox_sol.node_qualities())
            P_fuel_nodes_arr.append(fuel_sol.node_pressures())

            T1 = T2
            x1 = x2
            m1 = m2
            v1 = v2

            print('------')
            print()
            print(f'Thrust:           {F/1000:.3f} kN')
            print(f'O/F Ratio:        {massratio:.3f}')
            print(f'Chamber Pressure: {Pc/_psi2Pa:.2f} psi')
            print(f'Isp:              {Isp:.2f} s')
            print(f'Tank Pressure:    {P2/_psi2Pa:.2f} psi')
            print(f'Ox Manifold:      {ox_sol.states[-1].P/_psi2Pa:.2f} psi   (quality {ox_sol.states[-1].x:.4f})')
            print(f'Fuel Manifold:    {P2inj/_psi2Pa:.2f} psi')
            print(f'Mass Flow Rate:   {mdot_total:.2f} kg/s')
            print('------')

            if on_step is not None:
                on_step(dict(i=i, t=time, Pc=Pc, P2=P2, T2=T2, x2=x2, F=F, Isp=Isp,
                             mdot=mdot_total, MR=massratio, phase=phase,
                             Pe=Pe, sep_margin=sepmargin, dP_channel=-dPchannels,
                             mdot_ox=mdotox, mdot_fuel=mdot_coolant,
                             P_ox_inj=ox_sol.states[-1].P, x_ox_inj=ox_sol.states[-1].x,
                             P_fuel_inj=P2inj,
                             frac=(i + 1)/numsteps, snapshot=snap, snapshot_error=snap_error))

    Imp = 0

    for i in range(numsteps - 1):
        trap = 0.5*(F_arr[i] + F_arr[i+1])*dt

        Imp += trap

    print('===== SIMULATION RESULTS=====')

    print('Total Impulse (N*s):',Imp)

    for f in (Feed_ox, Feed_fuel):
        print(f'{f.name} feed: {f.total_evals/max(f.solves, 1):.2f} residual evaluations per solve')

    return {
        'timevec':       (np.arange(numsteps)*dt).tolist(),
        'Pc_arr':        np.asarray(Pc_arr).tolist(),
        'P2_arr':        np.asarray(P2_arr).tolist(),
        'T2_arr':        np.asarray(T2_arr).tolist(),
        'x2_arr':        np.asarray(x2_arr).tolist(),
        'F_arr':         np.asarray(F_arr).tolist(),
        'Isp_arr':       np.asarray(Isp_arr).tolist(),
        'mdot_arr':      np.asarray(mdot_arr).tolist(),
        'massratio_arr': np.asarray(massratio_arr).tolist(),
        'Pe_arr':        np.asarray(Pe_arr).tolist(),
        'sepmargin_arr': np.asarray(sepmargin_arr).tolist(),
        'dPchannel_arr': np.asarray(dPchannel_arr).tolist(),
        'mdot_ox_arr':   mdot_ox_arr,
        'mdot_fuel_arr': mdot_fuel_arr,
        'mdot_film_arr': mdot_film_arr,
        'feed': {f.name: {'nodes': f.node_names(), 'components': f.describe(),
                          'evals_per_solve': f.total_evals/max(f.solves, 1)}
                 for f in (Feed_ox, Feed_fuel)},
        'P_ox_nodes_arr':   P_ox_nodes_arr,
        'x_ox_nodes_arr':   x_ox_nodes_arr,
        'P_fuel_nodes_arr': P_fuel_nodes_arr,
        'P_amb':         float(P_amb),
        'thermal_model': bool(thermal),
        'impulse':       float(Imp),
        'z_array':       Regen_obj.z_array.tolist(),
        'regen_times':   regen_snap_times,
        'Tcool_3d':      Tcool_3d[:snap_count].tolist(),
        'Pcool_3d':      Pcool_3d[:snap_count].tolist(),
        'hg_3d':         hg_3d[:snap_count].tolist(),
        'Twall_3d':      Twall_3d[:snap_count].tolist(),
        'Qflux_3d':      Qflux_3d[:snap_count].tolist(),
        'filmcov_3d':    filmcov_3d[:snap_count].tolist(),
        'filmmdot_3d':   filmmdot_3d[:snap_count].tolist(),
        'psi_3d':        psi_3d,
        'qrad_3d':       qrad_3d,
        'transport_3d':  transport_3d,
        'tempsC_3d':     tempsC_3d,
        'film':          film_snaps,
        'snapshot_errors': snapshot_errors,
        'nozzle_R':      [float(Regen_obj.R(zi)) for zi in Regen_obj.z_array],
    }


def film_summary(t, film, Tcool_out, Twall):
    '''
    JSON-friendly scalars for one snapshot.
    '''

    s = {'t': float(t), 'active': False, 'Tcool_out': float(Tcool_out),
         'Twall_max': float(np.max(Twall)), 'warnings': []}

    if film is None:
        return s

    s['warnings'] = list(film['warnings'])
    s['active'] = bool(film['active'])
    s['iterations'] = int(film.get('iterations', 1))
    for k in ('psi_throat', 'xi_throat', 'q_rad_throat'):
        s[k] = float(film.get(k, 1.0 if k == 'psi_throat' else 0.0))

    if film['active']:
        for k in ('Lc', 'L_heat', 'z_dry', 'T_inj', 'T_vap', 'Q_vap', 'c_f', 'flash',
                  'Re_f_inj', 'eta_inj', 'q_conv_inj', 'q_rad_inj', 'eps_g'):
            s[k] = float(film[k])
        s['T_sat'] = s['T_vap']

    return s


######################################


def run(vehicle, sim, on_step=None, cancel=None):
    '''
    Runs CATNAP from vehicle and sim config dicts (see RXPI_CATNAP_Config).
    '''

    obj = Config.build(vehicle, sim)

    dt = sim['time']['duration_s']/sim['time']['steps']

    results = CATNAP(sim['tank']['temperature_K'], sim['tank']['vapor_quality'], dt,
                     int(sim['time']['steps']), sim['tank']['propellant_mass_kg'],
                     obj['regen'], obj['injector'], obj['props'], obj['feed'], obj['film'],
                     obj['geom'], obj['eps'],
                     P_amb=sim['environment']['ambient_pressure_bar']*1e5,
                     dP_piston_psi=sim['tank']['piston_dp_psi'],
                     Pc_init=sim['solver']['chamber_pressure_guess_psi']*_psi2Pa,
                     regen_times=sim['regen']['snapshot_times_s'],
                     Tcool_init=sim['regen']['coolant_inlet_temperature_K'],
                     film_tol=sim['solver']['film_coupling_tol_K'],
                     film_maxiter=int(sim['solver']['film_coupling_max_iter']),
                     thermal=bool(sim['regen']['thermal_model']),
                     on_step=on_step, cancel=cancel)

    results['engine'] = vehicle['name']
    results['propellants'] = vehicle['propellants']
    results['derived_geometry'] = Config.derived_geometry(vehicle)

    return results


def write_dashboard(results, outdir=HERE):
    '''
    Writes catnap_results.json and a self-contained catnap_results.html.
    Returns the HTML path, or None if the dashboard template is missing.
    '''

    with open(os.path.join(outdir, 'catnap_results.json'), 'w') as f:
        json.dump(results, f)

    try:
        with open(os.path.join(HERE, 'catnap_dashboard.html'), 'r', encoding='utf-8') as f:
            html = f.read()
    except FileNotFoundError:
        return None

    inject = '<script>var AUTOLOAD = ' + json.dumps(results) + ';</script>'
    html = html.replace('</head>', inject + '\n</head>')

    output_path = os.path.abspath(os.path.join(outdir, 'catnap_results.html'))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


def downloads_dir():
    path = os.path.join(os.path.expanduser('~'), 'Downloads')
    return path if os.path.isdir(path) else HERE


def write_csv(results, path=None):
    '''
    Bundles every plotted quantity into one comma-separated .txt file (Downloads by
    default). Time series, axial profiles and per-snapshot film values have different
    lengths, so they sit side by side and shorter columns are padded with blanks.
    Read it back with pandas.read_csv(path, comment='#').
    '''

    import csv
    from datetime import datetime

    if path is None:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        name = str(results.get('engine', 'engine')).replace(' ', '_')
        path = os.path.join(downloads_dir(), f'catnap_{name}_{stamp}.txt')

    r = results
    MR = np.asarray(r['massratio_arr'])
    mdot = np.asarray(r['mdot_arr'])

    cols = [
        ('t_s',                    r['timevec']),
        ('Pc_bar',                 np.asarray(r['Pc_arr'])/1e5),
        ('P_tank_bar',             np.asarray(r['P2_arr'])/1e5),
        ('T_tank_K',               r['T2_arr']),
        ('vapor_quality',          r['x2_arr']),
        ('thrust_N',               r['F_arr']),
        ('Isp_s',                  r['Isp_arr']),
        ('mdot_total_kg_s',        mdot),
        ('mdot_ox_kg_s',           mdot*MR/(1 + MR)),
        ('mdot_fuel_kg_s',         mdot/(1 + MR)),
        ('OF',                     MR),
        ('P_exit_bar',             np.asarray(r['Pe_arr'])/1e5),
        ('separation_margin_pct',  r['sepmargin_arr']),
        ('channel_dP_bar',         np.asarray(r['dPchannel_arr'])/1e5),
    ]

    if 'feed' in r:
        cols.append(('mdot_film_kg_s', r['mdot_film_arr']))
        for side, P_key in (('ox', 'P_ox_nodes_arr'), ('fuel', 'P_fuel_nodes_arr')):
            P = np.asarray(r[P_key])/1e5
            for j, node in enumerate(r['feed'][side]['nodes'][1:-1], start=1):
                cols.append((f'P_{side}_{node.replace(" ", "_")}_bar', P[:, j]))
        x = np.asarray(r['x_ox_nodes_arr'])
        for j, node in enumerate(r['feed']['ox']['nodes'][1:-1], start=1):
            cols.append((f'quality_ox_{node.replace(" ", "_")}', x[:, j]))

    if r['regen_times']:
        cols += [('z_m', r['z_array']), ('R_m', r['nozzle_R'])]
        for k, t in enumerate(r['regen_times']):
            tag = f'@t={t:g}s'
            cols += [
                ('T_wall_K' + tag,      r['Twall_3d'][k]),
                ('T_cool_K' + tag,      r['Tcool_3d'][k]),
                ('P_cool_bar' + tag,    np.asarray(r['Pcool_3d'][k])/1e5),
                ('q_flux_MW_m2' + tag,  np.asarray(r['Qflux_3d'][k])/1e6),
                ('hg_W_m2K' + tag,      r['hg_3d'][k]),
                ('film_coverage' + tag, r['filmcov_3d'][k]),
                ('convective_reduction_psi' + tag, r['psi_3d'][k]),
                ('q_rad_wall_MW_m2' + tag, np.asarray(r['qrad_3d'][k])/1e6),
            ]

        film = r['film']
        cols.append(('snapshot_t_s', [s['t'] for s in film]))
        for key, label, scale in (('Lc', 'film_Lc_m', 1), ('L_heat', 'film_heating_length_m', 1),
                                  ('eta_inj', 'film_stability_eta', 1),
                                  ('T_inj', 'film_T_inj_K', 1), ('T_vap', 'film_T_vap_K', 1),
                                  ('q_conv_inj', 'film_q_conv_MW_m2', 1e-6),
                                  ('q_rad_inj', 'film_q_rad_wall_MW_m2', 1e-6), ('eps_g', 'gas_emissivity', 1)):
            cols.append((label, [s[key]*scale if s.get('active') else '' for s in film]))
        cols.append(('T_wall_max_K', [s['Twall_max'] for s in film]))
        cols.append(('T_cool_out_K', [s['Tcool_out'] for s in film]))

    nrows = max(len(c[1]) for c in cols)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write(f"# CATNAP results  engine={r.get('engine', '')}  P_amb={r['P_amb']/1e5:g} bar  "
                f"thermal_model={r['thermal_model']}  impulse_Ns={r['impulse']:.6g}\n")
        f.write('# Separation criterion: P_exit < 0.4 P_amb; margin = 100 (P_exit/(0.4 P_amb) - 1), '
                'negative = separated. Columns of different length are blank-padded.\n')
        for e in r.get('snapshot_errors', []):
            f.write(f"# Regen snapshot at t={e['t']:g} s skipped: {e['error']}\n")
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for i in range(nrows):
            w.writerow([(f'{float(c[1][i]):.8g}' if c[1][i] != '' else '') if i < len(c[1]) else ''
                        for c in cols])

    return path


def plot_results(results, vehicle):
    '''
    Contour, feed system, performance and mass flow plots.
    '''

    with plt.style.context('dark_background'):

        R, Le, _ = Config.make_contour(vehicle['contour'])

        numpts = 2000
        z = np.linspace(0,Le,numpts)
        zout = np.array([R(zi) for zi in z])/Config._in2m

        z = z/Config._in2m

        nozzlecolor = 'steelblue'
        contournumber = 7
        gridnumber = 15

        plt.figure(figsize=(8,4))
        plt.plot(z,zout,color=nozzlecolor)
        plt.plot(z,-zout,color=nozzlecolor)
        plt.plot([0,0],[zout[0],-zout[0]],color=nozzlecolor)
        plt.plot([z[-1],z[-1]],[zout[-1],-zout[-1]],color=nozzlecolor)

        for i in range(contournumber):
            a = i/contournumber
            plt.plot(z,zout*a,color=nozzlecolor)
            plt.plot(z,-zout*a,color=nozzlecolor)

        for i in range(gridnumber):
            grdx = round((i/gridnumber)*numpts)
            plt.plot([z[grdx],z[grdx]],[zout[grdx],-zout[grdx]],color=nozzlecolor)

        plt.axis('equal')
        plt.title(f"{vehicle['name']} Nozzle Contour (inches)")
        plt.show()

        timevec = np.asarray(results['timevec'])
        Pc_arr = np.asarray(results['Pc_arr'])
        P2_arr = np.asarray(results['P2_arr'])
        T2_arr = np.asarray(results['T2_arr'])
        F_arr = np.asarray(results['F_arr'])
        Isp_arr = np.asarray(results['Isp_arr'])
        mdot_arr = np.asarray(results['mdot_arr'])
        massratio_arr = np.asarray(results['massratio_arr'])

        factor = massratio_arr/(1 + massratio_arr)
        mdot_oxarr = factor*mdot_arr
        mdot_fuelarr = (1-factor)*mdot_arr

        def twin(title, figsize, left, right):
            fig, ax1 = plt.subplots(figsize=figsize)
            plt.title(title, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.5)
            for y, label, color, ls in left[1]:
                ax1.plot(timevec, y, label=label, color=color, linewidth=2, linestyle=ls)
            ax1.set_ylabel(left[0], fontsize=14)
            ax1.set_ylim(bottom=0)
            ax1.set_xlabel('Time (s)', fontsize=14)
            ax2 = ax1.twinx()
            for y, label, color, ls in right[1]:
                ax2.plot(timevec, y, label=label, color=color, linewidth=2, linestyle=ls)
            ax2.set_ylabel(right[0], fontsize=14)
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='best')
            plt.show()

        twin('Feed System Temperatures and Pressures', (10, 6),
             ('Pressure (psi)', [(P2_arr/_psi2Pa, 'Tank Pressure (psi)', 'mediumblue', '-'),
                                 (Pc_arr/_psi2Pa, 'Chamber Pressure (psi)', 'mediumpurple', '-')]),
             ('Temperature (K)', [(T2_arr, 'Saturation Temperature (K)', 'red', '-')]))

        if 'feed' in results:
            names_ox = results['feed']['ox']['nodes']
            names_f = results['feed']['fuel']['nodes']
            P_ox = np.asarray(results['P_ox_nodes_arr'])/_psi2Pa
            P_f = np.asarray(results['P_fuel_nodes_arr'])/_psi2Pa
            x_ox = np.asarray(results['x_ox_nodes_arr'])
            ox_cols = plt.cm.Oranges(np.linspace(0.45, 0.95, P_ox.shape[1]))
            f_cols = plt.cm.Blues(np.linspace(0.45, 0.95, P_f.shape[1]))
            twin('Feed Network Node Pressures', (12, 7),
                 ('Pressure (psi)', [(P_ox[:, j], f'Ox: {n}', ox_cols[j], '-') for j, n in enumerate(names_ox)] +
                                    [(P_f[:, j], f'Fuel: {n}', f_cols[j], '--') for j, n in enumerate(names_f)]),
                 ('Ox Vapor Quality', [(x_ox[:, j], f'Ox quality: {n}', 'gainsboro', ls)
                                       for j, (n, ls) in enumerate(zip(names_ox[1:-1], (':', 'dashdot', '--')), start=1)]))

        twin('Engine Thrust and Specific Impulse', (10, 6),
             ('Specific Impulse (s)', [(Isp_arr, 'Specific Impulse (s)', 'orchid', '-')]),
             ('Thrust (kN)', [(F_arr/1000, 'Thrust (kN)', 'aquamarine', '-')]))

        p = vehicle['propellants']
        twin('Feed System Mass Flow Rates', (12, 9),
             ('Mass Flow Rate (kg/s)', [(mdot_oxarr, f"{p['oxidizer_coolprop']} Mass Flow Rate", 'lightsalmon', '-'),
                                        (mdot_fuelarr, f"{p['fuel_coolprop']} Mass Flow Rate", 'cornflowerblue', '-')]),
             ('Mass Ratio', [(massratio_arr, 'O/F Ratio', 'gainsboro', 'dashdot')]))

        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.title('Nozzle Exit Separation Margin  (criterion: P_exit < 0.4 P_amb)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        margin = np.asarray(results['sepmargin_arr'])
        ax1.plot(timevec, margin, color='tomato', linewidth=2, label='Separation margin (%)')
        ax1.axhline(0, color='white', linewidth=1, linestyle=':')
        ax1.fill_between(timevec, margin, 0, where=margin < 0, color='tomato', alpha=0.25,
                         label='Separated')
        ax1.set_ylabel('Margin (%)', fontsize=14)
        ax1.set_xlabel('Time (s)', fontsize=14)
        ax2 = ax1.twinx()
        ax2.plot(timevec, np.asarray(results['Pe_arr'])/1e5, color='gainsboro', linewidth=2,
                 linestyle='dashdot', label='Exit pressure (bar)')
        ax2.set_ylabel('Exit Pressure (bar)', fontsize=14)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='best')
        plt.show()

        twin('Regen Channel Pressure Drop (isothermal Darcy-Weisbach)', (10, 6),
             ('Channel dP (bar)', [(np.asarray(results['dPchannel_arr'])/1e5, 'Channel dP (bar)', 'mediumseagreen', '-')]),
             ('Tank Pressure (bar)', [(P2_arr/1e5, 'Tank Pressure (bar)', 'mediumblue', '--')]))


if __name__ == '__main__':

    import argparse

    ap = argparse.ArgumentParser(description='Run CATNAP from vehicle and sim YAML files.')
    ap.add_argument('--vehicle', default=os.path.join(HERE, 'configs', 'altair_vehicle.yaml'))
    ap.add_argument('--sim', default=os.path.join(HERE, 'configs', 'altair_sim.yaml'))
    args = ap.parse_args()

    vehicle, unknown_v = Config.load_config(args.vehicle, Config.VEHICLE_KIND)
    sim, unknown_s = Config.load_config(args.sim, Config.SIM_KIND)

    for k in unknown_v + unknown_s:
        print(f'Warning: unknown config key {k!r} ignored')

    results = run(vehicle, sim)

    for s in results['film']:
        if s['active']:
            print(f"t = {s['t']:6.2f} s   film L_c = {100*s['Lc']:6.2f} cm   "
                  f"heating {100*s['L_heat']:6.2f} cm   T_inj = {s['T_inj']:6.1f} K")

    if sim['output']['csv']:
        print(f'CSV exported to {write_csv(results)}')

    if sim['output']['plots']:
        plot_results(results, vehicle)

    if sim['output']['dashboard']:
        output_path = write_dashboard(results)
        if output_path:
            print(f'Dashboard exported to {output_path}')
            webbrowser.open('file:///' + output_path)
