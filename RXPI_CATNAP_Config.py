# -*- coding: utf-8 -*-
"""
CATNAP configuration files.

Two YAML files describe a run:
  vehicle - hardware: propellants, contour, injector, valves, feed lines, regen channels, film
  sim     - operating point and numerics: burn time, tank state, snapshots, solver settings

Every input lives in exactly one place. Quantities that follow from others (throat
and exit diameter, expansion ratio, chamber radius) are derived from the contour.
"""

import copy
from math import pi, tan, sqrt, atan, degrees

import numpy as np
import yaml


_in2m = 1 / 39.3700787

VEHICLE_KIND = 'catnap-vehicle'
SIM_KIND = 'catnap-sim'
SCHEMA_VERSION = 1


DEFAULT_VEHICLE = {
    'kind': VEHICLE_KIND,
    'version': SCHEMA_VERSION,
    'name': 'Altair',
    'propellants': {
        'oxidizer_cea': 'N2O',
        'fuel_cea': 'ETHANOL',
        'oxidizer_coolprop': 'N2O',
        'fuel_coolprop': 'ETHANOL',
    },
    # Altair, RPA conical-nozzle geometry: Dc 106.67, Dt 52.07, De 114.55 mm, R1 39.05, R2 82.37 mm,
    # Te 15 deg, L* 740 mm. The contour has two tangent arcs and no straight converging section
    # or downstream throat arc, so the arcs meet at 39.2 deg (RPA: 30 deg cone) and the cylinder
    # length is set to keep L* = 740 mm. Dt, De, Dc, R1, R2 and Te are exact.
    'contour': {                       # inches, to integrate with NX CAD
        'nozzle_length_in': 4.5901,
        'arc1_length_in': 0.9713,
        'arc2_length_in': 2.0488,
        'chamber_length_in': 5.0123,
        'exit_radius_in': 2.2549,
        'arc1_radius_in': 1.5374,
        'arc2_radius_in': 3.2429,
        'divergence_angle_deg': 15.0,
        'throat_curvature_radius_mm': 24.5,   # Bartz r_c, mean of R1 and Rn (9.95 mm)
    },
    'injector': {
        'ox_holes': 25,
        'ox_diameter_mm': 2.533,
        'ox_cd': 0.55,
        'fuel_holes': 20,
        'fuel_diameter_mm': 1.240,
        'fuel_cd': 0.65,
        'film_holes': 20,
        'film_diameter_mm': 0.480,
        'film_cd': 0.65,
    },
    'valves': {
        'ox_cv': 24.0,
        'fuel_cv': 12.0,
    },
    # Feed lines, AN/JIC tube (dash = OD in 1/16 in). Ox: tank -> main valve -> line -> injector.
    # Fuel: tank -> piston -> main valve -> line -> regen inlet (nozzle exit) -> channels -> injector + film.
    'lines': {
        'ox_dash': 12,
        'ox_wall_in': 0.049,
        'ox_length_in': 36.0,
        'ox_minor_loss_K': 0.0,        # fittings and bends, sum of K
        'fuel_dash': 8,
        'fuel_wall_in': 0.035,
        'fuel_length_in': 36.0,
        'fuel_minor_loss_K': 0.0,
        'roughness_um': 1.5,           # drawn stainless tube
    },
    'regen': {
        'wall_thickness_mm': 1.5,
        'channel_height_mm': 1.0,
        'min_channel_width_mm': 1.5,
        'channels': 90,
        'wall_conductivity_W_mK': 237.0,
        'roughness_um': 15.0,
        'generatrix_angle_deg': 35.0,
        'gas_radiation': True,
        'wall_emissivity': 0.8,        # hot wall (RPA table 1: oxidized copper 0.57-0.87)
        'gas_emissivity': None,        # null -> Leckner H2O/CO2
    },
    'film': {                          # RPA liquid film + Ievlev near-wall layer mixing
        'enabled': True,
        'mixing_coefficient': 1e-3,    # K_t, RPA: (0.05...0.20)e-2
        'near_wall_mass_fraction': 0.1,
        'near_wall_OF': None,          # null -> core mixture ratio
    },
    'tbc': {                           # PDMS fuel additive: scales the wall heat flux at every node
        'pdms_mass_fraction_pct': 0.0, # % of fuel mass; 8 % flux reduction per 1 % (conservative)
    },
}


DEFAULT_SIM = {
    'kind': SIM_KIND,
    'version': SCHEMA_VERSION,
    'time': {
        'duration_s': 25.0,
        'steps': 150,
    },
    'tank': {
        'propellant_mass_kg': 54.0,
        'temperature_K': 290.0,
        'vapor_quality': 0.01,
        'piston_dp_psi': 15.0,
    },
    'environment': {
        'ambient_pressure_bar': 0.810852,         # 0.8 atm; also sets the separation criterion
    },
    'regen': {
        'snapshot_times_s': [float(t) for t in np.round(np.linspace(0.5, 18, 7), 4)],
        'thermal_model': True,         # False: isothermal Darcy-Weisbach channel dP only
        'coolant_inlet_temperature_K': 290.0,
        'axial_points': 80,
    },
    'solver': {
        'chamber_pressure_guess_psi': 340.0,
        'film_coupling_tol_K': 0.5,
        'film_coupling_max_iter': 15,
    },
    'output': {
        'plots': True,
        'dashboard': True,
        'csv': True,                   # all plotted data as one CSV .txt in Downloads
    },
}


# AN / JIC tube: dash number = OD in 1/16 in
AN_DASH_SIZES = (3, 4, 5, 6, 8, 10, 12, 16)

# common wall thicknesses (in) of 304/316 stainless hydraulic tube, as GUI hints
AN_COMMON_WALLS_IN = {3: (0.020, 0.028, 0.035), 4: (0.028, 0.035, 0.049), 5: (0.035, 0.049),
                      6: (0.035, 0.049, 0.065), 8: (0.035, 0.049, 0.065), 10: (0.035, 0.049, 0.065),
                      12: (0.049, 0.065, 0.083), 16: (0.049, 0.065, 0.083)}


# 1 Cv (US gpm of water at 1 psi) as an equivalent CdA: Q = CdA*sqrt(2 dP/rho) with rho = 1000
CV_TO_CDA = 6.30901964e-5*sqrt(1000/(2*6894.75729))   # m^2 per Cv (~ 1/38 in^2)


def an_tube_id_in(dash, wall_in):
    return dash/16 - 2*wall_in


############ YAML I/O ############

class ConfigError(ValueError):
    pass


def _merge(defaults, user, path, unknown):
    out = copy.deepcopy(defaults)
    for k, v in user.items():
        if k not in defaults:
            unknown.append(path + k)
            out[k] = v
        elif isinstance(defaults[k], dict):
            if not isinstance(v, dict):
                raise ConfigError(f"'{path + k}' must be a section, got {v!r}")
            out[k] = _merge(defaults[k], v, path + k + '.', unknown)
        else:
            out[k] = v
    return out


def load_config(path, kind):
    '''
    Reads a vehicle or sim YAML file. Missing keys are filled from the defaults;
    unknown keys are kept and reported.

    Returns
    -------
    cfg : dict
    unknown : list of dotted key names that are not part of the schema
    '''

    defaults = DEFAULT_VEHICLE if kind == VEHICLE_KIND else DEFAULT_SIM

    with open(path, 'r', encoding='utf-8') as f:
        user = yaml.safe_load(f) or {}

    if not isinstance(user, dict):
        raise ConfigError(f'{path} does not contain a YAML mapping')

    # schema 1 stored ambient pressure in kPa
    env = user.get('environment')
    if kind == SIM_KIND and isinstance(env, dict) and 'ambient_pressure_kPa' in env:
        env.setdefault('ambient_pressure_bar', env.pop('ambient_pressure_kPa')/100)

    # radiation inputs moved from 'film' to 'regen'
    film = user.get('film')
    if kind == VEHICLE_KIND and isinstance(film, dict):
        reg = user.setdefault('regen', {})
        for k in ('wall_emissivity', 'gas_emissivity'):
            if k in film:
                reg.setdefault(k, film.pop(k))

    found = user.get('kind', kind)
    if found != kind:
        raise ConfigError(f"{path} is a '{found}' file, expected '{kind}'")

    unknown = []
    cfg = _merge(defaults, user, '', unknown)
    validate(cfg, kind)

    return cfg, unknown


def save_config(cfg, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(_plain(cfg), f, sort_keys=False, default_flow_style=False)


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def validate(cfg, kind):
    def num(sec, key, lo=None, hi=None, integer=False):
        v = cfg[sec][key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"'{sec}.{key}' must be a number, got {v!r}")
        if integer and int(v) != v:
            raise ConfigError(f"'{sec}.{key}' must be an integer, got {v!r}")
        if lo is not None and v < lo:
            raise ConfigError(f"'{sec}.{key}' must be >= {lo}, got {v}")
        if hi is not None and v > hi:
            raise ConfigError(f"'{sec}.{key}' must be <= {hi}, got {v}")

    if kind == VEHICLE_KIND:
        for k in cfg['contour']:
            num('contour', k, lo=0)
        num('contour', 'divergence_angle_deg', hi=60)
        for k in cfg['injector']:
            num('injector', k, lo=0, integer=k.endswith('_holes'))
        for k in cfg['valves']:
            num('valves', k, lo=1e-6)
        for k in cfg['lines']:
            num('lines', k, lo=0, integer=k.endswith('_dash'))
        for side in ('ox', 'fuel'):
            dash = cfg['lines'][f'{side}_dash']
            if dash not in AN_DASH_SIZES:
                raise ConfigError(f"'lines.{side}_dash' must be one of {', '.join(f'-{d}' for d in AN_DASH_SIZES)}, got {dash}")
            if not 0 < cfg['lines'][f'{side}_wall_in'] < dash/32:
                raise ConfigError(f"'lines.{side}_wall_in' must be between 0 and half the -{dash} tube OD ({dash/32:.4g} in)")
        for k in cfg['regen']:
            if k not in ('gas_radiation', 'gas_emissivity'):
                num('regen', k, lo=0, integer=(k == 'channels'))
        num('regen', 'wall_emissivity', lo=0, hi=1)
        num('film', 'mixing_coefficient', lo=0)
        num('film', 'near_wall_mass_fraction', lo=1e-6, hi=1)
        if cfg['film']['near_wall_OF'] is not None:
            num('film', 'near_wall_OF', lo=0)
        if cfg['regen']['gas_emissivity'] is not None:
            num('regen', 'gas_emissivity', lo=0, hi=1)
        num('tbc', 'pdms_mass_fraction_pct', lo=0, hi=10)   # linear 8 %/% model is not extrapolated past 10 %
        derived_geometry(cfg)   # raises on an impossible contour

    else:
        num('time', 'duration_s', lo=0)
        num('time', 'steps', lo=2, integer=True)
        num('tank', 'propellant_mass_kg', lo=0)
        num('tank', 'temperature_K', lo=0)
        num('tank', 'vapor_quality', lo=0, hi=1)
        num('tank', 'piston_dp_psi', lo=0)
        num('environment', 'ambient_pressure_bar', lo=0)
        num('regen', 'coolant_inlet_temperature_K', lo=0)
        num('regen', 'axial_points', lo=3, integer=True)
        num('solver', 'chamber_pressure_guess_psi', lo=0)
        num('solver', 'film_coupling_tol_K', lo=0)
        num('solver', 'film_coupling_max_iter', lo=1, integer=True)
        times = cfg['regen']['snapshot_times_s']
        if not isinstance(times, list) or not all(isinstance(t, (int, float)) for t in times):
            raise ConfigError("'regen.snapshot_times_s' must be a list of numbers")


############ Geometry ############

def make_contour(c):
    '''
    Thrust chamber contour from the vehicle 'contour' section. Two arcs for the
    converging section and a conical diverging section, z = 0 at the nozzle exit.

    Returns
    -------
    R : callable, R(z) in m
    Le : engine length in m
    geom : [Lnozzle, Lcon1, Lcon2, Lcham, Rexit, Rc1, Rc2, Rcham] in m
    '''

    Lnozzle = c['nozzle_length_in']*_in2m
    Lcon1 = c['arc1_length_in']*_in2m
    Lcon2 = c['arc2_length_in']*_in2m
    Lcham = c['chamber_length_in']*_in2m
    Rexit = c['exit_radius_in']*_in2m
    Rc1 = c['arc1_radius_in']*_in2m
    Rc2 = c['arc2_radius_in']*_in2m
    expansionangle = c['divergence_angle_deg']

    if Lcon1 > Rc1 or Lcon2 > Rc2:
        raise ConfigError('Each converging arc length must not exceed its arc radius')

    Le = Lnozzle + Lcon1 + Lcon2 + Lcham

    Rthroat = Rexit - tan(pi*expansionangle/180)*Lnozzle
    Rcon1 = Rthroat + Rc1 - sqrt(Rc1**2 - Lcon1**2)
    Rcham = Rcon1 + Rc2 - sqrt(Rc2**2 - Lcon2**2)

    if Rthroat <= 0:
        raise ConfigError('Nozzle length and divergence angle leave a non-positive throat radius')

    def R(z):
        if z >= 0 and z < Lnozzle:
            return Rexit - tan(pi*expansionangle/180)*z
        elif z >= Lnozzle and z < Lnozzle + Lcon1:
            z = z - Lnozzle
            return Rthroat + Rc1 - np.sqrt(Rc1**2 - z**2)
        elif z >= Lnozzle + Lcon1 and z <= Lnozzle + Lcon1 + Lcon2:
            z = z - (Lnozzle + Lcon1)
            return Rcon1 + np.sqrt(Rc2**2 - (z - Lcon2)**2) - np.sqrt(Rc2**2 - Lcon2**2)
        elif z >= Lnozzle + Lcon1 + Lcon2 and z < Le + 1e-2:
            return Rcham
        else:
            raise ValueError('Z value given is outside nozzle range!')

    geom = np.array([Lnozzle, Lcon1, Lcon2, Lcham, Rexit, Rc1, Rc2, Rcham])

    return R, Le, geom


def derived_geometry(vehicle):
    '''
    Quantities implied by the contour, for display and sanity checks.
    '''

    c = vehicle['contour']
    R, Le, geom = make_contour(c)
    Rt = R(geom[0])
    Re = geom[4]
    Rch = geom[7]

    # slope mismatch between the two converging arcs (0 = tangent)
    s1 = geom[1]/sqrt(geom[5]**2 - geom[1]**2) if geom[5] > geom[1] else float('inf')
    s2 = geom[2]/sqrt(geom[6]**2 - geom[2]**2) if geom[6] > geom[2] else float('inf')

    return {
        'throat_diameter_mm': float(2000*Rt),
        'exit_diameter_mm': float(2000*Re),
        'chamber_diameter_mm': float(2000*Rch),
        'expansion_ratio': float((Re/Rt)**2),
        'contraction_ratio': float((Rch/Rt)**2),
        'length_mm': float(1000*Le),
        'arc_kink_deg': float(abs(degrees(atan(s1) - atan(s2)))),
    }


############ Object builders ############

def build(vehicle, sim):
    '''
    Constructs the solver objects and scalar parameters from the two configs.
    '''

    from RXPI_CATNAP_Combustion import Props_obj
    from RXPI_CATNAP_Fluids import Injector_obj, Valve_obj
    from RXPI_CATNAP_Regen import Regen_obj, tbc_flux_scale
    from RXPI_CATNAP_Film import Film_obj

    p = vehicle['propellants']
    props = Props_obj(p['oxidizer_cea'], p['fuel_cea'], p['oxidizer_coolprop'], p['fuel_coolprop'])

    inj = vehicle['injector']
    injector = Injector_obj(inj['fuel_cd'], inj['ox_cd'], inj['film_cd'],
                            int(inj['ox_holes']), int(inj['fuel_holes']), int(inj['film_holes']),
                            inj['ox_diameter_mm']/1000, inj['fuel_diameter_mm']/1000,
                            inj['film_diameter_mm']/1000, props)

    valves = Valve_obj(vehicle['valves']['ox_cv'], vehicle['valves']['fuel_cv'], props)

    R, Le, geom = make_contour(vehicle['contour'])
    Rthroat = R(geom[0])
    eps = (geom[4]/Rthroat)**2

    rg = vehicle['regen']
    regen = Regen_obj(rg['wall_thickness_mm']/1000, rg['channel_height_mm']/1000,
                      rg['min_channel_width_mm']/1000, R, Rthroat,
                      vehicle['contour']['throat_curvature_radius_mm']/1000,
                      int(rg['channels']), props.fuel, rg['wall_conductivity_W_mK'],
                      rg['roughness_um']*1e-6, int(sim['regen']['axial_points']), Le,
                      rg['generatrix_angle_deg'],
                      tbc_scale=tbc_flux_scale(vehicle['tbc']['pdms_mass_fraction_pct']))

    f = vehicle['film']
    film = Film_obj(enabled=bool(f['enabled']), wall_emissivity=rg['wall_emissivity'],
                    gas_emissivity=rg['gas_emissivity'], gas_radiation=bool(rg['gas_radiation']),
                    mixing_coefficient=f['mixing_coefficient'],
                    near_wall_mass_fraction=f['near_wall_mass_fraction'], near_wall_OF=f['near_wall_OF'],
                    oxname=p['oxidizer_cea'], fuelname=p['fuel_cea'])

    feed = build_feed(vehicle, sim, props, injector, regen)

    return dict(props=props, injector=injector, valves=valves, regen=regen, film=film, feed=feed,
                R=R, Le=Le, geom=geom, eps=eps)


def build_feed(vehicle, sim, props, injector, regen):
    '''
    The ox and fuel feed networks (RXPI_CATNAP_Feed), tank to chamber.
    '''

    import RXPI_CATNAP_Feed as Feed

    ln = vehicle['lines']
    vv = vehicle['valves']

    ox = Feed.FeedSystem('ox', Feed.TwoPhaseFluid(props.ox), [
        Feed.Valve('Ox main valve', vv['ox_cv']),
        Feed.Line('Ox line', ln['ox_dash'], ln['ox_wall_in'], ln['ox_length_in'],
                  ln['roughness_um'], ln['ox_minor_loss_K']),
        Feed.Orifice('Ox injector', injector.CdA_ox),
    ])

    # the film is fed from the regen outlet manifold, in parallel with the fuel orifices
    fuel = Feed.FeedSystem('fuel', Feed.LiquidFluid(props.fuel), [
        Feed.FixedDrop('Piston', sim['tank']['piston_dp_psi']*6894.75729),
        Feed.Valve('Fuel main valve', vv['fuel_cv']),
        Feed.Line('Fuel line', ln['fuel_dash'], ln['fuel_wall_in'], ln['fuel_length_in'],
                  ln['roughness_um'], ln['fuel_minor_loss_K']),
        Feed.RegenChannels('Regen channels', regen, sim['regen']['coolant_inlet_temperature_K']),
        Feed.Orifice('Fuel + film injector', injector.CdA_fuel + injector.CdA_film),
    ])

    return dict(ox=ox, fuel=fuel)
