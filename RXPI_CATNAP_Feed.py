# -*- coding: utf-8 -*-


import math

import CoolProp.CoolProp as CP

from RXPI_CATNAP_Fluids import CV_TO_CDA, abstract_state, flow_state, dyer_mdot
from RXPI_CATNAP_Config import an_tube_id_in


_in2m = 0.0254




def secant(f, x0, slope=None, lo=-math.inf, hi=math.inf, x_pos=None, x_neg=None,
           ftol=1e-9, xtol=0.0, rtol=0.0, maxiter=60):


    n = 0
    at_bound = set()

    def ev(x):
        nonlocal n, x_pos, x_neg
        fx = f(x)
        n += 1
        if fx > 0:
            x_pos = x
        elif fx < 0:
            x_neg = x
        return fx

    x = x0
    fx = ev(x)
    if abs(fx) <= ftol + rtol*abs(x):
        return x, slope, n, True

    if slope is None or not math.isfinite(slope) or slope == 0:
        dx = 1e-4*abs(x) if x != 0 else 1e-6
        x1 = x + dx if x + dx < hi else x - dx
        f1 = ev(x1)
        slope = (f1 - fx)/(x1 - x)
        if abs(f1) < abs(fx):
            x, fx = x1, f1
        if abs(fx) <= ftol + rtol*abs(x):
            return x, slope, n, True

    for _ in range(maxiter):
        bracketed = x_pos is not None and x_neg is not None
        a, b = (min(x_pos, x_neg), max(x_pos, x_neg)) if bracketed else (lo, hi)

        xn = x - fx/slope if (slope != 0 and math.isfinite(slope)) else math.nan
        if not (a < xn < b):
            if bracketed:
                xn = 0.5*(a + b)
            elif math.isnan(xn):
                return x, slope, n, False
            else:
                bound = b if xn >= b else a
                if bound in at_bound or not math.isfinite(bound):
                    return x, slope, n, False      
                at_bound.add(bound)
                xn = bound

        fn = ev(xn)
        if xn != x and fn != fx:
            s = (fn - fx)/(xn - x)
            if math.isfinite(s) and s != 0:
                slope = s
        x, fx = xn, fn

        if abs(fx) <= ftol + rtol*abs(x):
            return x, slope, n, True
        if x_pos is not None and x_neg is not None and abs(x_pos - x_neg) <= xtol + rtol*abs(x):
            return x, slope, n, True

    return x, slope, n, False




class TwoPhaseFluid:

    incompressible = False

    def __init__(self, fluid):
        self.name = fluid
        self.F = abstract_state(fluid)

    def state(self, P, h):
        return flow_state(self.F, max(P, self.F.p_floor), h)

    def downstream(self, st, P):
        return self.state(P, st.h)

    def restriction_mdot(self, st, P2, CdA, k_override=None):
        return dyer_mdot(self.F, st, P2, CdA, k_override)

    def restriction_dP(self, st, mdot, CdA):
        return None     # implicit, see Orifice.outlet

    def saturated_liquid(self, T):
        AS = self.F.AS
        AS.update(CP.QT_INPUTS, 0, T)
        return self.state(AS.p(), AS.hmass())

    def tank_vapor(self, T, rho):
        AS = self.F.AS
        AS.update(CP.DmassT_INPUTS, rho, T)
        return self.state(AS.p(), AS.hmass())


class LiquidState:
    __slots__ = ('P', 'h', 'T', 'rho', 'mu', 'x', 'region')

    def __init__(self, P, T, rho, mu):
        self.P, self.T, self.rho, self.mu = P, T, rho, mu
        self.h, self.x, self.region = None, 0.0, 'liquid'

    def __repr__(self):
        return f'LiquidState(P={self.P:.6g}, T={self.T:.5g}, rho={self.rho:.5g})'


class LiquidFluid:

    incompressible = True

    def __init__(self, fluid):
        self.name = fluid
        self.AS = CP.AbstractState('HEOS', fluid)
        self.T = self.P = None
        self._cache = {}

    def props(self, T):
        p = self._cache.get(T)
        if p is None:
            AS = self.AS
            AS.update(CP.PT_INPUTS, self.P, T)
            try:
                mu = AS.viscosity()
            except ValueError:
                mu = CP.PropsSI('viscosity', 'T', T, 'P', self.P, self.name)
            p = self._cache[T] = (AS.rhomass(), mu)
        return p

    def set(self, T, P):
        if (T, P) != (self.T, self.P):
            self.T, self.P = T, P
            self._cache = {}
        rho, mu = self.props(T)
        return LiquidState(P, T, rho, mu)

    def downstream(self, st, P):
        return LiquidState(P, st.T, st.rho, st.mu)

    def restriction_mdot(self, st, P2, CdA, k_override=None):
        return CdA*math.sqrt(2*st.rho*max(st.P - P2, 0.0))

    def restriction_dP(self, st, mdot, CdA):
        return (mdot/CdA)**2/(2*st.rho)


############ Components ############

class Component:


    kind = 'component'

    def __init__(self, name):
        self.name = name

    def outlet(self, fluid, st, mdot):
        raise NotImplementedError

    def describe(self):
        return {'name': self.name, 'kind': self.kind}


class FixedDrop(Component):
    '''
    Constant pressure drop (piston, regulator offset).
    '''

    kind = 'fixed'

    def __init__(self, name, dP):
        super().__init__(name)
        self.dP = dP

    def outlet(self, fluid, st, mdot):
        return fluid.downstream(st, st.P - self.dP)


class Orifice(Component):
    '''
    Restriction of effective area CdA: SPI for a liquid, generalized Dyer NHNE for a
    flashing propellant (compressible orifice for vapour).

    mdot() is the direct form, mdot(P_in, P_out). outlet() inverts it for a given flow:
    explicit for SPI, otherwise a secant in ln(dP) warm-started from the SPI estimate
    scaled by the last solve's correction.
    '''

    kind = 'orifice'

    def __init__(self, name, CdA, k_override=None):
        super().__init__(name)
        self.CdA = CdA
        self.k_override = k_override
        self._ratio = 1.0       # dP/dP_SPI at the last solve
        self._slope = 0.5       # d ln(mdot)/d ln(dP); exactly 1/2 for SPI
        self.choked = False

    def mdot(self, fluid, st, P_out):
        return fluid.restriction_mdot(st, P_out, self.CdA, self.k_override)

    def outlet(self, fluid, st, mdot):
        self.choked = False
        if mdot <= 0:
            return st

        dP = fluid.restriction_dP(st, mdot, self.CdA)
        if dP is not None:
            return fluid.downstream(st, st.P - dP)

        P_min = fluid.F.p_floor
        dP_max = st.P - P_min
        dP_spi = (mdot/self.CdA)**2/(2*st.rho)
        u_hi = math.log(dP_max)
        u0 = min(math.log(self._ratio*dP_spi), u_hi - 1e-6)

        def g(u):
            m = self.mdot(fluid, st, st.P - math.exp(u))
            return math.log(m/mdot) if m > 0 else -50.0

        u, slope, _, ok = secant(g, u0, slope=self._slope, hi=u_hi, ftol=1e-10, xtol=1e-12)

        if not ok:
            # The flow can't pass this restriction at any back pressure (choked or past the
            # HEM critical flow): hand the outer solve a starved state so it lowers mdot.
            self.choked = True
            return fluid.downstream(st, P_min)

        dP = math.exp(u)
        self._ratio = dP/dP_spi
        if 0.05 < slope < 5:
            self._slope = slope
        return fluid.downstream(st, st.P - dP)

    def describe(self):
        return {'name': self.name, 'kind': self.kind, 'CdA_mm2': 1e6*self.CdA}


class Valve(Orifice):
    '''
    Valve from its flow coefficient Cv (US gpm, psi), as an orifice of CdA = Cv*CV_TO_CDA.
    For a liquid this is exactly dP_psi = SG (Q_gpm/Cv)^2.
    '''

    kind = 'valve'

    def __init__(self, name, Cv, k_override=None):
        super().__init__(name, Cv*CV_TO_CDA, k_override)
        self.Cv = Cv

    def describe(self):
        d = super().describe()
        d['Cv'] = self.Cv
        return d


def churchill_f(Re, rel_rough):
    '''
    Darcy friction factor, Churchill (1977). One smooth expression across laminar,
    transitional and turbulent flow (Colebrook within ~1 % when turbulent), so the
    feed residual stays differentiable for the secant solver.
    '''

    Re = max(Re, 1e-9)
    A = (2.457*math.log(1/((7/Re)**0.9 + 0.27*rel_rough)))**16
    B = (37530/Re)**16
    return 8*((8/Re)**12 + (A + B)**-1.5)**(1/12)


class Line(Component):
    '''
    Straight AN/JIC tube, Darcy-Weisbach with Churchill's friction factor, plus a lumped
    minor-loss coefficient K for fittings and bends:

        dP = (f L/D + K) G^2 v_m/2 + G^2 (v_out - v_in)

    For an incompressible liquid v_m = v_in and the acceleration term is zero. For a
    flashing propellant the flow is homogeneous: v_m is the mean of the inlet and outlet
    specific volumes (outlet from an isenthalpic flash, one predictor-corrector pass), and
    the second term is the acceleration of the expanding mixture.
    '''

    kind = 'line'

    def __init__(self, name, dash, wall_in, length_in, roughness_um=1.5, K=0.0):
        super().__init__(name)
        self.dash = int(dash)
        self.wall_in = wall_in
        self.length_in = length_in
        self.roughness_um = roughness_um
        self.K = K
        self.D = an_tube_id_in(self.dash, wall_in)*_in2m
        self.L = length_in*_in2m
        self.A = 0.25*math.pi*self.D**2
        self.rel_rough = roughness_um*1e-6/self.D

    def resistance(self, G, mu):
        if self.L <= 0:
            return self.K
        return churchill_f(G*self.D/mu, self.rel_rough)*self.L/self.D + self.K

    def outlet(self, fluid, st, mdot):
        if mdot <= 0 or (self.L <= 0 and self.K <= 0):
            return st

        G = mdot/self.A
        Rf = self.resistance(G, st.mu)
        v1 = 1/st.rho
        dP = 0.5*Rf*G**2*v1

        if fluid.incompressible:
            return fluid.downstream(st, st.P - dP)

        out = fluid.downstream(st, st.P - dP)
        v2 = 1/out.rho
        dP = 0.25*Rf*G**2*(v1 + v2) + G**2*(v2 - v1)
        return fluid.downstream(st, st.P - dP)

    def describe(self):
        return {'name': self.name, 'kind': self.kind, 'dash': self.dash, 'wall_in': self.wall_in,
                'ID_mm': 1000*self.D, 'length_in': self.length_in, 'K': self.K}


class RegenChannels(Component):

    kind = 'regen'

    def __init__(self, name, regen, T_coolant):
        super().__init__(name)
        self.regen = regen
        self.T_coolant = T_coolant

    def outlet(self, fluid, st, mdot):
        rho, mu = fluid.props(self.T_coolant)
        dP = -self.regen.dP_channel_isothermal(mdot, rho, mu)
        return fluid.downstream(st, st.P - dP)

    def describe(self):
        return {'name': self.name, 'kind': self.kind, 'channels': self.regen.numchannels}




class FeedSolution:
    __slots__ = ('mdot', 'states', 'names', 'P_end', 'evals', 'converged', 'choked')

    def node_pressures(self):
        return [st.P for st in self.states] + [self.P_end]

    def node_qualities(self):
        return [st.x for st in self.states]

    def dP(self, name):
        i = self.names.index(name)
        return self.states[i].P - (self.states[i + 1].P if i + 1 < len(self.states) else self.P_end)


class FeedSystem:
    '''
    Series feed network. components[-1] is the injector (an Orifice) discharging into the
    chamber; see the module docstring for the solution method.
    '''

    def __init__(self, name, fluid, components, rtol=1e-7):
        if not isinstance(components[-1], Orifice):
            raise ValueError('The last feed component must be the injector orifice')
        self.name = name
        self.fluid = fluid
        self.upstream = list(components[:-1])
        self.injector = components[-1]
        self.components = list(components)
        self.rtol = rtol
        self._slope = None
        self.total_evals = 0
        self.solves = 0

    def node_names(self):
        '''
        Names of the states in a FeedSolution: the tank, then each upstream outlet
        (= the next component's inlet), then the chamber.
        '''

        return ['Tank'] + [f'{c.name} outlet' for c in self.upstream] + ['Chamber']

    def march(self, st0, mdot):
        states = [st0]
        for c in self.upstream:
            states.append(c.outlet(self.fluid, states[-1], mdot))
        return states

    def solve(self, st0, P_end, mdot_guess=None):
        '''
        Flow through the network from the tank state st0 into P_end.
        mdot_guess (the previous step's flow) and the stored slope warm-start the secant.
        '''

        fl = self.fluid
        inj = self.injector

        # Upper bound: the injector alone at tank pressure. Every upstream loss only reduces it.
        m_max = inj.mdot(fl, st0, P_end)

        sol = FeedSolution()
        sol.names = [c.name for c in self.components]
        sol.P_end = P_end

        if m_max <= 0:
            sol.mdot, sol.states, sol.evals, sol.converged, sol.choked = 0.0, self.march(st0, 0.0), 0, True, False
            return sol

        last = {}

        def r(m):
            states = self.march(st0, m)
            last['m'], last['states'] = m, states
            return inj.mdot(fl, states[-1], P_end) - m

        x0 = mdot_guess if (mdot_guess is not None and 0 < mdot_guess < m_max) else 0.9*m_max
        m, slope, n, ok = secant(r, x0, slope=self._slope, lo=0.0, hi=m_max, x_pos=0.0, x_neg=m_max,
                                 ftol=0.0, rtol=self.rtol)

        if math.isfinite(slope) and slope < 0:
            self._slope = slope

        if last.get('m') != m:
            r(m)
            n += 1

        self.total_evals += n
        self.solves += 1

        sol.mdot = m
        sol.states = last['states']
        sol.evals = n
        sol.converged = ok
        sol.choked = any(getattr(c, 'choked', False) for c in self.upstream)
        return sol

    def describe(self):
        return [c.describe() for c in self.components]
