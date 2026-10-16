# CATNAP
Coupled Adiabatic Tank and Nozzle Analysis Program - A multiphase flow and regenerative cooling solver for dual-acting vapor pressurized propulsion systems.

CATNAP couples an iterative control volume solver with multiphase flow functions, regenerative cooling functions, liquid film cooling, and NASA CEA combustion analysis to form a high fidelity estimate of propulsion system characteristics over the course of a burn. Generally, it is meant to be used
for Nitrous Oxide (N2O) and a generic fuel (Ethanol, IPA, Jet-A). 

Install Dependancies with:
```sh
pip install -r requirements.txt
```

Run the GUI:
```sh
python catnap_gui.py
```

Or run from the command line with a vehicle and sim file:
```sh
python RXPI_CATNAP.py --vehicle configs/altair_vehicle.yaml --sim configs/altair_sim.yaml
```

Inputs are split into two YAML files, which the GUI can import and export:
-Vehicle: propellants, chamber contour, injector, valves, feed lines, regen channels, film cooling
-Sim: burn time and steps, initial tank state, ambient pressure, regen/film snapshot times, solver settings

Throat and exit diameter, expansion ratio and chamber radius are derived from the contour.

Each feed system is a chain of components in series (`RXPI_CATNAP_Feed.py`):
-Oxidizer: tank -> main valve -> line -> injector
-Fuel: tank -> piston -> main valve -> line -> regen inlet (nozzle exit) -> channels -> injector + film

Every step, the tank state and the previous step's chamber pressure fix the ends of each chain and the mass
flow is solved with a warm-started, bracketed secant (about 3 evaluations per step). The oxidizer valve and
injector use the generalized Dyer model and the line a homogeneous two-phase Darcy-Weisbach loss. The enthalpy
is carried down the chain, so N2O that flashes in the valve or line enters the injector as a two-phase mixture.
The fuel side uses SPI and Darcy-Weisbach. Lines are AN/JIC tube (dash size, wall, length, fitting K). The GUI's
"Feed system schematic" window shows the network with the current inputs, and comes up for review before a
vehicle file is exported.

The regen loop and the film cooling model are only solved at the snapshot times. Film cooling uses the
approach of RPA's thermal module (Ponomarenko, 2012): the liquid film is marched through heating and
vaporization (the wall under it receives only radiation), then its vapour mixes into the near-wall gas layer.
The convective flux is scaled by Ievlev's similarity parameter for that layer, with the layer properties from a
chemical equilibrium calculation (Cantera). Hot-wall temperatures, which the Bartz sigma factor, radiation and
Ievlev's parameter depend on, and the film inlet temperature (the film is fed from the regen manifold) are
iterated to convergence at each snapshot.

Setting `regen.thermal_model: false` skips the regen heat transfer and film model; the channel pressure drop is
then the isothermal Darcy-Weisbach estimate at the coolant inlet temperature.

Outputs include the nozzle exit separation margin, 100*(P_exit/(0.4*P_amb) - 1), over the burn (negative means
separation is predicted). With `output.csv: true` every plotted quantity is written to a single comma-separated
.txt file in Downloads; load it with `pandas.read_csv(path, comment='#')`.

The GUI uses IBM Plex Sans / Mono if installed, or if the .ttf files are placed in a `fonts` folder.

CATNAP is unfinished! There may be bugs I have not caught.
