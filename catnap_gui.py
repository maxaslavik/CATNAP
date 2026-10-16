# -*- coding: utf-8 -*-
"""
CATNAP GUI.

Edits the vehicle and sim configs (see RXPI_CATNAP_Config), imports/exports them as
YAML, and runs the same solver as the command line (RXPI_CATNAP.run).

Look: light, Windows XP / Office 2003 era. Tahoma 9pt, beige window face, white input
fields, blue group headers.
"""

import copy
import glob
import os
import sys
import threading
import time
import traceback
import webbrowser

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal

import numpy as np
import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import RXPI_CATNAP_Config as Config


HERE = os.path.dirname(os.path.abspath(__file__))

BG     = '#ECE9D8'   # window face
PANE   = '#FCFCFA'   # tab pages, forms, plot backgrounds
FIELD  = '#FFFFFF'
BORDER = '#7F9DB9'   # input borders
LINE   = '#ACA899'   # dividers, panel edges
NAVY   = '#15428B'
BLUE   = '#316AC5'   # selection
INK    = '#1F4E9C'   # secondary plot series
RED    = '#B22222'
TEXT   = '#000000'
MUTED  = '#5F5F5F'
GRID   = '#DADAD2'
OXC    = '#C8650A'   # oxidizer feed
FUELC  = '#2E7D4F'   # fuel feed
SCHEM_LABEL = '#4A6A94'   # feed schematic labels

_psi2MPa = 6894.75729/1e6


############ Input definitions ############
# (key, label, unit, type). Types: float, int, str, bool, optfloat, floatlist, choice:<a|b>,
# dash (AN tube size dropdown)

VEHICLE_FIELDS = [
    (None, 'Vehicle', [
        ('name', 'Name', '', 'str'),
    ]),
    ('propellants', 'Propellants', [
        ('oxidizer_cea',      'Oxidizer  (CEA)',      '', 'str'),
        ('fuel_cea',          'Fuel  (CEA)',          '', 'str'),
        ('oxidizer_coolprop', 'Oxidizer  (CoolProp)', '', 'str'),
        ('fuel_coolprop',     'Fuel  (CoolProp)',     '', 'str'),
    ]),
    ('contour', 'Chamber contour', [
        ('nozzle_length_in',           'Nozzle length',          'in',  'float'),
        ('arc1_length_in',             'Throat arc length',      'in',  'float'),
        ('arc1_radius_in',             'Throat arc radius',      'in',  'float'),
        ('arc2_length_in',             'Chamber arc length',     'in',  'float'),
        ('arc2_radius_in',             'Chamber arc radius',     'in',  'float'),
        ('chamber_length_in',          'Chamber length',         'in',  'float'),
        ('exit_radius_in',             'Exit radius',            'in',  'float'),
        ('divergence_angle_deg',       'Divergence half-angle',  'deg', 'float'),
        ('throat_curvature_radius_mm', 'Bartz throat r_c',       'mm',  'float'),
    ]),
    ('injector', 'Injector', [
        ('ox_holes',          'Oxidizer holes',     '',   'int'),
        ('ox_diameter_mm',    'Oxidizer hole Ø',    'mm', 'float'),
        ('ox_cd',             'Oxidizer Cd',        '',   'float'),
        ('fuel_holes',        'Fuel holes',         '',   'int'),
        ('fuel_diameter_mm',  'Fuel hole Ø',        'mm', 'float'),
        ('fuel_cd',           'Fuel Cd',            '',   'float'),
        ('film_holes',        'Film holes',         '',   'int'),
        ('film_diameter_mm',  'Film hole Ø',        'mm', 'float'),
        ('film_cd',           'Film Cd',            '',   'float'),
    ]),
    ('valves', 'Valves', [
        ('ox_cv',   'Oxidizer Cv', '', 'float'),
        ('fuel_cv', 'Fuel Cv',     '', 'float'),
    ]),
    ('lines', 'Feed lines  ·  Darcy-Weisbach', [
        ('ox_dash',           'Oxidizer line size',     '',   'dash'),
        ('ox_wall_in',        'Oxidizer wall',          'in', 'float'),
        ('ox_length_in',      'Oxidizer line length',   'in', 'float'),
        ('ox_minor_loss_K',   'Oxidizer fittings ΣK',   '',   'float'),
        ('fuel_dash',         'Fuel line size',         '',   'dash'),
        ('fuel_wall_in',      'Fuel wall',              'in', 'float'),
        ('fuel_length_in',    'Fuel line length',       'in', 'float'),
        ('fuel_minor_loss_K', 'Fuel fittings ΣK',       '',   'float'),
        ('roughness_um',      'Tube roughness',         'µm', 'float'),
    ]),
    ('regen', 'Regen channels', [
        ('wall_thickness_mm',      'Hot-wall thickness',   'mm',    'float'),
        ('channel_height_mm',      'Channel height',       'mm',    'float'),
        ('min_channel_width_mm',   'Throat channel width', 'mm',    'float'),
        ('channels',               'Channels',             '',      'int'),
        ('wall_conductivity_W_mK', 'Wall conductivity',    'W/m·K', 'float'),
        ('roughness_um',           'Surface roughness',    'µm',    'float'),
        ('generatrix_angle_deg',   'Generatrix angle',     'deg',   'float'),
        ('gas_radiation',          'Gas radiation',        '',      'bool'),
        ('wall_emissivity',        'Hot-wall emissivity ε_w', '',   'float'),
        ('gas_emissivity',         'Gas emissivity',       '',      'optfloat'),
    ]),
    ('film', 'Film cooling  ·  RPA / Ievlev', [
        ('enabled',              'Enable film model',     '',  'bool'),
        ('mixing_coefficient',      'Mixing intensity K_t',     '', 'float'),
        ('near_wall_mass_fraction', 'Near-wall layer mass fraction', '', 'float'),
        ('near_wall_OF',            'Near-wall layer O/F',      '', 'optfloat'),
    ]),
    ('tbc', 'Thermal barrier  ·  PDMS', [
        ('pdms_mass_fraction_pct', 'PDMS fuel mass fraction', '%', 'float'),
    ]),
]

SIM_FIELDS = [
    ('time', 'Burn', [
        ('duration_s', 'Duration',   's', 'float'),
        ('steps',      'Time steps', '',  'int'),
    ]),
    ('tank', 'Tank', [
        ('propellant_mass_kg', 'Pressurant mass',    'kg',  'float'),
        ('temperature_K',      'Initial temperature', 'K',   'float'),
        ('vapor_quality',      'Initial vapor quality', '', 'float'),
        ('piston_dp_psi',      'Piston ΔP',           'psi', 'float'),
    ]),
    ('environment', 'Environment', [
        ('ambient_pressure_bar', 'Ambient pressure', 'bar', 'float'),
    ]),
    ('regen', 'Regen + film snapshots', [
        ('thermal_model',               'Thermal model (regen + film)', '', 'bool'),
        ('snapshot_times_s',            'Snapshot times',      's',  'floatlist'),
        ('coolant_inlet_temperature_K', 'Coolant inlet T',     'K',  'float'),
        ('axial_points',                'Axial points',        '',   'int'),
    ]),
    ('solver', 'Solver', [
        ('chamber_pressure_guess_psi', 'Pc initial guess',       'psi', 'float'),
        ('film_coupling_tol_K',        'Film coupling tolerance', 'K',  'float'),
        ('film_coupling_max_iter',     'Film coupling max iter', '',    'int'),
    ]),
    ('output', 'Output', [
        ('dashboard', 'Open dashboard when done', '', 'bool'),
        ('csv',       'Export CSV to Downloads', '', 'bool'),
    ]),
]

TOOLTIPS = {
    'film.mixing_coefficient': 'RPA turbulent mixing intensity, (0.05...0.20)e-2. Higher = film vapour\n'
                               'dilutes faster into the near-wall layer = less protection downstream.',
    'film.near_wall_mass_fraction': 'Fraction of the total flow in the near-wall layer the film vapour mixes into.\n'
                                    'Sets the layer thickness (annulus at uniform mass flux).',
    'film.near_wall_OF': 'Mixture ratio of the near-wall layer. Blank = core mixture ratio.',
    'tbc.pdms_mass_fraction_pct': 'PDMS as % of fuel mass. Every node\'s wall heat flux is multiplied by\n'
                                  '1 - 0.08 x this value (8 % reduction per 1 %). Allowed range 0-10 %.',
    'regen.wall_emissivity': 'Absorbs gas radiation everywhere; under the liquid film it is the only heat\n'
                            'reaching the wall. RPA table 1: oxidized copper 0.57-0.87, oxidized steel 0.8.',
    'regen.gas_emissivity': 'Leave blank to use the Leckner H2O/CO2 correlation.',
    'regen.thermal_model': 'Off: no regen heat transfer or film model. The channel pressure drop is the\n'
                           'isothermal Darcy-Weisbach estimate at the coolant inlet temperature.',
    'environment.ambient_pressure_bar': 'Used for thrust and for the separation criterion P_exit < 0.4 P_amb.',
    'output.csv': 'Writes every plotted quantity to one comma-separated .txt file in Downloads.',
    'regen.snapshot_times_s': 'Comma-separated. The regen loop and film model are only solved here.',
    'contour.throat_curvature_radius_mm': 'Radius of curvature r_c used in the Bartz correlation.',
    'valves.ox_cv': 'Flow coefficient (US gpm of water at 1 psi), modelled as an orifice of CdA = Cv/38 in².\n'
                    'The oxidizer valve uses the generalized Dyer model, so the vapour it flashes is\n'
                    'carried into the line and injector.',
    'valves.fuel_cv': 'Flow coefficient (US gpm of water at 1 psi): dP_psi = SG (Q_gpm/Cv)².',
    'lines.ox_length_in': 'Tube length from the oxidizer main valve to the injector.',
    'lines.fuel_length_in': 'Tube length from the fuel main valve to the regen inlet at the nozzle exit.',
    'lines.ox_minor_loss_K': 'Sum of fitting and bend loss coefficients, dP = ΣK ρv²/2.',
    'lines.fuel_minor_loss_K': 'Sum of fitting and bend loss coefficients, dP = ΣK ρv²/2.',
    'lines.roughness_um': 'Absolute wall roughness. Drawn stainless tube ~1.5 µm.',
}

for _side in ('ox', 'fuel'):
    TOOLTIPS[f'lines.{_side}_dash'] = 'AN / JIC tube size: dash number = outer diameter in 1/16 in.'
    TOOLTIPS[f'lines.{_side}_wall_in'] = 'Tube wall thickness. Common stainless walls:\n' + '\n'.join(
        f'  -{d}: ' + ', '.join(f'{w:.3f}' for w in ws) + ' in' for d, ws in Config.AN_COMMON_WALLS_IN.items())

_OD_TEXT = {3: '3/16', 4: '1/4', 5: '5/16', 6: '3/8', 8: '1/2', 10: '5/8', 12: '3/4', 16: '1'}


def dash_text(d):
    return f'-{d}  ({_OD_TEXT.get(d, f"{d}/16")}" OD)'


def _get(cfg, sec, key):
    return cfg[key] if sec is None else cfg[sec][key]


def _set(cfg, sec, key, v):
    if sec is None:
        cfg[key] = v
    else:
        cfg[sec][key] = v


def _fmt(v):
    if isinstance(v, float):
        return format(v, '.10g')
    return str(v)


############ Fonts and theme ############

def load_fonts():
    try:
        families = set(QtGui.QFontDatabase.families())
    except TypeError:
        families = set(QtGui.QFontDatabase().families())

    sans = next((f for f in ('Tahoma', 'Segoe UI', 'Arial') if f in families), 'Sans Serif')
    mono = next((f for f in ('Lucida Console', 'Consolas', 'Courier New') if f in families), 'Monospace')

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [sans, 'DejaVu Sans']

    return sans, mono


def palette():
    p = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    for role, col in ((R.Window, BG), (R.WindowText, TEXT), (R.Base, FIELD), (R.AlternateBase, '#F3F3EC'),
                      (R.Text, TEXT), (R.Button, BG), (R.ButtonText, TEXT),
                      (R.Highlight, BLUE), (R.HighlightedText, '#FFFFFF'),
                      (R.ToolTipBase, '#FFFFE1'), (R.ToolTipText, TEXT)):
        p.setColor(role, QtGui.QColor(col))
    return p


def stylesheet(sans, mono):
    return f"""
    * {{ font-family: '{sans}'; font-size: 9pt; color: {TEXT}; }}
    QMainWindow, QWidget#root {{ background: {BG}; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: {PANE}; }}

    QFrame#banner {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0A3C9E, stop:1 #3F83E0);
                     border: 1px solid #0A2C78; border-radius: 3px; }}
    QFrame#banner QLabel {{ background: transparent; color: #FFFFFF; }}
    QLabel#brand {{ font-size: 15pt; font-weight: bold; }}
    QLabel#tagline {{ color: #D6E4FA; font-size: 8pt; }}

    QLabel#section {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #C9D8F0, stop:1 #EEF3FB);
                      color: {NAVY}; font-weight: bold; padding: 3px 6px; margin-top: 10px;
                      border: 1px solid #A9BFE3; border-radius: 2px; }}
    QLabel#unit {{ color: {MUTED}; }}
    QLabel#derivedVal {{ font-family: '{mono}'; font-size: 9pt; color: {NAVY}; }}
    QLabel#error {{ color: {RED}; font-weight: bold; }}

    QLineEdit, QComboBox {{ background: {FIELD}; border: 1px solid {BORDER}; padding: 2px 4px;
                            font-family: '{mono}'; selection-background-color: {BLUE};
                            selection-color: #FFFFFF; min-height: 16px; }}
    QLineEdit:focus, QComboBox:focus {{ border: 1px solid {BLUE}; }}
    QLineEdit:disabled, QComboBox:disabled {{ background: #F4F3EE; border-color: #C9C7BA; color: #9A9A92; }}
    QLineEdit[invalid="true"] {{ border: 1px solid {RED}; background: #FFE9E4; }}
    QComboBox QAbstractItemView {{ background: {FIELD}; border: 1px solid {BORDER};
                                   selection-background-color: {BLUE}; selection-color: #FFFFFF; }}

    QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:0.85 #ECEBE5, stop:1 #D6D0C5);
                   border: 1px solid #003C74; border-radius: 3px; padding: 3px 12px; min-height: 18px; }}
    QPushButton:hover {{ border: 1px solid #E5A126; }}
    QPushButton:pressed {{ background: #D6D0C5; }}
    QPushButton:disabled {{ background: #F4F3EE; border: 1px solid #C9C7BA; color: #9A9A92; }}
    QPushButton#run {{ font-weight: bold; padding: 6px 12px; border: 2px solid #003C74;
                       background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:0.85 #DCE8F8, stop:1 #B9CFEE); }}
    QPushButton#run:hover {{ border: 2px solid #E5A126; }}
    QPushButton#run:disabled {{ background: #F4F3EE; border: 1px solid #C9C7BA; }}
    QPushButton#stop {{ padding: 6px 16px; }}

    QTabWidget::pane {{ background: {PANE}; border: 1px solid {LINE}; top: -1px; }}
    QTabBar::tab {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6F5EE, stop:1 #E2DFCF);
                    border: 1px solid {LINE}; border-bottom: none; border-top-left-radius: 3px;
                    border-top-right-radius: 3px; padding: 4px 14px; margin-right: 1px; margin-top: 2px; }}
    QTabBar::tab:selected {{ background: {PANE}; border-top: 3px solid #F4A92B; margin-top: 0; padding-bottom: 6px; }}
    QTabBar::tab:hover:!selected {{ background: #FFF4D6; }}

    QProgressBar {{ background: {FIELD}; border: 1px solid #686868; border-radius: 2px; }}
    QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B7EC96, stop:0.5 #3DB93D, stop:1 #6FD36F);
                           margin: 1px; }}

    QStatusBar {{ background: {BG}; border-top: 1px solid {LINE}; }}
    QStatusBar::item {{ border: none; }}
    QStatusBar QLabel {{ font-size: 9pt; }}
    QLabel#phase {{ border: 1px solid {LINE}; border-right-color: #FFFFFF; border-bottom-color: #FFFFFF;
                    font-weight: bold; padding: 1px 8px; }}

    QTableWidget {{ background: {FIELD}; alternate-background-color: #F3F3EC; gridline-color: {GRID};
                    border: none; font-family: '{mono}'; selection-background-color: {BLUE};
                    selection-color: #FFFFFF; }}
    QHeaderView::section {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:1 #E2DFCF);
                            border: none; border-right: 1px solid {LINE}; border-bottom: 1px solid {LINE};
                            padding: 4px; }}

    QScrollBar:vertical {{ background: #F4F3EE; width: 16px; border-left: 1px solid #E2DFCF; margin: 16px 0 16px 0; }}
    QScrollBar::handle:vertical {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #DDE8FA, stop:1 #B4C9EC);
                                   border: 1px solid #7F9DB9; border-radius: 2px; min-height: 28px; margin: 0 1px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 16px; background: #DDE8FA;
                                   border: 1px solid #7F9DB9; border-radius: 2px; subcontrol-origin: margin; }}
    QScrollBar::add-line:vertical {{ subcontrol-position: bottom; }}
    QScrollBar::sub-line:vertical {{ subcontrol-position: top; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

    QToolTip {{ background: #FFFFE1; color: {TEXT}; border: 1px solid #000000; padding: 2px; }}
    QSplitter::handle {{ background: {BG}; }}
    QMessageBox, QDialog {{ background: {BG}; }}
    """


def style_axes(ax, ylabel, xlabel=None):
    ax.set_facecolor(FIELD)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    for sp in ax.spines.values():
        sp.set_color(LINE)
    ax.grid(color=GRID, linewidth=0.6)


############ Config form ############

class ConfigForm(QtWidgets.QScrollArea):
    '''
    Scrollable form for one config dict. Keys not listed in the field spec are kept
    untouched, so YAML files round-trip through the GUI.
    '''

    changed = Signal()

    def __init__(self, spec, cfg, extra=None):
        super().__init__()
        self.spec = spec
        self.cfg = copy.deepcopy(cfg)
        self.editors = {}

        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        body = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(body)
        self.grid.setContentsMargins(8, 0, 12, 10)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(5)
        self.grid.setColumnStretch(0, 1)
        self.row = 0

        for sec, title, fields in spec:
            self._section(title)
            for key, label, unit, typ in fields:
                self._field(sec, key, label, unit, typ)
            if extra and sec in extra:
                extra[sec](self)

        self.grid.setRowStretch(self.row, 1)
        self.setWidget(body)
        self.set_config(cfg)

    def _section(self, title):
        lbl = QtWidgets.QLabel(title)
        lbl.setObjectName('section')
        self.grid.addWidget(lbl, self.row, 0, 1, 3)
        self.row += 1

    def add_widget_row(self, label, widget):
        k = QtWidgets.QLabel(label)
        k.setObjectName('derivedKey')
        self.grid.addWidget(k, self.row, 0)
        self.grid.addWidget(widget, self.row, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        self.row += 1

    def _field(self, sec, key, label, unit, typ):
        if typ == 'bool':
            ed = QtWidgets.QCheckBox()
            ed.toggled.connect(self.changed.emit)
        elif typ.startswith('choice:'):
            ed = QtWidgets.QComboBox()
            ed.addItems(typ.split(':', 1)[1].split('|'))
            ed.currentIndexChanged.connect(self.changed.emit)
        elif typ == 'dash':
            ed = QtWidgets.QComboBox()
            for d in Config.AN_DASH_SIZES:
                ed.addItem(dash_text(d), d)
            ed.currentIndexChanged.connect(self.changed.emit)
        else:
            ed = QtWidgets.QLineEdit()
            if typ != 'str':
                ed.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            if typ == 'optfloat':
                ed.setPlaceholderText('auto (Leckner)')
            ed.textEdited.connect(self.changed.emit)

        if typ != 'bool':
            ed.setFixedWidth(190 if typ == 'floatlist' else 130 if typ == 'dash' else 110)

        tip = TOOLTIPS.get(f'{sec}.{key}')
        lbl = QtWidgets.QLabel(label)
        if tip:
            ed.setToolTip(tip)
            lbl.setToolTip(tip)

        u = QtWidgets.QLabel(unit)
        u.setObjectName('unit')
        u.setMinimumWidth(38)

        self.grid.addWidget(lbl, self.row, 0)
        self.grid.addWidget(ed, self.row, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        self.grid.addWidget(u, self.row, 2)
        self.row += 1

        self.editors[(sec, key)] = (ed, typ)

    def set_config(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        for (sec, key), (ed, typ) in self.editors.items():
            v = _get(cfg, sec, key)
            ed.blockSignals(True)
            if typ == 'bool':
                ed.setChecked(bool(v))
            elif typ.startswith('choice:'):
                ed.setCurrentText(str(v))
            elif typ == 'dash':
                ed.setCurrentIndex(max(ed.findData(int(v)), 0))
            elif typ == 'optfloat':
                ed.setText('' if v is None else _fmt(v))
            elif typ == 'floatlist':
                ed.setText(', '.join(_fmt(x) for x in v))
            else:
                ed.setText(_fmt(v))
            self._mark(ed, False)
            ed.blockSignals(False)

    @staticmethod
    def _mark(ed, bad):
        if isinstance(ed, QtWidgets.QLineEdit):
            ed.setProperty('invalid', 'true' if bad else 'false')
            ed.style().unpolish(ed)
            ed.style().polish(ed)

    def collect(self):
        '''
        Returns (cfg, errors). cfg is a deep copy with the edited values applied.
        '''

        cfg = copy.deepcopy(self.cfg)
        errors = []

        for (sec, key), (ed, typ) in self.editors.items():
            bad = False
            try:
                if typ == 'bool':
                    v = ed.isChecked()
                elif typ.startswith('choice:'):
                    v = ed.currentText()
                elif typ == 'dash':
                    v = ed.currentData()
                elif typ == 'str':
                    v = ed.text().strip()
                    if not v:
                        raise ValueError
                elif typ == 'int':
                    v = int(ed.text())
                elif typ == 'float':
                    v = float(ed.text())
                elif typ == 'optfloat':
                    v = float(ed.text()) if ed.text().strip() else None
                elif typ == 'floatlist':
                    v = [float(s) for s in ed.text().replace(';', ',').split(',') if s.strip()]
                _set(cfg, sec, key, v)
            except ValueError:
                bad = True
                errors.append(f'{sec or "vehicle"}.{key}: {ed.text()!r} is not a valid {typ}')
            self._mark(ed, bad)

        return cfg, errors


############ Plots ############

class LivePlots(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(facecolor=PANE)
        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.90, top=0.96, bottom=0.07, hspace=0.45)

        ax1 = self.fig.add_subplot(411)
        ax2 = self.fig.add_subplot(412, sharex=ax1)
        ax3 = self.fig.add_subplot(413, sharex=ax1)
        ax4 = self.fig.add_subplot(414, sharex=ax1)
        self.axes = [ax1, ax2, ax3, ax4]
        self.twins = [ax.twinx() for ax in self.axes]

        style_axes(ax1, 'Pressure  (MPa)')
        style_axes(ax2, 'Thrust  (kN)')
        style_axes(ax3, 'Vapor quality')
        style_axes(ax4, 'Separation margin  (%)', 'Time  (s)')
        ax4.axhline(0, color=MUTED, lw=0.8, ls=':')
        for tw, lab in zip(self.twins, ('Tank T  (K)', 'Isp  (s)', 'ṁ total  (kg/s)', 'Channel ΔP  (bar)')):
            style_axes(tw, lab)
            tw.grid(False)

        self.lines = {
            'Pc':   ax1.plot([], [], color=RED,   lw=1.8, label='Chamber')[0],
            'P2':   ax1.plot([], [], color=INK, lw=1.8, label='Tank')[0],
            'Pox':  ax1.plot([], [], color=OXC, lw=1.1, label='Ox manifold')[0],
            'Pf':   ax1.plot([], [], color=FUELC, lw=1.1, label='Fuel manifold')[0],
            'T2':   self.twins[0].plot([], [], color=MUTED, lw=1.3, ls='--', label='Tank T')[0],
            'F':    ax2.plot([], [], color=RED,   lw=1.8, label='Thrust')[0],
            'Isp':  self.twins[1].plot([], [], color=INK, lw=1.3, ls='--', label='Isp')[0],
            'x2':   ax3.plot([], [], color=INK, lw=1.8, label='Quality')[0],
            'mdot': self.twins[2].plot([], [], color=RED, lw=1.3, ls='--', label='ṁ')[0],
            'sep':  ax4.plot([], [], color=RED, lw=1.8, label='Separation margin')[0],
            'dPch': self.twins[3].plot([], [], color=INK, lw=1.3, ls='--', label='Channel ΔP')[0],
        }
        ax3.set_ylim(0, 1.05)

        for ax, tw in zip(self.axes, self.twins):
            h = [l for l in ax.get_lines() + tw.get_lines() if not l.get_label().startswith('_')]
            leg = ax.legend(h, [l.get_label() for l in h], loc='upper right', fontsize=8,
                            frameon=True, facecolor=FIELD, edgecolor=LINE, framealpha=0.9, labelcolor=TEXT, ncol=3)

        self.snap_marks = []
        self.clear()

    def clear(self):
        self.data = {k: [] for k in ('t', 'Pc', 'P2', 'Pox', 'Pf', 'T2', 'F', 'Isp', 'x2', 'mdot', 'sep', 'dPch')}
        for m in self.snap_marks:
            m.remove()
        self.snap_marks = []
        self.redraw()

    def add(self, s):
        d = self.data
        d['t'].append(s['t'])
        d['Pc'].append(s['Pc']/1e6)
        d['P2'].append(s['P2']/1e6)
        d['Pox'].append(s['P_ox_inj']/1e6)
        d['Pf'].append(s['P_fuel_inj']/1e6)
        d['T2'].append(s['T2'])
        d['F'].append(s['F']/1e3)
        d['Isp'].append(s['Isp'])
        d['x2'].append(s['x2'])
        d['mdot'].append(s['mdot'])
        d['sep'].append(s['sep_margin'])
        d['dPch'].append(s['dP_channel']/1e5)
        if s.get('snapshot') is not None:
            for ax in self.axes:
                self.snap_marks.append(ax.axvline(s['t'], color=RED, lw=0.7, alpha=0.35))

    def redraw(self):
        t = self.data['t']
        for k, ln in self.lines.items():
            ln.set_data(t, self.data[k])
        for ax in [a for k, a in enumerate(self.axes) if k != 2] + self.twins:
            ax.relim()
            ax.autoscale_view()
        self.axes[2].relim()
        self.axes[2].autoscale_view(scaley=False)
        self.draw_idle()


class ContourPlot(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(facecolor=PANE)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)

    def show_contour(self, vehicle, film=None):
        ax = self.ax
        ax.clear()
        style_axes(ax, 'Radius  (mm)', 'Axial position from exit  (mm)')

        try:
            R, Le, geom = Config.make_contour(vehicle['contour'])
        except Exception as e:
            ax.text(0.5, 0.5, f'Invalid contour\n{e}', color=RED, ha='center', va='center',
                    transform=ax.transAxes)
            self.draw_idle()
            return

        z = np.linspace(0, Le, 600)
        r = np.array([R(zi) for zi in z])
        zm, rm = 1000*z, 1000*r

        ax.fill_between(zm, -rm, rm, color='#E3EAF6', lw=0)
        ax.plot(zm, rm, color=NAVY, lw=1.6)
        ax.plot(zm, -rm, color=NAVY, lw=1.6)
        ax.axhline(0, color=LINE, lw=0.8, ls='-.')

        zt = 1000*geom[0]
        ax.axvline(zt, color=MUTED, lw=0.8, ls=':')
        ax.text(zt, 1000*R(geom[0])*0.15, ' throat', color=MUTED, fontsize=8)

        title = f"{vehicle['name']}  contour"
        if film and film.get('active'):
            zz = z >= film['z_dry']
            for sgn in (1, -1):
                ax.plot(zm[zz], sgn*rm[zz], color=RED, lw=4, solid_capstyle='butt')
            title += f"   ·   liquid film  L_c = {100*film['Lc']:.2f} cm  at  t = {film['t']:.2f} s"
            ax.plot([], [], color=RED, lw=4, label='Liquid film (wetted wall)')
            ax.legend(loc='lower left', frameon=True, facecolor=FIELD, edgecolor=LINE, labelcolor=TEXT, fontsize=8.5)

        ax.set_title(title, color=TEXT, fontsize=10, loc='left')
        ax.set_aspect('equal', adjustable='datalim')
        self.draw_idle()


############ Feed system schematic ############

def _mm2(A):
    return f'{1e6*A:.2f} mm²'


class FeedSchematic(FigureCanvasQTAgg):
    '''
    Tank-to-chamber schematic of both feed networks, annotated with the current inputs.
    Ox: tank -> main valve -> line -> injector. Fuel: tank -> piston -> main valve -> line ->
    regen inlet at the nozzle exit -> channels -> injector manifold (fuel + film orifices).
    '''

    def __init__(self):
        self.fig = Figure(facecolor=PANE)
        super().__init__(self.fig)
        self.ax = self.fig.add_axes([0.005, 0.005, 0.99, 0.99])
        self.setMinimumSize(1000, 600)

    def show_feed(self, veh, sim, errors=()):
        from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle

        ax = self.ax
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        ax.set_aspect('equal')
        ax.set_anchor('N')

        def txt(x, y, s, color=TEXT, size=8.5, **kw):
            return ax.text(x, y, s, color=color, fontsize=size, **kw)

        title = txt(1, 58, f"{veh.get('name', 'Vehicle')}  ·  feed system", NAVY, 12, weight='bold', va='top')

        try:
            if errors:
                raise Config.ConfigError(errors[0])
            Config.validate(veh, Config.VEHICLE_KIND)
            R, Le, geom = Config.make_contour(veh['contour'])
        except Exception as e:
            txt(50, 30, f'Invalid input\n{e}', RED, 10, ha='center', va='center')
            self.draw_idle()
            return

        inj, vv, ln, rg = veh['injector'], veh['valves'], veh['lines'], veh['regen']
        piston = sim['tank']['piston_dp_psi'] if sim else None

        # engine, injector face at x_inj, nozzle exit on the right
        z = np.linspace(0, Le, 300)
        r = np.array([R(zi) for zi in z])
        x_inj = 58.0
        sc = min(32/Le, 8.5/r.max())
        xe = x_inj + (Le - z)*sc
        re = r*sc
        yc = 25.0
        Rch, Rex = re[-1], re[0]
        x_exit = xe[0]

        INKS, FILL, LBL = '#000000', '#E4E4E4', SCHEM_LABEL

        def label(x, y, head, body=None, ha='center', va='bottom', size=8):
            # muted-blue label with its values in black underneath
            if body is None:
                return txt(x, y, head, LBL, size, ha=ha, va=va, weight='bold')
            n = body.count('\n') + 1
            dy = 1.25*n if va == 'bottom' else 0
            txt(x, y + dy, head, LBL, size, ha=ha, va='bottom' if va == 'bottom' else 'top', weight='bold')
            txt(x, y + dy - (0.15 if va == 'bottom' else 1.25), body, TEXT, size, ha=ha, va='top')

        ax.fill_between(xe, yc - re, yc + re, color=FILL, lw=0, zorder=1)
        for sgn in (1, -1):
            ax.plot(xe, yc + sgn*re, color=INKS, lw=1.6, zorder=3)
            ax.plot(xe, yc + sgn*(re + 0.9), color=INKS, lw=0.8, ls='--', zorder=2)   # regen jacket
        zt = geom[0]
        xt = x_inj + (Le - zt)*sc
        txt((x_inj + xt)/2, yc + 0.6, 'Pc', LBL, 11, ha='center', va='bottom', style='italic')
        d = Config.derived_geometry(veh)
        txt((x_inj + x_exit)/2, yc - 0.4, f"throat Ø {d['throat_diameter_mm']:.2f} mm   ε = {d['expansion_ratio']:.2f}",
            TEXT, 7.5, ha='center', va='top')

        # injector manifold block
        ax.add_patch(Rectangle((x_inj - 2.4, yc - Rch), 2.4, 2*Rch, fc=FIELD, ec=INKS, lw=1.2, zorder=4))
        label(x_inj - 1.2, yc - Rch - 0.6, 'Injector', va='top', size=7.5)

        # tank
        y_f = yc + max(Rch, Rex) + 9.5
        tank_top, tank_bot = y_f + 4.5, yc - 12
        ax.add_patch(FancyBboxPatch((2, tank_bot), 19, tank_top - tank_bot, boxstyle='round,pad=0,rounding_size=2.5',
                                    fc=FIELD, ec=INKS, lw=1.6))
        txt(11.5, (tank_top + tank_bot)/2 + 3, 'P_tank', LBL, 12, ha='center', style='italic')
        lines = [f"{veh['propellants']['oxidizer_coolprop']} / {veh['propellants']['fuel_coolprop']}"]
        if sim:
            lines += [f"T₀ = {sim['tank']['temperature_K']:g} K", f"{sim['tank']['propellant_mass_kg']:g} kg ox"]
        txt(11.5, (tank_top + tank_bot)/2 + 1, '\n'.join(lines), TEXT, 8, ha='center', va='top')

        def valve(x, y):
            w, h = 1.4, 1.2
            for sgn in (-1, 1):
                ax.add_patch(Polygon([(x + sgn*w, y - h), (x + sgn*w, y + h), (x, y)], closed=True,
                                     fc=FIELD, ec=INKS, lw=1.4, zorder=5))
            ax.add_patch(Circle((x, y), 0.45, fc=FIELD, ec=INKS, lw=1.2, zorder=6))

        def arrow(x0, y0, x1, y1, style='-|>', lw=1.6):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=6,
                        arrowprops=dict(arrowstyle=style, color=INKS, lw=lw, shrinkA=0, shrinkB=0))

        def line_label(side):
            dash, wall = ln[f'{side}_dash'], ln[f'{side}_wall_in']
            idm = Config.an_tube_id_in(dash, wall)*25.4
            return (f"{dash_text(dash)} × {wall:.3f}\" wall\n"
                    f"ID {idm:.2f} mm   L = {ln[f'{side}_length_in']:g} in\n"
                    f"ΣK = {ln[f'{side}_minor_loss_K']:g}   ε = {ln['roughness_um']:g} µm")

        def valve_label(cv):
            return f"Cv = {cv:g}\nCdA {_mm2(cv*Config.CV_TO_CDA)}"

        x_v = 31.0

        # fuel: tank -> valve -> line over the engine -> regen inlet at the nozzle exit
        x_turn = x_exit + 3.5
        y_lip = yc + Rex + 0.9
        ax.plot([21, x_turn, x_turn], [y_f, y_f, y_lip + 2.2], color=INKS, lw=2.0, zorder=4)
        arrow(x_turn, y_lip + 2.2, x_exit + 0.2, y_lip + 0.1)
        valve(x_v, y_f)
        label(x_v, y_f + 1.6, 'Fuel main valve', valve_label(vv['fuel_cv']))
        label(x_v + 7, y_f + 0.8, 'Fuel line', line_label('fuel'), ha='left')
        label(x_turn + 0.8, y_lip + 2.6, 'Regen\ninlet', ha='left')
        if piston is not None:
            txt(21.6, y_f - 0.8, f'piston ΔP = {piston:g} psi', TEXT, 7.5, ha='left', va='top')

        # coolant runs from the nozzle exit to the injector, then into the manifold
        for xa in (x_exit - 4, (x_exit + x_inj)/2):
            k = np.argmin(np.abs(xe - xa))
            arrow(xe[k], yc + re[k] + 0.9, xe[k] - 2.2, yc + re[min(k + 15, len(re) - 1)] + 0.9, '->', 1.2)
        ax.plot([x_inj, x_inj - 1.2], [yc + Rch + 0.9, yc + Rch + 0.9], color=INKS, lw=0.8, ls='--')
        arrow(x_inj - 1.2, yc + Rch + 0.9, x_inj - 1.2, yc + Rch - 0.2, '->', 1.2)
        y_rg = yc + max(Rch, Rex) + 2.2
        txt((x_inj + x_exit)/2, y_rg,
            f"{rg['channels']} × {rg['channel_height_mm']:g} mm deep, {rg['min_channel_width_mm']:g} mm wide at throat",
            TEXT, 7.5, ha='center', va='bottom')
        label((x_inj + x_exit)/2, y_rg + 1.15, 'Regen channels', size=7.5)

        # ox: tank -> valve -> line -> injector
        ax.plot([21, x_inj - 2.4], [yc, yc], color=INKS, lw=2.0, zorder=4)
        arrow(x_inj - 5, yc, x_inj - 2.4, yc)
        valve(x_v, yc)
        label(x_v, yc - 1.8, 'Ox main valve', valve_label(vv['ox_cv']), va='top')
        label(x_v + 3, yc + 0.8, 'Ox line', line_label('ox'), ha='left')

        # injector orifices, lower right
        def orifice(key, name):
            n, dmm, cd = inj[f'{key}_holes'], inj[f'{key}_diameter_mm'], inj[f'{key}_cd']
            A = 0.25*np.pi*n*(dmm/1000)**2
            return f'{name:<15s}{n} × Ø{dmm:g} mm   Cd {cd:g}   CdA {_mm2(cd*A)}'

        y_tab = yc - max(Rch, Rex) - 6.5
        rows = [orifice(k, n) for k, n in (('ox', 'Ox injector'), ('fuel', 'Fuel injector'), ('film', 'Film'))]
        txt(x_turn, y_tab, '\n'.join(rows), TEXT, 8, ha='right', va='top', family='monospace',
            multialignment='left', linespacing=1.8)

        y_bot = y_tab - 6.5
        ax.set_ylim(y_bot, tank_top + 6)
        title.set_position((1, tank_top + 5.5))
        title.set_color(LBL)

        self.draw_idle()


class FeedSystemWindow(QtWidgets.QDialog):
    '''
    Feed system schematic. Live (follows the form) or, before exporting the vehicle, a review
    of what will be written with Save / Cancel.
    '''

    def __init__(self, parent, review=False):
        super().__init__(parent)
        self.setWindowTitle('Review vehicle before saving' if review else 'Feed system')
        self.resize(1180, 660)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        self.canvas = FeedSchematic()
        v.addWidget(self.canvas, 1)

        box = QtWidgets.QDialogButtonBox()
        if review:
            save = box.addButton('Save…', QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
            save.setObjectName('run')
            box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        else:
            box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        v.addWidget(box)

    def show_config(self, veh, sim, errors=()):
        self.canvas.show_feed(veh, sim, errors)


############ Solver thread ############

class SolverThread(QtCore.QThread):
    step = Signal(object)
    done = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, vehicle, sim):
        super().__init__()
        self.vehicle = vehicle
        self.sim = sim
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        # Anything escaping QThread.run() makes PyQt abort the whole process, so
        # every failure, including imports, is reported through a signal.
        try:
            import RXPI_CATNAP as catnap
        except Exception as e:
            self.failed.emit(f'Could not load the solver: {e}', traceback.format_exc())
            return

        def on_step(s):
            self.step.emit(s)
            time.sleep(0.002)   # let the GUI thread take the GIL and repaint

        try:
            res = catnap.run(self.vehicle, self.sim, on_step=on_step, cancel=self._stop.is_set)
            res['_html'] = catnap.write_dashboard(res)
            res['_csv'] = catnap.write_csv(res) if self.sim['output']['csv'] else None
            self.done.emit(res)
        except catnap.Cancelled:
            self.cancelled.emit()
        except BaseException as e:
            self.failed.emit(str(e) or type(e).__name__, traceback.format_exc())


############ Main window ############

SNAP_COLUMNS = [
    ('t',          't\n(s)',              lambda s: f"{s['t']:.2f}"),
    ('Lc',         'Film L_c\n(cm)',      lambda s: f"{100*s['Lc']:.2f}"),
    ('L_heat',     'Heating L\n(cm)', lambda s: f"{100*s['L_heat']:.2f}"),
    ('eta_inj',    'Stability η\n(-)',    lambda s: f"{s['eta_inj']:.2f}"),
    ('T_inj',      'Film T_inj\n(K)',     lambda s: f"{s['T_inj']:.1f}"),
    ('T_vap',      'T_vap(Pc)\n(K)',      lambda s: f"{s['T_vap']:.1f}"),
    ('q_conv_inj', 'q film\n(MW/m²)',     lambda s: f"{s['q_conv_inj']/1e6:.3f}"),
    ('q_rad_inj',  'q_rad wall\n(MW/m²)',      lambda s: f"{s['q_rad_inj']/1e6:.3f}"),
    ('psi_throat', 'Throat q/q_bare\n(-)', lambda s: f"{s['psi_throat']:.2f}"),
    ('Twall_max',  'Peak T_wall\n(K)',    lambda s: f"{s['Twall_max']:.1f}"),
    ('Tcool_out',  'T_cool out\n(K)',     lambda s: f"{s['Tcool_out']:.1f}"),
    ('iterations', 'Coupling\niter',      lambda s: f"{s['iterations']}"),
]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1100, 700)
        self.resize(1440, 900)

        self.thread = None
        self.html_out = None
        self.snaps = []

        self._derived = {}
        self.feed_win = None
        self.vform = ConfigForm(VEHICLE_FIELDS, Config.DEFAULT_VEHICLE,
                                extra={'contour': self._derived_rows, 'lines': self._line_rows})
        self.sform = ConfigForm(SIM_FIELDS, Config.DEFAULT_SIM)

        root = QtWidgets.QWidget(objectName='root')
        self.setCentralWidget(root)
        lay = QtWidgets.QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self._left_panel())
        split.addWidget(self._right_panel())
        split.setStretchFactor(1, 1)
        split.setSizes([500, 940])
        split.setHandleWidth(5)
        lay.addWidget(split)

        self._status_bar()

        self._contour_timer = QtCore.QTimer(self, singleShot=True, interval=250)
        self._contour_timer.timeout.connect(self._update_geometry)
        self.vform.changed.connect(self._contour_timer.start)
        self.vform.changed.connect(self._set_title)
        self.sform.changed.connect(self._contour_timer.start)

        self._redraw_timer = QtCore.QTimer(self, interval=150)
        self._redraw_timer.timeout.connect(self._flush_plot)
        self._dirty = False

        thermal = self.sform.editors[('regen', 'thermal_model')][0]
        thermal.toggled.connect(self._thermal_toggled)
        self._thermal_toggled(thermal.isChecked())

        self._update_geometry()
        self._set_title()

    def _thermal_toggled(self, on):
        for sec, key in (('regen', 'snapshot_times_s'),
                         ('solver', 'film_coupling_tol_K'), ('solver', 'film_coupling_max_iter')):
            self.sform.editors[(sec, key)][0].setEnabled(on)
        for key in ('enabled', 'mixing_coefficient', 'near_wall_mass_fraction', 'near_wall_OF'):
            self.vform.editors[('film', key)][0].setEnabled(on)
        for key in ('gas_radiation', 'wall_emissivity', 'gas_emissivity'):
            self.vform.editors[('regen', key)][0].setEnabled(on)

    # ── layout ──

    def _left_panel(self):
        w = QtWidgets.QWidget()
        w.setMinimumWidth(460)
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(8, 8, 4, 8)
        v.setSpacing(6)

        banner = QtWidgets.QFrame(objectName='banner')
        bl = QtWidgets.QVBoxLayout(banner)
        bl.setContentsMargins(10, 6, 10, 6)
        bl.setSpacing(0)
        bl.addWidget(QtWidgets.QLabel('CATNAP', objectName='brand'))
        bl.addWidget(QtWidgets.QLabel('Coupled Adiabatic Tank and Nozzle Analysis Program',
                                      objectName='tagline'))
        v.addWidget(banner)

        files = QtWidgets.QGridLayout()
        files.setHorizontalSpacing(6)
        files.setVerticalSpacing(4)
        for col, (kind, name) in enumerate(((Config.VEHICLE_KIND, 'vehicle'), (Config.SIM_KIND, 'sim'))):
            imp = QtWidgets.QPushButton(f'Import {name}…')
            exp = QtWidgets.QPushButton(f'Export {name}…')
            imp.clicked.connect(lambda _=False, k=kind: self.import_config(k))
            exp.clicked.connect(lambda _=False, k=kind: self.export_config(k))
            files.addWidget(imp, 0, col)
            files.addWidget(exp, 1, col)
        feed = QtWidgets.QPushButton('Feed system schematic…')
        feed.clicked.connect(self.open_feed_window)
        files.addWidget(feed, 2, 0, 1, 2)
        v.addLayout(files)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.vform, 'Vehicle')
        self.tabs.addTab(self.sform, 'Simulation')
        v.addWidget(self.tabs, 1)

        btns = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton('Run simulation', objectName='run')
        self.stop_btn = QtWidgets.QPushButton('Stop', objectName='stop')
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.start_run)
        self.stop_btn.clicked.connect(self.stop_run)
        btns.addWidget(self.run_btn, 1)
        btns.addWidget(self.stop_btn)
        v.addLayout(btns)

        return w

    def _derived_rows(self, form):
        for key, label in (('throat_diameter_mm', 'Throat Ø'), ('exit_diameter_mm', 'Exit Ø'),
                           ('chamber_diameter_mm', 'Chamber Ø'), ('expansion_ratio', 'Expansion ratio ε'),
                           ('contraction_ratio', 'Contraction ratio'), ('length_mm', 'Engine length'),
                           ('arc_kink_deg', 'Arc tangency error')):
            lbl = QtWidgets.QLabel('—', objectName='derivedVal')
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self._derived[key] = lbl
            form.add_widget_row(label, lbl)
        self._geom_err = QtWidgets.QLabel('', objectName='error')
        self._geom_err.setWordWrap(True)
        form.grid.addWidget(self._geom_err, form.row, 0, 1, 3)
        form.row += 1

    def _line_rows(self, form):
        for key, label in (('ox_line_id', 'Oxidizer line ID'), ('fuel_line_id', 'Fuel line ID')):
            lbl = QtWidgets.QLabel('—', objectName='derivedVal')
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self._derived[key] = lbl
            form.add_widget_row(label, lbl)

    def _right_panel(self):
        self.right_tabs = QtWidgets.QTabWidget()

        self.live = LivePlots()
        self.right_tabs.addTab(self.live, 'Live')

        self.contour = ContourPlot()
        self.right_tabs.addTab(self.contour, 'Contour')

        self.table = QtWidgets.QTableWidget(0, len(SNAP_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in SNAP_COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._snapshot_selected)
        self.right_tabs.addTab(self.table, 'Film + regen snapshots')

        return self.right_tabs

    def _status_bar(self):
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.phase = QtWidgets.QLabel('Ready', objectName='phase')
        self.phase.setMinimumWidth(90)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(180)
        self.progress.setFixedHeight(14)
        self.msg = QtWidgets.QLabel('Edit the configuration and press Run.')
        self.dash_btn = QtWidgets.QPushButton('Open dashboard')
        self.dash_btn.setEnabled(False)
        self.dash_btn.clicked.connect(self.open_dashboard)
        sb.addWidget(self.phase)
        sb.addWidget(self.progress)
        sb.addWidget(self.msg, 1)
        sb.addPermanentWidget(self.dash_btn)

    def _set_title(self):
        name = self.vform.collect()[0].get('name') or 'vehicle'
        self.setWindowTitle(f'CATNAP  —  {name}')

    # ── geometry readout ──

    def _update_geometry(self):
        self._refresh_feed_window()
        veh, errors = self.vform.collect()
        try:
            if errors:
                raise Config.ConfigError(errors[0])
            Config.validate(veh, Config.VEHICLE_KIND)
            d = Config.derived_geometry(veh)
        except Exception as e:
            for lbl in self._derived.values():
                lbl.setText('—')
            self._geom_err.setText(str(e))
            return

        self._geom_err.setText('')
        f = self._derived
        for side in ('ox', 'fuel'):
            idin = Config.an_tube_id_in(veh['lines'][f'{side}_dash'], veh['lines'][f'{side}_wall_in'])
            f[f'{side}_line_id'].setText(f'{25.4*idin:.2f} mm  ({idin:.3f}")')
        f['throat_diameter_mm'].setText(f"{d['throat_diameter_mm']:.2f} mm")
        f['exit_diameter_mm'].setText(f"{d['exit_diameter_mm']:.2f} mm")
        f['chamber_diameter_mm'].setText(f"{d['chamber_diameter_mm']:.2f} mm")
        f['expansion_ratio'].setText(f"{d['expansion_ratio']:.3f}")
        f['contraction_ratio'].setText(f"{d['contraction_ratio']:.3f}")
        f['length_mm'].setText(f"{d['length_mm']:.1f} mm")
        f['arc_kink_deg'].setText(f"{d['arc_kink_deg']:.2f}°")
        if d['arc_kink_deg'] > 0.5:
            self._geom_err.setText('Converging arcs are not tangent; check the arc lengths and radii.')

        self.contour.show_contour(veh)

    # ── feed schematic ──

    def open_feed_window(self):
        if self.feed_win is None:
            self.feed_win = FeedSystemWindow(self)
        self.feed_win.show()
        self.feed_win.raise_()
        self.feed_win.activateWindow()
        self._refresh_feed_window()

    def _refresh_feed_window(self):
        if self.feed_win is None or not self.feed_win.isVisible():
            return
        veh, e1 = self.vform.collect()
        sim, e2 = self.sform.collect()
        self.feed_win.show_config(veh, None if e2 else sim, e1)

    # ── config files ──

    def _form(self, kind):
        return self.vform if kind == Config.VEHICLE_KIND else self.sform

    def import_config(self, kind):
        name = 'vehicle' if kind == Config.VEHICLE_KIND else 'sim'
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f'Import {name} config', os.path.join(HERE, 'configs'), 'YAML (*.yaml *.yml)')
        if not path:
            return
        try:
            cfg, unknown = Config.load_config(path, kind)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Import failed', str(e))
            return

        self._form(kind).set_config(cfg)
        self._thermal_toggled(self.sform.editors[('regen', 'thermal_model')][0].isChecked())
        if kind == Config.VEHICLE_KIND:
            self._set_title()
            self._update_geometry()
        msg = f'Imported {os.path.basename(path)}'
        if unknown:
            msg += f"  ·  kept {len(unknown)} unrecognised key(s): {', '.join(unknown)}"
        self.msg.setText(msg)

    def export_config(self, kind):
        name = 'vehicle' if kind == Config.VEHICLE_KIND else 'sim'
        cfg, errors = self._form(kind).collect()
        try:
            if errors:
                raise Config.ConfigError('\n'.join(errors))
            Config.validate(cfg, kind)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Cannot export', str(e))
            return

        if kind == Config.VEHICLE_KIND:
            sim, e2 = self.sform.collect()
            review = FeedSystemWindow(self, review=True)
            review.show_config(cfg, None if e2 else sim)
            if review.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return

        default = os.path.join(HERE, 'configs',
                               f"{self.vform.cfg.get('name', 'engine').lower()}_{name}.yaml")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f'Export {name} config', default,
                                                        'YAML (*.yaml *.yml)')
        if not path:
            return
        Config.save_config(cfg, path)
        self._form(kind).cfg = cfg
        self.msg.setText(f'Exported {path}')

    # ── run ──

    def _collect_all(self):
        veh, e1 = self.vform.collect()
        sim, e2 = self.sform.collect()
        errors = e1 + e2
        if not errors:
            for cfg, kind in ((veh, Config.VEHICLE_KIND), (sim, Config.SIM_KIND)):
                try:
                    Config.validate(cfg, kind)
                except Config.ConfigError as e:
                    errors.append(str(e))
        return veh, sim, errors

    def start_run(self):
        if self.thread is not None:
            return
        veh, sim, errors = self._collect_all()
        if errors:
            QtWidgets.QMessageBox.critical(self, 'Invalid input', '\n'.join(errors))
            return

        self.vform.cfg, self.sform.cfg = veh, sim
        self.live.clear()
        self.snaps = []
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.dash_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.phase.setText('Running')
        self.msg.setText('Starting solver…')
        self.right_tabs.setCurrentWidget(self.live)

        self.thread = SolverThread(veh, sim)
        self.thread.step.connect(self._on_step)
        self.thread.done.connect(self._on_done)
        self.thread.failed.connect(self._on_failed)
        self.thread.cancelled.connect(self._on_cancelled)
        self.thread.finished.connect(self._thread_finished)
        self._redraw_timer.start()
        self.thread.start()

    def stop_run(self):
        if self.thread is not None:
            self.thread.stop()
            self.phase.setText('Stopping')
            self.msg.setText('Stopping after the current step…')

    def _on_step(self, s):
        self.live.add(s)
        self._dirty = True
        self.progress.setValue(int(1000*s['frac']))
        self.phase.setText(s['phase'].capitalize())
        self.msg.setText(f"t = {s['t']:.2f} s   Pc = {s['Pc']/1e6:.3f} MPa   F = {s['F']/1e3:.2f} kN   "
                         f"Isp = {s['Isp']:.1f} s   O/F = {s['MR']:.2f}   x = {s['x2']:.4f}")
        if s.get('snapshot') is not None:
            self._add_snapshot(s['snapshot'])
        if s.get('snapshot_error') is not None:
            e = s['snapshot_error']
            self._add_snapshot({'t': e['t'], 'active': False, 'skipped': True,
                                'warnings': [f"Snapshot skipped: {e['error']}"]})

    def _flush_plot(self):
        if self._dirty:
            self._dirty = False
            self.live.redraw()

    def _add_snapshot(self, snap):
        self.snaps.append(snap)
        r = self.table.rowCount()
        self.table.insertRow(r)
        tip = '\n'.join(snap.get('warnings', [])) or 'No warnings'
        for c, (key, _, fmt) in enumerate(SNAP_COLUMNS):
            if key == 't' or snap.get('active') or (key in ('Twall_max', 'Tcool_out', 'psi_throat', 'iterations') and not snap.get('skipped')):
                text = fmt(snap)
            elif snap.get('skipped') and key == 'Lc':
                text = 'skipped'
            else:
                text = '—'
            item = QtWidgets.QTableWidgetItem(text)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item.setToolTip(tip)
            if snap.get('warnings') and c == 0:
                item.setForeground(QtGui.QColor(RED))
            self.table.setItem(r, c, item)

    def _snapshot_selected(self):
        rows = {i.row() for i in self.table.selectedItems()}
        if rows:
            snap = self.snaps[min(rows)]
            self.contour.show_contour(self.vform.cfg, snap if snap.get('active') else None)

    def _on_done(self, res):
        self._flush_plot()
        self.html_out = res.get('_html')
        self.progress.setValue(1000)
        self.phase.setText('Complete')
        msg = f"Total impulse {res['impulse']/1e3:.2f} kN·s   ·   min separation margin {min(res['sepmargin_arr']):+.0f} %"
        active = [s for s in res['film'] if s.get('active')]
        if active:
            msg += f"   ·   film L_c {100*min(s['Lc'] for s in active):.2f}–{100*max(s['Lc'] for s in active):.2f} cm"
            self.contour.show_contour(self.vform.cfg, active[-1])
        if res.get('_csv'):
            msg += f"   ·   CSV: {res['_csv']}"
        self.msg.setText(msg)
        self.dash_btn.setEnabled(bool(self.html_out))
        if self.html_out and self.sform.cfg['output']['dashboard']:
            self.open_dashboard()

    def _on_failed(self, err, tb):
        self.phase.setText('Error')
        self.msg.setText(err)
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle('Solver error')
        box.setText(err)
        box.setDetailedText(tb)
        box.exec()

    def _on_cancelled(self):
        self.phase.setText('Stopped')
        self.msg.setText('Run cancelled.')

    def _thread_finished(self):
        self._redraw_timer.stop()
        self._flush_plot()
        self.thread = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def open_dashboard(self):
        if self.html_out and os.path.exists(self.html_out):
            webbrowser.open('file:///' + self.html_out.replace('\\', '/'))

    def closeEvent(self, ev):
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(30000)
        super().closeEvent(ev)


def missing_solver_deps():
    import importlib.util
    return [m for m in ('CoolProp', 'rocketcea', 'scipy', 'yaml') if importlib.util.find_spec(m) is None]


def main():
    app = QtWidgets.QApplication(sys.argv)

    # PyQt aborts on an exception escaping a slot; show it instead
    def excepthook(etype, value, tb):
        text = ''.join(traceback.format_exception(etype, value, tb))
        sys.__stderr__ and sys.__stderr__.write(text)
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Critical, 'CATNAP error', str(value))
        box.setDetailedText(text)
        box.exec()
    sys.excepthook = excepthook

    app.setApplicationName('CATNAP')
    app.setStyle('Fusion')
    app.setPalette(palette())
    sans, mono = load_fonts()
    app.setFont(QtGui.QFont(sans, 9))
    app.setStyleSheet(stylesheet(sans, mono))
    w = MainWindow()
    w.show()

    missing = missing_solver_deps()
    if missing:
        w.run_btn.setEnabled(False)
        w.phase.setText('No solver')
        w.msg.setText(f"Missing in {sys.executable}: {', '.join(missing)}")
        QtWidgets.QMessageBox.critical(
            w, 'Missing dependencies',
            f"This Python environment is missing: {', '.join(missing)}\n\n{sys.executable}\n\n"
            'Install the project requirements (pip install -r requirements.txt) in the environment '
            'you launch the GUI from. Editing and YAML import/export still work.')
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
