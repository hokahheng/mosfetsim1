#!/usr/bin/env python3
"""2D MOSFET / 3D FinFET simulator.      python main.py   (run from this folder)

Requires numpy, matplotlib (tkinter ships with Python; Linux: sudo apt install python3-tk)

Flow:  choose 2D MOSFET for compact-model results or FinFET for the 3D solver.

FinFET sizes
  Only N3A, N7A and N15A are selectable; dimensions are read-only.
  MESD device records report L=15 nm for all three nodes and include a
  one-fin case (Nfin=1). Fin height / width / EOT are application assumptions,
  not dimensions reported by MESD. Assumed Hfin / Wfin / EOT (nm):
  N3A: 55 / 5 / 0.7; N7A: 50 / 7 / 0.8; N15A: 40 / 10 / 0.9 (default).
  S/D length is 30 nm. Source: MESD, IEEE Data Descriptions (2024),
  doi:10.1109/ieeedata.2024.3481356. Oxide material remains selectable;
  physical oxide thickness is adjusted to preserve the assumed EOT.

Electrical sweeps
  Set independent Vgs/Vds start, end and step values. Blank ends use Vdd;
  blank steps use automatic sampling. Curve counts select the displayed bias
  families. Temperature start/end/step overlays the runs on the same Id and SS
  axes; a blank temperature end runs only the starting temperature.
  SS(Vgs) = 1000 / (d log10(|Id|) / dVgs), in mV/dec, requires at least three
  gate points. Zero, missing and non-increasing current slopes are masked.
  Extracted DIBL uses log-interpolated thresholds at one reference current:
  1000 * (Vth_low - Vth_high) / (Vds_high - Vds_low), in mV/V. It is unavailable
  unless that current is crossed at two positive drain biases within the sweep.
  The compact model uses approximate smooth weak-inversion charge, not a fitted
  BSIM model. Temperature changes thermal voltage and intrinsic carrier density;
  the entered mobility is held fixed. 3D current/reference current is per fin.
  Each 3D temperature run is saved under finfet_out/<node>/T_<temperature>K; its NPZ
  includes SS_curves. Compact CSV export includes all temperatures and SS.

Layout
  mosfet_sim/inputs.py      input list, presets, per-model fields, validation
  mosfet_sim/models.py      PlanarMOSFET, FinFET (DIBL, SCE)   <- add devices here
  mosfet_sim/physics.py     Central editable electrical + thermal equations/properties
  mosfet_sim/equations.py   Compatibility API for Id, gm, go
  mosfet_sim/simulate.py    curve data, Vth/DIBL vs L, operating-point check
  mosfet_sim/geometry3d.py  3D drawings per device
  mosfet_sim/plotting.py    plot pages (PAGE_SPECS registry)
  mosfet_sim/io_utils.py    CSV export / empirical import
  mosfet_sim/gui.py         input window + results window
  mosfet_sim/packages.py    TO-247 / D2PAK / DPAK geometry
  mosfet_sim/thermal_*.py   package thermal: inputs, 3D transient model, plots
  mosfet_sim/gui_thermal.py thermal input + results windows (button on the main window)
  mosfet_sim/dataset.py     ANN training-data generator      (optional)
  mosfet_sim/ann.py         NumpyMLP + HybridModel (physics + ANN residual) (optional)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mosfet_sim.gui import run  # noqa: E402

if __name__ == "__main__":
    run()
