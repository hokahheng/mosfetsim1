"""Transient 3D thermal grid and result reporting for a power MOSFET.

The physical equations and tunable properties are implemented in physics.py.
This module builds the package grid and exposes the solver/results interface.

Model
  * Non-uniform finite-volume grid; cells: mold, copper tab, solder, silicon die.
  * Conduction between cells (harmonic mean of k); implicit (backward Euler) time
    stepping with a precomputed dense inverse, so large steps are stable.
  * Heat source: P = Id^2 * Rds(Tj),  Rds(T) = Rds25 (T/298.15)^n, uniform in the
    die, with Tj = mean die temperature (solved implicitly every step).
  * Exposed faces lose heat by convection to ambient air (flow in ONE direction):
      faces parallel to the flow : laminar flat plate  h(d)=0.332 k Pr^(1/3) sqrt(v/nu)/sqrt(d)
                                   (d = distance from the upstream edge)
      windward face              : 1.2 * average plate h     (assumption)
      leeward face               : 0.5 * average plate h     (assumption)
      natural convection + radiation floor H_NAT, combined as (hf^3 + hn^3)^(1/3)
      downward faces that are not on the mount surface: x0.5
  * The copper tab bottom is attached to a lumped sink/PCB node (heat capacity
    Csink, resistance to ambient Rsa/(1 + SINK_V_GAIN sqrt(v))) through a contact
    resistance R_CONTACT.
  * Leads are NOT in the thermal mesh (drawn for reference only; conservative).
Tunable assumptions are the module constants below.
"""
import numpy as np
from .packages import PACKAGES
from . import physics

MOLD, CU, SOLDER, SI = physics.MOLD, physics.CU, physics.SOLDER, physics.SI
MAT_K, MAT_RC = physics.MAT_K, physics.MAT_RC
H_NAT, R_CONTACT = physics.H_NAT, physics.R_CONTACT
SINK_V_GAIN, MAX_CELLS = physics.SINK_V_GAIN, physics.MAX_CELLS
Z_RUNAWAY_C = physics.Z_RUNAWAY_C


def air_props(t_c):
    """Thermal conductivity, kinematic viscosity, Prandtl of air near t_c (C)."""
    return physics.air_properties(t_c)


def make_edges(keys_mm, h_mm):
    keys = sorted({round(k, 6) for k in keys_mm})
    out = [keys[0]]
    for a, b in zip(keys, keys[1:]):
        n = max(1, int(np.ceil((b - a) / h_mm - 1e-9)))
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(out) * 1e-3                                         # metres


def build_grid(pkg, h_mm):
    m, t, (dx0, dx1, dy0, dy1) = pkg["mold"], pkg["tab"], pkg["die_xy"]
    zs = t[5] + pkg["solder_t"]
    xe = make_edges([m[0], m[1], t[0], t[1], dx0, dx1], h_mm)
    ye = make_edges([m[2], m[3], t[2], t[3], dy0, dy1], h_mm)
    ze = make_edges([m[4], m[5], t[4], t[5], zs, zs + pkg["die_t"]], max(h_mm * 0.7, 0.3))
    c = [(e[:-1] + e[1:]) / 2 * 1e3 for e in (xe, ye, ze)]               # centres, mm

    def box(x0, x1, y0, y1, z0, z1):
        return ((c[0] > x0) & (c[0] < x1))[:, None, None] & \
               ((c[1] > y0) & (c[1] < y1))[None, :, None] & \
               ((c[2] > z0) & (c[2] < z1))[None, None, :]
    mat = np.zeros((len(c[0]), len(c[1]), len(c[2])), int)
    mat[box(*m[:6])] = MOLD
    mat[box(*t[:6])] = CU
    mat[box(dx0, dx1, dy0, dy1, t[5], zs)] = SOLDER
    mat[box(dx0, dx1, dy0, dy1, zs, zs + pkg["die_t"])] = SI
    return (xe, ye, ze), mat


class ThermalModel:
    def __init__(self, p):
        self.p, self.pkg = p, PACKAGES[p["package"]]
        h = p["mesh_mm"]
        while True:                                   # auto-coarsen to keep it fast
            self.edges, self.mat = build_grid(self.pkg, h)
            if (self.mat > 0).sum() <= MAX_CELLS:
                break
            h *= 1.15
        self.mesh_used_mm = h
        self._assemble()

    # ------------------------------------------------------------ assembly
    def _assemble(self):
        self.solid = self.mat > 0
        self.idx = -np.ones(self.mat.shape, int)
        self.N = int(self.solid.sum())
        self.idx[self.solid] = np.arange(self.N)
        physics.assemble_thermal(self)

    # ------------------------------------------------------------ simulation
    def rds(self, t_c):
        return physics.rds_on(self.p, t_c)

    def run(self):
        return physics.run_thermal(self)

    def summary(self, res):
        p = self.p
        e = -1
        tl = np.nonzero(res["Tj_max"] > p["Tj_max_C"])[0]
        out = [f"Package   = {p['package']}",
               f"Cells     = {self.N}   mesh {self.mesh_used_mm:.2f} mm",
               f"Ambient   = {p['T_amb_C']:g} C   air {p['air_v']:g} m/s  {p['flow_name'].split()[0]}",
               f"Id        = {p['Id_A']:g} A   Rds25 = {p['Rds25_mOhm']:g} mOhm",
               f"Simulated = {res['t'][e]:.1f} s",
               f"Tj max    = {res['Tj_max'][e]:.1f} C (end)",
               f"Tj mean   = {res['Tj_mean'][e]:.1f} C",
               f"T tab     = {res['Tcase'][e]:.1f} C",
               f"T sink    = {res['Tsink'][e]:.1f} C",
               f"Rds(on)   = {res['Rds_mOhm'][0]:.2f} -> {res['Rds_mOhm'][e]:.2f} mOhm "
               f"(x{res['Rds_mOhm'][e] / res['Rds_mOhm'][0]:.2f})",
               f"Power     = {res['P'][0]:.2f} -> {res['P'][e]:.2f} W",
               f"Rth(j-a)  = {(res['Tj_mean'][e] - p['T_amb_C']) / res['P'][e]:.2f} K/W (at end)"]
        if len(tl):
            out.append(f"WARNING: Tj > {p['Tj_max_C']:g} C after {res['t'][tl[0]]:.1f} s")
        if res["runaway"]:
            out.append("THERMAL RUNAWAY - simulation stopped")
        return out

    def export_rows(self, res):
        hdr = ["time_s", "Tj_max_C", "Tj_mean_C", "Tcase_C", "Tsink_C", "Rds_on_mOhm", "P_W"]
        rows = zip(res["t"], res["Tj_max"], res["Tj_mean"], res["Tcase"], res["Tsink"],
                   res["Rds_mOhm"], res["P"])
        return hdr, rows
