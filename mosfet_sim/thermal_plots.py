"""Thermal visualisation (matplotlib only, no Tk): 3D heat map of the package surface
or a cut-away, and temperature / Rds(on) time curves."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .geometry3d import add_box
from .packages import PACKAGES
from . import physics

LEAD_GRAY = "#9aa0a6"
MATERIAL_SERIES = (
    (physics.MOLD, "Epoxy mold", "#537f8c"),
    (physics.CU, "Copper tab", "#b87333"),
    (physics.SOLDER, "Solder attach", "#c49a35"),
    (physics.SI, "Silicon die", "#d9534f"),
)
STEP_FILES = {
    "TO-247": "Fet TO247 KA3875.STEP",
    "D2PAK": "D2PACK.stp",
    "DPAK": "TO-252-2.stp",
}
STEP_AXIS_MAP = {"TO-247": (0, 1, 2), "D2PAK": (0, 1, 2), "DPAK": (1, 0, 2)}
STEP_AXIS_REVERSED = {"TO-247": (False, True, False),
                      "D2PAK": (False, False, False),
                      "DPAK": (False, False, False)}
STEP_DIR = Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=3)
def _load_step_mesh(step_file):
    import gmsh

    started_gmsh = not gmsh.isInitialized()
    if started_gmsh:
        gmsh.initialize(["gmsh", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("mosfet_step_surface")
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.8)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        for _, surface in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(2, surface)
        gmsh.model.mesh.generate(2)

        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coordinates).reshape(-1, 3)
        node_index = {int(tag): i for i, tag in enumerate(tags)}
        polygons = []
        for _, surface in gmsh.model.getEntities(2):
            element_types, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
            for element_type, node_tags in zip(element_types, element_nodes):
                _, dimension, _, node_count, _, _ = gmsh.model.mesh.getElementProperties(
                    int(element_type))
                if dimension != 2 or node_count not in (3, 4):
                    continue
                for element in np.asarray(node_tags).reshape(-1, node_count):
                    polygons.append(vertices[[node_index[int(tag)] for tag in element]])
        if not polygons:
            raise RuntimeError("The STEP file did not produce a surface mesh.")
        return polygons, vertices.min(axis=0), vertices.max(axis=0)
    finally:
        gmsh.model.remove()
        if started_gmsh:
            gmsh.finalize()


class HeatView:
    """Owns a figure with one 3D axes. set_frame(k) recolours the faces."""

    def __init__(self, fig, model, res, cutaway=False):
        self.model, self.res = model, res
        p, self.pkg = model.p, PACKAGES[model.p["package"]]
        fig.clf()
        self.fig = fig
        self.ax = ax = fig.add_subplot(111, projection="3d")
        self.package = p["package"]
        self.use_step = self.package in STEP_FILES and not cutaway
        self.step_error = None
        if self.use_step:
            try:
                step_file = STEP_DIR / STEP_FILES[self.package]
                self.step_polygons, self.step_min, self.step_max = _load_step_mesh(str(step_file))
                self.step_cells = self._map_step_to_grid()
                polys = self.step_polygons
                self.cells = None
            except Exception as exc:
                self.use_step = False
                self.step_error = str(exc)
        if not self.use_step:
            polys, cells = self._faces(cutaway)
            self.cells = np.array(cells)
        vmin = p["T_amb_C"]
        vmax = max(float(res["Tj_max"].max()), vmin + 1.0)
        self.norm, self.cmap = colors.Normalize(vmin, vmax), cm.turbo
        edgecolors = (0, 0, 0, 0) if self.use_step else (0, 0, 0, 0.25)
        self.coll = Poly3DCollection(polys, edgecolors=edgecolors, linewidths=0.3)
        ax.add_collection3d(self.coll)
        if self.use_step:
            self._step_arrow()
            self._step_limits()
        else:
            self._leads()
            self._arrow()
            self._limits()
        fig.colorbar(cm.ScalarMappable(self.norm, self.cmap), ax=ax, shrink=0.6,
                     pad=0.08, label="Temperature (C)")
        self.label = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)
        title = "STEP surface" if self.use_step else ("cut-away" if cutaway else "surface")
        ax.set_title(f"{p['package']}  -  {title}")
        if self.use_step:
            ax.text2D(0.02, 0.02, "STEP geometry; colors projected from the thermal grid",
                      transform=ax.transAxes, fontsize=8, color="gray")
        elif self.step_error:
            ax.text2D(0.02, 0.02, "STEP unavailable; showing simplified package",
                      transform=ax.transAxes, fontsize=8, color="tab:red")
        ax.view_init(elev=24, azim=-60)
        self.set_frame(0)

    def _map_step_to_grid(self):
        centers = np.array([polygon.mean(axis=0) for polygon in self.step_polygons])
        step_span = np.maximum(self.step_max - self.step_min, 1e-12)
        step_fractions = (centers - self.step_min) / step_span
        axis_map = STEP_AXIS_MAP[self.package]
        fractions = step_fractions[:, axis_map].copy()
        for axis, reverse in enumerate(STEP_AXIS_REVERSED[self.package]):
            if reverse:
                fractions[:, axis] = 1.0 - fractions[:, axis]
        cell_indices, mapped_positions = [], []
        for axis, edges_m in enumerate(self.model.edges):
            edges = edges_m * 1e3
            positions = edges[0] + fractions[:, axis] * (edges[-1] - edges[0])
            mapped_positions.append(positions)
            indices = np.searchsorted(edges, positions, side="right") - 1
            cell_indices.append(np.clip(indices, 0, len(edges) - 2))
        mapped_cells = self.model.idx[tuple(cell_indices)]
        unmapped = np.flatnonzero(mapped_cells < 0)
        if len(unmapped):
            mapped_positions = np.column_stack(mapped_positions)
            solid_indices = np.argwhere(self.model.solid)
            solid_centers = np.column_stack([
                (edges_m[solid_indices[:, axis]] + edges_m[solid_indices[:, axis] + 1]) * 500
                for axis, edges_m in enumerate(self.model.edges)
            ])
            for start in range(0, len(unmapped), 256):
                batch = unmapped[start:start + 256]
                distances = mapped_positions[batch, None, :] - solid_centers[None, :, :]
                nearest = np.argmin(np.sum(distances * distances, axis=2), axis=1)
                nearest_indices = solid_indices[nearest]
                mapped_cells[batch] = self.model.idx[tuple(nearest_indices.T)]
        return mapped_cells

    def _step_limits(self):
        span = np.maximum(self.step_max - self.step_min, 1.0)
        margin = span * 0.12
        limits = [(lo - pad, hi + pad)
                  for lo, hi, pad in zip(self.step_min, self.step_max, margin)]
        for axis, bounds in zip((self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim), limits):
            axis(*bounds)
        self.ax.set_box_aspect(tuple(span))
        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")
        self.ax.set_zlabel("z (mm)")

    def _step_arrow(self):
        axis, sign = self.model.p["flow_axis"], self.model.p["flow_sign"]
        step_axis = STEP_AXIS_MAP[self.package][axis]
        if STEP_AXIS_REVERSED[self.package][axis]:
            sign = -sign
        start = [(lo + hi) / 2 for lo, hi in zip(self.step_min, self.step_max)]
        start[2] = self.step_max[2] + 0.5
        start[step_axis] = (self.step_min[step_axis] - 2 if sign > 0
                            else self.step_max[step_axis] + 2)
        direction = [0.0, 0.0, 0.0]
        direction[step_axis] = sign * 5.0
        self.ax.quiver(*start, *direction, color="tab:blue", linewidth=2.5,
                       arrow_length_ratio=0.3)

    def _faces(self, cutaway):
        m = self.model
        xe, ye, ze = (e * 1e3 for e in m.edges)
        E = (xe, ye, ze)
        vis = m.solid.copy()
        if cutaway:
            xc = (xe[:-1] + xe[1:]) / 2
            vis &= (xc <= (self.pkg["mold"][0] + self.pkg["mold"][1]) / 2)[:, None, None]
        polys, cells = [], []
        for a in range(3):
            o1, o2 = [i for i in range(3) if i != a]
            for s in (-1, 1):
                nb = np.zeros_like(vis)
                src, dst = [slice(None)] * 3, [slice(None)] * 3
                if s < 0:
                    src[a], dst[a] = slice(0, -1), slice(1, None)
                else:
                    src[a], dst[a] = slice(1, None), slice(0, -1)
                nb[tuple(dst)] = vis[tuple(src)]
                for ijk in np.argwhere(vis & ~nb):
                    pos = E[a][ijk[a] + (1 if s > 0 else 0)]
                    r1 = (E[o1][ijk[o1]], E[o1][ijk[o1] + 1])
                    r2 = (E[o2][ijk[o2]], E[o2][ijk[o2] + 1])
                    quad = []
                    for u, w in ((r1[0], r2[0]), (r1[1], r2[0]), (r1[1], r2[1]), (r1[0], r2[1])):
                        pt = [0, 0, 0]
                        pt[a], pt[o1], pt[o2] = pos, u, w
                        quad.append(pt)
                    polys.append(quad)
                    cells.append(m.idx[tuple(ijk)])
        return polys, cells

    def _leads(self):
        g = self.pkg
        for x in g["lead_x"]:
            add_box(self.ax, x - g["lead_w"] / 2, x + g["lead_w"] / 2, g["lead_y"][0],
                    g["lead_y"][1] + 0.5, g["lead_z0"], g["lead_z0"] + g["lead_t"],
                    LEAD_GRAY, 1.0)

    def _arrow(self):
        p, mold, tab = self.model.p, self.pkg["mold"], self.pkg["tab"]
        x0, x1 = mold[0], mold[1]
        y0, y1 = self.pkg["lead_y"][0], max(mold[3], tab[3])
        z = mold[5] + 3
        L = 10
        if p["flow_axis"] == 1:
            ys = y0 - 2 if p["flow_sign"] > 0 else y1 + 2
            self.ax.quiver((x0 + x1) / 2, ys, z, 0, p["flow_sign"] * L, 0,
                           color="tab:blue", linewidth=2.5, arrow_length_ratio=0.3)
            self.ax.text((x0 + x1) / 2, ys, z + 1.5, "air flow", color="tab:blue", fontsize=8)
        else:
            xs = x0 - 2 - L if p["flow_sign"] > 0 else x1 + 2 + L
            ym = (y0 + y1) / 2
            self.ax.quiver(xs, ym, z, p["flow_sign"] * L, 0, 0,
                           color="tab:blue", linewidth=2.5, arrow_length_ratio=0.3)
            self.ax.text(xs, ym, z + 1.5, "air flow", color="tab:blue", fontsize=8)

    def _limits(self):
        g = self.pkg
        xs = [g["mold"][0], g["mold"][1], g["tab"][0], g["tab"][1]]
        ys = [g["lead_y"][0] - 3, g["tab"][3], g["mold"][3]]
        xr = (min(xs) - 12, max(xs) + 12) if self.model.p["flow_axis"] == 0 else (min(xs) - 2, max(xs) + 2)
        yr = (min(ys) - 2, max(ys) + 2)
        zr = (0, g["mold"][5] + 6)
        self.ax.set_xlim(*xr); self.ax.set_ylim(*yr); self.ax.set_zlim(*zr)
        sp = [xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]]
        big = max(sp)
        self.ax.set_box_aspect(tuple(max(s, 0.3 * big) for s in sp))
        self.ax.set_xlabel("x (mm)"); self.ax.set_ylabel("y (mm)"); self.ax.set_zlabel("z (mm)")

    def set_frame(self, k):
        r = self.res
        T = r["T"][k]
        if self.use_step:
            surface_t = np.full(len(self.step_cells), self.model.p["T_amb_C"])
            valid = self.step_cells >= 0
            surface_t[valid] = T[self.step_cells[valid]]
        else:
            surface_t = T[self.cells]
        self.coll.set_facecolor(self.cmap(self.norm(surface_t)))
        self.label.set_text(f"t = {r['t'][k]:6.1f} s   Tj = {r['Tj_max'][k]:.1f} C   "
                            f"Rds(on) = {r['Rds_mOhm'][k]:.2f} mOhm   P = {r['P'][k]:.2f} W")


def draw_series(fig, res, p):
    """Left: temperatures vs time. Right: Rds(on) and power vs time.
    Returns the two time-marker lines (move them with set_xdata)."""
    fig.clf()
    a1, a2 = fig.add_subplot(121), fig.add_subplot(122)
    t = res["t"]
    a1.plot(t, res["Tj_max"], color="tab:red", lw=2, label="Tj max (die)")
    a1.plot(t, res["Tj_mean"], color="tab:orange", lw=1.5, ls="--", label="Tj mean")
    a1.plot(t, res["Tcase"], color="tab:green", lw=1.5, label="tab (case)")
    a1.plot(t, res["Tsink"], color="tab:blue", lw=1.5, label="sink / PCB")
    a1.axhline(p["T_amb_C"], color="gray", ls=":", lw=1, label="ambient")
    a1.axhline(p["Tj_max_C"], color="red", ls=":", lw=1, label="Tj limit")
    a1.set(xlabel="time (s)", ylabel="Temperature (C)", title="Temperature vs time",
           xlim=(0, t[-1]))
    a1.legend(fontsize=8, loc="center right")
    a2.plot(t, res["Rds_mOhm"], color="tab:purple", lw=2, label="Rds(on)")
    a2.set(xlabel="time (s)", ylabel="Rds(on) (mOhm)", title="Rds(on) and power vs time",
           xlim=(0, t[-1]))
    a3 = a2.twinx()
    a3.plot(t, res["P"], color="tab:orange", lw=1.5, ls="--", label="P = Id^2 Rds")
    a3.set_ylabel("Power (W)")
    h = a2.get_legend_handles_labels()[0] + a3.get_legend_handles_labels()[0]
    a2.legend(h, ["Rds(on)", "Power"], fontsize=8, loc="center right")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    m1, m2 = a1.axvline(0, color="k", lw=1), a2.axvline(0, color="k", lw=1)
    fig.tight_layout()
    return m1, m2


def draw_material_series(fig, model, res):
    """Plot volume-weighted mean and hottest-cell temperature per material."""
    fig.clf()
    ax = fig.add_subplot(111)
    cell_volumes = (np.diff(model.edges[0])[:, None, None]
                    * np.diff(model.edges[1])[None, :, None]
                    * np.diff(model.edges[2])[None, None, :])
    time = res["t"]
    plotted = 0
    for material_id, label, color in MATERIAL_SERIES:
        mask = model.mat == material_id
        cell_indices = model.idx[mask]
        if not len(cell_indices):
            continue
        weights = cell_volumes[mask]
        weights = weights / weights.sum()
        temperatures = res["T"][:, cell_indices]
        mean_temperature = temperatures @ weights
        maximum_temperature = temperatures.max(axis=1)
        conductivity = physics.MAT_K[material_id]
        ax.plot(time, mean_temperature, color=color, lw=2,
                label=f"{label} mean (k={conductivity:g} W/mK)")
        ax.plot(time, maximum_temperature, color=color, lw=1.1, ls="--",
                label=f"{label} max")
        plotted += 1
    ax.set(xlabel="time (s)", ylabel="Temperature (C)",
           title="Package material temperatures", xlim=(0, time[-1]))
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8, ncol=2, loc="best")
    ax.text(0.01, -0.17,
            "Solid lines: volume-weighted mean; dashed: hottest cell. "
            "External leads are shown for reference but are not in the thermal mesh.",
            transform=ax.transAxes, fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
