"""Matplotlib pages. To add a page: write draw_xxx(page, ctx) and add a PageSpec
to PAGE_SPECS (when = function ctx -> bool deciding if the tab is shown). ctx is a namespace with: m (model), p, r (curves), sce, log,
emp, show3d."""
from collections import namedtuple
import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm
from .constants import BLUE, ORANGE, GRAY
from .geometry3d import draw_device
from .simulate import operating_check  # noqa: F401  (re-export for the GUI)

PageSpec = namedtuple("PageSpec", "title projection draw when")


# ---------------------------------------------------------------- helpers
def unit_for(ymax):
    return (1e3, "mA") if ymax >= 1e-3 else (1e6, "uA") if ymax >= 1e-6 else (1e9, "nA")


def cond_unit(ymax):
    return (1e3, "mS") if ymax >= 1e-3 else (1e6, "uS") if ymax >= 1e-6 else (1e9, "nS")


def alpha_i(i, n):
    return 0.4 + 0.6 * i / max(n - 1, 1)


def plot_family(pg, axi, x, curves, vgs_list, scale):
    """One blue curve per Vgs with increasing opacity."""
    n = len(curves)
    for i, (vg, c) in enumerate(zip(vgs_list, curves)):
        pg.add(axi, x, c * scale, color=BLUE, lw=2, alpha=alpha_i(i, n),
               label=f"Vgs={vg:.2f} V")


def mark_vth(ax, c):
    ax.axvline(float(c.m.vth(c.p["Vds_read_V"])), color=GRAY, ls=":", lw=1)


def curve_style(temperature_index, bias_index, temperature_count):
    color_index = temperature_index if temperature_count > 1 else bias_index
    return {"color": cm.tab10(color_index % 10),
            "ls": ("-", "--", "-.", ":")[bias_index % 4], "lw": 1.6}


class CurveList:
    """Keep large curve legends outside the figure in a scrollable list."""

    def __init__(self, parent, canvas):
        from tkinter import ttk
        self.canvas = canvas
        self.frame = ttk.Frame(parent)
        self.lines = {}
        controls = ttk.Frame(self.frame)
        controls.pack(side="right", fill="y", padx=4)
        ttk.Button(controls, text="Show all", command=lambda: self.set_visible(True)).pack(fill="x")
        ttk.Button(controls, text="Hide all", command=lambda: self.set_visible(False)).pack(fill="x")
        self.tree = ttk.Treeview(self.frame, columns=("plot", "curve"), height=5,
                                 selectmode="browse")
        self.tree.heading("#0", text="Visible")
        self.tree.column("#0", width=65, stretch=False)
        self.tree.heading("plot", text="Plot")
        self.tree.column("plot", width=160, stretch=False)
        self.tree.heading("curve", text="Curve")
        self.tree.column("curve", width=230)
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<ButtonRelease-1>", self.toggle)
        self.tree.bind("<space>", self.toggle)

    def update(self, axes):
        groups = [axis.get_legend_handles_labels() for axis in axes]
        crowded = any(len(labels) > 12 for _, labels in groups)
        self.tree.delete(*self.tree.get_children())
        self.lines.clear()
        if not crowded:
            self.frame.pack_forget()
            return
        for axis, (handles, labels) in zip(axes, groups):
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
            for line, label in zip(handles, labels):
                item = self.tree.insert("", "end", text="[x]" if line.get_visible() else "[ ]",
                                        values=(axis.get_title(), label))
                self.lines[item] = line
        self.frame.pack(side="bottom", fill="x", padx=4, pady=4)

    def toggle(self, event):
        if getattr(event, "keysym", "") == "space":
            item = self.tree.focus()
        else:
            if self.tree.identify_region(event.x, event.y) not in ("tree", "cell"):
                return
            item = self.tree.identify_row(event.y)
        if item in self.lines:
            line = self.lines[item]
            line.set_visible(not line.get_visible())
            self.tree.item(item, text="[x]" if line.get_visible() else "[ ]")
            self.canvas.draw_idle()
        return "break"

    def set_visible(self, visible):
        for item, line in self.lines.items():
            line.set_visible(visible)
            self.tree.item(item, text="[x]" if visible else "[ ]")
        self.canvas.draw_idle()


# ---------------------------------------------------------------- Page widget
class Page:
    """Notebook tab with one figure; 2 axes, or one 3D axes."""

    def __init__(self, notebook, title, projection=None):
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                       NavigationToolbar2Tk)
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        self.fig = Figure(figsize=(10, 4.8), dpi=100, constrained_layout=True)
        if projection:
            self.ax = [self.fig.add_subplot(111, projection=projection)]
        else:
            self.ax = [self.fig.add_subplot(121), self.fig.add_subplot(122)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, frame)
        self.curve_list = CurveList(frame, self.canvas)
        self.projection = projection

    def reset(self):
        for a in self.ax:
            a.clear()

    def add(self, i, x, y, **kw):
        self.ax[i].plot(x, y, **kw)

    def finish(self):
        self.curve_list.update(self.ax)
        if not self.projection:
            for a in self.ax:
                a.grid(alpha=0.3)
        self.canvas.draw_idle()


# ---------------------------------------------------------------- page 1: Id
def draw_id(pg, c):
    r, p = c.r, c.p
    pg.reset()
    ax1, ax2 = pg.ax
    cases = getattr(c, "sweeps", [c])
    ymax1 = max(cur.max() for case in cases for cur in case.r["id_vds"]) * 1.05 or 1e-9
    s1, u1 = unit_for(ymax1)
    ax1.set(xlim=(r["vd"][0], r["vd"][-1]), ylim=(0, ymax1 * s1), xlabel="Vds (V)",
            ylabel=f"Id ({u1})", title="Id - Vds")
    ax2.set(xlim=(r["vx"][0], r["vx"][-1]), xlabel="Vgs (V)", title="Id - Vgs")
    top = max(case.r["transfer"].max() for case in cases)
    if c.log:
        ax2.set_yscale("log")
        positive = np.concatenate([case.r["transfer"].ravel() for case in cases])
        positive = positive[positive > 0]
        if positive.size:
            ax2.set_ylim(positive.min() * 0.5, max(top * 2, positive.min() * 2))
        ax2.set_ylabel("Id (A, log)")
        s2 = 1
    else:
        s2, u2 = unit_for(top * 1.05 or 1e-9)
        ax2.set(ylim=(0, (top * 1.05 or 1e-9) * s2), ylabel=f"Id ({u2})")
    for temperature_index, case in enumerate(cases):
        data = case.r
        temperature = case.p["T_K"]
        for index, (vg, current) in enumerate(zip(data["vgs_list"], data["id_vds"])):
            pg.add(0, data["vd"], current * s1, label=f"{temperature:g} K, Vgs={vg:g} V",
                   **curve_style(temperature_index, index, len(cases)))
        for index, vd in enumerate(data["vds_list"]):
            pg.add(1, data["vx"], data["transfer"][:, index] * s2,
                   label=f"{temperature:g} K, Vds={vd:g} V",
                   **curve_style(temperature_index, index, len(cases)))
    if c.emp is not None:
        vg, vd, i = c.emp
        ax1.plot(vd, i * s1, "x", color=ORANGE, label="empirical")
        m = np.isclose(vd, p["Vds_read_V"])
        if m.any():
            ax2.plot(vg[m], i[m] * s2, "x", color=ORANGE)
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="best", fontsize=8)
    pg.finish()


def draw_ss(pg, c):
    pg.reset()
    cases = getattr(c, "sweeps", [c])
    for temperature_index, case in enumerate(cases):
        for index, vd in enumerate(case.r["vds_list"]):
            style = curve_style(temperature_index, index, len(cases))
            label = f"{case.p['T_K']:g} K, Vds={vd:g} V"
            pg.add(0, case.r["vx"], case.r["ss"][:, index], label=label, **style)
            pg.add(1, case.r["vx"], case.r["transfer"][:, index], label=label, **style)
    pg.ax[0].set(xlabel="Vgs (V)", ylabel="SS (mV/dec)",
                 title="SS = 1000 / (d log10(|Id|) / dVgs)")
    pg.ax[1].set(xlabel="Vgs (V)", ylabel="Id (A)", yscale="log", title="Transfer curves (log Id)")
    if not any(np.isfinite(case.r["ss"]).any() for case in cases):
        pg.ax[0].text(0.5, 0.5, "SS requires at least three Vgs points\nand positive, increasing current.",
                      transform=pg.ax[0].transAxes, ha="center")
    for axis in pg.ax:
        axis.legend(fontsize=8)
    pg.finish()


def draw_dibl(pg, c):
    pg.reset()
    cases = getattr(c, "sweeps", [c])
    for index, case in enumerate(cases):
        pg.add(0, case.r["vds_list"], case.r["Vth"], marker="o",
               label=f"{case.p['T_K']:g} K", **curve_style(index, 0, len(cases)))
    temperatures = [case.p["T_K"] for case in cases]
    dibl = [case.r["DIBL_mV_per_V"] for case in cases]
    pg.add(1, temperatures, dibl, marker="o", color=BLUE)
    pg.ax[0].set(xlabel="Vds (V)", ylabel="Vth (V)",
                 title=f"Constant-current threshold (Id={c.p['Icc_A']:g} A)")
    pg.ax[1].set(xlabel="Temperature (K)", ylabel="DIBL (mV/V)",
                 title="DIBL = 1000 (Vth_low - Vth_high) / (Vds_high - Vds_low)")
    pg.ax[0].legend(fontsize=8)
    if not np.isfinite(dibl).any():
        pg.ax[1].text(0.5, 0.5, "DIBL unavailable: reference current must be crossed\nat two drain biases.",
                      transform=pg.ax[1].transAxes, ha="center")
    pg.finish()


# ---------------------------------------------------------------- page 2: gm
def draw_gm(pg, c):
    r, p, m = c.r, c.p, c.m
    pg.reset()
    ax1, ax2 = pg.ax
    vx = r["vx"]
    top = max(np.max(g) for g in [r["gm_vgs"]] + r["gm_vds"])
    sc, un = cond_unit(top * 1.05 or 1e-9)
    ymax = top * 1.1 * sc or 1
    ax1.set(xlim=(vx[0], vx[-1]), ylim=(0, ymax), xlabel="Vgs (V)", ylabel=f"gm ({un})",
            title=f"gm vs Vgs  (Vds = {p['Vds_read_V']:g} V)")
    pg.add(0, vx, r["gm_vgs"] * sc, color=BLUE, lw=2, label="formula  gm = dId/dVgs")
    pg.add(0, vx, r["gm_vgs_num"] * sc, color=ORANGE, ls="none", marker=".", ms=4,
           label="numeric  dId/dVgs")
    edge = float(m.vth(p["Vds_read_V"])) + p["Vds_read_V"]
    if edge < p["Vdd_V"]:
        ax1.axvline(edge, color=GRAY, ls=":", lw=1)
        ax1.text(edge, ymax * 0.05, " sat. -> triode", fontsize=8, color=GRAY)
    mark_vth(ax1, c)
    ax1.legend(loc="upper left", fontsize=8)
    ax2.set(xlim=(r["vd"][0], r["vd"][-1]), ylim=(0, ymax), xlabel="Vds (V)", ylabel=f"gm ({un})",
            title="gm vs Vds  (one curve per Vgs)")
    plot_family(pg, 1, r["vd"], r["gm_vds"], r["vgs_list"], sc)
    ax2.legend(loc="upper right", fontsize=8)
    pg.finish()


# ---------------------------------------------------------------- page 3: go
def draw_go(pg, c):
    r, p = c.r, c.p
    pg.reset()
    ax1, ax2 = pg.ax
    vx = r["vx"]
    top = max(np.max(g) for g in r["go_vds"] + [r["go_vgs"]])
    sc, un = cond_unit(top * 1.05 or 1e-9)
    ymax = top * 1.1 * sc or 1
    ax1.set(xlim=(r["vd"][0], r["vd"][-1]), ylim=(0, ymax), xlabel="Vds (V)", ylabel=f"go ({un})",
            title="go vs Vds  (one curve per Vgs)")
    plot_family(pg, 0, r["vd"], r["go_vds"], r["vgs_list"], sc)
    pg.add(0, r["vd"], r["go_vds_num"][-1] * sc, color=ORANGE, ls="none", marker=".", ms=4,
           label=f"numeric dId/dVds (Vgs={r['vgs_list'][-1]:.2f} V)")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.set(xlim=(vx[0], vx[-1]), ylim=(0, ymax), xlabel="Vgs (V)", ylabel=f"go ({un})",
            title=f"go vs Vgs  (Vds = {p['Vds_read_V']:g} V)")
    pg.add(1, vx, r["go_vgs"] * sc, color=BLUE, lw=2, label="formula")
    mark_vth(ax2, c)
    ax2.legend(loc="upper left", fontsize=8)
    pg.finish()


# ---------------------------------------------------------------- page 4: SCE
def draw_sce(pg, c):
    s, p = c.sce, c.p
    pg.reset()
    ax1, ax2 = pg.ax
    L = s["L_nm"]
    pg.add(0, L, s["vth_lin"], color=BLUE, lw=2, label=f"Vds = {s['vlin']:g} V (linear)")
    pg.add(0, L, s["vth_hi"], color=ORANGE, lw=2, ls="--", label=f"Vds = {s['vhi']:g} V")
    pg.add(1, L, s["dibl"], color=BLUE, lw=2)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.axvline(p["L_nm"], color=GRAY, ls=":", lw=1)
        ax.set_xlabel("Gate length L (nm)")
    ax1.set(ylabel="Vth (V)", title="Threshold roll-off (SCE) and DIBL")
    ax2.set(ylabel="DIBL (mV/V)", title="DIBL = (Vth_lin - Vth_hi)/(Vhi - Vlin)")
    ax1.legend(loc="lower right", fontsize=8)
    if c.p["device"] == "2D MOSFET":
        ax2.text(0.5, 0.5, "ideal planar model:\nno short-channel effects",
                 transform=ax2.transAxes, ha="center", color=GRAY)
    pg.finish()


# ---------------------------------------------------------------- page 5: 3D
def draw_3d(pg, c):
    pg.reset()
    ax = pg.ax[0]
    draw_device(ax, c.p)
    pg.finish()


def _always(_c):
    return True


PAGE_SPECS = [PageSpec("Id curves", None, draw_id, _always),
              PageSpec("SS vs Vgs", None, draw_ss, _always),
              PageSpec("Extracted DIBL", None, draw_dibl, _always),
              PageSpec("gm", None, draw_gm, _always),
              PageSpec("go", None, draw_go, _always),
              PageSpec("SCE / DIBL", None, draw_sce, _always),
              PageSpec("3D structure", "3d", draw_3d, lambda c: c.show3d)]   # only if ticked
