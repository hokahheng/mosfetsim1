"""Tk front end.
  Window 1 (inputs):   choose model -> fill parameters -> Run
  Window 2 (results):  opens after Run with the plot pages and the numbers
"""
import os
import queue
import threading
from dataclasses import replace
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .constants import OXIDE_KAPPA
from .finfet_3d import Par, main_run
from .inputs import (DEVICES, DEV_FINFET, FINFET_NODES, FINFET_SIZE_KEYS,
                     GATE_TYPES, INPUTS, parse_inputs,
                     preset_values, visible_keys)
from .models import build_model
from .simulate import simulate, sce_sweep, operating_check
from .plotting import Page, PAGE_SPECS, CurveList, curve_style
from .io_utils import export_csv, load_empirical


class ResultsWindow(tk.Toplevel):
    """Second window: info text + notebook of plot pages (lazy drawing)."""

    def __init__(self, master, ctx, to_inputs):
        super().__init__(master)
        self.ctx = ctx
        self.title(f"Results - {ctx.m.label}")
        self.specs = [s for s in PAGE_SPECS if s.when(ctx)]
        bar = ttk.Frame(self, padding=4)
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="<- Edit inputs", command=to_inputs).pack(side="left")
        ttk.Button(bar, text="Export CSV (for ANN)", command=self.export).pack(side="left", padx=4)
        ttk.Button(bar, text="Load empirical overlay", command=self.load_emp).pack(side="left")
        self.log = tk.BooleanVar()
        ttk.Checkbutton(bar, text="Log Id axis", variable=self.log,
                        command=lambda: self.redraw(0)).pack(side="left", padx=8)
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.info = tk.Text(body, width=40, font=("Consolas", 9))
        self.info.pack(side="left", fill="y", padx=4, pady=4)
        self.nb = ttk.Notebook(body)
        self.nb.pack(side="right", fill="both", expand=True)
        self.pages = [Page(self.nb, s.title, s.projection) for s in self.specs]
        self.drawn = [False] * len(self.specs)
        self.show_info()
        self.nb.bind("<<NotebookTabChanged>>", self.on_tab)
        self.on_tab()

    def on_tab(self, _event=None):
        i = self.nb.index(self.nb.select())
        if not self.drawn[i]:
            self.ctx.log = self.log.get()
            self.specs[i].draw(self.pages[i], self.ctx)
            self.drawn[i] = True

    def redraw(self, i):
        self.drawn[i] = False
        self.on_tab()

    def show_info(self):
        m, p = self.ctx.m, self.ctx.p
        k = operating_check(m, p)
        lines = m.summary() + [
            f"\n-- check at Vgs={p['Vdd_V']:g} V, Vds={p['Vds_read_V']:g} V --",
            f"Id            = {k['Id']*1e3:.5g} mA",
            f"gm = 2Id/Vov  = {k['gm_2Id_Vov']*1e3:.5g} mS",
            f"gm = dId/dVgs = {k['gm_num']*1e3:.5g} mS",
            f"gm (formula)  = {k['gm_formula']*1e3:.5g} mS",
            f"go (formula)  = {k['go_formula']*1e6:.5g} uS",
            f"go = dId/dVds = {k['go_num']*1e6:.5g} uS"]
        lines.append(f"\n-- Constant-current DIBL (Id={p['Icc_A']:g} A) --")
        for case in self.ctx.sweeps:
            value = case.r["DIBL_mV_per_V"]
            text = f"{value:.3g} mV/V" if np.isfinite(value) else "unavailable (current not crossed at two biases)"
            lines.append(f"{case.p['T_K']:g} K: {text}")
        self.info.insert("end", "\n".join(lines))
        self.info.config(state="disabled")

    def export(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            export_csv(path, self.ctx.m, self.ctx.p, self.ctx.r,
                       runs=[(case.m, case.p, case.r) for case in self.ctx.sweeps])
            messagebox.showinfo("Saved", f"Wrote {path}")

    def load_emp(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.ctx.emp = load_empirical(path)
        except Exception as e:
            messagebox.showerror("CSV error", f"Need columns Vgs,Vds,Id (Id in A).\n{e}")
            return
        self.redraw(0)


class FinFETResultsWindow(tk.Toplevel):
    """Display the 3D solver's measured bias sweeps and threshold-derived DIBL."""

    def __init__(self, master, result):
        super().__init__(master)
        self.title("3D FinFET results")
        self.geometry("1100x760")
        self.results = result.get("temperature_results", [result])
        self.result = self.results[0]
        summaries = []
        for item in self.results:
            dibl = item["DIBL_mV_per_V"]
            value = f"{dibl:.2f} mV/V" if np.isfinite(dibl) else "unavailable (need two threshold crossings)"
            summaries.append(f"{item['T_K']:g} K: DIBL {value}")
        ttk.Label(self, text=";  ".join(summaries), wraplength=1050,
                  padding=8).pack(anchor="w")
        self.log = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Log Id-Vgs axis", variable=self.log,
                        command=self.redraw).pack(anchor="w", padx=8)
        self.plots = []
        self.curve_lists = []
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self._add_plot("Id vs Vgs", self.draw_id_vgs)
        self._add_plot("Id vs Vds", self.draw_id_vds)
        self._add_plot("SS vs Vgs", self.draw_ss)
        self._add_plot("DIBL", self.draw_dibl)

    def _add_plot(self, title, draw):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        figure = Figure(figsize=(9, 5.5), dpi=100, constrained_layout=True)
        axis = figure.add_subplot(111)
        draw(axis)
        axis.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, frame)
        curve_list = CurveList(frame, canvas)
        curve_list.update([axis])
        self.curve_lists.append(curve_list)
        canvas.draw()
        self.plots.append((axis, canvas, draw))

    def redraw(self):
        for (axis, canvas, draw), curve_list in zip(self.plots, self.curve_lists):
            axis.clear()
            draw(axis)
            axis.grid(True, alpha=0.3)
            curve_list.update([axis])
            canvas.draw_idle()

    def draw_id_vgs(self, axis):
        logarithmic = self.log.get()
        scale = 1 if logarithmic else 1e6
        for temperature_index, result in enumerate(self.results):
            for bias_index, index in enumerate(result["transfer_indices"]):
                vds = result["Vds"][index]
                current = result["Id"][:, index]
                if logarithmic:
                    current = np.where(current > 0, current, np.nan)
                axis.plot(result["Vgs"], current * scale,
                          label=f"{result['T_K']:g} K, Vds={vds:g} V",
                          **curve_style(temperature_index, bias_index, len(self.results)))
        axis.set(xlabel="Vgs (V)", ylabel="Ids (A/fin)" if logarithmic else "Ids (uA/fin)",
                 yscale="log" if logarithmic else "linear", title="3D FinFET Id vs Vgs")
        axis.legend(loc="best", fontsize=8, ncol=2)

    def draw_id_vds(self, axis):
        for temperature_index, result in enumerate(self.results):
            for bias_index, index in enumerate(result["output_indices"]):
                vgs = result["Vgs"][index]
                axis.plot(result["Vds"], result["Id"][index, :] * 1e6,
                          label=f"{result['T_K']:g} K, Vgs={vgs:g} V",
                          **curve_style(temperature_index, bias_index, len(self.results)))
        axis.set(xlabel="Vds (V)", ylabel="Ids (uA/fin)", title="3D FinFET Id vs Vds")
        axis.legend(loc="best", fontsize=8, ncol=2)

    def draw_ss(self, axis):
        for temperature_index, result in enumerate(self.results):
            for bias_index, index in enumerate(result["transfer_indices"]):
                axis.plot(result["Vgs"], result["SS_curves"][:, index],
                          label=f"{result['T_K']:g} K, Vds={result['Vds'][index]:g} V",
                          **curve_style(temperature_index, bias_index, len(self.results)))
        axis.set(xlabel="Vgs (V)", ylabel="SS (mV/dec)",
                 title="SS = 1000 / (d log10(|Id|) / dVgs)")
        axis.legend(loc="best", fontsize=8, ncol=2)
        if all(len(result["Vgs"]) < 3 for result in self.results):
            axis.text(0.5, 0.5, "SS requires at least three Vgs points.",
                      ha="center", transform=axis.transAxes)

    def draw_dibl(self, axis):
        for index, result in enumerate(self.results):
            dibl = result["DIBL_mV_per_V"]
            value = f"{dibl:.2f} mV/V" if np.isfinite(dibl) else "unavailable"
            axis.plot(result["Vds_threshold"], result["Vth"], marker="o",
                      label=f"{result['T_K']:g} K, DIBL {value}",
                      **curve_style(index, 0, len(self.results)))
        axis.set(xlabel="Vds (V)", ylabel="Vth (V)",
                 title=f"Constant-current threshold (Id={self.result['Icc_A_per_fin']:g} A/fin)")
        if not any(np.isfinite(result["DIBL_mV_per_V"]) for result in self.results):
            axis.text(0.5, 0.5, "DIBL requires a reference-current crossing at two drain biases.",
                      ha="center", transform=axis.transAxes)
        axis.legend(loc="best")


class App:
    """Input window."""

    def __init__(self, root):
        self.root = root
        root.title("MOSFET / FinFET simulator - inputs")
        self.vars, self.rows, self.results = {}, {}, None
        self._finfet_messages = queue.Queue()
        frame = ttk.Frame(root, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="1. Select model", font=("", 10, "bold")).pack(anchor="w")
        self.device = tk.StringVar(value=DEVICES[0])
        cb = ttk.Combobox(frame, textvariable=self.device, state="readonly",
                          values=DEVICES, width=20)
        cb.pack(anchor="w", pady=(2, 8))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.on_device())
        ttk.Label(frame, text="2. Enter parameters", font=("", 10, "bold")).pack(anchor="w")
        self.grid = ttk.Frame(frame)
        self.grid.pack(fill="x", pady=(2, 8))
        for key, label, default in INPUTS:
            self.vars[key] = tk.StringVar(value=default)
            self._row(key, label, ttk.Entry(self.grid, textvariable=self.vars[key], width=14))
        self.finfet_node = tk.StringVar(value="N15A")
        self.vars["finfet_node"] = self.finfet_node
        node_control = ttk.Combobox(self.grid, textvariable=self.finfet_node,
                                   state="readonly", width=12, values=list(FINFET_NODES))
        node_control.bind("<<ComboboxSelected>>", lambda _event: self.on_finfet_node())
        self._row("finfet_node", "FinFET node", node_control)
        self.oxide, self.gate = tk.StringVar(value="SiO2"), tk.StringVar(value="Tri-gate")
        self._row("oxide", "Oxide material", ttk.Combobox(
            self.grid, textvariable=self.oxide, state="readonly", width=12,
            values=list(OXIDE_KAPPA)))
        self.rows["oxide"][1].bind("<<ComboboxSelected>>", lambda _event: self.on_oxide())
        self._row("gate", "Gate type", ttk.Combobox(
            self.grid, textvariable=self.gate, state="readonly", width=12,
            values=list(GATE_TYPES)))
        self.show3d = tk.BooleanVar()
        self.show3d_control = ttk.Checkbutton(frame, text="Draw 3D structure in results",
                                              variable=self.show3d)
        self.show3d_control.pack(anchor="w")
        self.run_button = ttk.Button(frame, text="3. Run simulation  ->  open results window",
                                     command=self.run)
        self.run_button.pack(fill="x", pady=(8, 0))
        self.run_status = tk.StringVar()
        ttk.Label(frame, textvariable=self.run_status).pack(anchor="w", pady=(3, 0))
        ttk.Separator(frame).pack(fill="x", pady=10)
        ttk.Button(frame, text="Power MOSFET package thermal (TO-247 / D2PAK / DPAK) ...",
                   command=self.open_thermal).pack(fill="x")
        self.on_device()

    def _row(self, key, label, widget):
        self.rows[key] = (ttk.Label(self.grid, text=label), widget)

    def on_device(self):
        """Load the preset for the selected model and show only its fields."""
        dev = self.device.get()
        for k, v in preset_values(dev, self.finfet_node.get(), self.oxide.get()).items():
            self.vars[k].set(v)
        for key in FINFET_SIZE_KEYS:
            self.rows[key][1].configure(state="readonly" if dev == DEV_FINFET else "normal")
        self.rows["oxide"][1].configure(state="readonly")
        for lbl, w in self.rows.values():
            lbl.grid_remove()
            w.grid_remove()
        keys = visible_keys(dev)
        split = (len(keys) + 1) // 2
        for index, key in enumerate(keys):
            column = 2 * (index // split)
            row = index % split
            lbl, w = self.rows[key]
            lbl.grid(row=row, column=column, sticky="w", pady=1, padx=(8, 0))
            w.grid(row=row, column=column + 1, padx=6, sticky="w")
        if dev == DEV_FINFET:
            self.show3d_control.pack_forget()
            self.run_button.configure(text="3. Run 3D FinFET simulation")
            self.run_status.set("3D solver simulates one fin; current is reported per fin.")
        else:
            self.show3d_control.pack(anchor="w", before=self.run_button)
            self.run_button.configure(text="3. Run simulation  ->  open results window")
            self.run_status.set("")

    def on_finfet_node(self):
        values = preset_values(DEV_FINFET, self.finfet_node.get(), self.oxide.get())
        for key in FINFET_SIZE_KEYS:
            self.vars[key].set(values[key])

    def on_oxide(self):
        if self.device.get() == DEV_FINFET:
            values = preset_values(DEV_FINFET, self.finfet_node.get(), self.oxide.get())
            self.vars["tox_nm"].set(values["tox_nm"])

    def open_thermal(self):
        from .gui_thermal import ThermalInputWindow
        ThermalInputWindow(self.root)

    def run_finfet_3d(self, p):
        par = Par()
        par.outDir = os.path.join(par.outDir, p["finfet_node"])
        par.Lg = p["L_nm"]
        par.Lj = par.Lg / 2
        par.Lsd = p["Lsd_nm"]
        par.Wfin = p["Wfin_nm"]
        par.Hfin = p["Hfin_nm"]
        par.Na = p["Na_cm3"] * 1e-21
        par.Nsd = p["Nsd_cm3"] * 1e-21
        par.PhiM = p["phi_m_eV"]
        par.mu0 = p["mu_n"]
        par.tIL = p["tox_nm"]
        par.epsIL = OXIDE_KAPPA[p["oxide"]]
        par.tHf = 0.0
        par.vMax = p["Vdd_V"]
        par.vStep = p["Vdd_V"] / p["n_vgs"]
        par.vgs_values = tuple(p["Vgs_sweep"])
        par.vds_values = tuple(p["Vds_sweep"])
        par.Icc_A = p["Icc_A"]
        par.show_plots = False
        par.verbose = False
        self.run_button.configure(state="disabled")
        points = len(par.vgs_values) * np.count_nonzero(p["Vds_sweep"] > 0) * len(p["temperatures"])
        self.run_status.set(f"3D FinFET sweep running: {points} bias/temperature points...")
        threading.Thread(target=self._finfet_worker, args=(par, p), daemon=True).start()
        self.root.after(100, self._poll_finfet)

    def _finfet_worker(self, par, p):
        try:
            results = []
            for temperature in p["temperatures"]:
                folder = os.path.join(par.outDir, f"T_{temperature:.12g}K")
                result = main_run(replace(par, T_K=float(temperature), outDir=folder))
                positive = np.flatnonzero(result["Vds"] > 0)
                selected = np.linspace(0, len(positive) - 1, p["n_vds"], dtype=int)
                if p["n_vds"] == 1:
                    selected = np.array([len(positive) - 1])
                result["transfer_indices"] = positive[np.unique(selected)]
                result["output_indices"] = np.unique(np.linspace(
                    0, len(result["Vgs"]) - 1, p["n_vgs"], dtype=int))
                results.append(result)
            self._finfet_messages.put((True, {"temperature_results": results, "outDir": par.outDir}))
        except Exception as exc:
            self._finfet_messages.put((False, str(exc)))

    def _poll_finfet(self):
        try:
            success, detail = self._finfet_messages.get_nowait()
        except queue.Empty:
            if self.root.winfo_exists():
                self.root.after(100, self._poll_finfet)
            return
        self.run_button.configure(state="normal")
        if success:
            self.run_status.set(f"Finished. Results: {detail['outDir']}")
            self.finfet_results = FinFETResultsWindow(self.root, detail)
        else:
            self.run_status.set("3D FinFET simulation failed")
            messagebox.showerror("3D FinFET error", detail, parent=self.root)

    def run(self):
        try:
            raw = {k: v.get() for k, v in self.vars.items()}
            p = parse_inputs(raw, self.device.get(), self.oxide.get(), self.gate.get())
            if p["device"] == DEV_FINFET:
                self.run_finfet_3d(p)
                return
            sweeps = []
            for temperature in p["temperatures"]:
                point = dict(p, T_K=float(temperature))
                m = build_model(point)
                sweeps.append(SimpleNamespace(m=m, p=point, r=simulate(m, point),
                                              sce=sce_sweep(m, point)))
            ctx = SimpleNamespace(**vars(sweeps[0]), sweeps=sweeps,
                                  log=False, emp=None, show3d=self.show3d.get())
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return
        if self.results is not None and self.results.winfo_exists():
            self.results.destroy()
        self.results = ResultsWindow(self.root, ctx, to_inputs=self.root.lift)


def run():
    root = tk.Tk()
    App(root)
    root.mainloop()
