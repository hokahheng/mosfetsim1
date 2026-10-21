"""Tk windows for the power-MOSFET package thermal simulation.
  ThermalInputWindow    package / ambient / airflow / current ... -> Run
  ThermalResultsWindow  3D heat map with time slider + curves + numbers
"""
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .thermal_inputs import (THERMAL_PACKAGES, THERMAL_INPUTS, FLOW_DIRS, parse_thermal,
                             thermal_preset, MAX_DURATION_S)
from .thermal_model import ThermalModel
from .thermal_plots import HeatView, draw_material_series, draw_series


class ThermalInputWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Power MOSFET package thermal - inputs")
        self.results = None
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="1. Select package", font=("", 10, "bold")).pack(anchor="w")
        self.package = tk.StringVar(value=THERMAL_PACKAGES[0])
        cb = ttk.Combobox(f, textvariable=self.package, state="readonly", width=20,
                          values=THERMAL_PACKAGES)
        cb.pack(anchor="w", pady=(2, 8))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.load_preset())
        ttk.Label(f, text="2. Operating conditions", font=("", 10, "bold")).pack(anchor="w")
        g = ttk.Frame(f)
        g.pack(fill="x", pady=(2, 8))
        self.vars = {}
        for r, (key, label, default) in enumerate(THERMAL_INPUTS):
            ttk.Label(g, text=label).grid(row=r, column=0, sticky="w", pady=1)
            self.vars[key] = tk.StringVar(value=default)
            ttk.Entry(g, textvariable=self.vars[key], width=10).grid(row=r, column=1, padx=6)
        r = len(THERMAL_INPUTS)
        ttk.Label(g, text="Air flow direction").grid(row=r, column=0, sticky="w")
        self.flow = tk.StringVar(value=list(FLOW_DIRS)[0])
        ttk.Combobox(g, textvariable=self.flow, state="readonly", width=22,
                     values=list(FLOW_DIRS)).grid(row=r, column=1, padx=6)
        self.status = ttk.Label(f, text="")
        ttk.Button(f, text="3. Run thermal simulation  ->  open results",
                   command=self.run).pack(fill="x", pady=(8, 2))
        self.status.pack(anchor="w")
        self.load_preset()

    def load_preset(self):
        for k, v in thermal_preset(self.package.get()).items():
            self.vars[k].set(v)

    def run(self):
        try:
            p = parse_thermal({k: v.get() for k, v in self.vars.items()},
                              self.package.get(), self.flow.get())
            self.status.config(text="Simulating ...")
            self.update_idletasks()
            model = ThermalModel(p)
            res = model.run()
        except Exception as e:
            self.status.config(text="")
            messagebox.showerror("Input error", str(e))
            return
        self.status.config(text=f"Done ({len(res['t'])} time steps).")
        if self.results is not None and self.results.winfo_exists():
            self.results.destroy()
        self.results = ThermalResultsWindow(self, model, res, self.lift)


class ThermalResultsWindow(tk.Toplevel):
    def __init__(self, master, model, res, to_inputs):
        super().__init__(master)
        self.model, self.res, self.p = model, res, model.p
        self.title(f"Thermal results - {self.p['package']}")
        self.k, self.playing = 0, False
        self.nframes = len(res["t"])
        self.stride = max(1, self.nframes // 150)

        bar = ttk.Frame(self, padding=4)
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="<- Edit inputs", command=to_inputs).pack(side="left")
        ttk.Button(bar, text="Export CSV (time, Tj, Rds)", command=self.export).pack(side="left", padx=4)
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        info = tk.Text(body, width=44, font=("Consolas", 9))
        info.pack(side="left", fill="y", padx=4, pady=4)
        info.insert("end", "\n".join(model.summary(res)))
        info.config(state="disabled")
        nb = ttk.Notebook(body)
        nb.pack(side="right", fill="both", expand=True)

        # ---- tab 1: 3D heat map + time controls
        t1 = ttk.Frame(nb)
        nb.add(t1, text="3D heat map")
        ctl = ttk.Frame(t1, padding=2)
        ctl.pack(side="top", fill="x")
        self.play_btn = ttk.Button(ctl, text="Play", command=self.toggle_play, width=7)
        self.play_btn.pack(side="left")
        self.slider_var = tk.DoubleVar(value=0)
        ttk.Scale(ctl, from_=0, to=self.nframes - 1, variable=self.slider_var,
                  command=self.on_slider).pack(side="left", fill="x", expand=True, padx=6)
        self.view_mode = tk.StringVar(value="Surface")
        cb = ttk.Combobox(ctl, textvariable=self.view_mode, state="readonly", width=10,
                          values=["Surface", "Cut-away"])
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self.build_view())
        self.fig3 = Figure(figsize=(8, 5.2), dpi=100)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=t1)
        self.canvas3.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas3, t1)
        self.build_view()

        # ---- tab 2: curves
        t2 = ttk.Frame(nb)
        nb.add(t2, text="Temperature / Rds(on) vs time")
        self.fig2 = Figure(figsize=(10, 4.8), dpi=100)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=t2)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas2, t2)
        self.markers = draw_series(self.fig2, res, self.p)
        self.canvas2.draw_idle()

        # ---- tab 3: temperatures by material
        t3 = ttk.Frame(nb)
        nb.add(t3, text="Material temperatures")
        self.fig_material = Figure(figsize=(10, 4.8), dpi=100)
        self.canvas_material = FigureCanvasTkAgg(self.fig_material, master=t3)
        self.canvas_material.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_material, t3)
        draw_material_series(self.fig_material, model, res)
        self.canvas_material.draw_idle()

    def build_view(self):
        self.view = HeatView(self.fig3, self.model, self.res,
                             cutaway=self.view_mode.get() == "Cut-away")
        self.view.set_frame(self.k)
        self.canvas3.draw_idle()

    def set_frame(self, k):
        self.k = int(max(0, min(self.nframes - 1, k)))
        self.view.set_frame(self.k)
        t = float(self.res["t"][self.k])
        for m in self.markers:
            m.set_xdata([t, t])
        self.canvas3.draw_idle()
        self.canvas2.draw_idle()

    def on_slider(self, value):
        self.set_frame(round(float(value)))

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            if self.k >= self.nframes - 1:
                self.k = 0
            self.tick()

    def tick(self):
        if not self.playing:
            return
        self.set_frame(self.k + self.stride)
        self.slider_var.set(self.k)
        if self.k >= self.nframes - 1:
            self.playing = False
            self.play_btn.config(text="Play")
            return
        self.after(40, self.tick)

    def export(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        hdr, rows = self.model.export_rows(self.res)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(rows)
        messagebox.showinfo("Saved", f"Wrote {path}")
