"""Run with: python -m unittest discover -s mosfet_app -p test_sweeps.py."""
import csv
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

from mosfet_sim.inputs import DEVICES, DEV_MOSFET, parse_inputs, preset_values
from mosfet_sim.models import build_model
from mosfet_sim.simulate import constant_current_dibl, simulate, subthreshold_swing
from mosfet_sim.io_utils import export_csv


class SweepTests(unittest.TestCase):
    def params(self, device=DEV_MOSFET, **changes):
        raw = preset_values(device)
        raw.update(changes)
        return parse_inputs(raw, device, "SiO2", "Tri-gate")

    def test_input_ranges_and_endpoints(self):
        for device in DEVICES:
            params = self.params(device, T_stop_K="350", Vgs_min_V="0.1",
                                 Vgs_max_V="0.8", Vgs_step_V="0.1",
                                 Vds_max_V="1", Vds_step_V="0.3")
            np.testing.assert_allclose(params["temperatures"], [300, 325, 350])
            np.testing.assert_allclose(params["Vgs_sweep"], np.linspace(0.1, 0.8, 8))
            np.testing.assert_allclose(params["Vds_sweep"], [0, 0.3, 0.6, 0.9, 1])

    def test_invalid_inputs(self):
        for changes in ({"T_step_K": "0"}, {"T_stop_K": "250"},
                        {"Vgs_step_V": "-1"}, {"Vds_max_V": "nan"},
                        {"Icc_A": "0"}, {"n_vds": "13"},
                        {"Vgs_step_V": "1e-9"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.params(**changes)

    def test_finfet_node_sizes_are_enforced(self):
        from mosfet_sim.inputs import DEV_FINFET, FINFET_NODES, FINFET_SIZE_KEYS

        self.assertEqual(tuple(FINFET_NODES), ("N3A", "N7A", "N15A"))
        self.assertEqual({dimensions["L_nm"] for dimensions in FINFET_NODES.values()}, {"15"})
        for node in FINFET_NODES:
            raw = preset_values(DEV_FINFET, node)
            raw.update({key: "999" for key in FINFET_SIZE_KEYS})
            raw["finfet_node"] = node
            params = parse_inputs(raw, DEV_FINFET, "HfO2", "Tri-gate")
            expected = preset_values(DEV_FINFET, node, "HfO2")
            for key in FINFET_SIZE_KEYS:
                self.assertEqual(params[key], float(expected[key]))
            self.assertEqual(params["finfet_node"], node)
            self.assertEqual(params["oxide"], "HfO2")
        self.assertEqual(self.params(DEV_FINFET)["finfet_node"], "N15A")
        with self.assertRaises(ValueError):
            self.params(DEV_FINFET, finfet_node="N5A")
        with self.assertRaises(ValueError):
            preset_values(DEV_FINFET, "custom")
        self.assertEqual(self.params(L_nm="123")["L_nm"], 123)

    def test_finfet_node_selector_locks_only_size_inputs(self):
        import tkinter as tk
        from mosfet_sim.gui import App
        from mosfet_sim.inputs import DEV_FINFET, FINFET_NODES, FINFET_SIZE_KEYS, visible_keys

        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.addCleanup(root.destroy)
        root.withdraw()
        app = App(root)
        app.device.set(DEV_FINFET)
        app.on_device()
        self.assertEqual(tuple(app.rows["finfet_node"][1]["values"]), tuple(FINFET_NODES))
        self.assertEqual(str(app.rows["oxide"][1]["state"]), "readonly")
        app.vars["T_stop_K"].set("350")
        app.vars["Vgs_step_V"].set("0.02")
        app.oxide.set("HfO2")
        app.rows["oxide"][1].event_generate("<<ComboboxSelected>>")
        self.assertEqual(app.vars["tox_nm"].get(), preset_values(DEV_FINFET, "N15A", "HfO2")["tox_nm"])
        for node in FINFET_NODES:
            app.finfet_node.set(node)
            app.on_finfet_node()
            expected = preset_values(DEV_FINFET, node, "HfO2")
            for key in FINFET_SIZE_KEYS:
                self.assertEqual(app.vars[key].get(), expected[key])
                self.assertEqual(str(app.rows[key][1]["state"]), "readonly")
            self.assertEqual(app.vars["T_stop_K"].get(), "350")
            self.assertEqual(app.vars["Vgs_step_V"].get(), "0.02")
            self.assertEqual(str(app.rows["T_K"][1]["state"]), "normal")
            self.assertEqual(app.oxide.get(), "HfO2")
        raw = {key: value.get() for key, value in app.vars.items()}
        params = parse_inputs(raw, DEV_FINFET, app.oxide.get(), app.gate.get())
        with patch("mosfet_sim.gui.threading.Thread") as thread, patch.object(root, "after"):
            app.run_finfet_3d(params)
        solver_params = thread.call_args.kwargs["args"][0]
        self.assertAlmostEqual(solver_params.tIL, params["tox_nm"])
        self.assertEqual(solver_params.epsIL, 22.0)
        self.assertAlmostEqual(solver_params.tIL * 3.9 / solver_params.epsIL, 0.9)
        app.device.set(DEV_MOSFET)
        app.on_device()
        self.assertEqual(str(app.rows["L_nm"][1]["state"]), "normal")
        self.assertEqual(str(app.rows["oxide"][1]["state"]), "readonly")
        self.assertNotIn("finfet_node", visible_keys(DEV_MOSFET))

    def test_finfet_oxide_selection_preserves_node_eot(self):
        from mosfet_sim.inputs import DEV_FINFET, FINFET_NODES
        from mosfet_sim.constants import OXIDE_KAPPA

        for node, dimensions in FINFET_NODES.items():
            for oxide, kappa in OXIDE_KAPPA.items():
                with self.subTest(node=node, oxide=oxide):
                    raw = dict(preset_values(DEV_FINFET, node), finfet_node=node)
                    params = parse_inputs(raw, DEV_FINFET, oxide, "Tri-gate")
                    self.assertEqual(params["oxide"], oxide)
                    self.assertAlmostEqual(params["tox_nm"] * 3.9 / kappa,
                                           float(dimensions["tox_nm"]))
                    model = build_model(params)
                    self.assertAlmostEqual(model.EOT_nm, float(dimensions["tox_nm"]))
        with self.assertRaises(ValueError):
            parse_inputs(preset_values(DEV_FINFET), DEV_FINFET, "unknown", "Tri-gate")

    def test_ss_differentiates_log_current_at_each_temperature(self):
        gate = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1])
        swings = np.array([0.06, 0.075, 0.09])
        current = 1e-12 * 10 ** (gate[:, None] / swings)
        expected = np.broadcast_to(swings * 1000, current.shape)
        np.testing.assert_allclose(subthreshold_swing(gate, current), expected)

    def test_ss_invalid_currents_and_short_sweeps(self):
        gate = np.linspace(0, 1, 11)
        for current in (np.zeros(11), np.ones(11), np.exp(-gate)):
            self.assertTrue(np.isnan(subthreshold_swing(gate, current)).all())
        self.assertTrue(np.isnan(subthreshold_swing(gate[:2], np.ones(2))).all())
        current = 1e-12 * 10 ** (gate / 0.075)
        current[5] = np.nan
        self.assertTrue(np.isnan(subthreshold_swing(gate, current)[4:7]).all())

    def test_constant_current_dibl(self):
        gate = np.linspace(0, 1, 101)
        drain = np.array([0.05, 0.5, 1.0])
        current = 1e-7 * 10 ** ((gate[:, None] - (0.5 - 0.08 * drain)) / 0.075)
        thresholds, dibl = constant_current_dibl(gate, drain, current, 1e-7)
        np.testing.assert_allclose(thresholds, 0.5 - 0.08 * drain)
        self.assertAlmostEqual(dibl, 80)
        thresholds, dibl = constant_current_dibl(gate, drain, current, 1e20)
        self.assertTrue(np.isnan(thresholds).all())
        self.assertTrue(np.isnan(dibl))

    def test_threshold_does_not_bridge_failed_samples(self):
        gate = np.array([0, 0.2, 0.4, 0.6, 0.8])
        current = np.array([[1e-12], [1e-10], [np.nan], [1e-6], [1e-4]])
        thresholds, dibl = constant_current_dibl(gate, [1.0], current, 1e-8)
        self.assertTrue(np.isnan(thresholds).all())
        self.assertTrue(np.isnan(dibl))

    def test_compact_axes_and_derivatives(self):
        for device in DEVICES:
            params = self.params(device, Vgs_step_V="0.01", Vds_step_V="0.07", lam="0.1")
            model = build_model(params)
            curves = simulate(model, params)
            self.assertEqual(curves["transfer"].shape, (len(params["Vgs_sweep"]), 2))
            self.assertEqual(len(curves["id_vds"][0]), len(params["Vds_sweep"]))
            self.assertTrue(np.isfinite(curves["ss"]).any())
            gate = float(model.vth(0.5)) + np.array([-0.4, -0.1, 0, 0.1, 0.8])
            drain = np.array([0.05, 0.2, 0.5, 0.8, 1.0])
            step = 1e-6
            gm = (model.id(gate + step, drain) - model.id(gate - step, drain)) / (2 * step)
            go = (model.id(gate, drain + step) - model.id(gate, drain - step)) / (2 * step)
            np.testing.assert_allclose(model.gm(gate, drain), gm, rtol=1e-5, atol=1e-12)
            np.testing.assert_allclose(model.go(gate, drain), go, rtol=1e-5, atol=1e-12)

    def test_export_all_temperatures(self):
        params = self.params(T_stop_K="350", Vgs_step_V="0.1", Vds_step_V="0.2")
        runs = []
        for temperature in params["temperatures"]:
            point = dict(params, T_K=float(temperature))
            model = build_model(point)
            runs.append((model, point, simulate(model, point)))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "curves.csv")
            export_csv(path, *runs[0], runs=runs)
            with open(path, newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual({float(row["T_K"]) for row in rows}, {300, 325, 350})
        expected = sum(len(data["vgs_list"]) * len(data["vd"]) +
                       len(data["vx"]) * len(data["vds_list"]) for _, _, data in runs)
        self.assertEqual(len(rows), expected)
        self.assertTrue(any(np.isfinite(float(row["SS_mV_dec"])) for row in rows))

    def test_finfet_temperature_batch_and_plot_counts(self):
        from mosfet_sim.gui import App, FinFETResultsWindow
        import queue

        params = self.params(DEVICES[1], T_stop_K="350")
        from mosfet_sim.finfet_3d import Par
        gate = np.linspace(0, 1, 11)
        drain = np.array([0, 0.05, 1.0])

        def solver(par):
            current = 1e-7 * 10 ** ((gate[:, None] - (0.5 - 0.08 * drain)) / (0.075 * par.T_K / 300))
            current[:, 0] = 0
            thresholds, dibl = constant_current_dibl(gate, drain[1:], current[:, 1:], par.Icc_A)
            return dict(Vgs=gate, Vds=drain, Id=current, Vds_threshold=drain[1:],
                        Vth=thresholds, DIBL_mV_per_V=dibl, Icc_A_per_fin=par.Icc_A,
                        SS_curves=subthreshold_swing(gate, current), T_K=par.T_K)

        app = SimpleNamespace(_finfet_messages=queue.Queue())
        with patch("mosfet_sim.gui.main_run", side_effect=solver) as mocked:
            App._finfet_worker(app, Par(Icc_A=1e-7), params)
        success, result = app._finfet_messages.get_nowait()
        self.assertTrue(success, result)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(len({call.args[0].outDir for call in mocked.call_args_list}), 3)
        context = SimpleNamespace(results=result["temperature_results"], log=SimpleNamespace(get=lambda: False))
        context.result = context.results[0]
        for draw, count in ((FinFETResultsWindow.draw_id_vgs, 6),
                            (FinFETResultsWindow.draw_id_vds, 15),
                            (FinFETResultsWindow.draw_ss, 6),
                            (FinFETResultsWindow.draw_dibl, 3)):
            axis = Figure().subplots()
            draw(context, axis)
            self.assertEqual(len(axis.lines), count)


    def test_crowded_legends_scroll_without_shrinking_axes(self):
        import tkinter as tk
        from tkinter import ttk
        from mosfet_sim.plotting import Page

        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.addCleanup(root.destroy)
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        page = Page(notebook, "Crowded curves")
        for axis in page.ax:
            axis.set(title="Id curves", xlabel="Voltage (V)", ylabel="Id (A)")
            for index in range(240):
                axis.plot([0, 1], [0, index + 1], label=f"Curve {index}")
            axis.legend()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            page.finish()
            for size in ("1000x700", "800x600"):
                root.geometry(size)
                root.update()
                for axis in page.ax:
                    self.assertGreater(axis.bbox.width, 200)
                    self.assertGreater(axis.bbox.height, 200)
                    self.assertIsNone(axis.get_legend())
        self.assertFalse([item for item in caught if "layout" in str(item.message).lower()])
        self.assertEqual(len(page.curve_list.lines), 480)
        page.curve_list.set_visible(False)
        self.assertTrue(all(not line.get_visible() for line in page.curve_list.lines.values()))
        item = next(iter(page.curve_list.lines))
        page.curve_list.tree.focus(item)
        page.curve_list.toggle(SimpleNamespace(keysym="space"))
        self.assertTrue(page.curve_list.lines[item].get_visible())
        page.curve_list.set_visible(True)
        self.assertTrue(all(line.get_visible() for line in page.curve_list.lines.values()))
        for axis in page.ax:
            axis.clear()
            axis.plot([0, 1], [0, 1], label="Single curve")
            axis.legend()
        page.finish()
        root.update()
        self.assertFalse(page.curve_list.lines)
        self.assertFalse(page.curve_list.frame.winfo_ismapped())
        self.assertTrue(all(axis.get_legend() is not None for axis in page.ax))

    def test_crowded_finfet_legends_survive_log_redraw(self):
        import tkinter as tk
        from mosfet_sim.gui import FinFETResultsWindow

        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.addCleanup(root.destroy)
        root.withdraw()
        gate = np.linspace(0, 1, 12)
        drain = np.linspace(0.05, 1, 12)
        results = []
        for temperature in np.linspace(300, 490, 20):
            current = 1e-7 * 10 ** ((gate[:, None] - (0.5 - 0.08 * drain)) /
                                   (0.075 * temperature / 300))
            thresholds, dibl = constant_current_dibl(gate, drain, current, 1e-7)
            results.append(dict(Vgs=gate, Vds=drain, Id=current, Vds_threshold=drain,
                                Vth=thresholds, DIBL_mV_per_V=dibl, Icc_A_per_fin=1e-7,
                                SS_curves=subthreshold_swing(gate, current), T_K=temperature,
                                transfer_indices=list(range(12)), output_indices=list(range(12))))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            window = FinFETResultsWindow(root, dict(temperature_results=results))
            root.update()
            window.log.set(True)
            window.redraw()
            root.update()
        self.assertFalse([item for item in caught if "layout" in str(item.message).lower()])
        for (axis, _, _), curve_list in zip(window.plots[:3], window.curve_lists[:3]):
            self.assertEqual(len(axis.lines), 240)
            self.assertEqual(len(curve_list.lines), 240)
            self.assertIsNone(axis.get_legend())
        self.assertEqual(window.plots[0][0].get_yscale(), "log")


if __name__ == "__main__":
    unittest.main()