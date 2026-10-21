"""CSV export (training data for an ANN) and empirical CSV import."""
import csv
import numpy as np

_BASE = ("L_nm", "W_um", "Na_cm3", "T_K", "mu_n", "tox_nm", "phi_m_eV", "lam",
         "Lsd_nm", "Xj_nm", "Nsd_cm3")


def export_csv(path, m, p, r, runs=None):
    """Export every temperature and plotted bias family; SS is NaN on output sweeps."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for index, (model, params, curves) in enumerate(runs or [(m, p, r)]):
            feats = {k: params[k] for k in _BASE}
            feats.update({"is_finfet": int(params["device"] == "FinFET"),
                          "Hfin_nm": 0, "Wfin_nm": 0, "Nfin": 0, "n_gates": 0,
                          "lam_nat_nm": 0, "dVth_sce": 0, "eta_dibl": 0})
            feats.update(model.ann_features())
            feats["oxide_kappa"] = model.kappa
            feats["Vth_long"] = model.vth_long
            feats["Weff_um"] = model.width * 1e6
            gate_out, drain_out = np.meshgrid(curves["vgs_list"], curves.get("vd", curves["vx"]),
                                              indexing="ij")
            drain_biases = curves.get("vds_list", [params["Vds_read_V"]])
            gate_in, drain_in = np.meshgrid(curves["vx"], drain_biases, indexing="ij")
            vg = np.concatenate([gate_out.ravel(), gate_in.ravel()])
            vd = np.concatenate([drain_out.ravel(), drain_in.ravel()])
            ss = np.concatenate([np.full(gate_out.size, np.nan),
                                 curves.get("ss", np.full_like(gate_in, np.nan)).ravel()])
            cols = zip(vg, vd, model.id(vg, vd), model.gm(vg, vd), model.go(vg, vd), ss)
            if index == 0:
                w.writerow(list(feats) + ["Vgs", "Vds", "Id_A", "gm_S", "go_S", "SS_mV_dec"])
            row = list(feats.values())
            w.writerows(row + list(values) for values in cols)


def load_empirical(path):
    """CSV with columns Vgs,Vds,Id (Id in A) -> (vgs, vds, id) arrays."""
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a["Vgs"], a["Vds"], a["Id"]
