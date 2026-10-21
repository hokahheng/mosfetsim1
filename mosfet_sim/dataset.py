"""Training-data generation for an ANN (not used by the GUI).

    X, Y, cols = build_dataset("FinFET", n_devices=300, n_bias=100)
    X : [device features..., Vgs, Vds]      Y : [Id, gm, go]
The same feature vector can be built for any single device with feature_vector().
"""
import numpy as np
from .inputs import DEV_MOSFET, DEV_FINFET, parse_inputs, preset_values
from .models import build_model

# (low, high, "log" | "lin")  -- sampling ranges for every varied parameter
RANGES = {
    DEV_MOSFET: {"L_nm": (50, 500, "log"), "W_um": (1, 50, "log"),
                 "Na_cm3": (1e16, 1e18, "log"), "tox_nm": (2, 10, "lin"),
                 "mu_n": (200, 500, "lin"), "phi_m_eV": (4.2, 4.9, "lin"),
                 "lam": (0, 0.1, "lin"), "Vdd_V": (1, 3.3, "lin")},
    DEV_FINFET: {"L_nm": (10, 60, "log"), "Hfin_nm": (20, 60, "lin"),
                 "Wfin_nm": (4, 15, "lin"), "Nfin": (1, 6, "lin"),
                 "Na_cm3": (1e15, 1e17, "log"), "tox_nm": (0.8, 2.0, "lin"),
                 "mu_n": (100, 300, "lin"), "phi_m_eV": (4.4, 4.8, "lin"),
                 "lam": (0, 0.2, "lin"), "Vdd_V": (0.6, 1.2, "lin")}}
FEATURES = {k: list(v) for k, v in RANGES.items()}


def sample_params(device, rng, oxide="SiO2", gate="Tri-gate"):
    raw = preset_values(device)
    for k, (lo, hi, kind) in RANGES[device].items():
        raw[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if kind == "log" \
            else float(rng.uniform(lo, hi))
    raw["Vds_read_V"] = raw["Vdd_V"]
    return parse_inputs({k: str(v) for k, v in raw.items()}, device, oxide, gate)


def feature_vector(p):
    return [p[k] for k in FEATURES[p["device"]]]


def build_dataset(device, n_devices=200, n_bias=100, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_devices):
        p = sample_params(device, rng)
        m = build_model(p)
        vg = rng.uniform(0, p["Vdd_V"], n_bias)
        vd = rng.uniform(0, p["Vdd_V"], n_bias)
        feat = np.tile(feature_vector(p), (n_bias, 1))
        X.append(np.column_stack([feat, vg, vd]))
        Y.append(np.column_stack([m.id(vg, vd), m.gm(vg, vd), m.go(vg, vd)]))
    return np.vstack(X), np.vstack(Y), FEATURES[device] + ["Vgs", "Vds"]
