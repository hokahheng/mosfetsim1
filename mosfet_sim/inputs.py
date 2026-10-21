"""Input definitions, device presets and input validation (no GUI code).

MESD N3A, N7A and N15A records use L=15 nm and include Nfin=1. Their fin
cross-section and EOT are not specified in the records, so those remain
application assumptions. Oxide physical thickness preserves the assumed EOT
for the selected material. The 3D solver models one fin.
"""
import numpy as np
from .constants import OXIDE_KAPPA

DEV_MOSFET, DEV_FINFET = "2D MOSFET", "FinFET"
DEVICES = [DEV_MOSFET, DEV_FINFET]
GATE_TYPES = {"Tri-gate": 3, "Double-gate": 2}     # name -> number of gated fin faces

# (key, label, default for the planar MOSFET)
INPUTS = [
    ("L_nm", "Channel / gate length L (nm)", "100"),
    ("W_um", "Channel width W (um)", "10"),
    ("Na_cm3", "Body doping Na (cm^-3)", "5e16"),
    ("T_K", "Temperature start (K)", "300"),
    ("T_stop_K", "Temperature end (K, blank = start)", ""),
    ("T_step_K", "Temperature step (K)", "25"),
    ("mu_n", "Electron mobility mu_n (cm^2/Vs)", "300"),
    ("tox_nm", "Oxide thickness tox (nm)", "5"),
    ("phi_m_eV", "Gate work function (eV)", "4.5"),
    ("Vth_override", "Vth override (V, blank = computed)", ""),
    ("lam", "Chan.-length modulation lam (1/V)", "0"),
    ("Hfin_nm", "Fin height Hfin (nm)", "30"),
    ("Wfin_nm", "Fin width Wfin (nm)", "8"),
    ("Nfin", "Number of fins Nfin", "3"),
    ("Lsd_nm", "S/D length (nm) (3D drawing)", "100"),
    ("Xj_nm", "Junction depth Xj (nm) (3D drawing)", "30"),
    ("Nsd_cm3", "S/D doping Nsd (cm^-3)", "1e20"),
    ("Vdd_V", "Vgs / Vds max (V)", "3"),
    ("n_vgs", "Number of Vgs curves", "6"),
    ("Vds_read_V", "Vds for Id-Vgs, gm-Vgs curves (V)", "3"),
    ("Vgs_min_V", "Vgs sweep start (V)", "0"),
    ("Vgs_max_V", "Vgs sweep end (V, blank = max)", ""),
    ("Vgs_step_V", "Vgs step (V, blank = automatic)", ""),
    ("Vds_min_V", "Vds sweep start (V)", "0"),
    ("Vds_max_V", "Vds sweep end (V, blank = max)", ""),
    ("Vds_step_V", "Vds step (V, blank = automatic)", ""),
    ("n_vds", "Number of Id-Vgs / SS drain biases", "2"),
    ("Icc_A", "DIBL reference current (A; per fin for 3D)", "1e-7"),
]

# values that differ from the planar defaults when "FinFET" is selected
FINFET_PRESET = {"L_nm": "20", "Na_cm3": "1e16", "mu_n": "200", "tox_nm": "1.2",
                 "phi_m_eV": "4.6", "lam": "0.1", "Lsd_nm": "30",
                 "Vdd_V": "1", "n_vgs": "5", "Vds_read_V": "1"}

FINFET_NODES = {
    "N3A": {"L_nm": "15", "Hfin_nm": "55", "Wfin_nm": "5", "tox_nm": "0.7"},
    "N7A": {"L_nm": "15", "Hfin_nm": "50", "Wfin_nm": "7", "tox_nm": "0.8"},
    "N15A": {"L_nm": "15", "Hfin_nm": "40", "Wfin_nm": "10", "tox_nm": "0.9"},
}
FINFET_FIXED = {"Lsd_nm": "30", "Nfin": "1"}
FINFET_SIZE_KEYS = tuple(FINFET_NODES["N15A"]) + tuple(FINFET_FIXED)

_POSITIVE = ("L_nm", "W_um", "Na_cm3", "T_K", "mu_n", "tox_nm", "Vdd_V",
             "Vds_read_V", "Hfin_nm", "Wfin_nm", "Nfin", "Lsd_nm", "Xj_nm", "Nsd_cm3")


def preset_values(device, finfet_node="N15A", oxide="SiO2"):
    """Dict key -> default string for the chosen device."""
    vals = {k: d for k, _, d in INPUTS}
    if device == DEV_FINFET:
        if finfet_node not in FINFET_NODES:
            raise ValueError("FinFET node must be N3A, N7A or N15A")
        vals.update(FINFET_PRESET)
        vals.update(FINFET_FIXED)
        vals.update(FINFET_NODES[finfet_node])
        if oxide not in OXIDE_KAPPA:
            raise ValueError("Select a supported oxide material")
        thickness = float(vals["tox_nm"]) * OXIDE_KAPPA[oxide] / OXIDE_KAPPA["SiO2"]
        vals["tox_nm"] = f"{thickness:.12g}"
    return vals


def parse_inputs(raw, device, oxide, gate):
    """raw: dict key -> string. Returns validated dict of floats (+ choices)."""
    if oxide not in OXIDE_KAPPA:
        raise ValueError("Select a supported oxide material")
    if device == DEV_FINFET:
        node = raw.get("finfet_node", "N15A")
        sizes = preset_values(device, node, oxide)
        raw = dict(raw, **{key: sizes[key] for key in FINFET_SIZE_KEYS})
    p = {"device": device, "oxide": oxide, "gate": gate}
    if device == DEV_FINFET:
        p["finfet_node"] = node
    optional = {"Vth_override", "T_stop_K", "Vgs_max_V", "Vds_max_V",
                "Vgs_step_V", "Vds_step_V"}
    for key, _, default in INPUTS:
        s = str(raw.get(key, default)).strip()
        p[key] = (float(s) if s else None) if key in optional else float(s)
        if p[key] is not None and not np.isfinite(p[key]):
            raise ValueError(f"{key} must be finite")
    p["n_vgs"] = max(1, min(12, int(round(p["n_vgs"]))))
    p["n_vds"] = int(p["n_vds"])
    if not 1 <= p["n_vds"] <= 12:
        raise ValueError("Number of drain biases must be between 1 and 12")
    p["Nfin"] = max(1, int(round(p["Nfin"])))
    for k in _POSITIVE:
        if p[k] <= 0:
            raise ValueError(f"{k} must be > 0")
    if p["lam"] < 0:
        raise ValueError("lam must be >= 0")
    if p["Icc_A"] <= 0:
        raise ValueError("DIBL reference current must be > 0")
    stop = p["T_K"] if p["T_stop_K"] is None else p["T_stop_K"]
    p["temperatures"] = sweep_values(p["T_K"], stop, p["T_step_K"], 20)
    for bias in ("Vgs", "Vds"):
        start = p[f"{bias}_min_V"]
        stop = p["Vdd_V"] if p[f"{bias}_max_V"] is None else p[f"{bias}_max_V"]
        if start < 0 or stop <= start:
            raise ValueError(f"{bias} requires 0 <= start < end")
        step = p[f"{bias}_step_V"]
        if step is None:
            points = p["n_vgs"] + 1 if device == DEV_FINFET else 121
            p[f"{bias}_sweep"] = np.linspace(start, stop, points)
        else:
            p[f"{bias}_sweep"] = sweep_values(start, stop, step, 2001)
    return p


def sweep_values(start, stop, step, limit):
    if step <= 0 or stop < start:
        raise ValueError("Sweep step must be positive and end must be >= start")
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    if count > limit:
        raise ValueError(f"Sweep has too many points (maximum {limit})")
    values = start + step * np.arange(count)
    if not np.isclose(values[-1], stop, rtol=1e-10, atol=1e-12):
        if count == limit:
            raise ValueError(f"Sweep has too many points (maximum {limit})")
        values = np.append(values, stop)
    else:
        values[-1] = stop
    return values


# fields shown on the input page for each model
_SWEEPS = ("T_K", "T_stop_K", "T_step_K", "Vgs_min_V", "Vgs_max_V", "Vgs_step_V",
           "Vds_min_V", "Vds_max_V", "Vds_step_V", "n_vds", "Icc_A")
_COMMON = ("L_nm", "Na_cm3", "T_K", "mu_n", "tox_nm", "phi_m_eV", "Vth_override",
           "lam", "Lsd_nm", "Vdd_V", "n_vgs", "Vds_read_V")
DEVICE_KEYS = {DEV_MOSFET: _COMMON + ("W_um", "Xj_nm") + _SWEEPS,
               DEV_FINFET: ("L_nm", "Na_cm3", "mu_n", "tox_nm", "phi_m_eV", "Lsd_nm",
                            "Vdd_V", "n_vgs", "Hfin_nm", "Wfin_nm", "Nsd_cm3") + _SWEEPS}


def visible_keys(device):
    """Ordered list of fields used by the selected solver."""
    keys = [k for k, _, _ in INPUTS if k in DEVICE_KEYS[device]] + ["oxide"]
    if device == DEV_FINFET:
        keys.insert(0, "finfet_node")
    return keys
