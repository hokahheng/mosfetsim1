"""Inputs, presets and validation for the power-MOSFET package thermal tool."""

THERMAL_PACKAGES = ["TO-247", "D2PAK", "DPAK"]
# name -> (flow axis index 0=x 1=y, sign)
FLOW_DIRS = {"+y  (leads -> tab end)": (1, 1), "-y  (tab end -> leads)": (1, -1),
             "+x  (left -> right)": (0, 1), "-x  (right -> left)": (0, -1)}
MAX_DURATION_S = 180.0

# (key, label, default)
THERMAL_INPUTS = [
    ("T_amb_C", "Ambient temperature (C)", "25"),
    ("air_v", "Air speed (m/s, 0 = natural)", "1.0"),
    ("Id_A", "Drain current Id, DC (A)", "40"),
    ("Rds25_mOhm", "Rds(on) at 25 C (mOhm)", "25"),
    ("rds_exp", "Rds(on) temp. exponent n  (R ~ T^n)", "2.3"),
    ("Rsa_KW", "Sink / PCB Rth to ambient, still air (K/W)", "1.5"),
    ("Csink_JK", "Sink / PCB heat capacity (J/K)", "90"),
    ("duration_s", f"Total simulation time (s, max {MAX_DURATION_S:g})", "180"),
    ("mesh_mm", "Mesh size (mm, smaller = slower)", "1.5"),
    ("Tj_max_C", "Tj limit for warning (C)", "175"),
]
PACKAGE_PRESET = {
    "TO-247": {"Id_A": "40", "Rds25_mOhm": "25", "Rsa_KW": "1.5", "Csink_JK": "90"},
    "D2PAK": {"Id_A": "14", "Rds25_mOhm": "20", "Rsa_KW": "30", "Csink_JK": "12"},
    "DPAK": {"Id_A": "7", "Rds25_mOhm": "40", "Rsa_KW": "45", "Csink_JK": "8"},
}


def thermal_preset(package):
    vals = {k: d for k, _, d in THERMAL_INPUTS}
    vals.update(PACKAGE_PRESET[package])
    return vals


def parse_thermal(raw, package, flow_name):
    p = {"package": package, "flow_name": flow_name}
    p["flow_axis"], p["flow_sign"] = FLOW_DIRS[flow_name]
    for key, _, _ in THERMAL_INPUTS:
        p[key] = float(str(raw[key]).strip())
    for k in ("Id_A", "Rds25_mOhm", "rds_exp", "Rsa_KW", "Csink_JK", "duration_s", "mesh_mm"):
        if p[k] <= 0:
            raise ValueError(f"{k} must be > 0")
    if p["air_v"] < 0:
        raise ValueError("air_v must be >= 0")
    if p["duration_s"] > MAX_DURATION_S:
        raise ValueError(f"Duration is limited to {MAX_DURATION_S:g} s (3 minutes)")
    if p["mesh_mm"] < 0.5:
        raise ValueError("mesh size must be >= 0.5 mm")
    return p
