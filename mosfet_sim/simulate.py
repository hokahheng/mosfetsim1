"""Builds every data set the plot pages need (no plotting, no GUI)."""
import numpy as np


def subthreshold_swing(vgs, current):
    """Return 1000 / (d log10(|Id|) / dVgs), in mV/dec along axis 0."""
    vgs = np.asarray(vgs, dtype=float)
    current = np.abs(np.asarray(current, dtype=float))
    result = np.full_like(current, np.nan)
    if vgs.size < 3:
        return result
    valid = np.isfinite(current) & (current > 0)
    logarithm = np.full_like(current, np.nan)
    np.log10(current, out=logarithm, where=valid)
    slope = np.gradient(logarithm, vgs, axis=0, edge_order=2)
    np.divide(1000.0, slope, out=result, where=np.isfinite(slope) & (slope > 0))
    return result


def constant_current_dibl(vgs, vds, current, reference):
    """Extract log-interpolated thresholds at one common current and DIBL."""
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Reference current must be finite and positive")
    thresholds = np.full(len(vds), np.nan)
    for index, drain in enumerate(vds):
        curve = np.asarray(current)[:, index]
        valid = np.isfinite(curve) & (curve > 0)
        if drain <= 0 or valid.sum() < 2:
            continue
        gate = np.asarray(vgs)
        log_id = np.full_like(curve, np.nan, dtype=float)
        np.log10(curve, out=log_id, where=valid)
        crossings = np.flatnonzero((log_id[:-1] <= np.log10(reference)) &
                                  (log_id[1:] >= np.log10(reference)) &
                                  (np.diff(log_id) > 0))
        if crossings.size:
            lower = crossings[0]
            thresholds[index] = np.interp(np.log10(reference), log_id[lower:lower + 2],
                                          gate[lower:lower + 2])
    valid = np.flatnonzero(np.isfinite(thresholds))
    dibl = np.nan
    if valid.size >= 2:
        low, high = valid[0], valid[-1]
        dibl = 1000 * (thresholds[low] - thresholds[high]) / (vds[high] - vds[low])
    return thresholds, dibl


def simulate(m, p):
    vdd, vr = p["Vdd_V"], p["Vds_read_V"]
    vx = np.asarray(p.get("Vgs_sweep", np.linspace(0, vdd, 121)))
    vd = np.asarray(p.get("Vds_sweep", np.linspace(0, vdd, 121)))
    vgs_list = np.linspace(vx[0], vx[-1], p["n_vgs"] + 1)[1:]
    positive_vds = vd[vd > 0]
    vds_list = np.linspace(positive_vds[0], positive_vds[-1], p.get("n_vds", 2))
    if p.get("n_vds", 2) == 1:
        vds_list = np.array([min(max(vr, positive_vds[0]), positive_vds[-1])])
    transfer = m.id(vx[:, None], vds_list[None, :])
    vth, dibl = constant_current_dibl(vx, vds_list, transfer, p.get("Icc_A", 1e-7))
    r = {"vx": vx, "vd": vd, "vgs_list": vgs_list, "vds_list": vds_list,
         "transfer": transfer, "ss": subthreshold_swing(vx, transfer),
         "Vth": vth, "DIBL_mV_per_V": dibl}
    r["id_vds"] = [m.id(vg, vd) for vg in vgs_list]
    r["id_vgs"] = m.id(vx, vr)
    r["gm_vgs"] = m.gm(vx, vr)
    r["gm_vgs_num"] = np.gradient(r["id_vgs"], vx)
    r["gm_vds"] = [m.gm(vg, vd) for vg in vgs_list]
    r["go_vds"] = [m.go(vg, vd) for vg in vgs_list]
    r["go_vds_num"] = [np.gradient(c, vd) for c in r["id_vds"]]
    r["go_vgs"] = m.go(vx, vr)
    return r


def sce_sweep(m, p):
    """Vth and DIBL versus gate length (0.25 L ... 4 L)."""
    Ls = np.geomspace(0.25 * p["L_nm"], 4 * p["L_nm"], 60)
    vhi = p["Vdd_V"]
    vlin = min(0.05, 0.1 * vhi)
    zero = np.zeros_like(Ls)
    vth_lin = m.vth_at(Ls * 1e-9, vlin) + zero
    vth_hi = m.vth_at(Ls * 1e-9, vhi) + zero
    return {"L_nm": Ls, "vlin": vlin, "vhi": vhi, "vth_lin": vth_lin,
            "vth_hi": vth_hi, "dibl": (vth_lin - vth_hi) / (vhi - vlin) * 1e3}


def operating_check(m, p):
    """gm / go at Vgs = Vdd, Vds = Vds_read: closed form vs numerical derivative."""
    vg, vd, h = p["Vdd_V"], p["Vds_read_V"], 1e-4
    vov = vg - float(m.vth(vd))
    i = float(m.id(vg, vd))
    lo = max(vd - h, 0)
    return {"Id": i,
            "gm_formula": float(m.gm(vg, vd)),
            "gm_num": float((m.id(vg + h, vd) - m.id(vg - h, vd)) / (2 * h)),
            "gm_2Id_Vov": 2 * i / vov if vov > 0 else 0.0,
            "go_formula": float(m.go(vg, vd)),
            "go_num": float((m.id(vg, vd + h) - m.id(vg, lo)) / (vd + h - lo))}
