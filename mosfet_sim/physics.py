"""Developer-editable source for all simulation constants, properties, and equations.

Electrical model: derive_device_common, device widths/thresholds, short-channel
terms, drain_current, transconductance, and output_conductance.
Thermal model: material properties, air_properties, flat_plate_convection,
rds_on, assemble_thermal, and run_thermal.

Change this module to modify the physics. GUI inputs, plotting, package geometry,
mesh generation, and simulation orchestration remain in their own modules.
"""
import numpy as np


# Fundamental and silicon constants.
Q, KB, E0 = 1.602e-19, 1.381e-23, 8.854e-12
HBAR, M0 = 1.054571817e-34, 9.1093837e-31
ESI = 11.7 * E0
CHI_SI, EG_SI = 4.05, 1.12
OXIDE_KAPPA = {"SiO2": 3.9, "Si3N4": 7.5, "Al2O3": 9.0,
               "HfO2": 22.0, "ZrO2": 25.0, "La2O3": 27.0}

# Thermal material IDs, conductivity (W/m K), and volumetric heat capacity (J/m^3 K).
MOLD, CU, SOLDER, SI = 1, 2, 3, 4
MAT_K = np.array([0.0, 1.0, 390.0, 55.0, 130.0])
MAT_RC = np.array([0.0, 1900 * 900, 8960 * 385, 7400 * 220, 2330 * 700.0])

# Package/thermal assumptions.
H_NAT = 10.0
R_CONTACT = 0.5e-4
SINK_V_GAIN = 0.7
MAX_CELLS = 2600
Z_RUNAWAY_C = 800.0

# Nominal power-package geometry in mm; boxes are x0, x1, y0, y1, z0, z1.
PACKAGES = {
    "TO-247": dict(
        mold=(0, 15.9, 0, 16.0, 0, 5.0),
        tab=(0, 15.9, 3.0, 20.9, 0, 1.6),
        die_xy=(5.45, 10.45, 7.0, 12.0), solder_t=0.05, die_t=0.35,
        lead_x=[2.5, 7.95, 13.4], lead_w=1.2, lead_t=0.6,
        lead_y=(-15.0, 0.0), lead_z0=2.0),
    "D2PAK": dict(
        mold=(0, 10.2, 0, 9.0, 0, 4.4),
        tab=(0.6, 9.6, 0, 11.0, 0, 1.3),
        die_xy=(3.1, 7.1, 2.5, 6.5), solder_t=0.05, die_t=0.30,
        lead_x=[2.56, 5.1, 7.64], lead_w=0.8, lead_t=0.5,
        lead_y=(-3.5, 0.0), lead_z0=1.0),
    "DPAK": dict(
        mold=(0, 6.6, 0, 6.1, 0, 2.3),
        tab=(0.6, 6.0, 0, 9.6, 0, 0.9),
        die_xy=(2.0, 4.6, 1.7, 4.3), solder_t=0.04, die_t=0.25,
        lead_x=[1.02, 5.58], lead_w=0.8, lead_t=0.5,
        lead_y=(-2.5, 0.0), lead_z0=0.6),
}


# ---------------------------------------------------------------- electrical device physics
def derive_device_common(model, p):
    model.kappa = OXIDE_KAPPA[p["oxide"]]
    temperature = p["T_K"]
    model.Vt = KB * temperature / Q
    model.ni = 1.45e10 * (temperature / 300) ** 1.5 * np.exp(
        -(EG_SI * Q / (2 * KB)) * (1 / temperature - 1 / 300))
    model.Cox = model.kappa * E0 / (p["tox_nm"] * 1e-9)
    model.phiF = model.Vt * np.log(p["Na_cm3"] / model.ni)
    if model.phiF <= 0:
        raise ValueError("Na must be larger than ni(T)")
    model.Na_m3 = p["Na_cm3"] * 1e6
    model.gamma = np.sqrt(2 * Q * ESI * model.Na_m3) / model.Cox
    model.phi_ms = p["phi_m_eV"] - (CHI_SI + EG_SI / 2 + model.phiF)
    model.kprime = p["mu_n"] * 1e-4 * model.Cox
    model.EOT_nm = p["tox_nm"] * 3.9 / model.kappa
    model.Vbi = model.Vt * np.log(p["Nsd_cm3"] * p["Na_cm3"] / model.ni ** 2)


def planar_effective_width(p):
    return p["W_um"] * 1e-6


def planar_threshold_long(model):
    return model.phi_ms + 2 * model.phiF + model.gamma * np.sqrt(2 * model.phiF)


def finfet_effective_width(model):
    p = model.p
    height, width, count = p["Hfin_nm"] * 1e-9, p["Wfin_nm"] * 1e-9, p["Nfin"]
    return count * (2 * height + width) if model.n_gates == 3 else count * 2 * height


def finfet_threshold_long(model):
    width = model.p["Wfin_nm"] * 1e-9
    depletion_max = np.sqrt(4 * ESI * model.phiF / (Q * model.Na_m3))
    model.wd = min(width / 2, depletion_max)
    model.fully_depleted = model.wd < depletion_max
    model.quantum_vth_shift = np.pi ** 2 * HBAR ** 2 / (2 * 0.19 * M0 * width ** 2 * Q)
    return (model.phi_ms + 2 * model.phiF + Q * model.Na_m3 * model.wd / model.Cox
            + model.quantum_vth_shift)


def finfet_natural_length(model):
    eox = model.kappa * E0
    tsi, tox = model.p["Wfin_nm"] * 1e-9, model.p["tox_nm"] * 1e-9
    return np.sqrt(ESI / (model.n_gates * eox)
                   * (1 + eox * tsi / (4 * ESI * tox)) * tsi * tox)


def finfet_short_channel_terms(model, length_m):
    ratio = np.exp(-np.asarray(length_m, float) / (2 * finfet_natural_length(model)))
    threshold_shift = 2 * max(model.Vbi - 2 * model.phiF, 0.0) * ratio
    return threshold_shift, ratio


def threshold_voltage(model, vds):
    return model.vth_long - model.dsce - model.eta * np.asarray(vds, float)


def threshold_at_length(model, length_m, vds):
    shift, dibl = model.sce_terms(np.asarray(length_m, float))
    return model.vth_long - shift - dibl * vds


def _bias(model, vgs, vds):
    vgs, vds = np.broadcast_arrays(np.asarray(vgs, float), np.asarray(vds, float))
    return vgs, vds, vgs - model.vth(vds)


def _regions(overdrive, vds):
    on = overdrive > 0
    triode = on & (vds < overdrive)
    return triode, on & ~triode


def _effective_overdrive(model, overdrive):
    """Soft inversion charge with a depletion-capacitance slope approximation."""
    slope_factor = 1 + model.gamma / (2 * np.sqrt(2 * model.phiF))
    width = 2 * slope_factor * model.Vt
    normalized = overdrive / width
    effective = width * np.logaddexp(0, normalized)
    derivative = np.exp(-np.logaddexp(0, -normalized))
    return effective, derivative


def drain_current(model, vgs, vds):
    vgs, vds, overdrive = _bias(model, vgs, vds)
    positive_overdrive, _ = _effective_overdrive(model, overdrive)
    effective_vds = np.minimum(vds, positive_overdrive)
    current = model.beta * (positive_overdrive * effective_vds - effective_vds ** 2 / 2)
    current *= 1 + model.lam * np.maximum(vds - positive_overdrive, 0)
    return current


def transconductance(model, vgs, vds):
    _, vds, overdrive = _bias(model, vgs, vds)
    overdrive, derivative = _effective_overdrive(model, overdrive)
    beta, lam = model.beta, model.lam
    triode, saturation = _regions(overdrive, vds)
    g_sat = (beta * overdrive * (1 + lam * (vds - overdrive))
             - 0.5 * beta * lam * overdrive ** 2)
    return np.where(triode, beta * vds, np.where(saturation, g_sat, 0.0)) * derivative


def output_conductance(model, vgs, vds):
    _, vds, overdrive = _bias(model, vgs, vds)
    overdrive, derivative = _effective_overdrive(model, overdrive)
    beta, lam, eta = model.beta, model.lam, model.eta
    eta *= derivative
    triode, saturation = _regions(overdrive, vds)
    g_triode = beta * (overdrive - vds + eta * vds)
    g_sat = (beta * overdrive * eta * (1 + lam * (vds - overdrive))
             + 0.5 * beta * lam * overdrive ** 2 * (1 - eta))
    return np.where(triode, g_triode, np.where(saturation, g_sat, 0.0))


# ---------------------------------------------------------------- thermal material and heat-transfer physics
def air_properties(t_c):
    temperature_k = t_c + 273.15
    conductivity = 0.0242 + 7.2e-5 * t_c
    viscosity = 1.5e-5 * (temperature_k / 293.15) ** 1.7
    return conductivity, viscosity, 0.71


def flat_plate_convection(downstream_m, air_k, prandtl, velocity, nu):
    return (0.332 * air_k * prandtl ** (1 / 3) * np.sqrt(velocity / nu)
            / np.sqrt(np.maximum(downstream_m, 1e-3)))


def rds_on(p, temperature_c):
    return (p["Rds25_mOhm"] * 1e-3
            * ((temperature_c + 273.15) / 298.15) ** p["rds_exp"])


def assemble_thermal(model):
    p, (xe, ye, ze), mat = model.p, model.edges, model.mat
    solid = model.solid
    idx = model.idx
    dx, dy, dz = (np.diff(edge) for edge in (xe, ye, ze))
    DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing="ij")
    CX, CY, CZ = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2,
                             (ze[:-1] + ze[1:]) / 2, indexing="ij")
    LEN, CEN = (DX, DY, DZ), (CX, CY, CZ)
    AREA = (DY * DZ, DX * DZ, DX * DY)
    conductivity = MAT_K[mat]
    matrix = np.zeros((model.N + 1, model.N + 1))
    ambient = p["T_amb_C"]

    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(0, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        both = solid[left] & solid[right]
        with np.errstate(divide="ignore", invalid="ignore"):
            resistance = LEN[axis][left] / (2 * conductivity[left]) \
                + LEN[axis][right] / (2 * conductivity[right])
            conductance = np.where(both, AREA[axis][left] / resistance, 0.0)[both]
        i_left, i_right = idx[left][both], idx[right][both]
        np.add.at(matrix, (i_left, i_left), conductance)
        np.add.at(matrix, (i_right, i_right), conductance)
        np.add.at(matrix, (i_left, i_right), -conductance)
        np.add.at(matrix, (i_right, i_left), -conductance)

    flow_axis, flow_sign, velocity = p["flow_axis"], p["flow_sign"], p["air_v"]
    air_k, nu, prandtl = air_properties(ambient)
    leading_edge = xe[0] if flow_axis == 0 and flow_sign > 0 else \
        xe[-1] if flow_axis == 0 else ye[0] if flow_sign > 0 else ye[-1]
    flow_extent = (xe[-1] - xe[0]) if flow_axis == 0 else (ye[-1] - ye[0])
    h_average = 2 * flat_plate_convection(flow_extent, air_k, prandtl, velocity, nu)
    mount = np.zeros(mat.shape, bool)
    ambient_load = np.zeros(model.N + 1)

    for axis in range(3):
        for sign in (-1, 1):
            neighbor = np.zeros_like(solid)
            if sign < 0:
                dst = tuple(slice(1, None) if i == axis else slice(None) for i in range(3))
                src = tuple(slice(0, -1) if i == axis else slice(None) for i in range(3))
            else:
                dst = tuple(slice(0, -1) if i == axis else slice(None) for i in range(3))
                src = tuple(slice(1, None) if i == axis else slice(None) for i in range(3))
            neighbor[dst] = solid[src]
            exposed = solid & ~neighbor
            if axis == 2 and sign < 0:
                mount = exposed & (mat == CU)
                exposed &= ~mount
            if axis == flow_axis:
                forced_h = np.full(mat.shape, 1.2 * h_average if sign * flow_sign < 0
                                   else 0.5 * h_average)
            else:
                forced_h = flat_plate_convection(np.abs(CEN[flow_axis] - leading_edge),
                                                 air_k, prandtl, velocity, nu)
                if axis == 2 and sign < 0:
                    forced_h *= 0.5
            h = (forced_h ** 3 + H_NAT ** 3) ** (1 / 3)
            if axis == 2 and sign < 0:
                h *= 0.5
            exposed_conductance = AREA[axis] / (1 / h + LEN[axis] / (2 * conductivity + 1e-12))
            exposed_idx = idx[exposed]
            np.add.at(matrix, (exposed_idx, exposed_idx), exposed_conductance[exposed])
            np.add.at(ambient_load, exposed_idx, exposed_conductance[exposed] * ambient)

    sink = model.N
    mount_idx = idx[mount]
    mount_conductance = (AREA[2] / (R_CONTACT + LEN[2] / (2 * conductivity + 1e-12)))[mount]
    np.add.at(matrix, (mount_idx, mount_idx), mount_conductance)
    matrix[sink, sink] += mount_conductance.sum()
    np.add.at(matrix, (mount_idx, sink), -mount_conductance)
    np.add.at(matrix, (sink, mount_idx), -mount_conductance)
    sink_conductance = (1 + SINK_V_GAIN * np.sqrt(velocity)) / p["Rsa_KW"]
    matrix[sink, sink] += sink_conductance
    ambient_load[sink] += sink_conductance * ambient

    volume = DX * DY * DZ
    model.C = np.append((MAT_RC[mat] * volume)[solid], p["Csink_JK"])
    model.die = idx[mat == SI]
    model.die_w = volume[mat == SI] / volume[mat == SI].sum()
    model.mount_idx, model.mount_w = mount_idx, mount_conductance / mount_conductance.sum()
    model.K, model.b0 = matrix, ambient_load


def run_thermal(model):
    p, cell_count = model.p, model.N
    ambient, current_squared = p["T_amb_C"], p["Id_A"] ** 2
    duration = p["duration_s"]
    fast_steps = int(min(50, duration / 0.02))
    remaining = duration - fast_steps * 0.02
    slow_steps = int(np.ceil(remaining / 0.25 - 1e-9)) if remaining > 1e-9 else 0
    segments = [(0.02, fast_steps)] + \
        ([(remaining / slow_steps, slow_steps)] if slow_steps else [])
    temperature = np.full(cell_count + 1, ambient)
    time = 0.0
    times, frames, powers, runaway = [0.0], [temperature.astype(np.float32)], [], False
    powers.append(current_squared * rds_on(p, ambient))
    power = powers[0]
    heat_distribution = np.zeros(cell_count + 1)
    heat_distribution[model.die] = model.die_w

    for dt, step_count in segments:
        capacity_over_dt = model.C / dt
        inverse = np.linalg.inv(np.diag(capacity_over_dt) + model.K)
        response = inverse @ heat_distribution
        die_response, mean_response = response[model.die], response[model.die] @ model.die_w
        for _ in range(step_count):
            base_temperature = inverse @ (capacity_over_dt * temperature + model.b0)
            base_die_mean = base_temperature[model.die] @ model.die_w
            for _ in range(40):
                new_power = current_squared * rds_on(p, min(base_die_mean + power * mean_response,
                                                             1500.0))
                if abs(new_power - power) <= 1e-7 * max(power, 1.0):
                    break
                power = new_power
            temperature = base_temperature + power * response
            time += dt
            times.append(time)
            frames.append(temperature.astype(np.float32))
            powers.append(power)
            if temperature[model.die].max() > Z_RUNAWAY_C:
                runaway = True
                times.pop()
                frames.pop()
                powers.pop()
                break
        if runaway:
            break

    frame_array = np.array(frames)
    die_mean = frame_array[:, model.die] @ model.die_w
    return {"t": np.array(times), "T": frame_array, "P": np.array(powers),
            "Tj_mean": die_mean, "Tj_max": frame_array[:, model.die].max(1),
            "Tcase": frame_array[:, model.mount_idx] @ model.mount_w,
            "Tsink": frame_array[:, cell_count], "Rds_mOhm": rds_on(p, die_mean) * 1e3,
            "runaway": runaway}