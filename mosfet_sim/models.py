"""Device model classes and summaries.

Physical equations are implemented in physics.py. Add a device by subclassing
DeviceModel and registering it in MODELS. Each model supplies effective width,
long-channel Vth and short-channel terms (dsce, eta) so that

    Vth(Vds, L) = vth_long - dsce(L) - eta(L) * Vds        (eta = DIBL coefficient)
"""
import numpy as np
from .inputs import GATE_TYPES
from . import equations as eq
from . import physics


class DeviceModel:
    label = "Device"

    def __init__(self, p):
        self.p = p
        self.L = p["L_nm"] * 1e-9
        self._derive_common()
        self.width = self._width()
        self.beta = self.kprime * self.width / self.L
        self.vth_calc = self._vth_long_calc()
        self.vth_long = self.vth_calc if p["Vth_override"] is None else p["Vth_override"]
        self.lam = p["lam"]
        self.dsce, self.eta = self.sce_terms(self.L)

    # ---- shared derived quantities
    def _derive_common(self):
        physics.derive_device_common(self, self.p)

    # ---- hooks for subclasses
    def _width(self):
        raise NotImplementedError

    def _vth_long_calc(self):
        raise NotImplementedError

    def sce_terms(self, L):
        """(dsce [V], eta [V/V]) for gate length L [m] (scalar or array)."""
        return 0.0, 0.0

    def ann_features(self):
        return {}

    def extra_summary(self):
        return []

    # ---- electrical interface (also implemented by ann.HybridModel)
    def id(self, vgs, vds):
        return eq.drain_current(self, vgs, vds)

    def gm(self, vgs, vds):
        return eq.gm(self, vgs, vds)

    def go(self, vgs, vds):
        return eq.go(self, vgs, vds)

    # ---- threshold voltage
    def vth(self, vds):
        return physics.threshold_voltage(self, vds)

    def vth_at(self, L, vds):
        """Vth for another gate length L [m] (used for the SCE / DIBL sweep)."""
        return physics.threshold_at_length(self, L, vds)

    @property
    def dibl_mV_per_V(self):
        return float(self.eta) * 1e3

    def summary(self):
        s = [f"Device    = {self.label}",
             f"Vt        = {self.Vt:.4f} V",
             f"ni        = {self.ni:.3e} cm^-3",
             f"Cox       = {self.Cox*1e2:.4g} uF/cm^2",
             f"EOT       = {self.EOT_nm:.3g} nm",
             f"phiF      = {self.phiF:.4f} V",
             f"phi_ms    = {self.phi_ms:.4f} V"]
        s += self.extra_summary()
        s += [f"Vth(long) = {self.vth_long:.4f} V",
              f"Vth(Vds={self.p['Vds_read_V']:g}) = {float(self.vth(self.p['Vds_read_V'])):.4f} V",
              f"Weff      = {self.width*1e6:.4g} um",
              f"k'=mu*Cox = {self.kprime*1e6:.4g} uA/V^2",
              f"beta=k'W/L= {self.beta*1e3:.4g} mA/V^2"]
        return s


class PlanarMOSFET(DeviceModel):
    """Long-channel planar MOSFET with approximate smooth weak inversion; no SCE."""
    label = "2D MOSFET"

    def _width(self):
        return physics.planar_effective_width(self.p)

    def _vth_long_calc(self):
        return physics.planar_threshold_long(self)

    def extra_summary(self):
        return [f"gamma     = {self.gamma:.4f} V^0.5"]


class FinFET(DeviceModel):
    """FinFET with compact short-channel / DIBL model.

    Weff = Nfin (2 Hfin + Wfin)  (tri-gate)  or  Nfin 2 Hfin  (double-gate)
    Vth0 = phi_ms + 2 phiF + q Na Wd / Cox,  Wd = min(Wfin/2, Wdep_max)
    Natural length (compact analytical approximation)
        lam_n = sqrt[ eSi/(Ng eox) (1 + eox Wfin/(4 eSi tox)) Wfin tox ]
    Short-channel terms, x = exp(-L / (2 lam_n))
        dVth_SCE = 2 (Vbi - 2 phiF) x        (threshold roll-off)
        eta      = x                         (DIBL: dVth = -eta Vds)
    Quantum correction: first subband energy for an infinite well across Wfin,
        dVth_q = pi^2 hbar^2 / (2 m* Wfin^2 q), with m* = 0.19 m0.
    """
    label = "FinFET"

    def __init__(self, p):
        self.n_gates = GATE_TYPES[p["gate"]]
        super().__init__(p)

    def _width(self):
        return physics.finfet_effective_width(self)

    def _vth_long_calc(self):
        return physics.finfet_threshold_long(self)

    @property
    def lam_nat(self):
        return physics.finfet_natural_length(self)

    def sce_terms(self, L):
        return physics.finfet_short_channel_terms(self, L)

    def extra_summary(self):
        return [f"gates     = {self.n_gates}  ({self.p['gate']})",
                f"fully depl= {self.fully_depleted}",
                f"Q shift   = {self.quantum_vth_shift*1e3:.2f} mV (1D well, m*=0.19m0)",
                f"lam_nat   = {self.lam_nat*1e9:.3f} nm",
                f"L/lam_nat = {self.L/self.lam_nat:.2f}",
                f"Vbi       = {self.Vbi:.3f} V",
                f"dVth(SCE) = {-float(self.dsce)*1e3:.2f} mV",
                f"DIBL      = {self.dibl_mV_per_V:.2f} mV/V"]

    def ann_features(self):
        return {"Hfin_nm": self.p["Hfin_nm"], "Wfin_nm": self.p["Wfin_nm"],
                "Nfin": self.p["Nfin"], "n_gates": self.n_gates,
                "lam_nat_nm": self.lam_nat * 1e9,
                "dVth_sce": float(self.dsce), "eta_dibl": float(self.eta)}


MODELS = {"2D MOSFET": PlanarMOSFET, "FinFET": FinFET}


def build_model(p):
    return MODELS[p["device"]](p)
