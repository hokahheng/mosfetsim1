#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quasi-3D FinFET (N15A-like)  --  Python / CuPy (RTX GPU) port of finfet_3d.m
=============================================================================
SAME PHYSICS AS THE MATLAB FILE (v3):

  u1 = psi : div(eps_r grad psi) = -(q/eps0)(p - n + Nd(x) - Na)          (3D Poisson)
  u2 = w   : density-gradient correction, n = w^2   (only if useDG = True)
             2b*lap(w) + w*(psi - phi_n(x) - Vt*ln(w^2/ni)) = 0,  b = gamma*hbar^2/(12 q m*)
  p = ni*exp(-psi/Vt);  classical n = ni*exp((psi - phi_n)/Vt)
  gate stack = IL SiO2 + HfO2  -> Robin BC  eps_Si dpsi/dn + Cq (psi - (Vg - phiRef)) = 0
  (gate only over |x - xc| <= Lg/2, on the top face and the +y side wall; y = 0 is the
   symmetry plane (natural BC), bottom face insulated, S/D faces Dirichlet)
  phi_n(x): 1-D current continuity integrated exactly per cell, Caughey-Thomas velocity
  saturation; Poisson <-> continuity coupled by Gummel iteration with Anderson mixing,
  adaptive damping, step cap, divergence/stagnation guards, automatic bias-step halving.

WHAT IS DIFFERENT (numerics only, not physics):
  * MATLAB PDE Toolbox tetrahedral FEM  ->  structured hexahedral grid, node-centred
    finite-volume / lumped-mass FEM (7-point stencil), boundary integrals done exactly.
  * The linear algebra (Newton Jacobian solves, line density) runs on the GPU via CuPy.
  * Results agree with the MATLAB run only up to discretisation error: validate Vth, SS and
    Ids at 2-3 biases and refine (hx, hy, hz) if needed.

INSTALL (Windows or WSL2, NVIDIA driver already installed):
    pip install numpy scipy matplotlib
    pip install cupy-cuda12x            # (cupy-cuda11x for CUDA 11)

RUN:
    python finfet_3d_gpu.py             # full sweep
    python finfet_3d_gpu.py --test      # quick check (Vgs = 0.8 V, Vds = 0.05 and 0.1 V)
    python finfet_3d_gpu.py --cpu       # force CPU (NumPy/SciPy) - same code path

OUTPUT (in par.outDir):
    voltage current information.csv , 3D.csv (or 3D_npz/*.npz), Ids matrix Vgs x Vds.csv,
    results.npz, partial.npz, *.png
Units: nm, V, densities nm^-3 (same as the MATLAB file).
"""
import os
import sys
import time
import math
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.special import erf

if __package__:
    from .simulate import constant_current_dibl, subthreshold_swing
else:
    from simulate import constant_current_dibl, subthreshold_swing

# ======================================================================================
#  Parameters  (identical values to the MATLAB file; ALL DEVICE NUMBERS ARE ASSUMED)
# ======================================================================================
@dataclass
class Par:
    Lg: float = 16.0
    Lsd: float = 20.0                    # gate length, S/D length [nm]
    Wfin: float = 7.0
    Hfin: float = 30.0                   # fin width / height [nm]
    tIL: float = 0.5
    epsIL: float = 3.9                   # interfacial SiO2 [nm]
    tHf: float = 2.0
    epsHf: float = 22.0                  # HfO2 [nm]
    Na: float = 1e16 * 1e-21             # channel acceptors [nm^-3]
    Nsd: float = 1e20 * 1e-21            # S/D donors        [nm^-3]
    Lj: float = 8.0                      # junction position (= Lg/2, set in __post_init__)
    sigJ: float = 2.0                    # junction abruptness [nm]
    PhiM: float = 4.5                    # gate work function [eV]
    mu0: float = 200.0                   # [cm^2/Vs]
    vsat: float = 1e7                    # [cm/s]
    useDG: bool = False                  # False: classical;  True: quantum (density-gradient)
    gamma: float = 3.6
    mstar: float = 0.26                  # DG fitting factor, confinement mass [m0]

    # ---- structured grid spacing [nm]  (hx is also the x grid of the 1-D current integral)
    hx: float = 0.5
    hy: float = 0.5
    hz: float = 0.5

    # ---- Poisson Newton
    newton_tol: float = 1e-7             # max |dpsi| [V] for convergence
    newton_max: int = 100
    newton_cap: float = 0.5              # max potential step per Newton iteration [V]
    lin_tol: float = 1e-10               # relative tolerance of the linear solves
    lin_max: int = 5000

    # ---- Gummel
    gumTol: float = 1e-3
    gumMax: int = 80
    gumAlpha: float = 0.5
    gumAA: int = 3
    gumCap: float = 0.10
    verbose: bool = True

    # ---- sweep
    vStep: float = 0.1
    vMax: float = 1.5
    T_K: float = 300.0
    vgs_values: tuple = ()
    vds_values: tuple = ()
    Icc_A: float = 0.0
    log3D: str = 'csv'                   # 'csv' (3D.csv, huge), 'npz' (one file per bias, fast) or '' (off)
    test: bool = False
    use_gpu: bool = True
    show_plots: bool = False
    outDir: str = os.path.join('.', 'finfet_out')

    def __post_init__(self):
        self.Lj = self.Lg / 2


# ======================================================================================
#  Backend selection (CuPy on the GPU, NumPy/SciPy otherwise) -- same code for both
# ======================================================================================
xp = np
xsp = sps
GPU = False


def select_backend(use_gpu):
    global xp, xsp, GPU
    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cps
            if cp.cuda.runtime.getDeviceCount() > 0:
                xp, xsp, GPU = cp, cps, True
                prop = cp.cuda.runtime.getDeviceProperties(0)
                print('GPU backend: CuPy on %s' % prop['name'].decode())
                return
        except Exception as e:                        # noqa
            print('CuPy / GPU not usable (%s) -> falling back to CPU' % e)
    xp, xsp, GPU = np, sps, False
    print('CPU backend: NumPy / SciPy')


def to_np(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)


def to_xp(a):
    return xp.asarray(a)


def to_xsp(M):
    return xsp.csr_matrix(M)


class GummelError(Exception):
    pass


# ======================================================================================
#  Linear solvers (own implementations -> independent of CuPy / SciPy version changes)
# ======================================================================================
def pcg(matvec, b, x0, dinv, tol, maxit):
    """Jacobi-preconditioned conjugate gradient (SPD systems)."""
    x = x0.copy()
    r = b - matvec(x)
    z = dinv * r
    pvec = z.copy()
    rz = float(xp.dot(r, z))
    bn = float(xp.linalg.norm(b))
    if bn == 0.0:
        return x, 0, True
    for it in range(maxit):
        if float(xp.linalg.norm(r)) <= tol * bn:
            return x, it, True
        Ap = matvec(pvec)
        alpha = rz / float(xp.dot(pvec, Ap))
        x += alpha * pvec
        r -= alpha * Ap
        z = dinv * r
        rz_new = float(xp.dot(r, z))
        pvec = z + (rz_new / rz) * pvec
        rz = rz_new
    return x, maxit, float(xp.linalg.norm(r)) <= tol * bn


def bicgstab(matvec, b, x0, precond, tol, maxit):
    """Right-preconditioned BiCGStab (non-symmetric systems, used for the DG block system)."""
    x = x0.copy()
    r = b - matvec(x)
    bn = float(xp.linalg.norm(b))
    if bn == 0.0:
        return x, 0, True
    rhat = r.copy()
    rho = alpha = omega = 1.0
    v = xp.zeros_like(b)
    pvec = xp.zeros_like(b)
    for it in range(maxit):
        if float(xp.linalg.norm(r)) <= tol * bn:
            return x, it, True
        rho_new = float(xp.dot(rhat, r))
        if rho_new == 0.0:
            return x, it, False
        beta = (rho_new / rho) * (alpha / omega)
        pvec = r + beta * (pvec - omega * v)
        phat = precond(pvec)
        v = matvec(phat)
        den = float(xp.dot(rhat, v))
        if den == 0.0:
            return x, it, False
        alpha = rho_new / den
        s = r - alpha * v
        if float(xp.linalg.norm(s)) <= tol * bn:
            x += alpha * phat
            return x, it + 1, True
        shat = precond(s)
        t = matvec(shat)
        tt = float(xp.dot(t, t))
        if tt == 0.0:
            return x, it, False
        omega = float(xp.dot(t, s)) / tt
        x += alpha * phat + omega * shat
        r = s - omega * t
        rho = rho_new
        if omega == 0.0:
            return x, it, False
    return x, maxit, float(xp.linalg.norm(r)) <= tol * bn


class DiagCSR:
    """CSR matrix with fixed sparsity whose diagonal can be changed cheaply (Jacobian assembly)."""

    def __init__(self, M_cpu):
        M = sps.csr_matrix(M_cpu)
        M.sum_duplicates()
        M.sort_indices()
        n = M.shape[0]
        rows = np.repeat(np.arange(n), np.diff(M.indptr))
        pos = np.nonzero(rows == M.indices)[0]
        assert pos.size == n, 'every row needs a diagonal entry'
        self.shape = M.shape
        self.data = to_xp(M.data.astype(np.float64))
        self.indices = to_xp(M.indices.astype(np.int32))
        self.indptr = to_xp(M.indptr.astype(np.int32))
        self.pos = to_xp(pos.astype(np.int64))
        self.diag0 = self.data[self.pos].copy()

    def make(self, add):
        d = self.data.copy()
        d[self.pos] += add
        return xsp.csr_matrix((d, self.indices, self.indptr), shape=self.shape)


# ======================================================================================
#  Current continuity (1-D, exact cell-wise integration, Caughey-Thomas saturation)
# ======================================================================================
def march_current(N, Vd, phin_used, dx_cm, Vt, q, mu0, vsat):
    if Vd < 1e-9:
        return 0.0, np.zeros_like(phin_used)
    g = np.maximum(N * 1e7, 1e-300) * np.exp(phin_used / Vt)       # cm^-1, Slotboom-normalised
    g1 = g[:-1]
    g2 = g[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(g1 / g2)
        ge = g1 * g2 * r / (g1 - g2)                               # exact for exponential g
    sm = np.abs(r) < 1e-6
    ge[sm] = 0.5 * (g1[sm] + g2[sm])
    # Same cell equation  I*(dx/mu0 + d/vsat)/q = ge*Vt*(u_k - u_k+1),  u = exp(-phi/Vt),
    # with the saturation drop d lagged one Gummel step (exact at convergence): linear in u,
    # so phi_n(drain) = Vd is hit exactly for any bias instead of rescaling a truncated march.
    d_old = np.abs(np.diff(phin_used))
    R = (dx_cm / mu0 + d_old / vsat) / (q * ge * Vt)
    uD = math.exp(-Vd / Vt)
    I = (1.0 - uD) / R.sum()
    tail = np.concatenate([np.cumsum(R[::-1])[::-1], [0.0]])      # sum_{j>=k} R_j, accurate near drain
    u = uD + I * tail
    phin = -Vt * np.log(u)
    phin[0], phin[-1] = 0.0, Vd
    return I, phin


# ======================================================================================
#  Device: structured grid, matrices, Poisson / DG Newton solvers
# ======================================================================================
def lap1d(n, h):
    main = np.full(n, 2.0 / h)
    main[0] = main[-1] = 1.0 / h
    off = np.full(n - 1, -1.0 / h)
    return sps.diags([off, main, off], [-1, 0, 1], format='csr')


def wts1d(n, h):
    w = np.full(n, h)
    w[0] = w[-1] = h / 2
    return w


class Device:
    def __init__(self, par):
        p = self.par = par
        # ---- constants
        self.q = 1.602e-19
        self.eps0 = 8.854e-21
        kB = 8.617e-5
        if not np.isfinite(p.T_K) or p.T_K <= 0:
            raise ValueError('Temperature must be finite and positive')
        self.Vt = kB * p.T_K
        self.epsSi = 11.7
        chi, Eg = 4.05, 1.12
        self.ni = 1e-11 * (p.T_K / 300.0) ** 1.5 * np.exp(
            -Eg / (2 * kB) * (1 / p.T_K - 1 / 300.0))
        hbar, m0 = 1.0546e-34, 9.109e-31
        self.b = p.gamma * hbar ** 2 / (12 * self.q * p.mstar * m0) * 1e18     # [V nm^2]
        self.Cq = 1.0 / (p.tIL / p.epsIL + p.tHf / p.epsHf)       # [nm^-1] (= Cox/eps0)
        self.phiRef = p.PhiM - (chi + Eg / 2)                      # psi_gate = Vg - phiRef
        self.Ltot = p.Lg + 2 * p.Lsd
        self.Weff = (2 * p.Hfin + p.Wfin) * 1e-3                   # um (tri-gate)
        self.xc = self.Ltot / 2

        # ---- grid  (x: S->D, y: 0 = symmetry plane ... Wfin/2 = gated side wall, z: 0 = bottom ... Hfin = top gate)
        self.nx = int(round(self.Ltot / p.hx)) + 1
        self.ny = max(int(round((p.Wfin / 2) / p.hy)) + 1, 3)
        self.nz = int(round(p.Hfin / p.hz)) + 1
        self.x = np.linspace(0, self.Ltot, self.nx)
        self.y = np.linspace(0, p.Wfin / 2, self.ny)
        self.z = np.linspace(0, p.Hfin, self.nz)
        hx, hy, hz = self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0]
        nx, ny, nz = self.nx, self.ny, self.nz
        self.nyz = ny * nz
        self.n = nx * self.nyz
        print('Grid: %d x %d x %d = %d nodes (hx=%.3f hy=%.3f hz=%.3f nm)' % (nx, ny, nz, self.n, hx, hy, hz))

        wx, wy, wz = wts1d(nx, hx), wts1d(ny, hy), wts1d(nz, hz)
        self.wx, self.wy, self.wz = wx, wy, wz
        Wx, Wy, Wz = sps.diags(wx), sps.diags(wy), sps.diags(wz)
        Kx, Ky, Kz = lap1d(nx, hx), lap1d(ny, hy), lap1d(nz, hz)
        K0 = (sps.kron(Kx, sps.kron(Wy, Wz)) + sps.kron(Wx, sps.kron(Ky, Wz)) +
              sps.kron(Wx, sps.kron(Wy, Kz))).tocsr()             # unit-coefficient stiffness
        self.K0 = K0
        self.vol = np.einsum('i,j,k->ijk', wx, wy, wz).ravel()    # control volumes

        # ---- gate (Robin) weights: gate only where |x-xc| <= Lg/2, top face (z=H) and side wall (y=W/2)
        gl, gr = self.xc - p.Lg / 2, self.xc + p.Lg / 2
        lo = np.maximum.reduce([self.x - hx / 2, np.zeros(nx), np.full(nx, gl)])
        hi = np.minimum.reduce([self.x + hx / 2, np.full(nx, self.Ltot), np.full(nx, gr)])
        wxG = np.maximum(hi - lo, 0.0)                            # exact overlap of the control interval with the gate
        gate = np.zeros((nx, ny, nz))
        gate[:, :, -1] += wxG[:, None] * wy[None, :]
        gate[:, -1, :] += wxG[:, None] * wz[None, :]
        self.robB = self.Cq * gate.ravel()                         # Robin diagonal (per unit psi)
        wall = np.zeros((nx, ny, nz))                              # DG wall penalty: top, +y side, bottom
        wall[:, :, -1] += wx[:, None] * wy[None, :]
        wall[:, -1, :] += wx[:, None] * wz[None, :]
        wall[:, :, 0] += wx[:, None] * wy[None, :]
        self.wall = wall.ravel()

        # ---- doping
        self.Ndx = p.Nsd * 0.5 * (1 + erf((np.abs(self.x - self.xc) - p.Lj) / p.sigJ))
        self.Ndv = np.repeat(self.Ndx, self.nyz)
        NetS = self.Ndx[0] - p.Na
        NetD = self.Ndx[-1] - p.Na
        ni = self.ni
        self.psiS = self.Vt * np.arcsinh(NetS / (2 * ni))
        self.psiD = self.Vt * np.arcsinh(NetD / (2 * ni))
        self.wS = math.sqrt(0.5 * (NetS + math.sqrt(NetS ** 2 + 4 * ni ** 2)))
        self.wD = math.sqrt(0.5 * (NetD + math.sqrt(NetD ** 2 + 4 * ni ** 2)))

        # ---- free nodes (everything except the S and D faces) = contiguous slice
        self.fr = slice(self.nyz, (nx - 1) * self.nyz)

        # ---- coordinates (centred like the MATLAB mesh) for the output files
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.Xo = (X - self.xc).ravel()
        self.Yo = (Y - p.Wfin / 4).ravel()
        self.Zo = (Z - p.Hfin / 2).ravel()

        # ---- GPU copies
        self.A0 = to_xsp(self.epsSi * K0 + sps.diags(self.robB))           # eps*K + Robin diagonal
        self.A0ff = DiagCSR((self.epsSi * K0 + sps.diags(self.robB))[self.fr, self.fr])
        self.vol_g = to_xp(self.vol)
        self.Ndv_g = to_xp(self.Ndv)
        self.robB_g = to_xp(self.robB)
        self.wy_g, self.wz_g = to_xp(wy), to_xp(wz)
        self.Vc = self.vol_g * (self.q / self.eps0)                       # V*(q/eps0)

        # ---- quantum switches (set by equilibrium / used by every solve)
        self.bs = float(p.useDG)
        self.qpen_full = float(p.useDG) * 1e3
        self.qpen = self.qpen_full

    # ------------------------------------------------------------------ helpers
    def phin_col(self, phin_x):
        return to_xp(phin_x)[:, None]

    def init_guess(self, phin_x):
        Vt, ni, Na = self.Vt, self.ni, self.par.Na
        Nd = self.Ndv
        phin = np.repeat(phin_x, self.nyz)
        psi = Vt * np.arcsinh((Nd - Na) / (2 * ni)) + phin * Nd / (Nd + Na)
        w = np.sqrt(ni * np.exp(np.clip((psi - phin) / Vt, -80, 80)))
        return to_xp(psi), to_xp(w)

    def set_dirichlet(self, f, valS, valD):
        f = f.copy()
        f3 = f.reshape(self.nx, self.nyz)
        f3[0, :] = valS
        f3[-1, :] = valD
        return f

    def boltz_n(self, psi, phin_x):
        pr = psi.reshape(self.nx, self.nyz)
        return (self.ni * xp.exp(xp.clip((pr - self.phin_col(phin_x)) / self.Vt, -80, 80))).ravel()

    # ------------------------------------------------------------------ classical Poisson (1 unknown)
    def poisson_classical(self, psi0, phin_x, Vg, Vd):
        p, Vt, ni, Na = self.par, self.Vt, self.ni, self.par.Na
        fr = self.fr
        psi = self.set_dirichlet(psi0, self.psiS, self.psiD + Vd)
        gvec = self.robB_g * (Vg - self.phiRef)

        def resid(ps):
            n = self.boltz_n(ps, phin_x)
            pp = ni * xp.exp(xp.clip(-ps / Vt, -80, 80))
            F = self.A0 @ ps - self.Vc * (pp - n + self.Ndv_g - Na) - gvec
            return F[fr], n, pp

        for it in range(p.newton_max):
            F, n, pp = resid(psi)
            d = (self.Vc * (pp + n) / Vt)[fr]
            J = self.A0ff.make(d)
            dinv = 1.0 / (self.A0ff.diag0 + d)
            dpsi, _, ok = pcg(lambda v: J @ v, -F, xp.zeros_like(F), dinv, p.lin_tol, p.lin_max)
            mx = float(xp.abs(dpsi).max())
            if mx < p.newton_tol:
                return psi
            if mx > p.newton_cap:
                dpsi = dpsi * (p.newton_cap / mx)
            nF0 = float(xp.linalg.norm(F))
            lam = 1.0
            for ls in range(10):
                trial = psi.copy()
                trial[fr] += lam * dpsi
                nFt = float(xp.linalg.norm(resid(trial)[0]))
                if nFt < (1 - 1e-4 * lam) * nF0:
                    break
                lam *= 0.5
            psi = trial
        raise GummelError('Poisson Newton did not converge (Vg=%.2f Vd=%.2f)' % (Vg, Vd))

    # ------------------------------------------------------------------ density-gradient (2 unknowns)
    def poisson_dg(self, psi0, w0, phin_x, Vg, Vd):
        p, Vt, ni, Na = self.par, self.Vt, self.ni, self.par.Na
        fr = self.fr
        nf = (self.nx - 2) * self.nyz
        c2 = 2 * self.b * self.bs + 1e-6
        K2cpu = (c2 * self.K0 + sps.diags(self.qpen * self.wall)).tocsr()
        K2 = to_xsp(K2cpu)
        K2ff = DiagCSR(K2cpu[fr, fr])
        psi = self.set_dirichlet(psi0, self.psiS, self.psiD + Vd)
        w = self.set_dirichlet(w0, self.wS, self.wD)
        gvec = self.robB_g * (Vg - self.phiRef)
        phin3 = xp.repeat(to_xp(phin_x), self.nyz)
        vol = self.vol_g

        def resid(ps, ww):
            pp = ni * xp.exp(xp.clip(-ps / Vt, -80, 80))
            s = ps - phin3 - Vt * xp.log((ww * ww + 1e-40) / ni)
            F1 = self.A0 @ ps - self.Vc * (pp - ww * ww + self.Ndv_g - Na) - gvec
            F2 = K2 @ ww - vol * (ww * s)
            return F1[fr], F2[fr], pp, s

        for it in range(p.newton_max):
            F1, F2, pp, s = resid(psi, w)
            wf = w[fr]
            wsq = w * w
            d11 = (self.Vc * pp / Vt)[fr]
            d12 = (self.Vc * 2 * w)[fr]
            d21 = (-vol * w)[fr]
            d22 = (-vol * (s - 2 * Vt * wsq / (wsq + 1e-40)))[fr]
            A11 = self.A0ff.make(d11)
            A22 = K2ff.make(d22)
            e11 = self.A0ff.diag0 + d11
            e22 = K2ff.diag0 + d22
            det = e11 * e22 - d12 * d21

            def mv(v, A11=A11, A22=A22, d12=d12, d21=d21):
                a, bq = v[:nf], v[nf:]
                return xp.concatenate([A11 @ a + d12 * bq, d21 * a + A22 @ bq])

            def pre(v, e11=e11, e22=e22, d12=d12, d21=d21, det=det):
                a, bq = v[:nf], v[nf:]
                return xp.concatenate([(e22 * a - d12 * bq) / det, (-d21 * a + e11 * bq) / det])

            rhs = -xp.concatenate([F1, F2])
            dz, _, ok = bicgstab(mv, rhs, xp.zeros_like(rhs), pre, p.lin_tol, p.lin_max)
            if not ok:                                              # direct fallback (CPU, slow)
                Jc = sps.bmat([[to_np_sparse(A11), sps.diags(to_np(d12))],
                               [sps.diags(to_np(d21)), to_np_sparse(A22)]], format='csc')
                dz = to_xp(spla.spsolve(Jc, to_np(rhs)))
            dpsi, dw = dz[:nf], dz[nf:]
            mxp = float(xp.abs(dpsi).max())
            mxw = float((xp.abs(dw) / (xp.abs(wf) + 1e-30)).max())
            if mxp < p.newton_tol and mxw < 1e-6:
                return psi, w
            if mxp > p.newton_cap:
                sc = p.newton_cap / mxp
                dpsi, dw = dpsi * sc, dw * sc
            neg = dw < 0
            lam = min(1.0, float(xp.min(xp.where(neg, 0.95 * wf / (-dw + 1e-300), 1.0))))
            nF0 = math.hypot(float(xp.linalg.norm(F1)), float(xp.linalg.norm(F2)))
            for ls in range(10):
                tp, tw = psi.copy(), w.copy()
                tp[fr] += lam * dpsi
                tw[fr] += lam * dw
                G1, G2, _, _ = resid(tp, tw)
                nFt = math.hypot(float(xp.linalg.norm(G1)), float(xp.linalg.norm(G2)))
                if nFt < (1 - 1e-4 * lam) * nF0:
                    break
                lam *= 0.5
            psi, w = tp, tw
        raise GummelError('DG Newton did not converge (Vg=%.2f Vd=%.2f)' % (Vg, Vd))

    def solve_poisson(self, psi, w, phin_x, Vg, Vd):
        if psi is None:
            psi, w = self.init_guess(phin_x)
        if self.bs == 0:
            psi = self.poisson_classical(psi, phin_x, Vg, Vd)
            w = xp.sqrt(self.boltz_n(psi, phin_x))                 # exact nodal w = sqrt(n)
        else:
            psi, w = self.poisson_dg(psi, w, phin_x, Vg, Vd)
        return psi, w

    # ------------------------------------------------------------------ electrons per nm of channel (per fin)
    def line_density(self, psi, w, phin_x):
        if self.bs == 0:
            n = self.boltz_n(psi, phin_x)
        else:
            n = w * w
        t = (n.reshape(self.nx * self.ny, self.nz) @ self.wz_g).reshape(self.nx, self.ny) @ self.wy_g
        return 2.0 * to_np(t)                                     # x2: half fin


def to_np_sparse(M):
    return M.get() if GPU else M


# ======================================================================================
#  Gummel loop at one bias point
# ======================================================================================
def solve_bias(dev, Vg, Vd, guess=None, strict=False):
    p = dev.par
    xg = dev.x
    dx_cm = (xg[1] - xg[0]) * 1e-7
    erfp = Vd * 0.5 * (1 + erf((xg - dev.xc) / (p.Lg / 2)))
    if guess is None:
        phin, psi, w = erfp.copy(), None, None
    else:
        psi, w = guess['psi'], guess['w']
        phin = guess['phin'] * (Vd / guess['Vd']) if guess['Vd'] > 1e-6 else erfp.copy()
    if Vd < 1e-9:
        phin = np.zeros_like(xg)
    err = np.inf
    alpha = p.gumAlpha
    Xh, Fh = [], []
    errHist = []
    last_stagnation_restart = 0
    I = 0.0
    N = None
    for it in range(1, p.gumMax + 1):
        psi, w = dev.solve_poisson(psi, w, phin, Vg, Vd)
        N = dev.line_density(psi, w, phin)
        I, phin_new = march_current(N, Vd, phin, dx_cm, dev.Vt, dev.q, p.mu0, p.vsat)
        F = phin_new - phin                                       # fixed-point residual
        err = float(np.max(np.abs(F)))
        errHist.append(err)
        if p.verbose:
            print('    Gummel %2d: dphi_max=%.2e V   I=%.4e A  alpha=%.3f' % (it, err, I, alpha))
        if err < p.gumTol:
            break

        # --- strict mode rejects unconverged states so goto() can halve the bias step.
        if (strict and it >= 20 and it - last_stagnation_restart >= 10 and
                err > 3 * p.gumTol and
                min(errHist[it - 10:it]) > 0.8 * min(errHist[:it - 10])):
            alpha = max(alpha / 2, 0.01)
            Xh, Fh = [], []
            last_stagnation_restart = it

        # --- adaptive damping
        if it > 1 and err > 1.3 * errHist[it - 2]:
            alpha = max(alpha / 2, 0.05)
            Xh, Fh = [], []
        elif it > 1 and err < 0.7 * errHist[it - 2]:
            alpha = min(alpha * 1.3, 1.0)

        # --- Anderson mixing on phi_n
        Xh.append(phin.copy())
        Fh.append(F.copy())
        if len(Xh) > p.gumAA + 1:
            Xh.pop(0)
            Fh.pop(0)
        xnew = phin + alpha * F
        if len(Fh) >= 2:
            Xm, Fm = np.array(Xh).T, np.array(Fh).T
            dX, dF = np.diff(Xm, axis=1), np.diff(Fm, axis=1)
            A = dF.T @ dF
            gam = np.linalg.solve(A + 1e-8 * np.trace(A) * np.eye(A.shape[0]), dF.T @ F)
            xnew = xnew - (dX + alpha * dF) @ gam
        step = xnew - phin
        mx = float(np.max(np.abs(step)))
        if mx > p.gumCap:
            step *= p.gumCap / mx
        phin = phin + step
        phin[0] = 0.0
        phin[-1] = Vd

    if err >= p.gumTol:
        if strict:
            raise GummelError('Gummel not converged (Vg=%.2f Vd=%.2f, err=%.1e V)' % (Vg, Vd, err))
        warnings.warn('Vg=%.2f Vd=%.2f: Gummel not converged (err=%.1e V after %d it)' % (Vg, Vd, err, it))

    ic = int(np.argmin(np.abs(xg - dev.xc)))
    inG = np.abs(xg - dev.xc) <= p.Lg / 2
    xs, xd = dev.xc - p.Lg / 2, dev.xc + p.Lg / 2
    dSD = (np.interp(xs, xg, phin) - phin[0]) + (phin[-1] - np.interp(xd, xg, phin))
    return dict(Vg=Vg, Vd=Vd, psi=psi, w=w, I=I, phin=phin, N=N, Nc=N[ic], Nmin=float(N[inG].min()),
                fSD=dSD / max(Vd, 1e-12), nIter=it, err=err)


def equilibrium(dev):
    """Staged start: classical first, then ramp the quantum term."""
    full_bs, full_q = dev.bs, dev.qpen_full
    dev.bs, dev.qpen = 0.0, 0.0
    st = solve_bias(dev, 0.0, 0.0, None)
    if dev.par.useDG:
        bsv = [0.01, 0.05, 0.2, 0.5, 1.0]
        qpv = [full_q * f for f in (1e-3, 1e-2, 0.1, 0.5, 1.0)]
        for i in range(len(bsv)):
            dev.bs, dev.qpen = bsv[i], qpv[i]
            st = solve_bias(dev, 0.0, 0.0, st)
            print('  DG ramp %d/%d done' % (i + 1, len(bsv)))
    dev.bs, dev.qpen = full_bs, full_q
    return st


def goto(dev, st0, Vg, Vd, depth=0):
    """Continuation with automatic step halving."""
    try:
        return solve_bias(dev, Vg, Vd, st0, strict=True)
    except GummelError as e:
        if depth >= 2:
            raise
        print('    [goto] %s -> halving step (depth %d)' % (e, depth + 1))
        mid = goto(dev, st0, (st0['Vg'] + Vg) / 2, (st0['Vd'] + Vd) / 2, depth + 1)
        return goto(dev, mid, Vg, Vd, depth + 1)


# ======================================================================================
#  Output
# ======================================================================================
class Output:
    def __init__(self, dev):
        self.dev = dev
        p = dev.par
        os.makedirs(p.outDir, exist_ok=True)
        self.csvIV = os.path.join(p.outDir, 'voltage current information.csv')
        self.csv3D = os.path.join(p.outDir, '3D.csv')
        self.csvMX = os.path.join(p.outDir, 'Ids matrix Vgs x Vds.csv')
        self.fIV = open(self.csvIV, 'w')
        self.fIV.write('Vgs_V,Vds_V,Ids_A_per_fin,Ids_uA_per_um,N_center_e_per_nm,'
                       'N_barrier_min_e_per_nm,Vdrop_SD_percent,Gummel_iter,Gummel_err_V,Status\n')
        self.f3D = None
        if p.log3D == 'csv':
            self.f3D = open(self.csv3D, 'w')
            self.f3D.write('Vgs_V,Vds_V,x_nm,y_nm,z_nm,psi_V,w_nm^-1.5,n_cm-3,p_cm-3,Nd_cm-3\n')
        elif p.log3D == 'npz':
            os.makedirs(os.path.join(p.outDir, '3D_npz'), exist_ok=True)

    def point(self, st):
        dev, p = self.dev, self.dev.par
        status = 'NOT_CONVERGED' if st['err'] >= p.gumTol else 'OK'
        self.fIV.write('%.3f,%.3f,%.6e,%.6e,%.6e,%.6e,%.3f,%d,%.3e,%s\n' %
                       (st['Vg'], st['Vd'], st['I'], st['I'] * 1e6 / dev.Weff, st['Nc'], st['Nmin'],
                        100 * st['fSD'], st['nIter'], st['err'], status))
        self.fIV.flush()
        if self.f3D is None and p.log3D != 'npz':
            return
        psi, w = to_np(st['psi']), to_np(st['w'])
        n = w ** 2 * 1e21                                          # cm^-3
        pc = dev.ni * np.exp(np.clip(-psi / dev.Vt, -80, 80)) * 1e21
        Nd = dev.Ndv * 1e21
        if p.log3D == 'csv':
            o = np.ones_like(psi)
            M = np.column_stack([st['Vg'] * o, st['Vd'] * o, dev.Xo, dev.Yo, dev.Zo, psi, w, n, pc, Nd])
            np.savetxt(self.f3D, M, fmt='%.3f,%.3f,%.4f,%.4f,%.4f,%.6e,%.6e,%.6e,%.6e,%.6e')
            self.f3D.flush()
        else:
            np.savez(os.path.join(p.outDir, '3D_npz', 'Vg%.2f_Vd%.2f.npz' % (st['Vg'], st['Vd'])),
                     x=dev.Xo, y=dev.Yo, z=dev.Zo, psi=psi, w=w, n=n, p=pc, Nd=Nd)

    def fail(self, Vg, Vd):
        self.fIV.write('%.3f,%.3f,NaN,NaN,NaN,NaN,NaN,NaN,NaN,FAILED\n' % (Vg, Vd))
        self.fIV.flush()

    def close(self):
        for f in (self.fIV, self.f3D):
            if f is not None and not f.closed:
                f.close()


# ======================================================================================
#  Main
# ======================================================================================
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    par = Par()
    if '--test' in argv:
        par.test = True
    if '--cpu' in argv:
        par.use_gpu = False
    if '--dg' in argv:
        par.useDG = True
    if '--no3d' in argv:
        par.log3D = ''
    if '--npz' in argv:
        par.log3D = 'npz'
    for a in argv:                                  # --out=PATH  --h=0.5
        if a.startswith('--out='):
            par.outDir = a.split('=', 1)[1]
        if a.startswith('--h='):
            par.hx = par.hy = par.hz = float(a.split('=', 1)[1])
    par.__post_init__()
    main_run(par)


def extract_threshold_metrics(VgFix, VdsList, IdVgs, par):
    Icc = par.Icc_A or 1e-7 * (2 * par.Hfin + par.Wfin) / par.Lg
    Vth, dibl = constant_current_dibl(VgFix, VdsList, IdVgs.T, Icc)
    curves = subthreshold_swing(VgFix, IdVgs.T)
    SS = np.array([np.min(curve[np.isfinite(curve)]) if np.isfinite(curve).any()
                   else np.nan for curve in curves.T])
    return Icc, Vth, SS, dibl


def main_run(par):
    select_backend(par.use_gpu)

    VgFix = np.asarray(par.vgs_values) if par.vgs_values else np.round(
        np.arange(par.vStep, par.vMax + 1e-9, par.vStep), 6)
    VdsList = np.asarray(par.vds_values) if par.vds_values else np.round(
        np.arange(par.vStep, par.vMax + 1e-9, par.vStep), 6)
    VdsList = VdsList[VdsList > 0]
    if VgFix.size == 0 or VdsList.size == 0:
        raise ValueError('Voltage sweeps require gate points and positive drain points')
    if par.test:
        VgFix, VdsList = np.array([0.8]), np.array([0.05, 0.1])

    dev = Device(par)
    out = Output(dev)
    t0 = time.time()
    try:
        st0 = equilibrium(dev)
        print('Equilibrium solved (%.0f s)' % (time.time() - t0))

        nVg, nVd = len(VgFix), len(VdsList)
        IdVds = np.full((nVg, nVd), np.nan)
        stPlot, stLast = None, None
        hdr = ('Vgs=%.2f Vds=%.2f  Ids=%.3e A  N(centre)=%.3f  N(barrier min)=%.3f e/nm  '
               'Vdrop_SD=%.0f%%  Gummel it=%d err=%.1e (%.0f s)')
        print('\n===== Id-Vds: %d traces (Vgs = %.2f ... %.2f V), %d points each (Vds = %.2f ... %.2f V) =====' %
              (nVg, VgFix[0], VgFix[-1], nVd, VdsList[0], VdsList[-1]))
        stV0 = st0                                                # Vd = 0 state carried between Vgs traces
        for k in range(nVg):
            try:
                stV0 = goto(dev, stV0, float(VgFix[k]), 0.0, 0)
            except GummelError as e:
                warnings.warn('Vgs=%.2f could not be reached at Vd=0 (%s) - trace skipped' % (VgFix[k], e))
                for j in range(nVd):
                    out.fail(VgFix[k], VdsList[j])
                continue
            st = stV0
            for j in range(nVd):
                try:
                    st = goto(dev, st, float(VgFix[k]), float(VdsList[j]), 0)
                except GummelError as e:
                    warnings.warn('Vgs=%.2f Vds=%.2f failed (%s)' % (VgFix[k], VdsList[j], e))
                    out.fail(VgFix[k], VdsList[j])
                    continue
                IdVds[k, j] = st['I']
                stLast = st
                if abs(st['Vg'] - 0.8) < 1e-9 and abs(st['Vd'] - 1.5) < 1e-9:
                    stPlot = st
                print(hdr % (VgFix[k], VdsList[j], st['I'], st['Nc'], st['Nmin'], 100 * st['fSD'],
                             st['nIter'], st['err'], time.time() - t0))
                out.point(st)
                np.savez(os.path.join(par.outDir, 'partial.npz'), IdVds=IdVds, VgFix=VgFix, VdsList=VdsList)
        if stPlot is None:
            stPlot = stLast
    finally:
        out.close()

    # ---------------- full data set (Vds = 0 point added, Ids = 0 there)
    include_zero = not par.vds_values or par.vds_values[0] == 0
    VdsPlot = np.concatenate([[0.0], VdsList]) if include_zero else VdsList.copy()
    IdFull = np.hstack([np.zeros((nVg, 1)), IdVds]) if include_zero else IdVds.copy()
    IdVgs = IdVds.T                                               # rows: Vds, cols: Vgs  (same data)
    np.savez(os.path.join(par.outDir, 'results.npz'), IdVds=IdVds, IdFull=IdFull, IdVgs=IdVgs,
             VgFix=VgFix, VdsList=VdsList, VdsPlot=VdsPlot)
    with open(out.csvMX, 'w') as f:
        f.write('Vgs_V \\ Vds_V' + ''.join(',%.2f' % v for v in VdsPlot) + '\n')
        for k in range(nVg):
            f.write('%.2f' % VgFix[k] + ''.join(',%.6e' % v for v in IdFull[k]) + '\n')
    print('\nData written to:\n  %s\n  %s' % (out.csvIV, out.csvMX))
    if par.log3D == 'csv':
        print('  %s' % out.csv3D)
    elif par.log3D == 'npz':
        print('  %s' % os.path.join(par.outDir, '3D_npz'))

    # ---------------- constant-current threshold and DIBL metrics
    Icc, Vth, SS, dibl = extract_threshold_metrics(VgFix, VdsList, IdVgs, par)
    ss_curves = subthreshold_swing(VgFix, IdFull)
    np.savez(os.path.join(par.outDir, 'results.npz'), IdVds=IdVds, IdFull=IdFull,
             IdVgs=IdVgs, VgFix=VgFix, VdsList=VdsList, VdsPlot=VdsPlot,
             T_K=par.T_K, Vth=Vth, SS=SS, SS_curves=ss_curves,
             DIBL_mV_per_V=dibl, Icc_A_per_fin=Icc)
    for j in range(nVd):
        if np.isfinite(Vth[j]):
            print('Vds=%.2f V: Vth(CC)=%.3f V, SS(min)=%.1f mV/dec' %
                  (VdsList[j], Vth[j], SS[j]))
    valid_vth = np.flatnonzero(np.isfinite(Vth))
    if np.isfinite(dibl):
        print('DIBL = %.1f mV/V (Vds %.2f -> %.2f V)' %
              (dibl, VdsList[valid_vth[0]], VdsList[valid_vth[-1]]))

    # ---------------- plots
    import matplotlib.cm
    if par.show_plots:
        import matplotlib.pyplot as plt
        new_figure = plt.figure
    else:
        from matplotlib.figure import Figure
        new_figure = Figure
    cm = matplotlib.cm.viridis

    fig1 = new_figure(figsize=(7.6, 5.2))
    axis = fig1.subplots()
    for k in range(nVg):
        axis.plot(VdsPlot, IdFull[k] * 1e6, 'o-', color=cm(k / max(nVg - 1, 1)), lw=1.3, ms=4,
                 label='V_GS=%.1f V' % VgFix[k])
    axis.grid(True)
    axis.set(xlabel='V_DS [V]', ylabel='I_DS [\u00b5A/fin]',
             title='I_DS-V_DS: %d traces, %d points each' % (nVg, len(VdsPlot)))
    axis.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig1.savefig(os.path.join(par.outDir, 'Ids-Vds.png'), dpi=150, bbox_inches='tight')

    fig2 = new_figure(figsize=(11, 4.8))
    ax = fig2.subplots(1, 2)
    for j in range(nVd):
        Ij = np.abs(IdVgs[j, :])
        Ij = np.where(Ij < 1e-18, 1e-18, Ij)
        ax[0].semilogy(VgFix, Ij, 'o-', color=cm(j / max(nVd - 1, 1)), lw=1.3, ms=4, label='V_DS=%.2f V' % VdsList[j])
        ax[1].plot(VgFix, IdVgs[j, :] * 1e6 / dev.Weff, 'o-', color=cm(j / max(nVd - 1, 1)), lw=1.3, ms=4)
    ax[0].grid(True)
    ax[0].set_xlabel('V_GS [V]')
    ax[0].set_ylabel('I_DS [A/fin]')
    ax[0].set_title('I_DS-V_GS (log)')
    ax[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)
    ax[1].grid(True)
    ax[1].set_xlabel('V_GS [V]')
    ax[1].set_ylabel('I_DS [\u00b5A/\u00b5m]')
    ax[1].set_title('I_DS-V_GS (linear)')
    fig2.savefig(os.path.join(par.outDir, 'Ids-Vgs.png'), dpi=150, bbox_inches='tight')

    if stPlot is not None:
        xr = dev.x - dev.xc
        fig3 = new_figure(figsize=(7, 6))
        ax3 = fig3.subplots(2, 1, sharex=True)
        ax3[0].plot(xr, stPlot['phin'], lw=1.5)
        ax3[0].set_ylabel('\u03c6_n [V]')
        ax3[0].set_title('V_GS=%.2f V, V_DS=%.2f V' % (stPlot['Vg'], stPlot['Vd']))
        ax3[1].semilogy(xr, np.maximum(stPlot['N'], 1e-12), lw=1.5)
        ax3[1].set_ylabel('N(x) [e/nm per fin]')
        ax3[1].set_xlabel('x - x_c [nm]')
        for a in ax3:
            a.grid(True)
            a.axvline(-par.Lg / 2, ls='--', c='k')
            a.axvline(par.Lg / 2, ls='--', c='k')
        fig3.savefig(os.path.join(par.outDir, 'phin_N_profile.png'), dpi=150, bbox_inches='tight')

        psi = to_np(stPlot['psi']).reshape(dev.nx, dev.ny, dev.nz)
        w = to_np(stPlot['w']).reshape(dev.nx, dev.ny, dev.nz)
        fig4 = new_figure(figsize=(9, 7))
        ax4 = fig4.subplots(2, 1)
        ext = [xr[0], xr[-1], dev.z[0] - par.Hfin / 2, dev.z[-1] - par.Hfin / 2]
        im0 = ax4[0].imshow(psi[:, 0, :].T, origin='lower', extent=ext, aspect='auto')
        ax4[0].set_title('\u03c8 [V]  (symmetry plane y = 0)')
        fig4.colorbar(im0, ax=ax4[0])
        im1 = ax4[1].imshow(np.log10(w[:, 0, :].T ** 2 * 1e21 + 1), origin='lower', extent=ext, aspect='auto')
        ax4[1].set_title('log10 n [cm^-3]  (y = 0)')
        fig4.colorbar(im1, ax=ax4[1])
        for a in ax4:
            a.set_xlabel('x - x_c [nm]')
            a.set_ylabel('z [nm]')
        fig4.savefig(os.path.join(par.outDir, 'psi_n_slice.png'), dpi=150, bbox_inches='tight')
    if par.show_plots:
        plt.show()
    print('Total time: %.0f s' % (time.time() - t0))
    return {"Vgs": VgFix, "Vds": VdsPlot, "Id": IdFull, "Vds_threshold": VdsList,
            "Vth": Vth, "SS": SS, "DIBL_mV_per_V": dibl, "Icc_A_per_fin": Icc,
            "SS_curves": ss_curves, "T_K": par.T_K, "outDir": par.outDir}


if __name__ == '__main__':
    main()