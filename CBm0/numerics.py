from __future__ import annotations
import math
import numpy as np
import numba

# ---------------------------
# TVD limiter and operators
# ---------------------------

def _limiter(a, b, kind='minmod'):
    if kind == 'minmod':
        s = np.sign(a) + np.sign(b)
        return 0.5 * s * np.minimum(np.abs(a), np.abs(b))
    elif kind == 'vanleer':
        r_num = a * b * 2.0
        r_den = a + b + 1e-30
        return r_num / r_den
    elif kind == 'superbee':
        zero = np.zeros_like(a)
        return np.maximum.reduce([
            zero,
            np.minimum(2.0 * np.abs(a) * np.sign(b), np.abs(b) * np.sign(b)),
            np.minimum(np.abs(a) * np.sign(a), 2.0 * np.abs(b) * np.sign(a))
        ])
    else:
        s = np.sign(a) + np.sign(b)
        return 0.5 * s * np.minimum(np.abs(a), np.abs(b))


def tvd_div(phi, u, v, dx, dy, limiter_kind='minmod'):
    Ny, Nx = phi.shape
    ue = 0.5 * (u[:, 1:] + u[:, :-1])
    dphi_x_f = phi[:, 1:] - phi[:, :-1]
    dphi_x_b = np.zeros_like(phi); dphi_x_b[:, 1:] = dphi_x_f
    dphi_x_fpad = np.zeros_like(phi); dphi_x_fpad[:, :-1] = dphi_x_f
    slope_x = _limiter(dphi_x_b, dphi_x_fpad, kind=limiter_kind)
    phiL = phi[:, :-1] + 0.5 * slope_x[:, :-1]
    phiR = phi[:, 1:] - 0.5 * slope_x[:, 1:]
    phie = np.where(ue >= 0.0, phiL, phiR); Fe = ue * phie

    vn = 0.5 * (v[1:, :] + v[:-1, :])
    dphi_y_f = phi[1:, :] - phi[:-1, :]
    dphi_y_b = np.zeros_like(phi); dphi_y_b[1:, :] = dphi_y_f
    dphi_y_fpad = np.zeros_like(phi); dphi_y_fpad[:-1, :] = dphi_y_f
    slope_y = _limiter(dphi_y_b, dphi_y_fpad, kind=limiter_kind)
    phiS = phi[:-1, :] + 0.5 * slope_y[:-1, :]
    phiN = phi[1:, :]  - 0.5 * slope_y[1:,  :]
    phin = np.where(vn >= 0.0, phiS, phiN); Fn = vn * phin

    div = np.zeros_like(phi)
    div[:, 1:-1] += (Fe[:, 1:] - Fe[:, :-1]) / dx
    div[1:-1, :] += (Fn[1:, :] - Fn[:-1, :]) / dy
    return div


def apply_diffusion(f, k, dx, dy):
    ke = 0.5 * (k[:, 1:] + k[:, :-1]); kw = ke
    kn = 0.5 * (k[1:, :] + k[:-1, :]); ks = kn
    df_e = (f[:, 1:] - f[:, :-1]) / dx; df_w = df_e
    df_n = (f[1:, :] - f[:-1, :]) / dy; df_s = df_n
    Fe = ke * df_e; Fw = kw * df_w; Fn = kn * df_n; Fs = ks * df_s
    div = np.zeros_like(f)
    div[:, 1:-1] += (Fe[:, 1:] - Fw[:, :-1]) / dx
    div[1:-1, :] += (Fn[1:, :] - Fs[:-1, :]) / dy
    return div


def divergence(u, v, dx, dy):
    dudx = np.zeros_like(u); dvdy = np.zeros_like(v)
    if u.shape[1] >= 3:
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    if v.shape[0] >= 3:
        dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dy)
    dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
    dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    dvdy[0, :] = (v[1, :] - v[0, :]) / dy
    dvdy[-1, :] = (v[-1, :] - v[-2, :]) / dy
    return dudx + dvdy

# ---------------------------
# Multigrid (Numba JIT)
# ---------------------------

@numba.njit(parallel=True, cache=True)
def smooth_rbgs(p, rhs, k, dx, dy, iters=2):
    Ny, Nx = p.shape
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)

    for _ in range(iters):
        # Red color update
        for j in numba.prange(1, Ny-1):
            for i in range(1, Nx-1):
                if (i + j) % 2 == 0:
                    ke = 0.5 * (k[j, i+1] + k[j, i])
                    kw = 0.5 * (k[j, i-1] + k[j, i])
                    kn = 0.5 * (k[j+1, i] + k[j, i])
                    ks = 0.5 * (k[j-1, i] + k[j, i])
                    aE = ke * dx2_inv; aW = kw * dx2_inv; aN = kn * dy2_inv; aS = ks * dy2_inv
                    aC = aE + aW + aN + aS + 1e-30
                    p[j, i] = (aE*p[j, i+1] + aW*p[j, i-1] + aN*p[j+1, i] + aS*p[j-1, i] - rhs[j, i]) / aC
        # Black color update
        for j in numba.prange(1, Ny-1):
            for i in range(1, Nx-1):
                if (i + j) % 2 == 1:
                    ke = 0.5 * (k[j, i+1] + k[j, i])
                    kw = 0.5 * (k[j, i-1] + k[j, i])
                    kn = 0.5 * (k[j+1, i] + k[j, i])
                    ks = 0.5 * (k[j-1, i] + k[j, i])
                    aE = ke * dx2_inv; aW = kw * dx2_inv; aN = kn * dy2_inv; aS = ks * dy2_inv
                    aC = aE + aW + aN + aS + 1e-30
                    p[j, i] = (aE*p[j, i+1] + aW*p[j, i-1] + aN*p[j+1, i] + aS*p[j-1, i] - rhs[j, i]) / aC
        # Boundaries
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = 0.0
    return p

@numba.njit(parallel=True, cache=True)
def restrict_avg(f):
    Ny, Nx = f.shape
    Nc_x = (Nx - 1) // 2 + 1
    Nc_y = (Ny - 1) // 2 + 1
    fc = np.zeros((Nc_y, Nc_x))

    # Copy boundaries
    fc[0, :] = f[0, ::2]
    fc[-1, :] = f[-1, ::2]
    fc[:, 0] = f[::2, 0]
    fc[:, -1] = f[::2, -1]

    # Interior
    for J in numba.prange(1, Nc_y-1):
        for I in range(1, Nc_x-1):
            j = 2 * J
            i = 2 * I
            fc[J, I] = (f[j-1, i-1] + f[j-1, i] + f[j, i-1] + f[j, i]) * 0.25
    return fc


def prolong_bilinear(fc, Nf_shape):
    Nc_y, Nc_x = fc.shape; Nf_y, Nf_x = Nf_shape
    pf = np.zeros((Nf_y, Nf_x))
    pf[::2, ::2] = fc
    pf[::2, 1:-1:2] = 0.5 * (pf[::2, 0:-2:2] + pf[::2, 2::2])
    pf[1:-1:2, ::2] = 0.5 * (pf[0:-2:2, ::2] + pf[2::2, ::2])
    pf[1:-1:2, 1:-1:2] = 0.25 * (pf[0:-2:2, 0:-2:2] + pf[0:-2:2, 2::2] + pf[2::2, 0:-2:2] + pf[2::2, 2::2])
    pf[:, -1] = pf[:, -2]; pf[-1, :] = pf[-2, :]
    return pf


def vcycle(p, rhs, k, dx, dy, pre, post, mg_coarsest_min=8):
    Ny, Nx = p.shape
    if min(Nx, Ny) <= mg_coarsest_min:
        p = smooth_rbgs(p, rhs, k, dx, dy, iters=30)
        return p
    p = smooth_rbgs(p, rhs, k, dx, dy, iters=pre)
    res = rhs - apply_diffusion(p, k, dx, dy)
    rc = restrict_avg(res)
    kc = restrict_avg(k)
    p0c = np.zeros_like(rc)
    dxc, dyc = dx*2, dy*2
    ec = vcycle(p0c, rc, kc, dxc, dyc, pre, post, mg_coarsest_min)
    ef = prolong_bilinear(ec, p.shape)
    p = p + ef
    p = smooth_rbgs(p, rhs, k, dx, dy, iters=post)
    return p


def mg_solve(rhs, k, dx, dy, p0, cycles, pre, post, mgmin):
    p = p0.copy()
    for _ in range(cycles):
        p = vcycle(p, rhs, k, dx, dy, pre, post, mgmin)
    return p
