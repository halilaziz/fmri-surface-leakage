"""
Numerical checks of cortical-sheet sampling theory under MRI encoding.

Claims tested
  T1 (3D Fourier encoding, thin columnar sheet, noise-free): the measurable tangential
     band is the projection P_n(B) of the k-space box B onto the sheet's tangent plane;
     its inradius is rho(n) = (pi/h) * min_{u in T_n, |u|=1} ||u||_1  in [pi/h, sqrt(2) pi/h].
  T2 With realistic encoding (2D multislice slice profile, EPI T2* apodization), finite SNR,
     registration jitter and a BOLD-PSF-limited prior, how much of that survives?
  T3 Two opposite sulcal banks: antisymmetric (bank-specific) patterns are identifiable only
     when the mid-surface separation d is comparable to the voxel size.

Everything is linear-Gaussian: y = K f + e, f ~ N(0, C), e ~ N(0, s^2 I).
Averaging kernel A = (K'K/s^2 + C^-1)^-1 K'K/s^2 ; transfer tau(q) = <e_q, A e_q>/<e_q, e_q>
for windowed tangential plane waves e_q.
"""
import numpy as np
from scipy.special import erf
import scipy.linalg as sla
import time, json, sys, os

OUT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- geometry
def orthobasis(n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    a = np.array([1., 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1., 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2, n


def rho_ratio(n, h=(1., 1., 1.), nang=7200):
    """Inradius of the projected k-box, in units of pi (rad/mm) -> multiply by pi.
    Support function of box B=[-pi/hx,pi/hx]x... in direction u is pi*sum|u_d|/h_d."""
    e1, e2, _ = orthobasis(n)
    th = np.linspace(0, np.pi, nang, endpoint=False)
    U = np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2)
    supp = (np.abs(U) / np.asarray(h)).sum(1)
    return supp.min(), supp.max()


# ----------------------------------------------------------------------------- kernels
def kspace_kernel(t, N, FOV, W):
    """1-D image-domain kernel of a Fourier-encoded axis with k-space weights W
    (indices -(N-1)/2..(N-1)/2, N odd, symmetric W -> real kernel). Peak ~ mean(W), integral = FOV/N * W0."""
    ks = np.arange(N) - (N - 1) // 2
    sh = t.shape
    return (np.cos(2 * np.pi * np.outer(t.ravel(), ks) / FOV) @ W / N).reshape(sh)


def slice_profile(t, dz, edge_frac=0.15, jitter=0.0):
    s = np.sqrt((edge_frac * dz) ** 2 + jitter ** 2)
    return 0.5 * (erf((t + dz / 2) / (np.sqrt(2) * s)) - erf((t - dz / 2) / (np.sqrt(2) * s)))


def axis_weights(N, FOV, kind='ideal', t2s_ms=None, esp_eff_ms=None, fov_pe_real=None, jitter=0.0):
    """k-space weights for one Fourier-encoded axis.
    t2s blur: W(k) = exp(-|k| * (FOV_real/2pi) * ESP_eff / T2*), k in rad/mm (linear ordering, echo at k=0).
    jitter: Gaussian registration jitter sigma (mm) -> exp(-sigma^2 k^2 / 2)."""
    ks = np.arange(N) - (N - 1) // 2
    k = 2 * np.pi * ks / FOV  # rad/mm
    W = np.ones(N)
    if kind == 't2s':
        lines = np.abs(k) * fov_pe_real / (2 * np.pi)
        W *= np.exp(-lines * esp_eff_ms / t2s_ms)
    if jitter > 0:
        W *= np.exp(-0.5 * (jitter * k) ** 2)
    return W


# ----------------------------------------------------------------------------- forward operator
def build_K(sheets, h=2.0, N=25, enc='3D', pe_axis=0, pe_W=None, ro_W=None, sl_jitter=0.0,
            x0=(0.37, -0.21, 0.13), chunk=256):
    """sheets: list of dicts {n, offset (mm along n), T, nd, delta, M}; each sheet contributes M*M unknowns
    (columnar field constant across depth). Voxel lattice N^3, spacing h, centred at 0.
    enc '3D': Fourier on all axes. enc '2D': Fourier in-plane (axes 0,1), slice profile on axis 2."""
    FOV = N * h
    c = (np.arange(N) - (N - 1) // 2) * h
    Ws = []
    for ax in range(3):
        if ax == pe_axis and pe_W is not None:
            Ws.append(pe_W)
        elif ro_W is not None and ax != 2:
            Ws.append(ro_W)
        else:
            Ws.append(np.ones(N))
    cols = []
    for sh in sheets:
        e1, e2, nn = orthobasis(sh['n'])
        M, delta, T, nd = sh['M'], sh['delta'], sh['T'], sh['nd']
        u = (np.arange(M) - (M - 1) / 2) * delta
        U1, U2 = np.meshgrid(u, u, indexing='ij'); U1 = U1.ravel(); U2 = U2.ravel()
        s = ((np.arange(nd) + 0.5) / nd - 0.5) * T + sh.get('offset', 0.0)
        w = delta ** 2 * (T / nd) / h ** 3
        P = (np.asarray(x0)[None, None, :] + U1[:, None, None] * e1 + U2[:, None, None] * e2
             + s[None, :, None] * nn)  # (J, nd, 3)
        J = P.shape[0]
        Ksh = np.empty((N ** 3, J))
        for j0 in range(0, J, chunk):
            Pc = P[j0:j0 + chunk]  # (q, nd, 3)
            q = Pc.shape[0]
            ker = []
            for ax in range(3):
                t = c[:, None] - Pc[..., ax].reshape(1, -1)  # (N, q*nd)
                if enc == '2D' and ax == 2:
                    k1 = slice_profile(t, h, jitter=sl_jitter)
                else:
                    k1 = kspace_kernel(t, N, FOV, Ws[ax])
                ker.append(k1.reshape(N, q, nd))
            blk = np.einsum('iqm,jqm,kqm->ijkq', ker[0], ker[1], ker[2], optimize=True)
            Ksh[:, j0:j0 + q] = w * blk.reshape(N ** 3, q)
        cols.append(Ksh)
    return np.concatenate(cols, axis=1)


# ----------------------------------------------------------------------------- prior
def cov_from_spectrum(Sfun, delta, M, pad=8):
    L = M * pad
    q = 2 * np.pi * np.fft.fftfreq(L, d=delta)
    QX, QY = np.meshgrid(q, q, indexing='ij')
    S = Sfun(np.sqrt(QX ** 2 + QY ** 2))
    cf = np.real(np.fft.ifft2(S))
    cf /= cf[0, 0]  # unit marginal variance
    idx = np.arange(M)
    d1 = (idx[:, None] - idx[None, :]) % L
    C = cf[d1[:, None, :, None], d1[None, :, None, :]]
    return C.reshape(M * M, M * M)


def spectrum(kind, fwhm_bold=None, ell_c=1.0):
    """Tangential spectral density of the latent BOLD field (shape only)."""
    def S(q):
        out = np.ones_like(q)
        if kind in ('matern', 'matern_bold'):
            kap = 1.0 / ell_c
            out = (kap ** 2 + q ** 2) ** -2.0
        if kind in ('white_bold', 'matern_bold') and fwhm_bold:
            sb = fwhm_bold / 2.3548
            out = out * np.exp(-(sb * q) ** 2)  # |MTF|^2 of Gaussian PSF
        return out
    return S


# ----------------------------------------------------------------------------- posterior transfer
def averaging_ops(K, C, cnr, block_sizes):
    """Return function applying the averaging kernel A to (J, nq) arrays, and sigma used.
    sigma chosen so that voxel-level signal std / noise std = cnr (signal std under the prior)."""
    KC = K @ C
    sig_var = np.einsum('ij,ij->i', KC, K)
    mask = sig_var > 1e-6 * sig_var.max()
    sig_std = np.sqrt(sig_var[mask].mean())
    s = sig_std / cnr
    G = (K.T @ K) / s ** 2
    lam, U = np.linalg.eigh(C)
    lam = np.clip(lam, 0, None)
    Ch = (U * np.sqrt(lam)) @ U.T
    B = Ch @ G @ Ch
    Lc = sla.cho_factor(B + np.eye(B.shape[0]))
    dfs = np.trace(sla.cho_solve(Lc, B))  # degrees of freedom for signal = tr(A)

    def applyA(E):
        return Ch @ sla.cho_solve(Lc, Ch @ (G @ E))
    return applyA, s, dfs


def plane_waves(sheet, qs, thetas, R=9.0, which=0, nsheets=1, sign=None):
    M, delta = sheet['M'], sheet['delta']
    u = (np.arange(M) - (M - 1) / 2) * delta
    U1, U2 = np.meshgrid(u, u, indexing='ij'); U1 = U1.ravel(); U2 = U2.ravel()
    r = np.sqrt(U1 ** 2 + U2 ** 2)
    win = np.where(r < R, 0.5 * (1 + np.cos(np.pi * r / R)), 0.0)
    E = []
    for q in qs:
        for th in thetas:
            ph = q * (np.cos(th) * U1 + np.sin(th) * U2)
            e = win * np.cos(ph)
            if nsheets == 1:
                E.append(e)
            else:
                z = np.zeros(nsheets * M * M)
                z[:M * M] = e
                z[M * M:] = e if sign is None else sign * e
                E.append(z)
    return np.array(E).T


def transfer(applyA, E):
    AE = applyA(E)
    return np.einsum('ij,ij->j', E, AE) / np.einsum('ij,ij->j', E, E)


def resolvable_wavelengths(tau, qs, thetas, thr=0.5):
    """tau shaped (nq, nth). For each direction the largest q with tau>=thr (scanning from 0)."""
    tau = tau.reshape(len(qs), len(thetas))
    qmax = np.zeros(len(thetas))
    for j in range(len(thetas)):
        ok = tau[:, j] >= thr
        if not ok[0]:
            qmax[j] = 0.0
            continue
        k = np.argmin(ok) if not ok.all() else len(qs)
        qmax[j] = qs[k - 1]
    with np.errstate(divide='ignore'):
        lam = np.where(qmax > 0, 2 * np.pi / qmax, np.inf)
    return lam.max(), lam.min(), lam  # isotropic (worst direction), best direction, per-direction


if __name__ == '__main__':
    # quick analytic table + a timing test
    for n in [(0, 0, 1), (1, 1, 0), (1, 1, 1), (0.3, 0.5, 0.81)]:
        print(n, 'rho/(pi/h) min,max =', np.round(rho_ratio(n), 4))
    t0 = time.time()
    sheet = dict(n=(0.3, 0.5, 0.81), offset=0.0, T=2.5, nd=6, delta=0.5, M=64)
    K = build_K([sheet], h=2.0, N=25, enc='3D')
    print('K', K.shape, 'built in %.1fs' % (time.time() - t0))
    t0 = time.time()
    G = K.T @ K
    print('KtK in %.1fs' % (time.time() - t0))
