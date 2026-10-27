"""Week 3 library: model-based estimation of the cortical response field from voxel data.

Model (per wedge run, real and imaginary parts separately): y = K f + e, e ~ N(0, diag(s2)) per voxel,
prior f ~ Matern (nu = 1) on each hemisphere's midthickness surface via the SPDE / FEM construction
(Lindgren et al. 2011): precision Q = tau^2 (k^2 C + G) C^-1 (k^2 C + G), range rho = sqrt(8)/k, variance 1 / (4 pi k^2 tau^2).
Posterior mean f = (K' D K + Q)^-1 K' D y, D = diag(1/s2), solved by block preconditioned CG (K has ~4e8 nonzeros, so
K' D K cannot be formed). Unknowns are the column means (b0) of the atlasroi vertices of both hemispheres.
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_forward_operator'))
from w2lib import *

WEDGE = {'ccw': 0, 'cw': 300}        # frame offsets of RETCCW_AP and RETCW_PA in the concatenated series
BLANK, NWIN, CYC = 22, 256, 8        # 22 s blank, then 8 cycles x 32 s (TR 1 s); stimulus = DFT bin 8 of the window
NOISE_BINS = [4, 5, 6, 7, 9, 10, 11, 12]   # neighbouring frequencies (no stimulus harmonics): noise power at the stimulus bin
# Week 13: the same phase-encoded analysis for the ring runs (RETEXP_AP, RETCON_PA; measured period 32 s, src/13_hcp_cohort/
# ring_freq_check.py) and for parts of a run (noisier sources: first or second half, 4 cycles). Defaults reproduce weeks 2-9.
STIMULI = {'wedge': dict(runs={'ccw': 0, 'cw': 300}, w16tag={'ccw': 'gv', 'cw': 'gvcw'}, maps='week2_wedge', summary='week2_summary'),
           'ring': dict(runs={'exp': 600, 'con': 900}, w16tag={'exp': 'gvexp', 'con': 'gvcon'}, maps='week2_ring', summary='week13_ring')}
SEGMENTS = {'full': (0, 256, 8, [4, 5, 6, 7, 9, 10, 11, 12]),      # (offset after the blank, window length, stimulus bin, noise bins)
            'half1': (0, 128, 4, [2, 3, 5, 6]), 'half2': (128, 128, 4, [2, 3, 5, 6])}


def stim_runs(stimulus='wedge'):
    """(runs, other, w16tag) of a stimulus: run names with frame offsets, the other run of each, the goodvoxels weight tag."""
    st = STIMULI[stimulus]; names = list(st['runs'])
    return st['runs'], {names[0]: names[1], names[1]: names[0]}, st['w16tag']
GEO_THR = 15.0                       # 'far cortex' = other hemisphere or > 15 mm geodesic on the sphere (weeks 1-2)
COH_THR = 0.3


# ----------------------------------------------------------------------------- surface FEM and SPDE prior
def fem_matrices(V, F):
    """Lumped mass (vertex areas, mm^2) and cotangent stiffness matrix of a triangle mesh."""
    n = len(V)
    e0, e1, e2 = V[F[:, 2]] - V[F[:, 1]], V[F[:, 0]] - V[F[:, 2]], V[F[:, 1]] - V[F[:, 0]]
    cr = np.cross(e1, e2); dbl = np.maximum(np.linalg.norm(cr, axis=1), 1e-12)
    cot = lambda a, b: -np.einsum('ij,ij->i', a, b) / dbl      # cot of the angle between edges meeting at a vertex
    c0, c1, c2 = cot(e1, e2), cot(e2, e0), cot(e0, e1)           # angles at vertices 0, 1, 2
    I = np.concatenate([F[:, 1], F[:, 2], F[:, 0]]); J = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    w = 0.5 * np.concatenate([c0, c1, c2])                       # edge opposite a vertex gets half its cotangent
    G = sp.coo_matrix((-w, (I, J)), shape=(n, n)).tocsr(); G = G + G.T
    G = (G - sp.diags(np.asarray(G.sum(1)).ravel())).tocsr()
    area = 0.5 * dbl
    C = np.zeros(n)
    for k in range(3):
        np.add.at(C, F[:, k], area / 3)
    return C, G


def roi_submesh(V, F, roi):
    """Faces whose three vertices are in the ROI, re-indexed (natural boundary at the ROI edge). Vertices of the ROI that
    belong to no such face (isolated at the edge; zero lumped area) are dropped. Returns vertices, faces, kept indices."""
    Fr = F[roi[F].all(1)]
    idx = np.unique(Fr); loc = -np.ones(len(V), int); loc[idx] = np.arange(len(idx))
    return V[idx], loc[Fr], idx


def analysis_vertices(geo, ymax=None):
    """Unknowns per hemisphere: atlasroi vertices (optionally with MNI y < ymax) that lie on at least one ROI face."""
    out = {}
    for h in 'LR':
        m = geo[h]['roi'].copy()
        if ymax is not None:
            m &= geo[h]['mid'][:, 1] < ymax
        _, _, idx = roi_submesh(geo[h]['mid'], geo[h]['faces'], m)
        out[h] = np.zeros(len(m), bool); out[h][idx] = True
    return out


def spde_precision(C, G, rho, sigma):
    """Matern(nu=1) precision on a surface: range rho (mm), marginal s.d. sigma (data units)."""
    k2 = 8.0 / rho ** 2
    tau2 = 1.0 / (4 * np.pi * k2 * sigma ** 2)
    A = (k2 * sp.diags(C) + G).tocsr()
    return (tau2 * (A @ sp.diags(1.0 / C) @ A)).tocsr()


# ----------------------------------------------------------------------------- voxel-level data
def voxel_wedge_data(S, grids, workers=8, chunk=200000, runs=None, segment='full'):
    """Per voxel and run: complex response at the stimulus frequency, noise power at that frequency (mean of the neighbouring
    bins), and temporal mean; at 1.6 mm and the emulated grids. Also the in-mask fraction on each grid.
    runs: {name: frame offset} (default the wedge runs); segment: 'full' (as weeks 2-9), 'half1' or 'half2' (week 13)."""
    runs = WEDGE if runs is None else runs
    off, nwin, cyc, nbins = SEGMENTS[segment]
    img = nib.load(f'{HCP}/{S}/MNINonLinear/Results/{CONCAT}/{CONCAT}_hp2000_clean.nii.gz')
    out = {}
    for k, (run, o) in enumerate(runs.items()):
        Y = np.ascontiguousarray(np.asarray(img.dataobj[..., o + BLANK + off:o + BLANK + off + nwin], dtype=np.float32))
        ser = {'1.6': Y}
        ser.update(emulate_series(Y, {g: grids[g]['shape'] for g in grids if g != '1.6'}, workers=workers))
        if k == 0:          # mask: voxels with data at 1.6 mm; on coarse grids, voxels at least 90% inside that mask
            m16 = (np.abs(Y).max(3) > 0)
            out['mask'] = {'1.6': m16.ravel()}
            for g in grids:
                if g != '1.6':
                    out['mask'][g] = emulate(m16.astype(np.float32), grids[g]['shape']).ravel() > 0.9
        for g, X in ser.items():
            X = X.reshape(-1, nwin)
            yv = np.zeros(len(X), np.complex64); nv = np.zeros(len(X), np.float32); mv = np.zeros(len(X), np.float32)
            for c0 in range(0, len(X), chunk):
                Xi = X[c0:c0 + chunk].astype(np.float64); mi = Xi.mean(1)
                Fx = np.fft.rfft(Xi - mi[:, None], axis=1)
                yv[c0:c0 + chunk] = Fx[:, cyc]; nv[c0:c0 + chunk] = np.mean(np.abs(Fx[:, nbins]) ** 2, axis=1); mv[c0:c0 + chunk] = mi
            out[(g, run)] = dict(y=yv, noise=nv, mean=mv)
        del Y, ser
    return out


def _pad_axis(X, axis, n):
    n2 = X.shape[axis]
    Xs = np.fft.fftshift(X, axes=axis)
    shape = list(X.shape); shape[axis] = n
    out = np.zeros(shape, X.dtype)
    lo = n // 2 - n2 // 2
    sl = [slice(None)] * X.ndim; sl[axis] = slice(lo, lo + n2)
    out[tuple(sl)] = Xs
    return np.fft.ifftshift(out, axes=axis)


def upsample(vol, fine_shape, workers=1):
    """Band-limited (zero-padded k-space) interpolation of a coarse volume onto the fine grid; inverse of emulate's crop."""
    X = sfft.fftn(vol, axes=(0, 1, 2), workers=workers)
    for ax in range(3):
        X = _pad_axis(X, ax, fine_shape[ax])
    scale = np.prod([fine_shape[a] / vol.shape[a] for a in range(3)])
    return sfft.ifftn(X, axes=(0, 1, 2), workers=workers) * scale


# ----------------------------------------------------------------------------- restricted operator and solver
class Problem:
    """Posterior-mean problem on one grid: unknowns = ROI vertices of both hemispheres (L then R)."""

    def __init__(self, K, valid, noise, Cr, Gr, nroi, cols=None, max_outside=0.01):
        """K: CSR (all voxels x all modelled columns, float32); valid: voxel mask; noise: per-voxel noise power (complex
        coefficient); Cr, Gr: per hemisphere lumped mass and stiffness of the unknowns' submesh; nroi: unknowns per hemisphere.
        cols: indices of the columns that are unknowns (default all). Voxels receiving more than max_outside of their
        modelled signal from columns outside `cols` are dropped (their data would be explained by cortex we do not estimate)."""
        inside = np.ones(K.shape[1], np.float32) if cols is None else np.isin(np.arange(K.shape[1]), cols).astype(np.float32)
        Ka = abs(K)                                   # absolute contributions (sidelobes count as signal, not as zero)
        m_in = np.asarray(Ka @ inside).ravel(); m_out = np.asarray(Ka @ (1 - inside)).ravel(); del Ka
        self.rows = np.flatnonzero(valid & (m_in > 0) & (m_out <= max_outside * m_in))
        self.K = K[self.rows].tocsr() if cols is None else K[self.rows][:, cols].tocsr()
        self.Kt = self.K.T.tocsr()
        s2 = noise[self.rows].astype(np.float64)
        s2 = np.maximum(s2, np.percentile(s2[s2 > 0], 10))
        self.D = (1.0 / s2).astype(np.float32)
        self.Cr, self.Gr, self.nroi = Cr, Gr, nroi
        self.n = self.K.shape[1]
        self.lump = np.maximum(np.asarray(self.Kt @ (self.D * (self.K @ np.ones(self.n, np.float32)))).ravel().astype(np.float64), 0)

    def KtDK(self, X):
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        return np.asarray(self.Kt @ (self.D[:, None] * (self.K @ X32)), dtype=np.float64)

    def set_prior(self, rho, sigma):
        """Block-diagonal Matern precision (one block per hemisphere) and the preconditioner Q + diag(K'DK 1), factorised."""
        self.set_prior_blocks([(self.Cr[h], self.Gr[h], rho, sigma) for h in 'LR'])

    def set_prior_blocks(self, blocks):
        """General block-diagonal prior, blocks = [(C, G, rho, sigma), ...] in column order (e.g. b0 L, b0 R, b1 L, b1 R)."""
        self.blocks = blocks
        self.Q = sp.block_diag([spde_precision(C, G, r, s) for C, G, r, s in blocks], format='csr')
        M = (self.Q + sp.diags(self.lump)).tocsc()
        self.Mlu = spla.splu(M, permc_spec='COLAMD')   # MMD ordering is ~100x slower on these 2-ring matrices

    def prior_samples(self, m, rng):
        """m draws from N(0, Q^-1): per block Q = tau^2 A C^-1 A with A = k^2 C + G, so x = A^-1 C^(1/2) z / tau."""
        out = []
        for C, G, rho, sigma in self.blocks:
            k2 = 8.0 / rho ** 2; tau = np.sqrt(1.0 / (4 * np.pi * k2 * sigma ** 2))
            lu = spla.splu((k2 * sp.diags(C) + G).tocsc(), permc_spec='COLAMD')
            out.append(lu.solve(np.sqrt(C)[:, None] * rng.normal(size=(len(C), m))) / tau)
        return np.vstack(out)

    def uncertainty(self, m, rng, tol=1e-3):
        """Per-unknown posterior and noise-only s.d. (complex-amplitude units, as H is written) from m exact perturbation-
        optimisation samples: delta = H^-1 (K'D e + Q f0) with e ~ N(0, s2), f0 ~ N(0, Q^-1) has covariance H^-1;
        H^-1 K'D e alone has the noise-propagation covariance. Returns (posterior sd, noise sd, iterations)."""
        e = rng.normal(size=(len(self.rows), m)) * np.sqrt(1.0 / self.D)[:, None]
        KtDe = np.asarray(self.Kt @ (self.D[:, None] * e.astype(np.float32)), dtype=np.float64)
        R = np.concatenate([KtDe + self.Q @ self.prior_samples(m, rng), KtDe], 1)
        X, it, rel = self.solve(R, tol=tol, maxit=400)
        return np.sqrt((X[:, :m] ** 2).mean(1)), np.sqrt((X[:, m:] ** 2).mean(1)), it

    def H(self, X):
        return self.KtDK(X) + self.Q @ X

    def rhs(self, ydata):
        """K' D y for voxel data y (complex, all voxels) -> real (n x 2) [re, im]."""
        y = ydata[self.rows]
        B = self.Kt @ (self.D[:, None] * np.stack([y.real, y.imag], 1).astype(np.float32))
        return np.asarray(B, dtype=np.float64)

    def solve(self, B, tol=1e-4, maxit=400, X0=None, verbose=False):
        """Block PCG, columns independent. Returns X, iterations, final relative residuals."""
        X = np.zeros_like(B) if X0 is None else X0.copy()
        R = B - self.H(X) if X0 is not None else B.copy()
        Z = self.Mlu.solve(R); P = Z.copy()
        rz = np.einsum('ij,ij->j', R, Z); bn = np.maximum(np.linalg.norm(B, axis=0), 1e-30)
        t0 = time.time()
        for it in range(1, maxit + 1):
            AP = self.H(P)
            alpha = rz / np.einsum('ij,ij->j', P, AP)
            X += P * alpha; R -= AP * alpha
            rel = np.linalg.norm(R, axis=0) / bn
            if verbose and (it % 10 == 0 or it == 1):
                print(f'    PCG it {it:4d}  max rel residual {rel.max():.2e}  ({time.time() - t0:.0f} s)', flush=True)
            if (rel < tol).all():
                break
            Z = self.Mlu.solve(R); rzn = np.einsum('ij,ij->j', R, Z)
            P = Z + P * (rzn / rz); rz = rzn
        return X, it, rel


def surface_smooth(Cr, Gr, fwhm):
    """Implicit heat-equation smoothing on each hemisphere's ROI submesh, (C + t G)^-1 C with t = s^2 / 2, s = fwhm / 2.355.
    Returns a function for per-ROI-vertex arrays (L then R stacked; real or complex, 1-D)."""
    t = (fwhm / 2.3548) ** 2 / 2
    lus = {h: spla.splu((sp.diags(Cr[h]) + t * Gr[h]).tocsc()) for h in 'LR'}
    nL = len(Cr['L'])

    def real(x):
        return np.concatenate([lus['L'].solve(Cr['L'] * x[:nL]), lus['R'].solve(Cr['R'] * x[nL:])])

    def apply(x):
        return real(x.real) + 1j * real(x.imag) if np.iscomplexobj(x) else real(x)
    return apply


# ----------------------------------------------------------------------------- scoring helpers
def resid(z, p):
    """Residual energy fraction of z after the best complex scaling of predictor p."""
    c = np.vdot(p, z) / max(np.vdot(p, p).real, 1e-30)
    return float(np.sum(np.abs(z - c * p) ** 2) / np.sum(np.abs(z) ** 2))


def polar_angle(zccw, zcw, dhat):
    """Polar angle (up to a global offset) from the two wedge runs, per-vertex delay branch closest to dhat (week 2)."""
    a1, a2 = np.angle(zccw), np.angle(zcw)
    dv = np.angle(np.exp(-1j * (a1 + a2))) / 2; alt = dv + np.pi
    dv = np.where(np.abs(np.angle(np.exp(1j * (alt - dhat)))) < np.abs(np.angle(np.exp(1j * (dv - dhat)))), alt, dv)
    return np.angle(np.exp(1j * (-a1 - dv)))


def opposite_bank_pairs(geo, sel, dmax=4.0):
    """For selected vertices of one hemisphere, the nearest vertex (Euclidean, midthickness) that is geodesically far
    (> GEO_THR mm on the sphere): the other side of the sulcus. Returns (i, j, distance) for pairs closer than dmax."""
    from scipy.spatial import cKDTree
    mid, roi, s = geo['mid'], geo['roi'], geo['sphere']
    su = s / np.linalg.norm(s, axis=1, keepdims=True); Rs = np.linalg.norm(s, axis=1).mean()
    am, _ = vertex_areas_normals(mid, geo['faces'])
    fs = Rs * np.sqrt(am[roi].sum() / (4 * np.pi * Rs ** 2))
    cand = np.flatnonzero(roi); tree = cKDTree(mid[cand])
    src = np.flatnonzero(sel)
    d, k = tree.query(mid[src], k=256, distance_upper_bound=dmax)
    ok = np.isfinite(d)
    j = np.where(ok, cand[np.minimum(k, len(cand) - 1)], -1)
    geo_d = fs * np.arccos(np.clip(np.einsum('nd,nkd->nk', su[src], su[np.maximum(j, 0)]), -1, 1))
    far = ok & (geo_d > GEO_THR)
    first = np.where(far.any(1), far.argmax(1), -1)
    keep = first >= 0
    return src[keep], j[keep, first[keep]], d[keep, first[keep]]
