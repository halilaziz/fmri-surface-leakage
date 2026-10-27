"""Week 6: correlated-noise likelihood for the posterior mean.
Noise model on a voxel grid: e ~ N(0, S^1/2 C S^1/2), S = per-voxel marginal variance (as in week 5), C = stationary correlation
(circulant on the grid) with spectrum P(k), mean_k P = 1 (unit diagonal). P is estimated from the source's own run residuals
(unit variance per voxel), radially averaged in physical frequency and floored. C^-1 is applied with FFTs on the full grid;
voxels outside the data mask are zero-filled (an approximation near the mask edge).
The posterior mean solves (K' S^-1/2 C^-1 S^-1/2 K + Q) f = K' S^-1/2 C^-1 S^-1/2 y by block PCG (w3lib.Problem machinery).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '05_studyforrest_within_session'))
from w5lib import *

P_FLOOR = 0.1          # floor of the normalised noise spectrum (the emulation window leaves ~0.07-0.1 of white near Nyquist)


def noise_spectrum(w, valid, vox, shape, nbins=40, floor=P_FLOOR):
    """w: list of unit-variance residual volumes (real, 3-D, zero outside `valid`). Returns the normalised spectrum on the
    rfftn grid of `shape` and the radial profile (k in cycles/mm, power relative to white)."""
    P = np.zeros(tuple(shape[:2]) + (shape[2] // 2 + 1,))
    for x in w:
        P += np.abs(np.fft.rfftn(x)) ** 2
    P /= len(w) * valid.sum()
    k = np.meshgrid(np.fft.fftfreq(shape[0], vox[0]), np.fft.fftfreq(shape[1], vox[1]), np.fft.rfftfreq(shape[2], vox[2]), indexing='ij')
    kr = np.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)
    edges = np.linspace(0, kr.max() + 1e-9, nbins + 1)
    b = np.clip(np.digitize(kr, edges) - 1, 0, nbins - 1)
    # rfftn keeps half of the last axis: weight interior planes twice for the radial mean
    wgt = np.ones_like(P); wgt[..., 1:(shape[2] + 1) // 2] = 2.0
    prof = np.array([np.sum((P * wgt)[b == i]) / max(np.sum(wgt[b == i]), 1e-12) for i in range(nbins)])
    kc = 0.5 * (edges[1:] + edges[:-1])
    ok = np.isfinite(prof) & (prof > 0)
    Pk = np.interp(kr, kc[ok], prof[ok])
    Pk /= (np.sum(Pk * wgt) / np.sum(wgt))                         # mean over the full spectrum = 1  <=>  unit diagonal
    Pk = np.maximum(Pk, floor)
    Pk /= (np.sum(Pk * wgt) / np.sum(wgt))
    return Pk, kc, prof


class ProblemCorr(Problem):
    """Problem with the correlated-noise likelihood; `noise` is the per-voxel marginal variance (as for Problem).
    Unlike Problem, every valid voxel is kept, including voxels without cortex (K row = 0): under correlated noise their data are
    noise references that inform the correlated component at the cortical voxels."""

    def __init__(self, K, valid, noise, Cr, Gr, nroi, shape, Pk):
        self.rows = np.flatnonzero(valid)
        self.K = K[self.rows].tocsr(); self.Kt = self.K.T.tocsr()
        s2 = noise[self.rows].astype(np.float64)
        s2 = np.maximum(s2, np.percentile(s2[s2 > 0], 10))
        self.D = (1.0 / s2).astype(np.float32)
        self.Cr, self.Gr, self.nroi = Cr, Gr, nroi
        self.n = self.K.shape[1]
        self.lump = np.maximum(np.asarray(self.Kt @ (self.D * (self.K @ np.ones(self.n, np.float32)))).ravel().astype(np.float64), 0)
        self.shape = tuple(shape); self.iPk = 1.0 / Pk
        self.sqD = np.sqrt(self.D).astype(np.float64)
        wgt = np.ones_like(Pk); wgt[..., 1:(self.shape[2] + 1) // 2] = 2.0
        self.lump *= float(np.sum(self.iPk * wgt) / np.sum(wgt))   # diag(C^-1) = mean_k 1/P for the preconditioner
        self.nvox = int(np.prod(self.shape))

    def Cinv(self, V):
        """C^-1 applied to values on the problem's rows (one column per map), via the full grid."""
        out = np.empty((V.shape[0], V.shape[1]), np.float64)
        for j in range(V.shape[1]):
            g = np.zeros(self.nvox); g[self.rows] = V[:, j]
            G = np.fft.irfftn(np.fft.rfftn(g.reshape(self.shape)) * self.iPk, s=self.shape)
            out[:, j] = G.ravel()[self.rows]
        return out

    def KtDK(self, X):
        KX = np.asarray(self.K @ np.ascontiguousarray(X, dtype=np.float32), np.float64)
        Z = self.sqD[:, None] * self.Cinv(self.sqD[:, None] * KX)
        return np.asarray(self.Kt @ Z.astype(np.float32), dtype=np.float64)

    def rhs(self, ydata):
        y = ydata[self.rows]
        Y = np.stack([y.real, y.imag], 1).astype(np.float64)
        Z = self.sqD[:, None] * self.Cinv(self.sqD[:, None] * Y)
        return np.asarray(self.Kt @ Z.astype(np.float32), dtype=np.float64)


def unit_residuals(yruns, valid):
    """yruns: complex (x, y, z, runs) data of one half; residuals about the half mean, each part scaled to unit variance per voxel."""
    out = []
    for part in (np.real, np.imag):
        Y = part(yruns); r = Y - Y.mean(-1, keepdims=True)
        sd = np.sqrt((r ** 2).sum(-1) / (Y.shape[-1] - 1))
        ok = valid & (sd > 0)
        out += [np.where(ok, r[..., i] / np.where(ok, sd, 1.0), 0.0) for i in range(Y.shape[-1])]
    return out
