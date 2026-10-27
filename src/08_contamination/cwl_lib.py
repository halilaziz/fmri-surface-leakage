"""Week 8 shared code: contrast-weighted leak (CWL) of standard mapping (STATS_PLAN.md, part A).
Contamination of the standard map at vertex v by far cortex responding differently from v:
    c_v,j = sum_{u in F(v)} a_vu (x_u,j - x_v,j) / sum_u a_vu,     a_v = row v of W K (readout of the unknowns)
F(v) = other hemisphere, or > 15 mm on the registration sphere scaled to the cortical area (week 2).
CWL_v = sqrt(sum_j c_v,j^2) / sqrt(mean over the scored set of sum_j x_v,j^2).
"""
import numpy as np

GEO_THR = 15.0


def sphere_frames(geo, sph, area_fn):
    """Unit sphere coordinates and the area scale factor per hemisphere (as week 2 / sf_gate.far_leak)."""
    su, fs = {}, {}
    for h in 'LR':
        s = sph[h]; Rs = np.linalg.norm(s, axis=1).mean(); su[h] = s / np.linalg.norm(s, axis=1, keepdims=True)
        a = area_fn(geo[h]['mid'], geo[h]['faces'])
        fs[h] = Rs * np.sqrt(a[geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
    return su, fs


def far_contamination(W, K, idx, nL, su, fs, Xs, chunk=3000):
    """W: per hemisphere CSR (vertices of that hemisphere x voxels); K: CSR (voxels x unknowns); idx: unknown -> global vertex
    index (L then R, R offset by nL); Xs: {name: (n_unknowns x J) float arrays}. Returns {name: (n x J) contamination} and the
    far-leak fraction (the week-2 quantity, for checking)."""
    n = len(idx); colh = idx >= nL; colloc = np.where(colh, idx - nL, idx)
    out = {k: np.zeros((n, X.shape[1])) for k, X in Xs.items()}; far_frac = np.zeros(n)
    for h in 'LR':
        sel = np.flatnonzero(colh == (h == 'R')); rows = colloc[sel]
        for c0 in range(0, len(rows), chunk):
            r = rows[c0:c0 + chunk]; sv = sel[c0:c0 + chunk]; nr = len(r)
            A = (W[h][r] @ K).tocoo()
            val = A.data.astype(np.float64); j = A.col
            other = colh[j] != (h == 'R')
            cosang = np.einsum('nd,nd->n', su[h][r[A.row]], su[h][np.where(other, 0, colloc[j])])
            far = other | (fs[h] * np.arccos(np.clip(cosang, -1, 1)) > GEO_THR)
            tot = np.bincount(A.row, val, minlength=nr); den = np.maximum(np.abs(tot), 1e-12)          # as sf_gate.far_leak
            vf = val * far; wf = np.bincount(A.row, vf, minlength=nr)
            far_frac[sv] = wf / den
            for k, X in Xs.items():
                for q in range(X.shape[1]):
                    s = np.bincount(A.row, vf * X[j, q], minlength=nr)
                    out[k][sv, q] = (s - wf * X[sv, q]) / den
    return out, far_frac


def cwl(C, X, scored):
    """Contamination norm relative to the typical response over the scored vertex set."""
    ref = np.sqrt(np.mean((X[scored] ** 2).sum(1))) if scored.sum() else np.nan
    return np.sqrt((C ** 2).sum(1)) / max(ref, 1e-30)


def split_complex(Z):
    """Complex (n,) or (n, J) maps -> real (n, 2J): real and imaginary parts as separate maps."""
    Z = Z[:, None] if Z.ndim == 1 else Z
    return np.concatenate([Z.real, Z.imag], 1).astype(np.float64)
