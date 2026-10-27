"""MVE week 1, part 1: realistic resolvability lookup tables.

For every sheet orientation (normal in the positive octant; the encodings are symmetric under sign flips)
and each HCP encoding (3T: 2 mm, PE along x; 7T: 1.6 mm, PE along y; both 2D multislice with EPI T2*
apodisation), compute with the planar-patch simulator from theory_sims/sheet_sampling.py:
  - resolvable isotropic tangential wavelength (worst direction, transfer >= 0.5) and best-direction wavelength
  - degrees of freedom for signal per cm^2
for three regimes: acquisition-limited single-frame (white prior, CNR 1), acquisition-limited map-level
(white prior, CNR 10) and physiology-limited map-level (BOLD-PSF-limited prior, CNR 10).
Also a two-bank table (mid-surface separation d) per encoding for CNR 1 and 10.

Usage: pyrun lookup_sims.py OUTDIR [--workers 28] [--smoke]
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', '..', 'theory_sims'), os.path.expanduser('~/theory_sims')):
    if os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))
        break
from sheet_sampling import (build_K, cov_from_spectrum, spectrum, averaging_ops, plane_waves, transfer,
                            resolvable_wavelengths, axis_weights, orthobasis)

N_OFFSETS = 6   # random lattice positions averaged in the two-bank table

ACQS = {  # HCP-YA protocol values (S1200 manual); T2* of grey matter is an assumption
    '3T': dict(h=2.0, N=25, pe_axis=0, t2s=45.0, esp=0.58, fov_pe=180.0, fw_bold=3.0),
    '7T': dict(h=1.6, N=31, pe_axis=1, t2s=28.0, esp=0.32, fov_pe=208.0, fw_bold=1.5),
}
CONFIGS = {'white_cnr1': ('white', 1.0), 'white_cnr10': ('white', 10.0), 'bold_cnr10': ('bold', 10.0)}
BANK_D = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
BANK_CNR = [1.0, 10.0]
BANK_QS = [0.05, 0.4, 0.8, 1.2, 1.6]          # rad/mm -> 126, 15.7, 7.9, 5.2, 3.9 mm


def octant_dirs(n_full=600):
    i = np.arange(n_full) + 0.5
    phi = np.arccos(1 - 2 * i / n_full)
    th = np.pi * (1 + 5 ** 0.5) * i
    P = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    P = P[(P >= 0).all(1)]
    extra = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], float)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    return np.vstack([P, extra])


def acq_weights(a):
    return axis_weights(a['N'], a['N'] * a['h'], kind='t2s', t2s_ms=a['t2s'], esp_eff_ms=a['esp'], fov_pe_real=a['fov_pe'])


def crossing_wavelengths(tau, qs, ths, thr=0.5):
    """Per direction: wavelength where transfer first drops below thr, with linear interpolation between q samples."""
    tau = tau.reshape(len(qs), len(ths))
    lam = np.full(len(ths), np.inf)
    for j in range(len(ths)):
        t = tau[:, j]
        below = np.flatnonzero(t < thr)
        if len(below) == 0:
            lam[j] = 2 * np.pi / qs[-1]
            continue
        k = below[0]
        if k == 0:
            continue
        qx = qs[k - 1] + (t[k - 1] - thr) / (t[k - 1] - t[k]) * (qs[k] - qs[k - 1])
        lam[j] = 2 * np.pi / qx
    return lam


def dir_task(args):
    acq, cfg, n, smoke = args
    a = ACQS[acq]
    kind, cnr = CONFIGS[cfg]
    M, delta = (24, 0.75) if smoke else (64, 0.5)
    sheet = dict(n=tuple(n), offset=0.0, T=2.5, nd=6, delta=delta, M=M)
    K = build_K([sheet], h=a['h'], N=(13 if smoke else a['N']), enc='2D', pe_axis=a['pe_axis'], pe_W=acq_weights(a) if not smoke else None)
    prior = spectrum('white') if kind == 'white' else spectrum('white_bold', fwhm_bold=a['fw_bold'])
    C = cov_from_spectrum(prior, delta, M)
    applyA, s, dfs = averaging_ops(K, C, cnr, None)
    qs = np.linspace(0.05, 4.2, 12 if smoke else 72)
    ths = np.linspace(0, np.pi, 6 if smoke else 24, endpoint=False)
    tau = transfer(applyA, plane_waves(sheet, qs, ths, R=(4.0 if smoke else 9.0)))
    lam = crossing_wavelengths(tau, qs, ths)
    iso, best = float(np.max(lam)), float(np.min(lam))
    return dict(acq=acq, cfg=cfg, n=list(map(float, n)), iso_mm=iso, best_mm=best, lam_by_angle=[float(x) for x in lam],
                dfs_per_cm2=float(dfs / (M * M * delta * delta) * 100))


def box_weights(sheets, h, N, x0):
    """Voxel x vertex overlap counts of each vertex column's depth samples (non-negative, like ribbon mapping)."""
    rows, cols = [], []
    off = 0
    for sh in sheets:
        e1, e2, nn = orthobasis(sh['n'])
        M, delta, T, nd = sh['M'], sh['delta'], sh['T'], sh['nd']
        u = (np.arange(M) - (M - 1) / 2) * delta
        U1, U2 = np.meshgrid(u, u, indexing='ij'); U1 = U1.ravel(); U2 = U2.ravel()
        sd = ((np.arange(nd) + 0.5) / nd - 0.5) * T + sh.get('offset', 0.0)
        P = (np.asarray(x0)[None, None, :] + U1[:, None, None] * e1 + U2[:, None, None] * e2 + sd[None, :, None] * nn)
        idx = np.rint(P / h).astype(int) + (N - 1) // 2
        ok = ((idx >= 0) & (idx < N)).all(-1)
        lin = idx[..., 0] * N * N + idx[..., 1] * N + idx[..., 2]
        jj = np.broadcast_to(np.arange(len(U1))[:, None], ok.shape)
        rows.append(lin[ok]); cols.append(jj[ok] + off)
        off += len(U1)
    Wb = np.zeros((N ** 3, off))
    np.add.at(Wb, (np.concatenate(rows), np.concatenate(cols)), 1.0)
    return Wb


def bank_task(args):
    acq, d, cnr, smoke = args
    a = ACQS[acq]
    M, delta = (16, 1.0) if smoke else (40, 0.75)
    T = min(2.5, d)
    n = (0.3, 0.5, 0.81)
    sheets = [dict(n=n, offset=+d / 2, T=T, nd=6, delta=delta, M=M), dict(n=n, offset=-d / 2, T=T, nd=6, delta=delta, M=M)]
    Nl = 13 if smoke else a['N']
    J = M * M
    C1 = cov_from_spectrum(spectrum('white_bold', fwhm_bold=a['fw_bold']), delta, M)
    C = np.zeros((2 * J, 2 * J)); C[:J, :J] = C1; C[J:, J:] = C1
    qs = np.array(BANK_QS); ths = np.linspace(0, np.pi, 6, endpoint=False)
    R = 4.0 if smoke else 9.0
    Es = plane_waves(sheets[0], qs, ths, R=R, nsheets=2, sign=+1)
    Ea = plane_waves(sheets[0], qs, ths, R=R, nsheets=2, sign=-1)
    EA = Es.copy(); EA[J:] = 0
    nrm = np.einsum('ij,ij->j', EA[:J], EA[:J])
    f = lambda v: v.reshape(len(qs), -1).mean(1)
    rng = np.random.default_rng(int(1000 * d) + (0 if acq == '3T' else 7))
    acc = {k: [] for k in ('tau_anti', 'own_opt', 'leak_opt', 'own_naive', 'leak_naive')}
    for rep in range(2 if smoke else N_OFFSETS):
        x0 = tuple(rng.uniform(-a['h'] / 2, a['h'] / 2, 3))
        K = build_K(sheets, h=a['h'], N=Nl, enc='2D', pe_axis=a['pe_axis'], pe_W=acq_weights(a) if not smoke else None, x0=x0)
        applyA, s_, dfs = averaging_ops(K, C, cnr, None)
        AEA = applyA(EA)
        acc['tau_anti'].append(f(transfer(applyA, Ea)))
        acc['own_opt'].append(f(np.einsum('ij,ij->j', EA[:J], AEA[:J]) / nrm))
        acc['leak_opt'].append(f(np.einsum('ij,ij->j', Es[J:], AEA[J:]) / nrm))
        Wb = box_weights(sheets, a['h'], Nl, x0)
        AnEA = (Wb.T @ (K @ EA)) / np.maximum(Wb.sum(0), 1e-12)[:, None]
        acc['own_naive'].append(f(np.einsum('ij,ij->j', EA[:J], AnEA[:J]) / nrm))
        acc['leak_naive'].append(f(np.einsum('ij,ij->j', Es[J:], AnEA[J:]) / nrm))
    out = {k: np.mean(v, 0).tolist() for k, v in acc.items()}
    out.update({k + '_sd': np.std(v, 0).tolist() for k, v in acc.items()})
    return dict(acq=acq, d=d, cnr=cnr, T=T, qs=BANK_QS, n_offsets=len(acc['tau_anti']), **out)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir'); ap.add_argument('--workers', type=int, default=28); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dirs = octant_dirs(60 if a.smoke else 600)
    if a.smoke:
        dirs = dirs[[0, -1]]
    dtasks = [(acq, cfg, n, a.smoke) for acq in ACQS for cfg in CONFIGS for n in dirs]
    btasks = [(acq, d, c, a.smoke) for acq in ACQS for d in (BANK_D[:2] if a.smoke else BANK_D) for c in BANK_CNR]
    print(f'{len(dtasks)} orientation tasks, {len(btasks)} two-bank tasks, {a.workers} workers', flush=True)
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.workers) as pool:
        bres = pool.map(bank_task, btasks, chunksize=1)
        print(f'two-bank done in {time.time()-t0:.0f}s', flush=True)
        dres = []
        for i, r in enumerate(pool.imap_unordered(dir_task, dtasks, chunksize=1)):
            dres.append(r)
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(dtasks)} orientation tasks ({time.time()-t0:.0f}s)', flush=True)
    json.dump(dict(acqs=ACQS, configs=CONFIGS, results=dres), open(os.path.join(a.outdir, 'lookup_dirs.json'), 'w'))
    json.dump(dict(acqs=ACQS, results=bres), open(os.path.join(a.outdir, 'lookup_banks.json'), 'w'))
    print(f'saved to {a.outdir} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
