"""Week 7, ds002702: pre-specify the settings of the real cross-session test from the within-session emulation only.
For each emulation factor (source voxel / 0.8 mm: x3 for 3T 2.4 mm, x1.3125 for 7T SE 1.05 mm), the competitor and the model
setting with the lowest mean residual over all emulated subjects (visual vertices; mean of the two directions per subject) are
chosen, together with the gate threshold of the leak-gated estimator. Written before kay_cross.py results are looked at.
Usage (xcp_d container): python3 kay_prespec.py EMUL_DIR OUT_JSON
"""
import sys, os, re, json, glob, time, hashlib
import numpy as np

ED, OJ = sys.argv[1:3]
COMP = ['upsample', 'upsmooth2', 'upsmooth4', 'smooth4', 'smooth8']
TO_CROSS = {'upsample': 'resample', 'upsmooth2': 'resample2', 'upsmooth4': 'resample4', 'smooth4': 'smooth4', 'smooth8': 'smooth8'}
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]


def resid_multi(T, P, sel):
    num = den = 0.0
    for j in range(T.shape[1]):
        z, p = T[sel, j], P[sel, j]; c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), emulation_dir=ED, factors={})
files = sorted(glob.glob(f'{ED}/sub-*/*.kay_emul.npz'))
byfac = {}
for p in files:
    byfac.setdefault(re.search(r'\.x([0-9.]+?)\.kay_emul\.npz$', p).group(1), []).append(p)
for fac, ps in sorted(byfac.items()):
    Z = [np.load(p) for p in ps]
    mk = sorted(k[3:] for k in Z[0].files if k.startswith('AB_') and k[3:].startswith(('corr_', 'white_')))
    res = {}
    for st in ('visual', 'covered'):
        sel = lambda z, d: z[f'{d}_base'] & z['visual'] if st == 'visual' else z[f'{d}_base']
        sc = lambda key: np.array([np.mean([resid_multi(z[f'{d}_target'], z[f'{d}_{key}'], sel(z, d)) for d in ('AB', 'BA')]) for z in Z])
        C = {k: sc(k) for k in COMP}; M = {k: sc(k) for k in mk}
        bc = min(C, key=lambda k: C[k].mean()); bm = min(M, key=lambda k: M[k].mean())
        # gate threshold for the chosen pair (model scaled onto the competitor without target data)
        G = np.zeros((len(Z), len(TAUS)))
        for i, z in enumerate(Z):
            for j, tau in enumerate(TAUS):
                v = []
                for d in ('AB', 'BA'):
                    s_ = sel(z, d); m, c = z[f'{d}_{bm}'].astype(float), z[f'{d}_{bc}'].astype(float)
                    k = np.array([(m[s_, q] @ c[s_, q]) / max(m[s_, q] @ m[s_, q], 1e-30) for q in range(m.shape[1])])
                    v.append(resid_multi(z[f'{d}_target'], np.where((z['far_c'] > tau)[:, None], m * k, c), s_))
                G[i, j] = np.mean(v)
        bt = TAUS[int(np.argmin(G.mean(0)))]
        res[st] = dict(competitor=bc, competitor_cross=TO_CROSS[bc], model=bm, tau=bt,
                       emul_competitor={k: float(v.mean()) for k, v in C.items()}, emul_model={k: float(v.mean()) for k, v in M.items()},
                       emul_gate_curve=G.mean(0).tolist())
    out['factors'][fac] = dict(subjects=[os.path.basename(p).split('.')[0] for p in ps], **res,
                               files_sha1={os.path.basename(p): hashlib.sha1(open(p, 'rb').read()).hexdigest() for p in ps})
    print(fac, {st: (r['competitor_cross'], r['model'], r['tau']) for st, r in res.items()}, flush=True)
json.dump(out, open(OJ, 'w'), indent=1)
