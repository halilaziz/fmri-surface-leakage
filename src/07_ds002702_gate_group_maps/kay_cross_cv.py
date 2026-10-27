"""Week 7, ds002702 real cross-session test, second analysis (added after the pre-specified one showed that settings chosen within
a session do not transfer across sessions): every method's setting, and the gate threshold, chosen on the OTHER subject of the
same pair type and stage (3T -> 7T: sub-01 <-> sub-03; SE -> GE: sub-01 <-> sub-04). Also the far-leak bins with those choices.
Usage (xcp_d container): python3 kay_cross_cv.py CROSS_DIR OUTDIR
"""
import os, sys, json, glob
import numpy as np

CD, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]
BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.01)]


def resid_multi(T, P, sel):
    num = den = 0.0
    for j in range(T.shape[1]):
        z, p = T[sel, j], P[sel, j]; c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


def gated(z, mk, ck, tau, sel):
    m, c = z[mk].astype(float), z[ck].astype(float)
    k = np.array([(m[sel, q] @ c[sel, q]) / max(m[sel, q] @ m[sel, q], 1e-30) for q in range(m.shape[1])])
    return resid_multi(z['target'], np.where((z['far_s'] > tau)[:, None], m * k, c), sel)


cells = {}
for p in sorted(glob.glob(f'{CD}/sub-*/*.kay_cross.npz')):
    b = os.path.basename(p).split('.'); S, src = b[0], b[1].split('-')[0]; stage = 'field-map corrected' if b[2] == 'dc' else 'uncorrected'
    cells.setdefault((src, stage), {})[S] = np.load(p)
T = ['# ds002702 cross-session test with settings chosen on the other subject of the same pair type (second analysis)', '',
     'Residual of the 7T 0.8 mm target maps (lower is better). Each method\'s setting (competitor among source-map smoothing 0–8 mm '
     'and resampling to the target grid ± 2 / 4 mm; model among the emulation\'s prior grid) and the gate threshold are chosen on '
     'the other subject of the pair type, in the same stage and vertex set.', '',
     '| source → target | stage | vertices | subject | competitor (chosen on other) | model (chosen on other) | gated | model − competitor | target split-half |',
     '|---|---|---|---|---|---|---|---|---|']
res = {}
for (src, stage), Z in sorted(cells.items()):
    subs = sorted(Z)
    if len(subs) != 2:
        continue
    z0 = Z[subs[0]]
    comps = [k for k in z0.files if k.startswith(('smooth', 'resample', 'reslin'))]
    models = [k for k in z0.files if k.startswith(('white_', 'corr_'))]
    for st in ('visual', 'covered'):
        sel = {S: (Z[S]['base'] & Z[S]['visual']) if st == 'visual' else Z[S]['base'] for S in subs}
        sc = {S: {k: resid_multi(Z[S]['target'], Z[S][k], sel[S]) for k in comps + models} for S in subs}
        for i, S in enumerate(subs):
            O = subs[1 - i]
            ck = min(comps, key=sc[O].get); mk = min(models, key=sc[O].get)
            tau = TAUS[int(np.argmin([gated(Z[O], mk, ck, t, sel[O]) for t in TAUS]))]
            z = Z[S]; g = gated(z, mk, ck, tau, sel[S])
            ceil = 0.5 * (resid_multi(z['target_B'], z['target_A'], sel[S]) + resid_multi(z['target_A'], z['target_B'], sel[S]))
            bins = {}
            clean = sel[S] & (z['far_t'] < 0.02)
            for lo, hi in BINS:
                s_ = clean & (z['far_s'] >= lo) & (z['far_s'] < hi)
                if s_.sum() >= 30:
                    bins[f'{int(100 * lo)}-{int(100 * hi)}%'] = dict(n=int(s_.sum()), d=resid_multi(z['target'], z[mk], s_) - resid_multi(z['target'], z[ck], s_))
            r = dict(competitor=sc[S][ck], competitor_key=ck, model=sc[S][mk], model_key=mk, gated=g, tau=tau, split_half=ceil, bins=bins,
                     n=int(sel[S].sum()))
            res[f'{src}|{stage}|{st}|{S}'] = r
            T.append(f"| {src.replace('ring', '')} → 7T 0.8 mm | {stage} | {st} | {S} | {r['competitor']:.4f} ({ck}) | {r['model']:.4f} ({mk}) | "
                     f"{g:.4f} (τ {tau:g}) | {r['model'] - r['competitor']:+.4f} | {ceil:.4f} |")
T += ['', '## Model − competitor (settings chosen on the other subject) by far leak of standard mapping at the source grid (visual, clean at the target)', '',
      '| source | stage | subject | ' + ' | '.join(f'{int(100 * lo)}–{int(100 * hi)}%' for lo, hi in BINS) + ' |', '|---|---|---|' + '---|' * len(BINS)]
for k, r in res.items():
    src, stage, st, S = k.split('|')
    if st == 'visual':
        T.append(f"| {src.replace('ring', '')} | {stage} | {S} | " + ' | '.join(
            (f"{r['bins'][b]['d']:+.4f} (n {r['bins'][b]['n']})" if b in r['bins'] else '–') for b in [f'{int(100 * lo)}-{int(100 * hi)}%' for lo, hi in BINS]) + ' |')
T.append('')
open(f'{OUT}/kay_cross_cv_tables.md', 'w').write('\n'.join(T) + '\n')
json.dump(res, open(f'{OUT}/kay_cross_cv_results.json', 'w'), indent=1, default=float)
print('\n'.join(T))
