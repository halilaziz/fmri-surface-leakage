"""Week 7, ds002702 real cross-session test: pre-specified comparison (settings from kay_prespec.json, chosen on the within-session
emulation before these results were read), exploratory best-in-hindsight settings, the target's own split-half reference, and the
model's gain by far leak of standard mapping at the source grid.
Usage (xcp_d container): python3 summarize_kay_cross.py CROSS_DIR PRESPEC_JSON OUTDIR
"""
import os, sys, json, glob
import numpy as np

CD, PJ, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
PS = json.load(open(PJ))
FAC = {'ring3T': '3', 'ringSE': '1.3125'}
BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.01)]


def resid_multi(T, P, sel):
    num = den = 0.0
    for j in range(T.shape[1]):
        z, p = T[sel, j], P[sel, j]; c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


T = ['# ds002702 real second acquisition: source session → 7T 0.8 mm GE session (all runs), same people', '',
     'Residual energy fraction of the target maps (six ring conditions, each with its best scaling); lower is better. Settings '
     'pre-specified from the within-session emulation (kay_prespec.json, created ' + PS['created'] + '). "Best in hindsight" '
     'picks each method\'s setting on this pair itself and is exploratory. Target split-half: odd-run map vs even-run map of the '
     'target session (a reference for target noise, not a bound).', '',
     '| pair | vertices | n | competitor (pre-spec.) | model (pre-spec.) | gated (pre-spec.) | model − competitor | best competitor in hindsight | best model in hindsight | target split-half |',
     '|---|---|---|---|---|---|---|---|---|---|']
res = {}
for p in sorted(glob.glob(f'{CD}/sub-*/*.kay_cross.npz')):
    b = os.path.basename(p).split('.'); S, src = b[0], b[1].split('-')[0]
    pair = b[1] + (' (field-map corrected)' if b[2] == 'dc' else ' (uncorrected)')
    z = np.load(p); pf = PS['factors'][FAC[src]]
    comps = [k for k in z.files if k.startswith(('smooth', 'resample', 'reslin'))]
    models = [k for k in z.files if k.startswith(('white_', 'corr_'))]
    for st in ('visual', 'covered'):
        sel = z['base'] & z['visual'] if st == 'visual' else z['base']
        ch = pf[st]; ck, mk, tau = ch['competitor_cross'], ch['model'], ch['tau']
        sc = {k: resid_multi(z['target'], z[k], sel) for k in comps + models}
        m, c = z[mk].astype(float), z[ck].astype(float)
        kk = np.array([(m[sel, q] @ c[sel, q]) / max(m[sel, q] @ m[sel, q], 1e-30) for q in range(m.shape[1])])
        gated = resid_multi(z['target'], np.where((z['far_s'] > tau)[:, None], m * kk, c), sel)
        ceil = 0.5 * (resid_multi(z['target_B'], z['target_A'], sel) + resid_multi(z['target_A'], z['target_B'], sel))
        bc = min(comps, key=sc.get); bm = min(models, key=sc.get)
        bins = {}
        clean = sel & (z['far_t'] < 0.02)
        for lo, hi in BINS:
            s_ = clean & (z['far_s'] >= lo) & (z['far_s'] < hi)
            if s_.sum() >= 30:
                bins[f'{int(100 * lo)}-{int(100 * hi)}%'] = dict(n=int(s_.sum()), model=resid_multi(z['target'], z[mk], s_),
                                                                 competitor=resid_multi(z['target'], z[ck], s_))
        res[(S, pair, st)] = dict(n=int(sel.sum()), competitor=sc[ck], competitor_key=ck, model=sc[mk], model_key=mk, gated=gated, tau=tau,
                                  best_competitor=(bc, sc[bc]), best_model=(bm, sc[bm]), split_half=ceil, all=sc, bins=bins)
        r = res[(S, pair, st)]
        T.append(f"| {S} {pair} | {st} | {r['n']} | {r['competitor']:.4f} ({ck}) | {r['model']:.4f} ({mk}) | {gated:.4f} (τ {tau:g}) | "
                 f"{r['model'] - r['competitor']:+.4f} | {sc[bc]:.4f} ({bc}) | {sc[bm]:.4f} ({bm}) | {ceil:.4f} |")
T += ['', '## Model − competitor (pre-specified) by far leak of standard mapping at the source grid (visual, clean at the target)', '',
      '| pair | ' + ' | '.join(f'{int(100 * lo)}–{int(100 * hi)}%' for lo, hi in BINS) + ' |', '|---|' + '---|' * len(BINS)]
for (S, pair, st), r in res.items():
    if st != 'visual':
        continue
    cells = []
    for lo, hi in BINS:
        k = f'{int(100 * lo)}-{int(100 * hi)}%'
        cells.append(f"{r['bins'][k]['model'] - r['bins'][k]['competitor']:+.4f} (n {r['bins'][k]['n']})" if k in r['bins'] else '–')
    T.append(f'| {S} {pair} | ' + ' | '.join(cells) + ' |')
T.append('')
open(f'{OUT}/kay_cross_tables.md', 'w').write('\n'.join(T) + '\n')
json.dump({f'{S}|{pair}|{st}': r for (S, pair, st), r in res.items()}, open(f'{OUT}/kay_cross_results.json', 'w'), indent=1, default=float)
print('\n'.join(T))
