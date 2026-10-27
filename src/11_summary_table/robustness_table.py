"""Week 11: one set of statistics across datasets, as specified (and hashed) in results/week11/TABLE_SPEC.md before computing.
Per subject d = residual(model) - residual(best competitor) (< 0: model better). Per row and vertex set: n, mean, median, d < 0,
bootstrap 95% interval of the mean (10,000 resamples, seed 0 per row) and the one-sided Wilcoxon p.
Usage (local, from the project root): python3 src/11_summary_table/robustness_table.py results/week11
"""
import os, sys, json, glob, re
import numpy as np
from scipy.stats import wilcoxon

OUT = sys.argv[1]
R = 'results'
SPEC_SHA = open(f'{OUT}/TABLE_SPEC.sha1').read().split()[0]


def stats(per_subject):
    v = np.asarray([x for x in per_subject.values() if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    b = np.random.default_rng(0).choice(v, size=(10000, len(v)), replace=True).mean(1)
    p = float(wilcoxon(v, alternative='less').pvalue) if len(v) >= 2 and np.any(v != 0) else float('nan')
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), n_neg=int((v < 0).sum()),
                ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], p_one_sided=p,
                per_subject={S: float(x) for S, x in per_subject.items() if x is not None and np.isfinite(x)})


# ---------------------------------------------------------------- A1, A2, B3-B6: HCP held-out test (week 3)
T3 = json.load(open(f'{R}/week3/week3_test_scores.json')); S3 = T3['test_subjects']


def hcp(g, m, s):
    a = np.array([T3['scores'][S][g][m].get(s, np.nan) for S in S3], float)
    c = np.fmin(*[np.array([T3['scores'][S][g][k].get(s, np.nan) for S in S3], float) for k in ('smoothing', 'upsample_smoothing')])
    return dict(zip(S3, a - c))


# ---------------------------------------------------------------- B1, B2: HCP misregistration (week 4)
E4 = json.load(open(f'{R}/week4/hcp_eval/week4_hcp_eval.json')); S4 = E4['subjects']


def hcp_shift(g, s):
    sc = lambda lab: np.array([E4['shift'][S][g][lab].get(s, np.nan) if E4['shift'][S][g].get(lab) else np.nan for S in S4], float)
    return dict(zip(S4, sc('model_shift1') - np.fmin(sc('smoothing_shift1'), sc('upsample_smoothing_shift1'))))


# ---------------------------------------------------------------- A3: StudyForrest within session, correlated noise (week 6)
W6 = json.load(open(f'{R}/week6/w6_results.json'))


def sf_within(st):
    acc = {}
    for P in W6.values():
        if st in P['res']:
            r = P['res'][st]
            for S, x in zip(r['subjects'], np.array(r['corr_loso']) - np.minimum(r['smoothing_loso'], r['upsample_loso'])):
                acc.setdefault(S, []).append(float(x))
    return {S: float(np.mean(v)) for S, v in acc.items()}


# ---------------------------------------------------------------- A4-A6: ds002702 within session (procedure of src/07_ds002702_gate_group_maps/summarize_kay.py)
COMP = ['upsample', 'upsmooth2', 'upsmooth4', 'smooth4', 'smooth8']


def resid_multi(T, P, sel):
    num = den = 0.0
    for j in range(T.shape[1]):
        z, p = T[sel, j], P[sel, j]; c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


def loso(M):
    return np.array([M[i, int(np.argmin(np.delete(M, i, 0).mean(0)))] for i in range(len(M))])


cells = {}
for p in sorted(glob.glob(f'{R}/week7/ds002702/emul/sub-*/*.kay_emul.npz')):
    fac = re.search(r'\.x([0-9.]+?)\.kay_emul\.npz$', p).group(1)
    cells.setdefault(fac, {})[os.path.basename(p).split('.')[0]] = p
K7 = json.load(open(f'{R}/week7/ds002702/summary/kay_results.json'))
kay = {}
for fac, SS in sorted(cells.items(), key=lambda kv: float(kv[0])):
    subs = sorted(SS); Z = {S: np.load(SS[S]) for S in subs}
    mk = sorted(k[3:] for k in Z[subs[0]].files if k.startswith('AB_') and (k[3:].startswith('corr_') or k[3:].startswith('white_')))

    def sets(z, d):
        vis = z[f'{d}_base'] & z['visual']
        return {'visual': vis, 'high': vis & (z['far_n'] < 0.02) & (z['far_c'] >= 0.10)}
    score = lambda S, key, st: np.mean([resid_multi(Z[S][f'{d}_target'], Z[S][f'{d}_{key}'], sets(Z[S], d)[st]) for d in ('AB', 'BA')])
    kay[fac] = {}
    for st in ('visual', 'high'):
        nmin = min(int(sets(Z[S], d)[st].sum()) for S in subs for d in ('AB', 'BA'))
        if nmin < 30:
            kay[fac][st] = dict(too_few=nmin); continue
        Mc = np.array([[score(S, k, st) for k in COMP] for S in subs]); Mm = np.array([[score(S, k, st) for k in mk] for S in subs])
        kay[fac][st] = dict(zip(subs, loso(Mm) - loso(Mc)))
    # check: the visual set recomputed from the npz files equals the week-7 summary
    ref = np.array(K7[fac]['visual']['model']) - np.array(K7[fac]['visual']['comp'])
    assert np.allclose([kay[fac]['visual'][S] for S in subs], ref, atol=1e-12), f'ds002702 x{fac}: recomputation differs from kay_results.json'

# ---------------------------------------------------------------- C1-C4: second acquisitions
X8 = json.load(open(f'{R}/week8/sf_cross/sf_cross_loso.json'))['tests']
CV = json.load(open(f'{R}/week7/ds002702/summary/kay_cross_cv_results.json'))


def kay_cross(src):
    a, h = {}, {}
    for k, r in CV.items():
        s, stage, st, S = k.split('|')
        if s == src and stage == 'field-map corrected' and st == 'visual':
            a[S] = r['model'] - r['competitor']
            b = r['bins'].get('10-101%')
            h[S] = b['d'] if b else None
    return a, h


k3, k3h = kay_cross('ring3T'); kse, kseh = kay_cross('ringSE')
VOX = {'1.3125': '1.05', '2': '1.6', '3': '2.4'}
ROWS = [  # (block, id, label, setting rule, all, high mixing)
    ('A', 'A1', 'HCP 7T, emulated 2.4 mm', 'fixed on 5 other subjects', hcp('2.4', 'model', 'coherent'), hcp('2.4', 'model', 'clean16_leak10')),
    ('A', 'A2', 'HCP 7T, emulated 3.2 mm', 'fixed on 5 other subjects', hcp('3.2', 'model', 'coherent'), hcp('3.2', 'model', 'clean16_leak10')),
    ('A', 'A3', 'StudyForrest 7T, emulated 2.1–4.0 mm', 'leave-one-subject-out', sf_within('visual'), sf_within('visual_xhemi')),
] + [('A', f'A{4 + i}', f'ds002702 7T, emulated {VOX[f]} mm', 'leave-one-subject-out', kay[f]['visual'],
      kay[f]['high'] if 'too_few' not in kay[f]['high'] else None) for i, f in enumerate(sorted(kay, key=float))] + [
    ('B', 'B1', 'HCP 2.4 mm, 1 mm misregistration (all methods)', 'fixed on 5 other subjects', hcp_shift('2.4', 'coherent'), hcp_shift('2.4', 'clean16_leak10')),
    ('B', 'B2', 'HCP 3.2 mm, 1 mm misregistration (all methods)', 'fixed on 5 other subjects', hcp_shift('3.2', 'coherent'), hcp_shift('3.2', 'clean16_leak10')),
    ('B', 'B3', 'HCP 2.4 mm, model PSF 15% too narrow', 'fixed on 5 other subjects', hcp('2.4', 'model_psf0.85', 'coherent'), hcp('2.4', 'model_psf0.85', 'clean16_leak10')),
    ('B', 'B4', 'HCP 2.4 mm, model PSF 15% too wide', 'fixed on 5 other subjects', hcp('2.4', 'model_psf1.15', 'coherent'), hcp('2.4', 'model_psf1.15', 'clean16_leak10')),
    ('B', 'B5', 'HCP 3.2 mm, model PSF 15% too narrow', 'fixed on 5 other subjects', hcp('3.2', 'model_psf0.85', 'coherent'), hcp('3.2', 'model_psf0.85', 'clean16_leak10')),
    ('B', 'B6', 'HCP 3.2 mm, model PSF 15% too wide', 'fixed on 5 other subjects', hcp('3.2', 'model_psf1.15', 'coherent'), hcp('3.2', 'model_psf1.15', 'clean16_leak10')),
    ('C', 'C1', 'StudyForrest 3T → 7T, voxel ratio ≥ 2.5', 'other subjects of the pair type',
     X8['primary: 3T->7T, ratio >= 2.5, visual']['per_subject'], None),
    ('C', 'C2', 'StudyForrest 7T → 7T, all pairs', 'other subjects of the pair type', X8['secondary: all 7T->7T, visual']['per_subject'], None),
    ('C', 'C3', 'ds002702 3T 2.4 mm → 7T 0.8 mm', 'the other subject of the pair type', k3, k3h),
    ('C', 'C4', 'ds002702 7T SE 1.05 mm → GE 0.8 mm', 'the other subject of the pair type', kse, kseh),
]
res = {'spec_sha1': SPEC_SHA, 'rows': []}
for blk, rid, lab, rule, a, h in ROWS:
    res['rows'].append(dict(block=blk, id=rid, label=lab, settings=rule, all=stats(a), high=stats(h) if h else None))
# checks named in the specification
row = {r['id']: r for r in res['rows']}
# Deviation from the specification's check A1/A2: it expected the Section 4.3 counts (13/15, 14/15), which compare the model with
# the competitor that is best on average (upsampling + smoothing). The specified per-subject best competitor is stricter; on the
# same data it gives the values of Fig. 4A (right panel): 12/15 and 14/15, mean -0.0070 and -0.0127. The definition is kept.
assert (row['A1']['all']['n_neg'], row['A2']['all']['n_neg']) == (12, 14), 'HCP held-out counts (per-subject best competitor)'
assert (round(row['A1']['all']['mean'], 4), round(row['A2']['all']['mean'], 4)) == (-0.0070, -0.0127), 'HCP held-out means (Fig. 4A)'
res['deviations'] = ['Check A1/A2 corrected: the specification expected the Section 4.3 counts (13/15, 14/15), which use the competitor '
                     'best on average; with the specified per-subject best competitor the same data give 12/15 and 14/15, as in Fig. 4A '
                     '(right). The definition was kept.']
assert [round(row[i]['all']['mean'], 3) for i in ('A4', 'A5', 'A6')] == [0.033, 0.029, 0.003], 'ds002702 within-session means'
c1 = row['C1']['all']; assert (round(c1['mean'], 4), c1['n_neg'], round(c1['p_one_sided'], 3)) == (0.0141, 2, 0.844), 'pre-declared C1'
for f in kay:
    if 'too_few' in kay[f]['high']:
        res.setdefault('notes', []).append(f"ds002702 {VOX[f]} mm: too few high-mixing vertices (min {kay[f]['high']['too_few']} < 30)")
json.dump(res, open(f'{OUT}/robustness_table.json', 'w'), indent=1)

# ---------------------------------------------------------------- markdown table (copied into the manuscript by build_v1.py)
m = lambda x: f'{x:+.3f}'.replace('-', '−')
pf = lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'


def cellset(t):
    if not t:
        return '–', '–', '–'
    return f"{m(t['mean'])} [{m(t['ci95'][0])}, {m(t['ci95'][1])}]", f"{t['n_neg']}/{t['n']}", pf(t['p_one_sided'])


BL = {'A': 'Emulation within one acquisition', 'B': 'Robustness of the HCP result', 'C': 'Second acquisitions'}
L = ['| comparison | settings chosen | all vertices: model − best competitor [95% CI] | better | p | high mixing: model − best competitor [95% CI] | better | p |',
     '|---|---|---|---|---|---|---|---|']
for blk in 'ABC':
    L.append(f'| **{BL[blk]}** | | | | | | | |')
    for r in res['rows']:
        if r['block'] == blk:
            L.append(f"| {r['label']} | {r['settings']} | " + ' | '.join(cellset(r['all'])) + ' | ' + ' | '.join(cellset(r['high'])) + ' |')
open(f'{OUT}/robustness_table.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print('notes:', res.get('notes', []))
