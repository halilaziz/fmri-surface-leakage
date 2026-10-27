"""Week 13: the tests of COHORT_PLAN.md (H1-H4) from the per-batch evaluation files, with the family rule.
Runs locally (the family table is local-only). Per row: n, mean, median, subjects in the predicted direction, bootstrap 95%
interval of the mean (10,000 resamples, seed 0 per row), one-sided Wilcoxon p; Holm within each secondary family; for the
all-subject analysis a family-level bootstrap as well.
Usage: python3 src/13_hcp_cohort/cohort_stats.py OUT_PREFIX [--batches batch_01,batch_02,...] [--family-rule lowest|none]
       [--root results/week13/cohort] [--label "pilot (exploratory)"]
"""
import os, sys, json, csv, glob, argparse
import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--batches', default=''); ap.add_argument('--family-rule', default='lowest', choices=['lowest', 'none'])
ap.add_argument('--root', default='results/week13/cohort'); ap.add_argument('--label', default='')
ap.add_argument('--family-table', default='configs/splits_7T_family.tsv')
ap.add_argument('--exclude', default='pilot,batch_17', help='never part of the default batch list: the pilot (development set) and the sensitivity set')
a = ap.parse_args()
batches = a.batches.split(',') if a.batches else sorted(os.path.basename(d) for d in glob.glob(f'{a.root}/*')
                                                       if os.path.isdir(f'{d}/eval') and os.path.basename(d) not in a.exclude.split(','))
G = ('2.4', '3.2')
BIN0 = '0-0.02'


def load(name):
    """Merge one eval file over batches; returns {} if absent everywhere."""
    out = {}
    for b in batches:
        p = f'{a.root}/{b}/eval/{name}'
        if os.path.exists(p):
            J = json.load(open(p))
            out.setdefault('_files', []).append(p); out.setdefault('_json', []).append(J)
    return out


def cells(name):
    """cells[g][S] of a summarize_cwl(2) file, merged over batches."""
    C = {g: {} for g in G}
    for J in load(name).get('_json', []):
        for g in G:
            C[g].update(J['cells'].get(g, {}))
    return C


def dose(name):
    """res[S][g][bin] of a dose-response file, merged over batches."""
    D = {}
    for J in load(name).get('_json', []):
        D.update(J)
    return D


def scores(name):
    Sc = {}
    for J in load(name).get('_json', []):
        Sc.update(J['scores'])
    return Sc


def d_from_dose(D, S, g):
    """D_far from the far-leak dose-response: Delta(highest bin with >= 30 vertices) - Delta(0-2% bin), competitor upsampling + 8 mm."""
    r = D.get(S, {}).get(g)
    if not r or r[BIN0]['model'] is None:
        return None
    dl = {b: (v['model'] - v['upsample_smoothing']) if (v['model'] is not None and v['n'] >= 30) else None for b, v in r.items()}
    top = [b for b in r if dl[b] is not None][-1]
    return None if top == BIN0 else dl[top] - dl[BIN0]


def delta_clean(D, S, g):
    r = D.get(S, {}).get(g)
    if not r or r[BIN0]['model'] is None or r[BIN0]['n'] < 30:
        return None
    return r[BIN0]['model'] - r[BIN0]['upsample_smoothing']


def best_two(sc, S, g, lab='model', stratum='coherent'):
    """gain of `lab` against the per-subject better of the two smoothing competitors (Section 4.8)."""
    s = sc.get(S, {}).get(g)
    if not s or not s.get(lab, {}).get(stratum) or not s.get('smoothing', {}).get(stratum) or not s.get('upsample_smoothing', {}).get(stratum):
        return None
    v = [s[lab][stratum], s['smoothing'][stratum], s['upsample_smoothing'][stratum]]
    if any(x is None or not np.isfinite(x) for x in v):
        return None
    return v[0] - min(v[1], v[2])


# ---------------------------------------------------------------- subjects, families
fam = {}
if os.path.exists(a.family_table):                 # HCP restricted family IDs: never distributed with the code
    with open(a.family_table) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            fam[r['Subject']] = r['Family_ID']
elif a.family_rule == 'lowest':
    raise SystemExit(f'{a.family_table} not found: the one-subject-per-family rule needs the HCP restricted family table '
                     '(--family-table); without it, run with --family-rule none')
else:
    print(f'note: {a.family_table} not found; the family bootstrap resamples subjects, each its own family', flush=True)
CW, CW2, CR = cells('hcp_cwl_summary.json'), cells('hcp_cwl2_summary.json'), cells('ring_cwl_summary.json')
DW, DR, DH1, DH2 = dose('dose_wedge/hcp_dose_response.json'), dose('dose_ring/hcp_dose_response.json'), dose('dose_half1/hcp_dose_response.json'), dose('dose_half2/hcp_dose_response.json')
SW, SR = scores('wedge/week3_test_scores.json'), scores('ring/week3_test_scores.json')
subjects = sorted(set(CW['2.4']) | set(SW))
if a.family_rule == 'lowest':
    keep = {}
    for S in subjects:
        f = fam.get(S, S); keep[f] = min(keep.get(f, S), S)
    primary = sorted(keep.values())
else:
    primary = subjects
print(f'{len(subjects)} subjects with results in {len(batches)} batch(es); primary set ({a.family_rule}): {len(primary)}', flush=True)


# ---------------------------------------------------------------- statistics
def stats(vals, direction='less'):
    v = np.asarray([x for x in vals.values() if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0)
    b = np.random.default_rng(0).choice(v, size=(10000, len(v)), replace=True).mean(1)
    p = float(wilcoxon(v, alternative=direction).pvalue) if len(v) >= 2 and np.any(v != 0) else float('nan')
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), n_dir=int((v < 0).sum() if direction == 'less' else (v > 0).sum()),
                ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], p=p)


def family_boot(vals):
    """Interval of the mean when families, not subjects, are resampled."""
    by = {}
    for S, x in vals.items():
        if x is not None and np.isfinite(x):
            by.setdefault(fam.get(S, S), []).append(x)
    fams = list(by.values())
    if not fams:
        return None
    rng = np.random.default_rng(0); m = []
    for _ in range(10000):
        pick = rng.integers(0, len(fams), len(fams)); m.append(np.mean(np.concatenate([fams[i] for i in pick])))
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(rows):
    ps = [(r['p'], i) for i, r in enumerate(rows) if r.get('n') and np.isfinite(r['p'])]
    ps.sort(); m = len(ps); adj = {}
    run = 0.0
    for k, (p, i) in enumerate(ps):
        run = max(run, min(1.0, (m - k) * p)); adj[i] = run
    for i, r in enumerate(rows):
        r['p_holm'] = adj.get(i, float('nan'))


def dvals(C, g, var):
    return {S: r['bins'][var]['D'] for S, r in C[g].items()}


# rows: (hypothesis, family, label, direction, values over all subjects)
ROWS = []
for g in G:
    ROWS.append(('H1', 'primary', f'D_CWL-M < 0, {g} mm', 'less', dvals(CW, g, 'cwlM')))
for g in G:
    ROWS.append(('H1', 'secondary', f'D_CWL-P < 0, {g} mm', 'less', dvals(CW, g, 'cwlP')))
    ROWS.append(('H1', 'secondary', f'D_far < 0, {g} mm', 'less', dvals(CW, g, 'far')))
    ROWS.append(('H1', 'secondary', f'gain, all coherent < 0, {g} mm', 'less', {S: best_two(SW, S, g) for S in SW}))
    ROWS.append(('H1', 'descriptive', f'D_CWL-S4, {g} mm', 'less', dvals(CW2, g, 'cwlS4')))
    ROWS.append(('H1', 'descriptive', f'D_CWL-Q, {g} mm', 'less', dvals(CW2, g, 'cwlQ')))
for g in G:
    dw, dr = dvals(CW, g, 'far'), dvals(CR, g, 'far')
    diff = {S: (dw[S] - dr[S]) if (S in dw and S in dr and dw[S] is not None and dr[S] is not None) else None for S in sorted(set(dw) | set(dr))}
    ROWS.append(('H2', 'H2a' if g == '3.2' else 'interval', f'D_far(wedge) − D_far(ring){" < 0" if g == "3.2" else ""}, {g} mm', 'less', diff))
    ROWS.append(('H2', 'descriptive', f'D_far(ring), {g} mm', 'less', dr))
    ROWS.append(('H2', 'H2b', f'D_CWL-M(ring) < 0, {g} mm', 'less', dvals(CR, g, 'cwlM')))
dm, df_ = dvals(CR, '3.2', 'cwlM'), dvals(CR, '3.2', 'far')
ROWS.append(('H2', 'secondary', 'D_CWL-M(ring) − D_far(ring) < 0, 3.2 mm', 'less', {S: (dm[S] - df_[S]) if (dm.get(S) is not None and df_.get(S) is not None) else None for S in dm}))
for g in G:
    ROWS.append(('H2', 'descriptive', f'ring: gain, all coherent, {g} mm', 'less', {S: best_two(SR, S, g) for S in SR}))
for g in G:
    half_c = {S: np.mean([delta_clean(DH1, S, g), delta_clean(DH2, S, g)]) if (delta_clean(DH1, S, g) is not None and delta_clean(DH2, S, g) is not None) else None for S in sorted(set(DH1) & set(DH2))}
    full_c = {S: delta_clean(DW, S, g) for S in DW}
    ROWS.append(('H3', 'H3a', f'Δ_clean(half) − Δ_clean(full) > 0, {g} mm', 'greater', {S: (half_c[S] - full_c[S]) if (S in full_c and half_c.get(S) is not None and full_c[S] is not None) else None for S in half_c}))
    half_d = {S: np.mean([d_from_dose(DH1, S, g), d_from_dose(DH2, S, g)]) if (d_from_dose(DH1, S, g) is not None and d_from_dose(DH2, S, g) is not None) else None for S in sorted(set(DH1) & set(DH2))}
    full_d = {S: d_from_dose(DW, S, g) for S in DW}
    ROWS.append(('H3', 'interval', f'D_far(half) − D_far(full), {g} mm (interval should include 0 or be positive)', 'less', {S: (half_d[S] - full_d[S]) if (S in full_d and half_d.get(S) is not None and full_d[S] is not None) else None for S in half_d}))
    ROWS.append(('H3', 'descriptive', f'Δ_clean(full), {g} mm', 'less', full_c)); ROWS.append(('H3', 'secondary', f'Δ_clean(half) < 0, {g} mm', 'less', half_c))
ROWS.append(('H4', 'primary', 'PSF 15% too wide: gain, all coherent < 0, 2.4 mm', 'less', {S: best_two(SW, S, '2.4', 'model_psf1.15') for S in SW}))
ROWS.append(('H4', 'secondary', 'PSF 15% too wide: gain, all coherent < 0, 3.2 mm', 'less', {S: best_two(SW, S, '3.2', 'model_psf1.15') for S in SW}))
for g in G:
    ROWS.append(('H4', 'secondary', f'PSF 15% too narrow: gain, all coherent < 0, {g} mm', 'less', {S: best_two(SW, S, g, 'model_psf0.85') for S in SW}))

res = {'label': a.label, 'batches': batches, 'n_subjects': len(subjects), 'family_rule': a.family_rule, 'primary_set': primary, 'rows': []}
for hyp, famly, lab, direction, vals in ROWS:
    r = dict(hypothesis=hyp, family=famly, label=lab, direction=direction,
             primary=stats({S: v for S, v in vals.items() if S in primary}, direction),
             all=stats(vals, direction), all_family_boot=family_boot(vals), values={S: v for S, v in vals.items() if v is not None})
    res['rows'].append(r)
for hyp in ('H1', 'H2', 'H3', 'H4'):
    for famly in ('secondary',):
        grp = [r['primary'] for r in res['rows'] if r['hypothesis'] == hyp and r['family'] == famly]
        if grp:
            holm(grp)
json.dump(res, open(f'{a.out}.json', 'w'), indent=1, default=float)

m = lambda x: f'{x:+.4f}'.replace('-', '−')
pf = lambda p: '–' if p is None or not np.isfinite(p) else ('< 0.001' if p < 0.001 else f'{p:.3f}')
L = [f"# Week 13 cohort tests {('— ' + a.label) if a.label else ''}", '',
     f"{len(subjects)} subjects with results ({len(batches)} batches); primary set: {len(primary)} ({a.family_rule}: one subject per family)" if a.family_rule == 'lowest' else f"{len(subjects)} subjects; no family rule", '',
     '| hyp. | role | test | n | mean | median | in direction | bootstrap 95% | p | p (Holm) | all subjects: n, mean, family-bootstrap 95% |',
     '|---|---|---|---|---|---|---|---|---|---|---|']
for r in res['rows']:
    p = r['primary']; q = r['all']
    if not p.get('n'):
        L.append(f"| {r['hypothesis']} | {r['family']} | {r['label']} | 0 | – | – | – | – | – | – | – |"); continue
    fb = r['all_family_boot']
    L.append(f"| {r['hypothesis']} | {r['family']} | {r['label']} | {p['n']} | {m(p['mean'])} | {m(p['median'])} | {p['n_dir']}/{p['n']} | "
             f"[{m(p['ci95'][0])}, {m(p['ci95'][1])}] | {pf(p['p'])} | {pf(p.get('p_holm'))} | {q['n']}, {m(q['mean'])}, "
             + (f"[{m(fb[0])}, {m(fb[1])}]" if fb else '–') + ' |')
open(f'{a.out}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
