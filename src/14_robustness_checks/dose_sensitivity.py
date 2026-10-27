"""Week 14 (reviewer pass): how much the dose-response contrast D depends on its bin conventions, in the confirmatory cohort.

D = Δ(highest bin with ≥ 30 vertices) − Δ(0–2% bin). Alternatives computed from the same per-subject, per-bin gains (the
`bins` records of the contamination summaries, wedge and ring): the 2–5% bin as reference; the 10–20% bin as top bin for every
subject; a least-squares slope of Δ on bin index over the available bins (≥ 3); Spearman's ρ of Δ with bin index; and the number
of subjects whose highest available bin was 10–20% rather than > 20%. Statistics as in the plan (one subject per family, lowest
ID; one-sided Wilcoxon; bootstrap 95% of the mean, 10,000 resamples, seed 0). Also H2a (D_far(wedge) − D_far(ring), 3.2 mm)
under each definition.
Usage (local): python3 src/14_robustness_checks/dose_sensitivity.py OUT_PREFIX [--root results/week13/cohort] [--family-table configs/splits_7T_family.tsv]
"""
import os, sys, json, csv, glob, argparse
import numpy as np
from scipy.stats import wilcoxon, spearmanr

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--root', default='results/week13/cohort')
ap.add_argument('--family-table', default='configs/splits_7T_family.tsv')
a = ap.parse_args()
G = ('2.4', '3.2'); BL = ['0–2%', '2–5%', '5–10%', '10–20%', '> 20%']
batches = sorted(os.path.basename(d) for d in glob.glob(f'{a.root}/batch_*') if os.path.basename(d) != 'batch_17' and os.path.isdir(f'{d}/eval'))


def cells(name):
    C = {g: {} for g in G}
    for b in batches:
        p = f'{a.root}/{b}/eval/{name}'
        if os.path.exists(p):
            J = json.load(open(p))
            for g in G:
                C[g].update(J['cells'].get(g, {}))
    return C


CW, CR = cells('hcp_cwl_summary.json'), cells('ring_cwl_summary.json')
fam = {r['Subject']: r['Family_ID'] for r in csv.DictReader(open(a.family_table), delimiter='\t')}
subjects = sorted(CW['2.4']); keep = {}
for S in subjects:
    f = fam.get(S, S); keep[f] = min(keep.get(f, S), S)
primary = set(keep.values())


def defs(delta):
    """The contrast under each definition, from one subject's list of five bin gains (None where < 30 vertices)."""
    d = [None if x is None else float(x) for x in delta]
    avail = [i for i, x in enumerate(d) if x is not None]
    top = avail[-1] if avail else None
    out = {'top_bin': top}
    out['D (as planned)'] = None if (top in (None, 0) or d[0] is None) else d[top] - d[0]
    out['reference bin 2–5%'] = None if (top in (None, 0, 1) or d[1] is None) else d[top] - d[1]
    out['top bin fixed at 10–20%'] = None if (d[3] is None or d[0] is None) else d[3] - d[0]
    if len(avail) >= 3:
        x = np.array(avail, float); y = np.array([d[i] for i in avail])
        out['slope per bin step'] = float(np.polyfit(x, y, 1)[0]); out["Spearman ρ (Δ vs bin)"] = float(spearmanr(x, y).correlation)
    else:
        out['slope per bin step'] = None; out["Spearman ρ (Δ vs bin)"] = None
    return out


DEFS = ['D (as planned)', 'reference bin 2–5%', 'top bin fixed at 10–20%', 'slope per bin step', "Spearman ρ (Δ vs bin)"]


def stats(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return dict(n=int(len(v)))
    b = np.random.default_rng(0).choice(v, size=(10000, len(v)), replace=True).mean(1)
    return dict(n=int(len(v)), mean=float(v.mean()), n_dir=int((v < 0).sum()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                p=float(wilcoxon(v, alternative='less').pvalue) if np.any(v != 0) else float('nan'))


m = lambda x: f'{x:+.4f}'.replace('-', '−')
pf = lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'
res = {'batches': batches, 'n_subjects': len(subjects), 'n_primary': len(primary), 'rows': []}
L = ['# Week 14: the dose-response contrast under other bin conventions (confirmatory cohort, one subject per family)', '',
     f'{len(subjects)} subjects, primary set {len(primary)}. Per-subject gains Δ by bin (model − upsampling + 8 mm smoothing; vertices clean at 1.6 mm) from the',
     'contamination summaries of the cohort batches; a bin counts when it has ≥ 30 vertices. "D (as planned)": highest available bin − 0–2% bin.',
     'One-sided Wilcoxon (negative predicted), bootstrap 95% of the mean. Source: `src/14_robustness_checks/dose_sensitivity.py`.', '',
     '| stimulus | dose variable | grid | definition | n | mean | in direction | bootstrap 95% | p |', '|---|---|---|---|---|---|---|---|---|']
VARS = [('wedges', CW, 'cwlM', 'contamination (CWL-M)'), ('wedges', CW, 'cwlP', 'contamination, coarse data only (CWL-P)'), ('wedges', CW, 'far', 'plain far leak'),
        ('rings', CR, 'cwlM', 'contamination (CWL-M)'), ('rings', CR, 'far', 'plain far leak')]
per = {}
for stim, C, var, vname in VARS:
    for g in G:
        D = {S: defs(C[g][S]['bins'][var]['delta']) for S in C[g] if S in primary}
        per[(stim, var, g)] = D
        tops = [d['top_bin'] for d in D.values()]
        res['rows'].append(dict(stimulus=stim, variable=var, grid=g, n_top_1020=int(sum(t == 3 for t in tops)), n_top_20=int(sum(t == 4 for t in tops)),
                                n_top_lower=int(sum(t is not None and t < 3 for t in tops))))
        for k in DEFS:
            st = stats([d[k] for d in D.values()])
            res['rows'].append(dict(stimulus=stim, variable=var, grid=g, definition=k, **st))
            if st.get('n', 0) >= 2:
                L.append(f"| {stim} | {vname} | {g} mm | {k} | {st['n']} | {m(st['mean'])} | {st['n_dir']}/{st['n']} | [{m(st['ci95'][0])}, {m(st['ci95'][1])}] | {pf(st['p'])} |")
        L.append(f"| {stim} | {vname} | {g} mm | highest available bin | {len(tops)} | > 20% in {sum(t == 4 for t in tops)}, 10–20% in {sum(t == 3 for t in tops)}, lower in {sum(t is not None and t < 3 for t in tops)} | | | |")
# H2a under each definition
L += ['', '## H2a: D_far(wedges) − D_far(rings) at 3.2 mm under each definition', '', '| definition | n | mean | in direction | bootstrap 95% | p |', '|---|---|---|---|---|---|']
for k in DEFS:
    W, R = per[('wedges', 'far', '3.2')], per[('rings', 'far', '3.2')]
    vals = [W[S][k] - R[S][k] for S in W if S in R and W[S][k] is not None and R[S][k] is not None]
    st = stats(vals); res['rows'].append(dict(test='H2a', definition=k, **st))
    L.append(f"| {k} | {st['n']} | {m(st['mean'])} | {st['n_dir']}/{st['n']} | [{m(st['ci95'][0])}, {m(st['ci95'][1])}] | {pf(st['p'])} |")
open(f'{a.out}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{a.out}.json', 'w'), indent=1)
print('\n'.join(L))
