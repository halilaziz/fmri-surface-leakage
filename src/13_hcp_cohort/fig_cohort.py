"""Week 13 cohort figure and curve numbers: the model's gain over the competitor (upsampling + 1.6 mm mapping + 8 mm smoothing)
by far-leak bin and by contamination bin, for the wedge runs (full and half sources) and the ring runs, mean ± SE over subjects.
Usage (local): python3 src/13_hcp_cohort/fig_cohort.py OUT_PNG OUT_JSON [--batches b1,b2,...]  (default: batch_01..16, no pilot, no batch_17)
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('out_png'); ap.add_argument('out_json'); ap.add_argument('--batches', default='')
ap.add_argument('--root', default='results/week13/cohort')
a = ap.parse_args()
B = a.batches.split(',') if a.batches else sorted(os.path.basename(d) for d in glob.glob(f'{a.root}/batch_*') if os.path.basename(d) != 'batch_17' and os.path.isdir(f'{d}/eval'))
G = ('2.4', '3.2'); BL = ['0–2%', '2–5%', '5–10%', '10–20%', '> 20%']; DBINS = ['0-0.02', '0.02-0.05', '0.05-0.1', '0.1-0.2', '0.2-1.01']


def merge(name):
    out = {}
    for b in B:
        p = f'{a.root}/{b}/eval/{name}'
        if os.path.exists(p):
            J = json.load(open(p)); out.update(J if 'cells' not in J else {})
            if 'cells' in J:
                for g in G:
                    out.setdefault(g, {}).update(J['cells'].get(g, {}))
    return out


dose = {k: merge(f'dose_{k}/hcp_dose_response.json') for k in ('wedge', 'half1', 'half2', 'ring')}
cwl = {'wedge': merge('hcp_cwl_summary.json'), 'ring': merge('ring_cwl_summary.json')}


def curve_dose(D, g):
    rows = []
    for S, r in D.items():
        if g in r:
            rows.append([(r[g][b]['model'] - r[g][b]['upsample_smoothing']) if (r[g][b]['model'] is not None and r[g][b]['n'] >= 30) else np.nan for b in DBINS])
    M = np.array(rows, float)
    return np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt((~np.isnan(M)).sum(0)), (~np.isnan(M)).sum(0)


def curve_cwl(C, g, var='cwlM'):
    rows = [[(x if x is not None else np.nan) for x in r['bins'][var]['delta']] for S, r in C[g].items()]
    M = np.array(rows, float)
    return np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt((~np.isnan(M)).sum(0)), (~np.isnan(M)).sum(0)


res = {'batches': B, 'n_subjects': len(dose['wedge']), 'bins': BL}
FS = float(os.environ.get('GEO_FONT_SCALE', '1'))     # font scale for print-size figures (week 14; 1 = as developed)
plt.rcParams.update({'font.size': FS * 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
for gi, g in enumerate(G):
    aL, aR = ax[gi]
    x = np.arange(5)
    # left: far-leak bins
    for key, lab, st, col in (('wedge', 'polar angle (wedges), full source run', 'o-', '#C44E52'),
                              ('half', 'polar angle, half source run', 's--', '#E8A9AB'), ('ring', 'eccentricity (rings), full source run', '^-', '#4C72B0')):
        if key == 'half':
            m1, s1, n1 = curve_dose(dose['half1'], g); m2, s2, n2 = curve_dose(dose['half2'], g)
            m, se, n = 0.5 * (m1 + m2), 0.5 * np.sqrt(s1 ** 2 + s2 ** 2), n1
        else:
            m, se, n = curve_dose(dose[key], g)
        res[f'far_{key}_{g}'] = dict(mean=m.round(5).tolist(), se=se.round(5).tolist(), n=n.tolist())
        aL.errorbar(x + {'wedge': -0.08, 'half': 0, 'ring': 0.08}[key], m, yerr=se, fmt=st, color=col, ms=4, capsize=2, label=lab)
    aL.axhline(0, color='k', lw=0.8, ls=':'); aL.set_xticks(x); aL.set_xticklabels(BL)
    aL.set_xlabel('far leak of standard mapping at the coarse grid\n(vertices with < 2% far leak at 1.6 mm)')
    aL.set_ylabel('model − competitor (mean ± SE over subjects)'); aL.set_title(f'Emulated {g} mm: gain by far leak', fontsize=FS * 9)
    if gi == 0:
        aL.legend(frameon=False, fontsize=FS * 7.5)
    # right: contamination bins
    for key, lab, st, col in (('wedge', 'polar angle (wedges)', 'o-', '#C44E52'), ('ring', 'eccentricity (rings)', '^-', '#4C72B0')):
        m, se, n = curve_cwl(cwl[key], g)
        res[f'cwl_{key}_{g}'] = dict(mean=m.round(5).tolist(), se=se.round(5).tolist(), n=n.tolist())
        aR.errorbar(x + {'wedge': -0.06, 'ring': 0.06}[key], m, yerr=se, fmt=st, color=col, ms=4, capsize=2, label=lab)
    aR.axhline(0, color='k', lw=0.8, ls=':'); aR.set_xticks(x); aR.set_xticklabels(BL)
    aR.set_xlabel('contamination-weighted leak, raw response estimate (CWL-M)\n(same vertices)')
    aR.set_title(f'Emulated {g} mm: gain by contamination', fontsize=FS * 9)
    if gi == 0:
        aR.legend(frameon=False, fontsize=FS * 7.5)
fig.suptitle(f'HCP 7T confirmatory cohort ({res["n_subjects"]} subjects): the model\'s gain over upsampling + 8 mm smoothing', fontsize=FS * 9.5)
plt.tight_layout(); fig.savefig(a.out_png, dpi=170); plt.close(fig)
json.dump(res, open(a.out_json, 'w'), indent=1)
# the same numbers as a markdown table (the supplement copies it; the number check reads it)
CURVES = [('far_wedge', 'far leak, wedges (full source run)'), ('far_half', 'far leak, wedges (half source run)'), ('far_ring', 'far leak, rings'),
          ('cwl_wedge', 'contamination (CWL-M), wedges'), ('cwl_ring', 'contamination (CWL-M), rings')]
L = ["# Week 13 cohort: the model's gain by bin (Fig. 8)", '',
     f'Model − competitor (upsampling + 1.6 mm mapping + 8 mm smoothing), mean ± SE over subjects ({res["n_subjects"]} subjects; a subject '
     'enters a bin with ≥ 30 vertices), by bin of the dose variable at the coarse grid among vertices with < 2% far leak at 1.6 mm.', '',
     '| grid | gain by | ' + ' | '.join(BL) + ' |', '|---|---|' + '---|' * len(BL)]
for g in G:
    for k, name in CURVES:
        c = res[f'{k}_{g}']
        L.append(f'| {g} mm | {name} | ' + ' | '.join(f'{m:+.4f} ± {s:.4f} (n = {n})'.replace('-', '−') for m, s, n in zip(c['mean'], c['se'], c['n'])) + ' |')
open(a.out_json[:-5] + '.md' if a.out_json.endswith('.json') else a.out_json + '.md', 'w').write('\n'.join(L) + '\n')
for k, v in res.items():
    if isinstance(v, dict):
        print(k, [f'{m:+.4f}' for m in v['mean']], 'n', v['n'])
