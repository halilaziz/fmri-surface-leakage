"""Week 8: the pre-declared tests of STATS_PLAN.md (part A) from the per-dataset summaries of summarize_cwl.py, a table and a
figure. Part B (StudyForrest cross-session) is tested in sf_cross_loso.py and only copied into the table here.
Usage (local): python3 stats_w8.py HCP_JSON SF_JSON KAY_JSON SF_CROSS_JSON PLAN_SHA1_FILE OUTDIR
"""
import sys, os, json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HJ, SJ, KJ, XJ, SHA, OUT = sys.argv[1:7]; os.makedirs(OUT, exist_ok=True)
H, SF, K, X = (json.load(open(p)) for p in (HJ, SJ, KJ, XJ))
sha = open(SHA).read().split()[0]
rng = np.random.default_rng(0)
BLAB = H['bins']


def test(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return dict(n=0)
    b = rng.choice(v, size=(10000, len(v)), replace=True).mean(1)
    p = float(wilcoxon(v, alternative='less').pvalue) if len(v) >= 2 and np.any(v != 0) else float('nan')
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), n_neg=int((v < 0).sum()),
                ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], p_one_sided=p, values=v.tolist())


def per_subject(cells, key, var=None, fn=None):
    """Mean over a dataset's cells per subject of bins[var]['D'] (or fn(record))."""
    acc = {}
    for cell in cells:
        for S, r in cell.items():
            x = r['bins'][var]['D'] if var else fn(r)
            if x is not None:
                acc.setdefault(S, []).append(x)
    return {S: float(np.mean(v)) for S, v in acc.items()}


data = {'HCP 2.4 mm': [H['cells']['2.4']], 'HCP 3.2 mm': [H['cells']['3.2']],
        'StudyForrest (4 cells)': list(SF['cells'].values()), 'ds002702 2.4 mm': [K['cells']['3']]}
extra = {'ds002702 1.6 mm': [K['cells']['2']], 'ds002702 1.05 mm': [K['cells']['1.3125']]}
R = {'plan_sha1': sha, 'primary': {}, 'secondary': {}, 'descriptive': {}}
for name, cells in data.items():
    R['primary'][name] = dict(per_subject=per_subject(cells, 'D', 'cwlM'), **test(per_subject(cells, 'D', 'cwlM').values()))
# Holm across the two HCP grids
hp = sorted((R['primary'][k]['p_one_sided'], k) for k in ('HCP 2.4 mm', 'HCP 3.2 mm'))
for i, (p, k) in enumerate(hp):
    R['primary'][k]['p_holm_hcp'] = float(min(1.0, max(q * (len(hp) - j) for j, (q, _) in enumerate(hp[:i + 1]))))
for name, cells in data.items():
    R['secondary'][f'{name}: far-leak contrast'] = test(per_subject(cells, 'D', 'far').values())
    R['secondary'][f'{name}: CWL-P contrast'] = test(per_subject(cells, 'D', 'cwlP').values())
    R['secondary'][f'{name}: CWL-P gated - competitor'] = test(per_subject(cells, None, fn=lambda r: r['gated_loso'] - r['competitor_everywhere']).values())
ps = sorted((v['p_one_sided'], k) for k, v in R['secondary'].items() if v.get('n', 0) and np.isfinite(v['p_one_sided']))
for i, (p, k) in enumerate(ps):
    R['secondary'][k]['p_holm'] = float(min(1.0, max(q * (len(ps) - j) for j, (q, _) in enumerate(ps[:i + 1]))))
for name, cells in extra.items():
    R['descriptive'][f'{name}: CWL-M contrast'] = test(per_subject(cells, 'D', 'cwlM').values())
    R['descriptive'][f'{name}: far-leak contrast'] = test(per_subject(cells, 'D', 'far').values())
for name, cells in {**data, **extra}.items():
    R['descriptive'][f'{name}: model everywhere - competitor'] = test(per_subject(cells, None, fn=lambda r: r['model_everywhere'] - r['competitor_everywhere']).values())
json.dump(R, open(f'{OUT}/w8_stats.json', 'w'), indent=1)

f = lambda t: (f"{t['n']} | {t['mean']:+.4f} | {t['median']:+.4f} | {t['n_neg']}/{t['n']} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | "
               f"{t['p_one_sided']:.3f}") if t.get('n') else '0 | – | – | – | – | –'
T = ['# Week 8 pre-declared tests', '', f'Plan SHA-1 {sha} (`STATS_PLAN.md`). D = gain in the highest CWL bin − gain in the 0–2% bin, per '
     'subject (negative = the model gains more where far cortex with different responses mixes in). One-sided Wilcoxon '
     'signed-rank (H1: < 0), bootstrap 95% interval of the mean.', '',
     '## Primary (CWL-M)', '', '| dataset | n | mean D | median D | D < 0 | bootstrap 95% | p | p (Holm, HCP grids) |', '|---|---|---|---|---|---|---|---|']
for k, t in R['primary'].items():
    T.append(f"| {k} | {f(t)} | {t.get('p_holm_hcp', float('nan')):.3f} |" if 'HCP' in k else f"| {k} | {f(t)} | – |")
T += ['', '## Secondary (Holm over all rows)', '', '| test | n | mean | median | < 0 | bootstrap 95% | p | p (Holm) |', '|---|---|---|---|---|---|---|---|']
for k, t in R['secondary'].items():
    T.append(f"| {k} | {f(t)} | {t.get('p_holm', float('nan')):.3f} |")
T += ['', '## Descriptive', '', '| quantity | n | mean | median | < 0 | bootstrap 95% | p |', '|---|---|---|---|---|---|---|']
for k, t in R['descriptive'].items():
    T.append(f"| {k} | {f(t)} |")
T += ['', '## Plan B (StudyForrest cross-session, settings chosen on the other subjects; from sf_cross_loso.py)', '',
      '| test | n | mean Δ | median Δ | model better | bootstrap 95% | p |', '|---|---|---|---|---|---|---|']
for k, t in X['tests'].items():
    T.append(f"| {k} | {t['n']} | {t['mean']:+.4f} | {t['median']:+.4f} | {t['n_better']}/{t['n']} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | {t['p_one_sided']:.3f} |")
# per-bin gains (mean over subjects having the bin)
T += ['', '## Gain (model − competitor) by bin, mean over subjects (n subjects with the bin)', '',
      '| dataset | variable | ' + ' | '.join(BLAB) + ' |', '|---|---|' + '---|' * len(BLAB)]
curves = {}
for name, cells in {**data, **extra}.items():
    for var in ('cwlM', 'cwlP', 'far'):
        ps_ = {}                                         # per subject: mean over the dataset's cells, then over subjects
        for cell in cells:
            for S, r in cell.items():
                for i, v in enumerate(r['bins'][var]['delta']):
                    if v is not None:
                        ps_.setdefault((S, i), []).append(v)
        acc = [[] for _ in BLAB]
        for (S, i), v in ps_.items():
            acc[i].append(float(np.mean(v)))
        curves[(name, var)] = [(np.mean(a) if a else np.nan, np.std(a) / np.sqrt(len(a)) if len(a) > 1 else np.nan, len(a)) for a in acc]
        T.append(f"| {name} | {var} | " + ' | '.join((f'{m:+.4f} ({n})' if n else '–') for m, _, n in curves[(name, var)]) + ' |')
open(f'{OUT}/w8_stats.md', 'w').write('\n'.join(T) + '\n')
print('\n'.join(T))

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
for a, name in zip(ax, ['HCP 3.2 mm', 'HCP 2.4 mm', 'StudyForrest (4 cells)', 'ds002702 2.4 mm']):
    for var, st, lab in (('cwlM', 'o-', 'contrast-weighted leak (source native map)'), ('cwlP', 's--', 'contrast-weighted leak (coarse data only)'),
                         ('far', '^:', 'far leak (week 2)')):
        c = curves[(name, var)]; x = np.arange(len(BLAB))
        m = np.array([v[0] for v in c]); e = np.array([v[1] for v in c])
        a.errorbar(x, m, yerr=e, fmt=st, capsize=2, label=lab, ms=4)
    a.axhline(0, color='k', lw=0.8, ls=':'); a.set_xticks(range(len(BLAB))); a.set_xticklabels(BLAB, fontsize=7)
    a.set_title(name, fontsize=8); a.set_xlabel('bin (fraction of the typical response / leak fraction)')
ax[0].set_ylabel('model − best competitor (mean ± SE over subjects)'); ax[0].legend(frameon=False, fontsize=6.5)
fig.suptitle('Week 8: gain of the model by contrast-weighted leak vs plain far leak (vertices clean at the native grid)', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figW8-1_cwl_dose_response.png', dpi=150); plt.close()
