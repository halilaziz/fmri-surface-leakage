"""Week 9: the pre-declared tests of STATS_PLAN_w9.md from the summaries of summarize_cwl2.py (denoised CWL), with the week-8
summaries for the paired comparison. Writes a table, a JSON and a figure.
Usage (local): python3 stats_w9.py HCP2 SF2 KAY2 HCP8 SF8 KAY8 PLAN_SHA1_FILE OUTDIR [paper]
"""
import sys, os, json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H2, S2, K2, H8, S8, K8 = (json.load(open(p)) for p in sys.argv[1:7]); SHA, OUT = sys.argv[7:9]; os.makedirs(OUT, exist_ok=True)
sha = open(SHA).read().split()[0]
rng = np.random.default_rng(0)
BLAB = H2['bins']


def test(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return dict(n=0)
    b = rng.choice(v, size=(10000, len(v)), replace=True).mean(1)
    p = float(wilcoxon(v, alternative='less').pvalue) if len(v) >= 2 and np.any(v != 0) else float('nan')
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), n_neg=int((v < 0).sum()),
                ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], p_one_sided=p, values=v.tolist())


def per_subject(cells, var):
    acc = {}
    for cell in cells:
        for S, r in cell.items():
            x = r['bins'][var]['D']
            if x is not None:
                acc.setdefault(S, []).append(x)
    return {S: float(np.mean(v)) for S, v in acc.items()}


def holm(d):
    ps = sorted((v['p_one_sided'], k) for k, v in d.items() if v.get('n', 0) and np.isfinite(v['p_one_sided']))
    for i, (p, k) in enumerate(ps):
        d[k]['p_holm'] = float(min(1.0, max(q * (len(ps) - j) for j, (q, _) in enumerate(ps[:i + 1]))))


W9 = {'StudyForrest (4 cells)': list(S2['cells'].values()), 'ds002702 2.4 mm': [K2['cells']['3']],
      'HCP 2.4 mm': [H2['cells']['2.4']], 'HCP 3.2 mm': [H2['cells']['3.2']],
      'ds002702 1.6 mm': [K2['cells']['2']], 'ds002702 1.05 mm': [K2['cells']['1.3125']]}
W8 = {'StudyForrest (4 cells)': list(S8['cells'].values()), 'ds002702 2.4 mm': [K8['cells']['3']],
      'HCP 2.4 mm': [H8['cells']['2.4']], 'HCP 3.2 mm': [H8['cells']['3.2']],
      'ds002702 1.6 mm': [K8['cells']['2']], 'ds002702 1.05 mm': [K8['cells']['1.3125']]}
R = {'plan_sha1': sha, 'primary': {}, 'secondary': {}, 'descriptive': {}}
for name in ('StudyForrest (4 cells)', 'ds002702 2.4 mm'):
    ps = per_subject(W9[name], 'cwlS4'); R['primary'][f'{name}: S4'] = dict(per_subject=ps, **test(ps.values()))
for name in ('StudyForrest (4 cells)', 'ds002702 2.4 mm'):
    for v in ('cwlS2', 'cwlS8', 'cwlQ'):
        R['secondary'][f'{name}: {v[3:]}'] = test(per_subject(W9[name], v).values())
for name in ('HCP 2.4 mm', 'HCP 3.2 mm'):
    for v in ('cwlS4', 'cwlQ'):
        R['secondary'][f'{name}: {v[3:]}'] = test(per_subject(W9[name], v).values())
holm(R['secondary'])
for name in ('ds002702 1.6 mm', 'ds002702 1.05 mm'):
    R['descriptive'][f'{name}: S4'] = test(per_subject(W9[name], 'cwlS4').values())
for name in W9:                                                    # paired: D(S4) - D(week-8 CWL-M), per subject
    a, b = per_subject(W9[name], 'cwlS4'), per_subject(W8[name], 'cwlM')
    R['descriptive'][f'{name}: D(S4) - D(week-8 CWL-M)'] = test([a[S] - b[S] for S in a if S in b])
json.dump(R, open(f'{OUT}/w9_stats.json', 'w'), indent=1)

f = lambda t: (f"{t['n']} | {t['mean']:+.4f} | {t['median']:+.4f} | {t['n_neg']}/{t['n']} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | "
               f"{t['p_one_sided']:.3f}") if t.get('n') else '0 | – | – | – | – | –'
T = ['# Week 9 pre-declared tests: contrast-weighted leak with a denoised response estimate', '',
     f'Plan SHA-1 {sha} (`STATS_PLAN_w9.md`). D = gain in the highest bin − gain in the 0–2% bin per subject (negative = the model '
     'gains more where contamination is high). One-sided Wilcoxon signed-rank, bootstrap 95% interval of the mean.', '',
     '## Primary (S4: source native map smoothed 4 mm)', '', '| test | n | mean D | median D | D < 0 | bootstrap 95% | p |', '|---|---|---|---|---|---|---|']
T += [f'| {k} | {f(t)} |' for k, t in R['primary'].items()]
T += ['', '## Secondary (Holm over all rows)', '', '| test | n | mean D | median D | D < 0 | bootstrap 95% | p | p (Holm) |', '|---|---|---|---|---|---|---|---|']
T += [f"| {k} | {f(t)} | {t.get('p_holm', float('nan')):.3f} |" for k, t in R['secondary'].items()]
T += ['', '## Descriptive', '', '| quantity | n | mean | median | < 0 | bootstrap 95% | p |', '|---|---|---|---|---|---|---|']
T += [f'| {k} | {f(t)} |' for k, t in R['descriptive'].items()]
T += ['', '## Gain (model − competitor) by bin, mean over subjects (n subjects)', '', '| dataset | estimate | ' + ' | '.join(BLAB) + ' |',
      '|---|---|' + '---|' * len(BLAB)]
curves = {}
for name, cells in list(W9.items()):
    for var, src in (('cwlS4', W9), ('cwlQ', W9), ('cwlS8', W9), ('cwlM', W8), ('far', W9)):
        ps_ = {}
        for cell in src[name]:
            for S, r in cell.items():
                for i, v in enumerate(r['bins'][var]['delta']):
                    if v is not None:
                        ps_.setdefault((S, i), []).append(v)
        acc = [[] for _ in BLAB]
        for (S, i), v in ps_.items():
            acc[i].append(float(np.mean(v)))
        curves[(name, var)] = [(np.mean(a) if a else np.nan, np.std(a) / np.sqrt(len(a)) if len(a) > 1 else np.nan, len(a)) for a in acc]
        T.append(f"| {name} | {var} | " + ' | '.join((f'{m:+.4f} ({n})' if n else '–') for m, _, n in curves[(name, var)]) + ' |')
open(f'{OUT}/w9_stats.md', 'w').write('\n'.join(T) + '\n')
print('\n'.join(T))
PAPER = len(sys.argv) > 9 and sys.argv[9] == 'paper'        # neutral labels for the manuscript
LAB = ({'cwlM': 'contamination, raw source map', 'cwlS4': 'contamination, source map smoothed 4 mm (pre-declared)',
        'cwlQ': "contamination, model's own estimate", 'far': 'far leak'} if PAPER else
       {'cwlM': 'week 8: raw source map', 'cwlS4': 'week 9: source map smoothed 4 mm (primary)', 'cwlQ': "week 9: model's own estimate",
        'far': 'far leak (week 2)'})
PTITLE = {'HCP 3.2 mm': 'HCP 7T, emulated 3.2 mm', 'HCP 2.4 mm': 'HCP 7T, emulated 2.4 mm',
          'StudyForrest (4 cells)': 'StudyForrest 7T, within session (2.1–4.0 mm)', 'ds002702 2.4 mm': 'ds002702, emulated 2.4 mm'}
FS = float(os.environ.get('GEO_FONT_SCALE', '1'))     # font scale for print-size figures (week 14; 1 = as developed)
plt.rcParams.update({'font.size': FS * 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for a, name in zip(ax, ['HCP 3.2 mm', 'HCP 2.4 mm', 'StudyForrest (4 cells)', 'ds002702 2.4 mm']):
    for var, st in (('cwlM', 'o:'), ('cwlS4', 'o-'), ('cwlQ', 's--'), ('far', '^:')):
        lab = LAB[var]
        c = curves[(name, var)]; x = np.arange(len(BLAB))
        a.errorbar(x, [v[0] for v in c], yerr=[v[1] for v in c], fmt=st, capsize=2, ms=4, label=lab)
    a.axhline(0, color='k', lw=0.8, ls=':'); a.set_xticks(range(len(BLAB))); a.set_xticklabels(BLAB, fontsize=FS * 7)
    a.set_title(PTITLE[name] if PAPER else name, fontsize=FS * 8)
    a.set_xlabel('contamination-weighted or far-leak bin' if PAPER else
                 'contamination bin (fraction of the typical response) / leak bin')
ax[0].set_ylabel('model − best competitor (mean ± SE over subjects)'); ax[0].legend(frameon=False, fontsize=FS * 6.5)
fig.suptitle(('Gain of the model over the best simple method by contamination-weighted leak and by far leak (vertices with < 2% far '
              'leak at the native grid)') if PAPER else 'Week 9: dose-response on the contrast-weighted leak with denoised response estimates '
             '(vertices clean at the native grid)', fontsize=FS * 9)
plt.tight_layout(); plt.savefig(f"{OUT}/figW9-1_denoised_cwl{'_paper' if PAPER else ''}.png", dpi=150); plt.close()
