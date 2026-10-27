"""Week 17 summaries (local), as PLAN_NYU.md.
  tune IN_DIR OUT_PREFIX     selection on the 8 tuning subjects (week-3 rule) -> OUT_PREFIX.md/.json and SETTINGS_NYU.txt next to it
  test IN_DIR TUNE_JSON OUT_PREFIX   N1-N3 (Holm over the six primaries), secondary, descriptive, leak figures, on the test subjects
"""
import sys, os, json, glob
import numpy as np
from scipy.stats import wilcoxon

PROJECT = os.environ.get('GEO_PROJECT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # folder holding results/week17
SPLIT = json.load(open(os.path.join(PROJECT, 'results', 'week17', 'nyu_split.json')))
GRIDS = ('3.0', '4.0')
BN = ['0-0.02', '0.02-0.05', '0.05-0.1', '0.1-0.2', '0.2-inf']
fm = lambda x, d=4: ('–' if x is None or not np.isfinite(x) else f'{x:+.{d}f}'.replace('-', '−'))
fp = lambda p: '–' if p is None else ('< 0.001' if p < 0.001 else f'{p:.3f}')


def load(indir, subs):
    out = {}
    for S in subs:
        f = f'{indir}/{S}.nyu_scores.json'
        if os.path.exists(f):
            j = json.load(open(f))
            if not j.get('excluded') and 't_total_s' in j:
                out[S] = j
    return out


def stats(v, alt='less'):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return dict(n=int(len(v)))
    b = np.random.default_rng(0).choice(v, size=(10000, len(v)), replace=True).mean(1)
    d = dict(n=int(len(v)), mean=float(v.mean()), n_neg=int((v < 0).sum()), n_pos=int((v > 0).sum()),
             ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    if np.any(v != 0):
        d['p'] = float(wilcoxon(v, alternative=alt).pvalue); d['p_two_sided'] = float(wilcoxon(v).pvalue)
    return d


def holm(ps):
    """Holm-adjusted p for a dict name -> p."""
    items = sorted((p, k) for k, p in ps.items() if p is not None); m = len(items); adj = {}; run = 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); adj[k] = run
    return adj


def tune(indir, prefix):
    J = load(indir, SPLIT['tuning']); subs = sorted(J)
    res = {'subjects': subs, 'grids': {}}
    L = [f'# Week 17: NYU settings chosen on the tuning subjects ({len(subs)} of {len(SPLIT["tuning"])}; week-3 rule)', '',
         'Mean cross-half residual over responsive vertices (lower is better); the competitor of the tests is the better of the',
         'two chosen smoothing competitors.', '']
    settings = []
    for g in GRIDS:
        meth = sorted(set.intersection(*[set(J[S]['grids'][g]['scores']) for S in subs]))
        mean = {m: float(np.mean([J[S]['grids'][g]['scores'][m]['responsive'] for S in subs])) for m in meth
                if all(J[S]['grids'][g]['scores'][m]['responsive'] is not None for S in subs)}
        pick = lambda pre: min((m for m in mean if m.startswith(pre) and (pre != 'smooth' or not m.startswith('upsmooth'))), key=mean.get)
        model, smooth, ups = pick('model_'), pick('smooth'), pick('upsmooth')
        comp = smooth if mean[smooth] < mean[ups] else ups
        rho, ss = model[len('model_r'):].split('_s')
        settings.append(f'{g}={rho}:{ss}')
        res['grids'][g] = dict(model=model, smoothing=smooth, upsmooth=ups, competitor=comp, means=mean)
        L += [f'## {g} mm', '', f'Choice: model **{model}** ({mean[model]:.4f}), smoothing **{smooth}** ({mean[smooth]:.4f}), upsampling + smoothing **{ups}** '
              f'({mean[ups]:.4f}); competitor of the tests **{comp}**. Standard mapping {mean.get("standard", float("nan")):.4f}, upsampling {mean.get("upsample", float("nan")):.4f}, '
              f'native source half (no coarsening) {mean.get("native", float("nan")):.4f}.', '', '| model setting | residual |', '|---|---|']
        for m in sorted((m for m in mean if m.startswith('model_')), key=mean.get)[:8]:
            L.append(f'| {m} | {mean[m]:.4f} |')
        L.append('')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1)
    open(os.path.join(os.path.dirname(prefix), 'SETTINGS_NYU.txt'), 'w').write(';'.join(settings) + '\n')
    print('\n'.join(L)); print('SETTINGS:', ';'.join(settings))


def dcontrast(sc, model, comp, var):
    d = [((sc[model].get(f'{var}_{b}') - sc[comp].get(f'{var}_{b}')) if (sc[model].get(f'{var}_{b}') is not None and sc[comp].get(f'{var}_{b}') is not None) else None) for b in BN]
    av = [i for i, x in enumerate(d) if x is not None]
    return None if not av or av[-1] == 0 or d[0] is None else d[av[-1]] - d[0], d


def test(indir, tune_json, prefix):
    T = json.load(open(tune_json)); J = load(indir, SPLIT['test']); subs = sorted(J)
    excl = sorted(set(SPLIT['test']) - set(J))
    res = {'subjects': subs, 'not_included': excl, 'grids': {}, 'leak': {}}
    L = [f'# Week 17: NYU Retinotopy, external replication — test subjects ({len(subs)} of {len(SPLIT["test"])})', '',
         'Δ = residual(model) − residual(competitor), negative favours the model; one-sided Wilcoxon (predicted direction: less),',
         'bootstrap 95% interval of the mean (10,000 resamples, seed 0); Holm over the six primary tests (N1–N3 × two grids).', '',
         f'Not included (preprocessing exclusions or missing): {", ".join(excl) if excl else "none"}.', '',
         '| grid | test | competitor | model | mean | subjects in direction | bootstrap 95% | p (one-sided) | p (Holm) |', '|---|---|---|---|---|---|---|---|---|']
    prim, sec = {}, {}
    for g in GRIDS:
        model, comp = T['grids'][g]['model'], T['grids'][g]['competitor']; res['grids'][g] = {}
        rows = {}
        rows['N1: Δ responsive'] = [J[S]['grids'][g]['scores'][model]['responsive'] - J[S]['grids'][g]['scores'][comp]['responsive']
                                   if J[S]['grids'][g]['scores'][model]['responsive'] is not None and J[S]['grids'][g]['scores'][comp]['responsive'] is not None else None for S in subs]
        rows['N2: D_far'] = [dcontrast(J[S]['grids'][g]['scores'], model, comp, 'far')[0] for S in subs]
        rows['N3: D_CWL-M'] = [dcontrast(J[S]['grids'][g]['scores'], model, comp, 'cwlM')[0] for S in subs]
        rows['descriptive: Δ responsive vs plain upsampling'] = [J[S]['grids'][g]['scores'][model]['responsive'] - J[S]['grids'][g]['scores']['upsample']['responsive']
                                                                 if J[S]['grids'][g]['scores'][model]['responsive'] is not None and J[S]['grids'][g]['scores']['upsample']['responsive'] is not None else None for S in subs]
        for st in ('clean_leak10', 'calc'):
            rows[f'secondary: Δ {st}'] = [(J[S]['grids'][g]['scores'][model][st] - J[S]['grids'][g]['scores'][comp][st])
                                         if J[S]['grids'][g]['scores'][model][st] is not None and J[S]['grids'][g]['scores'][comp][st] is not None else None for S in subs]
        for k, v in rows.items():
            st_ = stats(v); res['grids'][g][k] = dict(st_, values=v)
            if k.startswith('N'):
                prim[f'{g}|{k}'] = st_.get('p')
            elif k.startswith('secondary'):
                sec[f'{g}|{k}'] = st_.get('p')
        res['grids'][g]['model'], res['grids'][g]['competitor'] = model, comp
        # descriptive: polar-angle error and per-bin curves
        ang = [(J[S]['grids'][g]['angle_deg'].get(model), J[S]['grids'][g]['angle_deg'].get(comp)) for S in subs]
        res['grids'][g]['angle_deg'] = dict(model=float(np.mean([a for a, _ in ang if a is not None])), competitor=float(np.mean([b for _, b in ang if b is not None])),
                                            diff=stats([a - b for a, b in ang if a is not None and b is not None]))
        for var in ('far', 'cwlM'):
            per = np.array([[np.nan if x is None else x for x in dcontrast(J[S]['grids'][g]['scores'], model, comp, var)[1]] for S in subs], float)
            res['grids'][g][f'bins_{var}'] = dict(mean=np.nanmean(per, 0).tolist(), n=np.isfinite(per).sum(0).tolist(), se=(np.nanstd(per, 0, ddof=1) / np.sqrt(np.maximum(np.isfinite(per).sum(0), 1))).tolist())
    hp, hs = holm(prim), holm(sec)
    for g in GRIDS:
        for k in res['grids'][g]:
            if not isinstance(res['grids'][g][k], dict) or 'n' not in res['grids'][g][k] or k.startswith('bins') or k == 'angle_deg':
                continue
            st_ = res['grids'][g][k]; key = f'{g}|{k}'; ph = hp.get(key, hs.get(key))
            st_['p_holm'] = ph
            L.append(f"| {g} mm | {k} | {res['grids'][g]['competitor']} | {res['grids'][g]['model']} | {fm(st_.get('mean'))} | {st_.get('n_neg', '–')}/{st_['n']} | "
                     f"[{fm(st_['ci95'][0]) if 'ci95' in st_ else '–'}, {fm(st_['ci95'][1]) if 'ci95' in st_ else '–'}] | {fp(st_.get('p'))} | {fp(ph)} |")
    L += ['', '## Descriptive', '', '| grid | polar-angle error, model / competitor (degrees) | difference (mean, subjects model better) |', '|---|---|---|']
    for g in GRIDS:
        a_ = res['grids'][g]['angle_deg']
        L.append(f"| {g} mm | {a_['model']:.2f} / {a_['competitor']:.2f} | {fm(a_['diff'].get('mean'), 2)}, {a_['diff'].get('n_neg', '–')}/{a_['diff']['n']} |")
    L += ['', 'Gain (model − competitor) by bin of the dose variable at the coarse grid (responsive vertices with < 2% far leak at 2.0 mm), mean (subjects with the bin):', '',
          '| grid | dose | 0–2% | 2–5% | 5–10% | 10–20% | > 20% |', '|---|---|---|---|---|---|---|']
    for g in GRIDS:
        for var, lab in (('far', 'far leak'), ('cwlM', 'CWL-M')):
            b = res['grids'][g][f'bins_{var}']
            L.append(f'| {g} mm | {lab} | ' + ' | '.join(f'{fm(m)} ({n})' if n else '–' for m, n in zip(b['mean'], b['n'])) + ' |')
    # leak figures of the 3T acquisition, all included subjects (tuning and test)
    allJ = load(indir, SPLIT['tuning'] + SPLIT['test'])
    L += ['', f'## Far leak of standard mapping for the 3T 2.0 mm acquisition and its emulations ({len(allJ)} subjects; area-weighted, mean ± SD)', '',
          '| grid | mean far leak (%) | cortex > 5% (%) | cortex > 10% (%) | other hemisphere (%) |', '|---|---|---|---|---|']
    for g in ('2.0',) + GRIDS:
        x = {k: np.array([100 * allJ[S]['leak'][g][k] for S in allJ]) for k in ('mean_far', 'gt5', 'gt10', 'mean_other_hemi')}
        res['leak'][g] = {k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1))) for k, v in x.items()}
        L.append(f'| {g} mm | ' + ' | '.join(f'{x[k].mean():.2f} ± {x[k].std(ddof=1):.2f}' for k in ('mean_far', 'gt5', 'gt10', 'mean_other_hemi')) + ' |')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    {'tune': lambda: tune(sys.argv[2], sys.argv[3]), 'test': lambda: test(sys.argv[2], sys.argv[3], sys.argv[4])}[sys.argv[1]]()
