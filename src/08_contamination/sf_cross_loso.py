"""Week 8, plan B: StudyForrest cross-session predictions (week 4, all settings logged) with each method's setting chosen on the
other subjects of the same pair type (leave-one-subject-out), the rule that gave the ds002702 3T -> 7T gain (week 7, 3c).
Primary: 3T -> 7T pairs with a voxel ratio >= 2.5 (3T 2.0 / 3.0 mm -> 7T 0.8 mm), visual vertices; one value per subject (mean
over its pair types); one-sided Wilcoxon signed-rank and a bootstrap interval. Secondary: all 3T -> 7T, all 7T -> 7T, covered.
Usage (local): python3 sf_cross_loso.py WEEK4_LOG_DIR PLAN_SHA1_FILE OUTDIR
"""
import sys, os, json, glob
import numpy as np
from scipy.stats import wilcoxon

LD, SHA, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
MM = {'r08': 0.8, 'r14': 1.4, 'r20': 2.0, 'r30': 3.0}
rng = np.random.default_rng(0)


def comp_keys(sc):
    return [k for k in sc if k in ('standard', 'smoothing', 'resample_smoothing') or k.startswith(('x_smooth', 'x_resample_smooth'))]


def model_keys(sc):
    return [k for k in sc if k == 'model' or (k.startswith('x_model_r'))]


def boot(v, n=10000):
    v = np.asarray(v, float); b = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def test(v):
    v = np.asarray(v, float)
    p = float(wilcoxon(v, alternative='less').pvalue) if len(v) >= 2 and np.any(v != 0) else float('nan')
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), n_better=int((v < 0).sum()), ci95=boot(v), p_one_sided=p)


logs = {os.path.basename(p).split('.')[0]: json.load(open(p)) for p in sorted(glob.glob(f'{LD}/sub-*/*.sf_model_log.json'))}
pairs = {}                                            # pair type -> subject -> scores
for S, L in logs.items():
    for pk, sc in L['pairs'].items():
        pairs.setdefault(pk, {})[S] = sc
res = {'plan_sha1': open(SHA).read().split()[0], 'pair_types': {}}
for st in ('visual', 'covered'):
    for pk, SS in sorted(pairs.items()):
        subs = sorted(SS)
        if len(subs) < 3:
            continue
        ck, mk = comp_keys(SS[subs[0]]), model_keys(SS[subs[0]])
        val = lambda S, k: SS[S][k][st] if SS[S][k].get(st) is not None else np.nan
        C = np.array([[val(S, k) for k in ck] for S in subs]); M = np.array([[val(S, k) for k in mk] for S in subs])
        rec = {}
        for i, S in enumerate(subs):
            o = [j for j in range(len(subs)) if j != i]
            ci = int(np.nanargmin(np.nanmean(C[o], 0))); mi = int(np.nanargmin(np.nanmean(M[o], 0)))
            rec[S] = dict(competitor=float(C[i, ci]), competitor_key=ck[ci], model=float(M[i, mi]), model_key=mk[mi],
                          delta=float(M[i, mi] - C[i, ci]), split_half=SS[S]['ceiling_split_half'][st])
        src, tgt = pk.split('->')
        res['pair_types'].setdefault(st, {})[pk] = dict(ratio=MM[src.split('-')[1]] / MM[tgt.split('-')[1]], subjects=rec)


def per_subject(st, cond):
    acc = {}
    for pk, r in res['pair_types'][st].items():
        if cond(pk, r):
            for S, x in r['subjects'].items():
                acc.setdefault(S, []).append(x['delta'])
    return {S: float(np.mean(v)) for S, v in acc.items()}


is3 = lambda pk: pk.startswith('3T') and '->7T' in pk
tests = {
    'primary: 3T->7T, ratio >= 2.5, visual': per_subject('visual', lambda pk, r: is3(pk) and r['ratio'] >= 2.5),
    'secondary: all 3T->7T, visual': per_subject('visual', lambda pk, r: is3(pk)),
    'secondary: all 7T->7T, visual': per_subject('visual', lambda pk, r: pk.startswith('7T')),
    'secondary: 3T->7T, ratio >= 2.5, covered': per_subject('covered', lambda pk, r: is3(pk) and r['ratio'] >= 2.5),
    'secondary: all 3T->7T, covered': per_subject('covered', lambda pk, r: is3(pk)),
}
res['tests'] = {k: dict(per_subject=v, **test(list(v.values()))) for k, v in tests.items()}
sec = [k for k in res['tests'] if k.startswith('secondary')]
ps = sorted((res['tests'][k]['p_one_sided'], k) for k in sec)
for i, (p, k) in enumerate(ps):                          # Holm over the secondary tests
    res['tests'][k]['p_holm'] = float(min(1.0, max(p * (len(ps) - j) for j, (p, _) in enumerate(ps[:i + 1]))))
json.dump(res, open(f'{OUT}/sf_cross_loso.json', 'w'), indent=1)
T = ['# StudyForrest cross-session, settings chosen on the other subjects of the pair type (week 8, plan B)', '',
     f"Plan SHA-1 {res['plan_sha1']}. Residual of the target map (lower is better); Δ = model − competitor per subject, mean over "
     'the pair types in the set. No distortion correction of the 3T sessions (no field maps).', '',
     '| test | n | mean Δ | median Δ | model better | bootstrap 95% | p (one-sided) | p (Holm) |', '|---|---|---|---|---|---|---|---|']
for k, t in res['tests'].items():
    T.append(f"| {k} | {t['n']} | {t['mean']:+.4f} | {t['median']:+.4f} | {t['n_better']}/{t['n']} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | "
             f"{t['p_one_sided']:.3f} | {t.get('p_holm', float('nan')):.3f} |")
T += ['', '## Per pair type (visual)', '', '| pair | ratio | subjects | mean competitor | mean model | mean Δ | model better | target split-half | settings chosen (most often) |',
      '|---|---|---|---|---|---|---|---|---|']
for pk, r in sorted(res['pair_types']['visual'].items(), key=lambda kv: (-kv[1]['ratio'], kv[0])):
    x = r['subjects'].values(); d = np.array([v['delta'] for v in x])
    cks = [v['competitor_key'] for v in r['subjects'].values()]; mks = [v['model_key'] for v in r['subjects'].values()]
    T.append(f"| {pk} | {r['ratio']:.2f} | {len(d)} | {np.mean([v['competitor'] for v in r['subjects'].values()]):.4f} | "
             f"{np.mean([v['model'] for v in r['subjects'].values()]):.4f} | {d.mean():+.4f} | {int((d < 0).sum())}/{len(d)} | "
             f"{np.mean([v['split_half'] for v in r['subjects'].values()]):.4f} | {max(set(cks), key=cks.count)}; {max(set(mks), key=mks.count)} |")
open(f'{OUT}/sf_cross_loso.md', 'w').write('\n'.join(T) + '\n')
print('\n'.join(T))
