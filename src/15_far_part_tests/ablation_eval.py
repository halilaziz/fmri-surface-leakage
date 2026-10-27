"""Week 15, analysis A: score the far-part-ablated inversions against the week-3 full model and the competitors.

subject mode (xcp_d container, TRUBA): merges the week-3 estimate file (references, competitors, full model) with the ablation
file (near / nearn renders) and computes the cross-run residual of every method in the strata of the dose-response (coherent
vertices clean at 1.6 mm, binned by the far leak at the coarse grid), over all coherent vertices, where leak > 10% and in the
pericalcarine. summary mode (local): the predictions P1–P4 of ABLATION_PLAN.md over subjects.
Usage: python3 ablation_eval.py subject S OUT_DIR [WEEK3_DIR ABLATION_DIR]
       python3 ablation_eval.py summary IN_DIR OUT_PREFIX
"""
import os, sys, json, glob
import numpy as np

CHOSEN = {'2.4': 'model_r12_s0.125', '3.2': 'model_r24_s0.25'}
BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
BINK = [f'bin_{lo:g}-{hi:g}' for lo, hi in BINS]
METHODS = ['model', 'near', 'nearn', 'upsample_smoothing', 'smoothing', 'upsample', 'standard']
LABEL = {'model': 'model, full operator', 'near': 'model, near-only operator', 'nearn': 'model, near-only renormalised', 'upsample_smoothing': 'upsampling + 1.6 mm mapping + smoothing 8 mm',
         'smoothing': 'standard + smoothing 10 mm', 'upsample': 'upsampling + 1.6 mm mapping', 'standard': 'standard mapping'}


def subject(S, out, est3, abl):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
    from evaluate import Merged, subject_context, methods, add_upsmooth, cross_run, strata
    from w3lib import COH_THR
    E = Merged([f'{est3}/{S}/{S}.week3_estimates.npz', f'{abl}/{S}/{S}.week15_ablation.npz'])
    ctx = subject_context(S, E['idx']); add_upsmooth(E, ctx)
    coh = ctx['coh'] > COH_THR; base = coh & (ctx['lf16'] < 0.02)
    res = {'subject': S, 'grids': {}}
    for g in ('2.4', '3.2'):
        mt = methods(E, g, ctx)
        names = {'model': CHOSEN[g], 'near': CHOSEN[g] + 'near', 'nearn': CHOSEN[g] + 'nearn', 'upsample_smoothing': 'upsmooth8',
                 'smoothing': 'smooth10', 'upsample': 'upsample', 'standard': 'standard'}
        st = dict(strata(ctx, g))
        for (lo, hi), k in zip(BINS, BINK):
            st[k] = base & (ctx['lf'][g] >= lo) & (ctx['lf'][g] < hi)
        res['grids'][g] = {'n': {k: int(sel.sum()) for k, sel in st.items()}}
        for lab, m in names.items():
            if m not in mt:
                print(S, g, 'missing', m, flush=True); continue
            res['grids'][g][lab] = {k: (cross_run(E, mt[m], sel) if sel.sum() >= 30 else None) for k, sel in st.items()}
        print(S, g, {lab: round(res['grids'][g][lab]['coherent'], 4) for lab in names if lab in res['grids'][g]}, flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f'{out}/{S}.ablation_eval.json', 'w'), indent=1)


def stats(vals, direction=None):
    from scipy.stats import wilcoxon
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return dict(n=int(len(v)))
    b = np.random.default_rng(0).choice(v, size=(10000, len(v)), replace=True).mean(1)
    d = dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
             n_pos=int((v > 0).sum()), n_neg=int((v < 0).sum()))
    if np.any(v != 0):
        d['p_two_sided'] = float(wilcoxon(v).pvalue)
        if direction:
            d['p_one_sided'] = float(wilcoxon(v, alternative=direction).pvalue)
    return d


def summary(indir, prefix):
    J = [json.load(open(f)) for f in sorted(glob.glob(f'{indir}/*.ablation_eval.json'))]
    subs = [j['subject'] for j in J]
    fm = lambda x: f'{x:+.4f}'.replace('-', '−')
    fp = lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'
    res = {'subjects': subs, 'grids': {}}
    L = [f'# Week 15 A: far-part ablation of the inversion ({len(subs)} development subjects; plan `ABLATION_PLAN.md`)', '',
         'Cross-run residual against the other run\'s 1.6 mm map, mean over subjects; Δ = model − competitor (upsampling + 8 mm smoothing);',
         'D_far = Δ(highest bin ≥ 30 vertices) − Δ(0–2%). "near": far entries of the operator removed per voxel; "nearn": the same, rows',
         'renormalised. Source: `src/15_far_part_tests/ablation_eval.py` from `ablation_eval/`.', '']
    for g in ('2.4', '3.2'):
        R = res['grids'][g] = {}
        # table of residuals by stratum
        strata = ['coherent', 'calc', 'clean16_leak10'] + BINK
        slab = {'coherent': 'all coherent', 'calc': 'pericalcarine', 'clean16_leak10': 'clean, leak > 10%', 'bin_0-0.02': '0–2%', 'bin_0.02-0.05': '2–5%',
                'bin_0.05-0.1': '5–10%', 'bin_0.1-0.2': '10–20%', 'bin_0.2-1.01': '> 20%'}
        L += [f'## Emulated {g} mm', '', '### Residuals (mean over subjects)', '',
              '| stratum | ' + ' | '.join(LABEL[m] for m in METHODS) + ' |', '|---|' + '---|' * len(METHODS)]
        for s in strata:
            cells = []
            for m in METHODS:
                v = np.array([j['grids'][g].get(m, {}).get(s, np.nan) if j['grids'][g].get(m, {}).get(s) is not None else np.nan for j in J], float)
                cells.append(f'{np.nanmean(v):.4f}' if np.isfinite(v).any() else '–')
            L.append(f'| {slab[s]} | ' + ' | '.join(cells) + ' |')
        # gains by bin and the contrasts
        def gain(j, m, s):
            a = j['grids'][g].get(m, {}).get(s); c = j['grids'][g]['upsample_smoothing'].get(s)
            return None if a is None or c is None else a - c
        def dfar(j, m):
            d = [gain(j, m, k) for k in BINK]
            avail = [i for i, x in enumerate(d) if x is not None]
            return None if not avail or avail[-1] == 0 or d[0] is None else d[avail[-1]] - d[0]
        L += ['', '### Gain over the competitor by far-leak bin (mean over subjects)', '', '| model | ' + ' | '.join(slab[k] for k in BINK) + ' | all coherent | D_far | Δ_clean |',
              '|---|' + '---|' * (len(BINK) + 3)]
        for m in ('model', 'near', 'nearn'):
            cells = [fm(np.nanmean([gain(j, m, k) if gain(j, m, k) is not None else np.nan for j in J])) for k in BINK]
            cells.append(fm(np.nanmean([gain(j, m, 'coherent') for j in J]))); cells.append(fm(np.nanmean([dfar(j, m) if dfar(j, m) is not None else np.nan for j in J])))
            cells.append(fm(np.nanmean([gain(j, m, 'bin_0-0.02') if gain(j, m, 'bin_0-0.02') is not None else np.nan for j in J])))
            L.append(f'| {LABEL[m]} | ' + ' | '.join(cells) + ' |')
        L += ['', '### Predictions', '', '| prediction | n | mean | median | in direction | bootstrap 95% | p (one-sided) | p (two-sided) |', '|---|---|---|---|---|---|---|---|']
        for vname in ('near', 'nearn'):
            tests = [(f'P1 D_far({vname}) − D_far(full) > 0', [dfar(j, vname) - dfar(j, 'model') if dfar(j, vname) is not None and dfar(j, 'model') is not None else None for j in J], 'greater'),
                     (f'P2 Δ_clean({vname}) − Δ_clean(full) (interval)', [gain(j, vname, 'bin_0-0.02') - gain(j, 'model', 'bin_0-0.02') if gain(j, vname, 'bin_0-0.02') is not None and gain(j, 'model', 'bin_0-0.02') is not None else None for j in J], None),
                     (f'P3 gain({vname}) − gain(full) > 0, all coherent', [gain(j, vname, 'coherent') - gain(j, 'model', 'coherent') for j in J], 'greater'),
                     (f'top-bin gain({vname}) − top-bin gain(full) > 0', [gain(j, vname, 'bin_0.2-1.01') - gain(j, 'model', 'bin_0.2-1.01') if gain(j, vname, 'bin_0.2-1.01') is not None and gain(j, 'model', 'bin_0.2-1.01') is not None else None for j in J], 'greater')]
            for name, vals, direction in tests:
                st = stats(vals, direction); R[name] = st
                if st.get('n', 0) >= 2:
                    nd = st['n_pos'] if direction == 'greater' else (st['n_neg'] if direction == 'less' else st['n_pos'])
                    L.append(f"| {name} | {st['n']} | {fm(st['mean'])} | {fm(st['median'])} | {nd}/{st['n']} | [{fm(st['ci95'][0])}, {fm(st['ci95'][1])}] | "
                             f"{fp(st['p_one_sided']) if 'p_one_sided' in st else '–'} | {fp(st['p_two_sided']) if 'p_two_sided' in st else '–'} |")
        L.append('')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'subject':
        W = '/path/to/geo_root/results'
        subject(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else f'{W}/week3/test_all', sys.argv[5] if len(sys.argv) > 5 else f'{W}/week15/ablation')
    else:
        summary(sys.argv[2], sys.argv[3])
