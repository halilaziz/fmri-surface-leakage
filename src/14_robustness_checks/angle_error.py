"""Week 14: the model's gain in physical units. Polar-angle error, in degrees, of the cross-run prediction against the other
run's 1.6 mm map, per vertex, for the week-3 methods at their chosen settings (no new fits: reads the week-3 test estimates).

Per subject, grid, method and run direction: the prediction is conjugated and aligned to the reference with the best complex
scaling c over all coherent vertices (one global phase per subject, grid, method and direction, as in the residual score of
week 3); the per-vertex error is e = angle(z_ref · conj(c · p)), and |e| in degrees is the polar-angle error, because the wedge
sweeps 360° of polar angle per stimulus cycle. Vertices are weighted equally (the residual score weights them by response
energy). Strata: all coherent vertices; coherent pericalcarine; coherent vertices clean at 1.6 mm (far leak < 2%) by the far leak
of standard mapping at the coarse grid (0–2, 2–5, 5–10, 10–20, > 20%; the bins of the dose-response analysis) and > 10%.
Also the run-to-run disagreement of the two 1.6 mm maps themselves (CCW against conj(CW), aligned the same way): the noise
floor of the reference. Means and medians per stratum are averaged over the two directions.

Usage (xcp_d container, TRUBA): python3 angle_error.py subject S OUT_DIR [EST_DIR]   -> OUT_DIR/S.angle_error.json
       (local)                  python3 angle_error.py summary IN_DIR OUT_PREFIX     -> OUT_PREFIX.md / .json
"""
import os, sys, json, glob
import numpy as np

BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
CHOSEN = {'2.4': 'model_r12_s0.125', '3.2': 'model_r24_s0.25'}          # week-3 settings (results/week3/_eval/week3_test_scores.json)
METHODS = ('standard', 'smoothing', 'upsample', 'upsample_smoothing', 'model', 'reference_runs')
LABEL = {'standard': 'standard mapping', 'smoothing': 'standard + smoothing 10 mm', 'upsample': 'upsampling + 1.6 mm mapping',
         'upsample_smoothing': 'upsampling + 1.6 mm mapping + smoothing 8 mm', 'model': 'model', 'reference_runs': '1.6 mm maps, run against run'}
STRATA_ORDER = ['coherent', 'calc', 'bin_0-0.02', 'bin_0.02-0.05', 'bin_0.05-0.1', 'bin_0.1-0.2', 'bin_0.2-1.01', 'clean16_leak10',
                'strong', 'strong_clean16_leak10']
STRATA_LABEL = {'coherent': 'all coherent vertices (coherence > 0.3)', 'calc': 'pericalcarine', 'bin_0-0.02': 'far leak 0–2%', 'bin_0.02-0.05': '2–5%',
                'bin_0.05-0.1': '5–10%', 'bin_0.1-0.2': '10–20%', 'bin_0.2-1.01': '> 20%', 'clean16_leak10': '> 10%',
                'strong': 'strongly coherent vertices (coherence > 0.5)', 'strong_clean16_leak10': 'coherence > 0.5, clean at 1.6 mm, far leak > 10%'}


def fnum(x, d=1, sign=False):
    return (f'{x:+.{d}f}' if sign else f'{x:.{d}f}').replace('-', '−')


def angle_error_deg(z, p, fit):
    """|angle| in degrees between reference z and predictor p after the best complex scaling of p fitted on `fit`."""
    c = np.vdot(p[fit], z[fit]) / max(np.vdot(p[fit], p[fit]).real, 1e-30)
    return np.degrees(np.abs(np.angle(z * np.conj(c * p))))


def subject(S, out, est):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
    from evaluate import Merged, subject_context, methods, RUNS, OTHER, R0, R1
    from w3lib import COH_THR, surface_smooth
    E = Merged([f'{est}/{S}/{S}.week3_estimates.npz'])
    ctx = subject_context(S, E['idx'])
    sm8 = surface_smooth(ctx['Cr'], ctx['Gr'], 8.0)                      # the chosen upsampling competitor: upsmooth8
    coh = ctx['coh'] > COH_THR; base = coh & (ctx['lf16'] < 0.02)
    res = {'subject': S, 'n_coherent': int(coh.sum()), 'grids': {}}
    for g in ('2.4', '3.2'):
        mt = methods(E, g)
        pred = {'standard': mt['standard'], 'smoothing': mt['smooth10'], 'upsample': mt['upsample'],
                'upsample_smoothing': {r: sm8(E[f'up_{g}_{r}'].astype(np.complex128)) for r in RUNS}, 'model': mt[CHOSEN[g]]}
        strong = ctx['coh'] > 0.5                                           # vertices whose single-run reference is reliable
        strata = {'coherent': coh, 'calc': coh & ctx['calc'], 'clean16_leak10': base & (ctx['lf'][g] > 0.10),
                  'strong': strong, 'strong_clean16_leak10': strong & (ctx['lf16'] < 0.02) & (ctx['lf'][g] > 0.10)}
        for lo, hi in BINS:
            strata[f'bin_{lo:g}-{hi:g}'] = base & (ctx['lf'][g] >= lo) & (ctx['lf'][g] < hi)
        errs = {lab: [angle_error_deg(E[f'z16_{OTHER[r]}'].astype(complex), np.conj(p[r].astype(complex)), coh) for r in RUNS]
                for lab, p in pred.items()}
        errs['reference_runs'] = [angle_error_deg(E[f'z16_{R0}'].astype(complex), np.conj(E[f'z16_{R1}'].astype(complex)), coh)]
        out_g = {}
        for lab, el in errs.items():
            out_g[lab] = {}
            for s, sel in strata.items():
                d = {'n': int(sel.sum())}
                if sel.sum() >= 30:
                    d['mean'] = float(np.mean([e[sel].mean() for e in el])); d['median'] = float(np.mean([np.median(e[sel]) for e in el]))
                    d['within30'] = float(np.mean([(e[sel] < 30).mean() for e in el]))
                out_g[lab][s] = d
        res['grids'][g] = out_g
        print(S, g, {lab: round(out_g[lab]['coherent'].get('mean', np.nan), 1) for lab in errs}, flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f'{out}/{S}.angle_error.json', 'w'), indent=1)


def summary(indir, prefix):
    files = sorted(glob.glob(f'{indir}/*.angle_error.json'))
    J = [json.load(open(f)) for f in files]
    subs = [j['subject'] for j in J]
    res = {'subjects': subs, 'grids': {}}
    L = [f'# Week 14: polar-angle error in degrees ({len(subs)} development subjects; week-3 fits, chosen settings)', '',
         'Mean over subjects of the per-subject mean absolute polar-angle error (degrees) of the cross-run prediction against the other',
         "run's 1.6 mm map, coherent vertices weighted equally, averaged over the two wedge directions; ± SE over subjects. Strata as in",
         'the far-leak dose-response (coherent, clean at 1.6 mm, binned by the far leak at the coarse grid). The last column is the',
         'run-to-run disagreement of the two 1.6 mm maps themselves. Δ = model − upsampling + 8 mm smoothing, with the subjects in which',
         'the model has the smaller error. Source: `src/14_robustness_checks/angle_error.py`; per-subject files in `angle_error/`.', '']
    for g in ('2.4', '3.2'):
        res['grids'][g] = {}
        L += [f'## Emulated {g} mm', '', '| stratum | n vertices (mean) | ' + ' | '.join(LABEL[m] for m in METHODS) + ' | Δ model − upsample+smooth | model better |',
              '|---|---|' + '---|' * (len(METHODS) + 2)]
        for s in STRATA_ORDER:
            row = {'n_vertices_mean': float(np.mean([j['grids'][g]['model'][s]['n'] for j in J]))}
            cells = []
            for m in METHODS:
                v = np.array([j['grids'][g][m][s].get('mean', np.nan) for j in J], float)
                ok = np.isfinite(v)
                row[m] = dict(mean=float(v[ok].mean()), se=float(v[ok].std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else None, n=int(ok.sum()))
                cells.append(f"{fnum(v[ok].mean())} ± {fnum(row[m]['se'])}" if ok.sum() > 1 else '–')
            d = np.array([j['grids'][g]['model'][s].get('mean', np.nan) - j['grids'][g]['upsample_smoothing'][s].get('mean', np.nan) for j in J], float)
            ok = np.isfinite(d)
            row['delta_model_upsmooth'] = dict(mean=float(d[ok].mean()), se=float(d[ok].std(ddof=1) / np.sqrt(ok.sum())), n_better=int((d[ok] < 0).sum()), n=int(ok.sum()))
            res['grids'][g][s] = row
            L.append(f"| {STRATA_LABEL[s]} | {row['n_vertices_mean']:.0f} | " + ' | '.join(cells)
                     + f" | {fnum(d[ok].mean(), sign=True)} ± {fnum(d[ok].std(ddof=1) / np.sqrt(ok.sum()))} | {(d[ok] < 0).sum()}/{ok.sum()} |")
        L.append('')
    # medians and the share of vertices within 30°, all coherent vertices
    L += ['## Medians and the share of vertices within 30° (all coherent vertices)', '', '| grid | method | median error (°) | within 30° |', '|---|---|---|---|']
    for g in ('2.4', '3.2'):
        for m in METHODS:
            med = np.array([j['grids'][g][m]['coherent']['median'] for j in J]); w = np.array([j['grids'][g][m]['coherent']['within30'] for j in J])
            res['grids'][g]['coherent_median'] = res['grids'][g].get('coherent_median', {}); res['grids'][g]['coherent_median'][m] = dict(median=float(med.mean()), within30=float(w.mean()))
            L.append(f'| {g} mm | {LABEL[m]} | {fnum(med.mean())} | {100 * w.mean():.0f}% |')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n')
    json.dump(res, open(f'{prefix}.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'subject':
        subject(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        summary(sys.argv[2], sys.argv[3])
