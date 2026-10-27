"""Week 4 evaluation on the HCP test subjects (emulated grids):
  1. uncertainty calibration: run-to-run differences of the estimates vs the predicted noise variance (model and standard mapping),
     in noise-dominated and in responsive vertices; posterior coverage against the known truth of the week-3 simulation;
  2. misregistration: cross-run score (week-3 metric) of the model with surfaces shifted 0.5 / 1.0 mm, vs nominal and competitors;
  3. depth component: b1/b0 ratio in responsive cortex, and the cross-run score with the depth-resolved model;
  4. theory link: averaging-kernel spread at the week-3 sample vertices vs week 1's orientation-dependent resolvable wavelength.
Usage (xcp_d container): python3 evaluate_hcp.py OUTDIR
"""
import os, sys, json, glob
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '01_resolvability'))
from evaluate import Merged, subject_context, strata, cross_run, methods, add_upsmooth, OTHER
from w3lib import *
from surface_maps import lam_ideal

R3 = f'{W_ROOT}/results/week3'
R4 = f'{W_ROOT}/results/week4/hcp'
CHOSEN = {'2.4': 'r12_s0.125', '3.2': 'r24_s0.25'}
Q = {0.5: -np.log(0.5), 0.8: -np.log(0.2), 0.95: -np.log(0.05)}      # Exp(1) quantiles of |complex Gaussian|^2 / variance


def ms(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return (float(x.mean()), float(x.std()), int(len(x))) if len(x) else (np.nan, np.nan, 0)


def smoother(ctx, fw):
    c = ctx.setdefault('smoothers', {})
    if fw not in c:
        c[fw] = surface_smooth(ctx['Cr'], ctx['Gr'], fw)
    return c[fw]


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    subs = sorted(os.path.basename(d) for d in glob.glob(f'{R4}/*') if os.path.exists(f'{d}/{os.path.basename(d)}.week4_uncertainty.npz'))
    R3json = json.load(open(f'{R3}/_eval/week3_test_scores.json'))
    res = {'subjects': subs, 'calibration': {}, 'shift': {}, 'depth': {}, 'theory': {}}
    for S in subs:
        files = [f'{R3}/test_all/{S}/{S}.week3_estimates.npz'] + [f for f in (f'{R4}/{S}/{S}.week4_shift.npz', f'{R4}/{S}/{S}.week4_depth.npz',
                                                                             f'{R4}/{S}/{S}.week4_shiftcomp.npz') if os.path.exists(f)]
        E = Merged(files); U = np.load(f'{R4}/{S}/{S}.week4_uncertainty.npz')
        ctx = subject_context(S, E['idx']); add_upsmooth(E, ctx)
        coh = ctx['coh']
        right = E['idx'] >= len(ctx['geo']['L']['white'])
        cal = {}
        for g in ('2.4', '3.2'):
            if f'noise_sd_{g}' not in U.files:
                continue
            nsd, psd, ssd = U[f'noise_sd_{g}'].astype(float), U[f'post_sd_{g}'].astype(float), U[f'std_noise_sd_{g}'].astype(float)
            c = {}
            for lab, fa, fb, sd in (('model', U[f'f_{g}_ccw'], U[f'f_{g}_cw'], nsd), ('standard', U[f'z_{g}_ccw'], U[f'z_{g}_cw'], ssd)):
                fa, fb = fa.astype(complex), np.conj(fb.astype(complex))
                sel = coh > COH_THR
                k = np.vdot(fb[sel], fa[sel]) / np.vdot(fb[sel], fb[sel])            # CW = k conj(CCW) up to noise
                r = np.abs(fa - k * fb) ** 2 / np.maximum((1 + abs(k) ** 2) * sd ** 2, 1e-30)
                # the model's noise s.d. is estimated from NSAMP = 16 real Monte-Carlo draws: E[sigma^2 / sigma_hat^2] = m / (m - 2),
                # so its mean ratio is also given corrected by (m - 2) / m; standard mapping's variance is exact
                corr = (16 - 2) / 16 if lab == 'model' else 1.0
                okv = sd > 1e-3 * np.median(sd[sd > 0])          # vertices mapped onto masked-out voxels have ~0 predicted variance
                for vs, m in (('noise_dominated', (coh < 0.1) & okv), ('coherent', (coh > COH_THR) & okv)):
                    c[f'{lab}:{vs}'] = dict(mean_ratio=float(np.mean(r[m])), mean_ratio_mc_corrected=float(corr * np.mean(r[m])),
                                            median_ratio_over_ln2=float(np.median(r[m]) / np.log(2)),
                                            frac_beyond_95=float(np.mean(r[m] > Q[0.95])), n=int(m.sum()))
                # one global variance inflation, estimated on the noise-dominated vertices of the other hemisphere
                m0 = (coh < 0.1) & okv; fb = []
                for hm in (right, ~right):
                    cf = np.mean(r[m0 & ~hm]); fb.append(np.mean(r[m0 & hm] / cf > Q[0.95]))
                c[f'{lab}:noise_dominated']['frac_beyond_95_after_other_hemisphere_scale'] = float(np.mean(fb))
            # posterior coverage in the week-3 known-truth simulation (same subject, grid and setting)
            if f'sim_f_{g}_ccw' in E.files:
                for vs, m in (('all', np.ones(len(psd), bool)), ('coherent', coh > COH_THR)):
                    zz = np.concatenate([np.abs(E[f'sim_f_{g}_{r}'].astype(complex) - E[f'sim_truth_f_{g}_{r}'].astype(complex))[m] ** 2 / psd[m] ** 2 for r in WEDGE])
                    c[f'sim_posterior:{vs}'] = {f'coverage_{p:g}': float(np.mean(zz < q)) for p, q in Q.items()}
            cal[g] = c
        res['calibration'][S] = cal
        # misregistration and depth: cross-run score, week-3 metric
        sh, dp = {}, {}
        for g in ('2.4', '3.2'):
            st = strata(ctx, g); mt = methods(E, g, ctx)
            # competitors under the same misregistration (run_hcp.py shiftcomp; week-3 chosen FWHMs 10 / 8 mm)
            for d in ('0.5', '1'):
                if f'zshift{d}_{g}_ccw' in E.files:
                    for key, src, fw in (('smooth10', 'zshift', 10.0), ('upsmooth8', 'upshift', 8.0)):
                        sm = smoother(ctx, fw)
                        mt[f'{key}_shift{d}'] = {r: sm(E[f'{src}{d}_{g}_{r}'].astype(np.complex128)).astype(np.complex64) for r in WEDGE}
                    mt[f'standard_shift{d}'] = {r: E[f'zshift{d}_{g}_{r}'] for r in WEDGE}
            names = {'model': f'model_{CHOSEN[g]}', 'model_shift0.5': 'model_shift0.5', 'model_shift1': 'model_shift1',
                     'model_depth_b0only': 'model_depth_b0only', 'model_depth_full': 'model_depth_full', 'upsample_smoothing': 'upsmooth8',
                     'standard': 'standard', 'smoothing': 'smooth10',
                     **{f'{lab}_shift{d}': f'{key}_shift{d}' for d in ('0.5', '1')
                        for lab, key in (('upsample_smoothing', 'upsmooth8'), ('smoothing', 'smooth10'), ('standard', 'standard'))}}
            sc = {lab: {s: (cross_run(E, mt[m], sel) if sel.sum() >= 30 else None) for s, sel in st.items()} for lab, m in names.items() if m in mt}
            sh[g] = sc
            if f'f0_{g}_ccw' in E.files:
                ratio = {}
                for vs, m in (('coherent', coh > COH_THR), ('pericalcarine', (coh > COH_THR) & ctx['calc'])):
                    rr = [float(np.real(np.vdot(E[f'f0_{g}_{r}'][m], E[f'f1_{g}_{r}'][m])) / np.vdot(E[f'f0_{g}_{r}'][m], E[f'f0_{g}_{r}'][m]).real) for r in WEDGE]
                    pos = [float(np.mean(np.real(E[f'f1_{g}_{r}'][m] * np.conj(E[f'f0_{g}_{r}'][m])) > 0)) for r in WEDGE]
                    ratio[vs] = dict(b1_over_b0=float(np.mean(rr)), frac_positive=float(np.mean(pos)))
                dp[g] = ratio
        res['shift'][S] = sh; res['depth'][S] = dp
        # theory link: kernel spread vs resolvable wavelength of the local sheet orientation (MNI normals, emulated voxel sizes)
        th = {}
        geo = ctx['geo']; idx = E['idx']; nL = len(geo['L']['white'])
        nrm = np.vstack([vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])[1] for h in 'LR'])[idx]
        for g, hvox in (('2.4', np.array([2.4107, 2.3912, 2.4107])), ('3.2', np.array([3.1719, 3.2, 3.1719]))):
            rows = R3json['kernels'].get(S, {}).get(g, [])
            if not rows:
                continue
            v = np.array([r['vertex'] for r in rows]); lam = lam_ideal(nrm[v], hvox)
            th[g] = dict(n=len(rows), rho_model_spread=float(spearmanr(lam, [r['model_spread_mm'] for r in rows])[0]),
                         rho_std_spread=float(spearmanr(lam, [r['std_spread_mm'] for r in rows])[0]),
                         lam_range_mm=[float(lam.min()), float(lam.max())])
        res['theory'][S] = th
        print(S, 'done', {g: {k: round(v['mean_ratio'], 3) for k, v in c.items() if 'mean_ratio' in v} for g, c in cal.items()}, flush=True)
    # group summaries
    G = {}
    for g in ('2.4', '3.2'):
        keys = sorted({k for S in subs for k in res['calibration'][S].get(g, {})})
        G[f'calibration_{g}'] = {k: {m: ms([res['calibration'][S][g][k].get(m) for S in subs if k in res['calibration'][S].get(g, {})])
                                     for m in next(res['calibration'][S][g][k] for S in subs if k in res['calibration'][S].get(g, {})).keys() if m != 'n'}
                                 for k in keys}
        labs = sorted({l for S in subs for l in res['shift'][S].get(g, {})})
        G[f'shift_{g}'] = {l: {s: ms([res['shift'][S][g][l][s] for S in subs if l in res['shift'][S].get(g, {})])
                               for s in ('coherent', 'clean16_leak5', 'clean16_leak10', 'calc')} for l in labs}
        G[f'depth_{g}'] = {vs: {m: ms([res['depth'][S][g][vs][m] for S in subs if g in res['depth'][S]]) for m in ('b1_over_b0', 'frac_positive')}
                           for vs in ('coherent', 'pericalcarine')}
        G[f'theory_{g}'] = {m: ms([res['theory'][S][g][m] for S in subs if g in res['theory'][S]]) for m in ('rho_model_spread', 'rho_std_spread')}
    res['group'] = G
    json.dump(res, open(f'{out}/week4_hcp_eval.json', 'w'), indent=1, default=float)
    print(json.dumps(G, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else x)[:6000])


if __name__ == '__main__':
    main()
