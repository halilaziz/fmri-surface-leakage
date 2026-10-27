"""Week 5, Part B: settings chosen from each source session's own data, for the week-4 cross-session StudyForrest pairs.
For every source session, every week-4 setting is scored on how well the source's odd-run half predicts its even-run half at the
source's own resolution, and vice versa (no target data):
  model       posterior mean from one half (prior s.d. scaled for the half's noise: alpha = runs / runs in half), rendered W_s K_s f
  smoothing   smoothing of one half's standard map (FWHM grid of week 4)
summarize_w5.py then picks, per source and vertex set, the setting with the lowest split-half residual and reads that setting's
target scores from the week-4 prior-grid logs (x_model_r*_s*, x_smooth*, x_resample_smooth*).
Unknowns, operator, noise and sigma0 are exactly week 4's (sf_model.py), so the chosen setting is the same estimator.
Usage (xcp_d container): python3 sf_selftune.py SUBJECT [--sources 7T-r30,7T-r20,...]
"""
import os, sys, json, glob, time, argparse
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w5lib import *

OUT = f'{ROOT}/results/week5/sf_selftune'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--sources', default='')
    a = ap.parse_args(); S = a.subject
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, f'{time.time() - T0:7.0f}s', *x, flush=True)
    geo, v2r = load_anatomy(S)
    keys = sorted(k for k in os.listdir(f'{PREP}/{S}') if os.path.exists(f'{PREP}/{S}/{k}/betaR_even.nii.gz'))
    ses = {k: load_session(S, k, v2r) for k in keys}
    U = unknowns(geo, [(s['A_anat'], s['shape']) for s in ses.values()])          # as week 4: cortex inside any session
    n = U['n']; ones = np.ones(n, np.float32); sm = smoother_cache(U['Cr'], U['Gr'])
    sources = a.sources.split(',') if a.sources else [k for k in keys if not k.endswith('r08')]
    w4 = json.load(open(f'{SFM.OUT}/{S}/{S}.sf_model_log.json'))
    say('unknowns', n, 'sources', sources)
    tag_out = '_'.join(s.replace('-', '') for s in sources)
    log = {'subject': S, 'sources': {}}
    for key in sources:
        t0 = time.time(); s = ses[key]; wd = s['wd']
        W, mp = ribbon_weights(geo, U, s['A_epi'], s['A_anat'], f'{wd}/session_mean.nii.gz', od, key)
        tabs, rad = SFM.session_tabs(s['acq'])
        K = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=s['shape'], affine=s['A_anat']), tabs, rad).tocsr()
        vol = lambda nm: np.asarray(nib.load(f'{wd}/{nm}.nii.gz').dataobj, np.float64).ravel()
        y = vol('betaL') + 1j * vol('betaR'); yo = vol('betaL_odd') + 1j * vol('betaR_odd'); ye = vol('betaL_even') + 1j * vol('betaR_even')
        s2 = vol('varL') + vol('varR'); mask = vol('mask') > 0.5
        nr = len(glob.glob(f'{wd}/mc_run-*.par'))                        # runs in the session's GLM (week 4 kept each run's motion file)
        a_odd, a_even = nr / ((nr + 1) // 2), nr / (nr // 2)          # half-mean noise = alpha x full-mean noise
        z, zo, ze = mp(y), mp(yo), mp(ye)
        dil = mp(K @ ones)
        cov = (dil > 0.5) & (np.abs(z) > 0)
        sets = {'covered': cov, 'visual': cov & U['visual']}
        cvs = lambda po, pe: {k: 0.5 * (resid_real(ze, po, v) + resid_real(zo, pe, v)) for k, v in sets.items()}
        C = {'standard': cvs(zo, ze)}
        for fw in FW_GRID:
            g = sm(fw); C[f'smooth{fw:g}'] = cvs(g(zo), g(ze))
        prob = Problem(K, mask, s2, U['Cr'], U['Gr'], U['nroi'])
        sig0 = w4['sessions'][key]['sigma0']                              # week 4's value (full data)
        Bo, Be = prob.rhs(yo), prob.rhs(ye)
        X0 = {'o': None, 'e': None}; its = {}
        for rho, gs in PRIOR_GRID:
            pr = {}
            for h, B, al in (('o', Bo, a_odd), ('e', Be, a_even)):
                prob.set_prior(rho, gs * sig0 / np.sqrt(al)); X, it, _ = prob.solve(B, tol=1e-4, maxit=400, X0=X0[h]); X0[h] = X
                f = X[:, 0] + 1j * X[:, 1]
                pr[h] = mp((K @ f.real.astype(np.float32)) + 1j * (K @ f.imag.astype(np.float32))); its[f'r{rho:g}_s{gs:g}_{h}'] = int(it)
            C[f'model_r{rho:g}_s{gs:g}'] = cvs(pr['o'], pr['e'])
        log['sources'][key] = dict(cv=C, sigma0=sig0, n_runs=nr, alpha=[a_odd, a_even], n={k: int(v.sum()) for k, v in sets.items()},
                                   iterations=its, t_s=round(time.time() - t0))
        del prob
        best = lambda pre, st: min((k for k in C if k.startswith(pre)), key=lambda k: C[k][st])
        say(key, 'split-half best (visual): model', best('model_', 'visual'), 'smoothing', best('smooth', 'visual'),
            round(time.time() - t0), 's')
        json.dump(log, open(f'{od}/{S}.w5_selftune_{tag_out}.json', 'w'), indent=1, default=float)
    say('done')


if __name__ == '__main__':
    main()
