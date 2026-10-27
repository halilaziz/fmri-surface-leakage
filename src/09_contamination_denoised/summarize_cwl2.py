"""Week 9 (STATS_PLAN_w9.md): as week-8 summarize_cwl.py, for the denoised response estimates (S2, S4, S8: smoothed source
native map; Q: the model's own estimate). Same settings, bins, vertex sets and contrast D; the far leak is kept for reference.
(Week-8 docstring follows.)
Week 8, plan A: per-subject dose-response of the model's gain on the contrast-weighted leak (STATS_PLAN.md), per dataset.
For every subject: gain Delta = residual(model) - residual(best competitor) per CWL bin (vertices clean at the native grid),
the primary contrast D = Delta(highest bin with >= 30 vertices) - Delta(0-2% bin), the same with far-leak bins (secondary a)
and with CWL-P (secondary b); other-hemisphere share bins where saved (exploratory, added after the plan), and a CWL-P-gated estimator with the threshold chosen leave-one-subject-out (secondary c).
Settings as in the original analyses (HCP week-3 choices; StudyForrest and ds002702 leave-one-subject-out).
Usage: python3 summarize_cwl2.py kay EMUL_DIR CWL_DIR OUT_JSON          (hebb, xcp_d container)
       python3 summarize_cwl2.py sf  GATE_DIR CWL_DIR OUT_JSON          (TRUBA)
       python3 summarize_cwl2.py hcp CWL_DIR OUT_JSON                   (TRUBA)
"""
import os, sys, re, json, glob
import numpy as np

BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, np.inf)]
VARS = ('cwlS2', 'cwlS4', 'cwlS8', 'cwlQ')
BLAB = ['0-2%', '2-5%', '5-10%', '10-20%', '>20%']
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, np.inf]
NMIN = 30


def resid_cols(T, P, sel):
    """Residual energy fraction pooled over real maps (columns), each with its own best scaling."""
    num = den = 0.0
    for j in range(T.shape[1]):
        z, p = T[sel, j], P[sel, j]; c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


def contrast(dl):
    """dl: list of per-bin Delta (None where the bin has < NMIN vertices). D = highest available bin - lowest bin."""
    ok = [i for i, v in enumerate(dl) if v is not None]
    return (dl[ok[-1]] - dl[0]) if (ok and ok[0] == 0 and ok[-1] > 0) else None


def loso_pick(M):
    return [int(np.argmin(np.delete(M, i, 0).mean(0))) for i in range(len(M))]


# --------------------------------------------------------------------------------------------------------- ds002702 / StudyForrest
def within_session(Z, C, subs, comps, models, score, sets_for, gate_pred, tag):
    """Common part for the two within-session datasets. score(S, key, d, sel) -> residual; sets_for(S, d) -> dict of vertex sets."""
    Mc = np.array([[np.mean([score(S, k, d, sets_for(S, d)['visual']) for d in ('AB', 'BA')]) for k in comps] for S in subs])
    Mm = np.array([[np.mean([score(S, k, d, sets_for(S, d)['visual']) for d in ('AB', 'BA')]) for k in models] for S in subs])
    ci, mi = loso_pick(Mc), loso_pick(Mm)
    res = {}
    for i, S in enumerate(subs):
        ck, mk = comps[ci[i]], models[mi[i]]
        r = dict(competitor=ck, model=mk, bins={})
        for var in VARS + ('far',):
            dl, nl = [], []
            for lo, hi in BINS:
                sels = []
                for d in ('AB', 'BA'):
                    s_ = sets_for(S, d)
                    v = Z[S]['far_c'] if var == 'far' else (Z[S]['xh_c'] if var == 'xh' else C[S][f'{var}_{d}'])
                    sels.append(s_['clean'] & (v >= lo) & (v < hi))
                if min(x.sum() for x in sels) >= NMIN:
                    dl.append(float(np.mean([score(S, mk, d, x) - score(S, ck, d, x) for d, x in zip(('AB', 'BA'), sels)])))
                else:
                    dl.append(None)
                nl.append(int(np.mean([x.sum() for x in sels])))
            r['bins'][var] = dict(delta=dl, n=nl, D=contrast(dl))
        r['competitor_everywhere'] = float(np.mean([score(S, ck, d, sets_for(S, d)['visual']) for d in ('AB', 'BA')]))
        r['model_everywhere'] = float(np.mean([score(S, mk, d, sets_for(S, d)['visual']) for d in ('AB', 'BA')]))
        res[S] = r
    return res


def run_kay(ED, CD):
    out = {}
    cells = {}
    for p in sorted(glob.glob(f'{ED}/sub-*/*.kay_emul.npz')):
        fac = re.search(r'\.x([0-9.]+?)\.kay_emul\.npz$', p).group(1); S = os.path.basename(p).split('.')[0]
        cells.setdefault(fac, {})[S] = p
    for fac, SS in sorted(cells.items(), key=lambda kv: float(kv[0])):
        subs = sorted(SS); Z = {S: np.load(SS[S]) for S in subs}
        C = {S: np.load(f'{CD}/{S}/{S}.ring7T.x{fac}.cwl2.npz') for S in subs}
        models = sorted(k[3:] for k in Z[subs[0]].files if k.startswith('AB_') and k[3:].startswith(('corr_', 'white_')))
        comps = ['upsample', 'upsmooth2', 'upsmooth4', 'smooth4', 'smooth8']
        sets_for = lambda S, d: dict(visual=Z[S][f'{d}_base'] & Z[S]['visual'],
                                     clean=Z[S][f'{d}_base'] & Z[S]['visual'] & (Z[S]['far_n'] < 0.02))
        score = lambda S, k, d, sel: resid_cols(Z[S][f'{d}_target'], Z[S][f'{d}_{k}'], sel)

        def gate_pred(S, mk, ck, d, tau):
            s_ = sets_for(S, d)['visual']; m, c = Z[S][f'{d}_{mk}'].astype(float), Z[S][f'{d}_{ck}'].astype(float)
            k = np.array([(m[s_, q] @ c[s_, q]) / max(m[s_, q] @ m[s_, q], 1e-30) for q in range(m.shape[1])])
            return resid_cols(Z[S][f'{d}_target'], np.where((C[S][f'cwlP_{d}'] > tau)[:, None], m * k, c), s_)
        out[fac] = within_session(Z, C, subs, comps, models, score, sets_for, gate_pred, 'kay')
        print('ds002702 x' + fac, {S: out[fac][S]['bins']['cwlS4']['D'] for S in subs}, flush=True)
    return out


def resid_complex(target, pred, sel):
    """weeks 5-7 resid_real: real and imaginary parts, each with its own best real scaling, pooled."""
    num = den = 0.0
    for part in (np.real, np.imag):
        z, p = part(target[sel]), part(pred[sel]); c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


def run_sf(GD, CD):
    out = {}
    cells = {}
    for p in sorted(glob.glob(f'{GD}/sub-*/*.w7_gate.npz')):
        b = os.path.basename(p).split('.'); S, key = b[0], b[1]
        fac = re.search(r'\.x([0-9.]+?)\.w7_gate\.npz$', p).group(1)
        cells.setdefault(f'{key}_x{fac}', {})[S] = (p, f'{CD}/{S}/{S}.{key}.x{fac}.cwl2.npz')
    for cell, SS in sorted(cells.items()):
        subs = sorted(S for S in SS if os.path.exists(SS[S][1]))
        Z = {S: np.load(SS[S][0]) for S in subs}; C = {S: np.load(SS[S][1]) for S in subs}
        models = sorted(k[3:] for k in Z[subs[0]].files if k.startswith('AB_') and k[3:].startswith(('corr_', 'white_')))
        comps = ['upsample', 'upsmooth2']
        sets_for = lambda S, d: dict(visual=Z[S][f'{d}_base'] & Z[S]['visual'],
                                     clean=Z[S][f'{d}_base'] & Z[S]['visual'] & (Z[S]['far_n'] < 0.02))
        score = lambda S, k, d, sel: resid_complex(Z[S][f'{d}_target'], Z[S][f'{d}_{k}'], sel)

        def gate_pred(S, mk, ck, d, tau):
            s_ = sets_for(S, d)['visual']; m, c = Z[S][f'{d}_{mk}'].astype(complex), Z[S][f'{d}_{ck}'].astype(complex)
            k = np.vdot(m[s_], c[s_]) / max(np.vdot(m[s_], m[s_]).real, 1e-30)
            return resid_complex(Z[S][f'{d}_target'], np.where(C[S][f'cwlP_{d}'] > tau, k * m, c), s_)
        out[cell] = within_session(Z, C, subs, comps, models, score, sets_for, gate_pred, 'sf')
        print('StudyForrest', cell, {S: out[cell][S]['bins']['cwlS4']['D'] for S in subs}, flush=True)
    return out


def run_hcp(CD, est_dir=None, subjects_file=None, stimulus='wedge'):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
    from evaluate import Merged, subject_context, cross_run, methods, add_upsmooth, set_stimulus
    from w3lib import W_ROOT, COH_THR, stim_runs
    set_stimulus(stimulus or 'wedge'); RUNS = stim_runs(stimulus or 'wedge')[0]; R0, R1 = list(RUNS)      # week 13: batches, rings
    R3 = f'{W_ROOT}/results/week3'
    CHOSEN = {'2.4': 'model_r12_s0.125', '3.2': 'model_r24_s0.25'}
    est = est_dir or f'{R3}/test_all'
    subs = ([l.strip() for l in open(subjects_file) if l.strip()] if subjects_file else
            json.load(open(f'{R3}/_eval/week3_test_scores.json'))['test_subjects'])
    res = {g: {} for g in CHOSEN}
    for S in subs:
        E = Merged([f'{est}/{S}/{S}.week3_estimates.npz']); Cw = np.load(f'{CD}/{S}.cwl2_hcp.npz')
        assert np.array_equal(Cw['idx'], E['idx'])
        ctx = subject_context(S, E['idx']); add_upsmooth(E, ctx)
        coh = ctx['coh'] > COH_THR; clean = coh & (ctx['lf16'] < 0.02)
        for g in CHOSEN:
            mt = methods(E, g, ctx); M, Cm = mt[CHOSEN[g]], mt['upsmooth8']
            var = {v: 0.5 * (Cw[f'{v}_{g}_{R0}'] + Cw[f'{v}_{g}_{R1}']) for v in VARS}; var['far'] = ctx['lf'][g]
            r = dict(competitor='upsmooth8', model=CHOSEN[g], bins={})
            for v, x in var.items():
                dl, nl = [], []
                for lo, hi in BINS:
                    sel = clean & (x >= lo) & (x < hi); nl.append(int(sel.sum()))
                    dl.append(float(cross_run(E, M, sel) - cross_run(E, Cm, sel)) if sel.sum() >= NMIN else None)
                r['bins'][v] = dict(delta=dl, n=nl, D=contrast(dl))
            r['competitor_everywhere'] = float(cross_run(E, Cm, coh)); r['model_everywhere'] = float(cross_run(E, M, coh))
            res[g][S] = r
        print('HCP', S, {g: res[g][S]['bins']['cwlS4']['D'] for g in CHOSEN}, flush=True)
    return res


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'kay':
        R = run_kay(sys.argv[2], sys.argv[3]); oj = sys.argv[4]
    elif which == 'sf':
        R = run_sf(sys.argv[2], sys.argv[3]); oj = sys.argv[4]
    else:
        R = run_hcp(sys.argv[2], *sys.argv[4:7]); oj = sys.argv[3]       # [EST_DIR [SUBJECTS_FILE [STIMULUS]]]
    json.dump(dict(dataset=which, bins=BLAB, taus=[t if np.isfinite(t) else 'never' for t in TAUS], cells=R),
              open(oj, 'w'), indent=1, default=float)
