"""Week 3 evaluation: tune on the tuning subjects, score on the test subjects.

Score (primary, unbiased): cross-run prediction in the reference domain. An estimate made from one wedge run is compared,
after the best complex scaling and conjugation (CW responses are conjugates of CCW responses up to a global phase), with the
other run's HCP-standard 1.6 mm surface map: residual energy fraction, averaged over the two directions. Noise in the
estimate and in the reference are independent. Strata: coherent vertices; vertices that are clean at 1.6 mm (standard
leak fraction < 2%) but contaminated at the coarse grid (> 5% or > 10%) - there the 1.6 mm reference is trustworthy and
unmixing is needed; coherent pericalcarine.
Also: opposite-bank pair contrast, averaging-kernel leakage (model vs standard mapping at the same vertices), and the
known-truth simulation.
Usage (xcp_d container): python3 evaluate.py TUNE_DIR[,TUNE_EXT_DIR] TEST_DIR OUTDIR [--tune-only]
"""
import os, sys, json, glob, argparse
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w3lib import *

W2SUB = f'{W_ROOT}/results/week2/subjects'
RUNS, OTHER, W16TAG = stim_runs('wedge'); R0, R1 = list(RUNS); STIM = 'wedge'
WEDGE = RUNS


def set_stimulus(name):
    """Week 13: score the ring runs (or the wedge runs, the default) with the same code."""
    global RUNS, OTHER, W16TAG, R0, R1, STIM, WEDGE
    RUNS, OTHER, W16TAG = stim_runs(name); R0, R1 = list(RUNS); STIM = name; WEDGE = RUNS
FWHMS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)


def gifti_maps(p):
    return {d.meta['Name']: np.asarray(d.data, np.float64) for d in nib.load(p).darrays}


def subject_context(S, idx):
    wk2 = {h: {**gifti_maps(f"{W2SUB}/{S}/{S}.{h}.{STIMULI[STIM]['maps']}.func.gii"), **gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_leakage.func.gii')}
           for h in 'LR'}
    cat = lambda k: np.concatenate([wk2['L'][k], wk2['R'][k]])[idx]
    lab = {}
    for h in 'LR':
        L = nib.load(f'{HCP}/{S}/MNINonLinear/Native/{S}.{h}.aparc.native.label.gii')
        key = [k for k, v in L.labeltable.get_labels_as_dict().items() if 'pericalcarine' in v.lower()][0]
        lab[h] = np.asarray(L.darrays[0].data) == key
    calc = np.concatenate([lab['L'], lab['R']])[idx]
    geo = {h: hemi_geometry(S, h) for h in 'LR'}
    av = analysis_vertices(geo)
    assert np.array_equal(idx, np.concatenate([np.flatnonzero(av['L']), len(geo['L']['white']) + np.flatnonzero(av['R'])]))
    fem = {}
    for h in 'LR':
        Vr, Fr, _ = roi_submesh(geo[h]['mid'], geo[h]['faces'], av[h]); fem[h] = fem_matrices(Vr, Fr)
    return dict(coh=cat('coherence_1.6_all'), lf16=cat('leak_fraction_1.6_gv'), calc=calc, geo=geo,
                Cr={h: fem[h][0] for h in 'LR'}, Gr={h: fem[h][1] for h in 'LR'},
                lf={g: cat(f'leak_fraction_{g}_all') for g in ('2.4', '3.2')},
                delay=json.load(open(f"{W2SUB}/{S}/{S}.{STIMULI[STIM]['summary']}.json"))[STIM]['delay_rad_mod_pi'])


def strata(ctx, g):
    b = ctx['coh'] > COH_THR
    return {'coherent': b, 'clean16_leak5': b & (ctx['lf16'] < 0.02) & (ctx['lf'][g] > 0.05),
            'clean16_leak10': b & (ctx['lf16'] < 0.02) & (ctx['lf'][g] > 0.10), 'calc': b & ctx['calc']}


def cross_run(E, pred, sel):
    """Mean over directions of resid(reference of the other run, conj(prediction from this run))."""
    return float(np.mean([resid(E[f'z16_{OTHER[r]}'][sel].astype(complex), np.conj(pred[r][sel].astype(complex))) for r in RUNS]))


def add_upsmooth(E, ctx):
    """Domain-fair competitor: k-space upsampling -> 1.6 mm HCP mapping (already in up_*) -> surface smoothing."""
    if 'upsmooth' in ctx:
        return
    ctx['upsmooth'] = {}
    for fw in FWHMS:
        sm = surface_smooth(ctx['Cr'], ctx['Gr'], fw)
        for g in ('2.4', '3.2'):
            if f'up_{g}_{R0}' in E.files:
                for r in RUNS:
                    ctx['upsmooth'][(g, fw, r)] = sm(E[f'up_{g}_{r}'].astype(np.complex128)).astype(np.complex64)


def methods(E, g, ctx=None):
    m = {'standard': {r: E[f'z_{g}_{r}'] for r in RUNS}, 'upsample': {r: E[f'up_{g}_{r}'] for r in RUNS}}
    if ctx is not None and 'upsmooth' in ctx:
        for fw in FWHMS:
            if (g, fw, R0) in ctx['upsmooth']:
                m[f'upsmooth{fw:g}'] = {r: ctx['upsmooth'][(g, fw, r)] for r in RUNS}
    for fw in FWHMS:
        if f'smooth{fw:g}_{g}_{R0}' in E.files:
            m[f'smooth{fw:g}'] = {r: E[f'smooth{fw:g}_{g}_{r}'] for r in RUNS}
    for k in E.files:
        if k.startswith(f'render_{g}_') and k.endswith(f'_{R0}'):
            key = k[len('render_'):-len(f'_{R0}')]
            m[f'model_{key[len(g) + 1:]}'] = {r: E[f'render_{key}_{r}'] for r in RUNS}
    return m


def score_subject(E, ctx):
    res = {}
    add_upsmooth(E, ctx)
    for g in ('2.4', '3.2'):
        if f'z_{g}_{R0}' not in E.files:
            continue
        st = strata(ctx, g)
        res[g] = {mname: {s: (cross_run(E, pred, sel) if sel.sum() >= 30 else np.nan) for s, sel in st.items()}
                  for mname, pred in methods(E, g, ctx).items()}
        res[g]['_n'] = {s: int(sel.sum()) for s, sel in st.items()}
    return res


class Merged:
    """Arrays of several estimate files of one subject (e.g. a tuning run and its extension) behind one npz-like view."""
    def __init__(self, files):
        self.d = {}
        for f in files:
            with np.load(f) as z:
                for k in z.files:
                    self.d.setdefault(k, z[k])
        self.files = list(self.d)

    def __getitem__(self, k):
        return self.d[k]


def load(dirpaths):
    """dirpaths: one directory or several, comma-separated; files of the same subject are merged."""
    per = {}
    for dp in dirpaths.split(','):
        for f in sorted(glob.glob(f'{dp}/*/*.week3_estimates.npz')):
            per.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    out = {}
    for S, files in sorted(per.items()):
        E = Merged(files)
        out[S] = (E, subject_context(S, E['idx']))
    return out


def mean_sd(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return (float(v.mean()), float(v.std()), int(len(v))) if len(v) else (np.nan, np.nan, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('tune_dir'); ap.add_argument('test_dir'); ap.add_argument('outdir')
    ap.add_argument('--tune-only', action='store_true')
    ap.add_argument('--stimulus', default='wedge', choices=sorted(STIMULI))
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

    # ---- tuning: always on the wedge data (the settings were chosen there in week 3, whatever the test stimulus)
    set_stimulus('wedge')
    T = load(a.tune_dir)
    tune_scores = {S: score_subject(E, ctx) for S, (E, ctx) in T.items()}
    choice = {}
    for g in ('2.4', '3.2'):
        names = sorted({m for S in tune_scores for m in tune_scores[S].get(g, {}) if not m.startswith('_')})
        if not names:
            continue
        # two pre-specified criteria, both evaluated on the tuning subjects only: overall accuracy (all coherent vertices)
        # and accuracy where unmixing is needed (clean at 1.6 mm, > 10% leakage at this grid)
        avg = {crit: {m: mean_sd([tune_scores[S][g][m][crit] for S in tune_scores if g in tune_scores[S] and m in tune_scores[S][g]])[0]
                      for m in names} for crit in ('coherent', 'clean16_leak10')}
        pick = lambda crit, pfx: min((m for m in names if m.startswith(pfx)), key=lambda m: avg[crit][m])
        choice[g] = dict(model=pick('coherent', 'model_'), smooth=pick('coherent', 'smooth'), upsmooth=pick('coherent', 'upsmooth'),
                         model_leak=pick('clean16_leak10', 'model_'), smooth_leak=pick('clean16_leak10', 'smooth'),
                         upsmooth_leak=pick('clean16_leak10', 'upsmooth'),
                         table={crit: dict(sorted(v.items(), key=lambda kv: kv[1])) for crit, v in avg.items()})
    json.dump(dict(subjects=sorted(T), choice=choice, per_subject=tune_scores), open(f'{a.outdir}/week3_tuning.json', 'w'), indent=1, default=float)
    for g, c in choice.items():
        print(f'tuning {g} mm: overall -> model {c["model"]}, smoothing {c["smooth"]}, upsample+smoothing {c["upsmooth"]}; '
              f'high-leak -> model {c["model_leak"]}, smoothing {c["smooth_leak"]}, upsample+smoothing {c["upsmooth_leak"]}')
        for m, v in c['table']['coherent'].items():
            print(f'   {m:24s} overall {v:.4f}   high-leak {c["table"]["clean16_leak10"][m]:.4f}')
    if a.tune_only:
        return

    # ---- test subjects: chosen model and smoothing vs standard mapping and upsampling (ring or wedge runs, week 13)
    set_stimulus(a.stimulus)
    X = load(a.test_dir)
    res = {'tuning_subjects': sorted(T), 'test_subjects': sorted(X), 'choice': {g: {k: v for k, v in c.items() if k != 'table'} for g, c in choice.items()},
           'scores': {}, 'pairs': {}, 'kernels': {}, 'simulation': {}}
    for S, (E, ctx) in X.items():
        sc = {g: v for g, v in score_subject(E, ctx).items() if g in choice}
        labs = lambda g: (('standard', 'standard'), ('upsample', 'upsample'), ('smoothing', choice[g]['smooth']), ('model', choice[g]['model']),
                          ('upsample_smoothing', choice[g]['upsmooth']),
                          ('smoothing_leak', choice[g]['smooth_leak']), ('model_leak', choice[g]['model_leak']),
                          ('model_psf0.85', choice[g]['model'] + '_psf0.85'), ('model_psf1.15', choice[g]['model'] + '_psf1.15'))
        res['scores'][S] = {g: {lab: sc[g].get(m, {}) for lab, m in labs(g)} for g in sc}
        for g in sc:
            res['scores'][S][g]['_n'] = sc[g]['_n']
        geo = ctx['geo']
        idx = E['idx']; nL = len(geo['L']['white'])
        loc = -np.ones(nL + len(geo['R']['white']), int); loc[idx] = np.arange(len(idx))
        # opposite-bank pairs among coherent vertices (both ends coherent, <= 4 mm apart)
        coh_full = {h: np.zeros(len(geo[h]['white']), bool) for h in 'LR'}
        cf = ctx['coh'] > COH_THR
        for h, off in (('L', 0), ('R', nL)):
            m = (idx >= off) & (idx < off + len(geo[h]['white'])); coh_full[h][idx[m] - off] = cf[m]
        pi_, pj_ = [], []
        for h, off in (('L', 0), ('R', nL)):
            i, j, d = opposite_bank_pairs(geo[h], coh_full[h], dmax=4.0)
            ok = coh_full[h][j]; pi_.append(loc[i[ok] + off]); pj_.append(loc[j[ok] + off])
        pi_, pj_ = np.concatenate(pi_), np.concatenate(pj_)
        calc_pair = ctx['calc'][pi_]
        dis = lambda zc, zw: 1 - np.cos(polar_angle(zc, zw, ctx['delay'])[pi_] - polar_angle(zc, zw, ctx['delay'])[pj_])
        ref = dis(E[f'z16_{R0}'], E[f'z16_{R1}'])
        pr = {'n_pairs': int(len(pi_)), 'n_pairs_calc': int(calc_pair.sum()), 'ref_1.6': float(ref.mean()), 'ref_1.6_calc': float(ref[calc_pair].mean())}
        for g in sc:
            mt = methods(E, g, ctx)
            for lab, m in labs(g):
                if m not in mt:
                    continue
                pred = mt[m]
                d = dis(pred[R0], pred[R1])
                pr[f'{g}_{lab}'] = float(d.mean()); pr[f'{g}_{lab}_calc'] = float(d[calc_pair].mean())
        res['pairs'][S] = pr
        # averaging kernels at sample vertices: leak fraction and spread, model vs standard mapping
        su, fs = {}, {}
        for h in 'LR':
            s_ = geo[h]['sphere']; Rs = np.linalg.norm(s_, axis=1).mean(); su[h] = s_ / np.linalg.norm(s_, axis=1, keepdims=True)
            am, _ = vertex_areas_normals(geo[h]['mid'], geo[h]['faces']); fs[h] = Rs * np.sqrt(am[geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
        mid = np.vstack([geo['L']['mid'], geo['R']['mid']])[idx]
        hemi = np.where(idx < nL, 0, 1); sph = np.vstack([su['L'], su['R']])[idx]; fsc = np.array([fs['L'], fs['R']])
        area = np.concatenate([vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])[0] for h in 'LR'])[idx]
        kr = {}
        for g in choice:
            if f'ak_samples_{g}' not in E.files:
                continue
            rows = []
            for k, i in enumerate(E[f'ak_samples_{g}']):
                far = (hemi != hemi[i]) | (fsc[hemi[i]] * np.arccos(np.clip(sph @ sph[i], -1, 1)) > GEO_THR)
                dd = np.linalg.norm(mid - mid[i], axis=1)
                row = {'vertex': int(i), 'std_leakfrac_week2': float(ctx['lf'][g][i])}
                for lab in ('model', 'std'):
                    akr = E[f'ak_{lab}_{g}'][k].astype(np.float64)
                    tot = akr.sum(); near = ~far
                    row[f'{lab}_leak_fraction'] = float(akr[far].sum() / tot) if abs(tot) > 0 else np.nan
                    row[f'{lab}_leak_abs_fraction'] = float(np.abs(akr[far]).sum() / np.abs(akr).sum())
                    w = np.abs(akr) * near
                    row[f'{lab}_spread_mm'] = float(np.sqrt((w * dd ** 2).sum() / w.sum()))
                    row[f'{lab}_peak_share'] = float(akr[i] / tot) if abs(tot) > 0 else np.nan
                rows.append(row)
            kr[g] = rows
        res['kernels'][S] = kr
        # known-truth simulation (same run; truth rendered noise-free through the 1.6 mm model)
        sim = {}
        for g in choice:
            if f'sim_z_{g}_{R0}' not in E.files:
                continue
            st = strata(ctx, g); sim[g] = {}
            smu = surface_smooth(ctx['Cr'], ctx['Gr'], float(choice[g]['upsmooth'][len('upsmooth'):]))
            pred = {'standard': {r: E[f'sim_z_{g}_{r}'] for r in RUNS}, 'upsample': {r: E[f'sim_up_{g}_{r}'] for r in RUNS},
                    'smoothing': {r: E[f"sim_{choice[g]['smooth']}_{g}_{r}"] for r in RUNS}, 'model': {r: E[f'sim_render_{g}_{r}'] for r in RUNS},
                    'upsample_smoothing': {r: smu(E[f'sim_up_{g}_{r}'].astype(np.complex128)) for r in RUNS}}
            for lab, pr_ in pred.items():
                sim[g][lab] = {s_: float(np.mean([resid(E[f'sim_truth_render_{g}_{r}'][sel].astype(complex), pr_[r][sel].astype(complex))
                                                   for r in RUNS])) if sel.sum() >= 30 else np.nan for s_, sel in st.items()}
            sim[g]['model_field_vs_truth_field'] = float(np.mean([resid(E[f'sim_truth_f_{g}_{r}'][st['coherent']].astype(complex),
                                                                       E[f'sim_f_{g}_{r}'][st['coherent']].astype(complex)) for r in RUNS]))
        res['simulation'][S] = sim
        print(S, 'scored', {g: {lab: round(v.get('coherent', np.nan), 4) for lab, v in res['scores'][S][g].items() if lab != '_n'} for g in res['scores'][S]}, flush=True)
    json.dump(res, open(f'{a.outdir}/week3_test_scores.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
