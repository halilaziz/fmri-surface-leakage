"""Week 15, analysis D: the operator's far part tested on real coarser acquisitions (StudyForrest 7T sessions), one subject.

For every ordered pair of 7T sessions fine -> coarse, the fine session's field estimate (standard map / dilution, as in week 2)
is pushed through the coarse session's operator A_c = W_c K_c, split per vertex into near and far columns (other hemisphere or
> GEO_THR mm on the area-scaled FreeSurfer sphere), and the predictions — naive (the fine map), near only, near + far, near
renormalised to the full row mass, near + far shuffled across the vertices of the stratum — are scored against the coarse
session's standard map (residual energy fraction after the best real scaling per hemifield map), by stratum. Geometry,
sessions, registrations, weights and operators exactly as in week 4 (sf_model.py).
Usage (xcp_d container): python3 sf_operator_test.py SUBJECT OUTDIR [--threads 8]
       (local)           python3 sf_operator_test.py summary IN_DIR OUT_PREFIX
"""
import os, sys, json, glob, time, argparse
import numpy as np, scipy.sparse as sp, nibabel as nib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '04_misregistration_studyforrest')); sys.path.insert(0, os.path.join(HERE, '..', '03_hcp_emulation'))

PAIRS = [('r08', 'r14'), ('r08', 'r20'), ('r08', 'r30'), ('r14', 'r20'), ('r14', 'r30'), ('r20', 'r30')]
PRED = ['naive', 'near', 'full', 'renorm', 'shuffled']
LABEL = {'naive': 'fine map itself', 'near': 'near part only', 'full': 'near + far', 'renorm': 'near, renormalised', 'shuffled': 'near + shuffled far'}
SETS = ['covered', 'visual', 'visual_lf_lt2', 'visual_lf_gt5', 'visual_lf_gt10', 'visual_lf_gt20', 'covered_lf_gt10', 'calc']
SLAB = {'covered': 'all covered', 'visual': 'visual cortex', 'visual_lf_lt2': 'visual, far leak < 2%', 'visual_lf_gt5': 'visual, > 5%',
        'visual_lf_gt10': 'visual, > 10%', 'visual_lf_gt20': 'visual, > 20%', 'covered_lf_gt10': 'all covered, > 10%', 'calc': 'pericalcarine'}


def subject(S, outdir, threads):
    from sf_model import read_lta, write_surf, hom, acq_params, session_tabs, resid_real, SF, PREP, RES_MM, VISUAL, OUT as W4OUT
    from w3lib import (analysis_vertices, roi_submesh, fem_matrices, subject_points, voxel_sizes, build_operator, wb_ribbon_weights,
                       parse_weights_text, vertex_areas_normals, GEO_THR)
    os.makedirs(outdir, exist_ok=True); od = f'{outdir}/{S}'; os.makedirs(od, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, f'{time.time() - T0:7.0f}s', *x, flush=True)
    FS = f'{SF}/freesurfer/{S}'
    orig = nib.load(f'{FS}/mri/orig.mgz'); v2r = orig.affine; tkr2scan = v2r @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    geo = {}
    for h, fh in (('L', 'lh'), ('R', 'rh')):
        W, F = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.white'); Pv, _ = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.pial')
        sph, _ = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.sphere')
        lab, _, names = nib.freesurfer.read_annot(f'{FS}/label/{fh}.aparc.annot')
        nm = np.array([names[l] if l >= 0 else b'unknown' for l in lab])
        W, Pv = hom(tkr2scan, W), hom(tkr2scan, Pv)
        geo[h] = dict(white=W, pial=Pv, mid=0.5 * (W + Pv), faces=F.astype(np.int64), sphere=sph,
                      roi=~np.isin(nm, [b'unknown', b'corpuscallosum']), visual=np.isin(nm, VISUAL), calc=nm == b'pericalcarine')
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    ses = {}
    for key in sorted(os.listdir(f'{PREP}/{S}')):
        wd = f'{PREP}/{S}/{key}'
        if not os.path.exists(f'{wd}/betaR_even.nii.gz') or not key.startswith('7T'):
            continue
        field, res = key.split('-')
        mimg = nib.load(f'{wd}/session_mean.nii.gz'); typ, M = read_lta(f'{wd}/bbr.lta')
        A_anat = v2r @ M if typ == 0 else M @ mimg.affine
        vol = lambda n: np.asarray(nib.load(f'{wd}/{n}.nii.gz').dataobj, np.float64).ravel()
        ses[key] = dict(field=field, res=res, shape=mimg.shape[:3], A_anat=A_anat, A_epi=mimg.affine, wd=wd,
                        y=vol('betaL') + 1j * vol('betaR'), mask=vol('mask') > 0.5)
        try:                                   # the raw sidecars were deleted with the raw volumes; the week-4 log kept the parameters
            ses[key]['acq'] = acq_params(field, res, S)
        except (IndexError, FileNotFoundError):
            ses[key]['acq'] = json.load(open(f'{W4OUT}/{S}/{S}.sf_model_log.json'))['sessions'][key]['acq']
    say('sessions', list(ses))
    inside = {h: np.zeros(nv[h], bool) for h in 'LR'}
    for s in ses.values():
        Ai = np.linalg.inv(s['A_anat']); vs = voxel_sizes(dict(affine=s['A_anat'])); mg = 5.0 / vs
        for h in 'LR':
            q = hom(Ai, geo[h]['mid']); inside[h] |= ((q > -0.5 - mg) & (q < np.array(s['shape']) - 0.5 + mg)).all(1)
    for h in 'LR':
        geo[h]['roi'] = geo[h]['roi'] & inside[h]
    av = analysis_vertices(geo)
    idx = np.concatenate([np.flatnonzero(av['L']), nv['L'] + np.flatnonzero(av['R'])]); n = len(idx)
    nroi = {h: int(av[h].sum()) for h in 'LR'}
    loc = -np.ones(nv['L'] + nv['R'], int); loc[idx] = np.arange(n)
    P, V, C, Z, _ = subject_points(geo)
    keep = loc[C] >= 0; P, V, Cc = P[keep], V[keep], loc[C[keep]]
    cat = lambda d: np.concatenate([d['L'], d['R']])[idx]
    visual = cat({h: geo[h]['visual'] for h in 'LR'}); calc = cat({h: geo[h]['calc'] for h in 'LR'})
    # sphere geometry of the unknowns for the far criterion
    su, fsc = {}, {}
    for h in 'LR':
        s_ = geo[h]['sphere']; Rs = np.linalg.norm(s_, axis=1).mean(); su[h] = s_ / np.linalg.norm(s_, axis=1, keepdims=True)
        am_, _ = vertex_areas_normals(geo[h]['mid'], geo[h]['faces']); fsc[h] = Rs * np.sqrt(am_[geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
    sph = np.vstack([su['L'], su['R']])[idx]; hemi = np.where(idx < nv['L'], 0, 1); fscale = np.array([fsc['L'], fsc['R']])
    say('unknowns', n, nroi)
    # per session: weights, operator, standard map, dilution; and the near / far split of A = W K on the unknowns
    for key, s in ses.items():
        t0 = time.time(); a2e = s['A_epi'] @ np.linalg.inv(s['A_anat']); W = {}
        for h in 'LR':
            for nm in ('white', 'pial', 'mid'):
                write_surf(f'{od}/{key}.{h}.{nm}.surf.gii', hom(a2e, geo[h][nm]), geo[h]['faces'], h)
            txt = f'{od}/{key}.{h}.w.txt'
            wb_ribbon_weights(f"{s['wd']}/session_mean.nii.gz", f'{od}/{key}.{h}.mid.surf.gii', f'{od}/{key}.{h}.white.surf.gii', f'{od}/{key}.{h}.pial.surf.gii', txt)
            W[h] = parse_weights_text(txt, nv[h], s['shape']).tocsr(); os.remove(txt)
            for nm in ('white', 'pial', 'mid'):
                os.remove(f'{od}/{key}.{h}.{nm}.surf.gii')
        s['map'] = lambda y, W=W: cat({h: W[h] @ y for h in 'LR'})
        tabs, rad = session_tabs(s['acq'])
        K = build_operator(P, V, Cc, n, dict(shape=s['shape'], affine=s['A_anat']), tabs, rad).tocsr()
        s['z'] = s['map'](s['y']); s['dil'] = s['map'](K @ np.ones(n, np.float32)); s['covered'] = (s['dil'] > 0.5) & (np.abs(s['z']) > 0)
        # rows of A for the unknowns, split near / far: store the row structure once per session (operator freed after)
        rows_h = {h: np.flatnonzero(av[h]) for h in 'LR'}
        An, Af = [], []
        for h, off in (('L', 0), ('R', nroi['L'])):
            Wm = W[h][rows_h[h]]
            for c0 in range(0, len(rows_h[h]), 2000):
                A = (Wm[c0:c0 + 2000] @ K).tocoo()
                r = A.row + off + c0; j = A.col; val = A.data.astype(np.float32)
                other = hemi[j] != hemi[r]
                cosang = np.einsum('nd,nd->n', sph[r], sph[j])
                far = other | (fscale[hemi[r]] * np.arccos(np.clip(cosang, -1, 1)) > GEO_THR)
                An.append(sp.coo_matrix((val[~far], (r[~far], j[~far])), shape=(n, n))); Af.append(sp.coo_matrix((val[far], (r[far], j[far])), shape=(n, n)))
        s['A_near'] = sum(An[1:], An[0]).tocsr(); s['A_far'] = sum(Af[1:], Af[0]).tocsr(); del An, Af, K
        own = np.asarray(s['A_near'].sum(1)).ravel(); leak = np.asarray(s['A_far'].sum(1)).ravel(); tot = own + leak
        s['lf'] = np.where(np.abs(tot) > 1e-6, leak / np.where(np.abs(tot) > 1e-6, tot, 1.0), 0.0); s['own'] = own; s['tot'] = tot
        say(key, 'weights, operator, near/far split', round(time.time() - t0), 's; covered', int(s['covered'].sum()),
            'mean far leak (visual, covered) %.2f%%' % (100 * s['lf'][visual & s['covered']].mean()))
    # pairs
    rng = np.random.default_rng(int(S.split('-')[1]) + 15)
    cmv = lambda M, x: (M @ np.ascontiguousarray(x.real, np.float32)) + 1j * (M @ np.ascontiguousarray(x.imag, np.float32))
    log = {'subject': S, 'sessions': {k: dict(covered=int(s['covered'].sum()), mean_lf_visual=float(s['lf'][visual & s['covered']].mean())) for k, s in ses.items()}, 'pairs': {}}
    for rf, rc in PAIRS:
        kf, kc = f'7T-{rf}', f'7T-{rc}'
        if kf not in ses or kc not in ses:
            continue
        f_, c_ = ses[kf], ses[kc]
        fhat = np.where(f_['covered'], f_['z'] / np.maximum(f_['dil'], 0.3), 0)
        pn = cmv(c_['A_near'], fhat); pf = cmv(c_['A_far'], fhat)
        scale = np.where(np.abs(c_['own']) > 1e-6, c_['tot'] / np.where(np.abs(c_['own']) > 1e-6, c_['own'], 1.0), 1.0)
        base = f_['covered'] & c_['covered']; lf = c_['lf']
        sets = {'covered': base, 'visual': base & visual, 'visual_lf_lt2': base & visual & (lf < 0.02), 'visual_lf_gt5': base & visual & (lf > 0.05),
                'visual_lf_gt10': base & visual & (lf > 0.10), 'visual_lf_gt20': base & visual & (lf > 0.20), 'covered_lf_gt10': base & (lf > 0.10), 'calc': base & calc}
        sc = {'n': {k: int(v.sum()) for k, v in sets.items()}}
        for k, sel in sets.items():
            ii = np.flatnonzero(sel)
            if len(ii) < 50:
                continue
            perm = rng.permutation(ii)
            preds = {'naive': f_['z'][ii], 'near': pn[ii], 'full': pn[ii] + pf[ii], 'renorm': pn[ii] * scale[ii], 'shuffled': pn[ii] + pf[perm]}
            zc = c_['z'][ii]; sc[k] = {m: resid_real(zc, p, np.ones(len(ii), bool)) for m, p in preds.items()}
        log['pairs'][f'{rf}->{rc}'] = sc
        say(f'{rf}->{rc}', {k: {m: round(v, 3) for m, v in sc[k].items()} for k in ('visual', 'visual_lf_gt10') if k in sc})
    log['t_total_s'] = round(time.time() - T0)
    json.dump(log, open(f'{outdir}/{S}.sf_operator_test.json', 'w'), indent=1, default=float)
    say('done')


def summary(indir, prefix):
    from scipy.stats import wilcoxon
    J = [json.load(open(f)) for f in sorted(glob.glob(f'{indir}/*.sf_operator_test.json'))]
    fm = lambda x: f'{x:+.4f}'.replace('-', '−'); fp = lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'

    def stats(v, direction):
        v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
        if len(v) < 2:
            return dict(n=int(len(v)))
        b = np.random.default_rng(0).choice(v, size=(10000, len(v)), replace=True).mean(1)
        d = dict(n=int(len(v)), mean=float(v.mean()), n_dir=int((v < 0).sum() if direction == 'less' else (v > 0).sum()),
                 ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
        if np.any(v != 0):
            d['p'] = float(wilcoxon(v, alternative=direction).pvalue)
        return d
    res = {'subjects': [j['subject'] for j in J], 'pairs': {}}
    L = [f'# Week 15 D: the operator\'s far part on real coarser StudyForrest sessions ({len(J)} subjects; plan `SF_OPERATOR_PLAN.md`)', '',
         'Residual energy fraction of the coarse session\'s standard map predicted from the fine session\'s field estimate through the coarse',
         'session\'s operator (mean over subjects); tests: D1 full < near, D2 shuffled > full, D3 renorm > full (one-sided Wilcoxon).', '']
    pair_keys = sorted({p for j in J for p in j['pairs']}, key=lambda p: PAIRS.index(tuple(p.split('->'))))
    for pk in pair_keys + ['pooled over pairs']:
        L += [f'## {pk}', '', '| stratum | n subj. | n vertices (mean) | ' + ' | '.join(LABEL[m] for m in PRED) + ' | D1 full − near | D2 shuffled − full | D3 renorm − full |',
              '|---|---|---|' + '---|' * (len(PRED) + 3)]
        res['pairs'][pk] = {}
        for st in SETS:
            rows = []
            for j in J:
                if pk == 'pooled over pairs':
                    vals = [j['pairs'][p][st] for p in j['pairs'] if st in j['pairs'][p]]
                    nv_ = [j['pairs'][p]['n'][st] for p in j['pairs'] if st in j['pairs'][p]]
                    if vals:
                        rows.append(({m: float(np.mean([v[m] for v in vals])) for m in PRED}, float(np.mean(nv_))))
                elif pk in j['pairs'] and st in j['pairs'][pk]:
                    rows.append((j['pairs'][pk][st], j['pairs'][pk]['n'][st]))
            if len(rows) < 2:
                continue
            means = {m: float(np.mean([r[0][m] for r in rows])) for m in PRED}
            d1 = stats([r[0]['full'] - r[0]['near'] for r in rows], 'less'); d2 = stats([r[0]['shuffled'] - r[0]['full'] for r in rows], 'greater')
            d3 = stats([r[0]['renorm'] - r[0]['full'] for r in rows], 'greater')
            res['pairs'][pk][st] = dict(n=len(rows), means=means, D1=d1, D2=d2, D3=d3)
            cell = lambda d: f"{fm(d['mean'])} ({d['n_dir']}/{d['n']}, p {fp(d['p'])})" if 'p' in d else '–'
            L.append(f"| {SLAB[st]} | {len(rows)} | {np.mean([r[1] for r in rows]):.0f} | " + ' | '.join(f'{means[m]:.3f}' for m in PRED) + f' | {cell(d1)} | {cell(d2)} | {cell(d3)} |')
        L.append('')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        summary(sys.argv[2], sys.argv[3])
    else:
        ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('outdir'); ap.add_argument('--threads', type=int, default=8)
        a = ap.parse_args(); subject(a.subject, a.outdir, a.threads)
