"""Week 2, per subject (HCP 7T retinotopy geometry, MNINonLinear space):
  1. Workbench ribbon-constrained weights on the native 1.6 mm grid (with and without HCP's per-run goodvoxels) and on the
     emulated 2.4 / 3.2 mm grids; 'standard mapping' W = row-normalised weights.
  2. Data: rotating-wedge runs (RETCCW, RETCW) at 1.6 mm and emulated 2.4 / 3.2 mm (k-space crop x window), standard
     mapping, phase-encoded analysis -> complex response at the stimulus frequency, coherence, polar angle, tSNR.
  3. Effective-PSF forward operator K (column-mean depth component b0; both hemispheres jointly, so midline mixing counts)
     and the real-geometry behaviour of standard mapping, A = W K: per vertex, the readout of a unit field on geodesically
     near cortex ('own') and on far cortex or the other hemisphere ('leak'); own + leak = partial-volume dilution.
     Compared with the week-1 proxy (planar-bank lookup indexed by distance to the opposite bank).
  4. Cross-resolution prediction (test of the operator on real data): the 1.6 mm wedge responses, divided by the 1.6 mm
     dilution, are pushed through A_2.4 / A_3.2 split into near and far parts. If the operator's cross-bank mixing is right,
     adding the far part improves the prediction of the emulated 2.4 / 3.2 mm maps where the model predicts leakage, and a
     far part shuffled across vertices does not.
  5. Geometry check: triangle-prism box weights (week-1 proxy of the ribbon method) vs Workbench weights.
Usage (xcp_d container): python3 run_subject.py SUBJECT OUTDIR WEEK1_MAPSDIR [--threads N] [--smoke]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2lib import *

GEO_THR = 15.0                       # mm on the registration sphere; same 'far cortex' criterion as week 1
WEDGE = {'ccw': 0, 'cw': 300}        # frame offsets of RETCCW_AP and RETCW_PA in the concatenated series
BLANK, NWIN, CYC = 22, 256, 8        # 22 s blank, then 8 cycles x 32 s (TR 1 s); stimulus = DFT bin 8 of the window
GV_RUN = {'gv': 'RETCCW_7T_AP', 'gvcw': 'RETCW_7T_PA'}     # run whose HCP goodvoxels mask a weight set uses
DBINS = [0, 2.25, 2.75, 3.25, 3.75, 4.5, 6.0, np.inf]   # week-1 d_opp bins (centres ~2, 2.5, 3, 3.5, 4, 5, >6 mm)
COMBOS = [('1.6', 'gv'), ('1.6', 'all'), ('2.4', 'all'), ('3.2', 'all')]
COH_THR = 0.3                        # 'coherent' vertices: mean wedge coherence > 0.3 at 1.6 mm (no mask)


def save_maps(path, maps, hemi):
    das = []
    for name, x in maps.items():
        da = nib.gifti.GiftiDataArray(np.asarray(x, np.float32), intent='NIFTI_INTENT_NONE', datatype='NIFTI_TYPE_FLOAT32')
        da.meta = nib.gifti.GiftiMetaData({'Name': name}); das.append(da)
    st = {'L': 'CortexLeft', 'R': 'CortexRight'}[hemi]
    nib.save(nib.gifti.GiftiImage(darrays=das, meta=nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary': st})), path)


def wmean(x, w):
    return float((x * w).sum() / max(w.sum(), 1e-12))


def cmean(theta, w):
    return (w * np.exp(1j * theta)).sum() / max(w.sum(), 1e-12)


def resid(z, p):
    """Residual energy fraction of z after the best complex scaling of predictor p."""
    c = np.vdot(p, z) / max(np.vdot(p, p).real, 1e-30)
    return float(np.sum(np.abs(z - c * p) ** 2) / np.sum(np.abs(z) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subject'); ap.add_argument('outdir'); ap.add_argument('w1maps'); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--smoke', action='store_true', help='occipital cortex only (MNI y < -70): quick end-to-end test')
    a = ap.parse_args(); S = a.subject; nth = a.threads
    sd = os.path.join(a.outdir, S); os.makedirs(sd, exist_ok=True)
    T0 = time.time(); log = {'subject': S}
    R = f'{HCP}/{S}/MNINonLinear/Results'; N = f'{HCP}/{S}/MNINonLinear/Native'

    sbref, grids = subject_grids(S)
    templ = {'1.6': sbref}
    sb = np.asarray(nib.load(sbref).dataobj, dtype=np.float64)
    for g in ('2.4', '3.2'):
        templ[g] = f'{sd}/grid_{g}mm_template.nii.gz'
        if not os.path.exists(templ[g]):
            save_nifti(emulate(sb, grids[g]['shape']), grids[g], templ[g])
    log['grids'] = {g: dict(shape=v['shape'], voxel_mm=voxel_sizes(v).round(4).tolist()) for g, v in grids.items()}

    geo = {h: hemi_geometry(S, h) for h in 'LR'}
    if a.smoke:
        for h in 'LR':
            geo[h]['roi'] = geo[h]['roi'] & (geo[h]['mid'][:, 1] < -70)
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    scales = subject_scales(S, geo, a.w1maps)
    log['mni_over_acpc_scales'] = scales.round(4).tolist()
    area = {h: vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])[0] for h in 'LR'}
    calc = {}
    for h in 'LR':
        lab = nib.load(f'{N}/{S}.{h}.aparc.native.label.gii')
        key = [k for k, v in lab.labeltable.get_labels_as_dict().items() if 'pericalcarine' in v.lower()][0]
        calc[h] = (np.asarray(lab.darrays[0].data) == key) & geo[h]['roi']
    w1 = {}
    for h in 'LR':
        g1 = nib.load(f'{a.w1maps}/{S}/{S}.{h}.week1_maps.func.gii')
        w1[h] = {d.meta.get('Name'): np.asarray(d.data, np.float64) for d in g1.darrays}

    # ---- 1. Workbench ribbon weights. HCP masks each run with its own goodvoxels (the intersection over the six runs keeps
    #         only ~40% of ribbon voxels): 'gv' = RETCCW mask (leakage maps, CCW run), 'gvcw' = RETCW mask (CW run).
    t0 = time.time(); Wts = {}
    for g, tag in COMBOS + [('1.6', 'gvcw')]:
        for h in 'LR':
            f = f'{sd}/hcpW_{g}_{h}_{tag}.npz'
            if not os.path.exists(f):
                txt = f'{sd}/w_{g}_{h}_{tag}.txt'
                gv = f'{R}/tfMRI_{GV_RUN[tag]}/RibbonVolumeToSurfaceMapping/goodvoxels.nii.gz' if tag in GV_RUN else None
                wb_ribbon_weights(templ[g], f'{N}/{S}.{h}.midthickness.native.surf.gii', f'{N}/{S}.{h}.white.native.surf.gii',
                                  f'{N}/{S}.{h}.pial.native.surf.gii', txt, gv)
                sp.save_npz(f, parse_weights_text(txt, nv[h], grids[g]['shape']).tocsr()); os.remove(txt)
            Wts[(g, h, tag)] = sp.load_npz(f).tocsr()
    log['t_weights_s'] = round(time.time() - t0, 1)
    log['voxels_per_vertex'] = {f'{g}_{tag}': float(np.median(np.concatenate([np.diff(Wts[(g, h, tag)].indptr)[geo[h]['roi']] for h in 'LR'])))
                                for g, tag in COMBOS}

    # ---- 2. wedge data at 1.6 mm and emulated 2.4 / 3.2 mm, standard mapping, phase-encoded analysis
    t0 = time.time()
    img = nib.load(f'{R}/{CONCAT}/{CONCAT}_hp2000_clean.nii.gz')
    F8, POW, MEAN, SD = {}, {}, {}, {}
    for run, o in WEDGE.items():
        Y = np.ascontiguousarray(np.asarray(img.dataobj[..., o + BLANK:o + BLANK + NWIN], dtype=np.float32))
        ser = {'1.6': Y}
        ser.update(emulate_series(Y, {g: grids[g]['shape'] for g in ('2.4', '3.2')}, workers=nth))
        for g, tag in COMBOS:
            Yf = ser[g].reshape(-1, NWIN)
            for h in 'LR':
                wtag = 'gvcw' if (tag == 'gv' and run == 'cw') else tag
                X = (Wts[(g, h, wtag)] @ Yf).astype(np.float64)
                m = X.mean(1); Xc = X - m[:, None]
                Ff = np.fft.rfft(Xc, axis=1)
                k = (g, tag, h, run)
                F8[k] = Ff[:, CYC]; POW[k] = (np.abs(Ff[:, 1:]) ** 2).sum(1); MEAN[k] = m; SD[k] = Xc.std(1)
        del Y, ser, Yf
    # global haemodynamic delay (mod pi) from 1.6 mm, pericalcarine, coherence-weighted; shared by all resolutions
    zs = 0j
    for h in 'LR':
        c1 = np.abs(F8[('1.6', 'all', h, 'ccw')]) / np.sqrt(np.maximum(POW[('1.6', 'all', h, 'ccw')], 1e-12))
        c2 = np.abs(F8[('1.6', 'all', h, 'cw')]) / np.sqrt(np.maximum(POW[('1.6', 'all', h, 'cw')], 1e-12))
        e = np.exp(-1j * (np.angle(F8[('1.6', 'all', h, 'ccw')]) + np.angle(F8[('1.6', 'all', h, 'cw')])))
        zs += (area[h] * calc[h] * c1 * c2 * e).sum()
    dhat = np.angle(zs) / 2
    log['wedge'] = {'delay_rad_mod_pi': float(dhat)}
    theta, coh = {}, {}
    for g, tag in COMBOS:
        for h in 'LR':
            k1, k2 = (g, tag, h, 'ccw'), (g, tag, h, 'cw')
            a1, a2 = np.angle(F8[k1]), np.angle(F8[k2])
            dv = np.angle(np.exp(-1j * (a1 + a2))) / 2
            alt = dv + np.pi
            dv = np.where(np.abs(np.angle(np.exp(1j * (alt - dhat)))) < np.abs(np.angle(np.exp(1j * (dv - dhat)))), alt, dv)
            theta[(g, tag, h)] = np.angle(np.exp(1j * (-a1 - dv)))
            coh[(g, tag, h)] = 0.5 * (np.abs(F8[k1]) / np.sqrt(np.maximum(POW[k1], 1e-12)) + np.abs(F8[k2]) / np.sqrt(np.maximum(POW[k2], 1e-12)))
    base = {h: geo[h]['roi'] & (coh[('1.6', 'all', h)] > COH_THR) for h in 'LR'}
    for h in 'LR':
        res = {'n_coherent': int(base[h].sum()), 'n_coherent_calc': int((base[h] & calc[h]).sum())}
        for g, tag in COMBOS:
            th = theta[(g, tag, h)]; c = coh[(g, tag, h)]
            agree = cmean(th[base[h]] - theta[('1.6', 'all', h)][base[h]], area[h][base[h]])
            dang = np.degrees(np.abs(np.angle(np.exp(1j * (th - theta[('1.6', 'all', h)])))))
            res[f'{g}_{tag}'] = dict(
                median_coh_calc=float(np.median(c[calc[h]])), median_coh_cortex=float(np.median(c[geo[h]['roi']])),
                frac_cortex_coherent=wmean((c > COH_THR) * 1.0, area[h] * geo[h]['roi']),
                agreement_with_1p6=float(np.abs(agree)), median_abs_dtheta_vs_1p6_deg=float(np.median(dang[base[h]])),
                median_abs_dtheta_vs_1p6_calc_deg=float(np.median(dang[base[h] & calc[h]])),
                tsnr_median_calc=float(np.median((MEAN[(g, tag, h, 'ccw')] / np.maximum(SD[(g, tag, h, 'ccw')], 1e-6))[calc[h]])))
        log['wedge'][h] = res
        save_maps(f'{sd}/{S}.{h}.week2_wedge.func.gii',
                  {f'{m}_{g}_{tag}': v for (g, tag, hh) in theta if hh == h
                   for m, v in (('theta', theta[(g, tag, h)]), ('coherence', coh[(g, tag, h)]),
                                ('tmean', MEAN[(g, tag, h, 'ccw')]), ('tsd', SD[(g, tag, h, 'ccw')]))}, h)
    log['t_wedge_s'] = round(time.time() - t0, 1)

    # ---- 3+4. operator per grid, real-geometry leakage of standard mapping, cross-resolution prediction
    P, V, C, Z, _ = subject_points(geo)
    ncols = nv['L'] + nv['R']
    log['ribbon_volume_cm3'] = round(float(V.sum() / 1000), 1)
    su, fscale = {}, {}
    for h in 'LR':
        s = geo[h]['sphere']; Rs = np.linalg.norm(s, axis=1).mean()
        su[h] = s / np.linalg.norm(s, axis=1, keepdims=True)
        fscale[h] = Rs * np.sqrt(area[h][geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
    log['operator'] = {}; leakmaps = {}; pred = {}; fhat = None
    for g, grid in grids.items():
        t0 = time.time()
        tabs, rad = grid_tabs(g, grid, scales)
        K = build_operator(P, V * depth_weight(Z, 'b0'), C, ncols, grid, tabs, rad).tocsr()
        out = {'nnz_K': int(K.nnz), 't_build_K_s': round(time.time() - t0, 1),
               'psf_fwhm_mm_xyz': [round(2 * float(tb[0][np.argmax(tb[1] < tb[1][0] / 2)]), 3) for tb in tabs]}
        for tag in [t for (gg, t) in COMBOS if gg == g]:
            for h in 'LR':
                Wm = Wts[(g, h, tag)]; roi = geo[h]['roi']
                own, leak, leak_abs, leak_oh = (np.zeros(nv[h]) for _ in range(4))
                do_pred = fhat is not None and tag == 'all'
                pn = {r: np.zeros(nv[h], complex) for r in WEDGE}; pf = {r: np.zeros(nv[h], complex) for r in WEDGE}
                idx = np.flatnonzero(roi)
                for c0 in range(0, len(idx), 3000):
                    rows = idx[c0:c0 + 3000]; nr = len(rows)
                    A = (Wm[rows] @ K).tocoo()
                    j = A.col; val = A.data.astype(np.float64)
                    other = (j >= nv['L']) if h == 'L' else (j < nv['L'])
                    jl = np.where(j >= nv['L'], j - nv['L'], j)
                    cosang = np.einsum('nd,nd->n', su[h][rows[A.row]], su[h][np.where(other, 0, jl)])
                    far = other | (fscale[h] * np.arccos(np.clip(cosang, -1, 1)) > GEO_THR)
                    vn, vf = val * ~far, val * far
                    own[rows] = np.bincount(A.row, vn, minlength=nr); leak[rows] = np.bincount(A.row, vf, minlength=nr)
                    leak_abs[rows] = np.bincount(A.row, np.abs(vf), minlength=nr)
                    leak_oh[rows] = np.bincount(A.row, val * other, minlength=nr)
                    if do_pred:
                        for r in WEDGE:
                            fj = fhat[r][j]
                            pn[r][rows] = np.bincount(A.row, vn * fj.real, minlength=nr) + 1j * np.bincount(A.row, vn * fj.imag, minlength=nr)
                            pf[r][rows] = np.bincount(A.row, vf * fj.real, minlength=nr) + 1j * np.bincount(A.row, vf * fj.imag, minlength=nr)
                if do_pred:
                    pred[(g, h)] = (pn, pf)
                tot = own + leak
                lf = np.where(np.abs(tot) > 1e-6, leak / np.where(np.abs(tot) > 1e-6, tot, 1.0), 0.0)
                leakmaps[(g, tag, h)] = dict(own=own, leak=leak, leak_abs=leak_abs, leak_other_hemi=leak_oh, dilution=tot, leak_fraction=lf)
                has = np.diff(Wm.indptr) > 0          # vertices with no usable voxel (all masked) are filled by dilation in HCP
                w = area[h] * roi * has; wc = area[h] * calc[h] * has
                d = np.nan_to_num(w1[h]['d_opp'], nan=np.inf, posinf=np.inf)
                bins = {}
                for b0, b1 in zip(DBINS[:-1], DBINS[1:]):
                    sel = roi & has & (d >= b0) & (d < b1)
                    bins[f'{b0:g}-{b1:g}'] = dict(area_frac=float(area[h][sel].sum() / area[h][roi].sum()),
                                                  leak=wmean(leak, area[h] * sel), leak_abs=wmean(leak_abs, area[h] * sel),
                                                  own=wmean(own, area[h] * sel))
                okr = roi & has & np.isfinite(w1[h]['leak_naive_7T'])
                out[f'{tag}_{h}'] = dict(
                    frac_cortex_no_weights=float(1 - (area[h] * roi * has).sum() / (area[h] * roi).sum()),
                    dilution_mean=wmean(tot, w), own_mean=wmean(own, w), leak_mean=wmean(leak, w), leak_abs_mean=wmean(leak_abs, w),
                    leak_other_hemi_mean=wmean(leak_oh, w), leak_frac_mean=wmean(lf, w),
                    frac_leak_frac_gt2=wmean((lf > 0.02) * 1.0, w), frac_leak_frac_gt5=wmean((lf > 0.05) * 1.0, w),
                    frac_leak_frac_gt10=wmean((lf > 0.10) * 1.0, w),
                    calc_dilution_mean=wmean(tot, wc), calc_leak_mean=wmean(leak, wc), calc_leak_abs_mean=wmean(leak_abs, wc),
                    calc_leak_frac_mean=wmean(lf, wc), calc_frac_leak_frac_gt5=wmean((lf > 0.05) * 1.0, wc),
                    by_week1_dopp=bins,
                    r_week1_leak7T=float(np.corrcoef(w1[h]['leak_naive_7T'][okr], leak[okr])[0, 1]))
        if g == '1.6':    # field estimate for the cross-resolution test: 1.6 mm standard-mapped response / 1.6 mm dilution
            fhat = {r: np.concatenate([np.where(geo[h]['roi'], F8[('1.6', 'all', h, r)] / np.maximum(leakmaps[('1.6', 'all', h)]['dilution'], 0.3), 0)
                                       for h in 'LR']) for r in WEDGE}
        out['t_total_s'] = round(time.time() - t0, 1)
        log['operator'][g] = out
        print(S, g, 'operator + leakage', out['t_total_s'], 's, nnz', out['nnz_K'], flush=True)
        del K
    for h in 'LR':
        save_maps(f'{sd}/{S}.{h}.week2_leakage.func.gii',
                  {f'{k}_{g}_{tag}': v for (g, tag, hh), mp in leakmaps.items() if hh == h for k, v in mp.items()}, h)

    # cross-resolution prediction of the emulated maps (both wedge runs pooled), overall and by model leak fraction
    rng = np.random.default_rng(int(S)); log['prediction'] = {}
    for g in ('2.4', '3.2'):
        for h in 'LR':
            pn, pf = pred[(g, h)]; lf = leakmaps[(g, 'all', h)]['leak_fraction']
            res = {}
            for sname, sel in (('coherent', base[h]), ('coherent_leakfrac_gt5', base[h] & (lf > 0.05)),
                               ('coherent_leakfrac_gt10', base[h] & (lf > 0.10)), ('coherent_leakfrac_lt2', base[h] & (lf < 0.02)),
                               ('coherent_calc', base[h] & calc[h])):
                ii = np.flatnonzero(sel)
                if len(ii) < 30:
                    continue
                perm = rng.permutation(ii)
                zc = np.concatenate([F8[(g, 'all', h, r)][ii] for r in WEDGE])
                p0 = np.concatenate([F8[('1.6', 'all', h, r)][ii] for r in WEDGE])
                pN = np.concatenate([pn[r][ii] for r in WEDGE])
                pF = np.concatenate([pn[r][ii] + pf[r][ii] for r in WEDGE])
                pS = np.concatenate([pn[r][ii] + pf[r][perm] for r in WEDGE])
                res[sname] = dict(n=int(len(ii)), naive=resid(zc, p0), model_near=resid(zc, pN), model_full=resid(zc, pF),
                                  model_far_shuffled=resid(zc, pS))
            log['prediction'][f'{g}_{h}'] = res

    # ---- 5. geometry check vs Workbench (native grid, no mask)
    t0 = time.time(); log['geometry_check'] = {}
    for h in 'LR':
        pts, vols, tv = triangle_points(geo[h])
        B = box_ribbon_weights(pts, vols, tv, nv[h], grids['1.6']); del pts, vols, tv
        s = np.asarray(B.sum(1)).ravel(); B = (sp.diags(np.where(s > 0, 1 / np.maximum(s, 1e-12), 0)) @ B).tocsr()
        Q = Wts[('1.6', h, 'all')]; roi = geo[h]['roi']
        ov = np.asarray(B.minimum(Q).sum(1)).ravel()
        spq = np.asarray(B.multiply(Q).sum(1)).ravel(); sb2 = np.asarray(B.multiply(B).sum(1)).ravel()
        sq2 = np.asarray(Q.multiply(Q).sum(1)).ravel(); nu = np.diff((B + Q).tocsr().indptr).astype(float)
        r = (spq - 1 / np.maximum(nu, 1)) / np.sqrt(np.maximum(sb2 - 1 / np.maximum(nu, 1), 1e-12) * np.maximum(sq2 - 1 / np.maximum(nu, 1), 1e-12))
        ok = roi & (nu > 1)
        ob = np.asarray(B.sum(0)).ravel(); oq = np.asarray(Q.sum(0)).ravel(); sel = (ob > 0) | (oq > 0)
        log['geometry_check'][h] = dict(median_weight_overlap=float(np.median(ov[ok])), median_r=float(np.median(r[ok])),
                                        voxel_occupancy_r=float(np.corrcoef(ob[sel], oq[sel])[0, 1]),
                                        voxels_per_vertex_box=float(np.median(np.diff(B.indptr)[roi])),
                                        voxels_per_vertex_wb=float(np.median(np.diff(Q.indptr)[roi])))
        del B
    log['t_geometry_check_s'] = round(time.time() - t0, 1)
    log['t_total_s'] = round(time.time() - T0, 1)
    json.dump(log, open(f'{sd}/{S}.week2_summary.json', 'w'), indent=1)
    op, pr = log['operator'], log['prediction']
    print(S, 'done in', log['t_total_s'], 's | leak mean L (1.6gv, 1.6all, 2.4, 3.2):',
          [round(100 * op[g][f'{t}_L']['leak_mean'], 2) for g, t in COMBOS],
          '| prediction 2.4 L (naive, near, full, shuffled):',
          [round(pr['2.4_L']['coherent'][k], 3) for k in ('naive', 'model_near', 'model_full', 'model_far_shuffled')], flush=True)


if __name__ == '__main__':
    main()
