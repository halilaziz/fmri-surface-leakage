"""Week 13: the week-2 phase-encoded analysis for the ring runs (RETEXP_AP, RETCON_PA) of one HCP subject, after week-2
run_subject.py has produced the operator products. Same window (22 s blank, 256 s, DFT bin 8: period 32 s, measured by
ring_freq_check.py), same combos (1.6 mm with the run's own goodvoxels, 1.6 / 2.4 / 3.2 mm without a mask), same statistics.
Writes, next to the week-2 files: ribbon weights with the ring runs' goodvoxels (hcpW_1.6_{h}_gvexp / gvcon), the maps
{S}.{h}.week2_ring.func.gii (theta, coherence, tmean, tsd per combo; key names as the wedge maps) and {S}.week13_ring.json.
Usage (xcp_d container): python3 ring_subject.py SUBJECT [--threads 8]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp, nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '03_hcp_emulation')); sys.path.insert(0, os.path.join(HERE, '..', '02_forward_operator'))
from w3lib import *                                                  # w2lib too (HCP, CONCAT, grids, geometry, weights)
from run_subject import save_maps, wmean, cmean, COMBOS, COH_THR      # week-2 helpers (the import is guarded by __main__)

W2SUB = f'{W_ROOT}/results/week2/subjects'
RUNS, OTHER, W16TAG = stim_runs('ring')                              # {'exp': 600, 'con': 900}; weights gvexp / gvcon
GV_RUN = {'gvexp': 'RETEXP_7T_AP', 'gvcon': 'RETCON_7T_PA'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); S = a.subject; nth = a.threads
    sd = f'{W2SUB}/{S}'; assert os.path.exists(f'{sd}/{S}.week2_summary.json'), 'run week-2 run_subject.py first'
    R = f'{HCP}/{S}/MNINonLinear/Results'; N = f'{HCP}/{S}/MNINonLinear/Native'
    T0 = time.time(); log = {'subject': S, 'runs': RUNS, 'window': dict(blank=BLANK, nwin=NWIN, cyc=CYC)}
    say = lambda *x: print(S, f'{time.time() - T0:6.0f}s', *x, flush=True)

    sbref, grids = subject_grids(S)
    templ = {'1.6': sbref, '2.4': f'{sd}/grid_2.4mm_template.nii.gz', '3.2': f'{sd}/grid_3.2mm_template.nii.gz'}
    geo = {h: hemi_geometry(S, h) for h in 'LR'}
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    area = {h: vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])[0] for h in 'LR'}
    calc = {}
    for h in 'LR':
        lab = nib.load(f'{N}/{S}.{h}.aparc.native.label.gii')
        key = [k for k, v in lab.labeltable.get_labels_as_dict().items() if 'pericalcarine' in v.lower()][0]
        calc[h] = (np.asarray(lab.darrays[0].data) == key) & geo[h]['roi']

    # ---- ribbon weights: the ring runs' own goodvoxels at 1.6 mm (new), the unmasked sets from week 2
    t0 = time.time(); Wts = {}
    for g, tag in [('1.6', 'gvexp'), ('1.6', 'gvcon')] + [(g, 'all') for g in ('1.6', '2.4', '3.2')]:
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

    # ---- ring data at 1.6 mm and emulated 2.4 / 3.2 mm, standard mapping, phase-encoded analysis (as week 2 for the wedges)
    t0 = time.time()
    img = nib.load(f'{R}/{CONCAT}/{CONCAT}_hp2000_clean.nii.gz')
    F8, POW, MEAN, SD = {}, {}, {}, {}
    r0, r1 = list(RUNS)
    for run, o in RUNS.items():
        Y = np.ascontiguousarray(np.asarray(img.dataobj[..., o + BLANK:o + BLANK + NWIN], dtype=np.float32))
        ser = {'1.6': Y}
        ser.update(emulate_series(Y, {g: grids[g]['shape'] for g in ('2.4', '3.2')}, workers=nth))
        for g, tag in COMBOS:
            Yf = ser[g].reshape(-1, NWIN)
            for h in 'LR':
                wtag = W16TAG[run] if tag == 'gv' else tag
                X = (Wts[(g, h, wtag)] @ Yf).astype(np.float64)
                m = X.mean(1); Xc = X - m[:, None]
                Ff = np.fft.rfft(Xc, axis=1)
                k = (g, tag, h, run)
                F8[k] = Ff[:, CYC]; POW[k] = (np.abs(Ff[:, 1:]) ** 2).sum(1); MEAN[k] = m; SD[k] = Xc.std(1)
        del Y, ser, Yf
    zs = 0j
    for h in 'LR':
        c1 = np.abs(F8[('1.6', 'all', h, r0)]) / np.sqrt(np.maximum(POW[('1.6', 'all', h, r0)], 1e-12))
        c2 = np.abs(F8[('1.6', 'all', h, r1)]) / np.sqrt(np.maximum(POW[('1.6', 'all', h, r1)], 1e-12))
        e = np.exp(-1j * (np.angle(F8[('1.6', 'all', h, r0)]) + np.angle(F8[('1.6', 'all', h, r1)])))
        zs += (area[h] * calc[h] * c1 * c2 * e).sum()
    dhat = np.angle(zs) / 2
    log['ring'] = {'delay_rad_mod_pi': float(dhat)}
    theta, coh = {}, {}
    for g, tag in COMBOS:
        for h in 'LR':
            k1, k2 = (g, tag, h, r0), (g, tag, h, r1)
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
                tsnr_median_calc=float(np.median((MEAN[(g, tag, h, r0)] / np.maximum(SD[(g, tag, h, r0)], 1e-6))[calc[h]])))
        log['ring'][h] = res
        save_maps(f'{sd}/{S}.{h}.week2_ring.func.gii',
                  {f'{m}_{g}_{tag}': v for (g, tag, hh) in theta if hh == h
                   for m, v in (('theta', theta[(g, tag, h)]), ('coherence', coh[(g, tag, h)]),
                                ('tmean', MEAN[(g, tag, h, r0)]), ('tsd', SD[(g, tag, h, r0)]))}, h)
    log['t_ring_s'] = round(time.time() - t0, 1); log['t_total_s'] = round(time.time() - T0, 1)
    json.dump(log, open(f'{sd}/{S}.week13_ring.json', 'w'), indent=1)
    say('done; coherent vertices', {h: log['ring'][h]['n_coherent'] for h in 'LR'}, 'delay', round(float(dhat), 3))


if __name__ == '__main__':
    main()
