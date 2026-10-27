"""Week 17b, PLAN_NYU_GATE.md C: group far-leak maps of the 3T 2.0 mm acquisition (NYU, ds003787) and its 3.0 / 4.0 mm
emulations on fsaverage (164k), and the region table, from the per-subject maps of nyu_emul.py --gate --save-maps.
Per subject and hemisphere the far leak of each analysis vertex is carried from the subject's FreeSurfer registration sphere
(?h.sphere.reg) to fsaverage's by the three nearest vertices on the unit sphere with inverse-distance weights; per fsaverage
vertex the group mean (vertices covered in at least half of the subjects) and the share of subjects above 5%. Region table: per
subject, area-weighted mean far leak and share above 10% per Desikan-Killiany region (subject space, midthickness areas), then
the mean over subjects. Writes GIFTI maps, a JSON summary and a markdown table into OUTDIR (inside GEO_CONST).
Usage (fmriprep container: nibabel, scipy, FreeSurfer's fsaverage): python3 nyu_groupmaps.py OUTDIR
"""
import os, sys, json, glob
import numpy as np, nibabel as nib
from scipy.spatial import cKDTree

G = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
FSD = f'{G}/data/ds003787/derivatives/freesurfer'
MAPS = os.environ.get('MAPS', f'{G}/results/nyu/emul_gate')                 # MAPS / LIMIT: for a test on other map files
FSAVG = os.environ.get('FSAVG', '/opt/freesurfer/subjects/fsaverage')
GRIDS = {'2.0': 'far_n', '3.0': 'far_3.0', '4.0': 'far_4.0'}


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def vertex_areas(V, F):
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    out = np.zeros(len(V)); np.add.at(out, F.ravel(), np.repeat(a / 3, 3)); return out


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(f'{MAPS}/*.nyu_maps.npz'))[:int(os.environ.get('LIMIT', '1000'))]; subs = [os.path.basename(f).split('.')[0] for f in files]
    fs_sph = {h: unit(nib.freesurfer.read_geometry(f'{FSAVG}/surf/{fh}.sphere.reg')[0]) for h, fh in (('L', 'lh'), ('R', 'rh'))}
    fsW = {h: nib.freesurfer.read_geometry(f'{FSAVG}/surf/{fh}.white') for h, fh in (('L', 'lh'), ('R', 'rh'))}
    fsP = {h: nib.freesurfer.read_geometry(f'{FSAVG}/surf/{fh}.pial')[0] for h, fh in (('L', 'lh'), ('R', 'rh'))}
    fs_area = {h: vertex_areas(0.5 * (fsW[h][0] + fsP[h]), fsW[h][1]) for h in 'LR'}
    acc = {g: {h: [np.zeros(len(fs_sph[h])), np.zeros(len(fs_sph[h])), np.zeros(len(fs_sph[h]))] for h in 'LR'} for g in GRIDS}   # sum, n, n>5%
    regions = {g: {} for g in GRIDS}
    for f, S in zip(files, subs):
        M = np.load(f); idx = M['idx']
        geo = {}
        for h, fh in (('L', 'lh'), ('R', 'rh')):
            Vw, F = nib.freesurfer.read_geometry(f'{FSD}/{S}/surf/{fh}.white'); Vp, _ = nib.freesurfer.read_geometry(f'{FSD}/{S}/surf/{fh}.pial')
            sph = unit(nib.freesurfer.read_geometry(f'{FSD}/{S}/surf/{fh}.sphere.reg')[0])
            lab, _, names = nib.freesurfer.read_annot(f'{FSD}/{S}/label/{fh}.aparc.annot')
            geo[h] = dict(n=len(Vw), sph=sph, area=vertex_areas(0.5 * (Vw + Vp), F), lab=lab, names=[n.decode() if isinstance(n, bytes) else n for n in names])
        nL = geo['L']['n']
        for g, key in GRIDS.items():
            vals = M[key].astype(np.float64)
            for h in 'LR':
                full = np.full(geo[h]['n'], np.nan)
                sel = (idx < nL) if h == 'L' else (idx >= nL)
                full[idx[sel] - (0 if h == 'L' else nL)] = vals[sel]
                ok = np.isfinite(full)
                tree = cKDTree(geo[h]['sph'][ok]); d, j = tree.query(fs_sph[h], k=3)
                w = 1.0 / np.maximum(d, 1e-6); v = full[ok][j]
                res = (w * v).sum(1) / w.sum(1)
                near = d[:, 0] < 0.02                                       # ~ 2 mm on a 100 mm sphere: outside the analysis vertices otherwise
                acc[g][h][0][near] += res[near]; acc[g][h][1][near] += 1; acc[g][h][2][near] += (res[near] > 0.05)
                # region table in subject space
                for r, name in enumerate(geo[h]['names']):
                    m = ok & (geo[h]['lab'] == r)
                    if m.sum() < 20 or name in ('unknown', 'corpuscallosum'):
                        continue
                    a = geo[h]['area'][m]; x = full[m]
                    regions[g].setdefault(name, []).append((float((a * x).sum() / a.sum()), float((a * (x > 0.10)).sum() / a.sum())))
        print(S, 'done', flush=True)
    summary = {'subjects': subs, 'n': len(subs), 'grids': {}}
    for g in GRIDS:
        summary['grids'][g] = {}
        cm, ca = [], []
        for h, fh in (('L', 'lh'), ('R', 'rh')):
            s_, n_, k_ = acc[g][h]
            mean = np.where(n_ >= 0.5 * len(subs), s_ / np.maximum(n_, 1), np.nan); share = np.where(n_ >= 0.5 * len(subs), k_ / np.maximum(n_, 1), np.nan)
            for arr, desc in ((mean, 'mean'), (share, 'share-gt5pct')):
                img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(arr.astype(np.float32), intent='NIFTI_INTENT_NONE', datatype='NIFTI_TYPE_FLOAT32')])
                nib.save(img, f'{out}/tpl-fsaverage_hemi-{h}_den-164k_desc-farleak3T{g.replace(".", "p")}mm-{desc}.func.gii')
            ok = np.isfinite(mean); cm.append((mean[ok], fs_area[h][ok]))
        m_all = np.concatenate([c[0] for c in cm]); a_all = np.concatenate([c[1] for c in cm])
        summary['grids'][g] = dict(group_mean_far=float((a_all * m_all).sum() / a_all.sum()), group_gt5=float((a_all * (m_all > 0.05)).sum() / a_all.sum()),
                                   group_gt10=float((a_all * (m_all > 0.10)).sum() / a_all.sum()))
        reg = {name: (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])), len(vals)) for name, vals in regions[g].items()}
        summary['grids'][g]['regions'] = {k: dict(mean_far=v[0], share_gt10=v[1], n=v[2]) for k, v in reg.items()}
    json.dump(summary, open(f'{out}/nyu_groupmaps.json', 'w'), indent=1)
    # fsaverage inflated surfaces for drawing the figure locally (not part of the release)
    os.makedirs(f'{out}/_draw', exist_ok=True)
    for h, fh in (('L', 'lh'), ('R', 'rh')):
        V, F = nib.freesurfer.read_geometry(f'{FSAVG}/surf/{fh}.inflated')
        np.savez_compressed(f'{out}/_draw/fsaverage_{h}_inflated.npz', V=V.astype(np.float32), F=F.astype(np.int32))
    L = [f'# Week 17b C: far leak of the 3T 2.0 mm acquisition (NYU) and its emulations — group maps on fsaverage ({len(subs)} subjects)', '',
         '| grid | group-mean far leak (%) | cortex > 5% in the group map (%) | cortex > 10% in the group map (%) |', '|---|---|---|---|']
    for g in GRIDS:
        x = summary['grids'][g]; L.append(f"| {g} mm | {100 * x['group_mean_far']:.2f} | {100 * x['group_gt5']:.1f} | {100 * x['group_gt10']:.1f} |")
    for g in ('2.0', '4.0'):
        top = sorted(summary['grids'][g]['regions'].items(), key=lambda kv: -kv[1]['mean_far'])[:8]
        L += ['', f'Regions with the highest far leak at {g} mm (area-weighted mean over the region, mean over subjects; share of the region above 10%):', '',
              '| region | mean far leak (%) | share above 10% (%) |', '|---|---|---|'] + [f"| {k} | {100 * v['mean_far']:.1f} | {100 * v['share_gt10']:.0f} |" for k, v in top]
    open(f'{out}/nyu_groupmaps.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
