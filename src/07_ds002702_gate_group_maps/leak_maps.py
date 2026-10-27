"""Week 7 deliverable: group maps of operator-predicted leakage of standard (ribbon-constrained) mapping on the fs_LR 32k surface.
Input: the week-2 per-subject leakage maps (HCP 7T geometry and PSF; native 1.6 mm and emulated 2.4 / 3.2 mm), resampled from each
subject's native mesh to fs_LR 32k with its MSMAll sphere (wb_command -metric-resample ADAP_BARY_AREA). Output per grid: mean and
subject-SD of the far-leak fraction, fraction of subjects above 10%; area-weighted summary per Desikan region (native aparc); a
figure (lateral and medial views of the very inflated fs_LR surfaces); GIFTI files for reuse.
Usage (xcp_d container): python3 leak_maps.py OUTDIR
"""
import os, sys, json, glob, subprocess, tempfile
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_forward_operator'))
from w2lib import HCP, W_ROOT, load_surf, vertex_areas_normals

W2SUB = f'{W_ROOT}/results/week2/subjects'
TPL = os.environ.get('TEMPLATEFLOW_HOME', '/home/xcp_d/.cache/templateflow') + '/tpl-fsLR'
GRIDS = ('1.6', '2.4', '3.2')


def gmaps(p):
    return {d.meta['Name']: np.asarray(d.data, np.float64) for d in nib.load(p).darrays}


def save_func(path, arrays, hemi):
    das = []
    for name, v in arrays.items():
        d = nib.gifti.GiftiDataArray(np.asarray(v, np.float32), intent='NIFTI_INTENT_NONE', datatype='NIFTI_TYPE_FLOAT32')
        d.meta = nib.gifti.GiftiMetaData({'Name': name}); das.append(d)
    img = nib.gifti.GiftiImage(darrays=das)
    img.meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary': {'L': 'CortexLeft', 'R': 'CortexRight'}[hemi]})
    nib.save(img, path)


def draw_figure(out, nsub):
    """Lateral / medial views of the very inflated fs_LR 32k surfaces, mean far-leak fraction per grid, one shared colour bar."""
    fig, ax = plt.subplots(len(GRIDS), 4, figsize=(15, 3.0 * len(GRIDS) + 0.8))
    for gi, g in enumerate(GRIDS):
        for hi, h in enumerate('LR'):
            sf = nib.load(f'{out}/fsLR32k.{h}.veryinflated.surf.gii')
            V, F = sf.agg_data('NIFTI_INTENT_POINTSET').astype(float), sf.agg_data('NIFTI_INTENT_TRIANGLE').astype(np.int64)
            fu = nib.load(f'{out}/group_leakage.fsLR32k.{h}.func.gii')
            val = {d.meta['Name']: np.asarray(d.data, float) for d in fu.darrays}[f'leak_mean_{g}']
            fc = np.ma.masked_where((val[F] == 0).all(1), val[F].mean(1))      # medial wall (stored as 0) drawn grey
            cm = plt.get_cmap('magma_r').copy(); cm.set_bad('0.8')
            fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
            for vi, (side, sgn) in enumerate((('lateral', -1 if h == 'L' else 1), ('medial', 1 if h == 'L' else -1))):
                a = ax[gi, 2 * hi + vi]
                vis = fn[:, 0] * sgn > 0
                order = np.argsort(sgn * V[F, 0].mean(1))
                order = order[vis[order]]
                xs = V[:, 1] * sgn                           # anterior to the left for left-lateral / right-medial views
                tpc = a.tripcolor(Triangulation(xs, V[:, 2], F[order]), facecolors=fc[order], cmap=cm, vmin=0, vmax=0.3)
                a.set_aspect('equal'); a.axis('off')
                a.set_title(f'{h} {side}, {g} mm', fontsize=9)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.02, hspace=0.12)
    cax = fig.add_axes([0.35, 0.05, 0.30, 0.018])
    fig.colorbar(tpc, cax=cax, orientation='horizontal', label='far-leak fraction of standard mapping (share of the mapped signal from > 15 mm or the other hemisphere)')
    fig.suptitle(f'Operator-predicted leakage of standard mapping, HCP 7T geometry (mean over {nsub} subjects, fs_LR 32k)', fontsize=10)
    plt.savefig(f'{out}/figW7-1_group_leakage.png', dpi=140); plt.close()


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); tmp = tempfile.mkdtemp(dir=out)
    subs = sorted(os.path.basename(d) for d in glob.glob(f'{W2SUB}/*') if os.path.exists(f'{d}/{os.path.basename(d)}.L.week2_leakage.func.gii'))
    stacks = {(g, h): [] for g in GRIDS for h in 'LR'}; regions = {}; skipped = []
    for S in list(subs):
        N = f'{HCP}/{S}/MNINonLinear/Native'
        if not all(os.path.exists(f'{N}/{S}.{h}.sphere.MSMAll.native.surf.gii') for h in 'LR'):
            skipped.append(S); subs.remove(S); print(S, 'skipped: no MSMAll sphere', flush=True); continue
        for h in 'LR':
            m = gmaps(f'{W2SUB}/{S}/{S}.{h}.week2_leakage.func.gii')
            keys = {g: f'leak_fraction_{g}_all' if f'leak_fraction_{g}_all' in m else f'leak_fraction_{g}_gv' for g in GRIDS}
            src = f'{tmp}/{S}.{h}.leak.func.gii'
            save_func(src, {g: m[keys[g]] for g in GRIDS}, h)
            dst = f'{tmp}/{S}.{h}.leak.32k.func.gii'
            subprocess.run(['wb_command', '-metric-resample', src, f'{N}/{S}.{h}.sphere.MSMAll.native.surf.gii',
                            f'{TPL}/tpl-fsLR_hemi-{h}_den-32k_sphere.surf.gii', 'ADAP_BARY_AREA', dst, '-area-surfs',
                            f'{N}/{S}.{h}.midthickness.native.surf.gii', f'{TPL}/tpl-fsLR_den-32k_hemi-{h}_midthickness.surf.gii'],
                           check=True, stdout=subprocess.DEVNULL)
            r = gmaps(dst)
            for g in GRIDS:
                stacks[(g, h)].append(r[g])
            roi32 = f'{tmp}/{S}.{h}.roi.32k.func.gii'           # the subject's cortex mask (atlasroi) on fs_LR 32k
            subprocess.run(['wb_command', '-metric-resample', f'{N}/{S}.{h}.atlasroi.native.shape.gii', f'{N}/{S}.{h}.sphere.MSMAll.native.surf.gii',
                            f'{TPL}/tpl-fsLR_hemi-{h}_den-32k_sphere.surf.gii', 'ADAP_BARY_AREA', roi32, '-area-surfs',
                            f'{N}/{S}.{h}.midthickness.native.surf.gii', f'{TPL}/tpl-fsLR_den-32k_hemi-{h}_midthickness.surf.gii'],
                           check=True, stdout=subprocess.DEVNULL)
            stacks.setdefault(('roi', h), []).append(np.asarray(nib.load(roi32).darrays[0].data, float))
            # regional summary on the native mesh (Desikan aparc), area-weighted over atlasroi vertices
            lab = nib.load(f'{N}/{S}.{h}.aparc.native.label.gii'); L = np.asarray(lab.darrays[0].data)
            names = lab.labeltable.get_labels_as_dict()
            roi = np.asarray(nib.load(f'{N}/{S}.{h}.atlasroi.native.shape.gii').agg_data()) > 0.5
            V, F = load_surf(f'{N}/{S}.{h}.midthickness.native.surf.gii'); area, _ = vertex_areas_normals(V, F)
            for k, nm in names.items():
                sel = (L == k) & roi
                if sel.sum() < 20 or 'unknown' in nm.lower() or 'medial' in nm.lower():
                    continue
                nm = nm.replace('L_', '').replace('R_', '')
                for g in GRIDS:
                    lf = m[keys[g]]
                    regions.setdefault(nm, {}).setdefault(g, []).append(dict(mean=float(np.sum(area[sel] * lf[sel]) / area[sel].sum()),
                                                                              frac10=float(np.sum(area[sel] * (lf[sel] > 0.10)) / area[sel].sum())))
        print(S, 'done', flush=True)
    medial = {h: np.mean(stacks[('roi', h)], 0) < 0.5 for h in 'LR'}        # medial wall = outside most subjects' cortex masks
    infl = {}
    for h in 'LR':                                                         # the template's inflated surfaces are not populated
        infl[h] = f'{out}/fsLR32k.{h}.veryinflated.surf.gii'
        subprocess.run(['wb_command', '-surface-generate-inflated', f'{TPL}/tpl-fsLR_den-32k_hemi-{h}_midthickness.surf.gii',
                        f'{tmp}/fsLR32k.{h}.inflated.surf.gii', infl[h]], check=True, stdout=subprocess.DEVNULL)
    va = {h: np.asarray(nib.load(f'{TPL}/tpl-fsLR_hemi-{h}_den-32k_desc-vaavg_midthickness.shape.gii').darrays[0].data, float) for h in 'LR'}
    summary = {'subjects': subs, 'skipped_no_msmall': skipped, 'grids': {}}
    for h in 'LR':
        arrs = {}
        for g in GRIDS:
            X = np.array(stacks[(g, h)]); X[:, medial[h]] = np.nan
            arrs[f'leak_mean_{g}'] = np.nanmean(X, 0); arrs[f'leak_sd_{g}'] = np.nanstd(X, 0)
            arrs[f'frac_subjects_leak_gt10_{g}'] = np.nanmean(X > 0.10, 0)
        save_func(f'{out}/group_leakage.fsLR32k.{h}.func.gii', {k: np.nan_to_num(v) for k, v in arrs.items()}, h)
        for g in GRIDS:
            m_ = arrs[f'leak_mean_{g}']; ok = ~medial[h] & np.isfinite(m_); w = va[h][ok]
            summary['grids'].setdefault(g, {})[h] = dict(
                mean=float(np.sum(w * m_[ok]) / w.sum()),
                **{f'area_frac_gt{int(100 * t)}': float(np.sum(w * (m_[ok] > t)) / w.sum()) for t in (0.05, 0.10, 0.20)})
    reg = {nm: {g: dict(mean=float(np.mean([x['mean'] for x in v[g]])), frac10=float(np.mean([x['frac10'] for x in v[g]])))
                for g in GRIDS} for nm, v in regions.items()}
    summary['regions'] = dict(sorted(reg.items(), key=lambda kv: -kv[1]['3.2']['mean']))
    json.dump(summary, open(f'{out}/group_leakage_summary.json', 'w'), indent=1)
    draw_figure(out, len(subs))
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(summary['grids'], indent=1))
    print('top regions (3.2 mm mean leak):', [(k, round(v['3.2']['mean'], 3)) for k, v in list(summary['regions'].items())[:10]])


if __name__ == '__main__':
    if sys.argv[1] == '--figure-only':          # redraw from the saved GIFTI files: python3 leak_maps.py --figure-only OUTDIR
        draw_figure(sys.argv[2], len(json.load(open(f'{sys.argv[2]}/group_leakage_summary.json'))['subjects']))
    else:
        main()
