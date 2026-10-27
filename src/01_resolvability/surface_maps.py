"""MVE week 1, part 2: per-subject resolvability and sulcal-bank-proximity maps on real HCP surfaces.

For one subject:
  1. warp the MNINonLinear native-mesh surfaces back to ACPC (T1w) space with the subject's HCP warps
     (the EPI grids were AutoAligned to ACPC, so ACPC axes approximate the acquisition lattice axes);
     sanity check: |pial - white| should match FreeSurfer thickness better in ACPC than in MNI space;
  2. per vertex (midthickness normal n):
       lam_ideal_<acq>   closed-form resolvable isotropic tangential wavelength, 3D-Fourier idealisation
       lam_<acq>_<cfg>   realistic value interpolated from the lookup tables (2D multislice + T2*)
       dof_<acq>_<cfg>   degrees of freedom for signal per cm^2 (lookup)
  3. d_opp: Euclidean distance (mm, midthickness) to the nearest cortex that is geodesically far
     (> 15 mm, judged on the registration sphere) or in the other hemisphere (midline);
     leak_naive_<acq>: predicted cross-bank leakage of ribbon-style averaging at that separation;
     tau8_<acq>_cnr<c>: predicted bank-specific transfer of the optimal estimator at 7.9 mm wavelength.
Outputs: one multi-map GIFTI per hemisphere, ACPC surfaces, and a JSON summary.

Usage (inside the xcp_d container): python3 surface_maps.py SUBJECT LOOKUPDIR OUTDIR
"""
import os, sys, json, subprocess, argparse
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

W = os.environ.get('GEO_ROOT', '/path/to/geo_root')
HCP = f'{W}/data/hcp_extra'
DG = f'{W}/data/hcp7t_rest_dgcon'
H = {'3T': np.array([2.0, 2.0, 2.0]), '7T': np.array([1.6, 1.6, 1.6])}
STRUCT = {'L': 'CortexLeft', 'R': 'CortexRight'}


def load_surf(p):
    g = nib.load(p)
    return g.agg_data('NIFTI_INTENT_POINTSET').astype(np.float64), g.agg_data('NIFTI_INTENT_TRIANGLE').astype(np.int64)


def load_shape(p):
    return np.asarray(nib.load(p).agg_data(), dtype=np.float64)


def save_maps(path, maps, hemi):
    das = []
    for name, x in maps.items():
        da = nib.gifti.GiftiDataArray(np.asarray(x, np.float32), intent='NIFTI_INTENT_NONE', datatype='NIFTI_TYPE_FLOAT32')
        da.meta = nib.gifti.GiftiMetaData({'Name': name})
        das.append(da)
    img = nib.gifti.GiftiImage(darrays=das, meta=nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary': STRUCT[hemi]}))
    nib.save(img, path)


def vertex_normals_areas(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fa = 0.5 * np.linalg.norm(fn, axis=1)
    vn = np.zeros_like(V); va = np.zeros(len(V))
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
        np.add.at(va, F[:, k], fa / 3)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    return vn, va


def lam_ideal(nrm, h):
    """2*pi/rho(n) with rho = pi * min_{u in T_n, |u|=1} sum_d |u_d|/h_d; the minimum sits at u ∝ n x e_i."""
    best = np.full(len(nrm), np.inf)
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        u = np.cross(nrm, e)
        L = np.linalg.norm(u, axis=1)
        ok = L > 1e-9
        val = np.full(len(nrm), np.inf)
        val[ok] = (np.abs(u[ok]) / h).sum(1) / L[ok]
        best = np.minimum(best, val)
    return 2.0 / best


def interp_dirs(nrm, D, vals, k=3):
    a = np.abs(nrm)
    k = min(k, len(D))
    d, i = cKDTree(D).query(a, k=k)
    if k == 1:
        return vals[i]
    w = 1.0 / np.maximum(d, 1e-6)
    return (vals[i] * w).sum(1) / w.sum(1)


def warp_to_acpc(S, hemi, surf, outdir):
    src = f'{HCP}/{S}/MNINonLinear/Native/{S}.{hemi}.{surf}.native.surf.gii'
    dst = f'{outdir}/{S}.{hemi}.{surf}.acpc.surf.gii'
    if not os.path.exists(dst):
        subprocess.run(['wb_command', '-surface-apply-warpfield', src,
                        f'{DG}/{S}/MNINonLinear/xfms/acpc_dc2standard.nii.gz', dst,
                        '-fnirt', f'{DG}/{S}/MNINonLinear/xfms/standard2acpc_dc.nii.gz'], check=True)
    return dst


def bank_distance(mid, sph, mask, area_tot, k=256, geo_thr=15.0):
    """mid/sph/mask: dicts hemi -> arrays. Returns dict hemi -> d_opp (inf if no far cortex among k nearest)."""
    hemis = ['L', 'R']
    pts = np.vstack([mid[h][mask[h]] for h in hemis])
    lab = np.concatenate([np.full(mask[h].sum(), j) for j, h in enumerate(hemis)])
    loc = np.concatenate([np.flatnonzero(mask[h]) for h in hemis])
    tree = cKDTree(pts)
    out = {}
    for j, h in enumerate(hemis):
        Rs = np.linalg.norm(sph[h], axis=1).mean()
        su = sph[h] / np.linalg.norm(sph[h], axis=1, keepdims=True)
        f = np.sqrt(area_tot[h] / (4 * np.pi * Rs ** 2))          # sphere -> cortex distance scale
        src = np.flatnonzero(mask[h])
        d_opp = np.full(len(mid[h]), np.nan)
        for c0 in range(0, len(src), 20000):
            s = src[c0:c0 + 20000]
            dist, idx = tree.query(mid[h][s], k=k, workers=4)
            other = lab[idx] != j
            same_loc = loc[idx]
            cosang = np.einsum('nd,nkd->nk', su[s], su[np.where(other, 0, same_loc)])
            geo = f * Rs * np.arccos(np.clip(cosang, -1, 1))
            far = other | (geo > geo_thr)
            dd = np.where(far, dist, np.inf).min(1)
            d_opp[s] = dd
        out[h] = d_opp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subject'); ap.add_argument('lookupdir'); ap.add_argument('outdir')
    a = ap.parse_args()
    S = a.subject
    sd = os.path.join(a.outdir, S); os.makedirs(sd, exist_ok=True)
    look = json.load(open(os.path.join(a.lookupdir, 'lookup_dirs.json')))
    banks = json.load(open(os.path.join(a.lookupdir, 'lookup_banks.json')))
    tables = {}
    for r in look['results']:
        tables.setdefault((r['acq'], r['cfg']), []).append(r)
    bank = {}
    for r in banks['results']:
        bank.setdefault((r['acq'], r['cnr']), []).append(r)

    mid, sph, mask, nrm, area, area_tot, summ = {}, {}, {}, {}, {}, {}, {'subject': S}
    for hemi in ['L', 'R']:
        surf = {s: load_surf(warp_to_acpc(S, hemi, s, sd)) for s in ['white', 'pial', 'midthickness']}
        mni = {s: load_surf(f'{HCP}/{S}/MNINonLinear/Native/{S}.{hemi}.{s}.native.surf.gii') for s in ['white', 'pial', 'midthickness']}
        V, F = surf['midthickness']
        thick = load_shape(f'{HCP}/{S}/MNINonLinear/Native/{S}.{hemi}.thickness.native.shape.gii')
        roi = load_shape(f'{HCP}/{S}/MNINonLinear/Native/{S}.{hemi}.atlasroi.native.shape.gii') > 0.5
        m = roi & (thick > 0.5)
        pw = np.linalg.norm(surf['pial'][0] - surf['white'][0], axis=1)
        pw_mni = np.linalg.norm(mni['pial'][0] - mni['white'][0], axis=1)
        n, ar = vertex_normals_areas(V, F)
        _, ar_mni = vertex_normals_areas(*mni['midthickness'])
        mid[hemi], mask[hemi], nrm[hemi], area[hemi] = V, m, n, ar
        area_tot[hemi] = ar[m].sum()
        sph[hemi] = load_surf(f'{HCP}/{S}/MNINonLinear/Native/{S}.{hemi}.sphere.native.surf.gii')[0]
        summ[hemi] = dict(n_vertices=int(len(V)), n_cortex=int(m.sum()),
                          area_cm2_acpc=float(ar[m].sum() / 100), area_cm2_mni=float(ar_mni[m].sum() / 100),
                          pw_over_thick_acpc=float(np.median(pw[m] / thick[m])), pw_over_thick_mni=float(np.median(pw_mni[m] / thick[m])),
                          r_pw_thick_acpc=float(np.corrcoef(pw[m], thick[m])[0, 1]), r_pw_thick_mni=float(np.corrcoef(pw_mni[m], thick[m])[0, 1]),
                          thickness_median=float(np.median(thick[m])))
        summ[hemi]['_thick'] = thick

    dopp = bank_distance(mid, sph, mask, area_tot)
    for hemi in ['L', 'R']:
        n, ar, m = nrm[hemi], area[hemi], mask[hemi]
        w = ar * m
        maps = {}
        for acq in H:
            maps[f'lam_ideal_{acq}'] = lam_ideal(n, H[acq])
            for cfg in ['white_cnr1', 'white_cnr10', 'bold_cnr10']:
                T = tables[(acq, cfg)]
                D = np.array([t['n'] for t in T]); D /= np.linalg.norm(D, axis=1, keepdims=True)
                maps[f'lam_{acq}_{cfg}'] = interp_dirs(n, D, np.array([t['iso_mm'] for t in T]))
                maps[f'dof_{acq}_{cfg}'] = interp_dirs(n, D, np.array([t['dfs_per_cm2'] for t in T]))
            for cnr in [1.0, 10.0]:
                B = sorted(bank[(acq, cnr)], key=lambda r: r['d'])
                dgrid = np.array([b['d'] for b in B])
                x = np.clip(np.nan_to_num(dopp[hemi], nan=dgrid[-1], posinf=dgrid[-1]), dgrid[0], dgrid[-1])
                for qi, tag in ((2, 'tau8'), (3, 'tau5'), (4, 'tau4')):
                    maps[f'{tag}_{acq}_cnr{int(cnr)}'] = np.interp(x, dgrid, [b['tau_anti'][qi] for b in B])
                if cnr == 1.0:
                    leak = np.interp(x, dgrid, [max(b['leak_naive'][0], 0) for b in B])
                    leak[np.nan_to_num(dopp[hemi], nan=99, posinf=99) >= dgrid[-1]] = 0.0
                    maps[f'leak_naive_{acq}'] = leak
        maps['d_opp'] = np.where(np.isfinite(dopp[hemi]), dopp[hemi], 99.0)
        maps['thickness'] = summ[hemi].pop('_thick')
        for kname in maps:
            maps[kname] = np.where(m, maps[kname], 0.0)
        save_maps(os.path.join(sd, f'{S}.{hemi}.week1_maps.func.gii'), maps, hemi)

        def wq(x, q):
            o = np.argsort(x[m]); cw = np.cumsum(w[m][o]) / w[m].sum()
            return float(x[m][o][np.searchsorted(cw, q)])
        s = summ[hemi]
        for acq in H:
            s[f'lam_ideal_{acq}_q'] = [wq(maps[f'lam_ideal_{acq}'], q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
            for cfg in ['white_cnr1', 'white_cnr10', 'bold_cnr10']:
                s[f'lam_{acq}_{cfg}_q'] = [wq(maps[f'lam_{acq}_{cfg}'], q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
                s[f'dfs_hemi_{acq}_{cfg}'] = float((maps[f'dof_{acq}_{cfg}'][m] * ar[m] / 100).sum())
            s[f'leak_naive_{acq}_mean'] = float((maps[f'leak_naive_{acq}'] * w).sum() / w.sum())
            for cnr in [1, 10]:
                for tag in ('tau8', 'tau5', 'tau4'):
                    s[f'frac_unrec_{tag}_{acq}_cnr{cnr}'] = float((w * (maps[f'{tag}_{acq}_cnr{cnr}'] < 0.5)).sum() / w.sum())
        d = maps['d_opp']
        for thr in (3, 4, 5, 6):
            s[f'frac_dopp_lt{thr}'] = float((w * (d < thr)).sum() / w.sum())
        s['dopp_median'] = wq(d, 0.5)
        np.save(os.path.join(sd, f'{S}.{hemi}.normals_acpc.npy'), n.astype(np.float32))
    json.dump(summ, open(os.path.join(sd, f'{S}.week1_summary.json'), 'w'), indent=1)
    print(S, 'done', {h: (summ[h]['area_cm2_acpc'], round(summ[h]['pw_over_thick_acpc'], 3), round(summ[h]['frac_dopp_lt4'], 3)) for h in 'LR'})


if __name__ == '__main__':
    main()
