"""Week 16, G2: the far criterion with true geodesic distances on the midthickness mesh against the area-scaled sphere proxy,
one subject, emulated kernels at 1.6 / 2.4 / 3.2 mm (operator only).

Geodesics: Dijkstra on the triangle graph of the native midthickness surface (edge lengths), from every ROI vertex, limited to
LIMIT mm; pairs beyond the limit count as far. The graph distance is calibrated against wb_command's exact geodesic (median ratio over
random source vertices) and the far label is also evaluated at the calibrated graph threshold. For every row of A = W K (vertex v, unknown u in the same hemisphere) the far
label is d_geo(v, u) > 15 mm; other-hemisphere unknowns are far under both criteria. Per vertex: leak fraction with the
geodesic criterion and with the sphere proxy; per subject: correlation and mean difference of the two maps, the share of
vertices whose > 5% / > 10% label changes, the share of operator mass whose far/near label flips, and the area-weighted
summaries of both.
Usage (xcp_d container): python3 geodesic_check.py SUBJECT OUTDIR
"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp, nibabel as nib
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_forward_operator'))
from w2lib import *

W2SUB = f'{W_ROOT}/results/week2/subjects'; W1MAPS = f'{W_ROOT}/results/week1/maps'
THR, LIMIT, BATCH = 15.0, 25.0, 400


def wmean(x, w):
    return float((x * w).sum() / max(w.sum(), 1e-30))


def mesh_graph(V, F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)          # each undirected edge once (two triangles share it; coo->csr would sum duplicates)
    d = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    G = sp.coo_matrix((np.concatenate([d, d]), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))), shape=(len(V), len(V)))
    return G.tocsr()


def main():
    S, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, f'{time.time() - T0:6.0f}s', *x, flush=True)
    sbref, grids = subject_grids(S)
    geo = {h: hemi_geometry(S, h) for h in 'LR'}; nv = {h: len(geo[h]['white']) for h in 'LR'}
    scales = subject_scales(S, geo, W1MAPS)
    area = {h: vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])[0] for h in 'LR'}
    P, V, C, Z, _ = subject_points(geo); ncols = nv['L'] + nv['R']
    su, fscale, Gm = {}, {}, {}
    for h in 'LR':
        s = geo[h]['sphere']; Rs = np.linalg.norm(s, axis=1).mean(); su[h] = s / np.linalg.norm(s, axis=1, keepdims=True)
        fscale[h] = Rs * np.sqrt(area[h][geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
        Gm[h] = mesh_graph(geo[h]['mid'], geo[h]['faces'])
    # calibration of the graph distance against Workbench's geodesic distance (exact on the mesh): 60 random ROI vertices per
    # hemisphere; the median ratio graph / exact over targets at 5-25 mm gives the graph distance that corresponds to 15 mm exact
    import subprocess, tempfile
    N = f'{HCP}/{S}/MNINonLinear/Native'; rng = np.random.default_rng(int(S)); ratios = []
    for h in 'LR':
        surf = f'{N}/{S}.{h}.midthickness.native.surf.gii'
        for v in rng.choice(np.flatnonzero(geo[h]['roi']), 60, replace=False):
            with tempfile.NamedTemporaryFile(suffix='.func.gii', delete=False) as tf:
                tmp = tf.name
            subprocess.run(['wb_command', '-surface-geodesic-distance', surf, str(int(v)), tmp, '-limit', str(LIMIT)], check=True, capture_output=True)
            dw = np.asarray(nib.load(tmp).darrays[0].data, float); os.remove(tmp)
            dg = dijkstra(Gm[h], directed=False, indices=[int(v)], limit=LIMIT)[0]
            ok = (dw > 5) & (dw < LIMIT) & np.isfinite(dg) & (dw >= 0)
            ratios.append(dg[ok] / dw[ok])
    ratios = np.concatenate(ratios); f_cal = float(np.median(ratios))
    THR_CAL = THR * f_cal
    log = {'subject': S, 'limit_mm': LIMIT, 'threshold_mm': THR, 'graph_over_exact_median': f_cal, 'graph_over_exact_iqr': [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
           'threshold_graph_calibrated_mm': THR_CAL, 'pairs': {}, 'grids': {}}
    say('graph / exact geodesic: median %.3f, IQR %.3f-%.3f -> graph threshold %.2f mm' % (f_cal, log['graph_over_exact_iqr'][0], log['graph_over_exact_iqr'][1], THR_CAL))
    # one grid at a time (memory): operator rows for both hemispheres, then a Dijkstra pass per hemisphere for that grid
    PB = [(5, 10), (10, 15), (15, 20), (20, 25)]
    for g in ('1.6', '2.4', '3.2'):
        tabs, rad = grid_tabs(g, grids[g], scales)
        K = build_operator(P, V * depth_weight(Z, 'b0'), C, ncols, grids[g], tabs, rad).tocsr()
        log['grids'][g] = {}
        for h in 'LR':
            Wm = sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr()
            idx = np.flatnonzero(geo[h]['roi']); has = np.diff(Wm.indptr)[idx] > 0
            A = (Wm[idx] @ K).tocsr(); del Wm
            off = 0 if h == 'L' else nv['L']
            a = dict(own_g=np.zeros(nv[h]), leak_g=np.zeros(nv[h]), own_s=np.zeros(nv[h]), leak_s=np.zeros(nv[h]), own_c=np.zeros(nv[h]), leak_c=np.zeros(nv[h]), gabs_g=np.zeros(nv[h]), gabs_s=np.zeros(nv[h]), abs_all=np.zeros(nv[h]), flip=0.0, mass=0.0)
            pb = {f'{lo}-{hi}': dict(n=0, sum_ratio=0.0, geo_far_sph_near=0, geo_near_sph_far=0) for lo, hi in PB}
            t0 = time.time()
            for b0 in range(0, len(idx), BATCH):
                src = idx[b0:b0 + BATCH]
                D = dijkstra(Gm[h], directed=False, indices=src, limit=LIMIT)       # (batch, nv): inf beyond the limit
                sub = A[b0:b0 + BATCH].tocoo(); r = sub.row; j = sub.col; val = sub.data.astype(np.float64)
                other = (j >= nv['L']) if h == 'L' else (j < nv['L']); jl = np.where(other, 0, j - off)
                dgeo = np.where(other, np.inf, D[r, jl]); del D
                dsph = fscale[h] * np.arccos(np.clip(np.einsum('nd,nd->n', su[h][src[r]], su[h][jl]), -1, 1))
                far_g = other | (dgeo > THR); far_s = other | (dsph > THR); far_c = other | (dgeo > THR_CAL)
                fin = np.isfinite(dgeo) & ~other
                for lo, hi in PB:
                    m = fin & (dsph >= lo) & (dsph < hi); k = f'{lo}-{hi}'
                    pb[k]['n'] += int(m.sum()); pb[k]['sum_ratio'] += float((dgeo[m] / dsph[m]).sum())
                    pb[k]['geo_far_sph_near'] += int((m & (dgeo > THR) & (dsph <= THR)).sum()); pb[k]['geo_near_sph_far'] += int((m & (dgeo <= THR) & (dsph > THR)).sum())
                a['own_c'][src] += np.bincount(r, val * ~far_c, minlength=len(src)); a['leak_c'][src] += np.bincount(r, val * far_c, minlength=len(src))
                a['own_g'][src] += np.bincount(r, val * ~far_g, minlength=len(src)); a['leak_g'][src] += np.bincount(r, val * far_g, minlength=len(src))
                a['own_s'][src] += np.bincount(r, val * ~far_s, minlength=len(src)); a['leak_s'][src] += np.bincount(r, val * far_s, minlength=len(src))
                a['flip'] += float(np.abs(val)[far_g != far_s].sum()); a['mass'] += float(np.abs(val).sum())
                av = np.abs(val); a['gabs_g'][src] += np.bincount(r, av * far_g, minlength=len(src)); a['gabs_s'][src] += np.bincount(r, av * far_s, minlength=len(src)); a['abs_all'][src] += np.bincount(r, av, minlength=len(src))
            del A
            w = np.zeros(nv[h]); w[idx] = area[h][idx] * has
            tg = a['own_g'] + a['leak_g']; ts = a['own_s'] + a['leak_s']; tc = a['own_c'] + a['leak_c']
            lfc = np.where(np.abs(tc) > 1e-6, a['leak_c'] / np.where(np.abs(tc) > 1e-6, tc, 1), 0)
            log['pairs'][f'{g}_{h}'] = {k: dict(n=v['n'], mean_ratio_geo_over_sph=(v['sum_ratio'] / v['n'] if v['n'] else None), geo_far_sph_near=v['geo_far_sph_near'], geo_near_sph_far=v['geo_near_sph_far']) for k, v in pb.items()}
            lfg = np.where(np.abs(tg) > 1e-6, a['leak_g'] / np.where(np.abs(tg) > 1e-6, tg, 1), 0)
            lfs = np.where(np.abs(ts) > 1e-6, a['leak_s'] / np.where(np.abs(ts) > 1e-6, ts, 1), 0)
            sel = w > 0
            gg = a['gabs_g'] / np.maximum(a['abs_all'], 1e-30); gs = a['gabs_s'] / np.maximum(a['abs_all'], 1e-30)
            log['grids'][g][h] = dict(
                leak_frac_mean_geo=wmean(lfg, w), leak_frac_mean_sph=wmean(lfs, w),
                gt5_geo=wmean((lfg > 0.05) * 1.0, w), gt5_sph=wmean((lfs > 0.05) * 1.0, w), gt10_geo=wmean((lfg > 0.10) * 1.0, w), gt10_sph=wmean((lfs > 0.10) * 1.0, w),
                corr=float(np.corrcoef(lfg[sel], lfs[sel])[0, 1]), mean_abs_diff=wmean(np.abs(lfg - lfs), w),
                label_change_gt5=wmean(((lfg > 0.05) != (lfs > 0.05)) * 1.0, w), label_change_gt10=wmean(((lfg > 0.10) != (lfs > 0.10)) * 1.0, w),
                mass_flip_share=a['flip'] / max(a['mass'], 1e-30),
                leak_frac_mean_cal=wmean(lfc, w), gt5_cal=wmean((lfc > 0.05) * 1.0, w), gt10_cal=wmean((lfc > 0.10) * 1.0, w),
                gross_far_mean_geo=wmean(gg, w), gross_far_mean_sph=wmean(gs, w), neg_share_geo=wmean((lfg < -0.01) * 1.0, w), neg_share_sph=wmean((lfs < -0.01) * 1.0, w),
                median_geo=float(np.median(lfg[sel])), median_sph=float(np.median(lfs[sel])), p90_geo=float(np.percentile(lfg[sel], 90)), p90_sph=float(np.percentile(lfs[sel], 90)))
            say(g, h, round(time.time() - t0), 's', {k: round(v, 4) for k, v in log['grids'][g][h].items()})
        del K
    log['t_total_s'] = round(time.time() - T0)
    json.dump(log, open(f'{outdir}/{S}.week16_geodesic_check.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
