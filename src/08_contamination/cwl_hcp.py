"""Week 8, plan A, HCP: contrast-weighted leak (CWL) of HCP standard mapping at the emulated 2.4 / 3.2 mm grids, for one week-3
test subject. Operator and unknowns exactly as week-3 run_subject.py (the model's world); coarse-grid ribbon weights from week 2
(hcpW_{grid}_{hemi}_all). Response estimates (complex wedge maps, real and imaginary parts as two maps), per run r:
  CWL-M  run r's own 1.6 mm HCP map ('z16_{r}' of the week-3 estimates)
  CWL-P  run r's coarse data upsampled in k-space and mapped at 1.6 mm ('up_{grid}_{r}')
The prediction from run r is scored against the other run; summarize_cwl.py averages the two runs' CWL per vertex.
Usage (xcp_d container, TRUBA): python3 cwl_hcp.py SUBJECT [--plan-sha1 SHA]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '03_hcp_emulation')); sys.path.insert(0, HERE)
from w3lib import *                     # (inserts week2 first on sys.path: week2 and week3 both have a run_subject.py)
import nibabel as nib
from cwl_lib import sphere_frames, far_contamination, cwl, split_complex

W1MAPS = f'{W_ROOT}/results/week1/maps'          # as week-3 run_subject.py
W2SUB = f'{W_ROOT}/results/week2/subjects'
R3 = f'{W_ROOT}/results/week3'
OUT = f'{W_ROOT}/results/week8/cwl_hcp'


def gifti_maps(p):
    return {d.meta['Name']: np.asarray(d.data, np.float64) for d in nib.load(p).darrays}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--plan-sha1', default='')
    ap.add_argument('--stimulus', default='wedge', choices=sorted(STIMULI))          # week 13: ring runs with the same code
    ap.add_argument('--est-dir', default=f'{R3}/test_all', help='directory of the week-3 estimate files')
    ap.add_argument('--out-dir', default=OUT)
    a = ap.parse_args(); S = a.subject
    RUNS, _, _ = stim_runs(a.stimulus); R0 = list(RUNS)[0]; out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, f'{time.time() - T0:6.0f}s', *x, flush=True)
    E = np.load(f'{a.est_dir}/{S}/{S}.week3_estimates.npz')
    sbref, grids = subject_grids(S)
    geo = {h: hemi_geometry(S, h) for h in 'LR'}
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    scales = subject_scales(S, geo, W1MAPS)
    av = analysis_vertices(geo)
    idx = np.concatenate([np.flatnonzero(av['L']), nv['L'] + np.flatnonzero(av['R'])])
    assert np.array_equal(idx, E['idx']), 'unknowns differ from week 3'
    loc = -np.ones(nv['L'] + nv['R'], int); loc[idx] = np.arange(len(idx))
    P, V, C, Z, _ = subject_points(geo)
    keep = loc[C] >= 0; P, V, Cc = P[keep], V[keep], loc[C[keep]]
    n = len(idx)
    cat = lambda d: np.concatenate([d['L'], d['R']])[idx]
    wk2 = {h: {**gifti_maps(f"{W2SUB}/{S}/{S}.{h}.{STIMULI[a.stimulus]['maps']}.func.gii"), **gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_leakage.func.gii')}
           for h in 'LR'}
    coherent = cat({h: wk2[h]['coherence_1.6_all'] for h in 'LR'}) > COH_THR
    su, fs = sphere_frames(geo, {h: geo[h]['sphere'] for h in 'LR'}, lambda Vv, F: vertex_areas_normals(Vv, F)[0])
    out = dict(idx=idx.astype(np.int32), plan_sha1=a.plan_sha1, coherent=coherent, stimulus=a.stimulus)
    for g in ('2.4', '3.2'):
        t0 = time.time(); tabs, rad = grid_tabs(g, grids[g], scales)
        K = build_operator(P, V, Cc, n, grids[g], tabs, rad).tocsr()
        W = {h: sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr() for h in 'LR'}
        say(g, 'operator', round(time.time() - t0), 's, nnz', K.nnz)
        Xs = {}
        for r in RUNS:
            Xs[f'M_{r}'] = split_complex(E[f'z16_{r}']); Xs[f'P_{r}'] = split_complex(E[f'up_{g}_{r}'])
        Cn, far_chk = far_contamination(W, K, idx, nv['L'], su, fs, Xs)
        lf2 = cat({h: wk2[h][f'leak_fraction_{g}_all'] for h in 'LR'})
        out[f'far_{g}'] = far_chk.astype(np.float32)
        out[f'far_check_{g}'] = np.array([float(np.median(np.abs(far_chk - lf2)[coherent])), float(np.corrcoef(far_chk[coherent], lf2[coherent])[0, 1])])
        for r in RUNS:
            for t in 'MP':
                out[f'cwl{t}_{g}_{r}'] = cwl(Cn[f'{t}_{r}'], Xs[f'{t}_{r}'], coherent).astype(np.float32)
        say(g, 'done; far vs week-2 leak (median |dev|, r):', np.round(out[f'far_check_{g}'], 4),
            '; median CWL-M / CWL-P (coherent, first run):', round(float(np.median(out[f'cwlM_{g}_{R0}'][coherent])), 4),
            round(float(np.median(out[f'cwlP_{g}_{R0}'][coherent])), 4))
        del K, W, Cn
    np.savez_compressed(f'{out_dir}/{S}.cwl_hcp.npz', **out)
    say('saved')


if __name__ == '__main__':
    main()
