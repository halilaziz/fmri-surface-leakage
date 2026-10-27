"""Week 2 library: effective-PSF kernels, sparse forward operators on real HCP geometry, k-space emulation of
coarser acquisitions, and parsing of Workbench ribbon-mapping weights.

Conventions
- World coordinates are MNI mm (HCP MNINonLinear space); grids are axis-aligned (diagonal affine), as for HCP MNI data.
- Voxel linear index = np.ravel_multi_index((i, j, k), shape) (C order over i, j, k).
- 1-D kernels have unit integral (1/mm), so a uniform unit field that fills space gives voxel value 1, and a
  quadrature point of volume w (mm^3) contributes w * kx * ky * kz to a voxel.
"""
import os, subprocess
import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
import nibabel as nib

W_ROOT = os.environ.get('GEO_ROOT', '/path/to/geo_root')
HCP = f'{W_ROOT}/data/hcp_extra'
DG = f'{W_ROOT}/data/hcp7t_rest_dgcon'
CONCAT = 'tfMRI_7T_RETCCW_AP_RETCW_PA_RETEXP_AP_RETCON_PA_RETBAR1_AP_RETBAR2_PA'

# HCP 7T fMRI acquisition (S1200 manual): 1.6 mm iso, PE along y (AP/PA), echo spacing 0.64 ms with iPAT 2,
# 130 PE lines over 208 mm. T2* of grey matter at 7T (~28 ms) and the resampling blur are modelling assumptions.
ACQ7T = dict(h=1.6, pe_axis=1, esp_eff_ms=0.32, fov_pe_mm=208.0, t2s_ms=28.0, slice_edge=0.15, taper=0.3, sigma_interp=0.4)
# Emulated coarser acquisitions: central k-space block of the 1.6 mm MNI data times a separable Tukey window (fraction 0.5,
# zero at the coarse Nyquist). A sharp crop would give sinc ringing that no compact operator can represent (truncation
# error > 10% at 5 voxels); with this window the coarse operators are accurate to ~1% at a 5-voxel support
# (effective FWHM ~1.6-1.7 x voxel instead of ~1.25 x for a sharp crop).
EMUL_TAPER = 0.5
SUPPORT_VOX = {"1.6": (4, 4, 3), "2.4": (5, 5, 5), "3.2": (5, 5, 5)}   # support radius (voxels, x y z) per grid; kernels roll off to 0 at its edge
ROLLOFF = {'native': 0.25, 'coarse': 0.15}   # kernels roll off to 0 over this last fraction of the support radius
# (native: gentler, column mass within ~1.5% across lattice positions; coarse: sharper, 3-D truncation error ~1%)


# ----------------------------------------------------------------------------- grids
def grid_from_nifti(path):
    img = nib.load(path)
    A = img.affine.copy()
    assert np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))), 'non-diagonal affine'
    return dict(shape=tuple(int(s) for s in img.shape[:3]), affine=A)


def coarse_grid(fine, target_mm):
    """Grid produced by cropping the central k-space block of `fine` to ~target_mm voxels (same FOV and first centre)."""
    n = np.array(fine['shape'])
    d = np.abs(np.diag(fine['affine'])[:3])
    n2 = np.floor(n * d / target_mm + 0.5).astype(int)
    A = fine['affine'].copy()
    A[:3, :3] = np.diag(np.sign(np.diag(fine['affine'])[:3]) * n * d / n2)
    return dict(shape=tuple(int(s) for s in n2), affine=A)


def voxel_sizes(grid):
    """Voxel edge lengths (mm) along the three grid axes; column norms, so oblique (rotated) grids work too."""
    return np.linalg.norm(grid['affine'][:3, :3], axis=0)


# ----------------------------------------------------------------------------- k-space emulation (operator S)
def _crop_axis(X, axis, n2):
    n = X.shape[axis]
    Xs = np.fft.fftshift(X, axes=axis)
    c = n // 2
    lo = c - n2 // 2
    sl = [slice(None)] * X.ndim
    sl[axis] = slice(lo, lo + n2)
    return np.fft.ifftshift(Xs[tuple(sl)], axes=axis)


def _window_dft(X, coarse_shape, taper):
    """Multiply a cropped spectrum (unshifted DFT order) by the separable Tukey window of the coarse grid."""
    if taper is None or taper == 0:
        return X
    for ax, n2 in enumerate(coarse_shape):
        f = np.abs(np.fft.fftfreq(n2)) / 0.5                         # |k| / k_Nyquist of the coarse grid
        w = _tukey(f, 1.0, taper)
        sh = [1] * X.ndim; sh[ax] = n2
        X = X * w.reshape(sh).astype(X.real.dtype)
    return X


def emulate(vol, coarse_shape, workers=1, taper=EMUL_TAPER):
    """k-space crop of a 3-D (or 4-D, frames last) volume to coarse_shape, times the emulation window; amplitude preserving."""
    X = sfft.fftn(vol, axes=(0, 1, 2), workers=workers)
    for ax in range(3):
        X = _crop_axis(X, ax, coarse_shape[ax])
    X = _window_dft(X, coarse_shape, taper)
    scale = np.prod([coarse_shape[a] / vol.shape[a] for a in range(3)])
    return np.real(sfft.ifftn(X, axes=(0, 1, 2), workers=workers)) * scale


def emulate_series(Y, shapes, workers=1, batch=32, taper=EMUL_TAPER):
    """Y (nx, ny, nz, T) -> {name: coarse series (float32, C order)}; one forward FFT per frame batch serves all grids."""
    T = Y.shape[3]
    out = {g: np.empty(tuple(s) + (T,), np.float32) for g, s in shapes.items()}
    for t0 in range(0, T, batch):
        X = sfft.fftn(np.asarray(Y[..., t0:t0 + batch], np.float32), axes=(0, 1, 2), workers=workers)
        for g, s in shapes.items():
            Z = X
            for ax in range(3):
                Z = _crop_axis(Z, ax, s[ax])
            Z = _window_dft(Z, s, taper)
            sc = np.prod([s[a] / Y.shape[a] for a in range(3)])
            out[g][..., t0:t0 + batch] = np.real(sfft.ifftn(Z, axes=(0, 1, 2), workers=workers)) * sc
    return out


# ----------------------------------------------------------------------------- 1-D effective PSF kernels
def _tukey(k, kmax, frac):
    """1 in |k| <= (1-frac) kmax, raised-cosine roll-off to 0 at kmax."""
    a = np.abs(k) / kmax
    out = np.ones_like(a)
    edge = (a > 1 - frac) & (a <= 1)
    out[edge] = 0.5 * (1 + np.cos(np.pi * (a[edge] - (1 - frac)) / frac))
    out[a > 1] = 0.0
    return out


def kernel_table(axis, scale, acq=ACQ7T, coarse_mm=None, box_mm=None, R_mm=8.0, dt=0.01, trunc_mm=None, native_trunc_mm=None,
                 rolloff=ROLLOFF['native']):
    """Tabulate the 1-D kernel along world axis `axis` (0 x, 1 y, 2 z) on t in [0, R_mm].
    scale = MNI/ACPC linear scale of that axis (acquisition voxel in MNI mm = acq['h'] * scale).
    box_mm: if given, return a box kernel of that width instead (used for the ribbon-geometry check).
    trunc_mm: if given, roll the kernel smoothly to 0 between (1-rolloff)*trunc_mm and trunc_mm and renormalise to unit integral
    (keeps a finite-support operator mass-conserving).
    coarse_mm (+ native_trunc_mm): kernel of the emulated acquisition = spectrum of the native kernel exactly as used on
    the native grid (truncated at native_trunc_mm) times the emulation window of a coarse_mm grid."""
    t = np.arange(0, R_mm + dt, dt)
    if box_mm is not None:
        return t, np.where(t < box_mm / 2, 1.0 / box_mm, 0.0)
    h = acq['h'] * scale
    kmax_big = 12 * np.pi / h
    k = np.linspace(-kmax_big, kmax_big, 24001)
    dk = k[1] - k[0]
    if coarse_mm is not None:
        tn, vn = kernel_table(axis, scale, acq, R_mm=native_trunc_mm + 0.5, dt=dt, trunc_mm=native_trunc_mm, rolloff=ROLLOFF['native'])
        Wk = (2 * np.cos(np.outer(k, tn[1:])) @ vn[1:] + vn[0]) * dt      # even kernel -> real spectrum
        Wk *= _tukey(k, np.pi / coarse_mm, EMUL_TAPER)
    elif axis == 2:   # slice direction: physical slice profile (erf-edged rect), interpolated between slices
        Wk = np.sinc(k * h / (2 * np.pi)) * np.exp(-0.5 * (acq['slice_edge'] * h * k) ** 2)
        Wk *= _tukey(k, 1.2 * np.pi / h, 0.35)
    else:             # Fourier-encoded axis with a mild taper; T2* decay along PE
        # acq['pf'] (week 15; default 1): partial Fourier along PE, approximated by the magnitude PSF of a zero-filled
        # reconstruction, whose main lobe is 1/pf wider: the sampled band is pf * kmax
        kmax = np.pi / h * (acq.get('pf', 1.0) if axis == acq['pe_axis'] else 1.0)
        Wk = _tukey(k, kmax, acq['taper'])
        if axis == acq['pe_axis']:
            lines = np.abs(k) * acq['fov_pe_mm'] * scale / (2 * np.pi)
            Wk *= np.exp(-lines * acq['esp_eff_ms'] / acq['t2s_ms'])
    if coarse_mm is None:
        Wk *= np.exp(-0.5 * (acq['sigma_interp'] * k) ** 2)
    Wk /= Wk[len(k) // 2]           # unit integral of the kernel  <=>  W(0) = 1
    vals = (np.cos(np.outer(t, k)) @ Wk) * dk / (2 * np.pi)
    if trunc_mm is not None:
        w = np.clip((trunc_mm - t) / (rolloff * trunc_mm), 0, 1)
        vals = vals * (0.5 - 0.5 * np.cos(np.pi * w))
        vals /= (2 * vals.sum() - vals[0]) * dt
    return t, vals


def eval_kernel(tab, x):
    t, v = tab
    return np.interp(np.abs(x), t, v, right=0.0)


def mni_acpc_scales(mni_mid, acpc_mid):
    """Per-axis linear scale MNI/ACPC from matched midthickness vertices (both hemispheres stacked)."""
    return np.std(mni_mid, axis=0) / np.std(acpc_mid, axis=0)


# ----------------------------------------------------------------------------- geometry
def load_surf(p):
    g = nib.load(p)
    return g.agg_data('NIFTI_INTENT_POINTSET').astype(np.float64), g.agg_data('NIFTI_INTENT_TRIANGLE').astype(np.int64)


def vertex_areas_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fa = 0.5 * np.linalg.norm(fn, axis=1)
    va = np.zeros(len(V)); vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(va, F[:, k], fa / 3)
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    return va, vn


def hemi_geometry(S, hemi):
    N = f'{HCP}/{S}/MNINonLinear/Native'
    wv, F = load_surf(f'{N}/{S}.{hemi}.white.native.surf.gii')
    pv, _ = load_surf(f'{N}/{S}.{hemi}.pial.native.surf.gii')
    mv, _ = load_surf(f'{N}/{S}.{hemi}.midthickness.native.surf.gii')
    roi = np.asarray(nib.load(f'{N}/{S}.{hemi}.atlasroi.native.shape.gii').agg_data()) > 0.5
    sph = load_surf(f'{N}/{S}.{hemi}.sphere.native.surf.gii')[0]
    return dict(white=wv, pial=pv, mid=mv, faces=F, roi=roi, sphere=sph)


def column_points(g, nd=5):
    """Quadrature points along each vertex's white->pial segment (MNI space). Returns points (P,3), volumes (P,),
    vertex index (P,), depth zeta (P,). Volume per vertex = midthickness area x normal thickness; medial wall dropped."""
    aw, _ = vertex_areas_normals(g['white'], g['faces'])
    ap, _ = vertex_areas_normals(g['pial'], g['faces'])
    am, nm = vertex_areas_normals(g['mid'], g['faces'])
    seg = g['pial'] - g['white']
    tn = np.abs(np.einsum('ij,ij->i', seg, nm))                    # thickness along the midthickness normal
    zeta = (np.arange(nd) + 0.5) / nd
    Az = (1 - zeta)[None, :] * aw[:, None] + zeta[None, :] * ap[:, None]
    wvol = (am * tn)[:, None] * Az / np.maximum(Az.sum(1, keepdims=True), 1e-12)
    keep = g['roi']
    vid = np.repeat(np.arange(len(aw)), nd).reshape(-1, nd)
    P = g['white'][:, None, :] + zeta[None, :, None] * seg[:, None, :]
    zz = np.broadcast_to(zeta, (len(aw), nd))
    m = np.repeat(keep, nd).reshape(-1, nd)
    return P[m], wvol[m], vid[m], zz[m]


def triangle_points(g, nd=5):
    """Points filling each triangle's prism (7 barycentric samples x nd depths), each assigned to all 3 vertices,
    mimicking Workbench's ribbon-constrained vertex polyhedra (1-ring). Returns points, volumes, triangle vertices."""
    B = np.array([[1/3, 1/3, 1/3], [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3], [1/2, 1/2, 0], [0, 1/2, 1/2], [1/2, 0, 1/2]])
    F = g['faces']
    Wt = np.einsum('bk,fkd->fbd', B, g['white'][F]); Pt = np.einsum('bk,fkd->fbd', B, g['pial'][F])
    zeta = (np.arange(nd) + 0.5) / nd
    pts = Wt[:, :, None, :] + zeta[None, None, :, None] * (Pt - Wt)[:, :, None, :]
    aw = 0.5 * np.linalg.norm(np.cross(g['white'][F[:, 1]] - g['white'][F[:, 0]], g['white'][F[:, 2]] - g['white'][F[:, 0]]), axis=1)
    ap = 0.5 * np.linalg.norm(np.cross(g['pial'][F[:, 1]] - g['pial'][F[:, 0]], g['pial'][F[:, 2]] - g['pial'][F[:, 0]]), axis=1)
    h = np.linalg.norm((Pt - Wt).mean(1), axis=1)
    vol = (aw + ap + np.sqrt(aw * ap)) / 3 * h / (len(B) * nd)      # frustum volume split over samples
    npf = len(B) * nd
    return pts.reshape(-1, 3), np.repeat(vol, npf), np.repeat(F, npf, axis=0)


# ----------------------------------------------------------------------------- sparse operators
def build_operator(points, vols, cols, ncols, grid, tabs, radius_vox, chunk_cols=4000):
    """K[v, c] = sum_{p: cols_p = c} vols_p * prod_d k_d(x_v,d - p_d). Columns must be grouped (sorted) so that a chunk
    of columns owns all its points; the result is assembled by stacking per-chunk CSC blocks."""
    shape = grid['shape']; A = grid['affine']
    Ainv = np.linalg.inv(A)
    ijk = points @ Ainv[:3, :3].T + Ainv[:3, 3]
    dvox = voxel_sizes(grid)
    order = np.argsort(cols, kind='stable')
    ijk, vols, cols = ijk[order], vols[order], cols[order]
    bounds = np.searchsorted(cols, np.arange(0, ncols + chunk_cols, chunk_cols))
    offs = [np.arange(-r, r + 1) for r in radius_vox]
    blocks = []
    nvox = int(np.prod(shape))
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        c0 = b * chunk_cols; c1 = min(ncols, c0 + chunk_cols)
        if hi <= lo:
            blocks.append(sp.csc_matrix((nvox, c1 - c0), dtype=np.float32)); continue
        q = ijk[lo:hi]
        base = np.rint(q).astype(np.int64)
        kv, ci = [], []
        for d in range(3):
            cand = base[:, d, None] + offs[d][None, :]
            kv.append(eval_kernel(tabs[d], (cand - q[:, d, None]) * dvox[d]))
            ci.append(cand)
        vals = (vols[lo:hi, None, None, None] * kv[0][:, :, None, None] * kv[1][:, None, :, None] * kv[2][:, None, None, :])
        I = np.broadcast_to(ci[0][:, :, None, None], vals.shape); J = np.broadcast_to(ci[1][:, None, :, None], vals.shape)
        Kz = np.broadcast_to(ci[2][:, None, None, :], vals.shape)
        ok = (I >= 0) & (I < shape[0]) & (J >= 0) & (J < shape[1]) & (Kz >= 0) & (Kz < shape[2]) & (vals != 0)
        rows = np.ravel_multi_index((I[ok], J[ok], Kz[ok]), shape)
        cc = np.broadcast_to((cols[lo:hi] - c0)[:, None, None, None], vals.shape)[ok]
        blocks.append(sp.csc_matrix((vals[ok].astype(np.float32), (rows, cc)), shape=(nvox, c1 - c0)))
    return sp.hstack(blocks, format='csc')


def box_ribbon_weights(points, vols, trivert, nverts, grid):
    """Vertex x voxel overlap volumes from triangle-prism samples (each sample counted for its 3 vertices)."""
    Ainv = np.linalg.inv(grid['affine'])
    ijk = np.rint(points @ Ainv[:3, :3].T + Ainv[:3, 3]).astype(np.int64)
    ok = ((ijk >= 0) & (ijk < np.array(grid['shape']))).all(1)
    lin = np.ravel_multi_index(tuple(ijk[ok].T), grid['shape'])
    rows = np.concatenate([trivert[ok, k] for k in range(3)])
    cols = np.tile(lin, 3)
    vals = np.tile(vols[ok], 3).astype(np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(nverts, int(np.prod(grid['shape']))))


def subject_grids(S):
    sbref = f'{HCP}/{S}/MNINonLinear/Results/{CONCAT}/{CONCAT}_SBRef.nii.gz'
    fine = grid_from_nifti(sbref)
    return sbref, {'1.6': fine, '2.4': coarse_grid(fine, 2.4), '3.2': coarse_grid(fine, 3.2)}


def subject_scales(S, geo, w1maps):
    acpc = np.vstack([load_surf(f'{w1maps}/{S}/{S}.{h}.midthickness.acpc.surf.gii')[0] for h in 'LR'])
    return mni_acpc_scales(np.vstack([geo[h]['mid'] for h in 'LR']), acpc)


def subject_points(geo, nd=5):
    """Column quadrature points of both hemispheres; operator columns are L vertices then R vertices."""
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    off = {'L': 0, 'R': nv['L']}
    P, V, C, Z = [], [], [], []
    for h in 'LR':
        p, w, vid, z = column_points(geo[h], nd)
        P.append(p); V.append(w); C.append(vid + off[h]); Z.append(z)
    return np.vstack(P), np.concatenate(V), np.concatenate(C), np.concatenate(Z), nv


def grid_tabs(g, grid, scales):
    """Kernel tables and support radius for grid g ('1.6' native; '2.4'/'3.2' k-space-cropped emulations)."""
    dv = voxel_sizes(grid); rad = SUPPORT_VOX[g]
    ntr = [(SUPPORT_VOX['1.6'][d] + 0.5) * ACQ7T['h'] for d in range(3)]      # native grid spacing = 1.6 mm (HCP 7T MNI)
    tabs = [kernel_table(d, scales[d], coarse_mm=(None if g == '1.6' else dv[d]), native_trunc_mm=ntr[d],
                         trunc_mm=(rad[d] + 0.5) * dv[d], R_mm=(rad[d] + 1) * dv[d],
                         rolloff=ROLLOFF['native' if g == '1.6' else 'coarse']) for d in range(3)]
    return tabs, rad


def depth_weight(z, comp):
    """Depth basis: b0 = 1 (column mean), b1 = sqrt(3) (2 zeta - 1) (white -> pial gradient, unit L2 norm on [0, 1])."""
    return np.ones_like(z) if comp == 'b0' else np.sqrt(3.0) * (2 * z - 1)


# ----------------------------------------------------------------------------- Workbench ribbon weights
WB_LOCK = os.environ.get('GEO_WB_LOCK', '/tmp/geo_model_res_wb.lock')   # node-local (host /tmp is shared by all containers on a node)


def wb_ribbon_weights(vol, mid, white, pial, out_txt, roi=None, retries=4):
    """Workbench ribbon-constrained mapping with the per-vertex weights written as text.
    On TRUBA nodes concurrent `-volume-to-surface-mapping` processes die with SIGBUS (reproduced 2026-09-27: alone always
    fine, 4 of 5 simultaneous calls crash), so calls are serialised per node with a lock file and retried."""
    import fcntl, random, time
    cmd = ['wb_command', '-volume-to-surface-mapping', vol, mid, out_txt + '.func.gii', '-ribbon-constrained', white, pial]
    if roi is not None:
        cmd += ['-volume-roi', roi]
    cmd += ['-output-weights-text', out_txt]
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get('WB_THREADS', os.environ.get('OMP_NUM_THREADS', '4')))
    for attempt in range(retries):
        if WB_LOCK == 'none':          # filesystems without flock (e.g. NFS without lockd): rely on the retries below
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
        else:
            with open(WB_LOCK, 'a') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                try:
                    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
                finally:
                    fcntl.flock(lk, fcntl.LOCK_UN)
        if r.returncode == 0:
            os.remove(out_txt + '.func.gii')
            return
        time.sleep(5 + 10 * random.random())
    raise RuntimeError(f'wb_command failed {retries} times (last exit {r.returncode}): {r.stderr.decode(errors="replace")[-400:]}')


def parse_weights_text(path, nverts, shape):
    """'vertex, n, i, j, k, w, ...' per line -> CSR (vertices x voxels), rows normalised to sum 1 (weighted mean)."""
    rows, cols, vals = [], [], []
    with open(path) as fh:
        for line in fh:
            a = np.array(line.replace(',', ' ').split(), dtype=np.float64)
            n = int(a[1])
            if n == 0:
                continue
            q = a[2:2 + 4 * n].reshape(n, 4)
            rows.append(np.full(n, int(a[0]))); cols.append(np.ravel_multi_index(tuple(q[:, :3].astype(int).T), shape)); vals.append(q[:, 3])
    Wm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nverts, int(np.prod(shape))))
    s = np.asarray(Wm.sum(1)).ravel()
    return sp.diags(np.where(s > 0, 1.0 / np.maximum(s, 1e-12), 0.0)) @ Wm


def save_nifti(data, grid, path):
    nib.save(nib.Nifti1Image(np.asarray(data, np.float32), grid['affine']), path)
