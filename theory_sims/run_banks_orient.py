"""T3: two opposite sulcal banks. T4: cortical orientation distribution on fsaverage5 -> predicted gain."""
import numpy as np, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_sampling import *
import nibabel as nib
import nilearn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
res = {}


def two_bank(d, h, N, n=(0.3, 0.5, 0.81), T=2.5, cnr=10.0, fw=None, M=40, delta=0.75):
    sheets = [dict(n=n, offset=+d / 2, T=T, nd=6, delta=delta, M=M),
              dict(n=n, offset=-d / 2, T=T, nd=6, delta=delta, M=M)]
    W = axis_weights(N, N * h, kind='t2s', t2s_ms=45.0, esp_eff_ms=0.58, fov_pe_real=180.0)
    K = build_K(sheets, h=h, N=N, enc='2D', pe_axis=0, pe_W=W)
    J = M * M
    pri = spectrum('white_bold', fwhm_bold=fw) if fw else spectrum('white')
    C1 = cov_from_spectrum(pri, delta, M)
    C = np.zeros((2 * J, 2 * J)); C[:J, :J] = C1; C[J:, J:] = C1   # banks a priori independent (geodesically far)
    applyA, s, dfs = averaging_ops(K, C, cnr, None)
    qs = np.array([0.05, 0.3, 0.6, 0.9]); ths = np.linspace(0, np.pi, 6, endpoint=False)
    Es = plane_waves(sheets[0], qs, ths, R=9.0, nsheets=2, sign=+1)
    Ea = plane_waves(sheets[0], qs, ths, R=9.0, nsheets=2, sign=-1)
    tau_s = transfer(applyA, Es).reshape(len(qs), -1).mean(1)
    tau_a = transfer(applyA, Ea).reshape(len(qs), -1).mean(1)
    # leakage: put a pattern on bank A only, measure what appears on bank B
    EA = Es.copy(); EA[J:] = 0
    AEA = applyA(EA)
    leak = (np.einsum('ij,ij->j', Es[J:], AEA[J:]) / np.einsum('ij,ij->j', EA[:J], EA[:J])).reshape(len(qs), -1).mean(1)
    own = (np.einsum('ij,ij->j', EA[:J], AEA[:J]) / np.einsum('ij,ij->j', EA[:J], EA[:J])).reshape(len(qs), -1).mean(1)
    # naive "ribbon-style" weighted-average mapping: W = rownorm(K^T) -> A_naive = W K
    Wn = K.T / np.maximum(K.T.sum(1, keepdims=True), 1e-12)
    An = Wn @ K
    AnEA = An @ EA
    leak_n = (np.einsum('ij,ij->j', Es[J:], AnEA[J:]) / np.einsum('ij,ij->j', EA[:J], EA[:J])).reshape(len(qs), -1).mean(1)
    own_n = (np.einsum('ij,ij->j', EA[:J], AnEA[:J]) / np.einsum('ij,ij->j', EA[:J], EA[:J])).reshape(len(qs), -1).mean(1)
    return dict(qs=qs.tolist(), tau_sym=tau_s.tolist(), tau_anti=tau_a.tolist(),
                leak_opt=leak.tolist(), own_opt=own.tolist(), leak_naive=leak_n.tolist(), own_naive=own_n.tolist())


print('== T3: two-bank separability (2D-MS + T2*, CNR 10, white prior) ==', flush=True)
for h, N in [(2.0, 25), (3.0, 17), (4.0, 13)]:
    for d in [2.5, 3.0, 3.5, 4.5, 6.0]:
        t0 = time.time()
        r = two_bank(d, h, N)
        res[f'T3|h{h}|d{d}'] = r
        print(f'h={h} d={d}: tau_anti(q~0)={r["tau_anti"][0]:.2f} tau_sym={r["tau_sym"][0]:.2f} '
              f'leak_opt={r["leak_opt"][0]:+.3f} own_opt={r["own_opt"][0]:.2f} | '
              f'leak_naive={r["leak_naive"][0]:+.3f} own_naive={r["own_naive"][0]:.2f}  ({time.time()-t0:.0f}s)', flush=True)

# ---------------------------------------------------------------- T4: fsaverage5 orientation distribution
print('== T4: orientation distribution on fsaverage5 midthickness ==', flush=True)
D = os.path.join(os.path.dirname(nilearn.__file__), 'datasets', 'data', 'fsaverage5')


def load(fn):
    g = nib.load(os.path.join(D, fn))
    return g.darrays[0].data.astype(float), (g.darrays[1].data.astype(int) if len(g.darrays) > 1 else None)


def vertex_normals_areas(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fa = 0.5 * np.linalg.norm(fn, axis=1)
    vn = np.zeros_like(V); va = np.zeros(len(V))
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
        np.add.at(va, F[:, k], fa / 3)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    return vn, va


allN, allA, allT = [], [], []
for hemi in ['left', 'right']:
    Vw, Fw = load(f'white_{hemi}.gii.gz')
    Vp, Fp = load(f'pial_{hemi}.gii.gz')
    th, _ = load(f'thick_{hemi}.gii.gz')
    Vm = 0.5 * (Vw + Vp)
    vn, va = vertex_normals_areas(Vm, Fw)
    allN.append(vn); allA.append(va); allT.append(th)
Nrm = np.concatenate(allN); Ar = np.concatenate(allA); Th = np.concatenate(allT)
print('vertices', len(Nrm), 'midthickness area (cm2) = %.0f' % (Ar.sum() / 100))


def gain_stats(h):
    r = np.array([rho_ratio(n, h=h, nang=720)[0] for n in Nrm])
    # reference: isotropic inradius of an aligned sheet = pi / max(h_x, h_y) for axial cortex; use pi/max(h)
    ref = 1.0 / np.max(h)
    g = r / ref
    w = Ar / Ar.sum()
    order = np.argsort(g); cw = np.cumsum(w[order])
    q = lambda p: g[order][np.searchsorted(cw, p)]
    return dict(mean=float((g * w).sum()), q25=float(q(0.25)), median=float(q(0.5)), q75=float(q(0.75)),
                frac_ge_1p2=float(w[g >= 1.2].sum()), frac_ge_1p3=float(w[g >= 1.3].sum())), g


for tag, h in [('iso2', (2., 2., 2.)), ('aniso_3x3x4', (3., 3., 4.)), ('aniso_2x2x3', (2., 2., 3.))]:
    st, g = gain_stats(h)
    res[f'T4|{tag}'] = st
    np.save(os.path.join(OUT, f'gain_{tag}.npy'), g)
    print(tag, {k: round(v, 3) for k, v in st.items()})
np.save(os.path.join(OUT, 'fs5_normals.npy'), Nrm); np.save(os.path.join(OUT, 'fs5_areas.npy'), Ar)

json.dump(res, open(os.path.join(OUT, 'results_T3_T4.json'), 'w'))
print('saved', flush=True)
