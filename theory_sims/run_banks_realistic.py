"""T3b: two-bank separability with a BOLD-limited prior (FWHM 2.5 mm) and realistic per-voxel CNR."""
import numpy as np, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_sampling import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
res = {}
qs = np.array([0.05, 0.4, 0.8, 1.2, 1.6]); ths = np.linspace(0, np.pi, 6, endpoint=False)

def two_bank(d, h, N, cnr, n=(0.3, 0.5, 0.81), T=2.5, fw=2.5, M=40, delta=0.75):
    sheets = [dict(n=n, offset=+d/2, T=T, nd=6, delta=delta, M=M), dict(n=n, offset=-d/2, T=T, nd=6, delta=delta, M=M)]
    W = axis_weights(N, N*h, kind='t2s', t2s_ms=45.0, esp_eff_ms=0.58, fov_pe_real=180.0)
    K = build_K(sheets, h=h, N=N, enc='2D', pe_axis=0, pe_W=W)
    J = M*M
    C1 = cov_from_spectrum(spectrum('white_bold', fwhm_bold=fw), delta, M)
    C = np.zeros((2*J, 2*J)); C[:J,:J] = C1; C[J:,J:] = C1
    applyA, s, dfs = averaging_ops(K, C, cnr, None)
    Es = plane_waves(sheets[0], qs, ths, R=9.0, nsheets=2, sign=+1)
    Ea = plane_waves(sheets[0], qs, ths, R=9.0, nsheets=2, sign=-1)
    EA = Es.copy(); EA[J:] = 0
    AEA = applyA(EA)
    nrm = np.einsum('ij,ij->j', EA[:J], EA[:J])
    f = lambda v: v.reshape(len(qs), -1).mean(1)
    out = dict(tau_sym=f(transfer(applyA, Es)), tau_anti=f(transfer(applyA, Ea)),
               own_opt=f(np.einsum('ij,ij->j', EA[:J], AEA[:J])/nrm), leak_opt=f(np.einsum('ij,ij->j', Es[J:], AEA[J:])/nrm))
    Wn = K.T / np.maximum(K.T.sum(1, keepdims=True), 1e-12); AnEA = Wn @ (K @ EA)
    out['own_naive'] = f(np.einsum('ij,ij->j', EA[:J], AnEA[:J])/nrm)
    out['leak_naive'] = f(np.einsum('ij,ij->j', Es[J:], AnEA[J:])/nrm)
    return {k: np.asarray(v).tolist() for k, v in out.items()}

print('wavelengths (mm):', np.round(2*np.pi/qs, 1), flush=True)
for cnr in [0.5, 2.0, 10.0]:
    for h, N in [(2.0, 25), (3.0, 17), (4.0, 13)]:
        for d in [2.5, 3.5, 5.0]:
            t0 = time.time(); r = two_bank(d, h, N, cnr); res[f'cnr{cnr}|h{h}|d{d}'] = r
            print(f'cnr={cnr:4} h={h} d={d}: tau_anti={np.round(r["tau_anti"],2)} own_opt={np.round(r["own_opt"],2)} '
                  f'leak_opt={np.round(r["leak_opt"],3)} | own_naive={np.round(r["own_naive"],2)} leak_naive={np.round(r["leak_naive"],3)} ({time.time()-t0:.0f}s)', flush=True)
json.dump(dict(qs=qs.tolist(), res=res), open(os.path.join(OUT, 'results_T3b.json'), 'w'))
print('saved', flush=True)
