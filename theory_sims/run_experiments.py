import numpy as np, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_sampling import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
res = {}

ORIENTS = {'aligned(0,0,1)': (0, 0, 1), 'tilt(1,1,0)': (1, 1, 0), 'diag(1,1,1)': (1, 1, 1),
           'oblique(.3,.5,.81)': (0.3, 0.5, 0.81), 'axial-tilt(0,.5,.87)': (0, 0.5, 0.866)}
qs = np.linspace(0.05, 3.2, 64)          # rad/mm  (wavelength 125mm .. 1.96mm)
ths = np.linspace(0, np.pi, 24, endpoint=False)

def run(tag, sheet_kw, enc_kw, prior, cnr, h=2.0, N=25):
    t0 = time.time()
    sheet = dict(offset=0.0, T=2.5, nd=6, delta=0.5, M=64); sheet.update(sheet_kw)
    K = build_K([sheet], h=h, N=N, **enc_kw)
    C = cov_from_spectrum(prior, sheet['delta'], sheet['M'])
    applyA, s, dfs = averaging_ops(K, C, cnr, None)
    E = plane_waves(sheet, qs, ths)
    tau = transfer(applyA, E)
    iso, best, per = resolvable_wavelengths(tau, qs, ths)
    area = sheet['M'] ** 2 * sheet['delta'] ** 2
    out = dict(iso_mm=float(iso), best_mm=float(best), dfs=float(dfs), dfs_per_cm2=float(dfs / area * 100),
               tau=tau.reshape(len(qs), len(ths)).tolist(), secs=time.time() - t0)
    print(f'{tag:70s} iso={iso:5.2f}mm best={best:5.2f}mm DFS/cm2={out["dfs_per_cm2"]:6.1f} ({out["secs"]:.0f}s)', flush=True)
    return out

white = spectrum('white')

# ---------------- T1: idealised check (3D Fourier, thin-ish sheet, white prior, very high CNR)
print('== T1: 3D Fourier encoding, white prior, CNR=1000 (idealised) ==')
for name, n in ORIENTS.items():
    res[f'T1|{name}'] = run(f'T1 3D-FT {name}', dict(n=n, T=1.0, nd=4), dict(enc='3D'), white, 1000)
    rmin, _ = rho_ratio(n)
    res[f'T1|{name}']['theory_iso_mm'] = float(2 * 2.0 / rmin)
    print('    theory iso wavelength = %.2f mm' % (2 * 2.0 / rmin))

# ---------------- T2: realistic acquisition factors at h=2 (HCP-3T-like)
# EPI T2* apodisation along PE: HCP 3T: PE FOV 180 mm, ESP 0.58 ms, no in-plane accel, T2*(GM,3T)~45 ms
W3T = axis_weights(25, 50.0, kind='t2s', t2s_ms=45.0, esp_eff_ms=0.58, fov_pe_real=180.0)
print('== T2: 2D multislice + T2* (HCP-3T-like), thickness 2.5 mm ==')
configs = {
    '3D-FT ideal':            dict(enc='3D'),
    '2D-MS ideal':            dict(enc='2D'),
    '2D-MS +T2*':             dict(enc='2D', pe_axis=0, pe_W=W3T),
}
for cname, enc in configs.items():
    for cnr in [1.0, 10.0]:
        for name in ['aligned(0,0,1)', 'diag(1,1,1)', 'oblique(.3,.5,.81)']:
            res[f'T2|{cname}|cnr{cnr}|{name}'] = run(f'T2 {cname} cnr={cnr} {name}',
                                                   dict(n=ORIENTS[name]), enc, white, cnr)

# registration jitter (0.5 mm) on top of 2D-MS + T2*
print('== T2b: + registration jitter sigma=0.5 mm ==')
Wj = axis_weights(25, 50.0, kind='t2s', t2s_ms=45.0, esp_eff_ms=0.58, fov_pe_real=180.0, jitter=0.5)
Wr = axis_weights(25, 50.0, jitter=0.5)
for cnr in [10.0]:
    for name in ['aligned(0,0,1)', 'diag(1,1,1)', 'oblique(.3,.5,.81)']:
        res[f'T2b|jit0.5|cnr{cnr}|{name}'] = run(f'T2b 2D-MS+T2*+jitter0.5 cnr={cnr} {name}',
                                               dict(n=ORIENTS[name]),
                                               dict(enc='2D', pe_axis=0, pe_W=Wj, ro_W=Wr, sl_jitter=0.5),
                                               white, cnr)

# ---------------- T2c: BOLD-PSF-limited prior (target = BOLD field with physiological blur)
print('== T2c: prior spectrum limited by BOLD PSF (FWHM 1.5 / 2.5 / 3.5 mm), 2D-MS + T2*, CNR 10 ==')
for fw in [1.5, 2.5, 3.5]:
    pri = spectrum('white_bold', fwhm_bold=fw)
    for name in ['aligned(0,0,1)', 'oblique(.3,.5,.81)']:
        res[f'T2c|bold{fw}|{name}'] = run(f'T2c bold FWHM={fw} {name}', dict(n=ORIENTS[name]),
                                          dict(enc='2D', pe_axis=0, pe_W=W3T), pri, 10.0)

# ---------------- voxel size series (legacy): h = 2, 3, 4 mm; 2D-MS + T2*, BOLD FWHM 2.5, CNR 10
print('== T2d: voxel size series h=2,3,4 mm (2D-MS+T2*, BOLD 2.5mm prior, CNR 10) ==')
pri = spectrum('white_bold', fwhm_bold=2.5)
for h, N in [(2.0, 25), (3.0, 17), (4.0, 13)]:
    FOV = N * h
    Wh = axis_weights(N, FOV, kind='t2s', t2s_ms=45.0, esp_eff_ms=0.58, fov_pe_real=180.0 * 2.0 / h * (h / 2.0))
    for name in ['aligned(0,0,1)', 'oblique(.3,.5,.81)']:
        res[f'T2d|h{h}|{name}'] = run(f'T2d h={h} {name}', dict(n=ORIENTS[name], delta=0.5, M=64),
                                      dict(enc='2D', pe_axis=0, pe_W=Wh), pri, 10.0, h=h, N=N)

json.dump(res, open(os.path.join(OUT, 'results_T1_T2.json'), 'w'))
print('saved')
