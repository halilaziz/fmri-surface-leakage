import numpy as np, json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_sampling import rho_ratio
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, 'figures')
os.chdir(RES)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C_AL, C_OB, C_DI = '#4C72B0', '#DD8452', '#55A868'

# ---------------- Fig A1: orientation law on the sphere + distribution over fsaverage5 cortex
lat = np.linspace(-89.5, 89.5, 90); lon = np.linspace(-179.5, 179.5, 180)
R = np.zeros((len(lat), len(lon)))
for i, la in enumerate(np.deg2rad(lat)):
    for j, lo in enumerate(np.deg2rad(lon)):
        n = (np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la))
        R[i, j] = rho_ratio(n, nang=360)[0]
A = np.load('fs5_areas.npy'); w = A/A.sum()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={'width_ratios': [1.5, 1, 1]})
im = ax[0].imshow(R, extent=[-180, 180, -90, 90], origin='lower', cmap='viridis', aspect='auto', vmin=1, vmax=np.sqrt(2))
ax[0].set_xlabel('normal azimuth (deg)'); ax[0].set_ylabel('normal elevation (deg)')
ax[0].set_title('Isotropic tangential bandwidth ρ(n) / (π/h)\n(3D Fourier box, thin columnar sheet)')
plt.colorbar(im, ax=ax[0], fraction=0.046)
for k, (tag, hmax, ref, lab) in enumerate([('iso2', 2.0, 4.0, '2 mm isotropic'), ('aniso_3x3x4', 4.0, 8.0, '3×3×4 mm (legacy)')]):
    g = np.load(f'gain_{tag}.npy'); ell = 2*hmax/g
    ax[k+1].hist(ell, bins=40, weights=w*100, color=C_OB, alpha=.85)
    ax[k+1].axvline(ref, color='k', ls='--', lw=1); ax[k+1].text(ref, ax[k+1].get_ylim()[1]*0.9, ' worst-case\n orientation', fontsize=7)
    med = ell[np.argsort(ell)][np.searchsorted(np.cumsum(w[np.argsort(ell)]), 0.5)]
    ax[k+1].axvline(med, color=C_AL, lw=1.5); ax[k+1].text(med, ax[k+1].get_ylim()[1]*0.6, f' median\n {med:.2f} mm', color=C_AL, fontsize=7)
    ax[k+1].set_xlabel('resolvable isotropic tangential wavelength (mm)'); ax[k+1].set_ylabel('% cortical area')
    ax[k+1].set_title(f'fsaverage5 cortex, {lab}')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'figA1_orientation_law.png'), dpi=180); plt.close()

# ---------------- Fig A2: transfer maps tau(q) aligned vs oblique
r = json.load(open('results_T1_T2.json'))
qs = np.linspace(0.05, 3.2, 64); ths = np.linspace(0, np.pi, 24, endpoint=False)
def polar_panel(axh, tau, title):
    tau = np.array(tau)
    TH = np.concatenate([ths, ths+np.pi, [2*np.pi]]); T = np.concatenate([tau, tau, tau[:, :1]], axis=1)
    Qg, THg = np.meshgrid(qs, TH, indexing='ij')
    pc = axh.pcolormesh(THg, Qg, T, cmap='magma', vmin=0, vmax=1, shading='auto')
    axh.contour(THg, Qg, T, levels=[0.5], colors='c', linewidths=1)
    for q0, ls in [(np.pi/2, ':'), (np.sqrt(2)*np.pi/2, '--')]:
        axh.plot(np.linspace(0, 2*np.pi, 200), np.full(200, q0), 'w', ls=ls, lw=0.8)
    axh.set_ylim(0, 3.2); axh.set_yticks([1, 2, 3]); axh.set_yticklabels(['1', '2', '3 rad/mm'], fontsize=6)
    axh.set_xticks([]); axh.set_title(title, fontsize=8)
    return pc
fig = plt.figure(figsize=(12, 3.3))
panels = [('T1|aligned(0,0,1)', 'Idealised, aligned'), ('T1|diag(1,1,1)', 'Idealised, oblique (1,1,1)'),
          ('T2|2D-MS +T2*|cnr1.0|aligned(0,0,1)', 'HCP-3T-like, aligned, CNR 1'), ('T2|2D-MS +T2*|cnr1.0|oblique(.3,.5,.81)', 'HCP-3T-like, oblique, CNR 1')]
for i, (key, title) in enumerate(panels):
    axh = fig.add_subplot(1, 4, i+1, projection='polar'); pc = polar_panel(axh, r[key]['tau'], title + f"\niso λ = {r[key]['iso_mm']:.2f} mm")
fig.colorbar(pc, ax=fig.axes, fraction=0.015, label='transfer τ(q)')
fig.text(0.01, 0.02, 'cyan: τ=0.5 contour; white dotted: voxel Nyquist π/h; white dashed: √2·π/h', fontsize=7)
plt.savefig(os.path.join(FIG, 'figA2_transfer_polar.png'), dpi=180, bbox_inches='tight'); plt.close()

# ---------------- Fig A3: voxel size and BOLD PSF dependence
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
hs = [2.0, 3.0, 4.0]
al = [r[f'T2d|h{h}|aligned(0,0,1)']['iso_mm'] for h in hs]; ob = [r[f'T2d|h{h}|oblique(.3,.5,.81)']['iso_mm'] for h in hs]
ax[0].plot(hs, al, 'o-', c=C_AL, label='grid-aligned sheet'); ax[0].plot(hs, ob, 's-', c=C_OB, label='oblique sheet')
ax[0].plot(hs, [2*h for h in hs], 'k:', lw=1, label='naive 2h')
ax[0].set_xlabel('voxel size h (mm)'); ax[0].set_ylabel('resolvable tangential wavelength (mm)')
ax[0].set_title('2D-MS + T2*, BOLD-PSF 2.5 mm prior, CNR 10'); ax[0].legend(frameon=False, fontsize=7)
fws = [1.5, 2.5, 3.5]
al = [r[f'T2c|bold{f}|aligned(0,0,1)']['iso_mm'] for f in fws]; ob = [r[f'T2c|bold{f}|oblique(.3,.5,.81)']['iso_mm'] for f in fws]
ax[1].plot(fws, al, 'o-', c=C_AL, label='aligned'); ax[1].plot(fws, ob, 's-', c=C_OB, label='oblique')
ax[1].set_xlabel('BOLD PSF FWHM of latent field (mm)'); ax[1].set_ylabel('resolvable wavelength (mm)')
ax[1].set_title('h = 2 mm: orientation gain shrinks as BOLD blur grows'); ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'figA3_voxel_bold.png'), dpi=180); plt.close()

# ---------------- Fig A4: two-bank separability
b = json.load(open('results_T3b.json')); qsb = np.array(b['qs']); lam = 2*np.pi/qsb
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
for k, cnr in enumerate([0.5, 2.0, 10.0]):
    for h, c in zip([2.0, 3.0, 4.0], [C_DI, C_OB, '#C44E52']):
        x = b['res'][f'cnr{cnr}|h{h}|d2.5']
        ax[k].plot(lam, x['tau_anti'], 'o-', c=c, label=f'optimal, h={h:g} mm')
        ax[k].plot(lam, x['own_naive'], 'x--', c=c, alpha=.6, label=f'naive own-bank, h={h:g}')
    ax[k].set_xscale('log'); ax[k].set_xticks([4, 8, 16, 32, 64, 125]); ax[k].set_xticklabels(['4', '8', '16', '32', '64', '125'])
    ax[k].set_xlabel('wavelength along bank (mm)'); ax[k].set_title(f'Touching banks (d = 2.5 mm), voxel CNR {cnr:g}')
ax[0].set_ylabel('bank-specific transfer'); ax[0].legend(frameon=False, fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'figA4_two_bank.png'), dpi=180); plt.close()

# leakage table (normalised)
rows = []
for cnr in [0.5, 2.0, 10.0]:
    for h in [2.0, 3.0, 4.0]:
        for d in [2.5, 3.5, 5.0]:
            x = b['res'][f'cnr{cnr}|h{h}|d{d}']
            rows.append((cnr, h, d, x['own_naive'][0], x['leak_naive'][0], x['own_opt'][0], x['leak_opt'][0], x['tau_anti'][2], x['tau_anti'][3]))
with open(os.path.join(RES, 'table_two_bank.md'), 'w') as f:
    f.write('| CNR | h (mm) | d (mm) | naive own | naive leak | optimal own | optimal leak | optimal bank-specific transfer @7.9 mm | @5.2 mm |\n|---|---|---|---|---|---|---|---|---|\n')
    for rr in rows:
        f.write('| %g | %g | %g | %.2f | %.3f | %.2f | %.3f | %.2f | %.2f |\n' % rr)
print('figures written')
