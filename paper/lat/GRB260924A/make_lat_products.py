"""Figures, tables and derived numbers for the LAT subsection of the GRB 260924A paper.

Run in the fermi environment from the repository root after executing
GRB260924A_Fermi_LAT_analysis.ipynb:  python paper/lat/make_lat_products.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from astropy.io import fits
import astropy.units as u
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lat_grb_tools as L

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / 'results/GRB260924A'
WORK = ROOT / 'work/GRB260924A'
GBM_DIR = ROOT / 'data/GRB260924A/gbm'
FT1 = ROOT / 'data/GRB260924A/lat/lat_extended.fits'
FT2 = ROOT / 'data/GRB260924A/lat/lat_spacecraft.fits'

T0 = 811905208.561746
RA, DEC, Z = 133.76, 12.04, 3.51
EMIN, EMAX = 100., 100_000.
TS_MIN = 25.
THETA_MAX, ZMAX = 70., 100.
ROI_EARLY, ROI_FULL = 10., 15.
GBM_DETS = ['n3', 'n5', 'n4']
GBM_EBAND = (10., 1000.)
BKG_WINDOWS = [(-150., -15.), (420., 600.)]
BKG_ORDER = 3
MEV_TO_ERG = 1.602176634e-6

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5, 'xtick.direction': 'in',
                     'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})
C_A, C_B, C_S = '#1b9e77', '#d95f02', '#7570b3'

fitsdf = pd.read_csv(RES / 'lat_fits.csv')
episodes = pd.read_csv(RES / 'gbm_episodes.csv')
windows = pd.read_csv(RES / 'analysis_windows.csv').set_index('window')
A_START, A_STOP = windows.loc['A: initial period', ['tmin', 'tmax']]
B_START, B_STOP = windows.loc['B: full LAT window', ['tmin', 'tmax']]

row = lambda name, evclass='TRANSIENT020': fitsdf[(fitsdf.approach == name) & (fitsdf.evclass == evclass)].iloc[0]
A_full, A_pre, A_com = row('A: initial period'), row('A: before 15 deg good time'), row('A: common interval')
B_full, B_com = row('B: full LAT window'), row('B: common interval')
bins = fitsdf[fitsdf.approach == 'B: bin'].sort_values('tmin').reset_index(drop=True)
bins_src = fitsdf[fitsdf.approach == 'B: bin, SOURCE'].sort_values('tmin').reset_index(drop=True)
common = fitsdf[fitsdf.approach.isin(['A: common interval', 'B: common interval',
                                      '10 deg common interval', '15 deg common interval'])]

N = {}

# ---------------------------------------------------------------- index constancy
w = 1 / bins.index_err ** 2
mean_idx = np.sum(w * bins['index']) / w.sum()
mean_err = 1 / np.sqrt(w.sum())
chi2 = float(np.sum(w * (bins['index'] - mean_idx) ** 2))
dof = len(bins) - 1
N['index_const'] = dict(mean=mean_idx, err=mean_err, chi2=chi2, dof=dof, p=float(stats.chi2.sf(chi2, dof)))
dev = (bins['index'] - mean_idx) / bins.index_err
k = int(np.argmax(np.abs(dev)))
N['index_extreme'] = dict(tmin=bins.tmin[k], tmax=bins.tmax[k], index=bins['index'][k], err=bins.index_err[k],
                          sigma=float(dev[k]),
                          p_trials=float(1 - (1 - 2 * stats.norm.sf(abs(dev[k]))) ** len(bins)))
for label, sel in (('prompt', bins.tmax <= A_STOP + 1e-6), ('late', bins.tmin >= A_STOP - 1e-6)):
    b = bins[sel]
    ww = 1 / b.index_err ** 2
    N[f'index_{label}'] = dict(mean=float(np.sum(ww * b['index']) / ww.sum()), err=float(1 / np.sqrt(ww.sum())),
                               n=int(len(b)))

# ---------------------------------------------------------------- SOURCE vs TRANSIENT020 bins
pair = bins.merge(bins_src, on=['tmin', 'tmax'], suffixes=('', '_s'))
det = pair[pair.ts_s >= TS_MIN]
d_idx = (det.index_s - det['index']) / np.hypot(det.index_err_s, det.index_err)
N['source_check'] = dict(n_det=int(len(det)), n_ul=int((pair.ts_s < TS_MIN).sum()),
                         flux_ratio_median=float(np.median(det.flux_s / det.flux)),
                         max_abs_dindex_sigma=float(np.max(np.abs(d_idx))))

# ---------------------------------------------------------------- common interval comparison
cmp = []
for cls, g in common.groupby('evclass'):
    a, b = g[g.roi == ROI_EARLY].iloc[0], g[g.roi == ROI_FULL].iloc[0]
    d = a['index'] - b['index']
    cmp.append(dict(evclass=cls, a=a, b=b, d=d, naive=d / np.hypot(a.index_err, b.index_err),
                    nested=d / np.sqrt(abs(a.index_err ** 2 - b.index_err ** 2))))
N['common_cmp'] = [dict(evclass=c['evclass'], d=c['d'], naive=c['naive'], nested=c['nested'],
                        flux_ratio=c['a'].flux / c['b'].flux) for c in cmp]

# ---------------------------------------------------------------- fluence and k-corrected isotropic energy
dl = Planck18.luminosity_distance(Z).to(u.cm).value
N['dl_cm'] = dl


def kcorr(gamma, e1=EMIN, e2=EMAX, r1=EMIN, r2=EMAX, z=Z):
    """Observed [e1,e2] energy fluence -> rest-frame [r1,r2], for dN/dE ~ E^gamma."""
    s = gamma + 2
    integ = (lambda a, b: np.log(b / a)) if abs(s) < 1e-9 else (lambda a, b: (b ** s - a ** s) / s)
    return integ(r1 / (1 + z), r2 / (1 + z)) / integ(e1, e2)


energetics = []
for name, r in (('initial period (10 deg)', A_full), ('full LAT window (15 deg)', B_full)):
    dur = r.tmax - r.tmin
    S, S_err = r.eflux * dur, r.eflux_err * dur
    k = kcorr(r['index'])
    draws = np.random.default_rng(2).normal(r['index'], r.index_err, 20000)
    k_err = np.std([kcorr(g) for g in draws[:4000]])
    E = 4 * np.pi * dl ** 2 * S * k / (1 + Z)
    E_err = E * np.hypot(S_err / S, k_err / k)
    L_avg = E / (dur / (1 + Z))
    energetics.append(dict(name=name, tmin=r.tmin, tmax=r.tmax, eflux=r.eflux, eflux_err=r.eflux_err,
                           S=S, S_err=S_err, k=k, k_err=k_err, Eiso=E, Eiso_err=E_err, Liso=L_avg,
                           Liso_err=L_avg * E_err / E))
N['energetics'] = energetics

# ---------------------------------------------------------------- temporal decay
def decay_fit(df, col='flux'):
    t = np.sqrt(df.tmin * df.tmax)
    y = np.log10(df[col])
    sy = df[f'{col}_err'] / df[col] / np.log(10)
    coef, cov = np.polyfit(np.log10(t), y, 1, w=1 / sy, cov='unscaled')
    resid = (y - np.polyval(coef, np.log10(t))) / sy
    c2 = float(np.sum(resid ** 2))
    return dict(alpha=-coef[0], alpha_err=float(np.sqrt(cov[0, 0])), logA=coef[1], chi2=c2, dof=len(df) - 2,
                n=len(df), tmin=float(df.tmin.min()), tmax=float(df.tmax.max()),
                p=float(stats.chi2.sf(c2, len(df) - 2)))


def broken_decay_fit(df):
    """Continuous broken power law in log space; returns indices, break time and F-test against a single law."""
    lt = np.log10(np.sqrt(df.tmin * df.tmax).values)
    y = np.log10(df.flux.values)
    sy = (df.flux_err / df.flux / np.log(10)).values
    model = lambda q, x: q[0] - np.where(x < q[3], q[1], q[2]) * (x - q[3])
    res = optimize.least_squares(lambda q: (y - model(q, lt)) / sy, [-3.8, 1.8, 1.0, 2.3])
    cov = np.linalg.inv(res.jac.T @ res.jac)
    err = np.sqrt(np.diag(cov))
    c2, dof = float(np.sum(res.fun ** 2)), len(df) - 4
    single = decay_fit(df)
    F = ((single['chi2'] - c2) / 2) / (c2 / dof)
    return dict(logA=res.x[0], alpha1=res.x[1], alpha1_err=err[1], alpha2=res.x[2], alpha2_err=err[2],
                tb=10 ** res.x[3], tb_lo=10 ** (res.x[3] - err[3]), tb_hi=10 ** (res.x[3] + err[3]),
                chi2=c2, dof=dof, dchi2=single['chi2'] - c2, p_ftest=float(stats.f.sf(F, 2, dof)),
                tmin=float(df.tmin.min()), tmax=float(df.tmax.max()))


peak = int(np.argmax(bins.flux))
LAST_EP = float(episodes.t_hi.max())
N['peak_bin'] = dict(tmin=bins.tmin[peak], tmax=bins.tmax[peak], flux=bins.flux[peak], err=bins.flux_err[peak])
N['decay'] = {
    'from_peak': decay_fit(bins.iloc[peak:]),
    'after_ep1': decay_fit(bins[bins.tmin >= A_STOP - 1e-6]),
    'after_last_ep': decay_fit(bins[bins.tmin >= LAST_EP - 1e-6]),
    'from_peak_eflux': decay_fit(bins.iloc[peak:], 'eflux'),
    'before_ep1_end': decay_fit(bins[bins.tmax <= A_STOP + 1e-6]),
    'broken': broken_decay_fit(bins.iloc[peak:]),
}
idx = N['index_const']
beta = -idx['mean'] - 1
N['closure'] = dict(beta=beta, beta_err=idx['err'], p_above_c=2 * beta, p_above_c_err=2 * idx['err'],
                    alpha_pred_above_c=(3 * 2 * beta - 2) / 4, alpha_pred_above_c_err=1.5 * idx['err'],
                    p_slow=2 * beta + 1, alpha_pred_slow=3 * (2 * beta) / 4, alpha_pred_slow_err=1.5 * idx['err'])

# ---------------------------------------------------------------- photons
probs_file = next(WORK.glob('roi15_TRANSIENT020_P00070p04_P00972p81/probs.fits'))
ph = L.read_events(probs_file, RA, DEC, T0)
with fits.open(probs_file) as h:
    ph['p'] = np.asarray(h['EVENTS'].data['GRB']).astype(float)
probs_A = next(WORK.glob('roi10_TRANSIENT020_M00008p16_P00231p46/probs.fits'))
ph_A = L.read_events(probs_A, RA, DEC, T0)
with fits.open(probs_A) as h:
    ph_A['p'] = np.asarray(h['EVENTS'].data['GRB']).astype(float)
ph_A = ph_A[ph_A.t_rel < B_START]
hep = pd.read_csv(RES / 'high_energy_photons.csv').sort_values('energy', ascending=False)
assoc = hep[(hep[['p_grb_A', 'p_grb_B']].max(axis=1) > 0.9)]
N['photons'] = dict(n_p90_A_pre=int((ph_A.p > 0.9).sum()), n_p90_B=int((ph.p > 0.9).sum()), n_p90_gt1GeV_B=int(((ph.p > 0.9) & (ph.energy > 1e3)).sum()),
                    n_p90_gt10GeV_B=int(((ph.p > 0.9) & (ph.energy > 1e4)).sum()),
                    emax=float(assoc.energy.iloc[0]), t_emax=float(assoc.t_rel.iloc[0]),
                    p_emax=float(assoc.p_grb_B.iloc[0]), theta_emax=float(assoc.theta.iloc[0]),
                    sep_emax=float(assoc.sep.iloc[0]))
N['meta'] = dict(A_START=A_START, A_STOP=A_STOP, B_START=B_START, B_STOP=B_STOP, LAST_EP=LAST_EP,
                 episodes=episodes[['t_lo', 't_hi']].values.tolist())

(OUT / 'lat_numbers.json').write_text(json.dumps(N, indent=2, default=float))

# ================================================================= FIGURES
pointing = L.pointing_table(FT2, RA, DEC, T0)
mid = 0.5 * (pointing.start + pointing.stop)

# Figure 1: geometry
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot(mid, pointing.theta, 'k-', lw=1.2, label=r'off-axis angle $\theta$')
ax.plot(mid, pointing.zenith, 'k--', lw=1.2, label=r'zenith angle $\zeta$')
ax.axhline(ZMAX - ROI_EARLY, color=C_A, ls=':', lw=1, label=r'$\zeta_{\max}-10^\circ$')
ax.axhline(ZMAX - ROI_FULL, color=C_B, ls=':', lw=1, label=r'$\zeta_{\max}-15^\circ$')
ax.axhline(THETA_MAX, color='0.5', ls='-.', lw=.8, label=r'$\theta_{\max}=70^\circ$')
ax.axvspan(A_START, A_STOP, color=C_A, alpha=.15, lw=0)
ax.axvspan(B_START, B_STOP, color=C_B, alpha=.08, lw=0)
ax.set_xlim(-100, 1100)
ax.set_ylim(30, 100)
ax.set_xlabel(r'$t - T_0$ (s)')
ax.set_ylabel('Angle (deg)')
ax.set_ylim(30, 120)
ax.legend(loc='upper center', ncol=3, frameon=False, fontsize=6, columnspacing=.8, handlelength=1.6)
fig.savefig(FIG / 'lat_geometry.pdf')
plt.close(fig)

# GBM light curve for Figure 2
files = {d: GBM_DIR / f'glg_ctime_{d}_bn260924051_v00.pha' for d in GBM_DETS}
lcs = [L.gbm_ctime_lightcurve(files[d], T0, *GBM_EBAND, -300., 1000.) for d in GBM_DETS]
lc = L.sum_lightcurves(lcs)
gbkg = np.sum([L.fit_background(x, BKG_WINDOWS, BKG_ORDER) for x in lcs], axis=0)
blocks = L.gbm_blocks(lc, gbkg, p0=0.01)
lmid = 0.5 * (lc.t_lo + lc.t_hi)
brate = gbkg / lc.livetime

# Figure 2: multi-panel light curve
fig, axes = plt.subplots(4, 1, figsize=(7, 8.2), sharex=True, gridspec_kw=dict(hspace=0,
                                                                                height_ratios=[1, 1, 1.2, 1]))
ax = axes[0]
ax.step(lmid, (lc.counts / lc.livetime - brate) / 1e3, where='mid', color='0.6', lw=.5)
bmid = 0.5 * (blocks.t_lo + blocks.t_hi)
ax.hlines(blocks.net_rate / 1e3, blocks.t_lo, blocks.t_hi, color='k', lw=1.1)
ax.axhline(0, color='0.3', lw=.5)
ax.set_ylabel('GBM NaI\nrate (kcts s$^{-1}$)')
ax.text(.99, .9, 'n3+n4+n5, 10-1000 keV', ha='right', va='top', transform=ax.transAxes, fontsize=7)

ax = axes[1]
m = (ph.p > 0.9)
ax.scatter(ph.t_rel[~m], ph.energy[~m] / 1e3, s=4, c='0.75', lw=0, label=r'$p<0.9$')
sc = ax.scatter(ph.t_rel[m], ph.energy[m] / 1e3, s=9, c='k', lw=0, label=r'$p>0.9$')
mA = ph_A.p > 0.9
ax.scatter(ph_A.t_rel[~mA], ph_A.energy[~mA] / 1e3, s=4, c='0.75', lw=0)
ax.scatter(ph_A.t_rel[mA], ph_A.energy[mA] / 1e3, s=12, marker='s', facecolors='none', edgecolors=C_A, lw=.8,
           label=r'$p>0.9$, $10^\circ$, $t<t_{15}$')
ax.set_yscale('log')
ax.set_ylim(0.08, 80)
ax.set_ylabel('Photon energy\n(GeV)')
ax.legend(loc='upper right', frameon=False, ncol=3)

ax = axes[2]
def errpts(ax, df, color, marker, label, ms=3.5):
    d = df[df.ts >= TS_MIN]
    tc = 0.5 * (d.tmin + d.tmax)
    ax.errorbar(tc, d.flux * 1e4, xerr=[tc - d.tmin, d.tmax - tc], yerr=d.flux_err * 1e4, fmt=marker, ms=ms,
                color=color, mfc=color, elinewidth=.8, capsize=0, label=label)
    u_ = df[df.ts < TS_MIN]
    if len(u_):
        tc = 0.5 * (u_.tmin + u_.tmax)
        ax.errorbar(tc, u_.flux_ul95 * 1e4, xerr=[tc - u_.tmin, u_.tmax - tc], yerr=0.35 * u_.flux_ul95 * 1e4,
                    uplims=True, fmt='none', color=color, elinewidth=.8)
errpts(ax, bins, C_B, 'o', r'B: $15^\circ$ adaptive bins')
errpts(ax, pd.DataFrame([A_full, A_pre]), C_A, 's', r'A: $10^\circ$ initial period / before $t_{15}$')
ax.set_yscale('log')
ax.set_ylabel(r'$F_{0.1-100\,\mathrm{GeV}}$' + '\n' + r'($10^{-4}$ ph cm$^{-2}$ s$^{-1}$)')
ax.legend(loc='upper right', frameon=False)

ax = axes[3]
for df, color, marker in ((bins, C_B, 'o'), (pd.DataFrame([A_full]), C_A, 's')):
    tc = 0.5 * (df.tmin + df.tmax)
    ax.errorbar(tc, df['index'], xerr=[tc - df.tmin, df.tmax - tc], yerr=df.index_err, fmt=marker, ms=3.5,
                color=color, elinewidth=.8)
ic = N['index_const']
ax.axhspan(ic['mean'] - ic['err'], ic['mean'] + ic['err'], color=C_B, alpha=.2, lw=0)
ax.axhline(ic['mean'], color=C_B, lw=.8)
ax.set_ylabel(r'Photon index $\gamma$')
ax.set_ylim(-3.9, -0.9)
ax.set_xlabel(r'$t - T_0$ (s)')
for ax in axes:
    for _, ep in episodes.iterrows():
        ax.axvspan(ep.t_lo, ep.t_hi, color='0.85', alpha=.5, lw=0, zorder=0)
    ax.axvline(B_START, color=C_B, ls=':', lw=.8)
    ax.axvline(B_STOP, color=C_B, ls=':', lw=.8)
axes[0].set_xlim(-30, 1000)
for a, lab in zip(axes, 'abcd'):
    a.text(.01, .9, f'({lab})', transform=a.transAxes, fontsize=9, va='top')
fig.align_ylabels(axes)
fig.savefig(FIG / 'lat_lightcurve.pdf')
plt.close(fig)

# Figure 3: SEDs on the common interval
energies = np.logspace(np.log10(EMIN), np.log10(EMAX), 80)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), gridspec_kw=dict(wspace=.28))
ax = axes[0]
for r, color, lab in ((A_com, C_A, r'$10^\circ$'), (B_com, C_B, r'$15^\circ$')):
    med, lo, hi = L.powerlaw2_band(r.flux, r.flux_err, r['index'], r.index_err, r.corr_flux_index, EMIN, EMAX, energies)
    ax.fill_between(energies / 1e3, lo, hi, color=color, alpha=.3, lw=0)
    ax.plot(energies / 1e3, med, color=color, lw=1.2,
            label=lab + rf' TRANSIENT020: $\gamma={r["index"]:.2f}\pm{r.index_err:.2f}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Energy (GeV)')
ax.set_ylabel(r'$E^2\,dN/dE$ (erg cm$^{-2}$ s$^{-1}$)')
ax.set_ylim(top=2e-7)
ax.legend(frameon=False, loc='upper right', fontsize=6.5)
ax.text(.04, .06, '(a)', transform=ax.transAxes, ha='left', va='bottom')

ax = axes[1]
cls_order = ['TRANSIENT020E', 'TRANSIENT020', 'SOURCE']
for j, (roi, color, off) in enumerate(((ROI_EARLY, C_A, -.08), (ROI_FULL, C_B, .08))):
    g = common[common.roi == roi].set_index('evclass').loc[cls_order]
    ax.errorbar(np.arange(3) + off, g['index'], yerr=g.index_err, fmt='so'[j], color=color, ms=4,
                label=rf'${roi:g}^\circ$')
ax.set_xticks(range(3))
ax.set_xticklabels(['TRANSIENT\n020E', 'TRANSIENT\n020', 'SOURCE'], fontsize=7.5)
ax.set_ylabel(r'Photon index $\gamma$')
ax.legend(frameon=False)
ax.text(.97, .95, '(b)', transform=ax.transAxes, ha='right', va='top')
fig.savefig(FIG / 'lat_sed_common.pdf')
plt.close(fig)

# Figure 4: decay
fig, ax = plt.subplots(figsize=(3.4, 2.7))
d = bins
tg = np.sqrt(d.tmin * d.tmax)
ax.errorbar(tg, d.flux * 1e4, xerr=[tg - d.tmin, d.tmax - tg], yerr=d.flux_err * 1e4, fmt='o', ms=3, color=C_B,
            elinewidth=.8)
f = N['decay']['from_peak']
tt = np.logspace(np.log10(f['tmin']), np.log10(f['tmax']), 100)
ax.plot(tt, 10 ** f['logA'] * tt ** (-f['alpha']) * 1e4, 'k--', lw=.9,
        label=rf'single: $\alpha={f["alpha"]:.2f}\pm{f["alpha_err"]:.2f}$, $\chi^2/\nu={f["chi2"]:.1f}/{f["dof"]}$')
f = N['decay']['broken']
lt = np.log10(tt)
lb = np.log10(f['tb'])
ax.plot(tt, 10 ** (f['logA'] - np.where(lt < lb, f['alpha1'], f['alpha2']) * (lt - lb)) * 1e4, 'k-', lw=1.1,
        label=rf'broken: $\alpha_1={f["alpha1"]:.2f}$, $\alpha_2={f["alpha2"]:.2f}$, $\chi^2/\nu={f["chi2"]:.1f}/{f["dof"]}$')
ax.axvline(f['tb'], color='k', lw=.5, ls=':')
for _, ep in episodes.iterrows():
    ax.axvspan(max(ep.t_lo, 60), ep.t_hi, color='0.85', alpha=.5, lw=0, zorder=0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$t - T_0$ (s)')
ax.set_ylabel(r'$F_{0.1-100\,\mathrm{GeV}}$ ($10^{-4}$ ph cm$^{-2}$ s$^{-1}$)')
ax.legend(frameon=False, fontsize=6.5, loc='lower left')
fig.savefig(FIG / 'lat_decay.pdf')
plt.close(fig)

# ================================================================= TABLES
def pm(v, e, nd=2):
    return rf'${v:.{nd}f}\pm{e:.{nd}f}$'


def trange(a, b):
    return rf'${a:.1f}$--${b:.1f}$'


def sci(v, e, exp):
    return rf'${v / 10 ** exp:.2f}\pm{e / 10 ** exp:.2f}$'


lines = []
for i, (r, s) in enumerate(zip(bins.itertuples(), pair.itertuples())):
    src = (pm(s.index_s, s.index_err_s) if s.ts_s >= TS_MIN else r'\ldots') + f' & {s.ts_s:.0f}'
    lines.append(rf'{i + 1} & {trange(r.tmin, r.tmax)} & {r.n_events} & {r.ts:.0f} & '
                 rf'{pm(r.index, r.index_err)} & {sci(r.flux, r.flux_err, -5)} & {sci(r.eflux, r.eflux_err, -8)} & {src} \\')
(OUT / 'tab_lat_bins.tex').write_text('\n'.join(lines) + '\n\\hline\n')

lines = []
for name, r in (('A', A_full), ('A', A_pre), ('A', A_com), ('B', B_com), ('B', B_full)):
    flux = sci(r.flux, r.flux_err, -5) if r.ts >= TS_MIN else rf'$<{r.flux_ul95 / 1e-5:.2f}$'
    idx = pm(r['index'], r.index_err) if r.ts >= TS_MIN else rf'$-2$ (fixed)'
    eflux = sci(r.eflux, r.eflux_err, -8) if r.ts >= TS_MIN else r'\ldots'
    lines.append(rf'{name} & ${r.roi:.0f}^\circ$ & {trange(r.tmin, r.tmax)} & {r.n_events} & {r.npred:.0f} & '
                 rf'{r.ts:.0f} & {idx} & {flux} & {eflux} \\')
(OUT / 'tab_lat_windows.tex').write_text('\n'.join(lines) + '\n\\hline\n')

lines = []
for c in sorted(cmp, key=lambda c: cls_order.index(c['evclass'])):
    a, b = c['a'], c['b']
    lines.append(rf'\texttt{{{c["evclass"]}}} & {a.n_events} / {b.n_events} & {a.ts:.0f} / {b.ts:.0f} & '
                 rf'{pm(a["index"], a.index_err)} & {pm(b["index"], b.index_err)} & '
                 rf'${c["naive"]:+.2f}$ / ${c["nested"]:+.2f}$ & {a.flux / b.flux:.3f} \\')
(OUT / 'tab_lat_common.tex').write_text('\n'.join(lines) + '\n\\hline\n')

lines = []
for r in assoc.head(10).itertuples():
    pa = f'{r.p_grb_A:.3f}' if np.isfinite(r.p_grb_A) else r'\ldots'
    lines.append(rf'{r.t_rel:.2f} & {r.energy / 1e3:.1f} & {r.source_frame_GeV:.0f} & {r.sep:.2f} & '
                 rf'{r.theta:.1f} & {pa} & {r.p_grb_B:.3f} \\')
(OUT / 'tab_lat_photons.tex').write_text('\n'.join(lines) + '\n\\hline\n')

print(json.dumps(N, indent=1, default=lambda x: round(float(x), 4)))
