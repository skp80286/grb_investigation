import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from jetsimpy_plot import band_colors, filt_freqs
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import scienceplots



def freq_to_legend(nu_Hz):
    """Format frequency in appropriate units for legend."""
    if nu_Hz >= 1e15:
        return f"{nu_Hz/1e15:.2f} PHz"
    if nu_Hz >= 1e12:
        return f"{nu_Hz/1e12:.1f} THz"
    if nu_Hz >= 1e9:
        return f"{nu_Hz/1e9:.2f} GHz"
    if nu_Hz >= 1e6:
        return f"{nu_Hz/1e6:.2f} MHz"
    return f"{nu_Hz:.0f} Hz"


def parse_args():
    p = argparse.ArgumentParser(description="Fit afterglow light curve with broken power law(s).")
    p.add_argument(
        "--model", "-m",
        choices=["single", "double"],
        default="single",
        help="Use single broken power law (one break) or double broken power law (two breaks). Default: single.",
    )
    return p.parse_args()


# -----------------------------
# LOAD DATA
# -----------------------------
args = parse_args()
df = pd.read_csv("data/GRB260131A_AB.csv")  

fit_bands = ["r", "R"]
fit_df = df[df["band"].isin(fit_bands)]

t = fit_df["time_s"].values
m = fit_df["mag_ab"].values
err = fit_df["mag_err"].values

# -----------------------------
# MODEL
# -----------------------------
def broken_powerlaw_single(t, m0, a1, a2, tb):
    """Single broken power law: slope a1 before tb, a2 after."""
    return np.where(
        t < tb,
        m0 + a1 * np.log10(t),
        m0 + a1 * np.log10(tb) + a2 * (np.log10(t) - np.log10(tb)),
    )


def broken_powerlaw_double(t, m0, a1, a2, a3, tb1, tb2):
    """Double broken power law: slopes a1 (t<tb1), a2 (tb1<=t<tb2), a3 (t>=tb2)."""
    logt = np.log10(t)
    log_tb1 = np.log10(tb1)
    log_tb2 = np.log10(tb2)
    seg1 = m0 + a1 * logt
    seg2 = m0 + a1 * log_tb1 + a2 * (logt - log_tb1)
    seg3 = m0 + a1 * log_tb1 + a2 * (log_tb2 - log_tb1) + a3 * (logt - log_tb2)
    return np.where(t < tb1, seg1, np.where(t < tb2, seg2, seg3))


t_min, t_max = t.min(), t.max()

if args.model == "single":
    params, _ = curve_fit(
        broken_powerlaw_single, t, m,
        sigma=err, p0=[18, 1, 2, (t_min + t_max) / 2],
        bounds=([-np.inf, -np.inf, -np.inf, t_min],
                [np.inf, np.inf, np.inf, t_max]),
        maxfev=10000,
    )
    model_fn = broken_powerlaw_single
else:
    mid = np.sqrt(t_min * t_max)
    tb1_guess = np.sqrt(t_min * mid)
    tb2_guess = np.sqrt(mid * t_max)
    params, _ = curve_fit(
        broken_powerlaw_double, t, m,
        sigma=err, p0=[18, 1, 2, 1.5, tb1_guess, tb2_guess],
        bounds=([-np.inf, -np.inf, -np.inf, -np.inf, t_min, t_min],
                [np.inf, np.inf, np.inf, np.inf, t_max, t_max]),
        maxfev=10000,
    )
    # Enforce tb1 < tb2 via bounds would need reparametrization; rely on initial guess
    if params[4] >= params[5]:
        params[4], params[5] = min(params[4], params[5]), max(params[4], params[5])
    model_fn = broken_powerlaw_double

t_fit = np.logspace(np.log10(min(t)), np.log10(max(t)), 400)
m_fit = model_fn(t_fit, *params)
res = m - model_fn(t, *params)

# magnitude → flux in mJy (display only)
def mag_to_flux(m):
    return 1000 * 3631 * 10**(-0.4*m)

# -----------------------------
# PLOT
# -----------------------------

plt.style.use(['science', 'high-vis'])
FONT_SIZE = 10
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": FONT_SIZE,
    "axes.titlesize": 20,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    #"pdf.fonttype": 42,  # embed fonts as TrueType
    #"ps.fonttype": 42,
    # "figure.dpi": 300,  # ensure high-res bitmap export when needed
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.75,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    # Longer ticks (default ~3.5)
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
})
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True,
    figsize=(6,6),
    gridspec_kw={"height_ratios": [3,1], "hspace": 0}
)

def band_freq(band, sub):
    """Get frequency in Hz for a band (from data or filt_freqs)."""
    if "median_frequency_Hz" in sub.columns:
        return sub["median_frequency_Hz"].iloc[0]
    return filt_freqs.get(band)

bands_sorted = sorted(
    df["band"].unique(),
    key=lambda b: band_freq(b, df[df["band"] == b]) or 0
)
for band in bands_sorted:
    sub = df[df["band"] == band]
    nu = band_freq(band, sub)
    label = f"{band} ({freq_to_legend(nu)})" if nu is not None else band
    ax1.errorbar(
        sub["time_s"], sub["mag_ab"],
        yerr=sub["mag_err"],
        fmt='o', label=label,
        color=band_colors.get(band, "gray")
    )

ax1.plot(t_fit, m_fit, linewidth=2, label="fit")
ylim = ax1.get_ylim()

if args.model == "single":
    tb = params[3]
    ax1.axvline(tb, linestyle="--", linewidth=1.5)
    ax1.text(tb * 1.12, np.mean(ylim), f"$t_b$ = {tb/86400:.2f} d",
             rotation=90, va="center", ha="right", fontsize=FONT_SIZE)
    a1, a2 = params[1], params[2]
    t_a1 = np.exp(0.5 * (np.log(t_min) + np.log(tb)))
    t_a2 = np.exp(0.5 * (np.log(tb) + np.log(t_max)))
    ax1.text(t_a1, model_fn(t_a1, *params), f"$\\alpha_1$ = {a1:.2f}",
             fontsize=FONT_SIZE, ha="center", va="bottom")
    ax1.text(t_a2, model_fn(t_a2, *params), f"$\\alpha_2$ = {a2:.2f}",
             fontsize=FONT_SIZE, ha="center", va="top")
else:
    tb1, tb2 = params[4], params[5]
    a1, a2, a3 = params[1], params[2], params[3]
    ax1.axvline(tb1, linestyle="--", linewidth=1.5, color="C0")
    ax1.axvline(tb2, linestyle="--", linewidth=1.5, color="C1")
    ax1.text(tb1 * 1.12, np.mean(ylim), f"$t_{{b1}}$ = {tb1/86400:.2f} d",
             rotation=90, va="center", ha="right", fontsize=FONT_SIZE)
    ax1.text(tb2 * 1.12, np.mean(ylim), f"$t_{{b2}}$ = {tb2/86400:.2f} d",
             rotation=90, va="center", ha="right", fontsize=FONT_SIZE)
    t_a1 = np.exp(0.5 * (np.log(t_min) + np.log(tb1)))
    t_a2 = np.exp(0.5 * (np.log(tb1) + np.log(tb2)))
    t_a3 = np.exp(0.5 * (np.log(tb2) + np.log(t_max)))
    ax1.text(t_a1, model_fn(t_a1, *params), f"$\\alpha_1$ = {a1:.2f}",
             fontsize=FONT_SIZE, ha="center", va="bottom")
    ax1.text(t_a2, model_fn(t_a2, *params), f"$\\alpha_2$ = {a2:.2f}",
             fontsize=FONT_SIZE, ha="center", va="center")
    ax1.text(t_a3, model_fn(t_a3, *params), f"$\\alpha_3$ = {a3:.2f}",
             fontsize=FONT_SIZE, ha="center", va="top")

ax1.set_xscale("log")
ax1.set_xlim(t_min*0.9, t_max*1.1)
sf = ScalarFormatter()
sf.set_scientific(False)
ax1.xaxis.set_major_formatter(sf)
ax1.invert_yaxis()
ax1.set_ylabel("Magnitude (AB)\n")
ax1.legend()

# Right axis (display only)
ax1r = ax1.twinx()
ax1r.set_ylim(mag_to_flux(ax1.get_ylim()[0]),
              mag_to_flux(ax1.get_ylim()[1]))
ax1r.set_ylabel("Flux (mJy)")

# Top axis (display only): same range in days
xlim_sec = ax1.get_xlim()
ax1t = ax1.twiny()
ax1t.set_xscale("log")
ax1t.set_xlim(xlim_sec[0] / 86400, xlim_sec[1] / 86400)

# Tick positions in days (same logical positions as bottom, in days)
sec_ticks = ax1.get_xticks()
day_ticks = sec_ticks / 86400
# Only use ticks that fall within the top axis range
day_ticks = day_ticks[(day_ticks >= ax1t.get_xlim()[0]) & (day_ticks <= ax1t.get_xlim()[1])]
if len(day_ticks) == 0:
    day_ticks = np.logspace(np.log10(xlim_sec[0] / 86400), np.log10(xlim_sec[1] / 86400), 5)
ax1t.set_xticks(day_ticks)
ax1t.set_xticklabels([f"{d:.2f}" for d in day_ticks])
ax1t.set_xlabel("Time (days)")

# Residuals (sharex with ax1, so x ticks match)
ax2.axhline(0)
ax2.errorbar(t, res, yerr=err, fmt='o')
ax2.set_xscale("log")
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.set_xlabel("Time after trigger (s)")
ax2.set_ylabel("Residual")

plt.tight_layout()

# Force same serif font on all tick labels (all axes).
# Do this after tight_layout so labels exist and are not recreated.
mpl.rcParams["font.family"] = "serif"
TICK_FONT = "serif"
for ax in (ax1, ax1r, ax1t, ax2):
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    for label in ax.get_xticklabels(minor=False) + ax.get_yticklabels(minor=False):
        label.set_fontfamily(TICK_FONT)
        label.set_fontsize(FONT_SIZE)

plt.savefig(f"afterglow_lc_powerlaw_fit_{args.model}_break.png", dpi=300, bbox_inches="tight")
