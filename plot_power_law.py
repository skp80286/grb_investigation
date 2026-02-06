import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.ticker as tck
import scienceplots

plt.style.use(['science', 'high-vis'])
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 5,  # minimum allowed by Nature
    "axes.titlesize": 6,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,  # embed fonts as TrueType
    "ps.fonttype": 42,
    # "figure.dpi": 300,  # ensure high-res bitmap export when needed
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.75,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
})
# ---------------- USER INPUT: set extinction (magnitudes) for r and R filters ----------------
# Subtract these from observed magnitudes to correct for extinction:
A_r = 0.117
A_R = 0.111
A_u = 0.217
A_g = 0.169
A_i = 0.087
A_J = 0.036
# ---------------------------------------------------------------------------------------------

# Read data
df = pd.read_csv("data/GRB250916A_AB_plaw.csv")
df.columns = df.columns.str.strip()  # remove whitespace in column names

# Boolean masks for filters
mask_R = df["Filter"] == "R"
mask_r = df["Filter"] == "r"
mask_L = df["Filter"] == "L"
mask_i = df["Filter"] == "i"
mask_J = df["Filter"] == "J"
mask_g = df["Filter"] == "g"
mask_u = df["Filter"] == "u"
mask_white = df["Filter"] == "white"
mask_vtr = df["Filter"] == "VT_R"
mask_vtb = df["Filter"] == "VT_B"

# combined mask for R + r used in your fit
mask_R_r = np.ma.mask_or(mask_R, mask_r)

# times and mags from dataframe
time = np.array(df["Time - t0 (hours)"])
mag = np.array(df["Magnitude"])
mag_err = np.array(df["Error"])

# X-ray file (unchanged)
df_xray = pd.read_csv("data/afterglow-xrt-density.csv")
flux10_x = np.array(df_xray["Flux_10"]) * 200  # already in mJy, scaled as you used
flux10_x_err = np.array(df_xray["Flux_10_err"]) * 200
time_x = np.array(df_xray["Times"]) / 3600  # seconds -> hours


# magnitude -> flux conversion (mJy)
def mag_to_flux(mag_arr, mag_err_arr):
    # flux zero point consistent with your original code (flux in mJy)
    flux = (10.0 ** (-0.4 * np.array(mag_arr))) * 3631000.0
    # propagate mag errors to flux errors (symmetric approximation)
    fluxerr = flux * (np.power(10.0, 0.4 * np.array(mag_err_arr)) - 1.0)
    return flux, fluxerr


def flux_to_mag(flux, flux_err):
    mag = -2.5 * np.log10(flux / 3631000.0)
    mag_err = (2.5 / np.log(10)) * (flux_err / flux)
    return mag, mag_err


# # ---------------- Apply extinction correction per-filter ----------------
# # copy observed mags so original data stays untouched
mag_corr = mag.copy()

# # subtract extinction (to correct for dimming, corrected_mag = observed_mag - A)
mag_corr[mask_r] = mag_corr[mask_r] - A_r
mag_corr[mask_R] = mag_corr[mask_R] - A_R
mag_corr[mask_u] = mag_corr[mask_u] - A_u
mag_corr[mask_g] = mag_corr[mask_g] - A_g
mag_corr[mask_i] = mag_corr[mask_i] - A_i
mag_corr[mask_J] = mag_corr[mask_J] - A_J

# convert corrected mags to fluxes (and errors)
flux_corr, flux_err_corr = mag_to_flux(mag_corr, mag_err)
# -------------------------------------------------------------------------

# # example predicted time/flux (kept from original script)
# time_pred = 194.0108333  # hours at 2025-09-24T15:30:00 (your value)
# pred_flux_r = 0.0015665723250779362
# pred_flux_xray = 3.1072508927165676e-07


# broken power law model
def broken_power_law(t, t_break, F_break, alpha_1, alpha_2):
    t = np.array(t)
    flux = np.where(
        t < t_break,
        F_break * (t / t_break) ** (-alpha_1),
        F_break * (t / t_break) ** (-alpha_2),
    )
    return flux


# initial guess and fit using corrected fluxes for R+r combined
initial_params = [30.0, np.median(flux_corr[mask_R_r]), 1.0, 2.0]
popt, pcov = curve_fit(
    broken_power_law,
    time[mask_R_r],
    flux_corr[mask_R_r],
    sigma=flux_err_corr[mask_R_r],
    p0=initial_params,
    absolute_sigma=True,  # interpret sigma as absolute errors (often better)
    maxfev=20000,
)

# fitted parameters and 1-sigma uncertainties from covariance matrix
t_break_fit, F_break_fit, alpha1_fit, alpha2_fit = popt
param_errors = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties
t_break_err, F_break_err, alpha1_err, alpha2_err = param_errors

print("Fitted parameters (with 1-sigma uncertainties):")
print(f"  t_break = {t_break_fit:.4f} +/- {t_break_err:.4f}  hours")
print(f"  F_break = {F_break_fit:.4e} +/- {F_break_err:.4e}  mJy")
print(f"  alpha_1 = {alpha1_fit:.4f} +/- {alpha1_err:.4f}")
print(f"  alpha_2 = {alpha2_fit:.4f} +/- {alpha2_err:.4f}")
print("\nCovariance matrix (pcov):\n", pcov)

# -------------------------------------------------------------------------
# Create fit curves for plotting
x_fitr = np.linspace(min(time[mask_R_r]) * 0.95, max(time[mask_R_r]) * 1.2, 300)
y_fitr = broken_power_law(x_fitr, *popt)

# x_fit_pred = np.linspace(max(time[mask_R_r]), time_pred + 10, 200)
# y_fit_pred = broken_power_law(x_fit_pred, *popt)

# some alternate curves that you used before (kept for display)
# x_fiti = np.linspace(min(time[mask_i]) * 0.95, max(time[mask_i]) * 1.05, 100)
# y_fiti = broken_power_law(x_fiti, popt[0], 0.02793296380431417, popt[2], popt[3])

# x_fit_x10 = np.linspace(min(time[mask_R_r]) * 0.95, max(time[mask_R_r]) * 1.05, 100)
# y_fit_x10 = broken_power_law(
#     x_fit_x10, popt[0], 4.536220981444408e-06, popt[2], popt[3]
# )

# -------------------------------------------------------------------------
# Plotting
fig, ax1 = plt.subplots(figsize=(10, 8))

# r band
ax1.errorbar(
    time[mask_r] / 24.0,
    flux_corr[mask_r],
    yerr=flux_err_corr[mask_r],
    fmt="o",
    label="r",
    capsize=4,
    color="tab:red",
)

# R band
ax1.errorbar(
    time[mask_R] / 24.0,
    flux_corr[mask_R],
    yerr=flux_err_corr[mask_R],
    fmt="o",
    label="R",
    capsize=4,
    color="magenta",
)

# u band
ax1.errorbar(
    time[mask_u] / 24.0,
    flux_corr[mask_u],
    yerr=flux_err_corr[mask_u],
    fmt="o",
    label="u",
    capsize=4,
    color="teal",
)

# g band
ax1.errorbar(
    time[mask_g] / 24.0,
    flux_corr[mask_g],
    yerr=flux_err_corr[mask_g],
    fmt="o",
    label="g",
    capsize=4,
    color="darkgreen",
)

# i band
ax1.errorbar(
    time[mask_i] / 24.0,
    flux_corr[mask_i],
    yerr=flux_err_corr[mask_i],
    fmt="o",
    label="i",
    capsize=4,
    color="darkgoldenrod",
)

# J band
ax1.errorbar(
    time[mask_J] / 24.0,
    flux_corr[mask_J],
    yerr=flux_err_corr[mask_J],
    fmt="o",
    label="J",
    capsize=4,
    color="olive",
)
ax1.errorbar(
    time[mask_L] / 24.0,
    flux_corr[mask_L],
    yerr=flux_err_corr[mask_L],
    fmt="o",
    mfc="none",
    label="L",
    capsize=4,
    color="saddlebrown",
)
ax1.errorbar(
    time[mask_vtr] / 24.0,
    flux_corr[mask_vtr],
    yerr=flux_err_corr[mask_vtr],
    fmt="o",
    label="VT_R",
    capsize=4,
    color="orange",
)
ax1.errorbar(
    time[mask_vtb] / 24.0,
    flux_corr[mask_vtb],
    yerr=flux_err_corr[mask_vtb],
    fmt="o",
    label="VT_B",
    capsize=4,
    color="royalblue",
)
ax1.errorbar(
    time[mask_white] / 24.0,
    flux_corr[mask_white],
    yerr=flux_err_corr[mask_white],
    fmt="o",
    mfc="none",
    label="white",
    capsize=4,
    color="dimgrey",
)
ax1.errorbar(
    time_x / 24.0,
    flux10_x,
    yerr=flux10_x_err,
    fmt="o",
    label="XRT 10 keV × 200",
    capsize=4,
    color="tab:purple",
)
# fitted model lines
ax1.plot(
    x_fitr / 24.0,
    y_fitr,
    linestyle="--",
    linewidth=2,
    alpha=0.8,
    color="tab:red",
    label="Broken PL fit (R+r filters)",
)

# axis labels and scales
ax1.set_xlabel("Time after trigger (days)", fontsize=18)
ax1.set_ylabel("Flux (mJy)", fontsize=18)
# ax1.set_title("Time vs Flux (extinction-corrected for r & R)", fontsize=16)
ax1.set_xscale("log")
ax1.set_yscale("log")

# vertical line and shaded band for t_break +/- 1-sigma
t_break_days = t_break_fit / 24.0
t_break_lo = max((t_break_fit - t_break_err) / 24.0, 1e-9)
t_break_hi = (t_break_fit + t_break_err) / 24.0

ax1.axvline(t_break_days, color="green", linestyle="--", linewidth=1.5)
ax1.axvspan(
    t_break_lo,
    t_break_hi,
    color="green",
    alpha=0.15,
    # label=r"$t_{\rm break} \pm 1\sigma$",
)

# annotate t_break and alphas with error bars shown
ax1.text(
    1.07 * t_break_days,
    ax1.get_ylim()[0] * 8,
    f"t_break = {t_break_fit:.2f} ± {t_break_err:.2f} h",
    color="green",
    fontsize=17,
    rotation=90,
    va="center",
    ha="left",
)

"""
# annotate alpha1 and alpha2 near the break (with +/-)
ax1.text(
    0.7 * t_break_days,
    max(y_fitr) * 1.5,
    r"$\alpha_1$ = {:.2f} $\pm$ {:.2f}".format(alpha1_fit, alpha1_err),
    color="blue",
    fontsize=17,
    ha="right",
    bbox=dict(facecolor="white", edgecolor="blue", boxstyle="round,pad=0.3"),
)

ax1.text(
    1.3 * t_break_days,
    min(y_fitr) * 1.6,
    r"$\alpha_2$ = {:.2f} $\pm$ {:.2f}".format(alpha2_fit, alpha2_err),
    color="blue",
    fontsize=17,
    ha="left",
    bbox=dict(facecolor="white", edgecolor="blue", boxstyle="round,pad=0.3"),
)
"""
# legend entries explaining symbol meaning
#ax1.plot([], [], "o", color="black", label="Filled: extinction-corrected")
#ax1.plot([], [], "o", mfc="none", color="black", label="Open: not extinction-corrected")

# Put "Broken PL fit (R+r filters)" at the bottom of the legend
handles, labels = ax1.get_legend_handles_labels()
broken_pl_label = "Broken PL fit (R+r filters)"
if broken_pl_label in labels:
    idx = labels.index(broken_pl_label)
    handles = handles[:idx] + handles[idx + 1 :] + [handles[idx]]
    labels = labels[:idx] + labels[idx + 1 :] + [labels[idx]]
ax1.legend(handles, labels, loc="lower left", fontsize=17)

# twin axes to show magnitudes and seconds like your original
ax2 = ax1.twinx()
ax3 = ax1.twiny()

y_limits = np.array(ax1.get_ylim())
x_limits = np.array(ax1.get_xlim())


def flux_to_mag_for_axis(flux):
    return -2.5 * np.log10(flux / 3631000.0)


max_mag = flux_to_mag_for_axis(y_limits[1])
min_mag = flux_to_mag_for_axis(y_limits[0])

min_time_sec = x_limits[0] * 3600.0 * 24.0
max_time_sec = x_limits[1] * 3600.0 * 24.0

ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()

ax1.tick_params(axis="both", which="major", labelsize=17)
ax2.tick_params(axis="y", which="major", labelsize=17)
ax3.tick_params(axis="x", which="major", labelsize=17)

ax2.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax2.set_ylim(min_mag, max_mag)
ax3.set_xlim(min_time_sec, max_time_sec)
ax3.set_xscale("log")
ax3.set_xlabel("Time after trigger (seconds)", fontsize=18)
ax2.set_ylabel("Apparent Magnitude", fontsize=18)

plt.tight_layout()
plt.savefig(f"output/multinest_powerlaw20_lf/power_law.png", format='png', bbox_inches='tight', dpi=300)
