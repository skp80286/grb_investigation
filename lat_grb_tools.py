"""Helpers for the Fermi-LAT analysis of a GRB with the Fermitools.

`setup_fermitools_env` must run before anything imports pyLikelihood, because
the likelihood library reads CALDB and the Fermitools data paths when it loads.
A Jupyter kernel started from a kernelspec does not run the conda activation
scripts that normally set those variables.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import bayesian_blocks
import astropy.units as u

GTI_FILTER = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'

EVCLASS_IRF = {
    8: 'P8R3_TRANSIENT020E_V3',
    16: 'P8R3_TRANSIENT020_V3',
    128: 'P8R3_SOURCE_V3',
}
EVCLASS_ISO = {
    8: 'iso_P8R3_TRANSIENT020E_V3_v1.txt',
    16: 'iso_P8R3_TRANSIENT020_V3_v1.txt',
    128: 'iso_P8R3_SOURCE_V3_v1.txt',
}
EVCLASS_NAME = {8: 'TRANSIENT020E', 16: 'TRANSIENT020', 128: 'SOURCE'}


# ---------------------------------------------------------------------------
# Environment and tool execution
# ---------------------------------------------------------------------------

def setup_fermitools_env(prefix: str | Path | None = None, pfiles_dir: str | Path | None = None) -> Path:
    """Set the variables that `conda activate` would set for the Fermitools."""
    prefix = Path(prefix or sys.prefix)
    inst = prefix / 'share' / 'fermitools'
    if not (inst / 'syspfiles').is_dir():
        raise RuntimeError(f'No Fermitools installation under {prefix}; run this kernel in the fermi environment')
    caldb = inst / 'data' / 'caldb'
    values = {
        'INST_DIR': inst,
        'FERMI_DIR': inst,
        'FERMI_INST_DIR': inst,
        'BASE_DIR': inst,
        'EXTFILESSYS': inst / 'refdata' / 'fermi',
        'GENERICSOURCESDATAPATH': inst / 'data' / 'genericSources',
        'TIMING_DIR': inst / 'refdata' / 'fermi' / 'jplephem',
        'CALDB': caldb,
        'CALDBROOT': caldb,
        'CALDBCONFIG': caldb / 'software' / 'tools' / 'caldb.config',
        'CALDBALIAS': caldb / 'software' / 'tools' / 'alias_config.fits',
    }
    for key, value in values.items():
        os.environ[key] = str(value)
    pfiles_dir = Path(pfiles_dir or Path.home() / 'pfiles')
    pfiles_dir.mkdir(parents=True, exist_ok=True)
    os.environ['PFILES'] = f'{pfiles_dir};{inst / "syspfiles"}'
    bindir = str(prefix / 'bin')
    if bindir not in os.environ.get('PATH', '').split(os.pathsep):
        os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')
    return inst


def galdiffuse_dir() -> Path:
    return Path(os.environ['FERMI_DIR']) / 'refdata' / 'fermi' / 'galdiffuse'


def run_tool(tool: str, pfiles_dir: Path | None = None, **pars) -> str:
    """Run one Fermitools command line tool; raise with its output on failure."""
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(f'{tool} not found on PATH')
    args = [exe] + [f'{k}={v}' for k, v in pars.items()]
    env = os.environ.copy()
    if pfiles_dir is not None:
        pfiles_dir.mkdir(parents=True, exist_ok=True)
        env['PFILES'] = f'{pfiles_dir};{Path(env["FERMI_DIR"]) / "syspfiles"}'
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f'{tool} failed ({proc.returncode})\n$ {" ".join(args)}\n{proc.stdout}\n{proc.stderr}')
    return proc.stdout


# ---------------------------------------------------------------------------
# Reading LAT files
# ---------------------------------------------------------------------------

def decode_bits(col: np.ndarray) -> np.ndarray:
    """EVENT_CLASS / EVENT_TYPE are stored as 32X bit arrays, most significant bit first."""
    col = np.asarray(col)
    if col.ndim == 1:
        return col.astype(np.int64)
    out = np.zeros(len(col), dtype=np.int64)
    for i in range(col.shape[1]):
        out |= col[:, -1 - i].astype(np.int64) << i
    return out


def read_events(path: str | Path, ra: float, dec: float, t0: float) -> pd.DataFrame:
    with fits.open(path, memmap=True) as h:
        d = h['EVENTS'].data
        out = {
            't_rel': np.asarray(d['TIME'], float) - t0,
            'energy': np.asarray(d['ENERGY'], float),
            'ra': np.asarray(d['RA'], float),
            'dec': np.asarray(d['DEC'], float),
            'theta': np.asarray(d['THETA'], float),
            'zenith': np.asarray(d['ZENITH_ANGLE'], float),
            'event_id': np.asarray(d['EVENT_ID']).astype(np.int64),
            'run_id': np.asarray(d['RUN_ID']).astype(np.int64),
        }
        if 'EVENT_CLASS' in d.names:
            out['event_class'] = decode_bits(d['EVENT_CLASS'])
        if 'EVENT_TYPE' in d.names:
            out['event_type'] = decode_bits(d['EVENT_TYPE'])
    df = pd.DataFrame(out)
    src = SkyCoord(ra * u.deg, dec * u.deg)
    df['sep'] = src.separation(SkyCoord(df.ra.values * u.deg, df.dec.values * u.deg)).deg
    return df


def read_gtis(path: str | Path, t0: float) -> list[tuple[float, float]]:
    with fits.open(path) as h:
        g = h['GTI'].data
        return [(float(a) - t0, float(b) - t0) for a, b in zip(g['START'], g['STOP'])]


def pointing_table(ft2: str | Path, ra: float, dec: float, t0: float) -> pd.DataFrame:
    """Off-axis angle and zenith angle of the source for every spacecraft row."""
    with fits.open(ft2) as h:
        d = h['SC_DATA'].data
        src = SkyCoord(ra * u.deg, dec * u.deg)
        bore = SkyCoord(np.asarray(d['RA_SCZ'], float) * u.deg, np.asarray(d['DEC_SCZ'], float) * u.deg)
        zen = SkyCoord(np.asarray(d['RA_ZENITH'], float) * u.deg, np.asarray(d['DEC_ZENITH'], float) * u.deg)
        return pd.DataFrame({
            'start': np.asarray(d['START'], float) - t0,
            'stop': np.asarray(d['STOP'], float) - t0,
            'theta': src.separation(bore).deg,
            'zenith': src.separation(zen).deg,
            'data_qual': np.asarray(d['DATA_QUAL']).astype(np.int64),
            'lat_config': np.asarray(d['LAT_CONFIG']).astype(np.int64),
            'in_saa': np.asarray(d['IN_SAA']).astype(bool),
            'livetime': np.asarray(d['LIVETIME'], float),
        })


def theta_crossing(pointing: pd.DataFrame, theta_max: float, after: float) -> float | None:
    """First time after `after` at which the off-axis angle rises above `theta_max`."""
    mid = 0.5 * (pointing.start.values + pointing.stop.values)
    th = pointing.theta.values
    for i in range(1, len(mid)):
        if mid[i] <= after:
            continue
        if th[i - 1] <= theta_max < th[i]:
            return float(mid[i - 1] + (theta_max - th[i - 1]) * (mid[i] - mid[i - 1]) / (th[i] - th[i - 1]))
    return None


# ---------------------------------------------------------------------------
# GBM light curves and episodes
# ---------------------------------------------------------------------------

def gbm_ctime_lightcurve(path: str | Path, trigtime: float, emin_kev: float, emax_kev: float,
                         tmin: float, tmax: float, dt: float = 1.024) -> pd.DataFrame:
    """Counts and livetime on a uniform grid, using channels fully inside [emin_kev, emax_kev].

    `dt` must be a multiple of the native CTIME resolution (0.256 s, 0.064 s after a trigger),
    otherwise the grid aliases against the native bins.
    """
    with fits.open(path) as h:
        eb = h['EBOUNDS'].data
        s = h['SPECTRUM'].data
        chans = (eb['E_MIN'] >= emin_kev - 1e-6) & (eb['E_MAX'] <= emax_kev + 1e-6)
        start = np.asarray(s['TIME'], float) - trigtime
        stop = np.asarray(s['ENDTIME'], float) - trigtime
        counts = np.asarray(s['COUNTS'], float)[:, chans].sum(axis=1)
        expo = np.asarray(s['EXPOSURE'], float)
    edges = np.arange(tmin, tmax + 0.5 * dt, dt)
    centre = 0.5 * (start + stop)
    idx = np.digitize(centre, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    n = np.bincount(idx[ok], weights=counts[ok], minlength=len(edges) - 1)
    live = np.bincount(idx[ok], weights=expo[ok], minlength=len(edges) - 1)
    return pd.DataFrame({'t_lo': edges[:-1], 't_hi': edges[1:], 'counts': n, 'livetime': live})


def rank_gbm_detectors(files: dict[str, Path], trigtime: float, emin_kev: float, emax_kev: float,
                       src_window: tuple[float, float], bkg_windows: list[tuple[float, float]],
                       dt: float = 1.024, order: int = 1) -> pd.DataFrame:
    rows = []
    for det, path in files.items():
        lc = gbm_ctime_lightcurve(path, trigtime, emin_kev, emax_kev, -1000., 1000., dt)
        bkg = fit_background(lc, bkg_windows, order)
        m = (lc.t_lo >= src_window[0]) & (lc.t_hi <= src_window[1])
        n = lc.counts[m].sum()
        b = bkg[m].sum()
        rows.append({'detector': det, 'net_counts': n - b, 'significance': (n - b) / np.sqrt(b)})
    return pd.DataFrame(rows).sort_values('significance', ascending=False).reset_index(drop=True)


def fit_background(lc: pd.DataFrame, windows: list[tuple[float, float]], order: int = 2) -> np.ndarray:
    """Polynomial in rate, fitted to the background windows, returned as expected counts per bin."""
    mid = 0.5 * (lc.t_lo.values + lc.t_hi.values)
    use = np.zeros(len(lc), bool)
    for a, b in windows:
        use |= (mid >= a) & (mid < b)
    use &= lc.livetime.values > 0
    rate = lc.counts.values[use] / lc.livetime.values[use]
    weights = lc.livetime.values[use] / np.sqrt(np.maximum(lc.counts.values[use], 1.0))
    coef = np.polyfit(mid[use], rate, order, w=weights)
    return np.polyval(coef, mid) * lc.livetime.values


def background_chi2(lc: pd.DataFrame, bkg_counts: np.ndarray, windows: list[tuple[float, float]]) -> float:
    """Mean squared Poisson residual of the background model inside its own fit windows."""
    mid = 0.5 * (lc.t_lo.values + lc.t_hi.values)
    use = np.zeros(len(lc), bool)
    for a, b in windows:
        use |= (mid >= a) & (mid < b)
    use &= bkg_counts > 0
    resid = (lc.counts.values[use] - bkg_counts[use]) / np.sqrt(bkg_counts[use])
    return float(np.mean(resid ** 2))


def sum_lightcurves(lcs: list[pd.DataFrame]) -> pd.DataFrame:
    """Summed counts; livetime averaged so counts/livetime is the summed rate."""
    out = lcs[0][['t_lo', 't_hi']].copy()
    out['counts'] = np.sum([lc.counts.values for lc in lcs], axis=0)
    out['livetime'] = np.mean([lc.livetime.values for lc in lcs], axis=0)
    return out


def gbm_blocks(lc: pd.DataFrame, bkg_counts: np.ndarray, p0: float = 0.01) -> pd.DataFrame:
    """Bayesian blocks on the livetime-corrected rate, with the excess over background per block."""
    good = lc.livetime.values > 0
    mid = 0.5 * (lc.t_lo.values + lc.t_hi.values)[good]
    rate = lc.counts.values[good] / lc.livetime.values[good]
    sigma = np.sqrt(np.maximum(lc.counts.values[good], 1.0)) / lc.livetime.values[good]
    edges = bayesian_blocks(mid, rate, sigma, fitness='measures', p0=p0)
    rows = []
    t_lo, t_hi = lc.t_lo.values[good], lc.t_hi.values[good]
    n_all, b_all = lc.counts.values[good], bkg_counts[good]
    live = lc.livetime.values[good]
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (mid >= a) & ((mid < b) if k < len(edges) - 2 else (mid <= b))
        if not m.any():
            continue
        n, bk = n_all[m].sum(), b_all[m].sum()
        rows.append({
            't_lo': float(t_lo[m].min()), 't_hi': float(t_hi[m].max()),
            'net_rate': (n - bk) / live[m].sum(),
            'significance': (n - bk) / np.sqrt(bk),
        })
    return pd.DataFrame(rows)


def episodes_from_blocks(blocks: pd.DataFrame, sig_min: float = 3.0, min_duration: float = 2.0,
                         search: tuple[float, float] | None = None) -> pd.DataFrame:
    """Maximal runs of consecutive blocks with excess significance >= sig_min."""
    active = blocks.significance.values >= sig_min
    if search is not None:
        active &= (blocks.t_hi.values > search[0]) & (blocks.t_lo.values < search[1])
    episodes, start = [], None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(active) - 1):
            stop = i if flag else i - 1
            t_lo, t_hi = blocks.t_lo.values[start], blocks.t_hi.values[stop]
            if t_hi - t_lo >= min_duration:
                episodes.append({'t_lo': t_lo, 't_hi': t_hi, 'n_blocks': stop - start + 1,
                                 'peak_net_rate': blocks.net_rate.values[start:stop + 1].max()})
            start = None
    return pd.DataFrame(episodes)


# ---------------------------------------------------------------------------
# Source model
# ---------------------------------------------------------------------------

def _scaled(value: float) -> tuple[float, float]:
    scale = 10.0 ** np.floor(np.log10(abs(value))) if value != 0 else 1.0
    return value / scale, scale


def _param(name, value, scale=1.0, vmin=None, vmax=None, free=False) -> str:
    if vmin is None:
        vmin = min(value, 0.0) - abs(value) * 10 - 10
    if vmax is None:
        vmax = max(value, 0.0) + abs(value) * 10 + 10
    return (f'    <parameter free="{int(free)}" max="{vmax:.8g}" min="{vmin:.8g}" '
            f'name="{name}" scale="{scale:.8g}" value="{value:.8g}"/>')


def _skydir(ra, dec) -> str:
    return ('  <spatialModel type="SkyDirFunction">\n'
            + _param('RA', ra, 1.0, -360, 360) + '\n'
            + _param('DEC', dec, 1.0, -90, 90) + '\n'
            + '  </spatialModel>')


def _catalog_spectrum(row) -> str | None:
    stype = row['SpectrumType'].strip()
    pivot = float(row['Pivot_Energy'])
    if stype == 'PowerLaw':
        n, s = _scaled(float(row['PL_Flux_Density']))
        pars = [_param('Prefactor', n, s, 1e-5, 1e5),
                _param('Index', -float(row['PL_Index']), 1.0, -10, 10),
                _param('Scale', pivot, 1.0, 30, 5e5)]
    elif stype == 'LogParabola':
        n, s = _scaled(float(row['LP_Flux_Density']))
        pars = [_param('norm', n, s, 1e-5, 1e5),
                _param('alpha', float(row['LP_Index']), 1.0, -10, 10),
                _param('beta', float(row['LP_beta']), 1.0, -10, 10),
                _param('Eb', pivot, 1.0, 30, 5e5)]
        stype = 'LogParabola'
    elif stype.startswith('PLSuperExpCutoff'):
        n, s = _scaled(float(row['PLEC_Flux_Density']))
        pars = [_param('Prefactor', n, s, 1e-5, 1e5),
                _param('IndexS', -float(row['PLEC_IndexS']), 1.0, -10, 10),
                _param('Scale', pivot, 1.0, 30, 5e5),
                _param('ExpfactorS', float(row['PLEC_ExpfactorS']), 1.0, -10, 10),
                _param('Index2', float(row['PLEC_Exp_Index']), 1.0, 1e-4, 10)]
        stype = 'PLSuperExpCutoff4'
    else:
        return None
    return f'  <spectrum type="{stype}">\n' + '\n'.join(pars) + '\n  </spectrum>'


def write_model_xml(path: str | Path, ra: float, dec: float, catalog: str | Path, model_radius: float,
                    evclass: int, emin: float, emax: float, grb_name: str = 'GRB',
                    gal_file: str = 'gll_iem_v07.fits', diffuse_max: float = 10.0) -> Path:
    """GRB power law at a fixed position, free diffuse normalizations, fixed 4FGL point sources."""
    path = Path(path)
    gdir = galdiffuse_dir()
    gal = gdir / gal_file
    iso = gdir / EVCLASS_ISO[evclass]
    for f in (gal, iso):
        if not f.exists():
            raise FileNotFoundError(f)
    lines = ['<?xml version="1.0" ?>', '<source_library title="source library">']
    lines += [
        f'<source name="{grb_name}" type="PointSource">',
        '  <spectrum type="PowerLaw2">',
        _param('Integral', 1.0, 1e-5, 1e-6, 1e5, free=True),
        _param('Index', -2.0, 1.0, -5.0, -0.5, free=True),
        _param('LowerLimit', emin, 1.0, 20, 1e6),
        _param('UpperLimit', emax, 1.0, 20, 1e6),
        '  </spectrum>',
        _skydir(ra, dec),
        '</source>',
        '<source name="GalDiffuse" type="DiffuseSource">',
        '  <spectrum type="ConstantValue">',
        _param('Value', 1.0, 1.0, 0.0, diffuse_max, free=True),
        '  </spectrum>',
        f'  <spatialModel file="{escape(str(gal))}" type="MapCubeFunction">',
        _param('Normalization', 1.0, 1.0, 1e-3, 1e3),
        '  </spatialModel>',
        '</source>',
        '<source name="IsoDiffuse" type="DiffuseSource">',
        f'  <spectrum file="{escape(str(iso))}" type="FileFunction">',
        _param('Normalization', 1.0, 1.0, 0.0, diffuse_max, free=True),
        '  </spectrum>',
        '  <spatialModel type="ConstantValue">',
        _param('Value', 1.0, 1.0, 0.0, 10.0),
        '  </spatialModel>',
        '</source>',
    ]
    with fits.open(catalog) as h:
        cat = h['LAT_Point_Source_Catalog'].data
        src = SkyCoord(ra * u.deg, dec * u.deg)
        sep = src.separation(SkyCoord(cat['RAJ2000'] * u.deg, cat['DEJ2000'] * u.deg)).deg
        skipped = []
        for i in np.where(sep < model_radius)[0]:
            row = cat[i]
            if row['Extended_Source_Name'].strip():
                skipped.append(row['Source_Name'].strip())
                continue
            spec = _catalog_spectrum(row)
            if spec is None:
                skipped.append(row['Source_Name'].strip())
                continue
            name = row['Source_Name'].strip().replace(' ', '_')
            lines += [f'<source name="{escape(name)}" type="PointSource">', spec,
                      _skydir(float(row['RAJ2000']), float(row['DEJ2000'])), '</source>']
    if skipped:
        raise ValueError(f'Catalog sources inside the model radius need templates or spectra not handled here: {skipped}')
    lines.append('</source_library>')
    path.write_text('\n'.join(lines) + '\n')
    return path


# ---------------------------------------------------------------------------
# Selection, exposure, likelihood
# ---------------------------------------------------------------------------

@dataclass
class Selection:
    ft1: Path
    ft2: Path
    ra: float
    dec: float
    t0: float
    roi: float
    evclass: int
    emin: float
    emax: float
    zmax: float = 100.0
    evtype: int = 3
    catalog: Path | None = None
    workdir: Path = Path('work')
    expmap_pixel: float = 0.5
    expmap_nenergies: int = 20
    model_margin: float = 10.0

    @property
    def irf(self) -> str:
        return EVCLASS_IRF[self.evclass]

    @property
    def label(self) -> str:
        return f'roi{self.roi:g}_{EVCLASS_NAME[self.evclass]}'

    @property
    def model_xml(self) -> Path:
        return self.workdir / f'model_{self.label}.xml'

    def write_model(self) -> Path:
        self.workdir.mkdir(parents=True, exist_ok=True)
        return write_model_xml(self.model_xml, self.ra, self.dec, self.catalog, self.roi + self.model_margin,
                               self.evclass, self.emin, self.emax)


def _tag(sel: Selection, tmin: float, tmax: float) -> str:
    return f'{sel.label}_{tmin:+09.2f}_{tmax:+09.2f}'.replace('.', 'p').replace('+', 'P').replace('-', 'M')


def good_time_intervals(sel: Selection, tmin: float, tmax: float) -> list[tuple[float, float]]:
    """GTIs after the spacecraft filter and the ROI-based zenith cut, without exposure products."""
    d = sel.workdir / f'gti_{_tag(sel, tmin, tmax)}'
    d.mkdir(parents=True, exist_ok=True)
    pf = d / 'pfiles'
    run_tool('gtselect', pf, infile=sel.ft1, outfile=d / 'selected.fits', ra=sel.ra, dec=sel.dec, rad=sel.roi,
             tmin=f'{sel.t0 + tmin:.6f}', tmax=f'{sel.t0 + tmax:.6f}', emin=sel.emin, emax=sel.emax,
             zmax=sel.zmax, evclass=sel.evclass, evtype=sel.evtype, chatter=1)
    run_tool('gtmktime', pf, evfile=d / 'selected.fits', scfile=sel.ft2, outfile=d / 'events.fits',
             filter=GTI_FILTER, roicut='yes', chatter=1)
    return read_gtis(d / 'events.fits', sel.t0)


@dataclass
class Prepared:
    sel: Selection
    tmin: float
    tmax: float
    evfile: Path
    ltcube: Path
    expmap: Path
    gtis: list = field(default_factory=list)
    n_events: int = 0


def prepare_interval(sel: Selection, tmin: float, tmax: float, reuse: bool = True) -> Prepared:
    """gtselect, gtmktime, gtdiffrsp, gtltcube and gtexpmap for one interval; reuses finished outputs."""
    tag = _tag(sel, tmin, tmax)
    d = sel.workdir / tag
    d.mkdir(parents=True, exist_ok=True)
    pf = d / 'pfiles'
    sel_file, gti_file = d / 'selected.fits', d / 'events.fits'
    ltcube, expmap, done = d / 'ltcube.fits', d / 'expmap.fits', d / '.done'
    if not (reuse and done.exists()):
        run_tool('gtselect', pf, infile=sel.ft1, outfile=sel_file, ra=sel.ra, dec=sel.dec, rad=sel.roi,
                 tmin=f'{sel.t0 + tmin:.6f}', tmax=f'{sel.t0 + tmax:.6f}', emin=sel.emin, emax=sel.emax,
                 zmax=sel.zmax, evclass=sel.evclass, evtype=sel.evtype, chatter=1)
        run_tool('gtmktime', pf, evfile=sel_file, scfile=sel.ft2, outfile=gti_file, filter=GTI_FILTER,
                 roicut='yes', chatter=1)
        if read_gtis(gti_file, sel.t0):
            if not sel.model_xml.exists():
                sel.write_model()
            run_tool('gtdiffrsp', pf, evfile=gti_file, scfile=sel.ft2, srcmdl=sel.model_xml, irfs=sel.irf, chatter=1)
            run_tool('gtltcube', pf, evfile=gti_file, scfile=sel.ft2, outfile=ltcube, dcostheta=0.025, binsz=1,
                     chatter=1)
            srcrad = sel.roi + sel.model_margin
            npix = int(np.ceil(2 * srcrad / sel.expmap_pixel))
            run_tool('gtexpmap', pf, evfile=gti_file, scfile=sel.ft2, expcube=ltcube, outfile=expmap,
                     irfs=sel.irf, srcrad=srcrad, nlong=npix, nlat=npix, nenergies=sel.expmap_nenergies, chatter=1)
        done.touch()
    gtis = read_gtis(gti_file, sel.t0)
    with fits.open(gti_file) as h:
        n_events = len(h['EVENTS'].data)
    return Prepared(sel, tmin, tmax, gti_file, ltcube, expmap, gtis, n_events)


def prepare_many(jobs: list[tuple[Selection, float, float]], workers: int = 6) -> list[Prepared]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda j: prepare_interval(*j), jobs))


def _free_index(like, src: str, par: str) -> int:
    full = like.par_index(src, par)
    return sum(1 for p in like.params()[:full] if p.isFree())


def fit_prepared(prep: Prepared, grb: str = 'GRB', ts_ul: float = 25.0, save_xml: Path | None = None) -> dict:
    """Unbinned likelihood for one prepared interval. Upper limit (95%, index -2) when TS < ts_ul."""
    sel = prep.sel
    row = {'approach': None, 'roi': sel.roi, 'evclass': EVCLASS_NAME[sel.evclass], 'tmin': prep.tmin,
           'tmax': prep.tmax, 'exposed_s': sum(b - a for a, b in prep.gtis), 'n_events': prep.n_events}
    if not prep.gtis or prep.n_events == 0:
        row.update(ts=np.nan, note='no good time or no events')
        return row
    import pyLikelihood as pyLike
    from UnbinnedAnalysis import UnbinnedObs, UnbinnedAnalysis

    obs = UnbinnedObs(str(prep.evfile), str(sel.ft2), expMap=str(prep.expmap), expCube=str(prep.ltcube),
                      irfs=sel.irf)
    like = UnbinnedAnalysis(obs, srcModel=str(sel.model_xml), optimizer='NewMinuit')
    try:
        like.fit(verbosity=0, covar=False, optimizer='DRMNFB')
    except Exception:
        pass
    opt = pyLike.NewMinuit(like.logLike)
    like.fit(verbosity=0, covar=True, optObject=opt)
    spec = like.model[grb].funcs['Spectrum']
    integral, index = spec.getParam('Integral'), spec.getParam('Index')
    cov = np.array(like.covariance)
    i_n, i_g = _free_index(like, grb, 'Integral'), _free_index(like, grb, 'Index')
    corr = cov[i_n, i_g] / np.sqrt(cov[i_n, i_n] * cov[i_g, i_g])
    at_bound = []
    for name, pname in (('GalDiffuse', 'Value'), ('IsoDiffuse', 'Normalization')):
        p = like.model[name].funcs['Spectrum'].getParam(pname)
        lo, hi = p.getBounds()
        if p.getValue() <= lo + 1e-3 or p.getValue() >= hi - 1e-3:
            at_bound.append(name)
    row.update(
        index=index.getValue(), index_err=index.error(),
        flux=integral.getTrueValue(), flux_err=integral.error() * integral.getScale(),
        eflux=like.energyFlux(grb, sel.emin, sel.emax) * 1.602176634e-6,
        eflux_err=like.energyFluxError(grb, sel.emin, sel.emax) * 1.602176634e-6,
        corr_flux_index=corr, npred=like.NpredValue(grb), fit_code=opt.getRetCode(),
        diffuse_at_bound=','.join(at_bound), log_like=-like(),
    )
    if save_xml is not None:
        like.writeXml(str(save_xml))
    row['ts'] = like.Ts(grb, reoptimize=True)
    row['flux_ul95'] = np.nan
    if row['ts'] < ts_ul:
        from UpperLimits import UpperLimits
        like.model[grb].funcs['Spectrum'].getParam('Index').setValue(-2.0)
        like.freeze(like.par_index(grb, 'Index'))
        like.fit(verbosity=0, covar=False, optObject=pyLike.NewMinuit(like.logLike))
        ul = UpperLimits(like)
        try:
            # delta = 2.71/2 in -log L is the one-sided 95% profile-likelihood limit.
            row['flux_ul95'] = ul[grb].compute(emin=sel.emin, emax=sel.emax, delta=2.71 / 2, verbosity=0)[0]
        except Exception as exc:
            row['note'] = f'upper limit failed: {exc}'
    return row


def adaptive_bins(sel: Selection, segments: list[tuple[float, float]], events: pd.DataFrame,
                  ts_min: float = 25.0, r_core: float = 3.0, min_duration: float = 5.0,
                  workers: int = 6, log=print) -> tuple[list[tuple[float, float]], pd.DataFrame]:
    """Split each segment in two at the median arrival time of events within r_core of the source,
    as long as both halves keep TS >= ts_min. Splits never cross segment boundaries."""
    fits_done: dict[tuple[float, float], dict] = {}

    def fit_all(intervals):
        todo = [iv for iv in intervals if iv not in fits_done]
        for prep in prepare_many([(sel, a, b) for a, b in todo], workers=workers):
            fits_done[(prep.tmin, prep.tmax)] = fit_prepared(prep)

    core = events[(events.sep <= r_core) & ((events.event_class & sel.evclass) != 0)
                  & events.energy.between(sel.emin, sel.emax) & (events.zenith <= sel.zmax)]

    def split_point(a, b):
        t = np.sort(core.t_rel[(core.t_rel >= a) & (core.t_rel < b)].values)
        if len(t) < 2:
            return None
        m = 0.5 * (t[len(t) // 2 - 1] + t[len(t) // 2]) if len(t) % 2 == 0 else t[len(t) // 2]
        m = round(float(m), 2)
        if m - a < min_duration or b - m < min_duration:
            return None
        return m

    fit_all(segments)
    final, pending = [], []
    for seg in segments:
        (pending if fits_done[seg].get('ts', 0) >= ts_min else final).append(seg)
    while pending:
        proposals = {}
        for a, b in pending:
            m = split_point(a, b)
            if m is None:
                final.append((a, b))
            else:
                proposals[(a, b)] = ((a, m), (m, b))
        fit_all([h for pair in proposals.values() for h in pair])
        pending = []
        for parent, (left, right) in proposals.items():
            if fits_done[left].get('ts', 0) >= ts_min and fits_done[right].get('ts', 0) >= ts_min:
                log(f'split {parent[0]:.2f}-{parent[1]:.2f} at {left[1]:.2f}: '
                    f'TS {fits_done[left]["ts"]:.1f} / {fits_done[right]["ts"]:.1f}')
                pending += [left, right]
            else:
                final.append(parent)
    final.sort()
    table = pd.DataFrame([fits_done[iv] for iv in final])
    return final, table


def powerlaw2_band(flux, flux_err, index, index_err, corr, emin, emax, energies, n_samples=4000, seed=1):
    """E^2 dN/dE (erg cm^-2 s^-1) median and 68% band for a PowerLaw2 with correlated (flux, index)."""
    rng = np.random.default_rng(seed)
    cov = np.array([[flux_err ** 2, corr * flux_err * index_err], [corr * flux_err * index_err, index_err ** 2]])
    draws = rng.multivariate_normal([flux, index], cov, size=n_samples)
    draws = draws[draws[:, 0] > 0]
    n, g = draws[:, :1], draws[:, 1:]
    e = np.asarray(energies)[None, :]
    norm = n * (g + 1) / (emax ** (g + 1) - emin ** (g + 1))
    sed = e ** 2 * norm * e ** g * 1.602176634e-6
    lo, med, hi = np.percentile(sed, [16, 50, 84], axis=0)
    return med, lo, hi


def source_probabilities(prep: Prepared, fitted_xml: Path, outfile: Path, ra: float, dec: float,
                         grb: str = 'GRB') -> pd.DataFrame:
    sel = prep.sel
    run_tool('gtsrcprob', prep.evfile.parent / 'pfiles', evfile=prep.evfile, scfile=sel.ft2, outfile=outfile,
             srcmdl=fitted_xml, irfs=sel.irf, chatter=1)
    df = read_events(outfile, ra, dec, sel.t0)
    with fits.open(outfile) as h:
        df['p_grb'] = np.asarray(h['EVENTS'].data[grb]).astype(float)
    return df
