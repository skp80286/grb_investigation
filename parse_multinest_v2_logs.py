#!/usr/bin/env python3
"""Parse multinest_.log files matching a glob pattern into a summary CSV on stdout."""

import argparse
import csv
import glob
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
DEG_PER_RAD = 180.0 / math.pi

LOG_TO_REAL = {
    "loge0": "e0",
    "logepsb": "epsb",
    "logepse": "epse",
    "logn0": "n0",
    "logthc": "theta_c",
    "logthv": "theta_v",
    "loglf": "lf",
    "logA": "A",
}


def sci2(x):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{float(x):.2e}"


def to_real(name, value):
    if name.startswith("log"):
        real_name = LOG_TO_REAL.get(name, name[3:])
        real_val = 10 ** float(value)
        if real_name in ("theta_c", "theta_v"):
            return real_name, real_val * DEG_PER_RAD
        return real_name, real_val
    return name, float(value)


def parse_ascii_table(lines, start_idx):
    rows = {}
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|") or "Parameter" in line or line.startswith("+"):
            if rows and not line.startswith("|"):
                break
            i += 1
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2:
            rows[parts[0]] = parts[1:]
        i += 1
    return rows, i


def parse_log_file(path):
    text = Path(path).read_text()
    lines = text.splitlines()

    dir_name = os.path.basename(os.path.dirname(path))

    obsfile = ""
    for line in reversed(lines):
        m = re.search(r"--obsfile\s+(\S+)", line)
        if m:
            obsfile = os.path.basename(m.group(1))
            break

    n_obs = ""
    for line in reversed(lines):
        m = re.search(r"Observations file has (\d+) records", line)
        if m:
            n_obs = int(m.group(1))
            break

    bic = ""
    for line in reversed(lines):
        m = re.search(r"BIC=([0-9.+-]+)", line)
        if m:
            bic = float(m.group(1))
            break

    fixed = {}
    inferred = {}
    for i, line in enumerate(lines):
        if line.strip() == "Fixed parameters":
            table, _ = parse_ascii_table(lines, i + 1)
            fixed = {k: v[0] for k, v in table.items()}
        if "Inferred parameters:" in line:
            table, _ = parse_ascii_table(lines, i + 1)
            for param, cols in table.items():
                if len(cols) >= 3:
                    inferred[param] = {
                        "median": float(cols[2]),
                        "rel_sigma": float(cols[3]),
                    }

    max_param = ""
    max_sigma = None
    if inferred:
        max_param = max(inferred, key=lambda p: inferred[p]["rel_sigma"])
        max_sigma = inferred[max_param]["rel_sigma"]

    return {
        "dir_name": dir_name,
        "obsfile": obsfile,
        "n_obs": n_obs,
        "bic": bic,
        "max_rel_sigma": max_sigma,
        "max_rel_sigma_param": max_param,
        "fixed": fixed,
        "inferred": inferred,
    }


def build_rows(runs):
    all_inferred = sorted({p for r in runs for p in r["inferred"]})
    all_fixed = sorted({p for r in runs for p in r["fixed"]})

    rows = [
        ("dir_name", lambda r: r["dir_name"]),
        ("obs_data_file", lambda r: r["obsfile"]),
        ("n_observations", lambda r: r["n_obs"]),
        ("BIC", lambda r: sci2(r["bic"])),
        ("max_rel_sigma", lambda r: sci2(r["max_rel_sigma"])),
        ("max_rel_sigma_param", lambda r: r["max_rel_sigma_param"]),
    ]

    for param in all_inferred:
        real_name, _ = to_real(param, 0)
        rows.append(
            (
                f"median_{real_name}",
                lambda r, p=param: sci2(
                    to_real(p, r["inferred"][p]["median"])[1]
                    if p in r["inferred"]
                    else None
                ),
            )
        )

    for param in all_fixed:
        if param == "jetType":
            rows.append((f"fixed_{param}", lambda r, p=param: r["fixed"].get(p, "")))
            continue
        real_name, _ = to_real(param, 0)
        rows.append(
            (
                f"fixed_{real_name}",
                lambda r, p=param: sci2(
                    to_real(p, float(r["fixed"][p]))[1] if p in r["fixed"] else None
                ),
            )
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Parse multinest_.log files into a summary CSV."
    )
    parser.add_argument(
        "log_pattern",
        help="Glob pattern for log files (e.g. output/*v2*ours3/multinest_.log)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    log_files = sorted(glob.glob(str(repo / args.log_pattern)))
    if not log_files:
        raise SystemExit(f"No log files matched {args.log_pattern}", file=sys.stderr)

    runs = [parse_log_file(p) for p in log_files]
    rows = build_rows(runs)

    writer = csv.writer(sys.stdout)
    writer.writerow(["field"] + [r["dir_name"] for r in runs])
    for label, getter in rows:
        writer.writerow([label] + [getter(r) for r in runs])


if __name__ == "__main__":
    main()
