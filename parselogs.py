#!/usr/bin/env python3
"""
parse_multinest_logs_final_with_nobs.py

Parses multinest-style logfiles and produces a summary CSV.

Behavior highlights:
 - Default output: output/summary.csv (created if needed)
 - Extracts jetType, label, livepoints from the Commandline: line
 - Extracts fixed_* entries from two-column '|' tables
 - Extracts inferred parameters (prior_low_*, prior_high_*, post_med_*, post_sigma_*)
 - Skips files that:
     * have no inferred parameters (after filtering spurious 'Parameter' header), OR
     * use 'thc' or 'thv' directly as parameter names (in fixed or inferred tables)
 - Parses lnZ and lnZErr from lines like:
     "... lnZ=-393.2138 lnZErr=0.1522" (last occurrence wins)
 - Parses number of observations from lines like:
     "... Observations file has 31 records." (last occurrence wins)
 - Removes 'filename' column from the CSV
 - Writes CSV with columns:
     jetType,label,livepoints, fixed_..., (inferred columns grouped by param), nObservations, lnZ, lnZErr
"""
import sys
import shlex
import re
import csv
import argparse
import os

# --- parsing helpers --------------------------------------------------------

def parse_commandline_line(line):
    res = {'jetType': '', 'label': '', 'livepoints': ''}
    if "Commandline:" not in line:
        return res
    cmd_part = line.split("Commandline:", 1)[1].strip()
    try:
        args = shlex.split(cmd_part)
    except ValueError:
        args = cmd_part.split()
    def get_arg(flag):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                return args[idx + 1]
        return ''
    res['jetType'] = get_arg("--jetType")
    res['label'] = get_arg("--label")
    res['livepoints'] = get_arg("--livepoints")
    return res

def parse_two_column_pipe_table(lines, start_idx):
    data = {}
    i = start_idx
    while i < len(lines):
        raw = lines[i].strip()
        if not raw or not raw.startswith("|"):
            break
        cols = [c.strip() for c in raw.split("|")]
        if len(cols) >= 3:
            key = cols[1]
            value = cols[2] if len(cols) > 2 else ''
            if key != '':
                data[key] = value
                i += 1
                continue
        break
    return data, i

def parse_five_column_pipe_table(lines, start_idx):
    """
    Return rows (list of 5-element lists) and next index after table.
    """
    rows = []
    i = start_idx
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        if raw.startswith("+"):
            i += 1
            continue
        if raw.startswith("|"):
            cols = [c.strip() for c in raw.split("|")]
            if len(cols) >= 6:
                five = cols[1:6]
                if five[0] != '':
                    rows.append(five)
                    i += 1
                    continue
            tokens = [t for t in cols if t != '']
            if len(tokens) >= 5:
                rows.append(tokens[:5])
                i += 1
                continue
            break
        else:
            break
    return rows, i

def extract_lnz_from_line(line):
    """
    Find lnZ and lnZErr in a line. Returns (lnZ, lnZErr) or (None, None).
    Accept formats like: lnZ=-393.2138 lnZErr=0.1522
    """
    lnz_re = re.compile(r"lnZ\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)")
    lnzerr_re = re.compile(r"lnZErr\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)")
    m1 = lnz_re.search(line)
    m2 = lnzerr_re.search(line)
    lnz = m1.group(1) if m1 else None
    lnzerr = m2.group(1) if m2 else None
    return lnz, lnzerr

def extract_nobs_from_line(line):
    """
    Find number of observations in a line.
    Matches phrases like: 'Observations file has 31 records.'
    Returns integer as string, or None.
    """
    # case-insensitive, allow some spacing variations
    m = re.search(r"Observations\s+file\s+has\s+(\d+)\s+records", line, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None

# --- file processing -------------------------------------------------------

def process_file(filename):
    """
    Parse filename and return a dict of collected values or None to indicate the file should be skipped.
    Returns:
      dict with keys: 'jetType','label','livepoints', fixed_..., prior_low_<param>, prior_high_<param>, post_med_<param>, post_sigma_<param>, 'nObservations', 'lnZ', 'lnZErr'
    Or None if file should be skipped.
    """
    simple = {'filename': filename, 'jetType': '', 'label': '', 'livepoints': ''}
    fixed_params = {}
    inferred = {}
    lnZ = None
    lnZErr = None
    nObservations = None

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"# ERROR reading {filename}: {e}", file=sys.stderr)
        return None

    # 1) commandline (first occurrence)
    for ln in lines:
        if "Commandline:" in ln:
            simple.update(parse_commandline_line(ln))
            break

    # 2) scan lines for tables, lnZ and nObservations
    i = 0
    L = len(lines)
    while i < L:
        line = lines[i]
        stripped = line.strip()

        # capture lnZ/lnZErr if present (keep last occurrence)
        lz, lze = extract_lnz_from_line(line)
        if lz is not None:
            lnZ = lz
        if lze is not None:
            lnZErr = lze

        # capture number of observations (keep last occurrence)
        nobs_found = extract_nobs_from_line(line)
        if nobs_found is not None:
            nObservations = nobs_found

        # parse tables
        if stripped.startswith("|") and stripped.count("|") >= 2:
            header_tokens = [t.strip() for t in stripped.split("|")]
            joined_header = " ".join([t.lower() for t in header_tokens if t])
            if "parameter" in joined_header and ("prior" in joined_header or "posterior" in joined_header):
                rows, next_i = parse_five_column_pipe_table(lines, i)
                for r in rows:
                    param = r[0]
                    # ignore spurious header row named 'Parameter'
                    if param.strip().lower() == "parameter":
                        continue
                    # store inferred
                    prior_low = r[1]
                    prior_high = r[2]
                    post_med = r[3]
                    post_sigma = r[4]
                    inferred[f"prior_low_{param}"] = prior_low
                    inferred[f"prior_high_{param}"] = prior_high
                    inferred[f"post_med_{param}"] = post_med
                    inferred[f"post_sigma_{param}"] = post_sigma
                i = next_i
                continue
            else:
                fixed, next_i = parse_two_column_pipe_table(lines, i)
                for k, v in fixed.items():
                    fixed_params[f"fixed_{k}"] = v
                i = next_i
                continue

        # sometimes tables start after a +----+ separator
        if stripped.startswith("+") and i+1 < L and "|" in lines[i+1]:
            rows, next_i = parse_five_column_pipe_table(lines, i+1)
            if rows:
                for r in rows:
                    param = r[0]
                    if param.strip().lower() == "parameter":
                        continue
                    prior_low = r[1]
                    prior_high = r[2]
                    post_med = r[3]
                    post_sigma = r[4]
                    inferred[f"prior_low_{param}"] = prior_low
                    inferred[f"prior_high_{param}"] = prior_high
                    inferred[f"post_med_{param}"] = post_med
                    inferred[f"post_sigma_{param}"] = post_sigma
                i = next_i
                continue

        i += 1

    # 3) Decide whether to skip:
    # - skip if no inferred parameters (after ignoring 'Parameter' header)
    if not inferred:
        return None

    # - skip if fixed params or inferred params contain parameter names 'thc' or 'thv' directly
    for fk in fixed_params.keys():
        if fk.lower().startswith("fixed_"):
            pname = fk[len("fixed_"):].strip().lower()
            if pname in ("thc", "thv"):
                return None

    for k in list(inferred.keys()):
        for prefix in ("prior_low_", "prior_high_", "post_med_", "post_sigma_"):
            if k.startswith(prefix):
                pname = k[len(prefix):].strip().lower()
                if pname in ("thc", "thv"):
                    return None
                break

    # 4) build combined output dict
    combined = {}
    combined.update(simple)
    combined.update(fixed_params)
    combined.update(inferred)
    combined['nObservations'] = nObservations if nObservations is not None else ""
    combined['lnZ'] = lnZ if lnZ is not None else ""
    combined['lnZErr'] = lnZErr if lnZErr is not None else ""

    return combined

# --- CSV writing -----------------------------------------------------------

def write_csv(rows, outpath):
    if not rows:
        print("# No parsed files meeting criteria. No CSV written.", file=sys.stderr)
        return

    # ensure output directory exists
    outdir = os.path.dirname(outpath)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # collect all keys
    keys = set()
    for r in rows:
        keys.update(r.keys())

    # remove filename column from keys if present
    keys.discard("filename")

    # build header: base fields, fixed columns, inferred grouped columns, nObservations, lnZ/lnZErr, then any remaining
    header = []
    for base in ("jetType", "label", "livepoints"):
        if base in keys:
            header.append(base)

    # fixed_ columns sorted
    fixed_cols = sorted(k for k in keys if k.startswith("fixed_"))
    header.extend(fixed_cols)

    # inferred parameter names
    param_names = set()
    for prefix in ("prior_low_", "prior_high_", "post_med_", "post_sigma_"):
        for k in keys:
            if k.startswith(prefix):
                param_names.add(k[len(prefix):])
    param_names = sorted(param_names)

    for p in param_names:
        header.extend([
            f"prior_low_{p}",
            f"prior_high_{p}",
            f"post_med_{p}",
            f"post_sigma_{p}"
        ])

    # add nObservations if present
    if 'nObservations' in keys and 'nObservations' not in header:
        header.append('nObservations')

    # ensure lnZ and lnZErr are present
    if 'lnZ' in keys and 'lnZ' not in header:
        header.append('lnZ')
    if 'lnZErr' in keys and 'lnZErr' not in header:
        header.append('lnZErr')

    # remaining keys not added yet
    remaining = [k for k in sorted(keys) if k not in header]
    header.extend(remaining)

    # write CSV
    with open(outpath, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r.get(c, "") for c in header])

# --- main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse multinest logfiles and produce filtered CSV (default output/summary.csv).")
    parser.add_argument('files', nargs='+', help='log files to parse')
    parser.add_argument('-o', '--output', default='output/summary.csv', help='output CSV filename (default output/summary.csv)')
    args = parser.parse_args()

    all_rows = []
    skipped_count = 0
    for fn in args.files:
        parsed = process_file(fn)
        if parsed is None:
            skipped_count += 1
            continue
        all_rows.append(parsed)

    write_csv(all_rows, args.output)
    print(f"Wrote summary for {len(all_rows)} files to {args.output}. Skipped {skipped_count} files.")

if __name__ == "__main__":
    main()
