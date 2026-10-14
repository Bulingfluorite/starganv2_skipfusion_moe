#!/usr/bin/env python3
"""format_metrics.py — 原始 CSV → 4-section 格式化 CSV

原始 CSV: modality,source_to_target,subject,nrmse,ssim,lpips
输出: 4 个 section（每个 subject + 平均值），每页按 modality 分组，
      20 行 pair + 1 行均值。

Usage:
    python format_metrics.py \
        --input raw.csv --output formatted.csv \
        --subjects "0006 0007 0009" \
        --modalities "T1W T2W T2FLAIR" \
        --fields "0.1T 1.5T 3T 5T 7T"
"""

import argparse
import csv
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--subjects", required=True,
                   help="Space-separated subject IDs, e.g. '0006 0007 0009'")
    p.add_argument("--modalities", required=True,
                   help="Space-separated modalities, e.g. 'T1W T2W T2FLAIR'")
    p.add_argument("--fields", required=True,
                   help="Space-separated fields, e.g. '0.1T 1.5T 3T 5T 7T'")
    return p.parse_args()


def build_pair_order(fields):
    pairs = []
    for src in fields:
        for tgt in fields:
            if src != tgt:
                pairs.append(f"{src}_to_{tgt}")
    return pairs


def main():
    args = parse_args()
    subjects = args.subjects.split()
    modalities = args.modalities.split()
    fields = args.fields.split()
    pair_order = build_pair_order(fields)

    # Read raw CSV into dict: (mod, pair, subj) -> (nrmse, ssim, lpips)
    data = {}
    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["modality"], row["source_to_target"], row["subject"])
            data[key] = (row["nrmse"], row["ssim"], row["lpips"])

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)

        # Section 1-3: per subject
        for subj in subjects:
            w.writerow([f"=== Subject {subj} ==="])
            w.writerow(["modality", "source_to_target", "nrmse", "ssim", "lpips"])
            _write_modality_blocks(w, modalities, pair_order, data, subj, subjects=None)

        # Section 4: average across subjects
        w.writerow([])
        w.writerow(["=== Average (3-subject mean) ==="])
        w.writerow(["modality", "source_to_target", "nrmse", "ssim", "lpips"])
        _write_modality_blocks(w, modalities, pair_order, data, subj=None, subjects=subjects)


def _write_modality_blocks(writer, modalities, pair_order, data, subj, subjects):
    for mod in modalities:
        pair_rows = []
        for pair in pair_order:
            if subj is not None:
                key = (mod, pair, subj)
                nrmse, ssim, lpips = data.get(key, ("NaN", "NaN", "NaN"))
            else:
                vals = []
                for s in subjects:
                    k = (mod, pair, s)
                    vals.append(data.get(k))
                vals = [v for v in vals if v is not None]
                if len(vals) == len(subjects):
                    nrmse = _safe_mean([v[0] for v in vals])
                    ssim = _safe_mean([v[1] for v in vals])
                    lpips = _safe_mean([v[2] for v in vals])
                else:
                    nrmse = ssim = lpips = "NaN"
            pair_rows.append((nrmse, ssim, lpips))

        for pair, (n, s, l) in zip(pair_order, pair_rows):
            writer.writerow([mod, pair, n, s, l])

        # Mean row for this modality
        n_vals = [v for v in [r[0] for r in pair_rows] if v != "NaN"]
        s_vals = [v for v in [r[1] for r in pair_rows] if v != "NaN"]
        l_vals = [v for v in [r[2] for r in pair_rows] if v != "NaN"]
        writer.writerow([
            f"{mod}_MEAN",
            "—",
            _safe_mean(n_vals),
            _safe_mean(s_vals),
            _safe_mean(l_vals),
        ])
        writer.writerow([])


def _safe_mean(vals):
    if not vals:
        return "NaN"
    try:
        return f"{sum(float(v) for v in vals) / len(vals):.6f}"
    except (ValueError, TypeError):
        return "NaN"


if __name__ == "__main__":
    main()
