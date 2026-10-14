#!/usr/bin/env python3
"""Build the Synapse submission directory tree from Baseline sweep predictions.

Source layout (Baseline experiment sweep):
    {predictions_dir}/{task}_{pair}_{MOD}/{method}/{mode}/{epoch_tag}[_seg]/
        P_{MOD}_{SRC_FIELD}_{ID}.nii.gz                # pred
        P_{MOD}_{SRC_FIELD}_{ID}_seg.nii.gz            # seg (task1/task2 only)

Target layout (Submission pack, ready to `zip -r task{N}.zip T1W T2W T2FLAIR`):
    {output_dir}/{task}/{MOD}/{pair}/pred/
        P_{MOD}_{TGT_FIELD}_{ID}.nii.gz
    {output_dir}/{task}/{MOD}/{pair}/seg/                # task1/task2 only
        P_{MOD}_{TGT_FIELD}_{ID}_seg.nii.gz

The two layouts differ in five ways: top-level naming, pred/seg directory names,
method/mode/epoch nesting (we pick one combo per task), file-name field tag
(SRC -> TGT), and the lack of a top-level zip (this script does NOT zip; user
runs `zip -r ~/task{N}.zip T1W T2W T2FLAIR` from inside `{output_dir}/task{N}`
when they're ready to upload).

See ../../Submission/README.md for the submission spec and the file-name rule:
"keep the source ID and replace the field tag with the target field".
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from mrixfields.zclip_constants import Z_CLIP_RANGE

MODS = ("T1W", "T2W", "T2FLAIR")

PAIRS: dict[str, list[str]] = {
    "task1": [
        "0.1T_to_7T",
        "1.5T_to_7T",
        "3T_to_7T",
        "5T_to_7T",
    ],
    "task2": [
        "0.1T_to_1.5T",
        "0.1T_to_3T",
        "0.1T_to_5T",
        "0.1T_to_7T",
    ],
    "task3": [
        f"{src}_to_{tgt}"
        for src in ("0.1T", "1.5T", "3T", "5T", "7T")
        for tgt in ("0.1T", "1.5T", "3T", "5T", "7T")
        if src != tgt
    ],
}

NEEDS_SEG = {"task1": True, "task2": True, "task3": False}

EXPECTED_FILES_PER_SUBTASK = 3

DEFAULTS = {
    "task1": ("cut", "pro_pretrained", "epoch100"),
    "task2": ("cut", "pro_pretrained", "epoch100"),
    "task3": ("stargan_v2", "pro_pretrained", "epoch50"),
}


def _copy_with_axial_clip(src: Path, dst: Path) -> None:
    """Real axial slice [z0, z1) — shape becomes (X, Y, z1 - z0).
    nibabel's slicer auto-shifts the affine origin so the new volume's
    voxel (0,0,0) is at the physical position of the old (0,0,z0)."""
    z0, z1 = Z_CLIP_RANGE
    img = nib.load(str(src))
    out = img.slicer[:, :, z0:z1]
    nib.save(out, str(dst))


def parse_pair(pair: str) -> tuple[str, str]:
    src, tgt = pair.split("_to_")
    return src, tgt


def extract_id(filename: str, mod: str, src_field: str) -> str | None:
    m = re.match(
        rf"^P_{re.escape(mod)}_{re.escape(src_field)}_(\d{{4}})(?:_seg)?\.nii\.gz$",
        filename,
    )
    return m.group(1) if m else None


def build_one_task(
    task: str,
    method: str,
    mode: str,
    epoch_tag: str,
    predictions_dir: Path,
    predictions_seg_dir: Path,
    output_dir: Path,
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    """Build one task tree. Returns (copied_pred, copied_seg, warnings)."""
    needs_seg = NEEDS_SEG[task]
    pairs = PAIRS[task]
    expected_pred = len(pairs) * len(MODS) * EXPECTED_FILES_PER_SUBTASK
    expected_seg = expected_pred if needs_seg else 0

    copied_pred = 0
    copied_seg = 0
    warnings: list[str] = []

    print(f"\n=== {task}: {method}/{mode}/{epoch_tag} ===")
    print(f"  expected pred files: {expected_pred}" + (f", seg: {expected_seg}" if needs_seg else ""))

    for mod in MODS:
        for pair in pairs:
            src_field, tgt_field = parse_pair(pair)
            src_pred_dir = (
                predictions_dir / f"{task}_{pair}_{mod}" / method / mode / epoch_tag
            )
            dst_pred_dir = output_dir / task / mod / pair / "pred"

            if not src_pred_dir.is_dir():
                warnings.append(f"missing src pred dir: {src_pred_dir}")
                continue

            src_pred_files = sorted(src_pred_dir.glob(f"P_{mod}_{src_field}_*.nii.gz"))
            src_pred_files = [f for f in src_pred_files if not f.name.endswith("_seg.nii.gz")]
            if len(src_pred_files) < EXPECTED_FILES_PER_SUBTASK:
                warnings.append(
                    f"only {len(src_pred_files)}/{EXPECTED_FILES_PER_SUBTASK} pred files in {src_pred_dir}"
                )

            if not dry_run:
                dst_pred_dir.mkdir(parents=True, exist_ok=True)

            for src_file in src_pred_files:
                file_id = extract_id(src_file.name, mod, src_field)
                if file_id is None:
                    warnings.append(f"unparseable filename: {src_file}")
                    continue
                dst_file = dst_pred_dir / f"P_{mod}_{tgt_field}_{file_id}.nii.gz"
                if dry_run:
                    print(f"  [dry-run] {src_file} -> {dst_file}")
                else:
                    _copy_with_axial_clip(src_file, dst_file)
                copied_pred += 1

            if needs_seg:
                src_seg_dir = (
                    predictions_seg_dir / f"{task}_{pair}_{mod}" / method / mode / epoch_tag
                )
                dst_seg_dir = output_dir / task / mod / pair / "seg"

                if not src_seg_dir.is_dir():
                    warnings.append(f"missing src seg dir: {src_seg_dir}")
                    continue

                src_seg_files = sorted(src_seg_dir.glob(f"P_{mod}_{src_field}_*_seg.nii.gz"))
                if len(src_seg_files) < EXPECTED_FILES_PER_SUBTASK:
                    warnings.append(
                        f"only {len(src_seg_files)}/{EXPECTED_FILES_PER_SUBTASK} seg files in {src_seg_dir}"
                    )

                if not dry_run:
                    dst_seg_dir.mkdir(parents=True, exist_ok=True)

                for src_file in src_seg_files:
                    file_id = extract_id(src_file.name, mod, src_field)
                    if file_id is None:
                        warnings.append(f"unparseable seg filename: {src_file}")
                        continue
                    dst_file = dst_seg_dir / f"P_{mod}_{tgt_field}_{file_id}_seg.nii.gz"
                    if dry_run:
                        print(f"  [dry-run] {src_file} -> {dst_file}")
                    else:
                        # Seg is at 0.5 mm (NN-resampled from SynthSeg by
                        # segment_predictions.py); the same axial slice keeps
                        # seg and pred on the same (364, 436, 30) grid.
                        _copy_with_axial_clip(src_file, dst_file)
                    copied_seg += 1

    print(f"  copied pred: {copied_pred}/{expected_pred}", end="")
    if needs_seg:
        print(f", seg: {copied_seg}/{expected_seg}")
    else:
        print()

    return copied_pred, copied_seg, warnings


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build Submission directory tree from Baseline sweep predictions."
    )
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Source of Baseline sweep predictions. Default: $INFERENCE_DIR from .env",
    )
    p.add_argument(
        "--predictions-seg-dir",
        type=Path,
        default=None,
        help="Source of SynthSeg segmentations (mirror tree of --predictions-dir, Task 1/2 only). Default: $PREDICTIONS_SEG_DIR from .env",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the packaged submission tree. Default: $SUBMISSION_DIR from .env",
    )
    p.add_argument(
        "--tasks",
        default="task1,task2,task3",
        help="comma-separated subset, default: task1,task2,task3",
    )
    for task, (m_def, mo_def, e_def) in DEFAULTS.items():
        p.add_argument(f"--{task}-method", default=m_def,
                       help=f"Method to pack for {task} (default: {m_def}). E.g. cut, cyclegan, stargan_v2.")
        p.add_argument(f"--{task}-mode", default=mo_def,
                       help=f"Training mode to pack for {task} (default: {mo_def}). E.g. retro_scratch, pro_scratch, pro_pretrained.")
        p.add_argument(f"--{task}-epoch", default=e_def,
                       help=f"Epoch tag to pack for {task} (default: {e_def}). E.g. epoch100, step200000.")
    p.add_argument("--dry-run", action="store_true", help="print src->dst mapping, copy nothing")
    p.add_argument(
        "--clean",
        action="store_true",
        help="remove output {task}/ dirs before building (default: overwrite per-file)",
    )
    args = p.parse_args()

    if args.predictions_dir is None or args.predictions_seg_dir is None or args.output_dir is None:
        from mrixfields.env import get_inference_dir, get_predictions_seg_dir, get_submission_dir
        if args.predictions_dir is None:
            args.predictions_dir = Path(get_inference_dir())
        if args.predictions_seg_dir is None:
            args.predictions_seg_dir = Path(get_predictions_seg_dir())
        if args.output_dir is None:
            args.output_dir = Path(get_submission_dir())

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in PAIRS:
            print(f"ERROR: unknown task {t!r}, expected one of {list(PAIRS)}", file=sys.stderr)
            return 2

    if not args.predictions_dir.is_dir():
        print(f"ERROR: --predictions-dir not found: {args.predictions_dir}", file=sys.stderr)
        return 2

    if args.clean and not args.dry_run:
        for t in tasks:
            target = args.output_dir / t
            if target.exists():
                print(f"  [clean] removing {target}")
                shutil.rmtree(target)

    all_warnings: list[str] = []
    totals = {"pred": 0, "seg": 0}
    for t in tasks:
        method = getattr(args, f"{t}_method")
        mode = getattr(args, f"{t}_mode")
        epoch_tag = getattr(args, f"{t}_epoch")
        cp, cs, ws = build_one_task(
            t, method, mode, epoch_tag,
            args.predictions_dir, args.predictions_seg_dir, args.output_dir,
            args.dry_run,
        )
        totals["pred"] += cp
        totals["seg"] += cs
        all_warnings.extend(f"[{t}] {w}" for w in ws)

    print("\n=== Summary ===")
    print(f"  total pred copied: {totals['pred']}")
    print(f"  total seg  copied: {totals['seg']}")
    if all_warnings:
        print(f"\n=== Warnings ({len(all_warnings)}) ===")
        for w in all_warnings:
            print(f"  {w}")
    else:
        print("  (no warnings)")

    if not args.dry_run:
        print(
            "\nNext step (when you're ready to upload to Synapse):"
            f"\n  cd {args.output_dir}/task1 && zip -r ~/task1.zip T1W T2W T2FLAIR"
            f"\n  cd {args.output_dir}/task2 && zip -r ~/task2.zip T1W T2W T2FLAIR"
            f"\n  cd {args.output_dir}/task3 && zip -r ~/task3.zip T1W T2W T2FLAIR"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
