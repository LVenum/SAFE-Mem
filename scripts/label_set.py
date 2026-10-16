#!/usr/bin/env python
"""Human labels for the gate's two classifiers (spec §7.4)."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

from ams.data.mpbench import load_mpbench
from ams.eval.labels import build, save_key, score, write_sheet

KEY = Path("data/labels/label_key_v1.json")
R = "results/ams-results"


def latest(pattern: str) -> Path:
    runs = [d for d in sorted(glob.glob(pattern)) if Path(d, "manifest.json").exists()]
    if not runs:
        sys.exit(f"no finished run matches {pattern}")
    return Path(runs[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("sheet", type=Path)
    sc.add_argument(
        "--exclude",
        type=Path,
        default=Path("data/labels/calibration_ids.txt"),
        help="ids not to score (the calibration round); pass /dev/null to score all",
    )
    args = ap.parse_args(argv)

    if args.cmd == "make":
        csv.field_size_limit(10**9)
        cases = {c.cid: c for c in load_mpbench("data/mpbench")[0]}
        mp = {
            bb: latest(f"{R}/v31/v31_mpbench_withhold_{bb}_pool6240-*") / "mpbench_outcomes.csv"
            for bb in ("gpt56", "4omini")
        }
        lc = {
            bb: latest(f"{R}/v31/v31_locomo_withhold_a_{bb}-*") / "locomo_outcomes.csv"
            for bb in ("gpt56", "4omini")
        }
        ret = latest("results/check_retention_split_gpt56-*") / "retention_split.json"
        sheet, key = build(
            mpbench_cases=cases,
            mpbench_outcomes=mp,
            locomo_outcomes=lc,
            retention_rows=json.loads(ret.read_text())["rows"],
        )
        write_sheet(sheet, args.out.expanduser())
        save_key(key, KEY)
        parts = {p: sum(r["part"] == p for r in sheet) for p in "AB"}
        print(f"{len(sheet)} items {parts} -> {args.out}\nkey (do not open while labelling): {KEY}")
        return 0

    key = json.loads(KEY.read_text())
    skip = set(args.exclude.read_text().split()) if args.exclude.exists() else set()
    key = {i: k for i, k in key.items() if i not in skip}
    print(f"scoring {len(key)} items ({len(skip)} calibration items excluded)")
    result = score(args.sheet.expanduser(), key)
    for name, s in result.items():
        print(f"{name:10} " + "  ".join(f"{k} {v}" for k, v in s.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
