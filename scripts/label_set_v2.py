#!/usr/bin/env python
"""Second labelling round: use type (150) and answer judge (200)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ams.data.mpbench import load_mpbench
from ams.eval.labels import _read, score
from ams.eval.labels_v2 import build_judge, build_use, score_judge, write

R = Path("results/ams-results")
KEY_USE, KEY_JUDGE = (
    Path("data/labels/label_key_v2_use.json"),
    Path("data/labels/label_key_v2_judge.json"),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--outdir", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--use", type=Path)
    sc.add_argument("--judge", type=Path)
    a = ap.parse_args(argv)
    csv.field_size_limit(10**9)
    if a.cmd == "make":
        cases = {c.cid: c for c in load_mpbench("data/mpbench")[0]}
        v31 = {
            bb: next(R.glob(f"v31/v31_mpbench_withhold_{bb}_pool6240-*"))
            for bb in ("gpt56", "4omini")
        }
        mp = {bb: _read(d / "mpbench_outcomes.csv", "cid") for bb, d in v31.items()}
        lc = {
            bb: _read(
                next(R.glob(f"v31/v31_locomo_withhold_a_{bb}-*")) / "locomo_outcomes.csv", "qid"
            )
            for bb in ("gpt56", "4omini")
        }
        old = json.loads(Path("data/labels/label_key_v1.json").read_text())
        exclude = {k["ref"] for k in old.values()}
        use, ukey = build_use(cases=cases, mp=mp, lc=lc, exclude=exclude)

        def nodef(bb):
            d = sorted(R.glob(f"final/mpbench_{bb}_pool6240-*/mpbench_outcomes.csv"))[-1]
            with open(d, newline="", encoding="utf-8") as f:
                return {r["cid"]: r for r in csv.DictReader(f) if r["defence"] == "no_defence"}

        runs = {f"nodef_{bb}": nodef(bb) for bb in ("gpt56", "4omini")}
        runs |= {f"ours_{bb}": mp[bb] for bb in ("gpt56", "4omini")}
        judge, jkey = build_judge(cases=cases, runs=runs)
        out = a.outdir.expanduser()
        write(use, out / "label_set_v2_use.csv")
        write(judge, out / "label_set_v2_judge.csv")
        KEY_USE.write_text(json.dumps(ukey, indent=1))
        KEY_JUDGE.write_text(json.dumps(jkey, indent=1))
        print(
            f"{len(use)} use items, {len(judge)} judge items -> {out}"
            f"\nkeys (do not open): {KEY_USE}, {KEY_JUDGE}"
        )
        return 0
    if a.use:
        for name, s in score(a.use.expanduser(), json.loads(KEY_USE.read_text())).items():
            print("use  ", name, s)
    if a.judge:
        print("judge", score_judge(a.judge.expanduser(), json.loads(KEY_JUDGE.read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
