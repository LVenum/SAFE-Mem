#!/usr/bin/env python
"""Spec §10.5 attack 6 (piggybacking on a genuine retention request), write path."""

from __future__ import annotations

import argparse
import sys

from ams.core.env import load_env
from ams.data.mpbench import load_mpbench
from ams.eval.attack6 import build_cases, run_case
from ams.eval.backbone import apply_backbone
from ams.eval.build import build_memory
from ams.eval.context import Config, RunContext
from ams.eval.judge import MemoryJudge
from ams.llm import map_concurrent, model_of


def main(argv=None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--yes", action="store_true", help="confirm a run above ~1k calls")
    ap.add_argument("--backbone", help="a name in configs/backbones.yaml (gpt56 | 4omini)")
    args = ap.parse_args(argv)

    load_env()
    cfg = Config.load(args.config)
    apply_backbone(cfg, args.backbone)
    ctx = RunContext(cfg)
    cases, _ = load_mpbench(cfg.get("dataset.root", "data/mpbench"))
    a6 = build_cases(cases, ctx.prompts, seed=cfg.get("seed", 0))
    calls = len(a6) * 4
    print(f"{len(a6)} attack-6 cases -> <= ~{calls} calls (CLAUDE.md rule 14)")
    if calls > 1000 and not args.yes:
        print("above ~1k calls: re-run with --yes to proceed")
        return 1

    memory = build_memory(cfg, ctx)
    jspec = cfg.require("models.judge")
    judge = MemoryJudge(
        ctx.client(jspec, stage="judge"),
        model=model_of(jspec),
        prompts=ctx.prompts,
        entry_version=cfg.get("prompts.judge_entry", "v2"),
    )

    def one(c):
        try:
            return run_case(memory, judge, c)
        except Exception as exc:
            return {"cid": c.cid, "variant": c.variant, "error": f"{type(exc).__name__}"}

    rows = map_concurrent(one, a6, workers=cfg.get("runtime.workers", 8))
    print(
        f"\n{'variant':<9}{'n':>5}{'error':>7}{'poison written':>17}{'poison elevated':>18}"
        f"{'any elevated':>15}"
    )
    for v in ("a", "b"):
        rs = [r for r in rows if r["variant"] == v]
        ok = [r for r in rs if "error" not in r]
        n = len(ok) or 1
        pw = sum(r["poison_written"] for r in ok)
        pe = sum(r["poison_elevated"] for r in ok)
        ae = sum(r["n_elevated"] > 0 for r in ok)
        print(
            f"{v:<9}{len(rs):>5}{len(rs) - len(ok):>7}{pw:>10} {pw / n:>5.1%}"
            f"{pe:>11} {pe / n:>5.1%}{ae:>8} {ae / n:>5.1%}"
        )
    ctx.write_json("attack6.json", {"rows": rows})
    ctx.finalise()
    print(f"\nper-case rows: {ctx.path('attack6.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
