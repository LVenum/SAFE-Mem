"""The MPBench baselines as MINJA arms, served from the main environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO

from ams.baselines.drift import InjectionIsolator
from ams.core.env import load_env
from ams.eval.backbone import apply_backbone
from ams.eval.context import Config, RunContext
from ams.eval.defences import build_defence
from ams.llm import model_of
from ams.memory.base import MemoryEntry

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "minja_baselines.yaml"
ARMS = {
    "DRIFT": "drift",
    "RandEC": "randec_distil",
    "LLM-sanitizer": "llm_sanitiser",
    "MemLineage": "memlineage_coarse",
}


def build(arm: str, cfg: Config, ctx: RunContext):
    agent = cfg.require("models.agent")
    if ARMS[arm] == "drift":
        return InjectionIsolator(
            ctx.client(agent, stage="drift_isolator"), model_of(agent), prompts=ctx.prompts
        )
    return build_defence(ARMS[arm], cfg, agent, ctx)


def filter_records(
    defence, query: str, records: list[dict]
) -> tuple[list[int], dict[int, str], dict[int, str]]:
    """(kept positions, rewritten text by position, reason by dropped position)."""
    kept, rewrite, why = [], {}, {}
    if isinstance(defence, InjectionIsolator):
        for i, r in enumerate(records):
            masked, removed = defence(query, r["text"])
            if not masked.strip():
                why[i] = "masked to nothing"
                continue
            kept.append(i)
            if removed:
                rewrite[i] = masked
        return kept, rewrite, why
    entries = [
        MemoryEntry(str(i), r["text"], metadata={"user_query": r["question"]})
        for i, r in enumerate(records)
    ]
    res = defence.filter(query, entries)
    out = {int(m.entry_id): m.text for m in res.kept}
    for i, r in enumerate(records):
        if i not in out:
            why[i] = res.reason
            continue
        kept.append(i)
        if out[i] != r["text"]:
            rewrite[i] = out[i]
    return kept, rewrite, why


def serve(defence, inp: IO[str], out: IO[str]) -> None:
    """One request per line in, one reply per line out, until EOF."""
    for line in inp:
        if not line.strip():
            continue
        req = json.loads(line)
        kept, rewrite, why = filter_records(defence, req["query"], req["records"])
        out.write(json.dumps({"kept": kept, "rewrite": rewrite, "why": why}) + "\n")
        out.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--backbone", help="a name in configs/backbones.yaml")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--results-root", required=True, help="the MINJA arm directory")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    reply, sys.stdout = sys.stdout, sys.stderr
    load_env()
    cfg = Config.load(args.config)
    apply_backbone(cfg, args.backbone)
    ctx = RunContext(cfg, results_root=args.results_root, run_id=args.run_id)
    serve(build(args.arm, cfg, ctx), sys.stdin, reply)
    ctx.finalise({"minja_arm": args.arm})
    return 0
