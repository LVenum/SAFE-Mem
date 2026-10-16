#!/usr/bin/env python
"""LoCoMo QA: memory systems beside the full-context baseline (spec §7.3)."""

from __future__ import annotations

import argparse
import sys

from ams.core.env import load_env
from ams.data.locomo import load_locomo
from ams.eval.backbone import apply_backbone
from ams.eval.build import build_memory
from ams.eval.context import Config, RunContext
from ams.eval.defences import build_defence
from ams.eval.locomo import LoCoMoQA, run_full_context, run_memory, summarise
from ams.eval.tables import write_csv
from ams.llm import ClientSpec, model_of
from ams.llm.pricing import estimate_cost


def main(argv=None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="first N conversations")
    ap.add_argument("--yes", action="store_true", help="confirm a run above ~1k calls")
    ap.add_argument("--backbone", help="a name in configs/backbones.yaml (gpt56 | 4omini)")
    args = ap.parse_args(argv)

    load_env()
    cfg = Config.load(args.config)
    apply_backbone(cfg, args.backbone)
    if cfg.get("dataset.kind", "locomo") == "longmemeval":
        from ams.data.longmemeval import load_longmemeval

        convs = load_longmemeval(cfg.require("dataset.path"))[: args.limit]
    else:
        convs = load_locomo(cfg.get("dataset.path", "data/locomo/locomo10.json"))[: args.limit]
    systems = cfg.get("systems", ["full_context", "memory"])
    n_qa = sum(len(c.qa) for c in convs)
    n_sessions = sum(len(c.sessions) for c in convs)
    full_tokens = sum(len(c.render()) // 4 * len(c.qa) for c in convs)

    calls = 2 * n_qa * len(systems) + n_sessions * ("memory" in systems)
    agent = ClientSpec.parse(cfg.require("models.agent"))
    cost = estimate_cost(agent.provider, agent.model, full_tokens, 40 * n_qa) * (
        "full_context" in systems
    ) + estimate_cost(agent.provider, agent.model, 1500 * n_qa * 2, 40 * n_qa * 2)
    print(
        f"LoCoMo: {len(convs)} conversations, {n_sessions} sessions, {n_qa} questions; "
        f"systems={systems}\nprojected ~{calls} calls, ~${cost:.2f} (cache hits are free)"
    )
    if calls > 1000 and not args.yes:
        print("above ~1k calls: re-run with --yes to proceed")
        return 1

    if args.limit:
        cfg.raw["limit"] = args.limit
    ctx = RunContext(cfg)
    m = {r: cfg.require(f"models.{r}") for r in ("agent", "judge")}
    qa = LoCoMoQA(
        agent=ctx.client(m["agent"], stage="agent_answer"),
        agent_model=model_of(m["agent"]),
        judge=ctx.client(m["judge"], stage="judge"),
        judge_model=model_of(m["judge"]),
        prompts=ctx.prompts,
    )
    workers = cfg.get("runtime.workers", 8)
    rows = []
    if "full_context" in systems:
        rows += run_full_context(convs, qa, workers=workers)
    if "memory" in systems:
        memory = build_memory(cfg, ctx)
        defences = [
            build_defence(kind, cfg, m["agent"], ctx)
            for kind in cfg.get("defences", ["no_defence"])
        ]
        rows += run_memory(
            convs,
            memory,
            qa,
            k=cfg.get("retrieve_k", 4),
            workers=workers,
            principal=cfg.get("locomo.principal"),
            defences=defences,
        )

    summary = summarise(rows)
    write_csv(ctx.path("locomo_outcomes.csv"), rows)
    if "memory" in systems and hasattr(memory, "graph_stats"):
        ctx.write_json("graph_stats.json", memory.graph_stats())
    ctx.write_json("locomo_summary.json", {"n_conversations": len(convs), "systems": summary})
    for name, s in summary.items():
        o = s["overall_1to4"]
        cats = "  ".join(
            f"c{c} {v['f1']:.3f}/{v['judge'] if v['judge'] is not None else '-'}"
            for c, v in s["by_category"].items()
        )
        print(f"{name:<40} F1 {o['f1']:.3f}  judge {o['judge']}  (n={o['n']})  {cats}")
    ctx.finalise()
    u = ctx.ledger.summary()
    print(
        f"\napi calls {u['total_calls'] - u['total_cache_hits']}  cost ${u['total_cost_usd']:.2f}"
        f"\nresults: {ctx.dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
