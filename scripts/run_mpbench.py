#!/usr/bin/env python
"""Defence baselines on MPBench through the write -> retrieve -> act lifecycle."""

from __future__ import annotations

import argparse
import sys

from ams.core.env import load_env
from ams.data.mpbench import load_mpbench
from ams.eval.backbone import apply_backbone
from ams.eval.build import build_memory
from ams.eval.context import Config, RunContext
from ams.eval.defences import build_defence
from ams.eval.judge import MemoryJudge
from ams.eval.lifecycle import run_lifecycle
from ams.eval.tables import write_csv
from ams.llm import ChatRequest, Message, model_of

_EXTRA_PER_READ = {
    "a_memguard": 6,
    "llm_sanitiser": 8,
    "memlineage_lm": 5,
    "memlineage_coarse": 1,
    "integrity_gate": 1,
}


def projected_calls(cfg: Config, cases, defences) -> int:
    """Upper bound: one extraction per write (typed adds a use-type call and
    conflict checks per read), then per defence and read: entry judge x2,
    answer with and without memory, answer judge x2, plus the defence's own."""
    reads = sum(bool((c.retrieval_query or "").strip()) for c in cases)
    typed = cfg.get("memory_system") == "typed"
    writes = len(cases) * (2 if cfg.get("write_isolation") == "drift" else 1)
    per_read = 6 + (3 if typed else 0)
    return writes + sum(reads * (per_read + _EXTRA_PER_READ.get(d, 0)) for d in defences)


def main(argv=None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="first N cases (stratified adv/benign by order)")
    ap.add_argument("--defences", help="comma-separated subset")
    ap.add_argument(
        "--pool-size", type=int, help="cases sharing one store (default: config pool_size or 1)"
    )
    ap.add_argument("--yes", action="store_true", help="confirm a run above ~1k calls")
    ap.add_argument("--backbone", help="a name in configs/backbones.yaml (gpt56 | 4omini)")
    args = ap.parse_args(argv)

    load_env()
    cfg = Config.load(args.config)
    apply_backbone(cfg, args.backbone)
    cases, canon = load_mpbench(cfg.get("dataset.root", "data/mpbench"))
    from ams.core.prompts import PromptLibrary
    from ams.eval.adaptive import apply as adapt

    cases = adapt(cases, cfg.get("adaptive.variant"), PromptLibrary())
    if args.limit:
        adv = [c for c in cases if c.is_adversarial][: args.limit // 2]
        ben = [c for c in cases if not c.is_adversarial][: args.limit - len(adv)]
        cases = adv + ben
    defences = args.defences.split(",") if args.defences else cfg.get("defences")
    pool_size = args.pool_size or cfg.get("pool_size", 1)
    cfg.raw["defences"] = list(defences)
    if args.limit:
        cfg.raw["limit"] = args.limit
    cfg.raw["pool_size"] = pool_size
    if pool_size > 1:
        cfg.raw["name"] = f"{cfg.get('name')}_pool{pool_size}"
    workers = cfg.get("runtime.workers", 6)
    print(
        f"MPBench: {len(cases)} cases ({sum(c.is_adversarial for c in cases)} adv)  "
        f"defences={defences}"
    )
    calls = projected_calls(cfg, cases, defences)
    print(f"projected <= ~{calls} model calls before cache hits (CLAUDE.md rule 14)")
    if calls > 1000 and not args.yes and not cfg.get("models.agent", "").startswith("echo"):
        print("above ~1k calls: re-run with --yes to proceed")
        return 1

    ctx = RunContext(cfg)
    m = {r: cfg.require(f"models.{r}") for r in ("memory", "agent", "judge", "embed")}
    memory = build_memory(cfg, ctx)
    system = cfg.get("memory_system", "simple")
    judge = MemoryJudge(
        ctx.client(m["judge"], stage="judge"),
        model=model_of(m["judge"]),
        prompts=ctx.prompts,
        entry_version=cfg.get("prompts.judge_entry", "v2"),
        answer_version=cfg.get("prompts.judge_answer", "v1"),
    )
    agent = ctx.client(m["agent"], stage="agent_answer")
    typed_answer = system == "typed" and cfg.get("typed.mode") != "off"
    answer_prompt = ctx.prompts.get(
        "answer_typed" if typed_answer else "answer",
        cfg.get("prompts.answer_typed" if typed_answer else "prompts.answer", "v1"),
    )

    def answer_fn(query: str, memories: list[str], *, required: str = "") -> str:
        body = "\n".join(f"- {t}" for t in memories) or "(nothing)"
        rendered = (
            answer_prompt.render(query=query, memory=body, required=required or "description")
            if typed_answer
            else answer_prompt.render(query=query, memory=body)
        )
        return agent.complete(
            ChatRequest(
                messages=(Message("user", rendered),),
                model=model_of(m["agent"]),
                temperature=0.0,
                max_tokens=200,
            )
        ).text.strip()

    rows, reports = [], []
    for kind in defences:
        defence = build_defence(kind, cfg, m["agent"], ctx)
        out, rep = run_lifecycle(
            cases,
            memory,
            defence,
            judge,
            answer_fn=answer_fn,
            workers=workers,
            pool_size=pool_size,
            seed=cfg.get("seed", 0),
            k=cfg.get("retrieve_k", 4),
        )
        rows += out
        reports.append(rep)
        print("  " + rep.line())

    write_csv(ctx.path("mpbench_outcomes.csv"), rows)
    stats = getattr(getattr(memory, "inner", memory), "graph_stats", None)
    if stats:
        ctx.write_json("graph_stats.json", stats())
    ctx.write_json(
        "mpbench_summary.json",
        {
            "n_cases": len(cases),
            "pool_size": pool_size,
            "write_gate": cfg.get("write_gate"),
            "reports": [r.to_dict() for r in reports],
            "canonicalisation": canon.to_dict(),
        },
    )
    print("\n" + "\n".join(r.line() for r in reports))
    ctx.finalise()
    u = ctx.ledger.summary()
    print(
        f"\napi calls {u['total_calls'] - u['total_cache_hits']}  cost ${u['total_cost_usd']:.2f}"
        f"\nresults: {ctx.dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
