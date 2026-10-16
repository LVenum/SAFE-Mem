#!/usr/bin/env python
"""Go / no-go: does MPBench actually need anything above `description`?"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys

from ams.core.env import load_env
from ams.data.mpbench import load_mpbench
from ams.eval.context import Config, RunContext
from ams.eval.use_type import UseTypeClassifier
from ams.llm import model_of

THRESHOLD = 0.25


def main(argv=None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mpbench_typed.yaml")
    ap.add_argument("--sample", type=int, default=200, help="stratified by attack goal")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    load_env()
    cfg = Config.load(args.config)
    cfg.override("name", "check_use_type")
    ctx = RunContext(cfg)
    spec = cfg.require("models.memory")
    clf = UseTypeClassifier(
        client=ctx.client(spec, stage="use_type"),
        model=model_of(spec),
        prompts=ctx.prompts,
        version=cfg.get("prompts.classify_use_type", "v1"),
    )

    cases, _ = load_mpbench(cfg.get("dataset.root", "data/mpbench"))
    adv = [c for c in cases if c.is_adversarial and c.retrieval_query.strip()]
    rng = random.Random(args.seed)
    by_goal: dict[str, list] = collections.defaultdict(list)
    for c in adv:
        by_goal[c.goal].append(c)
    per = max(1, args.sample // max(1, len(by_goal)))
    sample = [c for g in sorted(by_goal) for c in rng.sample(by_goal[g], min(per, len(by_goal[g])))]

    print(f"{len(sample)} adversarial queries, stratified over {len(by_goal)} goals\n")
    rows = []
    counts: collections.Counter = collections.Counter()
    per_goal: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for c in sample:
        rung = clf.required(c.retrieval_query)
        counts[rung.name.lower()] += 1
        per_goal[c.goal][rung.name.lower()] += 1
        rows.append(
            {
                "cid": c.cid,
                "goal": c.goal,
                "query": c.retrieval_query,
                "expected_memory": c.expected_memory,
                "required": rung.name.lower(),
            }
        )

    n = len(sample)
    print(f"{'required kind':<20}{'n':>6}{'share':>9}")
    print("-" * 35)
    for name, k in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{name:<20}{k:>6}{k / n:>8.1%}")
    print()
    print(f"{'goal':<28}" + "".join(f"{x[:6]:>8}" for x in sorted(counts)))
    for goal in sorted(per_goal):
        print(f"{goal:<28}" + "".join(f"{per_goal[goal][x]:>8}" for x in sorted(counts)))

    share = counts["description"] / n if n else 1.0
    print(
        f"\ndescription share: {share:.1%}   threshold: {THRESHOLD:.0%}   "
        f"unparsed replies: {clf.n_unparsed}"
    )
    verdict = "GO" if share <= THRESHOLD else "NO-GO"
    print(
        f"\n{verdict} — "
        + (
            "the gate has work to do on MPBench; proceed to the main runs."
            if verdict == "GO"
            else "most adversarial queries need only a description, so the use-dependent "
            "gate cannot carry MPBench. Return to the design (spec §2.1), do not tune."
        )
    )

    ctx.write_json(
        "use_type_sample.json",
        {
            "threshold": THRESHOLD,
            "share_description": share,
            "counts": dict(counts),
            "verdict": verdict,
            "rows": rows,
        },
    )
    print(f"\nper-query labels for human checking: {ctx.path('use_type_sample.json')}")
    print(json.dumps({"verdict": verdict, "share_description": round(share, 4)}))
    return 0 if verdict == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
