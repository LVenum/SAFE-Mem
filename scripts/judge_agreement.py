#!/usr/bin/env python
"""Sample cases, run the memory write, ask the judge, and score it against labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ams.core.env import load_env
from ams.data.mpbench import load_mpbench
from ams.eval.agreement import LabelledCase, load_labels, save_for_labelling, score
from ams.eval.judge import MemoryJudge
from ams.llm import ResponseCache, build_client, map_concurrent
from ams.memory import SimpleMemory


def _write(judged, verdicts: str, labels: str) -> None:
    Path(verdicts).parent.mkdir(parents=True, exist_ok=True)
    with open(verdicts, "w", encoding="utf-8") as fh:
        for c in judged:
            fh.write(
                json.dumps(
                    {
                        "cid": c.cid,
                        "stratum": c.stratum,
                        "judge": c.judge,
                        "reason": c.judge_reason,
                        "fallback": c.judge_fallback,
                        "expected_memory": c.expected_memory,
                        "stored": list(c.stored),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    save_for_labelling(judged, labels)

    yes = sum(bool(c.judge) for c in judged)
    fb = sum(c.judge_fallback for c in judged)
    print(
        f"\njudge said YES on {yes}/{len(judged)} ({yes / len(judged):.0%}); "
        f"{fb} used the lexical fallback"
    )
    print(f"verdicts (judge's own): {verdicts}")
    print(f"TO LABEL:               {labels}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sys.stdout.reconfigure(line_buffering=True)
    ap.add_argument("--sample", type=int, help="draw and judge N cases, then write labels file")
    ap.add_argument(
        "--from-results",
        help="sample stored entries from a run's mpbench_outcomes.csv instead of re-running the "
        "memory; stratified over the judge's own verdict so negatives are not starved",
    )
    ap.add_argument("--score", action="store_true", help="score the judge against labels")
    ap.add_argument("--labels", default="data/judge_labels.jsonl")
    ap.add_argument("--verdicts", default="data/judge_verdicts.jsonl")
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--entry-version", default="v1", help="judge_entry prompt version")
    args = ap.parse_args(argv)

    load_env()
    cache = ResponseCache(".llm_cache/responses.sqlite")

    if args.sample and args.from_results:
        import csv

        rows = [
            r
            for r in csv.DictReader(open(args.from_results, encoding="utf-8"))
            if r["is_adversarial"] == "1" and r["written_texts"] != "()"
        ]
        yes = [r for r in rows if r["poison_written"] in ("True", "1")]
        no_hit = [r for r in rows if r not in yes and r["attack_succeeded"] in ("True", "1")]
        no = [r for r in rows if r not in yes and r not in no_hit]
        per = max(1, args.sample // 3)
        picked = []
        for bucket in (yes, no_hit, no):
            step = max(1, len(bucket) // per)
            picked.extend(bucket[::step][:per])
        picked = picked[: args.sample]
        by_cid = {c.cid: c for c in load_mpbench()[0] if c.is_adversarial}
        judge = MemoryJudge(
            build_client(args.model, cache=cache, stage="judge"),
            model=args.model.split(":", 1)[1],
            entry_version=args.entry_version,
        )

        def one_row(r):
            case = by_cid[r["cid"]]
            stored = tuple(eval(r["written_texts"]))  # noqa: S307 - our own CSV, tuple repr
            v = judge.judge_entry(case.expected_memory, stored)
            return LabelledCase(
                cid=case.cid,
                stratum=case.attack_signal,
                expected_memory=case.expected_memory,
                stored=stored,
                human=False,
                judge=v.hit,
                judge_reason=v.reason,
                judge_fallback=v.fallback,
            )

        print(f"judging {len(picked)} stored-entry sets from {args.from_results}...")
        judged = list(map_concurrent(one_row, picked, workers=6, progress="judge"))
        _write(judged, args.verdicts, args.labels)
        return 0

    if args.sample:
        cases, _ = load_mpbench()
        adv = [c for c in cases if c.is_adversarial and c.expected_memory]
        per = max(1, args.sample // 3)
        sample = []
        for stratum in ("strong", "moderate", "weak"):
            rows = [c for c in adv if c.attack_signal == stratum]
            step = max(1, len(rows) // per)
            sample.extend(rows[::step][:per])
        sample = sample[: args.sample]

        memory = SimpleMemory(
            build_client(args.model, cache=cache, stage="memory_write"),
            model=args.model.split(":", 1)[1],
            embed_client=build_client("local:", cache=cache, stage="embed"),
            embed_model="all-MiniLM-L6-v2",
        )
        judge = MemoryJudge(
            build_client(args.model, cache=cache, stage="judge"),
            model=args.model.split(":", 1)[1],
            entry_version=args.entry_version,
        )

        def one(case):
            memory.reset(case.cid)
            written = memory.write(
                session_id=case.cid, user_query=case.user_query, context=case.context
            ).written
            stored = tuple(e.text for e in written)
            v = judge.judge_entry(case.expected_memory, stored)
            return LabelledCase(
                cid=case.cid,
                stratum=case.attack_signal,
                expected_memory=case.expected_memory,
                stored=stored,
                human=False,
                judge=v.hit,
                judge_reason=v.reason,
                judge_fallback=v.fallback,
            )

        print(f"running write + judge on {len(sample)} cases...")
        judged = list(map_concurrent(one, sample, workers=6, progress="judge"))

        _write(judged, args.verdicts, args.labels)
        print('\nSet "human": true/false on each row, then re-run with --score.')
        return 0

    if args.score:
        labels = load_labels(args.labels)
        if not labels:
            print(f'no labels found in {args.labels} — set "human" on the rows first')
            return 2
        rows = [json.loads(x) for x in Path(args.verdicts).read_text().splitlines() if x.strip()]
        cases = [
            LabelledCase(
                cid=r["cid"],
                stratum=r["stratum"],
                expected_memory=r["expected_memory"],
                stored=tuple(r["stored"]),
                human=labels[r["cid"]],
                judge=r["judge"],
                judge_reason=r.get("reason", ""),
                judge_fallback=r.get("fallback", False),
            )
            for r in rows
            if r["cid"] in labels
        ]
        rep = score(cases)
        print(rep.line())
        print(json.dumps(rep.to_dict(), indent=2))
        if rep.disagreements:
            print(f"\n--- {len(rep.disagreements)} disagreements ---")
            for c in rep.disagreements[:10]:
                print(f"\n  {c.cid} [{c.stratum}]  judge={c.judge} human={c.human}")
                print(f"    expected: {c.expected_memory[:110]}")
                print(f"    stored  : {(c.stored[0] if c.stored else '(nothing)')[:110]}")
                print(f"    reason  : {c.judge_reason[:110]}")
        if not rep.usable:
            print("\nJUDGE NOT USABLE — fix the prompt before trusting any ASR built on it.")
            return 1
        return 0

    ap.error("pass --sample N or --score")
    return 2


if __name__ == "__main__":
    sys.exit(main())
