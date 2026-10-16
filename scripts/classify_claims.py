#!/usr/bin/env python
"""Label every MPBench adversarial payload with its claim kind."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ams.core.env import load_env
from ams.data.mpbench import load_mpbench
from ams.eval.claims import classify
from ams.llm import build_client, model_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--out", default="data/mpbench/claim_types_v1.jsonl")
    args = ap.parse_args(argv)
    load_env()
    cases, _ = load_mpbench("data/mpbench")
    adv = [c for c in cases if c.is_adversarial and c.expected_memory.strip()]
    kinds = classify(
        build_client(args.model),
        model=model_of(args.model),
        payloads=[c.expected_memory for c in adv],
    )
    with Path(args.out).open("w") as f:
        for c, k in zip(adv, kinds, strict=True):
            f.write(
                json.dumps(
                    {
                        "cid": c.cid,
                        "attack_type": c.attack_type,
                        "payload": c.expected_memory,
                        "kind": k,
                    }
                )
                + "\n"
            )
    from collections import Counter

    print(Counter(kinds), "->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
