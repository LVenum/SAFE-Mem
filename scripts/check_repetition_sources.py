#!/usr/bin/env python
"""How MPBench's flooding attack is laid out, and therefore what a source cap buys."""

from __future__ import annotations

import collections

from ams.data.mpbench import load_mpbench
from ams.memory.typed import split_artifacts


def main() -> int:
    cases, _ = load_mpbench()
    rep = [c for c in cases if c.is_adversarial and c.raw.get("payload_variants")]
    print(f"{len(rep)} repetition cases\n")

    counts = collections.Counter(str(c.raw.get("repetition_count")) for c in rep)
    print("variants per case:", dict(sorted(counts.items())))

    same = multi = unplaced = 0
    arts = collections.Counter()
    for c in rep:
        parts = split_artifacts(c.context)
        arts[len(parts)] += 1
        where = set()
        for v in c.raw["payload_variants"]:
            head = v.strip()[:40]
            for i, p in enumerate(parts):
                if head in p:
                    where.add(i)
                    break
        if not where:
            unplaced += 1
        elif len(where) == 1:
            same += 1
        else:
            multi += 1

    total = same + multi
    print("artifacts per context:", dict(sorted(arts.items())[:8]))
    print(f"\nparaphrases confined to one artifact: {same}/{total} ({same / total:.0%})")
    print(f"spread over several artifacts:        {multi}/{total} ({multi / total:.0%})")
    print(f"not located in the context:           {unplaced}")
    print(
        f"\nA source cap of 1 collapses the flood in the {same / total:.0%} of cases where it "
        "sits in one artifact,\nand does nothing in the rest. Run the cap ablation if that "
        "fraction is worth a row."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
