#!/usr/bin/env python
"""Third labelling round (rebuttal): claim kind (50) and question target (30).

    python scripts/label_set_v3.py make --outdir ~/Downloads
    python scripts/label_set_v3.py score --claims ~/Downloads/label_set_v3_claims.csv \
        --targets ~/Downloads/label_set_v3_targets.csv

`make` re-runs the frozen v3.1 write path (GPT-4o-mini, the config of the reported
LoCoMo run) on a sample of inputs; with the shared cache it costs nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from ams.data.locomo import load_locomo
from ams.data.mpbench import load_mpbench
from ams.eval.build import build_memory
from ams.eval.context import Config, RunContext
from ams.eval.labels import _read
from ams.eval.labels_v2 import write
from ams.eval.labels_v3 import (
    build_claims,
    build_targets,
    claims_of,
    score_claims,
    score_targets,
)

R = Path("results/ams-results/v31")
KEY_C, KEY_T = (
    Path("data/labels/label_key_v3_claims.json"),
    Path("data/labels/label_key_v3_targets.json"),
)


def make(outdir: Path) -> None:
    run = next(R.glob("v31_locomo_withhold_a_4omini-*"))
    raw = json.loads((run / "manifest.json").read_text())["config"]
    raw["name"] = "label_set_v3"
    ctx = RunContext(Config(raw=raw))
    memory = build_memory(ctx.config, ctx)
    rng = random.Random(5)

    convs = load_locomo()
    lrows = []
    for c in rng.sample(convs, 5):
        sid = f"lab3:{c.sample_id}"
        memory.reset(sid)
        for s in c.sessions[:2]:
            memory.write_dialogue(
                session_id=sid,
                turns=[(t.speaker, t.render().split(" said, ", 1)[-1]) for t in s.turns],
                date=s.date_time,
                principal=c.speakers[0],
            )
        lrows += [{**r, "ref": c.sample_id} for r in claims_of(memory, sid)]

    cases = {c.cid: c for c in load_mpbench("data/mpbench")[0]}
    out = _read(
        next(R.glob("v31_mpbench_withhold_4omini_pool6240-*")) / "mpbench_outcomes.csv", "cid"
    )
    warranted = sorted(cid for cid, r in out.items() if r["benign_warranted"] == "1")
    mrows = []
    for cid in rng.sample(warranted, 30):
        sid, case = f"lab3:{cid}", cases[cid]
        memory.reset(sid)
        memory.write(session_id=sid, user_query=case.user_query, context=case.context)
        mrows += [{**r, "ref": cid} for r in claims_of(memory, sid)]

    csheet, ckey = build_claims(lrows, mrows)

    qa = _read(run / "locomo_outcomes.csv", "qid")
    by_conv = {c.sample_id: c for c in convs}
    qs = []
    for qid, r in qa.items():
        if r["judge_correct"] not in ("0", "1"):
            continue
        c = by_conv[r["sample_id"]]
        qs.append(
            {
                "qid": qid,
                "question": r["question"],
                "speakers": c.speakers,
                "principal": c.speakers[0],
                "required": r["required_type"],
            }
        )
    tsheet, tkey = build_targets(qs, memory.use_type.subject)

    outdir.mkdir(parents=True, exist_ok=True)
    write(csheet, outdir / "label_set_v3_claims.csv")
    write(tsheet, outdir / "label_set_v3_targets.csv")
    KEY_C.write_text(json.dumps(ckey, indent=1))
    KEY_T.write_text(json.dumps(tkey, indent=1))
    print(f"{len(csheet)} claims, {len(tsheet)} targets -> {outdir}; run dir {ctx.dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--outdir", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--claims", type=Path)
    sc.add_argument("--targets", type=Path)
    a = ap.parse_args(argv)
    csv.field_size_limit(10**9)
    if a.cmd == "make":
        make(a.outdir.expanduser())
        return 0
    if a.claims:
        print(
            "claims:", json.dumps(score_claims(a.claims, json.loads(KEY_C.read_text())), indent=1)
        )
    if a.targets:
        print(
            "targets:",
            json.dumps(score_targets(a.targets, json.loads(KEY_T.read_text())), indent=1),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
