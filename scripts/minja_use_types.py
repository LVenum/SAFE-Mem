#!/usr/bin/env python
"""Classify what kind of claim every MINJA question needs, before the run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ams.core.env import load_env
from ams.eval.backbone import apply_backbone
from ams.eval.context import Config, RunContext
from ams.eval.use_type import UseTypeClassifier
from ams.llm import map_concurrent, model_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bench", type=Path)
    ap.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[1] / "configs/minja_typed.yaml")
    )
    ap.add_argument("--backbone", help="a name in configs/backbones.yaml (gpt56 | 4omini)")
    args = ap.parse_args(argv)

    load_env()
    cfg = Config.load(args.config)
    apply_backbone(cfg, args.backbone)
    questions = sorted(
        {
            e["template"]
            for name in ("poison_stream.json", "clean_stream.json", "victim_test.json")
            if (args.bench / name).exists()
            for e in json.loads((args.bench / name).read_text())
        }
    )
    ctx = RunContext(cfg)
    clf = UseTypeClassifier(
        client=ctx.client(cfg.require("models.memory"), stage="use_type"),
        model=model_of(cfg.require("models.memory")),
        prompts=ctx.prompts,
        version=cfg.get("prompts.classify_use_type", "v1"),
    )
    rungs = map_concurrent(
        lambda q: clf.required(q).name.lower(), questions, workers=cfg.get("runtime.workers", 8)
    )
    out = dict(zip(questions, rungs, strict=True))
    (args.bench / "use_types.json").write_text(json.dumps(out, indent=1))
    ctx.write_json("use_types.json", out)
    dist = {r: sum(v == r for v in out.values()) for r in sorted(set(out.values()))}
    print(f"{len(out)} questions classified: {dist}")
    ctx.finalise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
