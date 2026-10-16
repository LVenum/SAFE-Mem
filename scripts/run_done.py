#!/usr/bin/env python
"""Has this exact experiment already finished? Exit 0 if yes, 1 if not."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

IGNORE = {"name", "fallbacks", "fallback_provider", "runtime"}


def normalise(cfg: dict) -> dict:
    out = {k: v for k, v in sorted(cfg.items()) if k not in IGNORE}
    if out.get("pool_size", 1) == 1:
        out.pop("pool_size", None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--defences")
    ap.add_argument("--pool-size", type=int)
    ap.add_argument("--backbone")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--results-root")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--explain", action="store_true", help="print how each run of this name differs"
    )
    args, _unknown = ap.parse_known_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.backbone:
        from ams.eval.backbone import apply_backbone
        from ams.eval.context import Config

        c = Config(raw=cfg)
        apply_backbone(c, args.backbone)
        cfg = c.raw
    if args.defences:
        cfg["defences"] = args.defences.split(",")
    if args.pool_size:
        cfg["pool_size"] = args.pool_size
    cfg.setdefault("pool_size", 1)
    if args.limit:
        cfg["limit"] = args.limit
    want = normalise(cfg)

    root = Path(
        args.results_root
        or os.environ.get("AMS_RESULTS_ROOT")
        or Path(__file__).resolve().parents[1] / "results"
    )
    if not root.is_dir():
        return 1

    if args.explain:
        prefix = str(cfg.get("name", ""))
        for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)):
            m = d / "manifest.json"
            if not m.is_file():
                print(f"{d.name}: no manifest (did not finish)")
                continue
            got = normalise(json.loads(m.read_text(encoding="utf-8")).get("config") or {})
            diff = [k for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)]
            print(f"{d.name}: " + ("SAME" if not diff else "differs in " + ", ".join(diff)))
            for k in diff:
                print(f"  {k} | now: {json.dumps(want.get(k))} | then: {json.dumps(got.get(k))}")

    for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        manifest = d / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            got = json.loads(manifest.read_text(encoding="utf-8")).get("config") or {}
        except (json.JSONDecodeError, OSError):
            continue
        if normalise(got) == want:
            if not args.quiet:
                print(f"    skip {cfg.get('name', args.config)} (done: {d.name})")
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
