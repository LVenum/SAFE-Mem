"""The paper's tables, read off finished run directories."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from ams.core.stats import wilson_interval

csv.field_size_limit(1 << 26)


def _b(v: str) -> bool:
    return v in ("1", "True", "true")


def _pct(k: int, n: int) -> str:
    return "–" if not n else f"{100 * k / n:.1f}"


def _ci(k: int, n: int) -> str:
    if not n:
        return "–"
    lo, hi = wilson_interval(k, n)
    return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def load_outcomes(run_dirs: Iterable[Path]) -> list[dict]:
    rows = []
    for d in run_dirs:
        f = Path(d) / "mpbench_outcomes.csv"
        if f.exists():
            for r in csv.DictReader(f.open(encoding="utf-8")):
                r["run"] = Path(d).name
                rows.append(r)
    return rows


def system(r: dict) -> str:
    return r["memory"] if r["defence"] == "no_defence" else f"{r['defence']} / {r['memory']}"


def _groups(rows: Sequence[dict]) -> tuple[dict[str, list[dict]], list[str]]:
    """Rows per system, keeping one run per system (the first given); the
    no-defence baseline's other runs become footnotes."""
    by: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[system(r)][r["run"]].append(r)
    out, notes = {}, []
    for name, runs in by.items():
        first, *rest = list(runs)
        out[name] = runs[first]
        for run in rest:
            adv = [r for r in runs[run] if _b(r["is_adversarial"])]
            k = sum(_b(r["attack_succeeded"]) for r in adv)
            notes.append(
                f"{name}: also run in `{run}`, ASR {_pct(k, len(adv))}% (shown: `{first}`)"
            )
    return out, notes


def main_table(rows: Sequence[dict]) -> str:
    groups, notes = _groups(rows)
    head = (
        "| system | n adv | ASR e2e % [95% CI] | poison in prompt % | benign recall % (n) "
        "| restraint % (n) | flagged benign % | masked at write, benign % "
        "| failed elevation % |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [head]
    order = sorted(groups, key=lambda s: (not s.startswith("simple_extract"), s))
    for name in order:
        rs = groups[name]
        adv = [r for r in rs if _b(r["is_adversarial"])]
        ben = [r for r in rs if not _b(r["is_adversarial"]) and _b(r["benign_scorable"])]
        war = [r for r in ben if r["benign_warranted"] in ("1", "True")]
        unw = [r for r in ben if r["benign_warranted"] in ("0", "False")]
        k = sum(_b(r["attack_succeeded"]) for r in adv)
        tracked = [r for r in war if int(r.get("n_written_above_description") or -1) >= 0]
        failed = sum(int(r["n_written_above_description"]) == 0 for r in tracked)
        lines.append(
            f"| {name} | {len(adv)} | {_ci(k, len(adv))} "
            f"| {_pct(sum(_b(r['poison_retrieved']) for r in adv), len(adv))} "
            f"| {_pct(sum(_b(r['answer_correct']) for r in war), len(war))} ({len(war)}) "
            f"| {_pct(sum(_b(r['answer_correct']) for r in unw), len(unw))} ({len(unw)}) "
            f"| {_pct(sum(int(r['n_flagged']) > 0 for r in ben), len(ben))} "
            f"| {_pct(sum(int(r.get('n_refused_at_write') or 0) > 0 for r in ben), len(ben))} "
            f"| {_pct(failed, len(tracked))} |"
        )
    if notes:
        lines.append("")
        lines += [f"- {n}" for n in notes]
    return "\n".join(lines)


def by_goal(rows: Sequence[dict]) -> str:
    groups, _ = _groups(rows)
    goals = sorted({r["goal"] for r in rows if _b(r["is_adversarial"])})
    head = "| system | " + " | ".join(goals) + " |\n|---|" + "---:|" * len(goals)
    lines = [
        "Each cell: ASR % / ASR % given the case's own poison reached the prompt (n of those).",
        "",
        head,
    ]
    for name in sorted(groups):
        adv = [r for r in groups[name] if _b(r["is_adversarial"])]
        cells = []
        for g in goals:
            rs = [r for r in adv if r["goal"] == g]
            hit = [r for r in rs if _b(r["poison_retrieved"])]
            cells.append(
                f"{_pct(sum(_b(r['attack_succeeded']) for r in rs), len(rs))} / "
                f"{_pct(sum(_b(r['attack_succeeded']) for r in hit), len(hit))} ({len(hit)})"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def repetition_sweep(rows: Sequence[dict], repetition_count: dict[str, int]) -> str:
    """Prediction 1: A-MemGuard's ASR rises with the number of paraphrases; ours is flat."""
    groups, _ = _groups(rows)
    counts = sorted(set(repetition_count.values()))
    lines = [
        "| system | " + " | ".join(f"x{c} ASR % (n)" for c in counts) + " |",
        "|---|" + "---:|" * len(counts),
    ]
    for name in sorted(groups):
        rs = [r for r in groups[name] if r["cid"] in repetition_count]
        if not rs:
            continue
        cells = []
        for c in counts:
            sel = [r for r in rs if repetition_count[r["cid"]] == c]
            cells.append(
                f"{_pct(sum(_b(r['attack_succeeded']) for r in sel), len(sel))} ({len(sel)})"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def eligible_empty(rows: Sequence[dict]) -> str:
    """§6.1: confine and withhold can only differ where nothing was eligible."""
    groups, _ = _groups(rows)
    lines = [
        "| system | eligible empty: ASR % (n) | eligible non-empty: ASR % (n) |",
        "|---|---:|---:|",
    ]
    for name in sorted(groups):
        adv = [r for r in groups[name] if _b(r["is_adversarial"]) and r.get("required_type")]
        if not adv:
            continue
        e = [r for r in adv if _b(r["eligible_empty"])]
        ne = [r for r in adv if not _b(r["eligible_empty"])]
        lines.append(
            f"| {name} | {_pct(sum(_b(r['attack_succeeded']) for r in e), len(e))} ({len(e)}) "
            f"| {_pct(sum(_b(r['attack_succeeded']) for r in ne), len(ne))} ({len(ne)}) |"
        )
    return "\n".join(lines)


def locomo_table(run_dirs: Iterable[Path]) -> str:
    seen: dict[str, tuple[str, dict]] = {}
    for d in run_dirs:
        f = Path(d) / "locomo_summary.json"
        if f.exists():
            for name, s in json.loads(f.read_text())["systems"].items():
                seen.setdefault(name, (Path(d).name, s))
    if not seen:
        return ""
    cats = ["1", "2", "3", "4", "5"]
    lines = [
        "| system | F1 (1-4) | judge (1-4) | " + " | ".join(f"c{c} F1/judge" for c in cats) + " |",
        "|---|---:|---:|" + "---:|" * len(cats),
    ]
    for name, (_, s) in sorted(seen.items(), key=lambda kv: kv[0] != "full_context"):
        o = s["overall_1to4"]
        cells = []
        for c in cats:
            v = s["by_category"].get(c)
            cells.append(
                "–" if not v else f"{v['f1']:.3f}/{v['judge'] if v['judge'] is not None else '–'}"
            )
        lines.append(f"| {name} | {o['f1']:.3f} | {o['judge']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Categories as in the released data: 1 multi-hop, 2 temporal, 3 open-domain, "
        "4 single-hop, 5 adversarial (string-scored only)."
    )
    return "\n".join(lines)


def graph_table(run_dirs: Iterable[Path]) -> str:
    rows = []
    for d in run_dirs:
        f = Path(d) / "graph_stats.json"
        if f.exists():
            s = json.loads(f.read_text())
            rows.append(
                f"| `{Path(d).name}` | {s['n_claims']} | {s['n_entities']} "
                f"| {s['share_with_object']} | {s['share_with_a_neighbour']} "
                f"| {s['claims_per_entity']} |"
            )
    if not rows:
        return ""
    return "\n".join(
        [
            "| run | claims | entities | share with object | share with a neighbour "
            "| claims per entity |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
        ]
    )
