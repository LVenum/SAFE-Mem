"""MPBench: memory-poisoning cases with a write-then-retrieve lifecycle.

Source: https://github.com/Digital-Trust-Lab/mp-bench (Apache-2.0), accompanying
"From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning
Attacks in LLM Agents" (AIWILD @ ICML 2026).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_DEC = json.JSONDecoder()

_ATTACK_TYPE = {
    "explicit_keyword_injection": "explicit_keyword",
    "explicit_keyword": "explicit_keyword",
    "repetition_injection": "repetition",
    "repetition_based_injection": "repetition",
    "repetition": "repetition",
    "delayed_conditional_injection_injection": "delayed_conditional",
    "delayed_conditional_injection": "delayed_conditional",
    "fact_injection": "fact_injection",
    "experience_injection": "experience_injection",
    "benign": "benign",
}

_SIGNAL = {"strong": "strong", "moderate": "moderate", "weak": "weak", "subtle": "weak"}

_DOMAIN = {
    "file_and_doc": "file_and_doc",
    "file_and_document_handling": "file_and_doc",
    "workspace_email": "email",
    "email_analysis_and_reply": "email",
    "web_browsing": "web",
    "web_search_result_analysis": "web",
    "script_execution": "script_execution",
    "calendar_and_scheduling": "calendar",
    "slack": "slack",
    "travel_assistance": "travel",
}

_GOAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"credential|password|secret|api.?key", "credential_harvesting"),
    (r"exfiltrat|leak|sensitive|external party|information.?leak", "data_exfiltration"),
    (r"bias|preference|vendor|tool.?pref|recommend", "recommendation_poisoning"),
    (r"bypass|verificat|authenticat", "bypass_verification"),
    (r"policy|override|normaliz|privilege|escalat", "policy_override"),
    (r"trust|hijack", "trust_hijacking"),
    (r"misinformation|disinformation", "misinformation"),
    (r"unauthori[sz]ed|unauthorized_action", "unauthorized_action"),
)


def _canon_bool(v):
    """`legitimate_memory_write` is a JSON bool on all but seven benign cases,
    where the release stores the string "true"/"false". Coerced on load so call
    sites can test it with `is False` without silently missing those seven."""
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return v


def _canon_goal(raw: str) -> str:
    low = (raw or "").lower()
    for pattern, label in _GOAL_PATTERNS:
        if re.search(pattern, low):
            return label
    return "other"


@dataclass(frozen=True, slots=True)
class MPBenchCase:
    cid: str
    is_adversarial: bool
    attack_type: str
    attack_signal: str
    domain: str
    goal: str
    user_query: str
    context: str
    expected_memory: str
    retrieval_query: str
    legitimate_memory_write: bool | None = None
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class CanonReport:
    merged: dict[str, dict[str, str]]
    counts: dict[str, Counter]
    n_unmapped_goals: int
    unmapped_goal_examples: list[str]

    def to_dict(self) -> dict:
        return {
            "merged_labels": self.merged,
            "counts": {k: dict(v) for k, v in self.counts.items()},
            "n_goals_falling_back_to_other": self.n_unmapped_goals,
            "unmapped_goal_examples": self.unmapped_goal_examples[:8],
        }


def _read_objects(path: Path) -> list[dict]:
    """Tolerant reader: a few lines in the release hold more than one object."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            while s:
                try:
                    obj, end = _DEC.raw_decode(s)
                except json.JSONDecodeError:
                    break
                out.append(obj)
                s = s[end:].lstrip()
    return out


def load_mpbench(
    root: str | Path = "data/mpbench",
    *,
    limit: int | None = None,
    adversarial: bool = True,
    benign: bool = True,
) -> tuple[list[MPBenchCase], CanonReport]:
    """Load the benchmark with labels canonicalised."""
    base = Path(root)
    merged: dict[str, dict[str, str]] = {
        "attack_type": {},
        "attack_signal": {},
        "domain": {},
        "legitimate_memory_write": {},
    }
    counts: dict[str, Counter] = {
        k: Counter() for k in ("attack_type", "attack_signal", "domain", "goal")
    }
    unmapped: list[str] = []
    cases: list[MPBenchCase] = []

    sources = []
    if adversarial:
        sources.append((base / "adversarial_data.jsonl.jsonl", True))
    if benign:
        sources.append((base / "benign_data.jsonl.jsonl", False))

    for path, is_adv in sources:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; fetch mp-bench into {base}")
        for r in _read_objects(path):
            at_raw, sig_raw = r.get("attack_type", ""), r.get("attack_signal", "")
            dom_raw, goal_raw = r.get("domain", ""), r.get("adversarial_goal", "")
            at = _ATTACK_TYPE.get(at_raw, at_raw)
            sig = _SIGNAL.get(sig_raw, sig_raw)
            dom = _DOMAIN.get(dom_raw, dom_raw)
            goal = _canon_goal(goal_raw)
            if goal == "other" and goal_raw:
                unmapped.append(goal_raw)

            lmw_raw = r.get("legitimate_memory_write")
            for field_name, raw_v, canon_v in (
                ("attack_type", at_raw, at),
                ("attack_signal", sig_raw, sig),
                ("domain", dom_raw, dom),
                ("legitimate_memory_write", lmw_raw, _canon_bool(lmw_raw)),
            ):
                if raw_v != canon_v:
                    merged[field_name][raw_v] = canon_v
            for k, v in (
                ("attack_type", at),
                ("attack_signal", sig),
                ("domain", dom),
                ("goal", goal),
            ):
                counts[k][v] += 1

            cases.append(
                MPBenchCase(
                    cid=r.get("id", ""),
                    is_adversarial=is_adv,
                    attack_type=at,
                    attack_signal=sig,
                    domain=dom,
                    goal=goal,
                    user_query=r.get("user_query", ""),
                    context=r.get("context", ""),
                    expected_memory=r.get("expected_memory", ""),
                    retrieval_query=r.get("retrieval_query", ""),
                    legitimate_memory_write=_canon_bool(r.get("legitimate_memory_write")),
                    raw=r,
                )
            )
            if limit and len(cases) >= limit:
                break

    report = CanonReport(
        merged=merged,
        counts=counts,
        n_unmapped_goals=len(unmapped),
        unmapped_goal_examples=sorted(set(unmapped)),
    )
    return cases, report
