"""LoCoMo (Maharana et al., ACL 2024): ten long conversations and their QA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CATEGORY = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


@dataclass(frozen=True, slots=True)
class Turn:
    dia_id: str
    speaker: str
    text: str
    blip_caption: str = ""

    def render(self) -> str:
        """The official turn format (`gpt_utils.get_input_context`)."""
        cap = f" and shared {self.blip_caption}." if self.blip_caption else ""
        return f'{self.speaker} said, "{self.text}"{cap}'


@dataclass(frozen=True, slots=True)
class Session:
    index: int
    date_time: str
    turns: tuple[Turn, ...]

    def render(self) -> str:
        body = "\n".join(t.render() for t in self.turns)
        return f"DATE: {self.date_time}\nCONVERSATION:\n{body}"


@dataclass(frozen=True, slots=True)
class QA:
    qid: str
    question: str
    answer: str
    category: int
    evidence: tuple[str, ...]
    kind: str = ""

    @property
    def category_name(self) -> str:
        return CATEGORY.get(self.category, str(self.category))


@dataclass(frozen=True, slots=True)
class Conversation:
    sample_id: str
    speakers: tuple[str, str]
    sessions: tuple[Session, ...]
    qa: tuple[QA, ...]

    def render(self) -> str:
        """Every session, oldest first."""
        return "\n\n".join(s.render() for s in self.sessions)


def load_locomo(path: str | Path = "data/locomo/locomo10.json") -> list[Conversation]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for sample in raw:
        conv = sample["conversation"]
        idx = sorted(
            int(k.rsplit("_", 1)[1])
            for k in conv
            if k.startswith("session_") and not k.endswith("date_time")
        )
        sessions = tuple(
            Session(
                index=i,
                date_time=conv.get(f"session_{i}_date_time", ""),
                turns=tuple(
                    Turn(
                        dia_id=t["dia_id"],
                        speaker=t["speaker"],
                        text=t["text"],
                        blip_caption=t.get("blip_caption", ""),
                    )
                    for t in conv[f"session_{i}"]
                ),
            )
            for i in idx
            if conv[f"session_{i}"]
        )
        qa = tuple(
            QA(
                qid=f"{sample['sample_id']}:{n}",
                question=q["question"],
                answer=str(q["answer"] if "answer" in q else q.get("adversarial_answer", "")),
                category=int(q["category"]),
                evidence=tuple(q.get("evidence", [])),
            )
            for n, q in enumerate(sample["qa"])
        )
        out.append(
            Conversation(
                sample_id=sample["sample_id"],
                speakers=(conv["speaker_a"], conv["speaker_b"]),
                sessions=sessions,
                qa=qa,
            )
        )
    return out
