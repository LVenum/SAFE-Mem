"""LongMemEval_S (Wu et al., ICLR 2025): 500 questions, each over its own history."""

from __future__ import annotations

import json
from pathlib import Path

from ams.data.locomo import QA, Conversation, Session, Turn

DEFAULT_PATH = Path("data/longmemeval/longmemeval_s.json")


def load_longmemeval(path: str | Path = DEFAULT_PATH) -> list[Conversation]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download longmemeval_s.json from the authors' release "
            "(Hugging Face dataset xiaowu0162/longmemeval) into data/longmemeval/."
        )
    out = []
    for item in json.loads(p.read_text(encoding="utf-8")):
        qid = str(item["question_id"])
        sessions = tuple(
            Session(
                index=i,
                date_time=str(date),
                turns=tuple(
                    Turn(dia_id=f"{sid}:{n}", speaker=t["role"], text=t["content"])
                    for n, t in enumerate(turns)
                ),
            )
            for i, (sid, date, turns) in enumerate(
                zip(
                    item["haystack_session_ids"],
                    item["haystack_dates"],
                    item["haystack_sessions"],
                    strict=True,
                )
            )
            if turns
        )
        question = f"(Today is {item['question_date']}.) {item['question']}"
        out.append(
            Conversation(
                sample_id=qid,
                speakers=("user", "assistant"),
                sessions=sessions,
                qa=(
                    QA(
                        qid=qid,
                        question=question,
                        answer=str(item["answer"]),
                        category=4,
                        evidence=tuple(item.get("answer_session_ids", [])),
                        kind=str(item.get("question_type", "")),
                    ),
                ),
            )
        )
    return out
