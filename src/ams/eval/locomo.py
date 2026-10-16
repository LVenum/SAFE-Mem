"""LoCoMo QA: a memory system against the full-context baseline (spec §7.3)."""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ams.core.prompts import PromptLibrary
from ams.data.locomo import QA, Conversation
from ams.llm import ChatRequest, LLMClient, Message, map_concurrent, progress_bar
from ams.memory.base import MemorySystem

NOT_MENTIONED = "Not mentioned in the conversation"


_stemmer = None


def _stem(w: str) -> str:
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer

        _stemmer = PorterStemmer()
    return _stemmer.stem(w)


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    p = [_stem(w) for w in normalize_answer(prediction).split()]
    g = [_stem(w) for w in normalize_answer(ground_truth).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def f1_multi(prediction: str, ground_truth: str) -> float:
    preds = [x.strip() for x in prediction.split(",")]
    golds = [x.strip() for x in ground_truth.split(",")]
    return sum(max(f1_score(p, g) for p in preds) for g in golds) / len(golds)


def official_score(category: int, prediction: str, answer: str) -> float:
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return f1_score(prediction, answer)
    if category == 1:
        return f1_multi(prediction, answer)
    if category == 5:
        low = prediction.lower()
        return float("no information available" in low or "not mentioned" in low)
    raise ValueError(f"unknown LoCoMo category {category}")


def cat5_answer(prediction: str, key: dict[str, str]) -> str:
    """Official `get_cat_5_answer`: map 'a' / '(a)' back to the option text."""
    p = prediction.strip().lower()
    if len(p) == 1:
        return key["a"] if "a" in p else key["b"]
    if len(p) == 3:
        return key["a"] if "(a)" in p else key["b"]
    return p


@dataclass(frozen=True, slots=True)
class LoCoMoOutcome:
    system: str
    sample_id: str
    qid: str
    category: int
    question: str
    gold: str
    prediction: str
    f1: float
    judge_correct: bool | None
    n_context_tokens_approx: int
    required_type: str = ""


@dataclass
class LoCoMoQA:
    agent: LLMClient
    agent_model: str
    judge: LLMClient
    judge_model: str
    prompts: PromptLibrary

    def _complete(self, client: LLMClient, model: str, text: str, max_tokens: int) -> str:
        return client.complete(
            ChatRequest(
                messages=(Message("user", text),),
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        ).text.strip()

    def answer(self, qa: QA, context: str) -> str:
        if qa.category == 5:
            a, b = qa.answer, NOT_MENTIONED
            if random.Random(qa.qid).random() < 0.5:
                a, b = b, a
            q = f"{qa.question} Select the correct answer: (a) {a} (b) {b}. "
            text = self.prompts.get("locomo_qa_cat5").render(context=context, question=q)
            return cat5_answer(
                self._complete(self.agent, self.agent_model, text, 32), {"a": a, "b": b}
            )
        q = qa.question
        if qa.category == 2:
            q += " Use DATE of CONVERSATION to answer with an approximate date."
        text = self.prompts.get("locomo_qa").render(context=context, question=q)
        return self._complete(self.agent, self.agent_model, text, 32)

    def judged(self, qa: QA, prediction: str) -> bool | None:
        if qa.category == 5:
            return None
        text = self.prompts.get("locomo_judge").render(
            question=qa.question, gold=qa.answer, prediction=prediction
        )
        return "CORRECT" in self._complete(self.judge, self.judge_model, text, 4).upper()

    def outcome(self, system: str, conv: Conversation, qa: QA, context: str, **kw) -> LoCoMoOutcome:
        pred = self.answer(qa, context)
        return LoCoMoOutcome(
            system=system,
            sample_id=conv.sample_id,
            qid=qa.qid,
            category=qa.category,
            question=qa.question,
            gold=qa.answer,
            prediction=pred,
            f1=official_score(qa.category, pred, qa.answer),
            judge_correct=self.judged(qa, pred),
            n_context_tokens_approx=len(context) // 4,
            **kw,
        )


def run_full_context(
    convs: Sequence[Conversation], qa: LoCoMoQA, *, workers: int = 8
) -> list[LoCoMoOutcome]:
    """No memory: the whole conversation in the prompt."""
    start = qa.prompts.get("locomo_conv_start")
    jobs = [
        (c, q, start.render(a=c.speakers[0], b=c.speakers[1]) + "\n\n" + c.render())
        for c in convs
        for q in c.qa
    ]
    return list(
        map_concurrent(
            lambda j: qa.outcome("full_context", *j), jobs, workers=workers, progress="full_context"
        )
    )


def run_memory(
    convs: Sequence[Conversation],
    memory: MemorySystem,
    qa: LoCoMoQA,
    *,
    k: int,
    workers: int = 8,
    principal: str | None = None,
    defences: Sequence | None = None,
) -> list[LoCoMoOutcome]:
    """One store per conversation, one write per session in order, then every
    question against that store, once per defence.
    """
    from ams.baselines.base import NoDefence

    defences = list(defences or [NoDefence()])
    dialogue = principal is not None and hasattr(memory, "write_dialogue")

    def who(c: Conversation) -> str:
        if not dialogue or principal == "all":
            return "user"
        return c.speakers[0 if principal == "speaker_a" else 1]

    def fill(c: Conversation) -> None:
        memory.reset(c.sample_id)
        for s in c.sessions:
            if dialogue:
                memory.write_dialogue(
                    session_id=c.sample_id,
                    turns=[(t.speaker, t.render().split(" said, ", 1)[-1]) for t in s.turns],
                    date=s.date_time,
                    principal="*" if principal == "all" else who(c),
                )
            else:
                memory.write(session_id=c.sample_id, user_query=s.render(), context="")
            bar.update()

    bar = progress_bar(sum(len(c.sessions) for c in convs), f"write/{memory.name}")
    try:
        map_concurrent(fill, convs, workers=workers)
    finally:
        bar.close()
    start = qa.prompts.get("locomo_memory_start")
    rows: list[LoCoMoOutcome] = []
    for defence in defences:
        label = memory.name if defence.name == "no_defence" else f"{defence.name}/{memory.name}"
        if dialogue:
            label += f"[principal={principal}]"

        def one(job: tuple[Conversation, QA], defence=defence, label=label) -> LoCoMoOutcome:
            c, q = job
            got = memory.retrieve(session_id=c.sample_id, query=q.question, k=k, principal=who(c))
            kept = defence.filter(q.question, list(got)).kept
            body = "\n".join(f"- {e.text}" for e in kept) or "(nothing)"
            context = start.render(a=c.speakers[0], b=c.speakers[1]) + "\n\n" + body
            return qa.outcome(label, c, q, context, required_type=getattr(got, "required", ""))

        jobs = [(c, q) for c in convs for q in c.qa]
        rows += map_concurrent(one, jobs, workers=workers, progress=f"qa/{label}")
    return rows


def summarise(rows: Sequence[LoCoMoOutcome]) -> dict:
    """Per system: F1 and judge accuracy per category, and over 1-4 (category 5
    is reported on its own; most published LoCoMo numbers exclude it)."""
    out: dict[str, dict] = {}
    for system in sorted({r.system for r in rows}):
        mine = [r for r in rows if r.system == system]

        def agg(sel: list[LoCoMoOutcome]) -> dict:
            judged = [r.judge_correct for r in sel if r.judge_correct is not None]
            return {
                "n": len(sel),
                "f1": round(sum(r.f1 for r in sel) / len(sel), 4) if sel else 0.0,
                "judge": round(sum(judged) / len(judged), 4) if judged else None,
            }

        out[system] = {
            "overall_1to4": agg([r for r in mine if r.category != 5]),
            "by_category": {
                str(c): agg([r for r in mine if r.category == c])
                for c in sorted({r.category for r in mine})
            },
        }
    return out
