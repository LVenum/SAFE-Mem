#!/usr/bin/env python3
"""Make the vendored MINJA EHRAgent runnable, without changing their attack."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "MINJA"
EHR = ROOT / "EHR" / "ehragent"


AUDIT_GLUE = r'''"""Defences for MINJA's EHRAgent: their monitors, our glue.

MINJA is a query-only attack. The attacker never writes to the store; he asks
questions, and the agent — working normally — records its own solutions. Every
poisoned record is therefore a genuine self-write, which is what makes this the
interesting setting for an origin-based defence.

$MINJA_AUDIT selects one of:

  A-MemGuard     their consistency check over per-record reasoning chains
                 (AMemGuard/EhrAgent/ehragent/consistency.py, unmodified), run
                 exactly as their own EHRAgent integration runs it: chains for
                 the retrieved records, an auditor LLM marking each consistent
                 or not, inconsistent records dropped from the prompt.
  TMA-NM         their origin/authority monitor over the act class. Records are
                 agent self-writes, so they carry Origin.AGENT and ActClass
                 INFORM, and nothing is withheld. This is not a bug in their
                 system: it is the boundary of what an origin label can see when
                 the attacker never touches the channel.
  TMA-NM-values  the same monitor used at value level, which is where it has
                 something to say here. The user's question endorses the values
                 it names; a retrieved record carrying an identifier the user
                 never named is unendorsed and is withheld. MINJA's payload
                 names the attacker's target patient, which the victim's query
                 never mentions.
  DRIFT, RandEC, LLM-sanitizer, MemLineage
                 the MPBench baselines, the same objects and prompts, served by
                 a worker in the main environment (ams.eval.minja_defence, which
                 says what each sees here). A record they rewrite replaces the
                 example in the prompt.

Verdicts are appended to $MINJA_AUDIT_LOG, one JSON line per retrieval.
"""

import functools
import importlib.util
import atexit
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_THIRD_PARTY = _HERE.parents[3]
_TMA = _THIRD_PARTY / "mem-inv-bench" / "v2" / "code" / "tma.py"
_AMEMGUARD = _THIRD_PARTY / "AMemGuard" / "EhrAgent" / "ehragent"

_IDENT = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{3,}\b")
_TOKEN = re.compile(r"[A-Za-z0-9_./:@-]{3,}")


def _record_text(rec):
    return "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(
        rec.get("question", ""), rec.get("knowledge", ""), rec.get("code", "")
    )


# ---------------------------------------------------------------- A-MemGuard

_CHECKER = None


def _checker():
    global _CHECKER
    if _CHECKER is None:
        if not _AMEMGUARD.exists():
            raise FileNotFoundError(
                f"AMemGuard not vendored at {_AMEMGUARD}; run scripts/fetch_third_party.sh"
            )
        # appended, never prepended: that directory holds its own config.py and
        # medagent.py, which would shadow MINJA's if they came first
        sys.path.append(str(_AMEMGUARD))
        from consistency import ConsistencyChecker, OpenAIModel, VLLMModel

        model = os.environ.get("AMEMGUARD_MODEL", "gpt-4o-mini")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            provider = VLLMModel(model_name=model, base_url=base_url,
                                 api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        else:
            provider = OpenAIModel(model_name=model, api_key=os.environ.get("OPENAI_API_KEY"))
        if os.environ.get("MINJA_EXTRA_BODY"):
            # OpenRouter backbones: reasoning off, as for the agent (minja_run.sh)
            provider.client.chat.completions.create = functools.partial(
                provider.client.chat.completions.create,
                extra_body=json.loads(os.environ["MINJA_EXTRA_BODY"]))
        _CHECKER = ConsistencyChecker(model_provider=provider)
    return _CHECKER


def _amemguard(query, memory, selected):
    texts = [_record_text(memory[i]) for i in selected]
    try:
        result = _checker().check(query=query, memories=texts, selected_indexes=list(selected),
                                  method=os.environ.get("AMEMGUARD_METHOD", "llm"))
    except Exception as e:
        # a judgment their parser cannot read beyond the JSON errors it catches
        # itself: drop every record, as their own fallback does (consistency.py,
        # "consider all as inconsistent") and as on MPBench; say so in the run log
        print(f"A-MemGuard parse failure, records dropped: {e!r}", file=sys.stderr)
        return [], {i: "A-MemGuard parse failure" for i in selected}
    kept = [int(item["index"]) for item in result.get("consistent_memories", [])]
    why = {int(item["index"]): item.get("reasoning_chain", "")[:400]
           for item in result.get("inconsistent_memories", [])}
    return kept, why


# -------------------------------------------------------------------- TMA-NM

_TMA_MOD = None


def _tma():
    global _TMA_MOD
    if _TMA_MOD is None:
        if not _TMA.exists():
            raise FileNotFoundError(
                f"mem-inv-bench not vendored at {_TMA}; run scripts/fetch_third_party.sh"
            )
        spec = importlib.util.spec_from_file_location("tma_upstream", _TMA)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tma_upstream"] = mod
        spec.loader.exec_module(mod)
        _TMA_MOD = mod
    return _TMA_MOD


def _monitor(t, query):
    """Their setup: the user's task is the one trusted principal, and every value
    it names is endorsed. Memory records arrive on the agent's own channel,
    because under a query-only attack that is literally where they came from."""
    chans = {
        "self": t.Channel("self", t.Origin.AGENT, "agent", b"k-self"),
        "user": t.Channel("user", t.Origin.USER, "principal", b"k-user"),
    }
    reg = t.Registry()
    for c in chans.values():
        reg.register(c)
    mon = t.Monitor(reg, k=2)
    mon.ingest(chans["user"].send(query))
    for tok in set(_TOKEN.findall(query)):
        mon.endorse(tok, "principal")
    mon.tick_epoch()
    return mon, chans


def _tma_actclass(query, memory, selected):
    t = _tma()
    mon, chans = _monitor(t, query)
    kept, why = [], {}
    for i in selected:
        item = mon.ingest(chans["self"].send(json.dumps(memory[i], sort_keys=True)))
        if int(item.act) >= int(t.ActClass.INFORM):
            kept.append(i)
        else:
            why[i] = f"act={item.act}"
    return kept, why


def _tma_values(query, memory, selected):
    """Value-level use of the same monitor: withhold a record that carries an
    identifier no trusted principal endorsed."""
    t = _tma()
    mon, _ = _monitor(t, query)
    kept, why = [], {}
    for i in selected:
        # mon.attribute is their allow-list: permitted iff a trusted channel
        # endorsed the value, under their canonicalisation
        unendorsed = sorted({
            tok for tok in _IDENT.findall(_record_text(memory[i]))
            if not mon.attribute(tok)[0]
        })
        if unendorsed:
            why[i] = "unendorsed values: " + ", ".join(unendorsed[:8])
        else:
            kept.append(i)
    return kept, why


# ------------------------------------------- typed standing (ours), isolation
#
# The gate logic is ours and lives in the `ams` package (src/ams/eval/callsite.py),
# pure Python with no dependencies beyond the standard library, so it imports in
# this environment unchanged. Per question the harness tells us who is asking
# (`begin_question`); every record carries who wrote it and its standing, the
# meet over everything that was in front of the model when it was written
# (`provenance`). What kind of claim each question needs was classified before
# the run, in the main environment, and is read from $MINJA_USE_TYPES.

# minja_run.sh passes the checkout it runs from: third_party/ may be a symlink to
# another volume, so the path of this file does not lead back to the repo.
_REPO = Path(os.environ.get("AMS_REPO") or _HERE.parents[4])
if str(_REPO / "src") not in sys.path:
    sys.path.append(str(_REPO / "src"))

_STATE = {"principal": "user", "context": [], "usable": {}, "rewrite": {}}
_USE_TYPES = None


def begin_question(principal):
    _STATE.update(principal=principal or "user", context=[], usable={}, rewrite={})


def _required(query):
    global _USE_TYPES
    from ams.core.lattice import Rung

    if _USE_TYPES is None:
        path = os.environ.get("MINJA_USE_TYPES")
        if not path or not Path(path).exists():
            raise FileNotFoundError("typed arms need $MINJA_USE_TYPES; see scripts/minja_use_types.py")
        _USE_TYPES = json.loads(Path(path).read_text())
    if query not in _USE_TYPES:
        raise KeyError(f"no use type precomputed for {query[:80]!r}")
    return Rung[_USE_TYPES[query].upper()]


def _typed(query, memory, selected, ranked, *, mode, scoped):
    from ams.eval import callsite

    picked = callsite.select(
        ranked or selected,
        [callsite.from_dict(m.get("standing")) for m in memory],
        required=_required(query), asker=_STATE["principal"],
        k=len(selected), mode=mode, scoped=scoped,
    )
    _STATE["usable"] = {i: u for i, u in picked}
    kept = [i for i, _ in picked]
    why = {i: "not eligible" for i in selected if i not in set(kept)}
    return kept, why


def _isolation(query, memory, selected, ranked):
    """The trivial solution: each principal sees only their own records (and the
    shipped demos). Top-k among those, not a filter on someone else's top-k."""
    mine = [i for i in (ranked or selected)
            if memory[i].get("principal") in (None, _STATE["principal"])]
    kept = mine[:len(selected)]
    return kept, {i: "another principal's record" for i in selected if i not in set(kept)}


# ------------------------------------------------ the MPBench baselines

_WORKER = None
_BASELINES = ("DRIFT", "RandEC", "LLM-sanitizer", "MemLineage")


def _close_worker():
    if _WORKER is not None and _WORKER.poll() is None:
        _WORKER.stdin.close()   # EOF: the worker writes its manifest and exits
        _WORKER.wait()


def _worker_died(err=""):
    # main.py catches every per-question exception and carries on, so raising here
    # would let the arm finish undefended and get its .done marker. End the
    # process instead: minja_run.sh then reports the arm as not finished.
    print(f"!! {os.environ['MINJA_AUDIT']} worker died {err}; arm aborted", file=sys.stderr)
    sys.stderr.flush()
    os._exit(1)


def _baseline(query, memory, selected):
    """One request to the worker; it fails loudly, so an arm whose defence
    could not answer is unfinished rather than silently undefended."""
    global _WORKER
    if _WORKER is None:
        log = Path(os.environ["MINJA_AUDIT_LOG"])   # <arm>/audit_{write,read}.jsonl
        try:
            _WORKER = subprocess.Popen(
                [os.environ.get("AMS_PY", str(_REPO / ".venv" / "bin" / "python")),
                 str(_REPO / "scripts" / "minja_defence.py"), os.environ["MINJA_AUDIT"],
                 "--backbone", os.environ.get("BACKBONE", "4omini"),
                 "--results-root", str(log.parent),
                 "--run-id", log.stem.replace("audit", "defence")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )
        except OSError as e:
            _worker_died(repr(e))
        atexit.register(_close_worker)
    records = [{"question": memory[i].get("question", ""), "text": _record_text(memory[i])}
               for i in selected]
    try:
        _WORKER.stdin.write(json.dumps({"query": query, "records": records}) + "\n")
        _WORKER.stdin.flush()
        line = _WORKER.stdout.readline()
    except OSError as e:
        _worker_died(repr(e))
    if not line:
        _worker_died()
    reply = json.loads(line)
    _STATE["rewrite"] = {selected[int(p)]: text for p, text in reply["rewrite"].items()}
    return ([selected[p] for p in reply["kept"]],
            {selected[int(p)]: why for p, why in reply["why"].items()})


_METHODS = {
    **{name: _baseline for name in _BASELINES},
    "A-MemGuard": _amemguard,
    "TMA-NM": _tma_actclass,
    "TMA-NM-values": _tma_values,
}
_RANKED = {
    "typed-confine": lambda q, m, s, r: _typed(q, m, s, r, mode="confine", scoped=True),
    "typed-withhold": lambda q, m, s, r: _typed(q, m, s, r, mode="withhold", scoped=True),
    # prediction 5: with scope off MINJA should return to the undefended level
    "typed-noscope": lambda q, m, s, r: _typed(q, m, s, r, mode="confine", scoped=False),
    "isolation": _isolation,
}


def _typed_arm():
    return os.environ.get("MINJA_AUDIT", "").startswith("typed-")


def render_example(i, template):
    """Typed arms wrap each record in a sentinel naming the use it may serve; a
    baseline that rewrote the record shows its rewrite instead."""
    if i in _STATE["rewrite"]:
        return _STATE["rewrite"][i]
    if not _typed_arm():
        return template
    from ams.core import sentinel
    from ams.core.lattice import Rung

    u = _STATE["usable"].get(i, Rung.DESCRIPTION)
    return sentinel.render(f"m{i}", template, usable=u.name.lower())


def render_knowledge(knowledge):
    """The knowledge line is the agent's own derivation from the retrieved
    records, so it may serve no more than the weakest of them (§5.3)."""
    if not _typed_arm():
        return knowledge
    from ams.core import sentinel
    from ams.core.lattice import Rung

    u = min(_STATE["usable"].values(), default=Rung.DESCRIPTION)
    return sentinel.render("knowledge", knowledge, usable=Rung(u).name.lower())


def provenance():
    from ams.eval import callsite

    return {
        "principal": _STATE["principal"],
        "standing": callsite.to_dict(
            callsite.record_standing(_STATE["principal"], _STATE["context"])
        ),
    }


def callsite_gate(question, code, prompt):
    """Refusal message if the call may not run, else None. Every execution is
    logged in every arm, so attack success can be scored on what ran."""
    refusal, detail = None, {}
    if _typed_arm():
        from ams.eval import callsite

        ok, detail = callsite.check_call(
            code, question=question, prompt=prompt, required=_required(question)
        )
        if not ok:
            refusal = ("call refused by the memory gate: "
                       + ", ".join(detail["refused_values"])
                       + " is supported only by memory that cannot authorise it")
    path = os.environ.get("MINJA_GATE_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"question": question, "principal": _STATE["principal"],
                                 "code": code, "refused": bool(refusal), **detail}) + "\n")
    return refusal


def audit_filter(query, memory, selected_indexes, ranked=None):
    """Return the subset of `selected_indexes` the defence lets into the prompt.

    Ranked defences (typed, isolation) choose from the whole similarity ranking,
    so they may serve records outside the plain top-k."""
    name = os.environ.get("MINJA_AUDIT", "")
    selected = [int(i) for i in selected_indexes]
    _STATE["rewrite"] = {}
    if not name or name in ("none", "None"):
        _STATE["context"] += [_standing_of(memory[i]) for i in selected]
        return selected
    if name in _RANKED:
        kept, why = _RANKED[name](query, memory, selected, ranked)
    elif name in _METHODS:
        kept, why = _METHODS[name](query, memory, selected)
        kept = [i for i in selected if i in set(kept)]  # keep retrieval order
    else:
        raise ValueError(f"unknown MINJA_AUDIT={name!r}; "
                         f"choose from {sorted(_METHODS) + sorted(_RANKED)}")
    _STATE["context"] += [_standing_of(memory[i]) for i in kept]
    path = os.environ.get("MINJA_AUDIT_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "audit": name, "query": query, "selected": selected, "kept": kept,
                "dropped": {str(k): v for k, v in why.items()},
                "rewritten": sorted(_STATE["rewrite"]),
            }) + "\n")
    return kept


def _standing_of(record):
    from ams.eval import callsite

    return callsite.from_dict(record.get("standing"))
'''


_MED_OLD = '                    temperature=0,\n                    **({"max_completion_tokens": 800,\n                        "extra_body": (__import__("json").loads(os.environ["MINJA_EXTRA_BODY"]) if os.environ.get("MINJA_EXTRA_BODY") else {"reasoning_effort": os.environ["MINJA_REASONING_EFFORT"]})}\n                       if os.environ.get("MINJA_REASONING_EFFORT") else\n                       {"max_tokens": 800, "top_p": 0.95, "frequency_penalty": 0,\n                        "presence_penalty": 0, "stop": None})\n                    )'
_MED_NEW = '                    temperature=0,\n                    **({"max_completion_tokens": 800,\n                        "extra_body": {"reasoning_effort": os.environ["MINJA_REASONING_EFFORT"]}}\n                       if os.environ.get("MINJA_REASONING_EFFORT") and not os.environ.get("MINJA_EXTRA_BODY") else\n                       {"max_tokens": 800, "top_p": 0.95, "frequency_penalty": 0,\n                        "presence_penalty": 0, "stop": None,\n                        **({"extra_body": json.loads(os.environ["MINJA_EXTRA_BODY"])}\n                           if os.environ.get("MINJA_EXTRA_BODY") else {})})\n                    )'


def sub(path: Path, old: str, new: str, *, count: int = 1) -> None:
    """Replace `old` with `new`; skipped once `new` is present, so reruns are safe."""
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        sys.exit(f"patch anchor not found in {path}: {old[:70]!r}")
    path.write_text(text.replace(old, new, -1 if count == 0 else count), encoding="utf-8")


def main() -> int:
    if not EHR.exists():
        sys.exit(f"MINJA not vendored at {ROOT}; run scripts/fetch_third_party.sh")

    sub(
        EHR / "config.py",
        """def openai_config(model):
    if model == 'gpt-4':
        config = {
            "model": "gpt-4",
            "api_key": "your-key"
        }
    elif model == 'gpt-4o':
        config = {
            "model": "gpt-4o",
            "api_key": "your-key"
        }
    else:
        config = {
            "model": "o1-preview",
            "api_key": "your-key"
        }     
    return config""",
        '''import os


def openai_config(model):
    """Any model name, key and base URL from the environment. The released file
    hard-codes three model names and the literal string "your-key"."""
    config = {"model": model, "api_key": os.environ.get("OPENAI_API_KEY", "")}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        config["base_url"] = base_url
    return config''',
    )

    tab = EHR / "tools" / "tabtools.py"
    sub(
        tab,
        "import Levenshtein\n",
        """import Levenshtein
import os

# The released file addresses the tables as "../ehrsql-ehragent/...", which only
# resolves from inside the author's working directory. Resolve them explicitly.
_EHRSQL = os.environ.get("EHRSQL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "ehrsql-ehragent")


def _db():
    return os.path.join(_EHRSQL, os.environ.get("EHRSQL_DB", "mimic_iii/mimic_iii.db"))
""",
    )
    sub(
        tab,
        "    data = pd.read_csv(ehr_dict[target_ehr])\n",
        """    # rebase onto $EHRSQL_ROOT; also repairs OUTPUTEVENTS.csv, which the release
    # points at "../mimic_iii/" while every other table sits under
    # "../ehrsql-ehragent/mimic_iii/"
    ehr_dict = {k: os.path.join(_EHRSQL, v.split("ehrsql-ehragent/")[-1].lstrip("./"))
                for k, v in ehr_dict.items()}
    data = pd.read_csv(ehr_dict[target_ehr])
""",
    )
    sub(
        tab,
        'sqlite3.connect("../ehrsql-ehragent/mimic_iii/mimic_iii.db")',
        "sqlite3.connect(_db())",
        count=0,
    )

    for name in ("data_create.py", "data_create_poison.py"):
        sub(
            EHR / "ehrsql-ehragent" / name,
            "import random\n",
            """import random
import os

# The release samples the attack and evaluation sets with an unseeded `random`,
# so a rerun builds a different benchmark. Seed it.
random.seed(int(os.environ.get("MINJA_SEED", "0")))
""",
        )

    med = EHR / "medagent.py"
    sub(
        med,
        "import json\nfrom openai import OpenAI\n",
        "import json\nimport os\nfrom openai import OpenAI\n",
    )
    sub(
        med,
        """        client = OpenAI(
            api_key=config["api_key"],
        )""",
        """        client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
        )""",
    )
    sub(
        med,
        """        # Prepare a list to store the top N most similar examples
        top_examples = []
        for idx, i in enumerate(sorted_indices[:self.num_shots]):""",
        """        # The defence seam. Unset $MINJA_AUDIT leaves retrieval exactly as
        # released; a defence may drop records, and a dropped record is not
        # replaced — the same semantics A-MemGuard's own EHRAgent uses.
        selected_indexes = [int(i) for i in sorted_indices[:self.num_shots]]
        if os.environ.get("MINJA_AUDIT"):
            from minja_audit import audit_filter
            selected_indexes = audit_filter(query, self.memory, selected_indexes,
                                            ranked=[int(i) for i in sorted_indices])

        # Prepare a list to store the top N most similar examples
        top_examples = []
        for idx, i in enumerate(selected_indexes):""",
    )

    sub(
        EHR / "main.py",
        """        result = judge(prediction, answer)
        """,
        """        result = judge(prediction, answer)

        # The release evaluates `result` and drops it on the floor, so the attack
        # is reported without the task accuracy it costs. One line per question.
        if os.environ.get("MINJA_RESULTS"):
            with open(os.environ["MINJA_RESULTS"], 'a') as rf:
                rf.write(json.dumps({
                    "id": contents[i]['id'], "question": question, "answer": answer,
                    "prediction": prediction, "correct": bool(result),
                    "knowledge": user_proxy.knowledge, "code": user_proxy.code,
                }) + "\\n")
        """,
    )

    sub(
        med,
        """            top_examples.append(template)""",
        """            if os.environ.get("MINJA_AUDIT"):
                from minja_audit import render_example
                template = render_example(i, template)
            top_examples.append(template)""",
    )
    sub(
        med,
        """        init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=knowledge, question=context["message"])
        return init_message""",
        """        shown = knowledge
        if os.environ.get("MINJA_AUDIT"):
            from minja_audit import render_knowledge
            shown = render_knowledge(knowledge)
        init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=shown, question=context["message"])
        self.init_prompt = init_message
        return init_message""",
    )
    sub(
        med,
        """                self.code = arguments["cell"]
                try:
                    content = func(**arguments)""",
        """                self.code = arguments["cell"]
                refusal = None
                if os.environ.get("MINJA_AUDIT"):
                    from minja_audit import callsite_gate
                    refusal = callsite_gate(self.question, self.code, getattr(self, "init_prompt", ""))
                try:
                    if refusal:
                        raise PermissionError(refusal)
                    content = func(**arguments)""",
    )
    sub(
        EHR / "main.py",
        """        try:
            user_proxy.update_memory(args.num_shots, long_term_memory)""",
        """        try:
            if os.environ.get("MINJA_AUDIT"):
                from minja_audit import begin_question
                begin_question(contents[i].get("principal", "user"))
            user_proxy.update_memory(args.num_shots, long_term_memory)""",
    )
    sub(
        EHR / "main.py",
        """        new_item = {"question": question, "knowledge": user_proxy.knowledge, "code": user_proxy.code}
        long_term_memory.append(new_item)""",
        """        new_item = {"question": question, "knowledge": user_proxy.knowledge, "code": user_proxy.code}
        if os.environ.get("MINJA_AUDIT"):
            from minja_audit import provenance
            new_item.update(provenance())
        long_term_memory.append(new_item)""",
    )

    for f in (EHR / "config.py", med):
        t = f.read_text(encoding="utf-8")
        if (
            '{"reasoning_effort": os.environ["MINJA_REASONING_EFFORT"]}' in t
            and "MINJA_EXTRA_BODY" not in t
        ):
            f.write_text(
                t.replace(
                    '{"reasoning_effort": os.environ["MINJA_REASONING_EFFORT"]}',
                    '(__import__("json").loads(os.environ["MINJA_EXTRA_BODY"]) if os.environ.get("MINJA_EXTRA_BODY") else {"reasoning_effort": os.environ["MINJA_REASONING_EFFORT"]})',
                ),
                encoding="utf-8",
            )
    t = med.read_text(encoding="utf-8")
    if _MED_OLD in t:
        med.write_text(t.replace(_MED_OLD, _MED_NEW), encoding="utf-8")
    sub(
        EHR / "config.py",
        """        "temperature": 0,
    }
    return llm_config_list""",
        """        "temperature": 0,
    }
    if os.environ.get("MINJA_REASONING_EFFORT"):
        llm_config_list["extra_body"] = (__import__("json").loads(os.environ["MINJA_EXTRA_BODY"]) if os.environ.get("MINJA_EXTRA_BODY") else {"reasoning_effort": os.environ["MINJA_REASONING_EFFORT"]})
    return llm_config_list""",
    )
    sub(
        med,
        """                    temperature=0,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                    )""",
        """                    temperature=0,
                    **({"max_completion_tokens": 800,
                        "extra_body": {"reasoning_effort": os.environ["MINJA_REASONING_EFFORT"]}}
                       if os.environ.get("MINJA_REASONING_EFFORT") and not os.environ.get("MINJA_EXTRA_BODY") else
                       {"max_tokens": 800, "top_p": 0.95, "frequency_penalty": 0,
                        "presence_penalty": 0, "stop": None,
                        **({"extra_body": json.loads(os.environ["MINJA_EXTRA_BODY"])}
                           if os.environ.get("MINJA_EXTRA_BODY") else {})})
                    )""",
    )

    calc = EHR / "tools" / "calculator.py"
    sub(
        calc,
        "import wolframalpha\n",
        r"""try:
    import wolframalpha
except ImportError:  # optional: see _local_formula
    wolframalpha = None
""",
    )
    sub(
        calc,
        r"""def WolframAlphaCalculator(input_query: str):
    try:""",
        r"""_FORMULA = re.compile(r"^[0-9eE.+\-*/(), ]*$")


def _local_formula(q: str):
    expr = q.replace("^", "**")
    if not _FORMULA.match(re.sub(r"\b(max|min|abs|round|sum)\b", "", expr)):
        raise Exception("Invalid input query for Calculator. Please check the input query or use other functions to do the computation.")
    fns = {"max": max, "min": min, "abs": abs, "round": round, "sum": sum}
    return str(eval(expr, {"__builtins__": {}}, fns))


def WolframAlphaCalculator(input_query: str):
    if wolframalpha is None or not os.environ.get("WOLFRAM_APP_ID"):
        return _local_formula(input_query)
    try:""",
    )
    sub(
        calc,
        '        wolfram_alpha_appid = "<YOUR_WOLFRAMALPHA_APP_ID>"',
        '        wolfram_alpha_appid = os.environ["WOLFRAM_APP_ID"]',
    )

    (EHR / "minja_audit.py").write_text(AUDIT_GLUE, encoding="utf-8")
    print(f"patched {EHR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
