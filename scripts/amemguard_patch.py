#!/usr/bin/env python
"""Make the vendored A-MemGuard checkout runnable, without changing its method."""

# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "AMemGuard"

GLUE = '''"""Glue the vendored agents expect but the release omits.

`local_wikienv.py` / `medagent.py` do `from consistency import check_consistency`;
no such function exists in the released `consistency.py`. This builds their
`ConsistencyChecker` on an OpenAI-compatible provider and forwards the call.
Model / method come from the environment so the same code serves both agents.
"""

import os

from consistency import ConsistencyChecker, OpenAIModel, VLLMModel

_checker = None


def _get_checker():
    global _checker
    if _checker is None:
        model = os.environ.get("AMEMGUARD_MODEL", "gpt-4o-mini")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            provider = VLLMModel(model_name=model, base_url=base_url,
                                 api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        else:
            provider = OpenAIModel(model_name=model, api_key=os.environ.get("OPENAI_API_KEY"))
        _checker = ConsistencyChecker(model_provider=provider)
    return _checker


def check_consistency(query, memories, selected_indexes, mode="example", knn=True):
    """Signature as called by the agents; `mode` and `knn` are accepted and unused,
    exactly as the release's call sites pass them."""
    return _get_checker().check(
        query=query,
        memories=list(memories),
        selected_indexes=list(selected_indexes),
        method=os.environ.get("AMEMGUARD_METHOD", "llm"),
    )
'''


REALM_SHIM = (
    "try:\n"
    "    from transformers import RealmEmbedder, RealmForOpenQA\n"
    "except ImportError:  # removed in transformers 5; only --model realm needs them\n"
    "    class RealmEmbedder:  # noqa: D101\n"
    "        pass\n\n"
    "    class RealmForOpenQA:  # noqa: D101\n"
    "        pass\n"
)


TMA_GLUE = r'''"""TMA-NM as an EHRAgent audit method: their monitor, our glue.

Memory records are the agent's own past solutions, so they arrive on the AGENT
channel (act=INFORM: may inform, may not drive an action). The poisoned records
are appended by the attack code directly, with no envelope; under TMA-NM an
unregistered write is UNTRUSTED (act=NONE) and is withheld from the prompt.
That is the AgentPoison threat model (direct store write).

TMA_POISON_ORIGIN=agent labels the same records as self-writes instead: the
label a query-only injection (MINJA) obtains, because the record really was
written by the agent in a user session. Nothing is withheld then. The two runs
together show what the mechanism rests on: the label, not the content.

Separately, for every generated program the ACTION gate is evaluated the way
their AgentDojo experiment does it (user words endorse values; a tool call is
allowed iff each field value is endorsed or vouched) and logged to
TMA_GATE_LOG. It is logged, not enforced: it shows whether value-level gating
can tell DeleteDB('patient') from LoadDB('patient') when 'patient' is in the
user's question.
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

_TMA = Path(__file__).resolve().parents[2].parent / "mem-inv-bench" / "v2" / "code" / "tma.py"


def _tma():
    if "tma_upstream" in sys.modules:
        return sys.modules["tma_upstream"]
    if not _TMA.exists():
        raise FileNotFoundError(f"mem-inv-bench not vendored at {_TMA}; run scripts/fetch_third_party.sh")
    spec = importlib.util.spec_from_file_location("tma_upstream", _TMA)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tma_upstream"] = mod
    spec.loader.exec_module(mod)
    return mod


def _poison_origin(t):
    return t.Origin.AGENT if os.environ.get("TMA_POISON_ORIGIN", "untrusted") == "agent" else t.Origin.UNTRUSTED


def _monitor(t, query):
    chans = {
        "self": t.Channel("self", t.Origin.AGENT, "agent", b"k-self"),
        "attacker": t.Channel("attacker", _poison_origin(t), "unknown", b"k-atk"),
        "user": t.Channel("user", t.Origin.USER, "principal", b"k-user"),
    }
    reg = t.Registry()
    for c in chans.values():
        reg.register(c)
    mon = t.Monitor(reg, k=2)
    mon.ingest(chans["user"].send(query))
    # as in their exp8: the user's task is the one trusted principal, and every
    # value it names is endorsed
    for tok in set(re.findall(r"[A-Za-z0-9_./:@-]{3,}", query)):
        mon.endorse(tok, "principal")
    mon.tick_epoch()
    return mon, chans


_STATE = {}
_CALL = re.compile(r"\b([A-Z][A-Za-z]+)\(([^()]*)\)")


def tma_filter(query, memory, selected_indexes, untrusted_from):
    """M1+M2: keep only records whose act class permits informing."""
    t = _tma()
    mon, chans = _monitor(t, query)
    kept, items = [], {}
    for i in selected_indexes:
        i = int(i)
        ch = chans["attacker"] if i >= untrusted_from else chans["self"]
        it = mon.ingest(ch.send(json.dumps(memory[i], sort_keys=True)))
        items[i] = it
        if int(it.act) >= int(t.ActClass.INFORM):
            kept.append(i)
    _STATE["last"] = {"query": query, "items": items, "kept": kept, "mon": mon,
                      "selected": [int(i) for i in selected_indexes], "untrusted_from": untrusted_from}
    return kept


def log_action_gate(code):
    """Evaluate (not enforce) their action gate on every tool call in `code`."""
    st = _STATE.get("last")
    path = os.environ.get("TMA_GATE_LOG")
    if not st or not path:
        return
    t = _tma()
    mon = st["mon"]
    supporting = [st["items"][i] for i in st["kept"]]
    rows = []
    for tool, args in _CALL.findall(code or ""):
        first = args.split(",")[0].strip().strip("'\"")
        if not first or tool in ("FilterDB", "GetValue", "Calculate", "Calendar", "SQLInterpreter", "print"):
            continue
        allowed, verdicts, _ = mon.authorize_action({"db": first}, supporting, tier=t.Tier.ROUTINE,
                                                    policy={"db": "identifier"})
        rows.append({"tool": tool, "arg": first, "allowed": bool(allowed), "why": verdicts["db"]["why"]})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"question": st["query"],
                             "poison_origin": os.environ.get("TMA_POISON_ORIGIN", "untrusted"),
                             "selected": st["selected"], "kept": st["kept"],
                             "untrusted_from": st["untrusted_from"], "calls": rows}) + "\n")
'''


def sub(path: Path, old: str, new: str, *, count: int = 1, again: bool = False) -> None:
    """Replace `old` with `new`. Skipped when `new` is already present, so the
    patch is idempotent; `again=True` forces it (for replace-all passes)."""
    text = path.read_text(encoding="utf-8")
    if new in text and not again:
        return
    if old not in text:
        sys.exit(f"patch anchor not found in {path}: {old[:70]!r}")
    path.write_text(text.replace(old, new, -1 if count == 0 else count), encoding="utf-8")


def guard_import(path: Path, line: str, name: str) -> None:
    sub(
        path,
        line,
        f"try:\n    {line}\nexcept Exception:  # other baselines' deps; not needed here\n"
        f"    {name} = None",
    )


def main() -> int:
    if not ROOT.exists():
        sys.exit(f"{ROOT} missing; run scripts/fetch_third_party.sh first")

    for d in ("ReAct", "EhrAgent/ehragent"):
        (ROOT / d / "consistency_glue.py").write_text(GLUE, encoding="utf-8")

    env = ROOT / "ReAct" / "local_wikienv.py"
    guard_import(
        env, "from auditor_token import audit_and_sanitize_item", "audit_and_sanitize_item"
    )
    guard_import(env, "from perplexity import audit_memory_ppl", "audit_memory_ppl")
    guard_import(env, "from classifier import RandECDetector", "RandECDetector")
    sub(
        env,
        "from consistency import check_consistency",
        "from consistency_glue import check_consistency",
    )
    sub(
        env,
        'AutoTokenizer.from_pretrained("/dpr-ctx_encoder-single-nq-base")',
        'AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")',
    )
    sub(
        env,
        'DPRContextEncoder.from_pretrained("dpr-ctx_encoder-single-nq-base")',
        'DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")',
    )
    sub(
        env,
        "import torch\n",
        'import torch\nDEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n',
    )
    sub(env, '"cuda"', "DEVICE", count=0, again=True)
    sub(
        env,
        'DEVICE if torch.cuda.is_available() else "cpu"',
        '"cuda" if torch.cuda.is_available() else "cpu"',
    )

    sub(
        env,
        "from transformers import AutoTokenizer, DPRContextEncoder, RealmEmbedder",
        "from transformers import AutoTokenizer, DPRContextEncoder\n" + REALM_SHIM,
    )
    for utils_py in (ROOT / "algo" / "utils.py", ROOT / "EhrAgent" / "ehragent" / "utils.py"):
        sub(
            utils_py,
            "                          DPRQuestionEncoder,\n"
            "                          RealmEmbedder,\n"
            "                          RealmForOpenQA)\n",
            "                          DPRQuestionEncoder)\n" + REALM_SHIM,
        )

    run = ROOT / "ReAct" / "run_strategyqa_gpt3.5.py"
    sub(run, 'OPENAI_API_KEY = ""', 'OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")')
    sub(run, 'BASE_URL = ""', 'BASE_URL = os.environ.get("OPENAI_BASE_URL") or None')
    sub(run, 'model="gpt-4o-mini",', 'model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),')
    sub(run, "logprobs=1,", "logprobs=True,")
    sub(
        run,
        "    return response.choices[0].message.content\n",
        "    content = response.choices[0].message.content or ''\n"
        "    return (content, []) if return_probs else content\n",
    )
    sub(
        run,
        'print("trigger_token_list", trigger_token_list)',
        'trigger_token_list = os.environ["TRIGGER_TOKENS"].split()\n'
        'print("trigger_token_list", trigger_token_list)',
    )

    sub(
        run,
        "            _, action = parse_thought_action(action, i)\n",
        "            _, parsed_action = parse_thought_action(action, i)\n"
        "            action = (parsed_action or action).strip() or 'think[]'\n"
        "            thought = thought or fallback_thought\n",
    )

    sub(
        run,
        "        try:\n            return env.step(action)\n",
        "        try:\n"
        '            if "search[" in action and '
        'os.environ.get("REACT_QUERY", "context") == "context":\n'
        '                action = f"search[{current_context}]"\n'
        "            return env.step(action)\n",
    )
    sub(
        run,
        "env = local_wikienv.WikiEnv(",
        'import random\nrandom.seed(int(os.environ.get("SEED", "0")))\n'
        "env = local_wikienv.WikiEnv(",
    )

    med = ROOT / "EhrAgent" / "ehragent" / "medagent.py"
    guard_import(med, "from auditor import audit_and_sanitize_item", "audit_and_sanitize_item")
    guard_import(med, "from perplexity import audit_memory_ppl", "audit_memory_ppl")
    guard_import(med, "from classifier import RandECDetector", "RandECDetector")
    sub(
        med,
        "from consistency import check_consistency",
        "from consistency_glue import check_consistency",
    )
    sub(med, 'OPENAI_API_KEY = ""', 'OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")')
    sub(med, 'ENGINE = "gpt-4o-mini"', 'ENGINE = os.environ.get("AGENT_MODEL", "gpt-4o-mini")')
    sub(med, 'BASE_URL = ""', 'BASE_URL = os.environ.get("OPENAI_BASE_URL") or None')
    sub(
        med,
        "import torch\n",
        'import os\nimport torch\nDEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n',
    )
    sub(med, '"cuda"', "DEVICE", count=0, again=True)
    sub(
        med,
        'DEVICE if torch.cuda.is_available() else "cpu"',
        '"cuda" if torch.cuda.is_available() else "cpu"',
    )

    (ROOT / "EhrAgent" / "ehragent" / "tma_glue.py").write_text(TMA_GLUE, encoding="utf-8")
    sub(
        med,
        "        self.memory = long_term_memory\n",
        "        self.memory = long_term_memory\n"
        "        self.untrusted_from = len(long_term_memory)  # rows appended after this arrived with no envelope\n",
    )
    sub(
        med,
        "        selected_indexes = sorted_indices[:self.num_shots]\n",
        "        selected_indexes = sorted_indices[:self.num_shots]\n"
        '        if self.audit_method == "TMA-NM":\n'
        "            from tma_glue import tma_filter\n"
        "            selected_indexes = tma_filter(query, self.memory, selected_indexes, self.untrusted_from)\n",
    )
    sub(
        med,
        "            code = self.generate_code(self.config_list[0], init_message)\n",
        "            code = self.generate_code(self.config_list[0], init_message)\n"
        '            if self.audit_method == "TMA-NM":\n'
        "                from tma_glue import log_action_gate\n"
        "                log_action_gate(code)\n",
    )

    main_py = ROOT / "EhrAgent" / "ehragent" / "main.py"
    sub(main_py, "enable_memory_audit=args.audit_method", "audit_method=args.audit_method")
    sub(
        main_py,
        "trigger_token_list = ['put', 'your', 'trigger', 'tokens', 'in', 'this', 'list']",
        'trigger_token_list = os.environ["TRIGGER_TOKENS"].split()',
    )
    cfg = ROOT / "EhrAgent" / "ehragent" / "config.py"
    sub(cfg, '"api_key": "",', '"api_key": __import__("os").environ.get("OPENAI_API_KEY", "x"),')
    utils = ROOT / "algo" / "utils.py"
    sub(utils, 'api_key = ""\n', 'api_key = __import__("os").environ.get("OPENAI_API_KEY", "")\n')

    opt = ROOT / "algo" / "trigger_optimization.py"
    sub(
        opt,
        "from agentdriver.reasoning.prompt_reasoning import *\nimport wandb\n",
        "try:\n    from agentdriver.reasoning.prompt_reasoning import *\n"
        "except ImportError:  # Agent-Driver is not vendored; only --agent ad needs it\n"
        "    pass\ntry:\n    import wandb\nexcept ImportError:\n    wandb = None\n",
    )
    sub(
        ROOT / "algo" / "utils.py",
        'max_memory_map = {\n        0: "140GiB",\n    }\n',
        "max_memory_map = None  # let accelerate place it on whatever exists\n",
    )

    for utils_py in (ROOT / "algo" / "utils.py", ROOT / "EhrAgent" / "ehragent" / "utils.py"):
        sub(
            utils_py,
            "    long_term_memory = [dict(t) for t in {tuple(d.items()) for d in long_term_memory}]\n",
            "    long_term_memory = [dict(t) for t in sorted({tuple(d.items()) for d in long_term_memory})]\n",
        )

    print(f"patched {ROOT}")
    sub(
        ROOT / "ReAct" / "consistency.py",
        "except (json.JSONDecodeError, TypeError) as e:",
        "except (json.JSONDecodeError, TypeError, AttributeError) as e:",
    )
    sub(
        ROOT / "ReAct" / "consistency.py",
        "            if 0 <= mem_idx < len(memories):",
        "            if isinstance(mem_idx, int) and 0 <= mem_idx < len(memories):",
    )
    sub(
        ROOT / "ReAct" / "consistency.py",
        "             if 0 <= mem_idx < len(memories):",
        "             if isinstance(mem_idx, int) and 0 <= mem_idx < len(memories):",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
