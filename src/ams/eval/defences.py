"""Defence construction, shared by every runner (MPBench, LoCoMo, LongMemEval)."""

from __future__ import annotations

from ams.baselines.base import NoDefence
from ams.eval.context import Config, RunContext
from ams.llm import model_of


def build_defence(kind: str, cfg: Config, spec: str, ctx: RunContext):
    def llm(stage: str):
        return ctx.client(spec, stage=stage), model_of(spec)

    def sensitive():
        from ams.baselines.memlineage import SensitiveAction

        c, mdl = llm("sensitive_action")
        return SensitiveAction(c, mdl, prompts=ctx.prompts)

    if kind == "no_defence":
        return NoDefence()
    if kind == "a_memguard":
        from ams.baselines.amemguard import AMemGuard

        return AMemGuard(model=model_of(spec), client=ctx.client(spec, stage="a_memguard"))
    if kind in ("tma_nm_values", "tma_nm_strict", "tma_nm_full"):
        from ams.baselines.tma_nm import TmaNm

        return TmaNm(mode=kind.rsplit("_", 1)[1])
    if kind == "randec_distil":
        from ams.baselines.amemguard_borrowed import RandECClassifier

        return RandECClassifier(weights=cfg.get("randec.weights"))
    if kind == "llm_sanitiser":
        from ams.baselines.amemguard_borrowed import LLMSanitiser

        c, mdl = llm("sanitiser")
        return LLMSanitiser(c, model=mdl, prompts=ctx.prompts)
    if kind == "memlineage_coarse":
        from ams.baselines.memlineage import MemLineage, coarse

        return MemLineage(coarse, sensitive(), tau=0.0, label="coarse")
    if kind == "memlineage_lm":
        from ams.baselines.memlineage import LmSelfEval, MemLineage

        c, mdl = llm("memlineage_attribution")
        return MemLineage(
            LmSelfEval(c, mdl, prompts=ctx.prompts), sensitive(), tau=0.3, label="lm_self_eval"
        )
    if kind in ("integrity_gate", "integrity_gate_all"):
        from ams.baselines.integrity_gate import IntegrityGate

        return IntegrityGate(consequential=None if kind.endswith("_all") else sensitive())
    raise ValueError(f"unknown defence {kind!r}")
