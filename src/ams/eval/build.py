"""Memory systems built from an experiment config."""

from __future__ import annotations

from ams.core.lattice import Rung
from ams.eval.context import Config, RunContext
from ams.llm import model_of
from ams.memory import RawMemory, SimpleMemory
from ams.memory.base import MemorySystem


def build_memory(cfg: Config, ctx: RunContext) -> MemorySystem:
    m = {r: cfg.require(f"models.{r}") for r in ("memory", "agent", "embed")}
    gate = None
    if cfg.get("write_gate") == "consistency":
        from ams.baselines.write_gate import ConsistencyGate

        gate = ConsistencyGate(
            ctx.client(m["memory"], stage="write_gate"),
            model=model_of(m["memory"]),
            k=cfg.get("write_gate_k", 5),
            tau=cfg.get("write_gate_tau", 0.7),
            prompts=ctx.prompts,
        )
    embed = ctx.client(m["embed"], stage="embed")
    system = cfg.get("memory_system", "simple")
    if system == "raw":
        memory = RawMemory(embed_client=embed, embed_model=model_of(m["embed"]))
    elif system == "typed":
        from ams.eval.use_type import FixedUse, UseTypeClassifier
        from ams.memory import TypedStandingMemory

        fixed = cfg.get("typed.required_fixed")
        use_type = (
            FixedUse(rung=Rung.parse(fixed))
            if fixed
            else UseTypeClassifier(
                client=ctx.client(m["memory"], stage="use_type"),
                model=model_of(m["memory"]),
                prompts=ctx.prompts,
                version=cfg.get("prompts.classify_use_type", "v1"),
            )
        )
        memory = TypedStandingMemory(
            client=ctx.client(m["memory"], stage="memory_write"),
            model=model_of(m["memory"]),
            embed_client=embed,
            embed_model=model_of(m["embed"]),
            use_type=use_type,
            mode=cfg.get("typed.mode", "confine"),
            granularity=cfg.get("typed.granularity", 5),
            source_cap=cfg.get("typed.source_cap"),
            hops=cfg.get("typed.hops", 0),
            coreference=cfg.get("typed.coreference", False),
            scoped=cfg.get("typed.scoped", True),
            max_claims=cfg.get("memory.max_facts", 6),
            detect_conflicts=cfg.get("typed.detect_conflicts", True),
            prompts=ctx.prompts,
            extract_version=cfg.get("prompts.extract_typed_claims", "v1"),
            conflict_version=cfg.get("prompts.detect_conflict", "v1"),
            retention_version=cfg.get("prompts.retention_check"),
            endorsement=cfg.get("typed.endorsement", "quote"),
            retention_sentence_version=cfg.get("prompts.retention_sentence", "v1"),
            evidence_slots=cfg.get("typed.evidence_slots", 0),
            index=cfg.get("typed.index", "flat"),
            external_evidence=cfg.get("typed.external_evidence", True),
            index_check=cfg.get("typed.index_check", False),
            hub_cap=cfg.get("typed.hub_cap"),
            entity_merge=cfg.get("typed.entity_merge"),
            dialogue_extract_version=cfg.get("prompts.extract_dialogue_claims", "v1"),
        )
    else:
        memory = SimpleMemory(
            ctx.client(m["memory"], stage="memory_write"),
            model=model_of(m["memory"]),
            embed_client=embed,
            embed_model=model_of(m["embed"]),
            prompts=ctx.prompts,
            prompt_version=cfg.get("prompts.memory_extract", "v1"),
            max_facts=cfg.get("memory.max_facts", 6),
            gate=gate,
        )
    if cfg.get("write_isolation") == "drift":
        from ams.baselines.drift import InjectionIsolator, IsolatedMemory

        memory = IsolatedMemory(
            memory,
            InjectionIsolator(
                ctx.client(m["agent"], stage="drift_isolator"),
                model=model_of(m["agent"]),
                prompts=ctx.prompts,
            ),
        )
    return memory
