"""Typed-standing memory: nothing is withheld as an entry, only as a use."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ams.core import sentinel
from ams.core.lattice import (
    GLOBAL,
    Domain,
    Rung,
    Standing,
    coarsen,
    meet_all,
    permits,
)
from ams.core.lattice import (
    principal as principal_domain,
)
from ams.core.prompts import PromptLibrary
from ams.core.spans import segment
from ams.core.standing import SpanRole, elevation_quote_is_genuine
from ams.llm import ChatRequest, EmbedRequest, LLMClient, Message
from ams.memory.ann import PartitionedIndex
from ams.memory.base import MemoryEntry, RetrievalResult, WriteResult
from ams.memory.evidence import Chunk, cut
from ams.memory.graph import Claim, ClaimGraph, ConflictVerdict, Source
from ams.memory.index import VectorIndex

_ARTIFACT_BREAK = re.compile(r"\n-{3,}\n|^(?=From:\s)", re.MULTILINE)


def split_artifacts(context: str) -> list[str]:
    """Cut a context into the artifacts it is made of."""
    parts = [p.strip() for p in _ARTIFACT_BREAK.split(context or "") if p and p.strip()]
    return parts or ([context.strip()] if (context or "").strip() else [])


def _overlap(a: str, b: str) -> float:
    """Crude lexical overlap, used only to attribute a claim to the artifact it
    most likely came from. Attribution affects `source_cap` and attribution
    rendering; it is not a gate input, because every artifact in a context has the
    same standing anyway."""
    wa = {w for w in re.findall(r"[a-z0-9]{3,}", a.lower())}
    wb = {w for w in re.findall(r"[a-z0-9]{3,}", b.lower())}
    return len(wa & wb) / len(wa) if wa else 0.0


def _parse_claims(text: str, limit: int) -> list[dict]:
    body = (text or "").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list):
        return []
    return [c for c in claims if isinstance(c, dict) and str(c.get("text", "")).strip()][:limit]


def _kind_cap(sentences: list[dict]) -> Rung | None:
    """The highest kind these retention sentences asked for; None if there are none.
    An unreadable kind caps at `description`, i.e. fails closed."""
    if not sentences:
        return None
    caps = []
    for s in sentences:
        try:
            caps.append(Rung.parse(s.get("kind") or ""))
        except ValueError:
            caps.append(Rung.DESCRIPTION)
    return max(caps)


@dataclass
class TypedStandingMemory:
    """Claims in a graph, gated by (kind, standing) against what a query needs."""

    client: LLMClient
    model: str
    embed_client: LLMClient
    embed_model: str
    use_type: object
    mode: str = "confine"
    source_cap: int | None = None
    hops: int = 0
    coreference: bool = False
    scoped: bool = True
    granularity: int = 5
    max_claims: int = 6
    detect_conflicts: bool = True
    prompts: PromptLibrary = field(default_factory=PromptLibrary)
    extract_version: str = "v1"
    conflict_version: str = "v1"
    retention_version: str | None = None
    endorsement: str = "quote"
    retention_sentence_version: str = "v1"
    evidence_slots: int = 0
    index: str = "flat"
    external_evidence: bool = True
    index_check: bool = False
    hub_cap: int | None = None
    entity_merge: float | None = None
    dialogue_extract_version: str = "v1"
    temperature: float = 0.0

    _graphs: dict[str, ClaimGraph] = field(default_factory=dict, repr=False)
    _vectors: dict[str, VectorIndex] = field(default_factory=dict, repr=False)
    _chunks: dict[str, list[Chunk]] = field(default_factory=dict, repr=False)
    _chunk_vecs: dict[str, VectorIndex] = field(default_factory=dict, repr=False)
    _parts: dict[str, PartitionedIndex] = field(default_factory=dict, repr=False)
    _chunk_parts: dict[str, PartitionedIndex] = field(default_factory=dict, repr=False)
    _chunk_by_id: dict[str, Chunk] = field(default_factory=dict, repr=False)
    _entity_names: dict[str, tuple[list[str], VectorIndex]] = field(
        default_factory=dict, repr=False
    )
    n_entity_merges: int = 0
    n_index_checks: int = 0
    index_overlap: float = 0.0
    n_unparsed_type: int = 0
    tracks_standing = True
    n_elevations: int = 0
    n_elevation_rejected: int = 0
    n_elevation_not_requested: int = 0
    n_principal_pass_claims: int = 0
    n_referential_elevations: int = 0

    @property
    def name(self) -> str:
        bits = [f"mode={self.mode}", f"hops={self.hops}", f"levels={self.granularity}"]
        if self.source_cap:
            bits.append(f"cap={self.source_cap}")
        if self.coreference:
            bits.append("coref")
        if not self.scoped:
            bits.append("noscope")
        fixed = getattr(self.use_type, "rung", None)
        if fixed is not None:
            bits.append(f"fixedT={fixed.name.lower()}")
        if self.retention_version and self.endorsement == "quote":
            bits.append(f"retain={self.retention_version}")
        if self.endorsement == "sentence":
            bits.append("endorse=sentence")
        if self.evidence_slots:
            bits.append(f"m={self.evidence_slots}")
        if self.index != "flat":
            bits.append(f"index={self.index}")
        if not self.external_evidence:
            bits.append("ext_evidence=off")
        return "typed_standing[" + ",".join(bits) + "]"

    def _graph(self, session_id: str) -> ClaimGraph:
        if session_id not in self._graphs:
            self._graphs[session_id] = ClaimGraph(coreference=self.coreference)
        return self._graphs[session_id]

    def write(
        self, *, session_id: str, user_query: str, context: str, principal: str = "user"
    ) -> WriteResult:
        graph = self._graph(session_id)
        turn = graph.add_source(
            Source(
                source_id=graph.new_source_id(),
                artifact=f"{session_id}:turn",
                role=SpanRole.PRINCIPAL_TURN,
                text=user_query or "",
                principal_id=principal,
                seq=len(graph),
            )
        )
        artifacts = [
            graph.add_source(
                Source(
                    source_id=graph.new_source_id(),
                    artifact=f"{session_id}:art{i}",
                    role=SpanRole.EXTERNAL_BODY,
                    text=part,
                    seq=len(graph),
                )
            )
            for i, part in enumerate(split_artifacts(context))
        ]

        if self.evidence_slots:
            first = artifacts if self.endorsement in ("sentence", "none") else [turn, *artifacts]
            for src in first:
                self._index_sentences(graph, session_id, src)
        if self.endorsement == "sentence":
            written = self._write_sentence_endorsed(
                graph, session_id, user_query or "", context or "", turn, artifacts
            )
            self._embed_written(graph, session_id, written)
            return WriteResult(written=tuple(written))

        raw_claims = self._extract(user_query or "", context or "")
        if not raw_claims:
            return WriteResult(written=())

        retain: list[bool] = []

        def asks_to_retain() -> bool:
            if not retain:
                retain.append(self._asks_to_retain(user_query or ""))
            return retain[0]

        written: list[MemoryEntry] = []
        for raw in raw_claims:
            text = str(raw.get("text", "")).strip()
            src = max(artifacts, key=lambda s: _overlap(text, s.text)) if artifacts else turn
            standing = src.standing()

            quote = str(raw.get("retention_quote", "")).strip()
            endorsed: tuple[str, ...] = ()
            if quote and self.endorsement != "none":
                if not elevation_quote_is_genuine(quote, user_query or ""):
                    self.n_elevation_rejected += 1
                elif not asks_to_retain():
                    self.n_elevation_not_requested += 1
                else:
                    es = turn.standing()
                    standing = Standing(max(standing.level, es.level), es.domain)
                    endorsed = (turn.source_id,)
                    self.n_elevations += 1
            written.append(self._store(graph, session_id, raw, standing, src, endorsed))

        self._embed_written(graph, session_id, written)
        return WriteResult(written=tuple(written))

    def prefetch(self, *, user_query: str, context: str) -> None:
        """Every model and embedding call `write` will make for this interaction,
        without touching the store. Pooled writes must be sequential (one store,
        session order), so the runner warms the cache concurrently first and the
        sequential pass then replays from it. Must mirror `write` exactly."""
        user_query, context = user_query or "", context or ""
        if self.evidence_slots:
            texts = split_artifacts(context)
            if self.endorsement != "sentence":
                texts = [user_query, *texts]
            for text in texts:
                sents = [sp.text for sp in segment(text)]
                if not sents:
                    continue
                vecs = self.embed_client.embed(
                    EmbedRequest(texts=tuple(sents), model=self.embed_model)
                ).vectors
                chunks = ["\n".join(sents[i] for i in run) for run in cut(sents, vecs)]
                self.embed_client.embed(EmbedRequest(texts=tuple(chunks), model=self.embed_model))
        if self.endorsement == "sentence":
            labels = (
                classify_retention_sentences(
                    self.client,
                    self.model,
                    self.prompts,
                    user_query,
                    version=self.retention_sentence_version,
                )
                if user_query.strip()
                else []
            )
            self_cap = _kind_cap([s for s in labels if s["keep"] and s["scope"] == "self"])
            if self.evidence_slots and any(s["keep"] for s in labels):
                sents = [sp.text for sp in segment(user_query)]
                vecs = self.embed_client.embed(
                    EmbedRequest(texts=tuple(sents), model=self.embed_model)
                ).vectors
                chunks = ["\n".join(sents[i] for i in run) for run in cut(sents, vecs)]
                self.embed_client.embed(EmbedRequest(texts=tuple(chunks), model=self.embed_model))
            raw = []
            if self_cap is not None:
                raw += self._extract(user_query, "")
            if context.strip() or self_cap is None:
                raw += self._extract(user_query, context)
        else:
            raw = self._extract(user_query, context)
            if self.retention_version and any(
                elevation_quote_is_genuine(str(r.get("retention_quote", "")).strip(), user_query)
                for r in raw
            ):
                self._asks_to_retain(user_query)
        if raw:
            texts = tuple(str(r.get("text", "")).strip() for r in raw)
            self.embed_client.embed(EmbedRequest(texts=texts, model=self.embed_model))
        if self.entity_merge is not None:
            for r in raw:
                for name in (
                    str(r.get("subject") or "unknown"),
                    str(r.get("object") or "").strip(),
                ):
                    if name.strip():
                        self.embed_client.embed(EmbedRequest(texts=(name,), model=self.embed_model))

    def _extract(self, query: str, context: str) -> list[dict]:
        rendered = self.prompts.get("extract_typed_claims", self.extract_version).render(
            query=query.strip(), context=context.strip(), max_claims=self.max_claims
        )
        return self._claims_from(rendered)

    def _claims_from(self, rendered: str) -> list[dict]:
        reply = self.client.complete(
            ChatRequest(
                messages=(Message("user", rendered),),
                model=self.model,
                temperature=self.temperature,
                max_tokens=900 if self.max_claims <= 6 else 2400,
            )
        ).text
        return _parse_claims(reply, self.max_claims)

    def _store(
        self,
        graph: ClaimGraph,
        session_id: str,
        raw: dict,
        standing: Standing,
        src: Source,
        endorsed: tuple[str, ...] = (),
    ) -> MemoryEntry:
        try:
            ctype = Rung.parse(str(raw.get("type", "")))
        except ValueError:
            self.n_unparsed_type += 1
            ctype = Rung.DESCRIPTION
        subject = graph.entity(self._entity_name(session_id, str(raw.get("subject") or "unknown")))
        obj_name = str(raw.get("object") or "").strip()
        obj = graph.entity(self._entity_name(session_id, obj_name)) if obj_name else None
        claim = graph.add_claim(
            Claim(
                claim_id=graph.new_claim_id(),
                text=str(raw.get("text", "")).strip(),
                claim_type=ctype,
                standing=standing,
                source_id=src.source_id,
                subject_id=subject.entity_id,
                object_id=obj.entity_id if obj else None,
                relation=str(raw.get("relation") or "").strip().lower(),
                endorsed_by=endorsed,
                session_id=session_id,
                seq=len(graph),
            )
        )
        return self._entry(claim, graph)

    def _entity_name(self, session_id: str, name: str) -> str:
        """The name to file an entity under. With `entity_merge`, a name whose
        embedding is close enough to an earlier one is filed under that one. Reads
        claim content, so it carries no security weight (§3): it only decides which
        claims are neighbours for traversal and conflict grouping."""
        if self.entity_merge is None or not name.strip():
            return name
        names, index = self._entity_names.setdefault(session_id, ([], VectorIndex()))
        if name in names:
            return name
        vec = self.embed_client.embed(EmbedRequest(texts=(name,), model=self.embed_model)).vectors[
            0
        ]
        best = max(index.scores(vec).items(), key=lambda kv: kv[1], default=(None, -1.0))
        if best[0] is not None and best[1] >= self.entity_merge:
            self.n_entity_merges += 1
            return names[best[0]]
        index.add([len(names)], [vec])
        names.append(name)
        return name

    def _embed_written(self, graph: ClaimGraph, session_id: str, written: list) -> None:
        if not written:
            return
        vecs = self.embed_client.embed(
            EmbedRequest(
                texts=tuple(graph.claim(e.entry_id).text for e in written),
                model=self.embed_model,
            )
        ).vectors
        self._vectors.setdefault(session_id, VectorIndex()).add([e.entry_id for e in written], vecs)
        if self.index != "flat":
            parts = self._parts.setdefault(session_id, PartitionedIndex(self.index))
            for e, v in zip(written, vecs, strict=True):
                c = graph.claim(e.entry_id)
                parts.add((c.claim_type, c.standing), [c.claim_id], [v])

    def _write_sentence_endorsed(
        self,
        graph: ClaimGraph,
        session_id: str,
        user_query: str,
        context: str,
        turn: Source,
        artifacts: list[Source],
    ) -> list[MemoryEntry]:
        """Spec §10.3. The principal's sentences are read on their own; only a
        self-contained retention request endorses, and it endorses only what was
        extracted from the principal's words with no artifact in the input. A
        referential request may raise artifact claims, capped at the kind asked."""
        labels = (
            classify_retention_sentences(
                self.client,
                self.model,
                self.prompts,
                user_query,
                version=self.retention_sentence_version,
            )
            if user_query.strip()
            else []
        )
        self_cap = _kind_cap([s for s in labels if s["keep"] and s["scope"] == "self"])
        ref = [s for s in labels if s["keep"] and s["scope"] == "referential"]
        ref_cap = _kind_cap(ref)
        es = turn.standing()
        if self.evidence_slots and any(s["keep"] for s in labels):
            self._index_sentences(graph, session_id, turn)
        written: list[MemoryEntry] = []

        if self_cap is not None:
            principal_standing = Standing(min(es.level, self_cap), es.domain)
            for raw in self._extract(user_query, ""):
                written.append(
                    self._store(graph, session_id, raw, principal_standing, turn, (turn.source_id,))
                )
                self.n_principal_pass_claims += 1
                self.n_elevations += 1

        if context.strip() or self_cap is None:
            for raw in self._extract(user_query, context):
                text = str(raw.get("text", "")).strip()
                src = max(artifacts, key=lambda s: _overlap(text, s.text)) if artifacts else turn
                standing = src.standing()
                endorsed: tuple[str, ...] = ()
                quote = str(raw.get("retention_quote", "")).strip()
                if (
                    ref_cap is not None
                    and quote
                    and elevation_quote_is_genuine(quote, user_query)
                    and any(quote in s["text"] or s["text"] in quote for s in ref)
                ):
                    standing = Standing(max(standing.level, min(es.level, ref_cap)), es.domain)
                    endorsed = (turn.source_id,)
                    self.n_referential_elevations += 1
                    self.n_elevations += 1
                elif quote:
                    self.n_elevation_not_requested += 1
                written.append(self._store(graph, session_id, raw, standing, src, endorsed))
        return written

    def write_dialogue(
        self,
        *,
        session_id: str,
        turns: list[tuple[str, str]],
        date: str = "",
        principal: str,
    ) -> WriteResult:
        """One dialogue session with a fixed principal (spec §10.2)."""
        graph = self._graph(session_id)
        p = principal.strip().lower()
        sources: list[tuple[str, Source]] = []
        prev: Source | None = None
        everyone = principal == "*"
        for speaker, text in turns:
            mine = everyone or speaker.strip().lower() == p
            src = graph.add_source(
                Source(
                    source_id=graph.new_source_id(),
                    artifact=f"{session_id}:{date}:{speaker}",
                    role=SpanRole.PRINCIPAL_TURN if mine else SpanRole.COUNTERPART_TURN,
                    text=text,
                    principal_id=("user" if everyone else principal) if mine else speaker,
                    seq=len(graph),
                )
            )
            if prev is not None:
                graph.link_next(prev.source_id, src.source_id)
            prev = src
            sources.append((f"{speaker}: {text}", src))
        if self.evidence_slots and sources:
            self._index_units(graph, session_id, sources, prefix=f"DATE: {date}")

        mine_lines = [line for line, s in sources if s.role == SpanRole.PRINCIPAL_TURN]
        theirs = [s for _, s in sources if s.role == SpanRole.COUNTERPART_TURN]
        written: list[MemoryEntry] = []
        prompt = self.prompts.get("extract_dialogue_claims", self.dialogue_extract_version)
        if mine_lines:
            anchor = next(s for _, s in sources if s.role == SpanRole.PRINCIPAL_TURN)
            for raw in self._claims_from(
                prompt.render(date=date, dialogue="\n".join(mine_lines), max_claims=self.max_claims)
            ):
                written.append(self._store(graph, session_id, raw, anchor.standing(), anchor))
        if theirs:
            standing = meet_all([s.standing() for s in theirs])
            for raw in self._claims_from(
                prompt.render(
                    date=date,
                    dialogue="\n".join(line for line, _ in sources),
                    max_claims=self.max_claims,
                )
            ):
                written.append(self._store(graph, session_id, raw, standing, theirs[0]))
        self._embed_written(graph, session_id, written)
        return WriteResult(written=tuple(written))

    def _index_sentences(self, graph: ClaimGraph, session_id: str, src: Source) -> None:
        """Sentence spans of one source, linked by `next`, cut into evidence chunks."""
        units: list[tuple[str, Source]] = []
        prev: Source | None = None
        for sp in segment(src.text):
            s = graph.add_source(
                Source(
                    source_id=graph.new_source_id(),
                    artifact=src.artifact,
                    role=src.role,
                    text=sp.text,
                    principal_id=src.principal_id,
                    tool_domain=src.tool_domain,
                    seq=len(graph),
                )
            )
            if prev is not None:
                graph.link_next(prev.source_id, s.source_id)
            prev = s
            units.append((sp.text, s))
        if units:
            self._index_units(graph, session_id, units, prefix="")

    def _index_units(
        self, graph: ClaimGraph, session_id: str, units: list[tuple[str, Source]], *, prefix: str
    ) -> None:
        texts = [t for t, _ in units]
        vecs = self.embed_client.embed(
            EmbedRequest(texts=tuple(texts), model=self.embed_model)
        ).vectors
        chunks: list[Chunk] = []
        for run in cut(texts, vecs):
            body = "\n".join(texts[i] for i in run)
            chunks.append(
                Chunk(
                    chunk_id=f"e{graph.new_claim_id()}",
                    text=f"{prefix}\n{body}" if prefix else body,
                    standing=meet_all([units[i][1].standing() for i in run]),
                    source_ids=tuple(units[i][1].source_id for i in run),
                    label=units[run[0]][1].artifact,
                )
            )
        cvecs = self.embed_client.embed(
            EmbedRequest(texts=tuple(c.text for c in chunks), model=self.embed_model)
        ).vectors
        self._chunks.setdefault(session_id, []).extend(chunks)
        self._chunk_vecs.setdefault(session_id, VectorIndex()).add(
            [c.chunk_id for c in chunks], cvecs
        )
        if self.index != "flat":
            parts = self._chunk_parts.setdefault(session_id, PartitionedIndex(self.index))
            for c, v in zip(chunks, cvecs, strict=True):
                parts.add(("chunk", c.standing), [c.chunk_id], [v])
                self._chunk_by_id[c.chunk_id] = c

    def retrieve(
        self, *, session_id: str, query: str, k: int = 4, principal: str = "user"
    ) -> RetrievalResult:
        graph = self._graphs.get(session_id)
        if graph is None or not (len(graph) or self._chunks.get(session_id)):
            return RetrievalResult(mode=self.mode)

        required = coarsen(self.use_type.required(query), self.granularity)
        asked_in: Domain = principal_domain(principal)
        about = (getattr(self.use_type, "subject", None) or (lambda q: ""))(query)
        qv = self.embed_client.embed(EmbedRequest(texts=(query,), model=self.embed_model)).vectors[
            0
        ]
        if self.index != "flat":
            if self.mode == "off":
                raise ValueError("mode 'off' is implemented on the flat index only")
            return self._retrieve_partitioned(graph, session_id, qv, k, required, asked_in, about)
        claims = list(graph.claims())
        eligible = (
            claims
            if self.mode == "off"
            else [
                c
                for c in claims
                if self._admits(c.claim_type, c.standing, required, asked_in, about)
            ]
        )
        index = self._vectors.get(session_id)
        sims = index.scores(qv) if index is not None else {}

        def score(c: Claim) -> float:
            return sims.get(c.claim_id, -1.0)

        ranked = sorted(eligible, key=score, reverse=True)
        seeds = ranked[:k]
        if self.hops:
            seeds = ranked[: max(1, k // 2)]
            eligible_ids = {c.claim_id for c in eligible}
            hop = graph.expand(
                seeds,
                hops=self.hops,
                keep=lambda c: c.claim_id in eligible_ids,
                per_entity=self.hub_cap,
                rank=score,
            )
            hop = sorted(hop, key=score, reverse=True)[: k - len(seeds)]
            chosen = {c.claim_id for c in seeds + hop}
            seeds = seeds + hop + [c for c in ranked if c.claim_id not in chosen]
        per_source: dict[str, int] = {}
        order = seeds if self.hops else sorted(seeds, key=score, reverse=True)
        served = self._cap(order, k, per_source)
        if self.mode == "confine" and len(served) < k:
            chosen = {c.claim_id for c in served}
            rest = [c for c in claims if c.claim_id not in chosen]
            served += self._cap(sorted(rest, key=score, reverse=True), k - len(served), per_source)

        verdicts = {
            c.claim_id: {
                "type": c.claim_type.name.lower(),
                "standing": str(c.standing),
                "usable_as": c.usable.name.lower(),
                "eligible": c in eligible,
                "source": c.source_id,
            }
            for c in served
        }
        if self.detect_conflicts and self.mode != "off":
            for v in self._resolve_conflicts(graph, served):
                for cid in v.contested:
                    verdicts.setdefault(cid, {})["contested"] = v.detail

        entries = [self._entry(c, graph, required=required) for c in served]
        if self.evidence_slots:
            entries += self._evidence(session_id, qv, required, asked_in, about)
        return RetrievalResult(
            entries=tuple(entries),
            required=required.name.lower(),
            eligible_empty=not eligible,
            mode=self.mode,
            verdicts=verdicts,
        )

    def _retrieve_partitioned(
        self,
        graph: ClaimGraph,
        session_id: str,
        qv,
        k: int,
        required: Rung,
        asked_in: Domain,
        about: str,
    ) -> RetrievalResult:
        """Spec §11. Same read path as the flat one, but the eligible set is never
        enumerated: only the partitions the gate admits are searched, so the cost
        does not grow with what the gate refuses."""
        parts = self._parts.get(session_id) or PartitionedIndex(self.index)
        admitted = [key for key in parts.keys() if self._admits(*key, required, asked_in, about)]
        width = k * (4 if (self.source_cap or self.hops) else 1)
        hits = parts.search(qv, width, admitted)
        sims = dict(hits)

        def score(c: Claim) -> float:
            if c.claim_id not in sims:
                sims[c.claim_id] = parts.score(qv, c.claim_id)
            return sims[c.claim_id]

        def eligible(c: Claim) -> bool:
            return self._admits(c.claim_type, c.standing, required, asked_in, about)

        ranked = [graph.claim(i) for i, _ in hits]
        seeds = ranked[:k]
        if self.hops:
            seeds = ranked[: max(1, k // 2)]
            hop = graph.expand(
                seeds, hops=self.hops, keep=eligible, per_entity=self.hub_cap, rank=score
            )
            hop = sorted(hop, key=score, reverse=True)[: k - len(seeds)]
            chosen = {c.claim_id for c in seeds + hop}
            seeds = seeds + hop + [c for c in ranked if c.claim_id not in chosen]
        per_source: dict[str, int] = {}
        order = seeds if self.hops else sorted(seeds, key=score, reverse=True)
        served = self._cap(order, k, per_source)
        if self.mode == "confine" and len(served) < k:
            chosen = {c.claim_id for c in served}
            rest = [
                graph.claim(i)
                for i, _ in parts.search(qv, width + k, parts.keys())
                if i not in chosen
            ]
            served += self._cap(rest, k - len(served), per_source)

        if self.index_check:
            flat = self._vectors.get(session_id)
            fs = flat.scores(qv) if flat is not None else {}
            exact = sorted(
                (c for c in graph.claims() if eligible(c)),
                key=lambda c: fs.get(c.claim_id, -1.0),
                reverse=True,
            )[:k]
            if exact:
                got = {c.claim_id for c in ranked[:k]}
                self.index_overlap += len(got & {c.claim_id for c in exact}) / len(exact)
                self.n_index_checks += 1

        verdicts = {
            c.claim_id: {
                "type": c.claim_type.name.lower(),
                "standing": str(c.standing),
                "usable_as": c.usable.name.lower(),
                "eligible": eligible(c),
                "source": c.source_id,
            }
            for c in served
        }
        if self.detect_conflicts:
            for v in self._resolve_conflicts(graph, served):
                for cid in v.contested:
                    verdicts.setdefault(cid, {})["contested"] = v.detail
        entries = [self._entry(c, graph, required=required) for c in served]
        if self.evidence_slots:
            entries += self._evidence(session_id, qv, required, asked_in, about)
        return RetrievalResult(
            entries=tuple(entries),
            required=required.name.lower(),
            eligible_empty=not hits,
            mode=self.mode,
            verdicts=verdicts,
        )

    def _admits(
        self, ctype: Rung, standing: Standing, required: Rung, asked_in: Domain, about: str
    ) -> bool:
        """The gate: `min(type, standing) >= required` in a covering domain. A
        counterpart's standing holds in their own domain, so it also admits when
        the question is about them (§10.2)."""
        t = coarsen(ctype, self.granularity)
        st = Standing(
            coarsen(standing.level, self.granularity),
            standing.domain if self.scoped else GLOBAL,
        )
        if permits(t, st, required=required, asked_in=asked_in):
            return True
        d = st.domain
        return (
            d.kind == "person"
            and bool(about)
            and d.name == about.strip().lower()
            and permits(t, st, required=required, asked_in=d)
        )

    def _evidence(
        self, session_id: str, qv, required: Rung, asked_in: Domain, about: str
    ) -> list[MemoryEntry]:
        """`m` evidence chunks (§10.1). Above description under withhold, only
        chunks whose own standing admits the use enter — never external text."""

        def admitted(ch: Chunk) -> bool:
            return self._admits(ch.standing.level, ch.standing, required, asked_in, about)

        open_all = required == Rung.DESCRIPTION or self.mode in ("confine", "off")

        def source_ok(standing: Standing) -> bool:
            return self.mode == "off" or self.external_evidence or standing.level > Rung.DESCRIPTION

        if self.index != "flat":
            parts = self._chunk_parts.get(session_id) or PartitionedIndex(self.index)
            keys = [
                key
                for key in parts.keys()
                if source_ok(key[1])
                and (open_all or self._admits(key[1].level, key[1], required, asked_in, about))
            ]
            pool = [self._chunk_by_id[i] for i, _ in parts.search(qv, self.evidence_slots, keys)]
        else:
            chunks = self._chunks.get(session_id, [])
            index = self._chunk_vecs.get(session_id)
            sims = index.scores(qv) if index is not None else {}
            pool = [ch for ch in chunks if source_ok(ch.standing) and (open_all or admitted(ch))]
            pool.sort(key=lambda ch: sims.get(ch.chunk_id, -1.0), reverse=True)
        out = []
        for ch in pool[: self.evidence_slots]:
            usable = required if required > Rung.DESCRIPTION and admitted(ch) else Rung.DESCRIPTION
            if self.mode == "off":
                out.append(
                    MemoryEntry(
                        entry_id=ch.chunk_id,
                        text=ch.text,
                        parent_ids=ch.source_ids,
                        metadata={
                            "evidence": True,
                            "standing": str(ch.standing),
                            "raw_text": ch.text,
                        },
                    )
                )
                continue
            out.append(
                MemoryEntry(
                    entry_id=ch.chunk_id,
                    text=sentinel.render(
                        ch.chunk_id,
                        f"source: {ch.label} | verbatim\n{ch.text}",
                        usable=usable.name.lower(),
                    ),
                    parent_ids=ch.source_ids,
                    metadata={
                        "evidence": True,
                        "standing": str(ch.standing),
                        "usable_as": usable.name.lower(),
                        "raw_text": ch.text,
                    },
                )
            )
        return out

    def _cap(self, ordered: list[Claim], k: int, seen: dict[str, int]) -> list[Claim]:
        """Take the best `k`, allowing at most `source_cap` from any one artifact."""
        if not self.source_cap:
            return ordered[:k]
        out: list[Claim] = []
        for c in ordered:
            key = c.source_id or ""
            if seen.get(key, 0) >= self.source_cap:
                continue
            seen[key] = seen.get(key, 0) + 1
            out.append(c)
            if len(out) >= k:
                break
        return out

    def _resolve_conflicts(self, graph: ClaimGraph, served: list[Claim]) -> list[ConflictVerdict]:
        """Downgrade, never adjudicate."""
        groups: dict[tuple, list[Claim]] = {}
        for c in served:
            if c.relation and c.subject_id:
                groups.setdefault(c.predicate, []).append(c)

        out: list[ConflictVerdict] = []
        for predicate, members in groups.items():
            if len(members) < 2:
                continue
            opposed = [
                (a, b)
                for i, a in enumerate(members)
                for b in members[i + 1 :]
                if self._contradicts(a, b)
            ]
            if not opposed:
                continue
            top = max(c.standing.level for c in members)
            at_top = [c for c in members if c.standing.level == top]
            disputed_at_top = any(
                a.standing.level == top and b.standing.level == top for a, b in opposed
            )
            if disputed_at_top:
                for c in members:
                    c.contested = True
                out.append(
                    ConflictVerdict(
                        predicate=predicate,
                        contested=tuple(c.claim_id for c in members),
                        detail="sources disagree at the same standing; none is served at its kind",
                        sources=tuple(c.source_id or "" for c in members),
                    )
                )
            else:
                held = at_top[0]
                losers = [c for c in members if c.claim_id != held.claim_id]
                for c in losers:
                    c.contested = True
                out.append(
                    ConflictVerdict(
                        predicate=predicate,
                        contested=tuple(c.claim_id for c in losers),
                        held=held.claim_id,
                        detail=f"outranked by {held.standing}",
                        sources=tuple(c.source_id or "" for c in members),
                    )
                )
        return out

    def _contradicts(self, a: Claim, b: Claim) -> bool:
        if a.text.strip().lower() == b.text.strip().lower():
            return False
        rendered = self.prompts.get("detect_conflict", self.conflict_version).render(
            claim_a=a.text, claim_b=b.text
        )
        reply = self.client.complete(
            ChatRequest(
                messages=(Message("user", rendered),),
                model=self.model,
                temperature=0.0,
                max_tokens=6,
            )
        ).text
        return "contradict" in reply.strip().lower()

    def _asks_to_retain(self, turn: str) -> bool:
        if self.retention_version is None:
            return True
        reply = self.client.complete(
            ChatRequest(
                messages=(
                    Message(
                        "user",
                        self.prompts.get("retention_check", self.retention_version).render(
                            turn=turn.strip()
                        ),
                    ),
                ),
                model=self.model,
                temperature=self.temperature,
                max_tokens=5,
            )
        ).text
        return reply.strip().lower().startswith("yes")

    def _entry(
        self, claim: Claim, graph: ClaimGraph, *, required: Rung | None = None
    ) -> MemoryEntry:
        """A typed record in a MemLineage-style sentinel (spec §2.0a, §6.2)."""
        if self.mode == "off":
            return MemoryEntry(
                entry_id=claim.claim_id,
                text=claim.text,
                parent_ids=(claim.source_id,) if claim.source_id else (),
                metadata={"standing": str(claim.standing), "raw_text": claim.text},
            )
        src = graph.source(claim.source_id) if claim.source_id else None
        marks = []
        if src:
            marks.append(f"source: {src.artifact}")
        if claim.subject_id:
            marks.append(f"about: {claim.subject_id}")
        if claim.contested:
            marks.append("contested")
        body = (" | ".join(marks) + "\n" if marks else "") + claim.text
        return MemoryEntry(
            entry_id=claim.claim_id,
            text=sentinel.render(claim.claim_id, body, usable=claim.usable.name.lower()),
            parent_ids=(claim.source_id,) if claim.source_id else (),
            metadata={
                "claim_type": claim.claim_type.name.lower(),
                "standing": str(claim.standing),
                "usable_as": claim.usable.name.lower(),
                "required": required.name.lower() if required else "",
                "contested": claim.contested,
                "source_id": claim.source_id or "",
                "subject": claim.subject_id or "",
                "raw_text": claim.text,
            },
        )

    def graph_stats(self) -> dict:
        """`about`-edge density, pooled over stores (spec §7.9, prediction 3):
        high on LoCoMo, near zero on MPBench. A dense MPBench graph would mean
        the extractor is inventing relations."""
        n = linked = objects = 0
        entities = set()
        for g in self._graphs.values():
            for c in g.claims():
                n += 1
                objects += bool(c.object_id)
                linked += bool(g.neighbours(c))
                entities.update(e for e in (c.subject_id, c.object_id) if e)
        return {
            "n_stores": len(self._graphs),
            "n_claims": n,
            "n_entities": len(entities),
            "share_with_object": round(objects / n, 4) if n else 0.0,
            "share_with_a_neighbour": round(linked / n, 4) if n else 0.0,
            "claims_per_entity": round(n / len(entities), 3) if entities else 0.0,
            "n_elevations": self.n_elevations,
            "n_elevation_rejected_quote": self.n_elevation_rejected,
            "n_elevation_not_requested": self.n_elevation_not_requested,
            "n_principal_pass_claims": self.n_principal_pass_claims,
            "n_referential_elevations": self.n_referential_elevations,
            "n_evidence_chunks": sum(len(v) for v in self._chunks.values()),
            "index": self.index,
            "n_entity_merges": self.n_entity_merges,
            "ann_recall_at_k": (
                round(self.index_overlap / self.n_index_checks, 4) if self.n_index_checks else None
            ),
        }

    def reset(self, session_id: str | None = None) -> None:
        stores = (
            self._graphs,
            self._vectors,
            self._chunks,
            self._chunk_vecs,
            self._parts,
            self._chunk_parts,
        )
        for store in stores:
            if session_id is None:
                store.clear()
            else:
                store.pop(session_id, None)


def classify_retention_sentences(
    client: LLMClient,
    model: str,
    prompts: PromptLibrary,
    turn: str,
    *,
    version: str = "v1",
) -> list[dict]:
    """Per-sentence retention reading of a principal turn (spec §10.3, step 1)."""
    from ams.core.spans import segment

    sents = [s.text for s in segment(turn)]
    if not sents:
        return []
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(sents))
    reply = client.complete(
        ChatRequest(
            messages=(
                Message(
                    "user",
                    prompts.get("retention_sentence", version).render(sentences=numbered),
                ),
            ),
            model=model,
            temperature=0.0,
            max_tokens=60 + 40 * len(sents),
        )
    ).text
    labels: dict[int, dict] = {}
    m = re.search(r"\{.*\}", reply, re.S)
    try:
        for d in json.loads(m.group(0)).get("sentences", []) if m else []:
            if isinstance(d, dict) and isinstance(d.get("i"), int):
                labels[d["i"]] = d
    except (json.JSONDecodeError, AttributeError):
        labels = {}
    out = []
    for i, t in enumerate(sents):
        d = labels.get(i, {})
        keep = d.get("keep") is True
        scope = d.get("scope") if keep and d.get("scope") in ("self", "referential") else ""
        out.append(
            {"text": t, "keep": keep and bool(scope), "scope": scope, "kind": d.get("kind") or ""}
        )
    return out
