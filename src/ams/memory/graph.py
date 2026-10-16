"""The claim graph: three node kinds, three edge kinds, one lattice."""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from ams.core.lattice import Rung, Standing, usable_as
from ams.core.standing import SpanRole, derived_standing, standing_for

_NORM = re.compile(r"[^a-z0-9]+")


def canonical_entity_key(name: str) -> str:
    """Normalised-string identity."""
    return _NORM.sub(" ", (name or "").strip().lower()).strip()


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    artifact: str
    role: SpanRole
    text: str = ""
    principal_id: str | None = None
    tool_domain: str | None = None
    seq: int = 0

    def standing(self) -> Standing:
        return standing_for(self.role, principal_id=self.principal_id, tool_domain=self.tool_domain)


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    name: str


@dataclass(slots=True)
class Claim:
    claim_id: str
    text: str
    claim_type: Rung
    standing: Standing
    source_id: str | None = None
    derived_from: tuple[str, ...] = ()
    subject_id: str | None = None
    object_id: str | None = None
    relation: str = ""
    endorsed_by: tuple[str, ...] = ()
    contested: bool = False
    session_id: str = ""
    seq: int = 0

    @property
    def usable(self) -> Rung:
        """The strongest rung this claim may actually serve."""
        return usable_as(self.claim_type, self.standing)

    @property
    def predicate(self) -> tuple[str | None, str]:
        """What two claims must share before they can contradict each other."""
        return (self.subject_id, self.relation)


class ClaimGraph:
    """In-memory graph. One instance per store; `reset` clears it."""

    def __init__(self, *, coreference: bool = False) -> None:
        self.coreference = coreference
        self._claims: dict[str, Claim] = {}
        self._sources: dict[str, Source] = {}
        self._entities: dict[str, Entity] = {}
        self._by_entity: dict[str, list[str]] = {}
        self._by_source: dict[str, list[str]] = {}
        self._after: dict[str, str] = {}
        self._prev: dict[str, str] = {}
        self._ids = itertools.count()

    def _next(self, prefix: str) -> str:
        return f"{prefix}{next(self._ids)}"

    def add_source(self, source: Source) -> Source:
        self._sources[source.source_id] = source
        return source

    def link_next(self, a: str, b: str) -> None:
        """`a` is immediately followed by `b` (spec §10.1). Order only; no weight."""
        self._after[a] = b
        self._prev[b] = a

    def neighbours_in_order(self, source_id: str, width: int = 1) -> list[Source]:
        """`source_id` with up to `width` predecessors and successors, in order."""
        before, after, cur = [], [], source_id
        for _ in range(width):
            cur = self._prev.get(cur, "")
            if not cur:
                break
            before.insert(0, cur)
        cur = source_id
        for _ in range(width):
            cur = self._after.get(cur, "")
            if not cur:
                break
            after.append(cur)
        return [self._sources[i] for i in [*before, source_id, *after] if i in self._sources]

    def entity(self, name: str) -> Entity:
        key = canonical_entity_key(name)
        if key not in self._entities and self.coreference:
            from ams.memory.entities import resolve, tokens

            hit = resolve(name, {k: tokens(e.name) for k, e in self._entities.items()})
            if hit is not None:
                return self._entities[hit]
        if key not in self._entities:
            self._entities[key] = Entity(entity_id=key, name=name)
        return self._entities[key]

    def add_claim(self, claim: Claim) -> Claim:
        self._claims[claim.claim_id] = claim
        for ent in (claim.subject_id, claim.object_id):
            if ent:
                self._by_entity.setdefault(ent, []).append(claim.claim_id)
        if claim.source_id:
            self._by_source.setdefault(claim.source_id, []).append(claim.claim_id)
        return claim

    def new_claim_id(self) -> str:
        return self._next("c")

    def new_source_id(self) -> str:
        return self._next("s")

    def __len__(self) -> int:
        return len(self._claims)

    def claims(self) -> Iterator[Claim]:
        return iter(self._claims.values())

    def claim(self, claim_id: str) -> Claim:
        return self._claims[claim_id]

    def source(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)

    def claims_about(self, entity_id: str) -> list[Claim]:
        return [self._claims[c] for c in self._by_entity.get(entity_id, ())]

    def neighbours(self, claim: Claim) -> list[Claim]:
        """Claims reachable in one hop: those sharing either of this claim's
        entities. Subject and object are both followed, which is the point of
        giving `about` a role."""
        out: dict[str, Claim] = {}
        for ent in (claim.subject_id, claim.object_id):
            for other in self.claims_about(ent) if ent else ():
                if other.claim_id != claim.claim_id:
                    out[other.claim_id] = other
        return list(out.values())

    def _step(self, claim: Claim, per_entity: int | None, rank, keep) -> list[Claim]:
        """One hop. With `per_entity`, at most that many claims are followed through
        each shared entity, best by `rank` first: a hub entity ("user", a common
        name) otherwise pulls in every claim that mentions it (spec §11)."""
        if not per_entity:
            return self.neighbours(claim)
        out: dict[str, Claim] = {}
        for ent in (claim.subject_id, claim.object_id):
            if not ent:
                continue
            others = [o for o in self.claims_about(ent) if o.claim_id != claim.claim_id and keep(o)]
            if rank is not None:
                others.sort(key=rank, reverse=True)
            for o in others[:per_entity]:
                out[o.claim_id] = o
        return list(out.values())

    def expand(
        self, seeds: list[Claim], *, hops: int, keep, per_entity: int | None = None, rank=None
    ) -> list[Claim]:
        """Breadth-first traversal from `seeds`, `hops` deep, admitting only
        claims `keep(claim)` accepts.
        """
        if hops <= 0:
            return []
        seen = {c.claim_id for c in seeds}
        frontier, out = list(seeds), []
        for _ in range(hops):
            nxt = []
            for claim in frontier:
                for other in self._step(claim, per_entity, rank, keep):
                    if other.claim_id in seen or not keep(other):
                        continue
                    seen.add(other.claim_id)
                    out.append(other)
                    nxt.append(other)
            frontier = nxt
            if not frontier:
                break
        return out

    def derive(self, *, context_claims: list[Claim]) -> Standing:
        """Standing for something written with these claims in front of the agent."""
        return derived_standing([c.standing for c in context_claims])

    def elevate(self, claim_id: str, *, source: Source) -> Claim:
        """Record a principal's endorsement of an existing claim."""
        claim = self._claims[claim_id]
        es = source.standing()
        raised = Standing(max(claim.standing.level, es.level), es.domain)
        self._claims[claim_id] = replace(
            claim, standing=raised, endorsed_by=(*claim.endorsed_by, source.source_id)
        )
        return self._claims[claim_id]

    def demote_source(self, source_id: str) -> list[str]:
        """A source turned out to be poisoned: walk `derives_from` forward from it
        and recompute standing over everything downstream.
        """
        changed: list[str] = []
        frontier = list(self._by_source.get(source_id, ()))
        floor = Standing(Rung.DESCRIPTION, self._sources[source_id].standing().domain)
        seen: set[str] = set()
        while frontier:
            cid = frontier.pop()
            if cid in seen:
                continue
            seen.add(cid)
            claim = self._claims[cid]
            new = claim.standing.meet(floor)
            if new != claim.standing:
                self._claims[cid] = replace(claim, standing=new)
                changed.append(cid)
            frontier.extend(
                other.claim_id for other in self._claims.values() if cid in other.derived_from
            )
        return changed

    def reset(self) -> None:
        self._claims.clear()
        self._sources.clear()
        self._entities.clear()
        self._by_entity.clear()
        self._by_source.clear()


@dataclass
class ConflictVerdict:
    """What §6.3 decided about one predicate on one entity."""

    predicate: tuple[str | None, str]
    contested: tuple[str, ...] = ()
    held: str | None = None
    detail: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)
