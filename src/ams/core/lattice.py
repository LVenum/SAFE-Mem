"""The two orders the gate compares, and the meet that propagates them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Rung(IntEnum):
    """What kind of claim. The discriminator is *who must comply with it*."""

    DESCRIPTION = 0
    ATTRIBUTE = 1
    PREFERENCE = 2
    RULE = 3
    PERMISSION_TRUST = 4

    @classmethod
    def parse(cls, text: str) -> Rung:
        """Accept the names a model emits; refuse anything else."""
        key = (text or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "description": cls.DESCRIPTION,
            "fact": cls.DESCRIPTION,
            "attribute": cls.ATTRIBUTE,
            "preference": cls.PREFERENCE,
            "rule": cls.RULE,
            "policy": cls.RULE,
            "permission": cls.PERMISSION_TRUST,
            "trust": cls.PERMISSION_TRUST,
            "permission_trust": cls.PERMISSION_TRUST,
        }
        if key not in aliases:
            raise ValueError(f"not a claim type: {text!r}")
        return aliases[key]


ClaimType = Rung


@dataclass(frozen=True, slots=True)
class Domain:
    """Where a standing holds."""

    kind: str
    name: str = ""

    def __str__(self) -> str:
        return self.kind if not self.name else f"{self.kind}:{self.name}"

    def covers(self, other: Domain) -> bool:
        """Can a standing held in `self` answer a question asked in `other`?"""
        if self.kind == "nowhere" or other.kind == "nowhere":
            return False
        if self.kind == "global":
            return True
        return self == other

    def meet(self, other: Domain) -> Domain:
        """The most permissive domain no more permissive than either."""
        if self == other:
            return self
        if self.kind == "global":
            return other
        if other.kind == "global":
            return self
        return NOWHERE


GLOBAL = Domain("global")
NOWHERE = Domain("nowhere")


def principal(name: str) -> Domain:
    return Domain("principal", name)


def tool(name: str) -> Domain:
    return Domain("tool", name)


def person(name: str) -> Domain:
    """A non-principal party speaking about themselves (spec §10.2)."""
    return Domain("person", (name or "").strip().lower())


@dataclass(frozen=True, slots=True)
class Standing:
    """The strongest claim a source is entitled to establish, and where."""

    level: Rung
    domain: Domain = GLOBAL

    def __str__(self) -> str:
        return f"{self.level.name.lower()}@{self.domain}"

    def meet(self, other: Standing) -> Standing:
        """Greatest lower bound: the weakest of the levels, in the narrower domain."""
        return Standing(min(self.level, other.level), self.domain.meet(other.domain))


UNKNOWN_STANDING = Standing(Rung.DESCRIPTION, GLOBAL)


def meet_all(standings: list[Standing]) -> Standing:
    """Meet over a whole write-time context."""
    if not standings:
        return UNKNOWN_STANDING
    out = standings[0]
    for s in standings[1:]:
        out = out.meet(s)
    return out


def usable_as(claim_type: Rung, standing: Standing) -> Rung:
    """The strongest rung this entry may actually be used at."""
    return Rung(min(int(claim_type), int(standing.level)))


def coarsen(rung: Rung, granularity: int) -> Rung:
    """Collapse the ladder to `granularity` levels."""
    if granularity >= len(Rung):
        return rung
    if granularity == 2:
        return Rung.DESCRIPTION if rung == Rung.DESCRIPTION else Rung.PERMISSION_TRUST
    raise ValueError(f"granularity must be 2 or {len(Rung)}, got {granularity}")


def permits(claim_type: Rung, standing: Standing, *, required: Rung, asked_in: Domain) -> bool:
    """May an entry with this type and standing serve a use requiring `required`?"""
    if usable_as(claim_type, standing) < required:
        return False
    return required == Rung.DESCRIPTION or standing.domain.covers(asked_in)
