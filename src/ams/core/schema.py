"""The data vocabulary shared by every experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Channel(StrEnum):
    USER = "user"
    TOOL = "tool"
    WEB = "web"
    FILE = "file"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class TrustPolicy:
    """Which channels a deployment treats as first-party."""

    trusted: frozenset[Channel] = frozenset({Channel.USER, Channel.TOOL})

    def is_trusted(self, channel: Channel) -> bool:
        return channel in self.trusted


DEFAULT_TRUST = TrustPolicy()


@dataclass(frozen=True, slots=True)
class Source:
    """One ingested item: the unit whose channel is known and whose text an
    adversary may or may not control."""

    sid: str
    channel: Channel
    text: str
    trusted: bool = True

    @classmethod
    def make(
        cls, sid: str, channel: Channel, text: str, policy: TrustPolicy = DEFAULT_TRUST
    ) -> Source:
        return cls(sid=sid, channel=channel, text=text, trusted=policy.is_trusted(channel))


@dataclass(frozen=True, slots=True)
class Variant:
    """One realisation of the attacker-controlled source."""

    vid: str
    text: str
    goal: str
    family: str
    strategy: str = ""
    is_benign: bool = False
    generator: str = "unknown"


@dataclass(frozen=True, slots=True)
class AttackerGoal:
    """An outcome the adversary is trying to produce."""

    gid: str
    description: str
    target_action: str


@dataclass(frozen=True, slots=True)
class Span:
    """A unit of a derived memory. Sentence-level by default."""

    index: int
    text: str


@dataclass(frozen=True, slots=True)
class DerivedMemory:
    """The output of one consolidation, with its parent set recorded."""

    text: str
    parent_sids: tuple[str, ...]
    spans: tuple[Span, ...] = ()


@dataclass(frozen=True, slots=True)
class SpanAnnotation:
    """One (span, source) edge of the annotation N(m) = {(m_j, s_i, û_ij)}."""

    span_index: int
    source_sid: str
    upper_bound: float
    fallback: bool = False


@dataclass(frozen=True, slots=True)
class Annotation:
    memory_id: str
    edges: tuple[SpanAnnotation, ...] = ()

    def controllable_spans(self, threshold: float) -> frozenset[int]:
        return frozenset(
            e.span_index for e in self.edges if e.upper_bound >= threshold or e.fallback
        )


@dataclass(frozen=True, slots=True)
class Instance:
    """One consolidation scenario: trusted sources, one attacker slot, a query."""

    iid: str
    trusted_sources: tuple[Source, ...]
    attacker_slot: Source
    query: str
    metadata: dict[str, str] = field(default_factory=dict)

    def sources_with(self, variant_text: str | None) -> tuple[Source, ...]:
        """All sources, with the attacker slot set to `variant_text`."""
        if variant_text is None:
            return self.trusted_sources
        attacker = Source(
            sid=self.attacker_slot.sid,
            channel=self.attacker_slot.channel,
            text=variant_text,
            trusted=self.attacker_slot.trusted,
        )
        return (*self.trusted_sources, attacker)
