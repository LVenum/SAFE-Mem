"""How provenance gets attached to data that does not come with any."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Granularity(StrEnum):
    """The unit a channel label may attach to."""

    SESSION = "session"
    DOCUMENT = "document"
    TURN = "turn"
    TOOL_CALL = "tool_call"
    SPAN = "span"


@dataclass(frozen=True, slots=True)
class ProvenancePolicy:
    """A documented, auditable rule for imposing provenance on a corpus."""

    name: str
    granularity: Granularity
    rule: str
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "granularity": str(self.granularity),
            "rule": self.rule,
            "assumptions": list(self.assumptions),
        }


SYNTHETIC_POLICY = ProvenancePolicy(
    name="native",
    granularity=Granularity.DOCUMENT,
    rule=(
        "Each source is constructed by the harness with a known channel: vendor and "
        "policy records arrive on the user channel, the invoice record on a first-party "
        "tool channel, and the third-party document on the web channel. A derived memory "
        "inherits the channels of the sources it was consolidated from."
    ),
    assumptions=(
        "Every case retrieves exactly one third-party document, clean or poisoned, so "
        "the presence of an untrusted parent carries no information about the label.",
    ),
)
