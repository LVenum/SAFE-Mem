"""Where a span sits -> what its source may establish. Reads no content."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from ams.core.lattice import (
    GLOBAL,
    UNKNOWN_STANDING,
    Domain,
    Rung,
    Standing,
    meet_all,
    person,
    principal,
    tool,
)


class SpanRole(StrEnum):
    """What kind of place a span sits in. Established by the harness, never parsed
    out of the text being classified."""

    PRINCIPAL_TURN = "principal_turn"
    AUTHENTICATED_PRINCIPAL = "authenticated_principal"
    EXTERNAL_BODY = "external_body"
    TOOL_OWN_DOMAIN = "tool_own_domain"
    TOOL_RELAY = "tool_relay"
    COUNTERPART_TURN = "counterpart_turn"
    AGENT_DERIVED = "agent_derived"
    UNKNOWN = "unknown"


def standing_for(
    role: SpanRole,
    *,
    principal_id: str | None = None,
    tool_domain: str | None = None,
) -> Standing:
    """The standing a span in this role confers."""
    match role:
        case SpanRole.PRINCIPAL_TURN | SpanRole.AUTHENTICATED_PRINCIPAL:
            if not principal_id:
                raise ValueError(f"{role} needs principal_id to scope the standing")
            return Standing(Rung.PERMISSION_TRUST, principal(principal_id))
        case SpanRole.TOOL_OWN_DOMAIN:
            if not tool_domain:
                raise ValueError("tool_own_domain needs tool_domain")
            return Standing(Rung.ATTRIBUTE, tool(tool_domain))
        case SpanRole.COUNTERPART_TURN:
            if not principal_id:
                raise ValueError("counterpart_turn needs the speaker's id")
            return Standing(Rung.ATTRIBUTE, person(principal_id))
        case SpanRole.EXTERNAL_BODY | SpanRole.TOOL_RELAY:
            return Standing(Rung.DESCRIPTION, GLOBAL)
        case SpanRole.AGENT_DERIVED | SpanRole.UNKNOWN:
            return UNKNOWN_STANDING
    raise ValueError(f"unhandled span role {role!r}")  # pragma: no cover


def derived_standing(context: Iterable[Standing]) -> Standing:
    """Standing for something the agent wrote, from everything that was in front
    of it.
    """
    return meet_all(list(context))


def elevation_quote_is_genuine(quote: str, principal_region: str) -> bool:
    """Did the principal actually write this, verbatim, in their own turn?"""
    q = (quote or "").strip()
    if len(q) < 8:
        return False
    return q in (principal_region or "")


def describe(standing: Standing) -> str:
    """One-line rendering for logs and for the prompt's provenance header."""
    return str(standing)


__all__ = [
    "SpanRole",
    "standing_for",
    "derived_standing",
    "elevation_quote_is_genuine",
    "describe",
    "Domain",
]
