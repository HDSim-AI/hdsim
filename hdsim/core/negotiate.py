"""Proposals and negotiation.

Two steps:

    propose(household)    every member answers alone, in parallel
    negotiate(household)  the members discuss until they agree on one value

Members propose in parallel rather than in turn. Asking them one after another lets later members
anchor on earlier ones, which produces agreement that comes from ordering rather than from the
household.

The value of the discussion is correction. A member who knows something the others do not says it,
and the household total moves. That only works if the members hold different information, which is
why the roster tells each member who the others are without telling them what those people plan.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from .backends import Client, get_client
from .domain import DomainConfig
from .household import Household, Member
from .prompts import (NEGOTIATION_SYSTEM, NEGOTIATION_USER,
                      PROPOSAL_SYSTEM, PROPOSAL_USER)


_TRUE = {"true", "yes", "will move", "move", "relocate"}
_FALSE = {"false", "no", "will not move", "stay", "remain"}


def parse_value(text: str, config: DomainConfig) -> float | bool | None:
    """Read the decided value out of a response.

    Falls back to the last number in the text, because models sometimes drop the label while
    still answering. A response with no answer at all returns None and is counted as a
    non-answer rather than silently treated as zero, which would otherwise turn a parsing failure
    into a confident prediction.
    """
    if not text:
        return None
    label = re.escape(config.task.final_label)

    # Not every decision is a quantity. Residential mobility answers whether the household moves.
    #
    # Unlike a number, the label is required here and there is no fallback to scanning the text.
    # Words like "no", "stay" and "move" occur constantly in ordinary prose, so a loose search
    # turns "no answer given" into a confident False. A missing label means the model did not
    # answer, and that is reported as a non-answer rather than guessed at.
    if config.task.value_type is bool:
        m = re.search(rf"{label}\s*[:=]\s*(.+)", text, re.IGNORECASE)
        if not m:
            return None
        candidate = m.group(1).strip().lower()
        # Longest first, so "will not move" is not read as "move".
        for word in sorted(_TRUE | _FALSE, key=len, reverse=True):
            if re.search(rf"\b{re.escape(word)}\b", candidate):
                return word in _TRUE
        return None

    m = re.search(rf"{label}\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if not m:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not numbers:
            return None
        value = float(numbers[-1])
    else:
        value = float(m.group(1))
    low, high = config.task.value_range
    value = max(low, min(high, value))
    return config.task.value_type(value)


def propose_one(member: Member, config: DomainConfig, client: Client) -> Member:
    """One member's independent answer."""
    task = config.task
    system = PROPOSAL_SYSTEM.format(
        domain=task.domain, context=task.context,
        target_description=task.target_description,
        final_label=task.final_label, value_type=task.value_type.__name__,
    )
    user = PROPOSAL_USER.format(
        persona=member.persona, unit=task.unit, final_label=task.final_label,
    )
    text = client.chat([{"role": "system", "content": system},
                        {"role": "user", "content": user}])
    member.proposal_text = text
    member.proposal_value = parse_value(text, config)
    return member


def propose(household: Household, config: DomainConfig,
            client: Client | None = None, max_workers: int = 4) -> Household:
    """Every member proposes, without seeing anyone else's answer."""
    client = client or get_client("member")
    if len(household) == 1 or max_workers == 1:
        for member in household:
            propose_one(member, config, client)
        return household
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda m: propose_one(m, config, client), household.members))
    return household


def _positions(household: Household, config: DomainConfig) -> str:
    """Render each member's opening position for the moderator.

    The label is what the moderator will use as a speaker name, so a loader that sets it from the
    relationship code produces a transcript reading "Mother:" instead of "Member 2:".
    """
    blocks = []
    for member in household:
        value = member.proposal_value
        stated = f"{value} {config.task.unit}" if value is not None else "no clear position"
        name = member.label or f"Member {member.person_id}"
        blocks.append(
            f"{name}\n"
            f"Profile: {member.capsule}\n"
            f"Opening position: {stated}\n"
            f"Reasoning: {member.proposal_text.strip()}"
        )
    return "\n\n".join(blocks)


def negotiate(household: Household, config: DomainConfig,
              client: Client | None = None, max_tokens: int = 1500) -> Household:
    """Run the discussion and record the agreed value.

    Members must have proposed first. The moderator is a separate role so a stronger model can be
    used here than for the member turns.
    """
    if any(m.proposal_value is None and not m.proposal_text for m in household):
        raise RuntimeError("propose() must run before negotiate()")

    client = client or get_client("moderator")
    task = config.task
    text = client.chat([
        {"role": "system", "content": NEGOTIATION_SYSTEM.format(
            target_description=task.target_description, context=task.context,
            final_label=task.final_label, value_type=task.value_type.__name__)},
        {"role": "user", "content": NEGOTIATION_USER.format(
            positions=_positions(household, config), final_label=task.final_label)},
    ], max_tokens=max_tokens)

    household.transcript = parse_transcript(text)
    household.rounds = max((t.get("round", 1) for t in household.transcript), default=1)
    household.consensus_value = parse_value(text, config)
    return household


def parse_transcript(text: str) -> list[dict]:
    """Split a discussion into speaker turns, keeping round numbers where they are marked."""
    turns, current_round = [], 1
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\**\s*Round\s+(\d+)", line, re.IGNORECASE)
        if m:
            current_round = int(m.group(1))
            continue
        # Models emit speaker labels in several markdown shapes: "Name:", "- Name:",
        # "**Name:**", "- **Name**:". The colon may fall inside or outside the bold markers, so
        # emphasis is allowed on both sides and stripped from the text that follows.
        m = re.match(r"^[-*]?\s*\**\s*([A-Z][A-Za-z0-9 ']{0,30}?)\s*\**\s*:\s*(.+)$", line)
        if m and not m.group(1).upper().startswith("FINAL"):
            # Emphasis markers land on either side of the colon depending on the model, and
            # sometimes on both, separated by spaces. Strip any run of asterisks and whitespace
            # from each end rather than trying to enumerate markdown shapes in the pattern.
            text = re.sub(r"^[\s*]+|[\s*]+$", "", m.group(2))
            if text:
                turns.append({"round": current_round,
                              "speaker": re.sub(r"^[\s*]+|[\s*]+$", "", m.group(1)),
                              "text": text})
    return turns


def simulate(household: Household, config: DomainConfig, *,
             member_client: Client | None = None,
             moderator_client: Client | None = None) -> Household:
    """Propose, then negotiate. Personas must already be built."""
    propose(household, config, member_client)
    negotiate(household, config, moderator_client)
    return household
