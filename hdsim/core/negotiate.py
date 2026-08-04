"""Proposals and household negotiation.

    propose(household)    every member answers alone, in parallel
    negotiate(household)  the household discusses and converges on one total

The prompts, labels and parsers all come from `stage2`, which is the PEMAND implementation carried
over unchanged. Nothing here rewrites them. This module's job is to move `Household` objects in and
out of the shapes those functions expect, and to run the member calls concurrently.

Members propose in parallel rather than in turn. Asking them one after another lets later members
anchor on earlier ones, which produces agreement that comes from ordering rather than from the
household.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import stage2
from .backends import Client, get_client
from .domain import DomainConfig
from .household import Household, Member


def _task_dict(config: DomainConfig) -> dict[str, str]:
    """The domain's decision, in the shape the stage-two prompt builders take."""
    task = config.task
    return {
        "domain": task.domain,
        "context": task.context,
        "target_description": task.target_description,
        "unit": task.unit,
        "final_label": task.final_label,
    }


def _member_dict(member: Member) -> dict[str, Any]:
    """A member, in the shape the stage-two prompt builders take."""
    return {
        "PERSONID": member.person_id,
        "persona": member.persona,
        "tpb": member.tpb,
    }


def parse_value(text: str, config: DomainConfig) -> float | bool | None:
    """Read a decided value out of a response.

    Numbers go through the stage-two extractor. Booleans are handled here because residential
    mobility answers yes or no, and unlike a number a bare "no" appears constantly in ordinary
    prose. The label is therefore required for booleans, and a response without one is reported as
    a non-answer rather than guessed at.
    """
    if not text:
        return None

    if config.task.value_type is bool:
        import re
        m = re.search(rf"{re.escape(config.task.final_label)}\s*[:=]\s*(.+)", text, re.IGNORECASE)
        if not m:
            return None
        candidate = m.group(1).strip().lower()
        true_words = {"true", "yes", "will move", "move", "relocate"}
        false_words = {"false", "no", "will not move", "stay", "remain"}
        for word in sorted(true_words | false_words, key=len, reverse=True):
            if re.search(rf"\b{re.escape(word)}\b", candidate):
                return word in true_words
        return None

    value = stage2.extract_decision_value(text, _task_dict(config))
    if value is None:
        return None
    low, high = config.task.value_range
    return config.task.value_type(max(low, min(high, value)))


def propose_one(member: Member, config: DomainConfig, client: Client) -> Member:
    """One member's independent answer, using the stage-two proposal prompts."""
    task = _task_dict(config)
    text = client.chat([
        {"role": "system", "content": stage2.build_system_prompt(task)},
        {"role": "user", "content": stage2.build_user_prompt(member.persona, member.tpb, task)},
    ])
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


def negotiate(household: Household, config: DomainConfig,
              client: Client | None = None, max_tokens: int = 1500) -> Household:
    """Run the household discussion and record the agreed total.

    Uses the stage-two consensus prompt, which distinguishes a member's own value from that
    member's view of the household figure from the agreed total, and states that the final number
    is the sum across all members.
    """
    client = client or get_client("moderator")
    members = [_member_dict(m) for m in household]

    text = client.chat([
        {"role": "system", "content": stage2.build_consensus_system(_task_dict(config))},
        {"role": "user", "content": stage2.build_household_prompt(members)},
    ], max_tokens=max_tokens)

    result = stage2.parse_completion_to_result(text, len(household))
    household.transcript = [{"round": 1, "speaker": "", "text": line}
                            for line in result["transcript"]]
    household.rounds = result["n_rounds"]
    value = result["final_value"]
    if value is not None and config.task.value_type is not bool:
        low, high = config.task.value_range
        value = config.task.value_type(max(low, min(high, value)))
    household.consensus_value = value
    return household


def simulate(household: Household, config: DomainConfig, *,
             member_client: Client | None = None,
             moderator_client: Client | None = None) -> Household:
    """Propose, then negotiate. Personas must already be built."""
    propose(household, config, member_client)
    negotiate(household, config, moderator_client)
    return household
