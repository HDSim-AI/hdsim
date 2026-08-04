"""Persona construction.

Two steps, in order:

    facts   -> capsule   a survey record becomes a first-person paragraph
    capsule -> tpb       the paragraph gains Attitude, Subjective Norm and Perceived Control

Both are checked before they are accepted. A capsule that drops facts, invents circumstances or
mentions the quantity the household is about to decide is regenerated rather than used, because
every one of those failures shows up later as a fabricated justification during negotiation.

In the research code this pipeline exists three times, once per dataset. Here it exists once and
takes a `DomainConfig`.
"""

from __future__ import annotations

import re

from .backends import Client, get_client
from .domain import DomainConfig
from .household import Household, Member
from .prompts import CAPSULE_SYSTEM, CAPSULE_USER, TPB_SYSTEM, TPB_USER

# A capsule is too short when it cannot plausibly contain the facts it was given. That depends on
# how many facts there were, so a fixed word count is wrong: a household described by seven short
# fields produces a correct capsule of about thirty words, while one described by twenty fields
# should produce far more. Scale the floor with the input and keep a small absolute minimum.
MIN_WORDS_FLOOR = 5
WORDS_PER_FACT = 3
MAX_ATTEMPTS = 3

_TPB_SECTIONS = ("Attitude", "Subjective Norm", "Perceived Behavioral Control")


class PersonaError(RuntimeError):
    """Raised when a persona cannot be produced within the attempt budget."""


# --- capsule --------------------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip preambles the model adds despite instructions."""
    text = text.strip()
    text = re.sub(r"^(Persona|Here is.*?|Sure.*?)[:\s]*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```.*?\n|```$", "", text, flags=re.DOTALL).strip()
    return text


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def check_capsule(text: str, facts: list[str], config: DomainConfig) -> str | None:
    """Return the reason a capsule is unusable, or None if it passes.

    Three failures matter, in order of how much damage they do downstream:

    Dropped facts. The prompt says to include every one. A capsule that quietly loses "my household
    owns 2 vehicles" produces an agent that later invents its own vehicle access. Numbers are the
    part most often dropped and the easiest to verify, so they are checked directly rather than
    inferred from length.

    Leaked outcome. Persona text is written before the household decides.

    Length. Only a weak backstop now that coverage is checked properly.
    """
    words = len(text.split())
    floor = max(MIN_WORDS_FLOOR, WORDS_PER_FACT * len(facts))
    if words < floor:
        return f"too short ({words} words for {len(facts)} facts, expected at least {floor})"

    missing = _numbers(" ".join(facts)) - _numbers(text)
    if missing:
        return f"dropped facts containing {sorted(missing)}"

    leak = config.leaks_outcome(text)
    if leak:
        return f"mentions the outcome being decided ({leak!r})"

    if re.search(r"\bFinal Answer\b", text, re.IGNORECASE):
        return "contains 'Final Answer'"
    return None


def build_capsule(member: Member, config: DomainConfig,
                  client: Client | None = None) -> Member:
    """Fill in `member.facts` and `member.capsule`."""
    client = client or get_client("persona")
    # A loader may have written richer facts already. Some surveys describe a person partly through
    # household-level fields that only make sense once the whole household is in hand, and the
    # loader is the only place that sees the group. Recomputing here would silently discard them.
    if not member.facts:
        member.facts = config.facts_for(member.record)
    if not member.facts:
        raise PersonaError(f"member {member.person_id}: no facts could be rendered from the record")

    banned_subject = ", ".join(f"'{p}'" for p in config.banned_patterns) or "the decision outcome"
    system = CAPSULE_SYSTEM.format(banned_subject=banned_subject)
    user = CAPSULE_USER.format(facts="\n".join(f"- {f}" for f in member.facts))

    problems = []
    for attempt in range(MAX_ATTEMPTS):
        text = _clean(client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0 if attempt == 0 else 0.4,
        ))
        problem = check_capsule(text, member.facts, config)
        if problem is None:
            member.capsule = text
            return member
        problems.append(problem)

    raise PersonaError(
        f"member {member.person_id}: no valid capsule after {MAX_ATTEMPTS} attempts "
        f"({'; '.join(problems)})"
    )


# --- Theory of Planned Behavior constructs ---------------------------------------------------------

def parse_tpb(text: str) -> dict[str, str]:
    """Pull the three labelled sections out of a response."""
    out: dict[str, str] = {}
    for i, name in enumerate(_TPB_SECTIONS):
        nxt = _TPB_SECTIONS[i + 1] if i + 1 < len(_TPB_SECTIONS) else None
        pattern = rf"{name}\s*:\s*(.*?)(?=\n\s*{nxt}\s*:|$)" if nxt else rf"{name}\s*:\s*(.*?)$"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            key = name.lower().replace(" ", "_")
            out[key] = " ".join(m.group(1).split())
    return out


def enrich(member: Member, config: DomainConfig, client: Client | None = None) -> Member:
    """Add the three TPB constructs to a member that already has a capsule.

    The roster is included in the prompt on purpose. Subjective Norm is defined as what the other
    household members expect of this person, so an agent that has not been told who they are cannot
    answer the question it is being asked.
    """
    if not member.capsule:
        raise PersonaError(f"member {member.person_id}: build_capsule must run first")
    client = client or get_client("persona")

    anchor = config.anchor(member.record)
    anchor_note = (
        f"\n4. For reference, people with similar characteristics average {anchor:.2f} "
        f"{config.task.unit}. Treat this as a starting expectation, not a rule."
        if anchor is not None else ""
    )
    roster_block = f"\n{member.roster}\n" if member.roster else "\n"
    marker_block = ""
    if config.markers:
        marker = config.markers[hash(str(member.person_id)) % len(config.markers)]
        marker_block = (
            f"\nBehavioral tendency inferred from similar profiles: "
            f"\"{marker.get('statement', '')}\"\n"
            f"This suggests: {marker.get('implication', '')}\n"
            f"Check this against the persona facts. The facts take precedence.\n"
        )

    text = client.chat([
        {"role": "system", "content": TPB_SYSTEM.format(
            domain=config.task.domain, anchor_note=anchor_note)},
        {"role": "user", "content": TPB_USER.format(
            persona=member.capsule, roster_block=roster_block, marker_block=marker_block,
            domain=config.task.domain,
            target_description=config.task.target_description)},
    ], temperature=0.3)

    tpb = parse_tpb(text)
    missing = [s for s in _TPB_SECTIONS if s.lower().replace(" ", "_") not in tpb]
    if missing:
        raise PersonaError(f"member {member.person_id}: TPB response missing {missing}")
    member.tpb = tpb
    return member


# --- household level --------------------------------------------------------------------------------

def build_personas(household: Household, config: DomainConfig,
                   client: Client | None = None) -> Household:
    """Build every member's persona, then give each of them the roster.

    Order matters. The roster is written before enrichment so Subjective Norm can refer to it.
    """
    client = client or get_client("persona")

    for member in household:
        build_capsule(member, config, client)

    if config.describe_member and config.relate_members:
        household.build_roster(config.describe_member, config.relate_members)

    for member in household:
        enrich(member, config, client)

    return household
