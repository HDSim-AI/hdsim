"""Persona construction.

    facts   -> capsule   a survey record becomes a first-person paragraph
    capsule -> tpb       the paragraph gains Attitude, Subjective Norm and Perceived Control

The prompts, the text cleanup and the section parser all come from `stage1`, which is the PEMAND
implementation carried over unchanged. Nothing here rewrites them. This module moves `Household`
and `Member` objects in and out of the shapes those functions expect, and enforces the retry
policy when a generation fails validation.

In the research code this pipeline exists three times, once per dataset. Here it exists once and
takes a `DomainConfig`.
"""

from __future__ import annotations

from . import stage1
from .backends import Client, get_client
from .domain import DomainConfig
from .household import Household, Member

MIN_WORDS_FLOOR = 5
WORDS_PER_FACT = 3
MAX_ATTEMPTS = 3


class PersonaError(RuntimeError):
    """Raised when a persona cannot be produced within the attempt budget."""


# --- capsule --------------------------------------------------------------------------------------

_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}


def _numbers(text: str) -> set[str]:
    """The numeric content of a passage, independent of how it is written.

    A fact and a faithful rendering of it can spell the same number differently, and neither
    difference means anything was dropped:

        "17000"      written back as "17,000"
        "2 vehicles" written back as "two vehicles"

    Comparing raw digit runs treats both as losses. The first is worse than it looks, because a
    naive match reads "17,000" as the two numbers 17 and 000 and then reports the original as
    missing, which is how a correct persona gets rejected.
    """
    import re as _re

    normalized = _re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    found = set(_re.findall(r"\d+", normalized))
    for word, digit in _WORD_NUMBERS.items():
        if _re.search(rf"\b{word}\b", normalized):
            found.add(digit)
    return {n.lstrip("0") or "0" for n in found}


def check_capsule(text: str, facts: list[str], config: DomainConfig) -> str | None:
    """Return the reason a capsule is unusable, or None if it passes.

    Runs the published validator first, then two additional checks. The length floor scales with
    the number of facts, because a household described by seven short fields produces a correct
    capsule of about thirty words while one described by twenty fields should produce far more.
    The numeric check catches dropped facts directly: a capsule that quietly loses "my household
    owns 2 vehicles" produces an agent that later invents its own vehicle access.
    """
    import re

    ok, _ = stage1.validate_and_clean_persona(text, min_words=1)
    if not ok:
        return "failed the published validator (banned phrase, or no usable text)"

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
    return None


def build_capsule(member: Member, config: DomainConfig,
                  client: Client | None = None) -> Member:
    """Fill in `member.facts` and `member.capsule`."""
    client = client or get_client("persona")
    # A loader may have written richer facts already. Some surveys describe a person partly through
    # household-level fields that only make sense once the whole household is in hand, and the
    # loader is the only place that sees the group.
    if not member.facts:
        member.facts = config.facts_for(member.record)
    if not member.facts:
        raise PersonaError(f"member {member.person_id}: no facts could be rendered from the record")

    problems = []
    for attempt in range(MAX_ATTEMPTS):
        # stage1.build_prompt composes these same two messages and then flattens them through a
        # local tokenizer's chat template, which an HTTP client neither needs nor can use. The
        # message content is taken from stage1 unchanged; only the flattening is skipped.
        bullet_block = "\n".join(f"- {fact}" for fact in member.facts)
        raw = client.chat(
            [{"role": "system", "content": stage1.SYSTEM_PROMPT},
             {"role": "user", "content": stage1.USER_TEMPLATE.format(facts=bullet_block)}],
            temperature=0.0 if attempt == 0 else 0.4,
        )
        text = stage1.clean_persona_text(raw)
        problem = check_capsule(text, member.facts, config)
        if problem is None:
            member.capsule = text
            return member
        problems.append(problem)

    raise PersonaError(
        f"member {member.person_id}: no valid capsule after {MAX_ATTEMPTS} attempts "
        f"({'; '.join(problems)})"
    )


# --- Chain-of-Planned-Behaviour enrichment ----------------------------------------------------------

def enrich(member: Member, config: DomainConfig, client: Client | None = None) -> Member:
    """Add the three behavioural constructs to a member that already has a capsule.

    Uses the published Chain-of-Planned-Behaviour prompts as written. That template also asks for a
    final numeric answer; it is parsed out and kept on the member, but the household's decision
    comes from stage two, not from here.
    """
    if not member.capsule:
        raise PersonaError(f"member {member.person_id}: build_capsule must run first")
    client = client or get_client("persona")

    if not config.copb_system or not config.copb_user:
        raise PersonaError(
            f"domain {config.name!r} has no Chain-of-Planned-Behaviour prompts. Each domain "
            f"supplies its own: travel asks about daily trips, mobility about relocation, and "
            f"running one domain's prompt on another is a silent wrong answer."
        )

    anchor = config.anchor(member.record)
    system = config.copb_system.replace(
        "{{ANCHOR}}", str(round(anchor, 1)) if anchor is not None else "unknown")

    marker = {"statement": "", "implication": ""}
    if config.markers:
        picked = stage1.get_attitudinal_marker(member.record, config.markers)
        if isinstance(picked, dict):
            marker = picked

    fields = {
        "anchor": anchor if anchor is not None else 0.0,
        "persona": member.persona,
        "marker_statement": marker.get("statement", ""),
        "marker_implication": marker.get("implication", ""),
        "role_hint": "household member",
        "veh_count": "unknown",
        "driver_status": "unknown",
    }
    if config.copb_fields:
        fields.update({k: v for k, v in config.copb_fields(member.record).items()
                       if v is not None})

    text = client.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": config.copb_user.format(**fields)},
    ], temperature=0.3)

    parsed = stage1.parse_tpb_sections(text)
    ok, reason = stage1.validate_tpb_extraction(parsed)
    if not ok:
        raise PersonaError(f"member {member.person_id}: TPB extraction failed ({reason})")

    member.tpb = {k: v for k, v in parsed.items()
                  if k in ("attitude", "subjective_norm", "perceived_behavioral_control")}
    return member


# --- household level --------------------------------------------------------------------------------

def build_personas(household: Household, config: DomainConfig,
                   client: Client | None = None, roster: bool = False) -> Household:
    """Build every member's persona.

    `roster` is off by default, which reproduces the published pipeline: personas there are built
    one member at a time, and the household roster is assembled later, at negotiation time, by
    `stage2.build_household_roster`.

    Turning it on writes each member's view of the others into the persona itself, before the
    behavioural constructs, so Subjective Norm can refer to them. That construct is defined as what
    the other household members expect of a person, and a persona that has not been told who they
    are cannot answer it. This changes persona content, so it is opt-in and any result produced
    with it should say so.
    """
    client = client or get_client("persona")

    for member in household:
        build_capsule(member, config, client)

    if roster and config.describe_member and config.relate_members:
        household.build_roster(config.describe_member, config.relate_members)

    for member in household:
        enrich(member, config, client)

    return household
