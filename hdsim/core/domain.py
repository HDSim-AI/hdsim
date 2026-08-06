"""Domain configuration.

A decision domain is defined by data, not by code. Travel and residential mobility run the same
persona construction, the same proposal step and the same negotiation. What differs is which
survey columns matter, how coded values read in English, what the household is deciding, and how
to parse the answer.

`DomainConfig` holds exactly those differences. One instance per domain lives in the domain
package (`hdsim.travel.config`, `hdsim.mobility.config`), which is why the core carries no
dataset-specific code.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionTask:
    """What the household is being asked to decide."""

    name: str                     # "trip_count"
    domain: str                   # "household travel"
    context: str                  # "tomorrow"
    target_description: str       # "the total number of one-way trips the household will make"
    unit: str                     # "trips"

    # Three quantities, three labels. A member's own count, that member's view of the household
    # total, and the agreed total are different numbers, and a discussion that cannot name them
    # separately collapses them: asked for "one number", a model averages the members instead of
    # summing them. Kept aligned with the negotiation implementation in the research pipeline.
    member_label: str = "MY_VALUE"
    total_label: str = "PREFERRED_TOTAL"
    final_label: str = "FINAL_VALUE"
    value_type: type = int
    value_range: tuple[float, float] = (0, 30)


@dataclass
class DomainConfig:
    """Everything the core needs in order to run a domain it has never seen."""

    name: str
    task: DecisionTask

    # Survey columns that become persona facts, in the order they should be read.
    fact_columns: list[str] = field(default_factory=list)

    # column -> {coded value -> English}, or column -> callable(value) -> str.
    # A column absent here is rendered by `default_renderer`.
    translations: dict[str, Any] = field(default_factory=dict)

    # Empirical priors: mean outcome for people like this. Used as a qualitative starting point,
    # never as a threshold. Keys are domain defined; `anchor_for` resolves a record to one.
    anchors: dict[str, float] = field(default_factory=dict)
    anchor_for: Callable[[dict], float] | None = None

    # Optional behavioural tendency injected into persona construction.
    markers: list[dict[str, str]] = field(default_factory=list)

    # Phrases that must never appear in a generated persona because they leak the outcome.
    # Travel bans "trip"; mobility bans "move" and "relocate".
    banned_patterns: list[str] = field(default_factory=list)

    # How a member is described to housemates, and how relationships are named.
    describe_member: Callable[[Any], str] | None = None
    relate_members: Callable[[Any, Any], str] | None = None

    # The household roster handed to the negotiation, as `household -> str`. Left unset, the
    # stage-two roster is used, which recovers attributes from persona text with travel-survey
    # patterns and reports "unknown" for anything it does not recognise. Set this in any domain
    # that is not travel: an unrecognised attribute is not neutral, it is stated to the model as
    # canonical fact and contradicts the member's own persona.
    household_roster: Callable[[Any], str] | None = None

    # The Chain-of-Planned-Behaviour prompts. These are domain specific: the travel version asks a
    # transportation psychologist about daily trips, the mobility version asks a demographic
    # sociologist about relocation, and each carries its own empirical rules. Core supplies no
    # default, because a domain running another domain's prompt is a silent wrong answer.
    copb_system: str = ""
    copb_user: str = ""

    # Values the Chain-of-Planned-Behaviour user template needs, which only the domain knows:
    # role_hint, veh_count and driver_status. Returning {} lets the defaults stand.
    copb_fields: Callable[[dict], dict] | None = None

    # Escape hatch for surveys whose facts cannot be rendered one column at a time.
    # Age and sex together give "a 44-year-old man"; relationship and sex together give "the wife
    # of the household head"; household size and adult count together give the composition
    # sentence. Per-column rendering cannot express any of those. A domain that needs it supplies
    # `facts_fn(record) -> list[str]`, which replaces `fact_columns` and `translations` entirely.
    facts_fn: Callable[[dict], list[str]] | None = None

    def __post_init__(self) -> None:
        self._banned = [re.compile(p, re.IGNORECASE) for p in self.banned_patterns]

    # ---- fact rendering ------------------------------------------------------------------

    def default_renderer(self, column: str, value: Any) -> str | None:
        """Fallback when a column has no entry in `translations`."""
        if value is None or value == "":
            return None
        return f"My {column.lower().replace('_', ' ')} is {value}."

    def render_fact(self, column: str, value: Any) -> str | None:
        """One survey field to one first-person sentence, or None to skip it."""
        if value is None or value == "":
            return None
        rule = self.translations.get(column)
        if rule is None:
            return self.default_renderer(column, value)
        if callable(rule):
            return rule(value)
        key = str(value)
        text = rule.get(key)
        if text is None:
            try:
                text = rule.get(str(int(float(value))))
            except (TypeError, ValueError):
                text = None
        return text

    def facts_for(self, record: dict[str, Any]) -> list[str]:
        """Survey record to an ordered list of first-person facts."""
        if self.facts_fn is not None:
            return self.facts_fn(record)
        facts = []
        for col in self.fact_columns:
            if col not in record:
                continue
            fact = self.render_fact(col, record[col])
            if fact:
                facts.append(fact)
        return facts

    # ---- guards --------------------------------------------------------------------------

    def leaks_outcome(self, text: str) -> str | None:
        """Return the first banned phrase found, or None.

        Persona text is written before the household decides. If it mentions the quantity being
        decided, the decision is no longer being made by the agents.
        """
        for pat in self._banned:
            m = pat.search(text)
            if m:
                return m.group(0)
        return None

    def anchor(self, record: dict[str, Any]) -> float | None:
        if self.anchor_for is None:
            return None
        return self.anchor_for(record)
