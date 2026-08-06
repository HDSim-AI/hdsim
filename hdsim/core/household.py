"""Household and member data model.

Every stage of the pipeline reads and writes these two objects. Survey loaders in the domain
packages produce them, persona construction fills them in, and the negotiation consumes them.
Keeping one shared model is what lets a single pipeline serve travel, mobility and any domain
added later.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Member:
    """One person in a household.

    `record` holds the raw survey fields. `facts`, `capsule` and `tpb` are filled in by persona
    construction. `roster` describes the other members and is written by `Household.build_roster`.
    """

    person_id: int
    record: dict[str, Any] = field(default_factory=dict)

    # How this person is named in a transcript. Survey loaders set it from the relationship code,
    # so a discussion reads "Mother:" rather than "Member 2:". Falls back to a numbered label.
    label: str = ""

    facts: list[str] = field(default_factory=list)
    capsule: str = ""
    tpb: dict[str, str] = field(default_factory=dict)
    roster: str = ""

    proposal_text: str = ""
    proposal_value: float | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.record.get(key, default)

    @property
    def persona(self) -> str:
        """The full text handed to an agent: capsule, roster, then TPB constructs."""
        parts = [self.capsule]
        if self.roster:
            parts.append(self.roster)
        if self.tpb:
            parts.append("\n".join(f"{k.replace('_', ' ').title()}: {v}"
                                   for k, v in self.tpb.items()))
        return "\n\n".join(p for p in parts if p)


@dataclass
class Household:
    """A set of members that decides together."""

    household_id: str
    members: list[Member] = field(default_factory=list)
    ground_truth: float | None = None

    consensus_value: float | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0

    # What the decided value counts, e.g. "trips". Set from the domain when the household
    # negotiates, so a saved record can be read back without also carrying its DomainConfig.
    unit: str = ""

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[Member]:
        return iter(self.members)

    def member(self, person_id: int) -> Member | None:
        for m in self.members:
            if m.person_id == person_id:
                return m
        return None

    @property
    def proposal_sum(self) -> float:
        """Household total implied by independent proposals, before any negotiation."""
        return sum(m.proposal_value or 0.0 for m in self.members)

    def build_roster(self, describe, relate) -> None:
        """Write each member's view of the others.

        `describe(member) -> str` renders one person. `relate(a, b) -> str` names b's relation to a.
        Both are supplied by the domain, since relationship coding is survey specific.
        """
        for me in self.members:
            others = [o for o in self.members if o.person_id != me.person_id]
            if not others:
                me.roster = "I live alone. There are no other members of my household."
                continue
            lines = [f"- {relate(me, o)}: {describe(o)}" for o in others]
            me.roster = (
                "The other members of my household are:\n" + "\n".join(lines) +
                "\nI know my own plans in detail. I do not automatically know theirs."
            )

    # ---- serialisation -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        """The household as a replay recording.

        Keeps what a reader of the transcript needs and drops the rest: no raw survey record, no
        roster, no TPB block. `label` is written out as `role` because that is what a transcript
        calls a person. This is the shape `hdsim demo` plays back.
        """
        return {
            "household_id": self.household_id,
            "unit": self.unit,
            "consensus_value": self.consensus_value,
            "ground_truth": self.ground_truth,
            "members": [
                {
                    "person_id": str(m.person_id),
                    "role": m.label or f"Member {m.person_id}",
                    "proposal_value": m.proposal_value,
                    "capsule": m.capsule,
                }
                for m in self.members
            ],
            "transcript": list(self.transcript),
        }

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> Household:
        """Read back what `to_record` wrote.

        A record is deliberately lossy: it keeps the transcript and drops the survey row, the
        roster and the TPB block. It is the format `hdsim demo` plays, so a run you record and a
        run you replay have to be the same shape.
        """
        return cls(
            household_id=str(d["household_id"]),
            members=[
                Member(
                    person_id=int(m["person_id"]),
                    label=m.get("role", ""),
                    capsule=m.get("capsule", ""),
                    proposal_value=m.get("proposal_value"),
                )
                for m in d.get("members", [])
            ],
            ground_truth=d.get("ground_truth"),
            consensus_value=d.get("consensus_value"),
            transcript=d.get("transcript", []),
            unit=d.get("unit", ""),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Household:
        if d.get("members") and "role" in d["members"][0]:
            return cls.from_record(d)          # a record, not a full dump
        members = [Member(**m) for m in d.get("members", [])]
        return cls(
            household_id=str(d["household_id"]),
            members=members,
            ground_truth=d.get("ground_truth"),
            consensus_value=d.get("consensus_value"),
            transcript=d.get("transcript", []),
            rounds=d.get("rounds", 0),
            unit=d.get("unit", ""),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Household:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def load_households(path: str | Path) -> list[Household]:
    """Read a JSON Lines file of households."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(Household.from_dict(json.loads(line)))
    return out
