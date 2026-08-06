"""Offline replay.

A recorded household negotiation, played back turn by turn with no model, no key and no data
download. This is what `hdsim demo` runs, so someone who has just installed the package can see
what the method produces before deciding whether to configure a provider.

The recordings are real stage-two outputs, selected and formatted for reading rather than
generated for this package. They are multi-member households that reached consensus within two
rounds, and each turn keeps the member's dialogue while dropping the per-turn structured fields
(`MY_VALUE`, `PREFERRED_TOTAL`) that drive the ReST reward and are not meant to be read.

A live run will not sound like these. The negotiation prompt was written for a model fine-tuned by
that ReST loop, which learned to fill the acknowledgement slot with dialogue. A stock chat model
tends to fill it with a bare cross-reference instead, so it converges on a sensible number while
saying very little. Nothing here calls a network.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

FIXTURE = "negotiations.json"


def _load() -> list[dict[str, Any]]:
    try:
        text = resources.files("hdsim.core.fixtures").joinpath(FIXTURE).read_text()
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        local = Path(__file__).parent / "fixtures" / FIXTURE
        if not local.is_file():
            raise FileNotFoundError(
                "No bundled recordings found. Reinstall the package, or run a live "
                "simulation with hdsim.simulate()."
            )
        text = local.read_text()
    return json.loads(text)


def available() -> list[dict[str, str]]:
    """List the recordings, with a one-line summary of each."""
    return [
        {
            "id": r["household_id"],
            "members": str(len(r["members"])),
            "rounds": str(max((t["round"] for t in r["transcript"]), default=1)),
            "result": f"{r['consensus_value']} {r.get('unit', '')}".strip(),
        }
        for r in _load()
    ]


def get(household_id: str | None = None) -> dict[str, Any]:
    """Return one recording. Defaults to the first."""
    records = _load()
    if household_id is None:
        return records[0]
    for record in records:
        if record["household_id"] == household_id:
            return record
    known = ", ".join(r["household_id"] for r in records)
    raise KeyError(f"No recording {household_id!r}. Available: {known}")


def _say(value: Any, unit: str) -> str:
    """A decided value as a reader sees it.

    A yes or no domain stores True and False, and its unit describes the answer rather than
    counting anything, so "True a yes or no" is not a sentence. Booleans print as yes or no on
    their own; counts keep their unit.
    """
    if value is None:
        return "no answer"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value} {unit}".strip()


def render(record: dict[str, Any], show_personas: bool = False) -> str:
    """Format a recording for a terminal."""
    lines: list[str] = []
    unit = record.get("unit", "")
    lines.append(f"Household {record['household_id']}  ({len(record['members'])} members)")
    lines.append("")

    lines.append("Opening positions, proposed independently")
    for member in record["members"]:
        role = member.get("role", f"Member {member['person_id']}")
        lines.append(f"  {role:<22} {_say(member['proposal_value'], unit)}")
        if show_personas and member.get("capsule"):
            lines.append(f"      {member['capsule']}")
    if show_personas and not any(m.get("capsule") for m in record["members"]):
        # Better to say so than to look like the flag was ignored. The bundled recordings keep the
        # dialogue only; a recording you make yourself carries its personas.
        lines.append("  (this recording does not carry personas)")
    lines.append("")

    current = None
    for turn in record["transcript"]:
        if turn["round"] != current:
            current = turn["round"]
            lines.append(f"Round {current}")
        lines.append(f"  {turn['speaker']}: {turn['text']}")
    lines.append("")

    lines.append(f"Agreed: {_say(record['consensus_value'], unit)}")
    truth = record.get("ground_truth")
    if truth is not None and record.get("consensus_value") is not None:
        line = f"Recorded in the survey: {_say(truth, unit)}"
        if not isinstance(truth, bool):
            line += f"   (error {abs(record['consensus_value'] - truth)})"
        lines.append(line)
    return "\n".join(lines)
