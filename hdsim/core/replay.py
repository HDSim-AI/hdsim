"""Offline replay.

A recorded household negotiation, played back turn by turn with no model, no key and no data
download. This is what `hdsim demo` runs, so someone who has just installed the package can see
what the method produces before deciding whether to configure a provider.

The recordings are real runs, shipped as fixtures. Nothing here calls a network.
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


def render(record: dict[str, Any], show_personas: bool = False) -> str:
    """Format a recording for a terminal."""
    lines: list[str] = []
    unit = record.get("unit", "")
    lines.append(f"Household {record['household_id']}  ({len(record['members'])} members)")
    lines.append("")

    lines.append("Opening positions, proposed independently")
    for member in record["members"]:
        role = member.get("role", f"Member {member['person_id']}")
        lines.append(f"  {role:<22} {member['proposal_value']} {unit}")
        if show_personas and member.get("capsule"):
            lines.append(f"      {member['capsule']}")
    lines.append("")

    current = None
    for turn in record["transcript"]:
        if turn["round"] != current:
            current = turn["round"]
            lines.append(f"Round {current}")
        lines.append(f"  {turn['speaker']}: {turn['text']}")
    lines.append("")

    lines.append(f"Agreed: {record['consensus_value']} {unit}")
    if record.get("ground_truth") is not None:
        truth = record["ground_truth"]
        error = abs(record["consensus_value"] - truth)
        lines.append(f"Recorded in the survey: {truth} {unit}   (error {error})")
    return "\n".join(lines)
