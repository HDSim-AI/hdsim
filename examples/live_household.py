"""Run a full simulation against a real model.

Needs a key. Set one first:

    cp .env.example .env        # then fill in HDSIM_API_KEY
    python examples/live_household.py

The household here is made up, so this file runs anywhere without a survey download. Swapping in
real data means replacing `build_household` with a loader from a domain package, for example
`hdsim.travel.load_nhts`.

The domain is defined inline rather than imported, to show what defining one involves. It is a
configuration object. No pipeline code changes.
"""

from __future__ import annotations

from hdsim.core import DecisionTask, DomainConfig, Household, Member, build_personas, simulate

# --- the domain ---------------------------------------------------------------------------------

TRIP_COUNT = DecisionTask(
    name="trip_count",
    domain="household travel",
    context="tomorrow",
    target_description="the total number of one-way trips everyone in the household will make",
    unit="trips",
    value_type=int,
    value_range=(0, 30),
)

AGE_WORDS = {"child": "I am a child.", "teen": "I am a teenager.", "adult": "I am an adult."}


def describe(member: Member) -> str:
    """How this person is introduced to their housemates."""
    bits = [f"{member.get('age')} years old"]
    if member.get("works"):
        bits.append("works full time")
    if member.get("in_school"):
        bits.append("in school")
    bits.append("drives" if member.get("drives") else "does not drive")
    return ", ".join(bits)


def relate(me: Member, other: Member) -> str:
    """How `other` relates to `me`. Real domains read this from the survey's relationship codes."""
    return other.get("role", "another member of my household")


TRAVEL = DomainConfig(
    name="travel",
    task=TRIP_COUNT,
    fact_columns=["age", "role", "works", "in_school", "drives", "vehicles", "household_size"],
    translations={
        "age": lambda v: f"I am {v} years old.",
        "role": lambda v: f"I am the {v} in my household.",
        "works": lambda v: "I work full time." if v else "I do not work.",
        "in_school": lambda v: "I attend school." if v else None,
        "drives": lambda v: ("I have a driver's licence." if v
                             else "I do not have a driver's licence."),
        "vehicles": lambda v: f"My household owns {v} vehicle(s).",
        "household_size": lambda v: f"There are {v} people in my household.",
    },
    # Persona text is written before the household decides. If it names the quantity under
    # discussion, the agents are no longer deciding anything.
    banned_patterns=[r"\btrips?\b", r"\btravel diary\b"],
    describe_member=describe,
    relate_members=relate,
)


# --- the household ------------------------------------------------------------------------------

def build_household() -> Household:
    return Household(
        household_id="example-1",
        members=[
            Member(1, {"age": 44, "role": "father", "works": True, "in_school": False,
                       "drives": True, "vehicles": 2, "household_size": 3}),
            Member(2, {"age": 41, "role": "mother", "works": True, "in_school": False,
                       "drives": True, "vehicles": 2, "household_size": 3}),
            Member(3, {"age": 15, "role": "son", "works": False, "in_school": True,
                       "drives": False, "vehicles": 2, "household_size": 3}),
        ],
    )


def main() -> None:
    household = build_household()

    print("Building personas")
    build_personas(household, TRAVEL)
    for member in household:
        print(f"\n  Member {member.person_id} ({member.get('role')})")
        print(f"    facts   {len(member.facts)}")
        print(f"    capsule {member.capsule}")
        print(f"    roster  {member.roster.splitlines()[0] if member.roster else '(none)'}")
        for name, text in member.tpb.items():
            print(f"    {name:<30} {text[:110]}")

    print("\nSimulating")
    simulate(household, TRAVEL)

    print("\nOpening positions, proposed independently")
    for member in household:
        print(f"  {member.get('role'):<8} {member.proposal_value} trips")
    print(f"  sum before discussion: {household.proposal_sum}")

    print("\nDiscussion")
    current = None
    for turn in household.transcript:
        if turn["round"] != current:
            current = turn["round"]
            print(f"  Round {current}")
        print(f"    {turn['speaker']}: {turn['text']}")

    print(f"\nAgreed: {household.consensus_value} trips")


if __name__ == "__main__":
    main()
