"""A complete decision domain, start to finish.

    python examples/minimal_domain.py

Runs with no API key: it builds the domain and shows what each member will say about itself. The
last few lines need a key, and are the same two calls every domain uses.

Copy this file, change the four marked places, and you have a new domain. Nothing in `hdsim.core`
needs to change.
"""

from hdsim.core import DecisionTask, DomainConfig, Household, Member

# 1. WHAT IS BEING DECIDED -----------------------------------------------------------------------
# `unit` and `context` are dropped straight into the prompts, so write them as English.
# `value_type` is int for a count or bool for a yes or no.

THERMOSTAT = DecisionTask(
    name="thermostat",
    domain="household energy use",
    context="this winter",
    target_description="the temperature the household sets the thermostat to",
    unit="degrees",
    value_type=int,
    value_range=(55, 80),
)

# 2. HOW A SURVEY ROW READS IN ENGLISH -----------------------------------------------------------
# One entry per column you want the persona to know about. A value maps through a dict or a
# function. Columns you leave out are simply not mentioned.

TRANSLATIONS = {
    "AGE": lambda v: f"I am {v} years old.",
    "WORKS_FROM_HOME": {1: "I work from home most days.", 2: "I go into a workplace."},
    "HOME_TYPE": {1: "I live in a detached house.", 2: "I live in an apartment."},
}

# 3. WHAT A PERSONA MAY NEVER SAY ----------------------------------------------------------------
# Persona text is written before the household decides. If it names the quantity under discussion,
# the agents are no longer deciding anything and the result looks excellent while meaning nothing.
# This is the single easiest way to get a wrong answer that passes review.

BANNED = [r"\bthermostat\b", r"\bdegrees?\b", r"\btemperature\b"]

# 4. HOW MEMBERS ARE INTRODUCED TO EACH OTHER ----------------------------------------------------
# Where the survey does not establish a relationship, say something weaker and true. A roster that
# guesses causes the confusion it exists to prevent.


def describe(member: Member) -> str:
    return f"a {member.get('AGE')}-year-old member of the household"


def relate(_me: Member, other: Member) -> str:
    return "someone I live with" if other.get("AGE", 0) >= 18 else "a child I live with"


ENERGY = DomainConfig(
    name="energy",
    task=THERMOSTAT,
    fact_columns=["AGE", "WORKS_FROM_HOME", "HOME_TYPE"],
    translations=TRANSLATIONS,
    banned_patterns=BANNED,
    describe_member=describe,
    relate_members=relate,
    # copb_system and copb_user hold the prompts that turn facts into attitude, subjective norm
    # and perceived control. Core ships no default on purpose: a domain running another domain's
    # prompt produces a confident wrong answer with nothing to flag it. See hdsim/travel/copb.py.
)


# --- that is the whole domain. everything below is just running it ------------------------------

household = Household(
    household_id="example",
    members=[
        Member(person_id=1, label="Partner A", record={"AGE": 34, "WORKS_FROM_HOME": 1,
                                                       "HOME_TYPE": 1}),
        Member(person_id=2, label="Partner B", record={"AGE": 36, "WORKS_FROM_HOME": 2,
                                                       "HOME_TYPE": 1}),
    ],
)

for member in household:
    member.facts = [
        ENERGY.translations[column](member.get(column))
        if callable(ENERGY.translations[column])
        else ENERGY.translations[column][member.get(column)]
        for column in ENERGY.fact_columns
        if member.get(column) is not None
    ]

household.build_roster(ENERGY.describe_member, ENERGY.relate_members)

print(f"Domain: {ENERGY.name}   deciding: {ENERGY.task.target_description}\n")
for member in household:
    print(member.label)
    for fact in member.facts:
        print(f"  {fact}")
    print(f"  roster: {member.roster.splitlines()[1] if member.roster else '(none)'}\n")

print("To run the negotiation, set HDSIM_API_KEY and add:\n")
print("    from hdsim.core import build_personas, simulate")
print("    build_personas(household, ENERGY)")
print("    simulate(household, ENERGY)")
print("    print(household.consensus_value)")
