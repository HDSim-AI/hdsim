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

# Keys are STRINGS. A survey row can arrive as 1, "1" or 1.0 depending on the reader, so
# `render_fact` matches on the string form. An int-keyed dict silently renders no fact at all.
TRANSLATIONS = {
    "AGE": lambda v: f"I am {v} years old.",
    "WORKS_FROM_HOME": {"1": "I work from home most days.", "2": "I go into a workplace."},
    "HOME_TYPE": {"1": "I live in a detached house.", "2": "I live in an apartment."},
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


# 5. THE PROMPTS THAT TURN FACTS INTO BEHAVIOURAL CONSTRUCTS -------------------------------------
# Core ships no default on purpose: a domain running another domain's prompt produces a confident
# wrong answer with nothing to flag it. The parser needs exactly these four labels back. Keep
# `{{ANCHOR}}` if you set `anchors`/`anchor_for`, or the prior you measured is silently discarded.

COPB_SYSTEM = """You are a behavioral scientist applying the Theory of Planned Behavior to
household energy use. The share of comparable households that set a low winter thermostat is
{{ANCHOR}}. Treat that as a starting point to reason against, never as a threshold.

Respond STRICTLY in this format:
Attitude: <2-3 sentences on how this person feels about heating cost and comfort>
Subjective Norm: <2-3 sentences on what the rest of the household expects>
Perceived Behavioral Control: <2-3 sentences on what they can actually change>
Final Answer: <integer between 55 and 80>
"""

COPB_USER = """Persona:
{persona}

They are a {role_hint} in this household. Analyse them using the format above.
"""

ENERGY = DomainConfig(
    name="energy",
    task=THERMOSTAT,
    fact_columns=["AGE", "WORKS_FROM_HOME", "HOME_TYPE"],
    translations=TRANSLATIONS,
    banned_patterns=BANNED,
    describe_member=describe,
    relate_members=relate,
    copb_system=COPB_SYSTEM,
    copb_user=COPB_USER,
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
    # facts_for skips a column it cannot render rather than raising, so one unmapped survey code
    # costs you that sentence and not the run
    member.facts = ENERGY.facts_for(member.record)

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
