"""Stage two: proposals and household negotiation.

Ported unchanged from the PEMAND stage-two implementation. The prompt wording, the label scheme and
the parsers are the method, so they are carried over rather than rewritten.

The three labels matter and are easy to collapse by accident. MY_VALUE is a member's own count,
PREFERRED_TOTAL is that member's view of the household figure, and FINAL_CONSENSUS is the agreed
household total, defined as the sum across all members. A prompt that asks only for "one number"
gets an average of the members instead of a sum.

`build_household_roster` reads each member's attributes back out of their persona text, so the
discussion can reference who is present without inventing anyone. The training half of the original
module, the ReST grow and improve loop and its reward model, is not included here.
"""

from __future__ import annotations

import json
import math
import re

# --- decision task and labels ------------------------------------------------------------

DECISION_TASK = {
    "domain": "{DECISION_DOMAIN}",
    "context": "{the time frame or situation of the decision}",
    "target_description": "{describe the quantity each member decides, and how to count it}",
    "unit": "{unit of the decided quantity}",
}
MEMBER_LABEL = "MY_VALUE"         
TOTAL_LABEL  = "PREFERRED_TOTAL"   
FINAL_LABEL  = "FINAL_CONSENSUS"  


# --- rendering --------------------------------------------------------------------------

def format_tpb(tpb) -> str:
    if isinstance(tpb, str):
        try:
            tpb = json.loads(tpb.replace("'", '"'))
        except Exception:
            return tpb
    if not isinstance(tpb, dict):
        return str(tpb)
    return "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in tpb.items())



def parse_vehicle_count(persona: str):
    m = re.search(r"household owns (\d+)\s*vehicle", persona, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r"\b(0|no)\s*vehicles?\b", persona, re.IGNORECASE):
        return 0
    return None


def parse_household_income(persona: str) -> str:
    m = re.search(r"household income is (.+?)\.", persona)
    return m.group(1).strip() if m else "unknown"


def parse_household_size(persona: str, fallback: int):
    m = re.search(r"There are (\d+) people in my household", persona)
    return int(m.group(1)) if m else fallback


def parse_member_attrs(persona: str) -> dict:
    a = {}
    if re.search(r"\b(woman|girl|female)\b", persona, re.IGNORECASE):
        a["gender"] = "female"
    elif re.search(r"\b(man|boy|male)\b", persona, re.IGNORECASE):
        a["gender"] = "male"
    age = re.search(r"between (\d+) and (\d+) years old", persona)
    if age:
        a["age_range"] = f"{age.group(1)}-{age.group(2)}"
    for role in ["head of my household", "wife", "husband", "spouse",
                 "son", "daughter", "child"]:
        if role in persona.lower():
            a["role"] = role.replace("head of my household", "head")
            break
    a["worker"] = "I am a worker" in persona
    a["driver"] = "I am a licensed driver" in persona
    return a



def build_household_roster(members: list) -> str:
    persona_with_household_info = next(
        (m["persona"] for m in members if "household" in m["persona"].lower()),
        members[0]["persona"],
    )
    veh   = parse_vehicle_count(persona_with_household_info)
    inc   = parse_household_income(persona_with_household_info)
    size  = parse_household_size(persona_with_household_info, len(members))
    n_drv = sum(1 for m in members if "I am a licensed driver" in m["persona"])
    lines = [
        "=== HOUSEHOLD ROSTER ===",
        "(canonical; do NOT invent any attribute not stated below; do NOT add "
        "members or resources)",
        f"Household size: {size}.  Income: {inc}.  "
        f"Vehicles owned: {'unknown' if veh is None else veh}.  "
        f"Licensed drivers in household: {n_drv}.",
        "",
        "Members:",
    ]
    for m in members:
        a = parse_member_attrs(m["persona"])
        gender = a.get("gender", "?")
        age    = a.get("age_range", "?")
        role   = a.get("role", "?")
        wk     = "worker" if a.get("worker") else "non-worker"
        dr     = "licensed driver" if a.get("driver") else "non-driver"
        lines.append(
            f"  - Member {m['PERSONID']}: {gender}, age {age}, role={role}, "
            f"{wk}, {dr}"
        )
    lines.append("=========================")
    return "\n".join(lines)



# --- proposal prompts (from initial_proposal.py) ----------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "You are role-playing as a single household member. "
    "Stay strictly in character based on the sociodemographic profile and perceptions "
    "given to you. You are making a {domain} decision for {context}. "
    "Think about {target_description}. "
    "First explain your reasoning in 2-4 sentences grounded in your persona and "
    "perceptions, then on the LAST line output exactly:\n"
    "{final_label}: <integer>"
)


def build_system_prompt(task=DECISION_TASK):
    return SYSTEM_PROMPT_TEMPLATE.format(**task)


def build_user_prompt(persona, tpb, task=DECISION_TASK):
    return (
        "Your sociodemographic profile:\n"
        f"{persona}\n\n"
        f"Your perceptions about {task['domain']} (Theory of Planned Behavior):\n"
        f"{format_tpb(tpb)}\n\n"
        f"Propose your initial decision (in {task['unit']}). "
        "Justify briefly from your persona and perceptions, then end with "
        f"the required {task['final_label']} line."
    )



def extract_decision_value(proposal_text, task=DECISION_TASK):
    """Pull the integer decision value out of the proposal."""
    if not proposal_text:
        return None
    label = re.escape(task["final_label"])
    m = re.search(rf"{label}\s*[:=]\s*(-?\d+)", proposal_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.findall(r"\b(\d+)\b", proposal_text)
    return int(m[-1]) if m else None



# --- consensus prompts -------------------------------------------------------------------

CONSENSUS_SYSTEM_TEMPLATE = (
    "You simulate an ENTIRE household discussing {target_description}. "
    "Use ONLY the attributes in the HOUSEHOLD ROSTER; never invent gender, "
    "role, age, worker status, or other attributes, and do not add members or "
    "resources that aren't listed.\n\n"
    "Write a short structured discussion in this exact format:\n"
    "  [Member <PERSONID>] ACKNOWLEDGE: <ref to another member or the roster> "
    "{member_label}: <int> {total_label}: <int>\n"
    "Have every member speak at least once. The household must converge to a "
    "single household total. After the discussion, on the LAST line output "
    "EXACTLY:\n"
    "{final_label}: <integer>\n\n"
    "{final_label} is the household's total {unit} for {context} (the sum "
    "across ALL members). It must be a single non-negative integer."
)


def build_consensus_system(task=DECISION_TASK):
    return CONSENSUS_SYSTEM_TEMPLATE.format(
        target_description=task["target_description"],
        unit=task["unit"],
        context=task["context"],
        member_label=MEMBER_LABEL,
        total_label=TOTAL_LABEL,
        final_label=FINAL_LABEL,
    )


CONSENSUS_SYSTEM = build_consensus_system()


def build_household_prompt(members: list) -> str:
    roster = build_household_roster(members)
    blocks = []
    for m in members:
        blocks.append(
            f"--- Member {m['PERSONID']} ---\n"
            f"Persona: {m['persona']}\n"
            f"TPB perceptions:\n{format_tpb(m.get('tpb', ''))}"
        )
    return (
        f"{roster}\n\n"
        + "\n\n".join(blocks)
        + "\n\nNow write the structured household discussion (one block per "
        "member, every member speaks at least once) and end with "
        f"{FINAL_LABEL}: <int>."
    )




# --- parsing the completion -------------------------------------------------------------

_FINAL_RE        = re.compile(rf"{re.escape(FINAL_LABEL)}\s*[:=]\s*(-?\d+)", re.IGNORECASE)
_MEMBER_BLOCK_RE = re.compile(r"\[Member\s+\d+\]", re.IGNORECASE)


def parse_final_consensus(completion: str):
    if not completion:
        return None
    matches = _FINAL_RE.findall(completion)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def parse_completion_to_result(completion: str, n_members: int) -> dict:
    """Convert a single-shot completion into the dict shape the reward
    function expects: {final_value, transcript (list), n_rounds}."""
    transcript = [ln.strip() for ln in completion.split("\n") if ln.strip()]
    n_blocks   = len(_MEMBER_BLOCK_RE.findall(completion))
    n_rounds   = max(1, math.ceil(n_blocks / max(1, n_members))) if n_blocks else 1
    return {
        "final_value": parse_final_consensus(completion),
        "transcript":  transcript,
        "n_rounds":    n_rounds,
    }


