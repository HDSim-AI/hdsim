"""Prompt templates.

Kept in one file so the wording can be reviewed and changed without reading the pipeline. The
capsule templates are carried over unchanged from the research code, including the banned phrase
lists, because those constraints are what keep generated personas factual.

Anything domain specific is filled in from `DomainConfig` at call time. Nothing here mentions
travel, trips, mobility or any dataset.
"""

from __future__ import annotations

# --- persona capsule: facts to first-person narrative ------------------------------------------
# Carried over verbatim. The banned lists exist because early versions produced personas full of
# invented emotional content ("proud", "grateful") and invented responsibilities ("responsible
# for"), which then showed up as fabricated justifications during negotiation.

CAPSULE_SYSTEM = (
    "You are a factual data-to-text writer. Convert bullet points into a plain first-person "
    "paragraph. Rules: Use every fact exactly as given. Add NO new information. "
    "BANNED phrases: 'responsible for', 'manage/managing', 'take care of', 'On average', "
    "'participate in', 'did not express', 'positive/negative opinion about'. "
    "BANNED words: proud, fortunate, blessed, grateful, love, enjoy, excited, happy, thrilled, "
    "thankful. "
    "NEVER mention {banned_subject}, or 'Final Answer'."
)

CAPSULE_USER = (
    "Facts (include ALL of these, do not drop any):\n{facts}\n\n"
    "Output instructions:\n"
    "- Write in the first person (e.g., 'I am', 'I live').\n"
    "- Include EVERY fact above. Do NOT skip any facts.\n"
    "- Keep the meaning of each fact exactly. Do not introduce new wording that changes meaning.\n"
    "- For agree/disagree/neither statements, preserve the same stance (do not paraphrase into "
    "different sentiment).\n"
    "- Be neutral and objective. No emotional or subjective language.\n"
    "- Do not infer relationships not stated in facts.\n"
    "Persona:"
)

# --- Theory of Planned Behavior constructs ------------------------------------------------------
# Structure follows Ajzen. The three constructs give an agent explicit grounds to argue from during
# negotiation, which is what the moderator later checks its statements against.

TPB_SYSTEM = """You are an expert behavioral scientist analyzing {domain} decisions.

You will apply the Theory of Planned Behavior. Behavior follows from Attitude, Subjective Norm and
Perceived Behavioral Control.

PRIORITY RULES:
1. Perceived Behavioral Control overrides Attitude. Someone who wants to act but lacks the
   resources will not act.
2. Subjective Norm matters. Household roles create obligations.
3. Base every statement on a fact in the persona. Do not invent circumstances.
{anchor_note}"""

TPB_USER = """Persona:
{persona}
{roster_block}{marker_block}
Analyze this person using the Theory of Planned Behavior, with respect to {target_description}.

1. Attitude: Do they view {domain} positively or negatively, and why? Ground it in their facts.
2. Subjective Norm: What do the other household members listed above expect of them, and how much
   pressure is there to meet those expectations?
3. Perceived Behavioral Control: How easy or hard is it for them to act? Consider resources they
   control, resources shared with the household, and external constraints.

Respond in exactly this format and nothing else:
Attitude: <2-3 sentences>
Subjective Norm: <2-3 sentences>
Perceived Behavioral Control: <2-3 sentences>"""

# --- independent proposal ------------------------------------------------------------------------
# Each member answers alone, before hearing anyone else. Running these in parallel rather than in
# sequence is deliberate: it removes the turn-order bias you get when later members anchor on
# earlier ones.

PROPOSAL_SYSTEM = (
    "You are role-playing as a single household member. Stay strictly in character based on the "
    "profile and perceptions given to you. You are making a {domain} decision for {context}. "
    "Think about {target_description}. "
    "First explain your reasoning in 2-4 sentences grounded in your persona and perceptions, then "
    "on the LAST line output exactly:\n{final_label}: <{value_type}>"
)

PROPOSAL_USER = (
    "Your profile:\n{persona}\n\n"
    "Propose your initial decision, in {unit}. Justify it briefly from your profile and "
    "perceptions, then end with the required {final_label} line."
)

# --- negotiation ---------------------------------------------------------------------------------

NEGOTIATION_SYSTEM = (
    "You are moderating a household discussion about {target_description} for {context}.\n"
    "Each member has already proposed a value independently. Your job is to run the discussion "
    "until the household agrees on one number.\n\n"
    "Rules:\n"
    "- Members may only cite facts from their own profile. Do not let them invent circumstances.\n"
    "- A member who knows something the others do not should say it. That is how the household "
    "corrects a wrong estimate.\n"
    "- Respect shared resource limits. The household cannot use more of something than it has.\n"
    "- Stop when the members agree, and state the agreed value on the last line as\n"
    "  {final_label}: <{value_type}>"
)

NEGOTIATION_USER = (
    "Household members and their opening positions:\n\n{positions}\n\n"
    "Run the discussion. Write it as dialogue, one line per speaker, grouped into rounds. "
    "End with the {final_label} line."
)
