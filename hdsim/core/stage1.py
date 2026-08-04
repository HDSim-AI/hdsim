"""Stage one: persona construction.

Ported unchanged from the PEMAND stage-one implementation. The prompt wording, the text cleanup and
the section parser are the method, so they are carried over rather than rewritten.

`clean_persona_text` is longer than it looks like it should be, and every rule in it was added
because a model produced something that broke a downstream step: preambles, restated bullet lists,
second-person drift, duplicated sentences. A short replacement passes casual inspection and lets
those back in.

`parse_tpb_sections` is strict about the three section headers because the enrichment step writes
into a fixed schema. A response that loses a header is a failed generation, not something to
salvage with a looser regex.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List

# --- persona capsule prompts -------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a factual data-to-text writer. Convert bullet points into a plain first-person paragraph. "
    "Rules: Use every fact exactly as given. Add NO new information. "
    "BANNED phrases: 'responsible for', 'manage/managing', 'take care of', 'On average', 'participate in', "
    "'did not express', 'positive/negative opinion about'. "
    "BANNED words: proud, fortunate, blessed, grateful, love, enjoy, excited, happy, thrilled, thankful. "
    "NEVER mention trip counts, travel diaries, or 'Final Answer'."
)
USER_TEMPLATE = (
    "Facts (include ALL of these, do not drop any):\n{facts}\n\n"
    "Output instructions:\n"
    "- Write in the first person (e.g., 'I am', 'I live').\n"
    "- Include EVERY fact above. Do NOT skip any facts.\n"
    "- Keep the meaning of each fact exactly. Do not introduce new wording that changes meaning.\n"
    "- For agree/disagree/neither statements, preserve the same stance (do not paraphrase into different sentiment).\n"
    "- Be neutral and objective. No emotional or subjective language.\n"
    "- Do not infer relationships not stated in facts.\n"
    "Persona:"
)
INVALID_PATTERNS = [re.compile(r"\btrip(s)?\b", re.IGNORECASE), re.compile(r"Final Answer", re.IGNORECASE)]

def build_prompt(facts: List[str]) -> str:
    bullet_block = "\n".join(f"- {fact}" for fact in facts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(facts=bullet_block)},
    ]
    tokenizer = build_prompt.tokenizer
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)




# --- persona text cleanup and validation ------------------------------------------------

def clean_persona_text(text: str) -> str:
    """Remove common model artifacts and prompt leakage from persona text."""
    if not text:
        return text
    
    # Strategy 1: Find first occurrence of persona-starting patterns
    # The actual persona should start with "I am", "I'm", "As a/an", "My name"
    persona_start_patterns = [
        r'\bI am an? \d+',         # "I am a 45-year-old" or "I am an 18-year-old"
        r'\bI\'m an? \d+',         # "I'm a 45-year-old" or "I'm an 80-year-old"
        r'\bAs an? \d+',           # "As a 45-year-old"
        r'\bI am \d+ years',       # "I am 45 years old"
        r'\bI am an? [A-Z][a-z]+', # "I am a White" or "I am an Asian"
    ]
    
    best_match = None
    best_pos = len(text)
    
    for pattern in persona_start_patterns:
        match = re.search(pattern, text)
        if match and match.start() < best_pos:
            best_match = match
            best_pos = match.start()
    
    if best_match:
        text = text[best_match.start():]
        # Capitalize first letter if needed
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text.strip()
    
    # Strategy 2: Remove known prompt leakage patterns
    leakage_patterns = [
        r".*?don't assume.*?\)\.\s*",          # "...don't assume 'spouse'...)."
        r".*?Only '[^']+' is mentioned\)\.\s*", # "Only 'adults' is mentioned)."
        r".*?do not infer.*?\)\.\s*",          # "do not infer relationships..."
        r".*?Output instructions:.*?\n",        # Output instructions block
        r".*?Facts \(.*?\)\.\s*",              # "Facts (use all...)."
        r"^[^I]*?(?=I am|I'm)",                # Everything before "I am/I'm"
    ]
    
    for pattern in leakage_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Strategy 3: Fallback to iterative cleaning of known prefixes
    prefixes_to_remove = [
        r'^assistant\s*:?\s*',
        r'^persona\s*:?\s*',
        r'^daily counts\.?\s*',
        r'^counts\.?\s*',
        r'^here is the (rewritten )?paragraph:?\s*',
        r'^here\'s the (rewritten )?paragraph:?\s*',
        r'^the persona is:?\s*',
        r'^\d+\s+(plain|factual)\s+sentences.*?\n',  # "5 plain, factual sentences"
        r'^-\s*Be neutral.*?\n',                     # "- Be neutral..."
    ]
    
    while True:
        original_text = text
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = text.lstrip(':- \t\n.')
        if text == original_text:
            break
    
    # Final cleanup: ensure starts with capital letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text.strip()


def validate_and_clean_persona(text: str, min_words: int) -> tuple[bool, str]:
    """Validate and clean persona text. Returns (is_valid, cleaned_text)."""
    if not text:
        return False, ""
    
    # Clean the text once
    cleaned_text = clean_persona_text(text)
    if not cleaned_text:
        return False, ""
        
    if len(cleaned_text.split()) < min_words:
        return False, ""
    for pattern in INVALID_PATTERNS:
        if pattern.search(cleaned_text):
            return False, ""
    return True, cleaned_text




# --- Chain-of-Planned-Behaviour prompts -------------------------------------------------

ACTOR_SYSTEM_PROMPT = """You are an expert Transportation Behavioral Psychologist predicting daily travel behavior.

REFERENCE ANCHOR (from 2009 NHTS - note: predicting for 2017):
People with similar demographics made an average of {{ANCHOR}} trips per day in 2009.
Since 2009: ride-sharing increased zero-vehicle mobility, seniors are more active, WFH is more common.
Use this as a starting point, then adjust based on the persona's specific circumstances.

PRIORITY RULES:
1. PBC overrides Attitude. Someone who wants to travel but lacks resources (no car, health issues) will make fewer trips.
2. Subjective Norms matter. Household roles create obligations (parent = school drop-offs, head = errands).
3. Zero trips is valid. If constraints are high (no vehicle, poor health, WFH), predict 0.

CITATION: Every claim must reference facts from the persona capsule.
"""

# System prompt for Demographics Only baseline (SIMPLE - no TPB/CoT)

COPB_USER_TEMPLATE = """Persona Capsule:
{persona}

Behavioral Tendency (inferred from demographics):
Based on similar demographic profiles, this individual may tend toward: "{marker_statement}"
This suggests: {marker_implication}
NOTE: Verify this tendency against the persona facts. The persona details take precedence.

Predict total trips on the recorded travel day using Theory of Planned Behavior:

1. **Attitude**: Analyze their attitude toward travel.
   - Do they view travel positively (convenient, necessary, enjoyable) or negatively (costly, stressful)?
   - Does the behavioral tendency above align with or contradict their persona facts?

2. **Subjective Norm**: What do household members expect regarding travel?
   - Based on their role ({role_hint}), what obligations exist?
   - How much pressure to conform to these expectations?

3. **Perceived Behavioral Control (PBC)**: How easy or difficult is travel?
   - Control: Are travel decisions under their control?
   - Ability: Resources available? (vehicles: {veh_count}, driver: {driver_status}, health, income)
   - Constraints: Location factors, transit access, costs?

Respond STRICTLY in this format:
Attitude: <2-3 sentences analyzing their attitude toward travel>
Subjective Norm: <2-3 sentences on household expectations and conformity pressure>
Perceived Behavioral Control: <2-3 sentences on control, ability, and constraints>
Final Answer: <integer between 0 and 30>
"""



# --- parsing the TPB response -----------------------------------------------------------

def _strip_final_answer(text: str) -> str:
    """Remove Final Answer and everything after it from the text."""
    if not text:
        return ""
    
    # Handle various Final Answer formats
    patterns = [
        r"Final Answer:\s*\d+.*$",
        r"\*\*Final Answer\*\*:\s*\d+.*$",
        r"Final Answer\s*:\s*\n?\s*\d+.*$",
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def _clean_section_content(content: str) -> str:
    """Clean extracted section content."""
    if not content:
        return ""
    
    # Remove leading/trailing whitespace
    content = content.strip()
    
    # Remove multiple consecutive newlines
    content = re.sub(r'\n\s*\n+', '\n', content)
    
    # Remove any trailing section markers that got included
    content = re.sub(r'\n(?:\d+\.\s*)?\*?\*?(?:Attitude|Subjective|Perceived).*$', '', content, flags=re.IGNORECASE)
    
    return content.strip()


def parse_tpb_sections(response_text: str) -> Dict[str, str]:
    """
    Extract TPB sections from model output.
    
    Expected headings (flexible matching):
      - Attitude:
      - Subjective Norm:
      - Perceived Behavioral Control: (or PBC:)
    
    Handles:
      - Numbered headings: "1. Attitude:", "2. Subjective Norm:"
      - Bold markdown: "**Attitude**:", "__Attitude__:"
      - Variations in spacing and casing
      - Missing sections (returns empty string)
    
    Returns dict with keys: attitude, subjective_norm, perceived_behavioral_control, reasoning_trace
    """
    trace = _strip_final_answer(response_text or "")
    out = {
        "attitude": "", 
        "subjective_norm": "", 
        "perceived_behavioral_control": "", 
        "reasoning_trace": trace
    }
    
    if not trace:
        return out

    # Robust patterns that handle:
    # - Optional numbering (1., 2., 3., etc.)
    # - Optional markdown bold (**...**) or (__...__)
    # - Optional dash or bullet prefix
    # - Case insensitive
    section_markers = [
        ("attitude", r"(?:[-•]\s*)?(?:\d+\.\s*)?(?:\*\*|__)?Attitude(?:\*\*|__)?[:\s]"),
        ("subjective_norm", r"(?:[-•]\s*)?(?:\d+\.\s*)?(?:\*\*|__)?Subjective\s*Norm[s]?(?:\*\*|__)?[:\s]"),
        ("perceived_behavioral_control", r"(?:[-•]\s*)?(?:\d+\.\s*)?(?:\*\*|__)?(?:Perceived\s*Behavioral\s*Control|PBC)(?:\*\*|__)?[:\s]"),
    ]
    
    # Find all section starts
    section_positions = []
    for key, pattern in section_markers:
        match = re.search(pattern, trace, flags=re.IGNORECASE)
        if match:
            section_positions.append((match.start(), match.end(), key))
    
    # Sort by position in text
    section_positions.sort(key=lambda x: x[0])
    
    # Extract content between sections
    for i, (start, end, key) in enumerate(section_positions):
        content_start = end
        # Content ends at the next section or end of trace
        if i + 1 < len(section_positions):
            content_end = section_positions[i + 1][0]
        else:
            content_end = len(trace)
        
        content = trace[content_start:content_end]
        out[key] = _clean_section_content(content)

    return out


def validate_tpb_extraction(parsed: Dict[str, str]) -> tuple:
    """
    Validate TPB extraction and return (is_complete, issues_list).
    """
    issues = []
    
    if not parsed.get("attitude"):
        issues.append("Missing Attitude section")
    elif len(parsed["attitude"]) < 20:
        issues.append(f"Attitude too short ({len(parsed['attitude'])} chars)")
    
    if not parsed.get("subjective_norm"):
        issues.append("Missing Subjective Norm section")
    elif len(parsed["subjective_norm"]) < 20:
        issues.append(f"Subjective Norm too short ({len(parsed['subjective_norm'])} chars)")
    
    if not parsed.get("perceived_behavioral_control"):
        issues.append("Missing PBC section")
    elif len(parsed["perceived_behavioral_control"]) < 20:
        issues.append(f"PBC too short ({len(parsed['perceived_behavioral_control'])} chars)")
    
    is_complete = len(issues) == 0
    return is_complete, issues



# --- attitudinal marker selection ------------------------------------------------------

def get_attitudinal_marker(persona_dict, markers_list):
    """
    Returns a weighted random choice of Attitudinal Marker based on demographics.
    
    The markers are from attitudinal_markers.json and represent psychological
    constructs that influence travel behavior according to Theory of Planned Behavior.
    
    Markers:
    - Tech-Savvy: High comfort with app-based mobility
    - Polychronic: Tolerance for multitasking during travel
    - Materialistic: Preference for private vehicles, status-seeking
    - Pro-Car: Positive attitude toward driving
    - Pro-Transit: Positive attitude toward public transit
    - Wait Tolerant: Lower disutility of travel time
    
    Args:
        persona_dict: Dictionary with demographic fields (URBRUR, HHVEHCNT, HHFAMINC, R_AGE_IMP, USEPUBTR, WORKER)
        markers_list: List of dicts [{"construct": "Name", "statement": "...", "implication": "..."}]
    
    Returns:
        dict: The selected marker object
    """
    
    # Init weights (uniform start)
    weights = {m['construct']: 1.0 for m in markers_list}
    
    # Demographic extraction
    urbrur = persona_dict.get('URBRUR')  # 1=Urban, 2=Rural
    veh_cnt = persona_dict.get('HHVEHCNT', 0)
    income = persona_dict.get('HHFAMINC', 5)  # Default mid-range
    age = persona_dict.get('R_AGE_IMP', 40)
    use_transit = persona_dict.get('USEPUBTR') == 1
    
    # --- LOGIC GATES ---
    
    # Pro-Transit
    if urbrur == 1:  # Urban
        weights['Pro-Transit'] += 2.0
    if veh_cnt == 0:
        weights['Pro-Transit'] += 5.0  # Strong boost
        weights['Pro-Car'] = 0.0  # Exclusion: Can't be pro-car if no car
    if use_transit:
        weights['Pro-Transit'] += 3.0
        
    # Pro-Car
    if urbrur == 2:  # Rural
        weights['Pro-Car'] += 3.0
        weights['Pro-Transit'] = 0.1  # Very unlikely
    if veh_cnt >= 2:
        weights['Pro-Car'] += 2.0
        
    # Tech-Savvy (Young + High Income)
    if age < 35 and income >= 7:  # >$75k
        weights['Tech-Savvy'] += 3.0
    elif age > 70:
        weights['Tech-Savvy'] = 0.1
        
    # Materialistic (High Income)
    if income >= 9:  # >$125k
        weights['Materialistic'] += 2.0
        
    # Wait Tolerant (Older or Low Income)
    if age > 65:
        weights['Wait Tolerant'] += 2.0
        
    # Polychronic (Busy Worker)
    if persona_dict.get('WORKER') == 1:
        weights['Polychronic'] += 1.5
        
    # Normalize and Sample
    choices = list(weights.keys())
    probs = list(weights.values())
    
    # Safety: ensure at least one positive weight
    if sum(probs) == 0:
        probs = [1.0] * len(choices)
        
    selected_construct = random.choices(choices, weights=probs, k=1)[0]
    
    # Return the full marker object
    for m in markers_list:
        if m['construct'] == selected_construct:
            return m
            
    return markers_list[0]  # Fallback

