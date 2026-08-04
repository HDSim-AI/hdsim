<div align="center">

# 🏠 hdsim

**Household decision simulation through multi-agent negotiation.**

<p>
<a href="https://arxiv.org/abs/2604.10475"><img src="https://img.shields.io/badge/arXiv-2604.10475-b31b1b?style=flat-square"></a>
<img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square">
<img src="https://img.shields.io/badge/python-3.10%2B-informational?style=flat-square">
<img src="https://img.shields.io/badge/HDSim-method%20core-17212b?style=flat-square">
</p>

</div>

---

A household decides together. Classical models treat it as a single unit with a regression
coefficient. This library treats it as several people who hold different information and have to
reconcile it, which is closer to how the decision is actually made.

`hdsim` is the method core of the [HDSim](https://github.com/HDSim-AI) ecosystem. It implements
PEMAND: persona-enriched multi-agent negotiation. Domain packages add the survey loaders and
configuration for a specific decision; the pipeline itself does not change between them.

## Quick start

No API key and no data download:

```bash
pip install hdsim
hdsim demo
```

```
Household 23001767  (2 members)

Opening positions, proposed independently
  Husband                2 trips
  Wife                   4 trips

Round 1
  Husband: I know you're planning to go grocery shopping tomorrow, so that's two one-way
           trips. Off the top of my head, I'd say we have 2 trips planned.
  Wife:    That sounds a little low. Don't forget you'll be driving to work in the morning
           and coming back in the afternoon. That's another two trips, so I think the total
           should actually be 4 trips.
Round 2
  Husband: You're right, I completely forgot to count my commute. Adding my drive to work
           and back with your grocery trip brings us to 4 trips.
  Wife:    Exactly. I don't think we've missed anything else, so 4 trips sounds right to me.

Agreed: 4 trips
```

That is the point of the method in one example. The husband is wrong at the start, the wife knows
something he has forgotten, and the household total moves because she says it.

## Running it live

```bash
cp .env.example .env      # then add a key
hdsim config              # check what each role will use
```

```python
from hdsim.travel import NHTS, Household, build_personas, simulate   # pip install hdsim-travel

household = Household.from_json("household.json")

build_personas(household, NHTS)        # facts -> capsule -> roster -> TPB constructs
simulate(household, NHTS)              # independent proposals, then negotiation

print(household.consensus_value)       # 4
print(household.proposal_sum)          # 6, before anyone talked
for turn in household.transcript:
    print(turn["round"], turn["speaker"], turn["text"])
```

## How it works

```
survey record
     │
     ├─ [1] facts        every field becomes a first-person sentence
     ├─ [2] capsule      an LLM writes them into a paragraph, checked for invented content
     ├─ [3] roster       each member is told who else is in the household
     └─ [4] constructs   attitude, subjective norm, perceived behavioral control
     │
     ▼
[5] proposal      every member answers alone, in parallel
     │
     ▼
[6] negotiation   members discuss and correct each other, moderated
     │
     ▼
household decision
```

Three details matter more than they look.

**Proposals run in parallel.** Asking members one after another lets later ones anchor on earlier
ones, so the household agrees because of turn order rather than because of anything about the
household.

**Personas are checked, not trusted.** A capsule that drops facts, adds emotional language, or
mentions the quantity being decided is regenerated. Each of those failures shows up later as a
fabricated justification during the discussion.

**Each member is told who the others are, not what they plan.** That asymmetry is what gives the
discussion something to correct.

## Configuration

Any provider that speaks the OpenAI chat API works. Point the base URL at it.

```bash
HDSIM_API_KEY=sk-...
HDSIM_MODEL=gpt-4o-mini
HDSIM_BASE_URL=https://api.openai.com/v1
```

| Endpoint | `HDSIM_BASE_URL` |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Ollama | `http://localhost:11434/v1` (key can stay empty) |
| vLLM | `http://localhost:8000/v1` |
| Together | `https://api.together.xyz/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

A member proposing a number is a cheap call. The moderator checking consistency and feasibility is
the one worth paying for. Set a role to override it; anything left blank falls back to
`HDSIM_MODEL`.

```bash
HDSIM_PERSONA_MODEL=
HDSIM_MEMBER_MODEL=
HDSIM_MODERATOR_MODEL=
```

Precedence is argument, then environment variable, then `.env`, then default.

To run weights locally instead: `pip install hdsim[local]`.

## Adding a domain

A decision domain is data, not code. Supply a `DomainConfig` and the pipeline runs unchanged.

```python
from hdsim import DecisionTask, DomainConfig

ENERGY = DomainConfig(
    name="energy",
    task=DecisionTask(
        name="thermostat",
        domain="household energy use",
        context="this winter",
        target_description="the temperature the household sets the thermostat to",
        unit="degrees",
    ),
    fact_columns=["AGE", "INCOME", "HOME_TYPE"],
    translations={"AGE": lambda v: f"I am {v} years old."},
    banned_patterns=[r"\bthermostat\b"],   # never let the persona state the answer
)
```

`banned_patterns` is not optional in practice. Persona text is written before the household
decides, so if it names the quantity under discussion the agents are no longer deciding anything.

## Results

| Dataset | Metric | Best baseline | PEMAND |
|---|---|---|---|
| NHTS 2017 | MAE ↓ | 3.07 (Gradient Boosting) | **2.38** |
| NHTS 2017 | sMAPE ↓ | 50.84 (MLP) | **34.48** |
| Puget Sound 2023 | MAE ↓ | 2.75 (Random Forest) | **1.99** |
| Puget Sound 2023 | ±2 Accuracy ↑ | 0.59 (Random Forest) | **0.78** |

Table 1, [arXiv:2604.10475v2](https://arxiv.org/abs/2604.10475).

## Domain packages

| Package | Repository | Decision |
|---|---|---|
| `hdsim.travel` | [travel-decision](https://github.com/HDSim-AI/travel-decision) | Trips the household makes tomorrow |
| `hdsim.mobility` | [residential-mobility](https://github.com/HDSim-AI/residential-mobility) | Whether the household relocates |

## Layout

```
hdsim/
├── household.py    Household and Member, shared by every stage
├── domain.py       DomainConfig, everything that differs between domains
├── persona.py      facts -> capsule -> constructs, with validation
├── negotiate.py    parallel proposals, then moderated discussion
├── backends.py     model access and role configuration
├── prompts.py      every template, in one file
├── replay.py       offline recordings for the demo
├── evaluate.py     metrics and a paired bootstrap
└── cli.py          hdsim demo, hdsim config
```

## Contributing

New decision domains, survey loaders and evaluations are all welcome. Open an issue describing the
decision you want to model before writing much code, so the `DomainConfig` shape can be checked
first.

## Citation

```bibtex
@article{sun2026pemand,
  title   = {PEMAND: Persona-Enriched Multi-Agent Negotiation for Household Decision-Making},
  author  = {Sun, Yuran and Sameen, Mustafa and Zhang, Yaotian and Gu, Rongguan and
             Vibhute, Mrunal and Wu, Chia-yu and Lei, Yuanyuan and Zhao, Xilei},
  journal = {arXiv preprint arXiv:2604.10475},
  year    = {2026}
}
```

MIT licensed.
