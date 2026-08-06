<div align="center">

# 🏠 hdsim

**Household decision simulation through multi-agent negotiation.**

<p>
<a href="https://github.com/HDSim-AI/hdsim"><img src="https://img.shields.io/github/stars/HDSim-AI/hdsim?style=flat-square&amp;logo=github" alt="Stars"></a>
<a href="https://github.com/HDSim-AI/hdsim/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/HDSim-AI/hdsim/ci.yml?branch=main&amp;style=flat-square&amp;label=CI" alt="CI"></a>
<a href="https://github.com/HDSim-AI/hdsim/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python 3.10+"></a>
<a href="./LICENSE"><img src="https://img.shields.io/github/license/HDSim-AI/hdsim?style=flat-square" alt="MIT License"></a>
<a href="https://arxiv.org/abs/2604.10475"><img src="https://img.shields.io/badge/arXiv-2604.10475-b31b1b?style=flat-square" alt="Paper"></a>
<a href="https://yushundong.github.io/pemand_simulation/pemand_official_site.html"><img src="https://img.shields.io/badge/Live%20Demo-HDSim-2f7d5f?style=flat-square" alt="Live Demo"></a>
</p>

<!-- Uncomment both once the package is published to PyPI:
<a href="https://pypi.org/project/hdsim/"><img src="https://img.shields.io/pypi/v/hdsim?style=flat-square" alt="PyPI version"></a>
<a href="https://pepy.tech/project/hdsim"><img src="https://static.pepy.tech/badge/hdsim" alt="PyPI downloads"></a>
-->


<img src="./docs/demo.gif" width="100%" alt="hdsim demo replaying a recorded household negotiation that settles on 4 trips">

</div>

---

## 🧭 What can hdsim do?

`hdsim` predicts what a household decides, from the survey records you already have. Feed it rows
from a travel survey or a panel study and it returns a decision for each household, plus the
conversation among the household members that produced it.

- predicts one household at a time, rather than an average over a segment
- returns the transcript behind every number, so a prediction can be inspected and argued with
- answers questions the survey never asked, on the households it already covers
- runs offline from bundled recordings, so you can see the output before configuring anything
- takes a new decision domain as configuration, not as a new pipeline

**Where this helps**

| You are trying to… | What you get |
|---|---|
| Forecast trip generation under a new road price, fare change, or transit line | Per-household trip counts under the scenario you describe |
| Plan for evacuation or post-disaster relocation | Move or stay, household by household |
| Test a policy you cannot field a new survey for | A counterfactual run on households already in your data |
| Fill in a group your survey covers thinly | Decisions for those households, from the records you do have |

**Start here**

| You want to… | Go to |
|---|---|
| Watch a household negotiate, with no API key | [Quick start](#quick-start) |
| Run it against your own model | [Running it live](#running-it-live) |
| Predict household trips | [travel-decision](https://github.com/HDSim-AI/travel-decision) |
| Predict whether a household moves | [residential-mobility](https://github.com/HDSim-AI/residential-mobility) |
| Point it at OpenAI, Ollama, vLLM, or a university proxy | [Configuration](#configuration) |
| Model a decision that is neither travel nor moving | [Adding a domain](#adding-a-domain) |
| See how it scores against classical baselines | [Results](#results) |

## The approach

A household decides together. Classical models treat it as a single unit with a regression
coefficient. This library treats it as several people who hold different information and have to
reconcile it, which is closer to how the decision is actually made.

`hdsim` is the method core of the [HDSim](https://github.com/HDSim-AI) ecosystem. It implements
PEMAND: persona-enriched multi-agent negotiation. Domain packages add the survey loaders and
configuration for a specific decision; the pipeline itself does not change between them.

## Quick start

No API key and no data download:

```bash
git clone https://github.com/HDSim-AI/hdsim && cd hdsim
pip install -e .
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
from hdsim.travel import NHTS, Household, build_personas, simulate   # see travel-decision

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

To run weights locally instead: `pip install -e '.[local]'`.

## Adding a domain

A decision domain is data, not code. Supply a `DomainConfig` and the pipeline runs unchanged.

```python
from hdsim.core import DecisionTask, DomainConfig

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
| PSID 2021–2023 | F1 ↑ | 0.55 (Gradient Boosting) | **0.73** |

Table 1, [arXiv:2604.10475v2](https://arxiv.org/abs/2604.10475). The first four rows are trip
generation, the last is the move-or-stay decision.

## Domain packages

**This repository is the core.** It holds the method and knows nothing about any particular
decision. A domain package adds a survey loader and one `DomainConfig`; the pipeline is unchanged.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/architecture-dark.svg">
  <img src="./docs/architecture-light.svg" width="100%" alt="hdsim is the method core; travel-decision and residential-mobility are domain packages built on it, and a new domain slots in alongside them">
</picture>
</p>

| Package | Repository | Decides |
|---|---|---|
| `hdsim.travel` | [travel-decision](https://github.com/HDSim-AI/travel-decision) | How many trips the household makes tomorrow |
| `hdsim.mobility` | [residential-mobility](https://github.com/HDSim-AI/residential-mobility) | Whether the household moves |

All three install alongside each other: `hdsim` owns `hdsim.core`, and each domain owns its own
subpackage under the same namespace.

## Layout

`hdsim` is a namespace package. This distribution owns `hdsim.core`; the domain packages own
`hdsim.travel` and `hdsim.mobility`, so all three install alongside each other.

```
hdsim/core/
├── household.py    Household and Member, shared by every stage
├── domain.py       DomainConfig, everything that differs between domains
├── persona.py      facts -> capsule -> constructs, with validation
├── stage1.py       the published persona prompts, labels and parsers
├── negotiate.py    parallel proposals, then moderated discussion
├── stage2.py       the published negotiation prompts, labels and parsers
├── backends.py     model access and role configuration
├── replay.py       offline recordings for the demo
├── evaluate.py     metrics and a paired bootstrap
├── cli.py          hdsim demo, hdsim config
└── fixtures/       bundled recordings the offline demo plays
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
