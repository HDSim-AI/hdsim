# Contributing to HDSim

HDSim exists so that the next household decision use case is cheaper to build than the last one.

**Start by finding yourself here.**

| You want to… | Go to |
|---|---|
| Model a decision nobody has built yet | [Add a new domain](#add-a-new-domain) |
| Improve trips or move-or-stay | [Improve an existing domain](#improve-an-existing-domain) |
| Support another survey or panel | [Improve an existing domain](#improve-an-existing-domain) |
| Change persona construction, the negotiation, or a backend | [Change the core](#change-the-core) |
| Report something wrong | [Open an issue](https://github.com/HDSim-AI/hdsim/issues) |

---

## Add a new domain

A domain is configuration, not a new pipeline. Nothing in `hdsim.core` changes.

**Copy [`examples/minimal_domain.py`](examples/minimal_domain.py) and change six things.** It is a
complete working domain in one file and it runs with no API key, so you can see the shape before
committing to anything:

```bash
python examples/minimal_domain.py
```

The six places are marked in the file:

1. **What is being decided** — a `DecisionTask`. `value_type=int` for a count, as travel does, or
   `bool` for a yes or no, as residential mobility does. Both are parsed, differently.
2. **How a survey row reads in English** — one entry per column, mapped through a dict or a
   function.
3. **What a persona may never say** — `banned_patterns`. Persona text is written before the
   household decides, so a persona that names the outcome has already answered the question the
   agents are meant to negotiate. **This is the easiest way to produce a result that looks
   excellent and means nothing.**
4. **How members are introduced to each other** — `describe_member` and `relate_members`. Where the
   survey does not establish a relationship, return something weaker and true rather than guessing.
   A roster that invents a relationship causes the confusion it exists to prevent.
5. **The Chain-of-Planned-Behaviour prompts** — `copb_system` and `copb_user`. Core ships no
   default on purpose: a domain running another domain's prompt gives a confident wrong answer with
   nothing to flag it.
6. **The household roster** — `household_roster`. Leave it unset and the negotiation uses the
   core's, which recovers attributes by matching persona text against travel-survey phrasings. On
   any other domain those miss and the household is told "non-worker, non-driver, age ?" **as
   canonical fact**, contradicting the personas you just wrote. Print it once before trusting a run.

Then two more things before it is a domain rather than a demo:

- **A loader** that turns real survey rows into `Household` and `Member` objects. See
  [`hdsim/travel/loaders.py`](https://github.com/HDSim-AI/travel-decision/blob/main/hdsim/travel/loaders.py).
  Give every member a distinct `person_id`, and set `ground_truth` to `None` rather than to a guess
  when the survey does not answer.
- **At least one classical baseline.** A domain without a baseline is a demo, not a result. Score it
  with `from hdsim.core import score` for a count, or `score_binary` for a yes or no.

[`travel-decision`](https://github.com/HDSim-AI/travel-decision) is the reference implementation and
the best thing to read once the minimal example makes sense.

## Improve an existing domain

Work in that domain's repository, not here. Loaders for another survey, better fact translations,
new scenarios and evaluations are all welcome, and none of them require touching the core.

- [`travel-decision`](https://github.com/HDSim-AI/travel-decision) — trips
- [`residential-mobility`](https://github.com/HDSim-AI/residential-mobility) — move or stay

## Change the core

Persona construction, the negotiation protocol and the model backends live here. Two of these files
are the published method rather than ordinary code: `hdsim/core/stage1.py` and
`hdsim/core/stage2.py` carry the prompt wording, the label scheme and the parsers exactly as
published, and are excluded from lint for that reason. Changing them changes results, so open an
issue first.

## Data

Do not commit survey microdata. Most household surveys carry redistribution terms and PSID requires
registration. Ship a loader and download instructions, plus a small synthetic example so the package
runs before anyone downloads anything.

## Pull requests

- Branch from `main`, one logical change per pull request.
- Include tests. They must pass with no API key and no network.
- If you change persona construction or the negotiation protocol, re-run the evaluation and include
  the numbers. Those changes move published metrics.

## Reproducibility

Record the model and version behind any reported result. Results produced by a language model are
not reproducible without it, and reproducibility is the point of this project.

## Questions and comments

If something is unclear, or you think a design choice is wrong, or you want to talk through a use
case before writing code, send an email to mustafasameen@ufl.edu. We would rather have the
conversation than have you guess, and we are happy to talk through anything in the method.

## Code of conduct

Be decent to each other. Research infrastructure is a long game and the community is small.
