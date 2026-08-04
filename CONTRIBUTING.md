# Contributing to HDSim

HDSim exists so that the next household decision use case is cheaper to build than the last one.
Contributions that make that true are the most valuable kind.

## Ways to contribute

| Kind | What it looks like |
|---|---|
| **New decision domain** | High-value purchases, energy use, family planning, evacuation |
| **Survey loader** | Support for another household survey or panel |
| **Evaluation** | Baselines, ablations, or annotation protocols |
| **Scenario** | A replayable household case for the demo |
| **Core improvement** | Persona construction, the negotiation protocol, model backends |

## Adding a decision domain

A domain is configuration, not a new pipeline. Nothing in `hdsim.core` needs to change.

1. Write a `DomainConfig`. It holds what the household is deciding, how survey codes read in
   English, the empirical anchor, and which words a persona may never use.
2. Write a loader that turns survey rows into `Household` and `Member` objects.
3. Set `DecisionTask.value_type`. Use `int` for a count, as travel does, or `bool` for a yes or no,
   as residential mobility does. Both are supported and parsed differently.
4. Set `banned_patterns` to the words that would give the answer away. Persona text is written
   before the household decides, so a persona that states the outcome has already answered the
   question the agents are meant to negotiate over. This is the easiest way to produce a result
   that looks excellent and means nothing.
5. Add `describe_member` and `relate_members` so members can be introduced to each other. Where the
   survey does not determine a relationship, return something weaker and true rather than guessing.
   A roster that invents a relationship causes the confusion it exists to prevent.
6. Evaluate against at least one classical baseline. A domain without a baseline is a demo, not a
   result.

`travel-decision` is the reference implementation. Read it before starting.

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
