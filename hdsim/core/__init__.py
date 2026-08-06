"""hdsim: household decision simulation through multi-agent negotiation.

A household decides together. Classical models treat it as one unit with a coefficient; this
library treats it as several people who each hold different information and have to reconcile it.

The pipeline is four steps:

    survey record  ->  persona      each member gets facts, a roster and behavioural constructs
                   ->  proposal     each member answers alone, in parallel
                   ->  negotiation  the members discuss and correct each other
                   ->  decision     one household level value

Quick start, no API key required:

    hdsim demo

With a key, on your own data:

    from hdsim import Household, build_personas, simulate
    from hdsim.travel import NHTS                    # pip install hdsim-travel

    household = Household.from_json("household.json")
    build_personas(household, NHTS)
    simulate(household, NHTS)
    print(household.consensus_value)

The method is described in arXiv:2604.10475. Domain packages (`hdsim.travel`,
`hdsim.mobility`) supply survey loaders and a `DomainConfig`; everything else lives here.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from . import stage1, stage2
from .domain import DecisionTask, DomainConfig
from .household import Household, Member, load_households
from .negotiate import negotiate, parse_value, propose, simulate
from .persona import build_capsule, build_personas, enrich

__all__ = [
    "DecisionTask",
    "DomainConfig",
    "Household",
    "Member",
    "__version__",
    "build_capsule",
    "build_personas",
    "enrich",
    "load_households",
    "negotiate",
    "parse_value",
    "propose",
    "simulate",
    "stage1",
    "stage2",
]
