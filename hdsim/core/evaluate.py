"""Metrics.

Scores a set of simulated households against the values the survey recorded. Two things are worth
knowing before reading a number here.

Household counts are small integers with a long right tail, so mean absolute error is easier to
interpret than squared error, and a tolerance accuracy (within one, within two) often says more
than exact match.

Comparing two methods on the same households calls for a paired test. The households differ far
more from each other than the methods differ from each other, so an unpaired comparison mostly
measures which households were sampled. `paired_bootstrap` resamples households, not predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .household import Household


@dataclass
class Scores:
    n: int
    mae: float
    rmse: float
    smape: float
    exact: float
    within_1: float
    within_2: float
    bias: float

    def __str__(self) -> str:
        return (f"n={self.n}  MAE={self.mae:.3f}  RMSE={self.rmse:.3f}  "
                f"sMAPE={self.smape:.2f}  exact={self.exact:.3f}  "
                f"within1={self.within_1:.3f}  within2={self.within_2:.3f}  "
                f"bias={self.bias:+.3f}")


def _pairs(households: Sequence[Household]) -> tuple[list[float], list[float]]:
    predicted, actual = [], []
    for h in households:
        if h.consensus_value is None or h.ground_truth is None:
            continue
        predicted.append(float(h.consensus_value))
        actual.append(float(h.ground_truth))
    return predicted, actual


def score(households: Sequence[Household]) -> Scores:
    """Score simulated households against their recorded values.

    Households missing either a consensus or a ground truth are skipped, and `n` reports how many
    were actually used so a silently shrinking denominator is visible.
    """
    predicted, actual = _pairs(households)
    n = len(predicted)
    if n == 0:
        raise ValueError("No households have both a consensus value and a ground truth.")

    errors = [p - a for p, a in zip(predicted, actual)]
    abs_errors = [abs(e) for e in errors]

    # sMAPE with both terms in the denominator, and pairs where both are zero treated as exact
    # rather than undefined.
    smape_terms = []
    for p, a in zip(predicted, actual):
        denom = (abs(p) + abs(a)) / 2
        smape_terms.append(0.0 if denom == 0 else abs(p - a) / denom)

    return Scores(
        n=n,
        mae=sum(abs_errors) / n,
        rmse=(sum(e * e for e in errors) / n) ** 0.5,
        smape=100 * sum(smape_terms) / n,
        exact=sum(1 for e in abs_errors if e == 0) / n,
        within_1=sum(1 for e in abs_errors if e <= 1) / n,
        within_2=sum(1 for e in abs_errors if e <= 2) / n,
        bias=sum(errors) / n,
    )


def paired_bootstrap(a: Sequence[Household], b: Sequence[Household],
                     metric: str = "mae", n_boot: int = 5000,
                     seed: int = 0) -> dict[str, float]:
    """Compare two methods on the same households.

    `a` and `b` must cover the same household ids. Returns the difference a minus b with a 95
    percent interval. For error metrics a negative difference means `a` is better.
    """
    import random

    index_a = {h.household_id: h for h in a}
    index_b = {h.household_id: h for h in b}
    shared = sorted(set(index_a) & set(index_b))
    if not shared:
        raise ValueError("The two sets share no household ids, so they cannot be paired.")

    def value(households: list[Household]) -> float:
        return getattr(score(households), metric)

    observed = value([index_a[i] for i in shared]) - value([index_b[i] for i in shared])

    rng = random.Random(seed)
    differences = []
    for _ in range(n_boot):
        sample = [shared[rng.randrange(len(shared))] for _ in shared]
        try:
            differences.append(value([index_a[i] for i in sample])
                               - value([index_b[i] for i in sample]))
        except ValueError:
            continue
    differences.sort()

    def percentile(p: float) -> float:
        if not differences:
            return float("nan")
        k = min(len(differences) - 1, max(0, int(p * len(differences))))
        return differences[k]

    return {
        "n_households": len(shared),
        "difference": observed,
        "ci_low": percentile(0.025),
        "ci_high": percentile(0.975),
        "metric": metric,
    }
