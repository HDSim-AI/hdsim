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
    import math

    predicted, actual = [], []
    for h in households:
        if h.consensus_value is None or h.ground_truth is None:
            continue
        # A loader that summed a blank cell yields NaN. It compares false against everything, so
        # it survives every guard above and turns MAE into nan while `n` still counts the
        # household. Drop it here rather than report a metric nobody can read.
        if not (math.isfinite(float(h.consensus_value)) and math.isfinite(float(h.ground_truth))):
            continue
        predicted.append(float(h.consensus_value))
        actual.append(float(h.ground_truth))
    return predicted, actual


@dataclass
class BinaryScores:
    """Scores for a yes or no decision, such as whether a household moves."""

    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        return (f"n={self.n}  accuracy={self.accuracy:.3f}  precision={self.precision:.3f}  "
                f"recall={self.recall:.3f}  F1={self.f1:.3f}")


def is_binary(households: Sequence[Household]) -> bool:
    """True when the decision is a yes or no rather than a count.

    A survey codes a yes or no as 1 and 0, so a set can arrive with `True` predictions against
    integer ground truth, or as 0/1 throughout. Testing for `bool` alone misses both, and missing
    them is not harmless: the count metrics then report within_1 = within_2 = 1.0 on data where
    those numbers cannot mean anything.
    """
    values = [h.ground_truth for h in households if h.ground_truth is not None]
    values += [h.consensus_value for h in households if h.consensus_value is not None]
    if not values:
        return False
    return all(isinstance(v, bool) or (isinstance(v, (int, float)) and v in (0, 1))
               for v in values)


def score_binary(households: Sequence[Household]) -> BinaryScores:
    """Score a yes or no decision.

    Precision, recall and F1 all treat yes as the positive class, because the interesting error in
    a relocation study is missing a household that moves, not one that stays.
    """
    # bool() so a set coded 1/0 scores the same as one using True/False
    pairs = [(bool(h.consensus_value), bool(h.ground_truth)) for h in households
             if h.consensus_value is not None and h.ground_truth is not None]
    n = len(pairs)
    if n == 0:
        raise ValueError("No households have both a consensus value and a ground truth.")

    tp = sum(1 for p, a in pairs if p and a)
    fp = sum(1 for p, a in pairs if p and not a)
    fn = sum(1 for p, a in pairs if not p and a)
    tn = sum(1 for p, a in pairs if not p and not a)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryScores(n=n, accuracy=(tp + tn) / n, precision=precision, recall=recall, f1=f1)


def score(households: Sequence[Household]) -> Scores:
    """Score simulated households against their recorded values.

    Households missing either a consensus or a ground truth are skipped, and `n` reports how many
    were actually used so a silently shrinking denominator is visible.

    Counts only. A yes or no decision is refused rather than scored, because these metrics look
    fine on one and mean nothing: `within_1` and `within_2` are 1.0 for every possible 0/1
    prediction, including one that gets every household wrong.
    """
    if is_binary(households):
        raise TypeError(
            "These households decided yes or no, not a count. MAE, sMAPE and the within-N "
            "counts do not measure anything on a 0/1 outcome: within_1 and within_2 are 1.0 "
            "however wrong the prediction is. Use score_binary() instead."
        )
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

    scorer = score_binary if is_binary(list(a) + list(b)) else score
    if metric == "mae" and scorer is score_binary:
        metric = "f1"        # the default only makes sense for counts

    def value(households: list[Household]) -> float:
        return getattr(scorer(households), metric)

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
