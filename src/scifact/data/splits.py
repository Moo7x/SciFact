"""Deterministic splits.

`train` is divided into `train_fit` and `train_tune`. Every hyperparameter is chosen on
`train_tune`. The official `dev` split is the holdout and is touched once per stage.

Stratified on whether a claim carries evidence (37.6% do not), so both halves keep the same
abstention balance. Seed is fixed and recorded; the split is a pure function of it.
"""

from __future__ import annotations

import random

from scifact.data.schema import Claim

SPLIT_SEED = 20260919
TUNE_FRACTION = 0.20


def split_train(
    claims: list[Claim], seed: int = SPLIT_SEED, tune_fraction: float = TUNE_FRACTION
) -> tuple[list[Claim], list[Claim]]:
    """Return (train_fit, train_tune), stratified by whether the claim has evidence."""
    rng = random.Random(seed)  # noqa: S311

    fit: list[Claim] = []
    tune: list[Claim] = []
    for has_ev in (True, False):
        group = sorted((c for c in claims if c.has_evidence is has_ev), key=lambda c: c.id)
        rng.shuffle(group)
        cut = round(len(group) * tune_fraction)
        tune.extend(group[:cut])
        fit.extend(group[cut:])

    fit.sort(key=lambda c: c.id)
    tune.sort(key=lambda c: c.id)
    return fit, tune
