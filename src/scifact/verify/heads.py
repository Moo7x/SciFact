"""Aligning a pretrained 3-way head with this project's label order.

NLI models arrive with a head that already distinguishes entailment, contradiction and neutral,
which map onto SUPPORT, CONTRADICT and NOT_ENOUGH_INFO. But each model stores those three outputs
in its own order -- for the candidates in ADR-0002 it is (contradiction, entailment, neutral) --
while this project's labels are (SUPPORT, CONTRADICT, NEI) = (entailment, contradiction, neutral).

If the order is not aligned, training still runs and the loss still falls. Nothing crashes. But
row 0 of the head, which learned "contradiction" in pretraining, is now trained as SUPPORT: the
model must first unlearn its own head before it can learn anything else, and the zero-shot
advantage that motivated the switch is thrown away. A silent bug of the same class as the
`.get()` default in schema.py: plausible output, wrong meaning.

The fix is to permute the head's output rows once, at load time, so row i means our label i.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

# Our label id -> the NLI label that means the same thing.
OUR_ORDER_AS_NLI: tuple[str, str, str] = ("entailment", "contradiction", "neutral")


def nli_permutation(id2label: Mapping[Any, str]) -> list[int] | None:
    """Rows to take from an NLI head, in our label order. None if the head is not NLI-shaped.

    `perm[i]` is the index of the model's output that should become our label i.
    """
    normalised = {int(k): str(v).strip().lower() for k, v in id2label.items()}
    if len(normalised) != 3 or set(normalised.values()) != set(OUR_ORDER_AS_NLI):
        return None
    by_name = {name: idx for idx, name in normalised.items()}
    return [by_name[name] for name in OUR_ORDER_AS_NLI]


def output_layer(classifier: torch.nn.Module) -> torch.nn.Linear:
    """The final Linear with 3 outputs inside a model's classifier, whatever its structure.

    BERT and DeBERTa use a single Linear; RoBERTa wraps it as `classifier.out_proj`.
    """
    linears = [m for m in classifier.modules() if isinstance(m, torch.nn.Linear)]
    candidates = [m for m in linears if m.out_features == 3]
    if not candidates:
        raise ValueError("classifier has no Linear layer with 3 outputs")
    return candidates[-1]


@torch.no_grad()
def permute_rows(layer: torch.nn.Linear, perm: list[int]) -> None:
    """Reorder a Linear layer's outputs in place: new row i = old row perm[i]."""
    index = torch.tensor(perm, device=layer.weight.device)
    layer.weight.copy_(layer.weight.index_select(0, index))
    if layer.bias is not None:
        layer.bias.copy_(layer.bias.index_select(0, index))
