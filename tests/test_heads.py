"""Tests for aligning an NLI head with this project's label order."""

from __future__ import annotations

import pytest
import torch

from scifact.verify.heads import nli_permutation, output_layer, permute_rows


def test_permutation_for_the_common_nli_order() -> None:
    # The order used by every NLI candidate in ADR-0002.
    perm = nli_permutation({0: "contradiction", 1: "entailment", 2: "neutral"})
    # Our order is SUPPORT, CONTRADICT, NEI = entailment, contradiction, neutral.
    assert perm == [1, 0, 2]


def test_permutation_is_case_and_key_type_insensitive() -> None:
    assert nli_permutation({"0": "ENTAILMENT", "1": "Neutral", "2": "contradiction"}) == [0, 2, 1]


def test_non_nli_heads_are_left_alone() -> None:
    assert nli_permutation({0: "LABEL_0"}) is None
    assert nli_permutation({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}) is None


def test_permute_rows_moves_whole_rows_including_bias() -> None:
    layer = torch.nn.Linear(4, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(12, dtype=torch.float).reshape(3, 4))
        layer.bias.copy_(torch.tensor([10.0, 20.0, 30.0]))
    permute_rows(layer, [1, 0, 2])
    assert layer.weight[0].tolist() == [4.0, 5.0, 6.0, 7.0]
    assert layer.weight[1].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert layer.bias.tolist() == [20.0, 10.0, 30.0]


def test_permuted_head_gives_the_same_scores_in_the_new_order() -> None:
    """The point of the fix: same knowledge, relabelled -- not a different model."""
    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 3)
    x = torch.randn(5, 8)
    before = layer(x)
    permute_rows(layer, [1, 0, 2])
    after = layer(x)
    assert torch.allclose(after[:, 0], before[:, 1])
    assert torch.allclose(after[:, 1], before[:, 0])
    assert torch.allclose(after[:, 2], before[:, 2])


def test_output_layer_finds_a_nested_head() -> None:
    class RobertaStyleHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = torch.nn.Linear(8, 8)
            self.out_proj = torch.nn.Linear(8, 3)

    head = RobertaStyleHead()
    assert output_layer(head) is head.out_proj
    with pytest.raises(ValueError, match="3 outputs"):
        output_layer(torch.nn.Linear(8, 1))
