from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor


class ContrastPairMetadata(TypedDict, total=False):
    concept: str
    dataset: str
    source: str
    pair_index: int
    original_question: str


@dataclass
class ContrastPair:
    positive: str
    negative: str
    metadata: ContrastPairMetadata


@dataclass
class SteeringVector:
    layer_activations: dict[int, Tensor]
    model_name: str
    concept: str
    method: str
