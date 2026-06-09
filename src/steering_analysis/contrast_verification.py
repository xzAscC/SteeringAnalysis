"""Contrast pair verification experiment.

Computes cosine similarity between the positive/negative text hidden states
(used to build the steering vector) and the steering vector itself, then
replicates Assumption 1's Exp1 (token-level) and Exp2 (layer-level).

This validates that the steering vector directionally separates the positive
and negative texts it was constructed from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from .assumption_verification import (
    LayerExistenceVerdict,
    TokenLevelVerdict,
    compute_empirical_thresholds,
    get_all_layer_activations,
    run_experiment1_token_level,
    run_experiment2_layer_existence,
)
from .config import ExtractionConfig, VerificationConfig
from .extract import extract_steering_vector, load_contrast_pairs
from .models import HookedModel


@dataclass
class ContrastLayerResult:
    """Results for one steering layer in the contrast verification experiment."""

    steering_layer: int
    cos_matrix_positive: Tensor  # (num_layers, total_pos_tokens)
    cos_matrix_negative: Tensor  # (num_layers, total_neg_tokens)
    experiment1_verdicts: dict[float, list[TokenLevelVerdict]] = field(default_factory=dict)
    experiment2_verdicts: dict[float, LayerExistenceVerdict | None] = field(default_factory=dict)
    empirical_thresholds: dict[float, float] = field(default_factory=dict)


@dataclass
class ContrastVerificationResult:
    """Full contrast verification results across all steering layers."""

    model_name: str
    concept: str
    steering_layers_tested: list[int]
    per_layer_results: dict[int, ContrastLayerResult] = field(default_factory=dict)


def compute_pair_cosine_matrices(
    positive_activations: list[dict[int, Tensor]],
    negative_activations: list[dict[int, Tensor]],
    steering_vector: Tensor,
    num_layers: int,
) -> tuple[Tensor, Tensor]:
    """Compute cosine similarity matrices for positive and negative text activations.

    Args:
        positive_activations: List of {layer: (1, seq_len, hidden_dim)} dicts, one per positive text.
        negative_activations: Same structure for negative texts.
        steering_vector: The steering vector (hidden_dim,).
        num_layers: Total number of model layers.

    Returns:
        (pos_cos, neg_cos) each shaped (num_layers, total_tokens).

    Raises:
        ValueError: If activations are empty or counts mismatch.
    """
    if not positive_activations or not negative_activations:
        raise ValueError("No activations provided")
    if len(positive_activations) != len(negative_activations):
        raise ValueError(
            f"Mismatched positive/negative activation counts: "
            f"{len(positive_activations)} vs {len(negative_activations)}"
        )

    # Steering vector shaped for broadcasting: (1, 1, hidden_dim)
    sv = steering_vector.unsqueeze(0).unsqueeze(0)

    all_pos_rows = []
    all_neg_rows = []

    for layer_idx in range(num_layers):
        pos_cosines = []
        neg_cosines = []

        for pos_act, neg_act in zip(positive_activations, negative_activations):
            pos_h = pos_act[layer_idx]  # (1, seq_len, hidden_dim)
            neg_h = neg_act[layer_idx]  # (1, seq_len, hidden_dim)

            # Cosine similarity per token: (1, seq_len)
            pos_cos = functional.cosine_similarity(pos_h.float(), sv.float().expand_as(pos_h), dim=-1)
            neg_cos = functional.cosine_similarity(neg_h.float(), sv.float().expand_as(neg_h), dim=-1)

            pos_cosines.append(pos_cos.squeeze(0))  # (seq_len,)
            neg_cosines.append(neg_cos.squeeze(0))  # (seq_len,)

        all_pos_rows.append(torch.cat(pos_cosines, dim=0))  # (total_tokens,)
        all_neg_rows.append(torch.cat(neg_cosines, dim=0))  # (total_tokens,)

    pos_matrix = torch.stack(all_pos_rows)  # (num_layers, total_tokens)
    neg_matrix = torch.stack(all_neg_rows)  # (num_layers, total_tokens)

    return pos_matrix, neg_matrix


def run_contrast_verification(
    model: HookedModel,
    concept: str,
    config: VerificationConfig,
) -> ContrastVerificationResult:
    """Run contrast pair verification experiment.

    For each steering layer:
    1. Build steering vector from contrast pairs.
    2. Extract hidden states from positive and negative texts at all layers.
    3. Compute cosine similarity between those hidden states and the steering vector.
    4. Run Exp1 (token-level) and Exp2 (layer-level) comparing positive vs negative.
    """
    pairs = load_contrast_pairs(concept, config.extraction_num_pairs, seed=config.seed)

    extraction_config = ExtractionConfig(
        layers=config.extraction_layers,
        method=config.extraction_method,
        num_pairs=config.extraction_num_pairs,
        seed=config.seed,
    )
    steering_vector = extract_steering_vector(model, pairs, extraction_config)

    steering_layers = sorted(steering_vector.layer_activations.keys())

    num_samples = min(config.num_samples, len(pairs))
    sample_pairs = pairs[:num_samples]

    # Extract hidden states once — they don't depend on the steering layer
    positive_activations: list[dict[int, Tensor]] = []
    negative_activations: list[dict[int, Tensor]] = []
    for pair in sample_pairs:
        pos_act = get_all_layer_activations(model, pair.positive)
        neg_act = get_all_layer_activations(model, pair.negative)
        positive_activations.append(pos_act)
        negative_activations.append(neg_act)

    per_layer_results: dict[int, ContrastLayerResult] = {}

    for s_layer in steering_layers:
        steering_vec = steering_vector.layer_activations[s_layer]

        # Compute cosine similarity matrices
        cos_matrix_positive, cos_matrix_negative = compute_pair_cosine_matrices(
            positive_activations,
            negative_activations,
            steering_vec,
            model.num_layers,
        )

        # Run Exp1 and Exp2 (positive = "steered", negative = "unsteered")
        exp1_verdicts: dict[float, list[TokenLevelVerdict]] = {}
        exp2_verdicts: dict[float, LayerExistenceVerdict | None] = {}

        for threshold in config.thresholds:
            exp1_verdicts[threshold] = run_experiment1_token_level(cos_matrix_positive, cos_matrix_negative, threshold)
            exp2_verdicts[threshold] = run_experiment2_layer_existence(
                cos_matrix_positive, cos_matrix_negative, threshold
            )

        # Compute empirical thresholds from negative (unsteered) distribution
        empirical_thresholds = compute_empirical_thresholds(cos_matrix_negative, config.empirical_percentiles)

        per_layer_results[s_layer] = ContrastLayerResult(
            steering_layer=s_layer,
            cos_matrix_positive=cos_matrix_positive,
            cos_matrix_negative=cos_matrix_negative,
            experiment1_verdicts=exp1_verdicts,
            experiment2_verdicts=exp2_verdicts,
            empirical_thresholds=empirical_thresholds,
        )

    return ContrastVerificationResult(
        model_name=model.config.model_name,
        concept=concept,
        steering_layers_tested=steering_layers,
        per_layer_results=per_layer_results,
    )


def save_contrast_results(
    result: ContrastVerificationResult,
    output_dir: Path,
    label: str,
) -> None:
    """Save contrast verification results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cosine matrices as .pt
    matrices: dict[str, Tensor] = {}
    for s_layer, lr in result.per_layer_results.items():
        matrices[f"positive_L{s_layer}"] = lr.cos_matrix_positive
        matrices[f"negative_L{s_layer}"] = lr.cos_matrix_negative
    torch.save(matrices, output_dir / f"{label}_cosine_matrices.pt")

    # Save summary JSON
    summary: dict[str, Any] = {
        "model_name": result.model_name,
        "concept": result.concept,
        "steering_layers_tested": result.steering_layers_tested,
        "per_layer_summary": {},
    }
    for s_layer, lr in result.per_layer_results.items():
        summary["per_layer_summary"][str(s_layer)] = {
            "steering_layer": s_layer,
            "empirical_thresholds": lr.empirical_thresholds,
            "experiment1_holds_count": {},
            "experiment2_holds": {},
        }
        for thresh, verdicts in lr.experiment1_verdicts.items():
            holds = sum(1 for v in verdicts if v.assumption_holds)
            summary["per_layer_summary"][str(s_layer)]["experiment1_holds_count"][str(thresh)] = {
                "holds": holds,
                "total": len(verdicts),
            }
        for thresh, verdict in lr.experiment2_verdicts.items():
            if verdict is not None:
                summary["per_layer_summary"][str(s_layer)]["experiment2_holds"][str(thresh)] = {
                    "holds": verdict.assumption_holds,
                    "positive_layers_above": verdict.steered_layers_above,
                    "negative_layers_above": verdict.unsteered_layers_above,
                }

    with open(output_dir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results JSON
    full_data: dict[str, Any] = {
        "model_name": result.model_name,
        "concept": result.concept,
        "steering_layers_tested": result.steering_layers_tested,
        "per_layer_results": {},
    }
    for s_layer, lr in result.per_layer_results.items():
        layer_data: dict[str, Any] = {
            "steering_layer": s_layer,
            "cos_matrix_positive": lr.cos_matrix_positive.tolist(),
            "cos_matrix_negative": lr.cos_matrix_negative.tolist(),
            "empirical_thresholds": lr.empirical_thresholds,
            "experiment1_token_level": {
                str(thresh): [
                    {
                        "layer_idx": v.layer_idx,
                        "threshold": v.threshold,
                        "steered_some_above": v.steered_some_above,
                        "steered_all_above": v.steered_all_above,
                        "steered_fraction_above": v.steered_fraction_above,
                        "unsteered_all_below": v.unsteered_all_below,
                        "unsteered_max_cosine": v.unsteered_max_cosine,
                        "assumption_holds": v.assumption_holds,
                        "verdict": v.verdict,
                    }
                    for v in verdicts
                ]
                for thresh, verdicts in lr.experiment1_verdicts.items()
            },
            "experiment2_layer_existence": {
                str(thresh): {
                    "threshold": v.threshold,
                    "exists_positive_layer": v.exists_steered_layer,
                    "exists_negative_layer": v.exists_unsteered_layer,
                    "positive_layers_above": v.steered_layers_above,
                    "negative_layers_above": v.unsteered_layers_above,
                    "assumption_holds": v.assumption_holds,
                }
                for thresh, v in lr.experiment2_verdicts.items()
                if v is not None
            },
        }
        full_data["per_layer_results"][str(s_layer)] = layer_data

    with open(output_dir / f"{label}_full_results.json", "w") as f:
        json.dump(full_data, f, indent=2)
