from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from .config import ExtractionConfig, VerificationConfig
from .extract import extract_steering_vector, load_contrast_pairs
from .models import HookedModel


@dataclass
class LayerThresholdResult:
    layer_idx: int
    fraction_above: float
    all_above: bool
    max_cosine: float
    mean_cosine: float
    threshold: float


@dataclass
class VerificationResult:
    model_name: str
    concept: str
    threshold_results: list[LayerThresholdResult]
    empirical_thresholds: dict[float, float]
    cos_matrix_steered: Tensor
    cos_matrix_unsteered: Tensor


def get_all_layer_activations(model: HookedModel, text: str) -> dict[int, Tensor]:
    """Extract hidden states from all layers for a single text input."""
    inputs = model.tokenizer(text, return_tensors="pt")
    device = next(model.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    activations: dict[int, Tensor] = {}
    handles = []
    model_layers = model._get_layers_module()

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            tensor_output = output[0] if isinstance(output, tuple) else output
            activations[layer_idx] = tensor_output.detach().clone()
        return hook_fn

    for layer_idx in range(model.num_layers):
        handle = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    try:
        with torch.no_grad():
            _ = model.model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return activations


def compute_cosine_similarities(
    layer_activations: dict[int, Tensor], concept_vector: Tensor
) -> dict[int, Tensor]:
    """Compute per-token cosine similarity between each layer's hidden states and a concept vector."""
    result = {}
    concept = concept_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
    for layer_idx, activations in layer_activations.items():
        # activations: (1, seq_len, hidden_dim), concept: (1, 1, hidden_dim)
        cos = functional.cosine_similarity(activations, concept.expand_as(activations), dim=-1)
        result[layer_idx] = cos.squeeze(0)  # (seq_len,)
    return result


def compute_threshold_violations(
    cos_matrix: Tensor, threshold: float
) -> list[LayerThresholdResult]:
    """Analyze threshold violations per layer from a (num_layers, seq_len) cosine matrix."""
    results = []
    for layer_idx in range(cos_matrix.shape[0]):
        layer_cos = cos_matrix[layer_idx]
        above = (layer_cos >= threshold).float()
        fraction_above = above.mean().item()
        all_above = bool((layer_cos >= threshold).all().item())
        max_cosine = layer_cos.max().item()
        mean_cosine = layer_cos.mean().item()
        results.append(LayerThresholdResult(
            layer_idx=layer_idx,
            fraction_above=fraction_above,
            all_above=all_above,
            max_cosine=max_cosine,
            mean_cosine=mean_cosine,
            threshold=threshold,
        ))
    return results


def compute_empirical_thresholds(
    unsteered_cosines: Tensor, percentiles: list[float]
) -> dict[float, float]:
    """Compute percentile thresholds from the flattened unsteered cosine distribution."""
    flat = unsteered_cosines.flatten().float().cpu().numpy()
    import numpy as np
    return {p: float(np.percentile(flat, p)) for p in percentiles}


def run_verification(
    model: HookedModel, concept: str, config: VerificationConfig
) -> VerificationResult:
    """Run full verification pipeline: extract steering vector, generate steered/unsteered text, compare."""
    pairs = load_contrast_pairs(concept, config.extraction_num_pairs, seed=config.seed)

    extraction_config = ExtractionConfig(
        layers=config.extraction_layers,
        method=config.extraction_method,
        num_pairs=config.extraction_num_pairs,
        seed=config.seed,
    )
    steering_vector = extract_steering_vector(model, pairs, extraction_config)

    # Find layer with highest-norm steering vector
    best_layer = max(
        steering_vector.layer_activations.keys(),
        key=lambda k: steering_vector.layer_activations[k].norm().item(),
    )
    steering_vec = steering_vector.layer_activations[best_layer]

    # Compute average activation norm at the best layer for scaling
    all_texts = [p.positive for p in pairs] + [p.negative for p in pairs]
    avg_activations = model.get_activations(all_texts, [best_layer])
    avg_norm = avg_activations[best_layer].norm(dim=-1).mean().item()
    scale = config.steering_multiplier * avg_norm

    prompts = [p.negative for p in pairs[: config.num_samples]]

    all_steered_cos = []
    all_unsteered_cos = []

    for prompt in prompts:
        steered_text = model.generate_with_steering(
            prompt, best_layer, steering_vec, scale,
            max_new_tokens=config.max_new_tokens, temperature=config.temperature,
        )
        unsteered_text = model.generate_with_steering(
            prompt, best_layer, steering_vec, scale=0.0,
            max_new_tokens=config.max_new_tokens, temperature=config.temperature,
        )

        steered_activations = get_all_layer_activations(model, steered_text)
        unsteered_activations = get_all_layer_activations(model, unsteered_text)

        # Use the steering vector at best_layer as concept vector
        concept_vec = steering_vector.layer_activations[best_layer]

        steered_cos = compute_cosine_similarities(steered_activations, concept_vec)
        unsteered_cos = compute_cosine_similarities(unsteered_activations, concept_vec)

        num_layers = model.num_layers
        steered_row = torch.stack([steered_cos[i] for i in range(num_layers)])
        unsteered_row = torch.stack([unsteered_cos[i] for i in range(num_layers)])

        all_steered_cos.append(steered_row)
        all_unsteered_cos.append(unsteered_row)

    cos_matrix_steered = torch.cat(all_steered_cos, dim=1)
    cos_matrix_unsteered = torch.cat(all_unsteered_cos, dim=1)

    threshold_results = []
    for threshold in config.thresholds:
        violations = compute_threshold_violations(cos_matrix_steered, threshold)
        threshold_results.extend(violations)

    empirical_thresholds = compute_empirical_thresholds(
        cos_matrix_unsteered, config.empirical_percentiles
    )

    return VerificationResult(
        model_name=model.config.model_name,
        concept=concept,
        threshold_results=threshold_results,
        empirical_thresholds=empirical_thresholds,
        cos_matrix_steered=cos_matrix_steered,
        cos_matrix_unsteered=cos_matrix_unsteered,
    )


def save_results(result: VerificationResult, output_dir: Path, label: str) -> None:
    """Save verification results to disk: .pt for tensors, .json for summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"steered": result.cos_matrix_steered, "unsteered": result.cos_matrix_unsteered},
        output_dir / f"{label}_cosine_matrices.pt",
    )

    per_threshold = {}
    for r in result.threshold_results:
        key = str(r.threshold)
        if key not in per_threshold:
            per_threshold[key] = {"fraction_above": [], "all_above_count": 0}
        per_threshold[key]["fraction_above"].append(r.fraction_above)
        if r.all_above:
            per_threshold[key]["all_above_count"] += 1

    summary_analysis = {}
    for key, vals in per_threshold.items():
        summary_analysis[key] = {
            "mean_fraction_above": sum(vals["fraction_above"]) / len(vals["fraction_above"]),
            "all_above_count": vals["all_above_count"],
        }

    summary = {
        "model_name": result.model_name,
        "concept": result.concept,
        "per_threshold_analysis": summary_analysis,
        "empirical_thresholds": result.empirical_thresholds,
    }
    with open(output_dir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    full_data: dict[str, Any] = {
        "model_name": result.model_name,
        "concept": result.concept,
        "threshold_results": [
            {
                "layer_idx": r.layer_idx,
                "fraction_above": r.fraction_above,
                "all_above": r.all_above,
                "max_cosine": r.max_cosine,
                "mean_cosine": r.mean_cosine,
                "threshold": r.threshold,
            }
            for r in result.threshold_results
        ],
        "empirical_thresholds": result.empirical_thresholds,
        "cos_matrix_steered": result.cos_matrix_steered.tolist(),
        "cos_matrix_unsteered": result.cos_matrix_unsteered.tolist(),
    }
    with open(output_dir / f"{label}_full_results.json", "w") as f:
        json.dump(full_data, f, indent=2)
