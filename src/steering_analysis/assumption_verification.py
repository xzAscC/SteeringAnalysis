from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
class SteeringLayerResult:
    steering_layer: int
    threshold_results: list[LayerThresholdResult]
    empirical_thresholds: dict[float, float]
    cos_matrix_steered: Tensor
    cos_matrix_unsteered: Tensor


@dataclass
class VerificationResult:
    model_name: str
    concept: str
    steering_layers_tested: list[int]
    per_layer_results: dict[int, SteeringLayerResult] = field(default_factory=dict)


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


def get_steered_activations(
    model: HookedModel,
    text: str,
    layer_idx: int,
    steering_vector: Tensor,
    scale: float,
) -> dict[int, Tensor]:
    """Forward pass over text with steering hook active at layer_idx, capturing all layers.

    The steering hook is registered FIRST so capture hooks see the steered output.
    """
    inputs = model.tokenizer(text, return_tensors="pt")
    device = next(model.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    activations: dict[int, Tensor] = {}
    handles = []
    model_layers = model._get_layers_module()
    sv = steering_vector.to(device=device)

    def steering_hook(module, input, output):
        tensor_output = output[0] if isinstance(output, tuple) else output
        perturbed = tensor_output + sv.to(dtype=tensor_output.dtype) * scale
        if isinstance(output, tuple):
            return (perturbed,) + output[1:]
        return perturbed

    handle_steer = model_layers[layer_idx].register_forward_hook(steering_hook)
    handles.append(handle_steer)

    def make_capture(idx: int):
        def hook_fn(module, input, output):
            tensor_output = output[0] if isinstance(output, tuple) else output
            activations[idx] = tensor_output.detach().clone()
        return hook_fn

    for li in range(model.num_layers):
        handle = model_layers[li].register_forward_hook(make_capture(li))
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
    concept = concept_vector.unsqueeze(0).unsqueeze(0)
    for layer_idx, activations in layer_activations.items():
        cos = functional.cosine_similarity(activations, concept.expand_as(activations), dim=-1)
        result[layer_idx] = cos.squeeze(0)
    return result


def compute_threshold_violations(
    cos_matrix: Tensor, threshold: float
) -> list[LayerThresholdResult]:
    """Analyze threshold violations per layer from a (num_layers, seq_len) cosine matrix."""
    results = []
    for layer_idx in range(cos_matrix.shape[0]):
        layer_cos = cos_matrix[layer_idx]
        if layer_cos.numel() == 0:
            results.append(LayerThresholdResult(
                layer_idx=layer_idx, fraction_above=0.0, all_above=False,
                max_cosine=0.0, mean_cosine=0.0, threshold=threshold,
            ))
            continue
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
    if flat.size == 0:
        return {p: 0.0 for p in percentiles}
    return {p: float(np.percentile(flat, p)) for p in percentiles}


def run_verification(
    model: HookedModel, concept: str, config: VerificationConfig
) -> VerificationResult:
    """Run full verification pipeline across ALL extracted steering layers.

    For each steering layer:
    1. Generate steered text (steering at that layer)
    2. Generate unsteered text (no steering)
    3. Forward-pass with steering hook active -> steered hidden states
    4. Forward-pass without steering -> unsteered hidden states
    5. Compute cosine similarities and threshold analysis
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

    all_texts = [p.positive for p in pairs] + [p.negative for p in pairs]
    avg_activations = model.get_activations(all_texts, steering_layers)
    avg_norms = {li: avg_activations[li].norm(dim=-1).mean().item() for li in steering_layers}

    prompts = [p.negative for p in pairs[: config.num_samples]]

    per_layer_results: dict[int, SteeringLayerResult] = {}

    for s_layer in steering_layers:
        steering_vec = steering_vector.layer_activations[s_layer]
        scale = config.steering_multiplier * avg_norms[s_layer]

        all_steered_cos = []
        all_unsteered_cos = []

        for prompt in prompts:
            steered_text = model.generate_with_steering(
                prompt, s_layer, steering_vec, scale,
                max_new_tokens=config.max_new_tokens, temperature=config.temperature,
            )
            unsteered_text = model.generate_with_steering(
                prompt, s_layer, steering_vec, scale=0.0,
                max_new_tokens=config.max_new_tokens, temperature=config.temperature,
            )

            steered_act = get_steered_activations(model, steered_text, s_layer, steering_vec, scale)
            unsteered_act = get_all_layer_activations(model, unsteered_text)

            concept_vec = steering_vector.layer_activations[s_layer]

            steered_cos = compute_cosine_similarities(steered_act, concept_vec)
            unsteered_cos = compute_cosine_similarities(unsteered_act, concept_vec)

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

        per_layer_results[s_layer] = SteeringLayerResult(
            steering_layer=s_layer,
            threshold_results=threshold_results,
            empirical_thresholds=empirical_thresholds,
            cos_matrix_steered=cos_matrix_steered,
            cos_matrix_unsteered=cos_matrix_unsteered,
        )

    return VerificationResult(
        model_name=model.config.model_name,
        concept=concept,
        steering_layers_tested=steering_layers,
        per_layer_results=per_layer_results,
    )


def save_results(result: VerificationResult, output_dir: Path, label: str) -> None:
    """Save verification results to disk: .pt for tensors, .json for summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_steered = {}
    all_unsteered = {}
    for s_layer, lr in result.per_layer_results.items():
        all_steered[f"steered_L{s_layer}"] = lr.cos_matrix_steered
        all_unsteered[f"unsteered_L{s_layer}"] = lr.cos_matrix_unsteered
    torch.save({**all_steered, **all_unsteered}, output_dir / f"{label}_cosine_matrices.pt")

    summary = {
        "model_name": result.model_name,
        "concept": result.concept,
        "steering_layers_tested": result.steering_layers_tested,
        "per_layer_summary": {},
    }
    for s_layer, lr in result.per_layer_results.items():
        per_threshold: dict[str, dict[str, Any]] = {}
        for r in lr.threshold_results:
            key = str(r.threshold)
            if key not in per_threshold:
                per_threshold[key] = {"fraction_above": [], "all_above_count": 0}
            per_threshold[key]["fraction_above"].append(r.fraction_above)
            if r.all_above:
                per_threshold[key]["all_above_count"] += 1

        layer_summary: dict[str, Any] = {}
        for key, vals in per_threshold.items():
            layer_summary[key] = {
                "mean_fraction_above": sum(vals["fraction_above"]) / len(vals["fraction_above"]),
                "all_above_count": vals["all_above_count"],
            }
        summary["per_layer_summary"][str(s_layer)] = {
            "steering_layer": s_layer,
            "empirical_thresholds": lr.empirical_thresholds,
            "per_threshold_analysis": layer_summary,
        }

    with open(output_dir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    full_data: dict[str, Any] = {
        "model_name": result.model_name,
        "concept": result.concept,
        "steering_layers_tested": result.steering_layers_tested,
        "per_layer_results": {},
    }
    for s_layer, lr in result.per_layer_results.items():
        full_data["per_layer_results"][str(s_layer)] = {
            "steering_layer": s_layer,
            "threshold_results": [
                {
                    "layer_idx": r.layer_idx,
                    "fraction_above": r.fraction_above,
                    "all_above": r.all_above,
                    "max_cosine": r.max_cosine,
                    "mean_cosine": r.mean_cosine,
                    "threshold": r.threshold,
                }
                for r in lr.threshold_results
            ],
            "empirical_thresholds": lr.empirical_thresholds,
            "cos_matrix_steered": lr.cos_matrix_steered.tolist(),
            "cos_matrix_unsteered": lr.cos_matrix_unsteered.tolist(),
        }

    with open(output_dir / f"{label}_full_results.json", "w") as f:
        json.dump(full_data, f, indent=2)
