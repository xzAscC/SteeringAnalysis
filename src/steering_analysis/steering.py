from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from .config import SteeringConfig
from .models import HookedModel
from .types import SteeringVector


def _normalize_vectors(vectors: dict[int, Tensor]) -> dict[int, Tensor]:
    normalized = {}
    for layer_idx, tensor in vectors.items():
        norm = tensor.norm()
        if norm > 0:
            normalized[layer_idx] = tensor / norm
        else:
            normalized[layer_idx] = tensor.clone()
    return normalized


def _compute_avg_activation(model: HookedModel, texts: list[str], layers: list[int]) -> dict[int, float]:
    activations = model.get_activations(texts, layers)
    result = {}
    for layer_idx, tensor in activations.items():
        result[layer_idx] = float(tensor.norm(dim=-1).mean().item())
    return result


def apply_steering(
    model: HookedModel,
    vector: SteeringVector,
    prompts: list[str],
    config: SteeringConfig,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    normalized = _normalize_vectors(vector.layer_activations)
    layer_indices = list(normalized.keys())
    avg_activation = _compute_avg_activation(model, prompts, layer_indices)
    for layer_idx, steering_vec in normalized.items():
        avg_norm = avg_activation[layer_idx]
        records = []
        for multiplier in config.multipliers:
            scale = multiplier * avg_norm
            for i, prompt in enumerate(prompts[: config.num_samples]):
                text = model.generate_with_steering(
                    prompt=prompt,
                    layer_idx=layer_idx,
                    steering_vector=steering_vec,
                    scale=scale,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    steer_tokens=config.steer_tokens,
                    steering_method=config.steering_method,
                )
                records.append(
                    {
                        "prompt": prompt,
                        "generated_text": text,
                        "layer": layer_idx,
                        "multiplier": multiplier,
                        "avg_activation": avg_norm,
                        "steer_tokens": config.steer_tokens,
                        "steering_method": config.steering_method,
                        "sample_index": i,
                    }
                )
        out_file = output_dir / f"layer_{layer_idx}.jsonl"
        with open(out_file, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
