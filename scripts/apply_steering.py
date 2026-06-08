"""CLI entry point for applying steering vectors to model generation.

Usage:
    uv run python scripts/apply_steering.py \\
        --vector results/vectors/sentiment.pt --concept sentiment
    uv run python scripts/apply_steering.py \\
        --vector results/vectors/refusal.pt --concept refusal --steer-tokens 5
    uv run python scripts/apply_steering.py \\
        --vector results/vectors/polite.pt --concept polite \\
        --multipliers 0.1 1.0 10.0
"""

import argparse

import torch

from steering_analysis.config import ModelConfig, SteeringConfig
from steering_analysis.extract import load_contrast_pairs
from steering_analysis.models import HookedModel
from steering_analysis.steering import apply_steering
from steering_analysis.types import SteeringVector
from steering_analysis.utils import safe_model_name, sample_with_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply steering vectors to model generation")
    parser.add_argument(
        "--vector",
        required=True,
        help="Path to saved steering vector (.pt file)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--concept",
        required=True,
        choices=["refusal", "sentiment", "polite"],
        help="Concept to steer",
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0, 10.0],
        help="Steering scale multipliers (default: 0.01 0.1 1.0 10.0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of prompts to steer (default: 10)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max tokens to generate per sample (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    parser.add_argument(
        "--steer-tokens",
        type=int,
        default=None,
        help="Number of generation steps to apply steering (default: None = full steering)",
    )
    parser.add_argument(
        "--steering-method",
        choices=["additive", "angular"],
        default="additive",
        help="Steering method: additive (default) or angular (norm-preserving)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="results/steering/",
        help="Output directory (default: results/steering/)",
    )
    args = parser.parse_args()

    torch.serialization.add_safe_globals([SteeringVector])
    data = torch.load(args.vector, weights_only=True)
    vector = data["vector"] if isinstance(data, dict) and "vector" in data else data

    model_config = ModelConfig(model_name=args.model)
    model = HookedModel(model_config)

    steering_config = SteeringConfig(
        multipliers=args.multipliers,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        steer_tokens=args.steer_tokens,
        seed=args.seed,
        steering_method=args.steering_method,
    )

    pairs = load_contrast_pairs(args.concept, args.num_samples, seed=args.seed)
    prompts = sample_with_seed([p.negative for p in pairs], args.num_samples, seed=args.seed)

    model_slug = safe_model_name(args.model)
    method_label = f"_{args.steering_method}" if args.steering_method != "additive" else ""
    steer_label = f"prefix{args.steer_tokens}" if args.steer_tokens is not None else "full"
    output_dir = f"{args.output.rstrip('/')}/{args.concept}_{model_slug}_{steer_label}{method_label}"

    apply_steering(model, vector, prompts, steering_config, output_dir)
    print(f"Saved steering results to {output_dir}")


if __name__ == "__main__":
    main()
