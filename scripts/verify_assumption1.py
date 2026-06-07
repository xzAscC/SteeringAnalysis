"""CLI entry point for verifying Assumption 1 of the steering analysis.

Assumption 1: Output is steered toward concept c at layer l iff cos(o_i^(l), c) > tau
for all (or some) token positions i, where tau is a predefined threshold.

Usage:
    uv run python scripts/verify_assumption1.py \
        --model EleutherAI/pythia-70m-deduped --concept refusal
    uv run python scripts/verify_assumption1.py \
        --model EleutherAI/pythia-160m-deduped --concept sentiment \
        --thresholds 0.1 0.3 0.5
"""

import argparse

from steering_analysis.assumption_verification import run_verification, save_results
from steering_analysis.config import ModelConfig, VerificationConfig
from steering_analysis.models import HookedModel
from steering_analysis.utils import safe_model_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Assumption 1: cosine similarity steering detection",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--concept",
        required=True,
        choices=["refusal", "sentiment", "polite"],
        help="Concept to verify",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Cosine similarity thresholds (default: 0.1 0.3 0.5 0.7 0.9)",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="Steering multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples (default: 5)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="results/assumption1/",
        help="Output directory (default: results/assumption1/)",
    )
    args = parser.parse_args()

    model_config = ModelConfig(model_name=args.model)
    model = HookedModel(model_config)

    config = VerificationConfig(
        thresholds=args.thresholds,
        steering_multiplier=args.multiplier,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    result = run_verification(model, args.concept, config)

    model_slug = safe_model_name(args.model)
    label = f"{args.concept}_{model_slug}"
    save_results(result, args.output, label)
    print(f"Results saved to {args.output}/{label}_*")


if __name__ == "__main__":
    main()
