"""CLI entry point for extracting concept steering vectors.

Usage:
    uv run python scripts/extract_vectors.py --concept sentiment --model Qwen/Qwen3-1.7B
    uv run python scripts/extract_vectors.py --concept refusal --method pca --num-pairs 50
"""

import argparse
from pathlib import Path

import torch

from steering_analysis.config import ExtractionConfig, ModelConfig
from steering_analysis.extract import extract_steering_vector, load_contrast_pairs
from steering_analysis.models import HookedModel
from steering_analysis.utils import safe_model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract concept steering vectors")
    parser.add_argument(
        "--concept",
        required=True,
        choices=["refusal", "sentiment", "polite"],
        help="Concept to extract",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--method",
        choices=["mean", "pca"],
        default="mean",
        help="Aggregation method (default: mean)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=50,
        help="Number of contrast pairs (default: 50)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=float,
        default=None,
        help="Relative layer positions (e.g., 0.5 0.7) (default: [0.4, 0.5, 0.6, 0.7, 0.8])",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="results/vectors/",
        help="Output directory (default: results/vectors/)",
    )
    args = parser.parse_args()

    # Load contrast pairs
    pairs = load_contrast_pairs(args.concept, args.num_pairs, seed=args.seed)
    print(f"Loaded {len(pairs)} contrast pairs for '{args.concept}'")

    # Load model
    model_config = ModelConfig(model_name=args.model)
    model = HookedModel(model_config)

    # Configure extraction
    extraction_config = ExtractionConfig(
        layers=args.layers if args.layers else [0.4, 0.5, 0.6, 0.7, 0.8],
        method=args.method,
        batch_size=args.batch_size,
        num_pairs=args.num_pairs,
        seed=args.seed,
    )

    # Extract steering vector
    vector = extract_steering_vector(model, pairs, extraction_config)

    # Save
    model_slug = safe_model_name(args.model)
    output_path = f"{args.output.rstrip('/')}/{args.concept}_{model_slug}_{args.method}_n{args.num_pairs}.pt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": vector, "num_pairs": len(pairs)}, output_path)
    print(f"Saved steering vector to {output_path}")
    print(f"Layers: {list(vector.layer_activations.keys())}")
    for layer, activation in vector.layer_activations.items():
        print(f"  Layer {layer}: shape={activation.shape}, norm={activation.norm():.4f}")


if __name__ == "__main__":
    main()
