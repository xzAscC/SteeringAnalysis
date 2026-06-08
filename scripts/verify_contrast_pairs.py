"""CLI entry point for contrast pair verification experiment.

Computes cosine similarity between the positive/negative text hidden states
(used to build the steering vector) and the steering vector itself, then
replicates Assumption 1's Exp1 (token-level) and Exp2 (layer-level).

Usage:
    uv run python scripts/verify_contrast_pairs.py \
        --model EleutherAI/pythia-70m-deduped --concept refusal
    uv run python scripts/verify_contrast_pairs.py \
        --model EleutherAI/pythia-160m-deduped --concept sentiment \
        --thresholds 0.1 0.3 0.5
"""

import argparse

from steering_analysis.contrast_verification import run_contrast_verification, save_contrast_results
from steering_analysis.config import ModelConfig, VerificationConfig
from steering_analysis.models import HookedModel
from steering_analysis.utils import safe_model_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrast pair verification: positive/negative text hidden states vs steering vector",
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
        help="Number of contrast pairs to sample (default: 5)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=50,
        help="Number of pairs for steering vector extraction (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="results/contrast_pairs/",
        help="Output directory (default: results/contrast_pairs/)",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error(f"--num-samples must be positive, got {args.num_samples}")

    model_config = ModelConfig(model_name=args.model)
    model = HookedModel(model_config)

    config = VerificationConfig(
        thresholds=args.thresholds,
        steering_multiplier=args.multiplier,
        num_samples=args.num_samples,
        extraction_num_pairs=args.num_pairs,
        seed=args.seed,
    )

    result = run_contrast_verification(model, args.concept, config)

    model_slug = safe_model_name(args.model)
    label = f"{args.concept}_{model_slug}"
    save_contrast_results(result, args.output, label)
    print(f"Results saved to {args.output}/{label}_*")

    print(f"\n{'=' * 60}")
    print(f"Contrast Pair Verification: {args.concept} / {args.model}")
    print(f"{'=' * 60}")
    for s_layer, lr in result.per_layer_results.items():
        print(f"\n  Steering layer: {s_layer}")
        for threshold in sorted(lr.experiment1_verdicts.keys()):
            e1 = lr.experiment1_verdicts[threshold]
            e2 = lr.experiment2_verdicts[threshold]
            if e2 is None:
                continue
            pos_all = sum(1 for v in e1 if v.verdict == "ALL")
            pos_some = sum(1 for v in e1 if v.verdict == "SOME")
            pos_none = sum(1 for v in e1 if v.verdict == "NONE")
            holds_count = sum(1 for v in e1 if v.assumption_holds)
            print(
                f"    τ={threshold:.1f}  "
                f"Exp1: ALL={pos_all} SOME={pos_some} "
                f"NONE={pos_none} holds={holds_count}/{len(e1)}  |  "
                f"Exp2: pos_above={e2.exists_steered_layer} "
                f"neg_above={e2.exists_unsteered_layer} "
                f"holds={e2.assumption_holds}"
            )


if __name__ == "__main__":
    main()
