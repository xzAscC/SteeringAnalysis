from collections.abc import Callable

import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from torch import Tensor

from .config import VALID_CONCEPTS, ExtractionConfig
from .models import HookedModel
from .types import ContrastPair, ContrastPairMetadata, SteeringVector
from .utils import sample_with_seed

Aggregator = Callable[[Tensor, Tensor], Tensor]


def mean_aggregator(pos: Tensor, neg: Tensor) -> Tensor:
    """Mean difference aggregator: mean of pairwise differences."""
    return (pos - neg).mean(dim=0)


def pca_aggregator(pos: Tensor, neg: Tensor) -> Tensor:
    """PCA-based aggregator: first principal component of differences."""
    deltas = pos - neg
    pca = PCA(n_components=1)
    pca.fit(deltas.detach().cpu().numpy())
    component = torch.from_numpy(pca.components_[0])
    return component.to(device=deltas.device, dtype=deltas.dtype)


def _resolve_aggregator(method: str) -> Aggregator:
    """Resolve aggregator by method name."""
    aggregators: dict[str, Aggregator] = {"mean": mean_aggregator, "pca": pca_aggregator}
    if method not in aggregators:
        raise ValueError(f"Unsupported method: {method}. Choose from: {list(aggregators.keys())}")
    return aggregators[method]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _load_sentiment_data(concept_name: str, num_pairs: int, seed: int = 42) -> list[ContrastPair]:
    dataset = load_dataset("glue", "sst2")
    positives = [row["sentence"].strip() for row in dataset["train"] if row["sentence"] and row["label"] == 1]
    negatives = [row["sentence"].strip() for row in dataset["train"] if row["sentence"] and row["label"] == 0]
    k = min(num_pairs, min(len(positives), len(negatives)))
    pos_sampled = sample_with_seed(positives, k, seed=seed)
    neg_sampled = sample_with_seed(negatives, k, seed=seed)
    return [
        ContrastPair(
            positive=p,
            negative=n,
            metadata=ContrastPairMetadata(
                concept=concept_name, dataset="sentiment", source="glue/sst2", pair_index=i
            ),
        )
        for i, (p, n) in enumerate(zip(pos_sampled, neg_sampled))
    ]


def _load_refusal_data(concept_name: str, num_pairs: int, seed: int = 42) -> list[ContrastPair]:
    benign = load_dataset("LLM-LAT/benign-dataset")
    harmful = load_dataset("LLM-LAT/harmful-dataset")
    benign_texts = [row["prompt"].strip() for row in benign["train"] if row.get("prompt")]
    harmful_texts = [row["prompt"].strip() for row in harmful["train"] if row.get("prompt")]
    k = min(num_pairs, min(len(benign_texts), len(harmful_texts)))
    pos_sampled = sample_with_seed(benign_texts, k, seed=seed)
    neg_sampled = sample_with_seed(harmful_texts, k, seed=seed)
    return [
        ContrastPair(
            positive=p,
            negative=n,
            metadata=ContrastPairMetadata(
                concept=concept_name, dataset="refusal", source="LLM-LAT/benign+harmful", pair_index=i
            ),
        )
        for i, (p, n) in enumerate(zip(pos_sampled, neg_sampled))
    ]


def _load_polite_data(concept_name: str, num_pairs: int, seed: int = 42) -> list[ContrastPair]:
    dataset = load_dataset("Intel/polite-guard", split="train")
    polite = [row["text"].strip() for row in dataset if row.get("text") and row["label"] == "polite"]
    impolite = [row["text"].strip() for row in dataset if row.get("text") and row["label"] == "impolite"]
    k = min(num_pairs, min(len(polite), len(impolite)))
    pos_sampled = sample_with_seed(polite, k, seed=seed)
    neg_sampled = sample_with_seed(impolite, k, seed=seed)
    return [
        ContrastPair(
            positive=p,
            negative=n,
            metadata=ContrastPairMetadata(
                concept=concept_name, dataset="polite", source="Intel/polite-guard", pair_index=i
            ),
        )
        for i, (p, n) in enumerate(zip(pos_sampled, neg_sampled))
    ]


_DATASET_LOADERS: dict[str, Callable[..., list[ContrastPair]]] = {
    "sentiment": _load_sentiment_data,
    "refusal": _load_refusal_data,
    "polite": _load_polite_data,
}


def load_contrast_pairs(concept: str, num_pairs: int, seed: int = 42) -> list[ContrastPair]:
    """Load contrast pairs for a given concept from the appropriate dataset."""
    if concept not in VALID_CONCEPTS:
        raise ValueError(f"Invalid concept: {concept}. Valid: {sorted(VALID_CONCEPTS)}")
    return _DATASET_LOADERS[concept](concept, num_pairs, seed=seed)


def extract_steering_vector(
    model: HookedModel,
    pairs: list[ContrastPair],
    config: ExtractionConfig,
) -> SteeringVector:
    """Extract steering vector from contrast pairs using last token only."""
    if not pairs:
        raise ValueError("Contrast pairs cannot be empty")

    layers = model.resolve_layers(config.layers)
    aggregator = _resolve_aggregator(config.method)

    positive_per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    negative_per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}

    for start in range(0, len(pairs), config.batch_size):
        batch = pairs[start : start + config.batch_size]
        positive_texts = [pair.positive for pair in batch]
        negative_texts = [pair.negative for pair in batch]

        pos_activations = model.get_activations(positive_texts, layers)
        neg_activations = model.get_activations(negative_texts, layers)

        for layer in layers:
            pos_act = pos_activations[layer]
            neg_act = neg_activations[layer]
            pos_token_idx = config.read_token_index
            neg_token_idx = config.read_token_index
            if pos_token_idx < 0:
                pos_token_idx = pos_act.shape[1] + pos_token_idx
            if neg_token_idx < 0:
                neg_token_idx = neg_act.shape[1] + neg_token_idx
            positive_per_layer[layer].append(pos_act[:, pos_token_idx, :])
            negative_per_layer[layer].append(neg_act[:, neg_token_idx, :])

    layer_activations: dict[int, Tensor] = {}
    for layer in layers:
        pos_all = torch.cat(positive_per_layer[layer], dim=0)
        neg_all = torch.cat(negative_per_layer[layer], dim=0)
        layer_activations[layer] = aggregator(pos_all, neg_all)

    concept_value = pairs[0].metadata.get("concept")
    concept = concept_value if isinstance(concept_value, str) else "unknown"

    return SteeringVector(
        layer_activations=layer_activations,
        model_name=model.config.model_name,
        concept=concept,
        method=config.method,
    )
