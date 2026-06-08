import torch

from steering_analysis.types import ContrastPair, ContrastPairMetadata, SteeringVector


def test_contrast_pair_construction():
    metadata: ContrastPairMetadata = {
        "concept": "sentiment",
        "dataset": " imdb",
        "source": "manual",
        "pair_index": 0,
    }
    pair = ContrastPair(positive="hello", negative="bye", metadata=metadata)
    assert pair.positive == "hello"
    assert pair.negative == "bye"
    assert pair.metadata == metadata


def test_contrast_pair_metadata_access():
    metadata: ContrastPairMetadata = {
        "concept": "sentiment",
        "dataset": "imdb",
        "source": "manual",
        "pair_index": 0,
    }
    pair = ContrastPair(positive="hello", negative="bye", metadata=metadata)
    assert pair.metadata["concept"] == "sentiment"
    assert pair.metadata["dataset"] == "imdb"
    assert pair.metadata["source"] == "manual"
    assert pair.metadata["pair_index"] == 0


def test_steering_vector_construction():
    activations = {0: torch.randn(8)}
    vec = SteeringVector(
        layer_activations=activations,
        model_name="test",
        concept="sentiment",
        method="mean",
    )
    assert vec.model_name == "test"
    assert vec.concept == "sentiment"
    assert vec.method == "mean"
    assert 0 in vec.layer_activations


def test_steering_vector_layer_activations_shape():
    activations = {0: torch.randn(16), 3: torch.randn(16)}
    vec = SteeringVector(
        layer_activations=activations,
        model_name="test",
        concept="sentiment",
        method="mean",
    )
    for layer, tensor in vec.layer_activations.items():
        assert tensor.dim() == 1, f"Layer {layer} tensor is not 1D"


def test_contrast_pair_metadata_typed_dict():
    metadata: ContrastPairMetadata = {
        "concept": "toxicity",
        "dataset": "civil_comments",
        "source": "auto",
        "pair_index": 5,
        "original_question": "Is this toxic?",
    }
    assert metadata["concept"] == "toxicity"
    assert metadata["original_question"] == "Is this toxic?"
