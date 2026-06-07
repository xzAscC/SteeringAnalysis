from steering_analysis.config import ModelConfig
from steering_analysis.models import HookedModel


def test_resolve_layers_middle(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    hm = HookedModel(config)
    result = hm.resolve_layers([0.5])
    assert result == [2], f"Expected [2], got {result}"


def test_resolve_layers_boundaries(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    hm = HookedModel(config)
    result = hm.resolve_layers([0.0, 1.0])
    assert result == [0, 3], f"Expected [0, 3], got {result}"


def test_resolve_layers_multiple(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    hm = HookedModel(config)
    result = hm.resolve_layers([0.25, 0.5, 0.75])
    assert result == [1, 2, 2], f"Expected [1, 2, 2], got {result}"


def test_get_activations_returns_dict(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    hm = HookedModel(config)
    activations = hm.get_activations(["hello", "world"], [0, 2])
    assert isinstance(activations, dict)
    assert set(activations.keys()) == {0, 2}


def test_get_activations_shape(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    hm = HookedModel(config)
    texts = ["hello", "world"]
    activations = hm.get_activations(texts, [0, 2])
    max_len = max(len(t) for t in texts)
    batch_size = len(texts)
    hidden_dim = 8
    for layer_idx, tensor in activations.items():
        assert tensor.shape == (batch_size, max_len, hidden_dim), (
            f"Layer {layer_idx}: expected ({batch_size}, {max_len}, {hidden_dim}), got {tensor.shape}"
        )


def test_get_activations_different_layers(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    hm = HookedModel(config)
    activations = hm.get_activations(["hello"], [0, 3])
    act_0 = activations[0]
    act_3 = activations[3]
    assert not act_0.allclose(act_3), "Activations from layer 0 and 3 should differ"


def test_hooks_cleaned_up(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    hm = HookedModel(config)
    hm.get_activations(["hello", "world"], [0, 2])
    for i in range(4):
        hooks = list(mock_hooked_model.model.layers[i]._forward_hooks.keys())
        assert len(hooks) == 0, f"Layer {i} still has {len(hooks)} forward hooks after cleanup"
