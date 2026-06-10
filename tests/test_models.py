from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

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


# ---------------------------------------------------------------------------
# Pythia (GPT-NeoX) architecture tests
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    pad_token = None
    eos_token = "</s>"
    pad_token_id = None
    eos_token_id = 2
    bos_token_id = 1
    _vocab_size = 32

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        if isinstance(texts, str):
            texts = [texts]
        all_ids = []
        for text in texts:
            ids = [ord(c) % self._vocab_size for c in text]
            all_ids.append(ids)
        max_len = max(len(ids) for ids in all_ids)
        padded = [ids + [0] * (max_len - len(ids)) for ids in all_ids]
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(
            [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids],
            dtype=torch.long,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            return " ".join(self.decode(ids) for ids in token_ids)
        chars = [chr(t) for t in token_ids if t != 0]
        return "".join(chars)


@pytest.fixture
def mock_pythia_model(monkeypatch):
    import transformers

    class _FakeLayer(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            return (self.linear(x),)

    class _FakeGPTNeoX(nn.Module):
        def __init__(self, hidden_dim, num_layers, vocab_size):
            super().__init__()
            self.layers = nn.ModuleList([_FakeLayer(hidden_dim) for _ in range(num_layers)])
            self.embed_in = nn.Embedding(vocab_size, hidden_dim)

    model = nn.Module()
    model.gpt_neox = _FakeGPTNeoX(8, 4, 32)

    def fake_forward(input_ids, **kwargs):
        x = model.gpt_neox.embed_in(input_ids)
        for layer in model.gpt_neox.layers:
            x = layer(x)[0]
        return SimpleNamespace(logits=x)

    def fake_generate(input_ids=None, max_new_tokens=20, pad_token_id=None, **kwargs):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            x = model.gpt_neox.embed_in(generated)
            for layer in model.gpt_neox.layers:
                x = layer(x)[0]
            logits = x[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    model.forward = fake_forward
    model.generate = fake_generate
    model.device = next(model.parameters()).device

    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    return model


def test_pythia_num_layers(mock_pythia_model):
    hm = HookedModel(ModelConfig(model_name="fake-pythia"))
    assert hm.num_layers == 4


def test_pythia_resolve_layers(mock_pythia_model):
    hm = HookedModel(ModelConfig(model_name="fake-pythia"))
    result = hm.resolve_layers([0.0, 0.5, 1.0])
    assert result == [0, 2, 3], f"Expected [0, 2, 3], got {result}"


def test_pythia_get_activations_returns_dict(mock_pythia_model):
    hm = HookedModel(ModelConfig(model_name="fake-pythia"))
    activations = hm.get_activations(["hello"], [0, 2])
    assert isinstance(activations, dict)
    assert set(activations.keys()) == {0, 2}


def test_pythia_get_activations_shape(mock_pythia_model):
    hm = HookedModel(ModelConfig(model_name="fake-pythia"))
    texts = ["hello", "world"]
    activations = hm.get_activations(texts, [0, 3])
    batch_size = 2
    max_len = max(len(t) for t in texts)
    hidden_dim = 8
    for layer_idx, tensor in activations.items():
        assert tensor.shape == (batch_size, max_len, hidden_dim), (
            f"Layer {layer_idx}: expected ({batch_size}, {max_len}, {hidden_dim}), got {tensor.shape}"
        )


def test_pythia_hooks_cleaned_up(mock_pythia_model):
    hm = HookedModel(ModelConfig(model_name="fake-pythia"))
    hm.get_activations(["hello"], [0, 2])
    for i in range(4):
        hooks = list(mock_pythia_model.gpt_neox.layers[i]._forward_hooks.keys())
        assert len(hooks) == 0, f"Layer {i} still has {len(hooks)} forward hooks after cleanup"


def test_pythia_different_layers_different_activations(mock_pythia_model):
    hm = HookedModel(ModelConfig(model_name="fake-pythia"))
    activations = hm.get_activations(["hello"], [0, 3])
    assert not activations[0].allclose(activations[3]), "Activations from layer 0 and 3 should differ"


class TestDeviceConfig:
    """Issue 14: config.device should be forwarded to device_map in from_pretrained."""

    def test_auto_device_uses_device_map_auto(self, monkeypatch):
        import transformers

        captured = {}

        def capture_model(*args, **kwargs):
            captured.update(kwargs)
            from conftest import FakeCausalLM

            return FakeCausalLM()

        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", capture_model)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: _FakeTokenizer())

        config = ModelConfig(model_name="fake-model", device="auto")
        HookedModel(config)
        assert captured["device_map"] == "auto"

    def test_custom_device_uses_device_map_dict(self, monkeypatch):
        import transformers

        captured = {}

        def capture_model(*args, **kwargs):
            captured.update(kwargs)
            from conftest import FakeCausalLM

            return FakeCausalLM()

        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", capture_model)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: _FakeTokenizer())

        config = ModelConfig(model_name="fake-model", device="cuda:0")
        HookedModel(config)
        assert captured["device_map"] == {"": "cuda:0"}


class TestAngularSteering:
    def test_angular_steering_produces_output(self, mock_hooked_model):
        hm = HookedModel(ModelConfig(model_name="fake-model"))
        sv = torch.randn(8)
        text = hm.generate_with_steering(
            "hello",
            layer_idx=2,
            steering_vector=sv,
            scale=1.0,
            steering_method="angular",
            max_new_tokens=5,
        )
        assert isinstance(text, str)

    def test_angular_steering_preserves_norm(self, mock_hooked_model):
        hm = HookedModel(ModelConfig(model_name="fake-model"))
        sv = torch.randn(8)
        layer_idx = 2
        scale = 5.0

        activations_clean = hm.get_activations(["hello"], [layer_idx])
        original_norm = activations_clean[layer_idx].norm(dim=-1)

        activations_angular: dict[int, torch.Tensor] = {}
        handles = []
        model_layers = hm._get_layers_module()

        def angular_hook(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            orig_norm = t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            shifted = t + sv.to(dtype=t.dtype, device=t.device) * scale
            shifted_norm = shifted.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            t = shifted * (orig_norm / shifted_norm)
            if isinstance(output, tuple):
                return (t,) + output[1:]
            return t

        handles.append(model_layers[layer_idx].register_forward_hook(angular_hook))
        try:
            with torch.no_grad():
                _ = hm.model(**hm.tokenizer("hello", return_tensors="pt"))
            for h in handles:
                h.remove()
            activations_angular = {layer_idx: hm.get_activations(["hello"], [layer_idx])[layer_idx]}
        finally:
            for h in handles:
                h.remove()

        steered_norm = activations_angular[layer_idx].norm(dim=-1)
        assert torch.allclose(original_norm, steered_norm, atol=0.5), (
            f"Angular steering should preserve norm: original={original_norm.mean():.4f}, "
            f"angular={steered_norm.mean():.4f}"
        )

    def test_angular_steering_changes_direction(self, mock_hooked_model):
        hm = HookedModel(ModelConfig(model_name="fake-model"))
        sv = torch.randn(8)
        text = "hello"

        text_angular = hm.generate_with_steering(
            text,
            layer_idx=2,
            steering_vector=sv,
            scale=50.0,
            steering_method="angular",
            max_new_tokens=5,
        )
        text_clean = hm.generate_with_steering(
            text,
            layer_idx=2,
            steering_vector=sv,
            scale=0.0,
            max_new_tokens=5,
        )
        assert text_angular != text_clean, "Angular steering should change output from unsteered at high scale"

    def test_angular_differs_from_additive(self, mock_hooked_model):
        from steering_analysis.assumption_verification import get_steered_activations

        hm = HookedModel(ModelConfig(model_name="fake-model"))
        sv = torch.randn(8)

        additive_act = get_steered_activations(hm, "hello", layer_idx=2, steering_vector=sv, scale=10.0)
        angular_act = get_steered_activations(
            hm,
            "hello",
            layer_idx=2,
            steering_vector=sv,
            scale=10.0,
            steering_method="angular",
        )
        assert not additive_act[2].allclose(angular_act[2]), "Angular and additive should produce different activations"

    def test_default_steering_method_is_additive(self, mock_hooked_model):
        hm = HookedModel(ModelConfig(model_name="fake-model"))
        sv = torch.randn(8)

        text_default = hm.generate_with_steering(
            "hello",
            layer_idx=2,
            steering_vector=sv,
            scale=5.0,
            max_new_tokens=5,
        )
        text_explicit_additive = hm.generate_with_steering(
            "hello",
            layer_idx=2,
            steering_vector=sv,
            scale=5.0,
            steering_method="additive",
            max_new_tokens=5,
        )
        assert text_default == text_explicit_additive, "Default steering_method should be additive"


def test_generate_with_steering_rejects_unsupported_method(mock_hooked_model):
    hm = HookedModel(ModelConfig(model_name="fake-model"))
    sv = torch.randn(8)
    with pytest.raises(ValueError, match="Unsupported steering_method"):
        hm.generate_with_steering(
            "hello",
            layer_idx=2,
            steering_vector=sv,
            scale=5.0,
            max_new_tokens=5,
            steering_method="prefix",
        )
