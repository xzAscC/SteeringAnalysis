import os
import sys

import pytest
import torch

from steering_analysis.config import ModelConfig
from steering_analysis.models import HookedModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import FakeCausalLM, FakeTokenizer  # noqa: E402


def test_fake_generate_produces_tokens():
    model = FakeCausalLM()
    tok = FakeTokenizer()
    encoded = tok("hello", return_tensors="pt")
    input_ids = encoded["input_ids"]
    output = model.generate(input_ids=input_ids, max_new_tokens=5)
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] > input_ids.shape[1]


def test_fake_generate_respects_max_new_tokens():
    model = FakeCausalLM()
    tok = FakeTokenizer()
    encoded = tok("hello", return_tensors="pt")
    input_ids = encoded["input_ids"]
    input_len = input_ids.shape[1]
    output = model.generate(input_ids=input_ids, max_new_tokens=5)
    assert output.shape[1] == input_len + 5


def test_fake_generate_greedy_deterministic():
    model = FakeCausalLM()
    tok = FakeTokenizer()
    encoded = tok("hello", return_tensors="pt")
    input_ids = encoded["input_ids"]
    out1 = model.generate(input_ids=input_ids, max_new_tokens=5)
    out2 = model.generate(input_ids=input_ids, max_new_tokens=5)
    assert torch.equal(out1, out2)


def test_fake_generate_returns_batch_dim():
    model = FakeCausalLM()
    tok = FakeTokenizer()
    encoded = tok(["hello", "world"], return_tensors="pt")
    input_ids = encoded["input_ids"]
    output = model.generate(input_ids=input_ids, max_new_tokens=5)
    assert output.shape[0] == 2



@pytest.fixture
def hooked_model(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    return HookedModel(config)


def test_generate_with_steering_returns_string(hooked_model):
    result = hooked_model.generate_with_steering(
        prompt="hello",
        layer_idx=0,
        steering_vector=torch.randn(8),
        scale=1.0,
        max_new_tokens=5,
    )
    assert isinstance(result, str)


def test_generate_with_steering_scale_zero_changes_nothing(hooked_model):
    tok = hooked_model.tokenizer
    inputs = tok("hello", return_tensors="pt")
    with torch.no_grad():
        baseline_ids = hooked_model.model.generate(**inputs, max_new_tokens=5)
    baseline_text = tok.decode(baseline_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    steered_text = hooked_model.generate_with_steering(
        "hello", layer_idx=0, steering_vector=torch.randn(8), scale=0.0, max_new_tokens=5,
    )
    assert steered_text == baseline_text


def test_generate_with_steering_nonzero_scale_changes_output(hooked_model):
    sv = torch.ones(8) * 50.0
    steered = hooked_model.generate_with_steering(
        "hello", layer_idx=0, steering_vector=sv, scale=1.0, max_new_tokens=5,
    )
    unsteered = hooked_model.generate_with_steering(
        "hello", layer_idx=0, steering_vector=sv, scale=0.0, max_new_tokens=5,
    )
    assert steered != unsteered


def test_generate_with_steering_full_steering(hooked_model):
    result = hooked_model.generate_with_steering(
        "hello", layer_idx=0, steering_vector=torch.randn(8), scale=1.0,
        max_new_tokens=5, steer_tokens=None,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_steering_prefix_steering(hooked_model):
    result = hooked_model.generate_with_steering(
        "hello", layer_idx=0, steering_vector=torch.randn(8), scale=1.0,
        max_new_tokens=5, steer_tokens=1,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_steering_hook_removed_after(hooked_model):
    hooked_model.generate_with_steering(
        "hello", layer_idx=0, steering_vector=torch.randn(8), scale=1.0, max_new_tokens=5,
    )
    layer = hooked_model._get_layers_module()[0]
    assert len(list(layer._forward_hooks.keys())) == 0


def test_generate_with_steering_different_layers(hooked_model):
    for layer_idx in [0, 3]:
        result = hooked_model.generate_with_steering(
            "hello", layer_idx=layer_idx, steering_vector=torch.randn(8),
            scale=1.0, max_new_tokens=5,
        )
        assert isinstance(result, str)
        assert len(result) > 0
