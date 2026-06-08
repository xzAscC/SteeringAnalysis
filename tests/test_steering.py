import pytest
import torch

from steering_analysis.config import ModelConfig
from steering_analysis.models import HookedModel


def test_normalize_vectors_unit_norm():
    from steering_analysis.steering import _normalize_vectors

    vectors = {0: torch.tensor([3.0, 4.0])}
    result = _normalize_vectors(vectors)
    norm = result[0].norm().item()
    assert abs(norm - 1.0) < 1e-6, f"Expected norm ~1.0, got {norm}"


def test_normalize_vectors_preserves_direction():
    from steering_analysis.steering import _normalize_vectors

    original = torch.tensor([3.0, 4.0])
    vectors = {0: original}
    result = _normalize_vectors(vectors)
    normalized = result[0]
    # Same direction means element-wise ratio is constant
    ratio = normalized / original
    assert torch.allclose(ratio, ratio.expand_as(ratio)), "Direction should be preserved"
    assert normalized[0].item() > 0 and original[0].item() > 0, "Signs should match"


def test_normalize_vectors_multiple_layers():
    from steering_analysis.steering import _normalize_vectors

    torch.manual_seed(42)
    vectors = {0: torch.randn(8), 2: torch.randn(8)}
    result = _normalize_vectors(vectors)
    assert set(result.keys()) == {0, 2}
    assert abs(result[0].norm().item() - 1.0) < 1e-6
    assert abs(result[2].norm().item() - 1.0) < 1e-6


def test_normalize_vectors_zero_vector():
    from steering_analysis.steering import _normalize_vectors

    vectors = {0: torch.zeros(4)}
    result = _normalize_vectors(vectors)
    assert torch.allclose(result[0], torch.zeros(4)), "Zero vector should stay zeros"


def test_compute_avg_activation_uses_all_tokens(mock_hooked_model):
    from steering_analysis.steering import _compute_avg_activation

    hm = HookedModel(ModelConfig(model_name="fake-model"))
    texts = ["hello", "world"]
    layers = [0]
    result = _compute_avg_activation(hm, texts, layers=layers)
    activations = hm.get_activations(texts, layers)
    expected = float(activations[0].norm(dim=-1).mean().item())
    assert abs(result[0] - expected) < 1e-5, f"Expected avg norm over ALL tokens ({expected}), got {result[0]}"


def test_compute_avg_activation_returns_per_layer(mock_hooked_model):
    from steering_analysis.steering import _compute_avg_activation

    hm = HookedModel(ModelConfig(model_name="fake-model"))
    texts = ["hello", "world"]
    result = _compute_avg_activation(hm, texts, layers=[0, 2])
    assert set(result.keys()) == {0, 2}


def test_compute_avg_activation_positive_values(mock_hooked_model):
    from steering_analysis.steering import _compute_avg_activation

    hm = HookedModel(ModelConfig(model_name="fake-model"))
    texts = ["hello", "world"]
    result = _compute_avg_activation(hm, texts, layers=[0, 2])
    for layer_idx, val in result.items():
        assert val > 0, f"Layer {layer_idx}: expected positive value, got {val}"


def test_compute_avg_activation_returns_float_dict(mock_hooked_model):
    from steering_analysis.steering import _compute_avg_activation

    hm = HookedModel(ModelConfig(model_name="fake-model"))
    texts = ["hello", "world"]
    result = _compute_avg_activation(hm, texts, layers=[0, 2])
    for layer_idx, val in result.items():
        assert isinstance(val, float), f"Layer {layer_idx}: expected float, got {type(val)}"


@pytest.fixture
def hooked_model(mock_hooked_model):
    return HookedModel(ModelConfig(model_name="fake-model"))


@pytest.fixture
def sample_vector():
    from steering_analysis.types import SteeringVector

    return SteeringVector(
        layer_activations={0: torch.randn(8), 2: torch.randn(8)},
        model_name="fake",
        concept="test",
        method="mean",
    )


def test_apply_steering_creates_jsonl_files(hooked_model, sample_vector, tmp_path):
    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=2)
    apply_steering(hooked_model, sample_vector, prompts=["hello", "world"], config=config, output_dir=tmp_path)
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) > 0, "Expected at least one .jsonl file in output_dir"


def test_apply_steering_jsonl_has_required_fields(hooked_model, sample_vector, tmp_path):
    import json

    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=2, steer_tokens=5)
    apply_steering(hooked_model, sample_vector, prompts=["hello", "world"], config=config, output_dir=tmp_path)
    for jsonl_file in tmp_path.glob("*.jsonl"):
        for line in jsonl_file.read_text().strip().splitlines():
            record = json.loads(line)
            assert "prompt" in record
            assert "generated_text" in record
            assert "layer" in record
            assert "multiplier" in record
            assert "steer_tokens" in record
            assert "sample_index" in record
            assert isinstance(record["prompt"], str)
            assert isinstance(record["generated_text"], str)
            assert isinstance(record["layer"], int)
            assert isinstance(record["multiplier"], float)
            assert isinstance(record["sample_index"], int)


def test_apply_steering_one_file_per_layer(hooked_model, sample_vector, tmp_path):
    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=1)
    apply_steering(hooked_model, sample_vector, prompts=["hello"], config=config, output_dir=tmp_path)
    expected_files = {tmp_path / "layer_0.jsonl", tmp_path / "layer_2.jsonl"}
    actual_files = set(tmp_path.glob("layer_*.jsonl"))
    assert set(actual_files) == expected_files, f"Expected {expected_files}, got {actual_files}"


def test_apply_steering_respects_multipliers(hooked_model, sample_vector, tmp_path):
    import json

    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[0.5, 1.0], num_samples=1)
    apply_steering(hooked_model, sample_vector, prompts=["hello"], config=config, output_dir=tmp_path)
    for jsonl_file in tmp_path.glob("layer_*.jsonl"):
        lines = jsonl_file.read_text().strip().splitlines()
        assert len(lines) == 2, f"Expected 2 lines (2 multipliers), got {len(lines)}"
        multipliers = [json.loads(line)["multiplier"] for line in lines]
        assert set(multipliers) == {0.5, 1.0}


def test_apply_steering_respects_num_samples(hooked_model, sample_vector, tmp_path):
    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=2)
    apply_steering(hooked_model, sample_vector, prompts=["hello", "world"], config=config, output_dir=tmp_path)
    for jsonl_file in tmp_path.glob("layer_*.jsonl"):
        lines = jsonl_file.read_text().strip().splitlines()
        assert len(lines) == 2, f"Expected 2 lines (num_samples=2), got {len(lines)}"


def test_apply_steering_scales_by_avg_activation(hooked_model, sample_vector, tmp_path):
    from unittest.mock import patch

    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=1)
    with patch("steering_analysis.steering._compute_avg_activation", return_value={0: 100.0, 2: 100.0}):
        apply_steering(
            hooked_model,
            sample_vector,
            prompts=["hello"],
            config=config,
            output_dir=tmp_path / "high",
        )
    with patch("steering_analysis.steering._compute_avg_activation", return_value={0: 0.001, 2: 0.001}):
        apply_steering(
            hooked_model,
            sample_vector,
            prompts=["hello"],
            config=config,
            output_dir=tmp_path / "low",
        )
    high_file = tmp_path / "high" / "layer_0.jsonl"
    low_file = tmp_path / "low" / "layer_0.jsonl"
    import json

    high_text = json.loads(high_file.read_text().strip())["generated_text"]
    low_text = json.loads(low_file.read_text().strip())["generated_text"]
    assert high_text != low_text, "Expected different outputs when avg_activation scaling differs (100.0 vs 0.001)"


def test_apply_steering_records_avg_activation_in_output(hooked_model, sample_vector, tmp_path):
    import json

    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=1)
    apply_steering(
        hooked_model,
        sample_vector,
        prompts=["hello"],
        config=config,
        output_dir=tmp_path,
    )
    jsonl_file = tmp_path / "layer_0.jsonl"
    record = json.loads(jsonl_file.read_text().strip())
    assert "avg_activation" in record, "Output should include avg_activation for reproducibility"
    assert isinstance(record["avg_activation"], float)


def test_apply_steering_uses_seed(hooked_model, sample_vector, tmp_path):
    """apply_steering should call torch.manual_seed with config.seed for reproducibility."""
    from unittest.mock import patch

    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=1, seed=123)
    with patch("steering_analysis.steering.torch.manual_seed") as mock_seed:
        apply_steering(
            hooked_model,
            sample_vector,
            prompts=["hello"],
            config=config,
            output_dir=tmp_path,
        )
        mock_seed.assert_called_with(123)


def test_apply_steering_angular_produces_jsonl(hooked_model, sample_vector, tmp_path):
    import json

    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=1, steering_method="angular")
    apply_steering(hooked_model, sample_vector, prompts=["hello"], config=config, output_dir=tmp_path)
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) > 0, "Expected at least one .jsonl file for angular steering"
    for jsonl_file in jsonl_files:
        for line in jsonl_file.read_text().strip().splitlines():
            record = json.loads(line)
            assert record["steering_method"] == "angular"


def test_apply_steering_records_steering_method(hooked_model, sample_vector, tmp_path):
    import json

    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config = SteeringConfig(multipliers=[1.0], num_samples=1)
    apply_steering(hooked_model, sample_vector, prompts=["hello"], config=config, output_dir=tmp_path)
    for jsonl_file in tmp_path.glob("*.jsonl"):
        record = json.loads(jsonl_file.read_text().strip().splitlines()[0])
        assert "steering_method" in record
        assert record["steering_method"] == "additive"


def test_apply_steering_angular_differs_from_additive(hooked_model, sample_vector, tmp_path):
    import json
    from unittest.mock import patch

    from steering_analysis.config import SteeringConfig
    from steering_analysis.steering import apply_steering

    config_add = SteeringConfig(multipliers=[10.0], num_samples=1, steering_method="additive")
    config_ang = SteeringConfig(multipliers=[10.0], num_samples=1, steering_method="angular")

    with patch("steering_analysis.steering._compute_avg_activation", return_value={0: 100.0, 2: 100.0}):
        apply_steering(hooked_model, sample_vector, prompts=["hello"], config=config_add, output_dir=tmp_path / "add")
        apply_steering(hooked_model, sample_vector, prompts=["hello"], config=config_ang, output_dir=tmp_path / "ang")

    add_text = json.loads((tmp_path / "add" / "layer_0.jsonl").read_text().strip())["generated_text"]
    ang_text = json.loads((tmp_path / "ang" / "layer_0.jsonl").read_text().strip())["generated_text"]
    assert add_text != ang_text, "Angular and additive steering should produce different outputs with strong scale"
