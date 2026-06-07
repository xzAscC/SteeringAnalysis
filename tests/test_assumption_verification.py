import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from steering_analysis.config import ModelConfig, VerificationConfig
from steering_analysis.models import HookedModel
from steering_analysis.types import ContrastPair, ContrastPairMetadata

# ---------------------------------------------------------------------------
# Reusable helpers for tests
# ---------------------------------------------------------------------------


def _make_layer_activations(num_layers=4, seq_len=5, hidden_dim=8):
    return {i: torch.randn(1, seq_len, hidden_dim) for i in range(num_layers)}


def _make_concept_vector(hidden_dim=8):
    return torch.randn(hidden_dim)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestLayerThresholdResult:
    def test_creation(self):
        from steering_analysis.assumption_verification import LayerThresholdResult

        r = LayerThresholdResult(
            layer_idx=2, fraction_above=0.75, all_above=False, max_cosine=0.95, mean_cosine=0.6, threshold=0.5
        )
        assert r.layer_idx == 2
        assert r.fraction_above == 0.75
        assert r.all_above is False
        assert r.max_cosine == 0.95
        assert r.mean_cosine == 0.6
        assert r.threshold == 0.5


class TestVerificationResult:
    def test_creation(self):
        from steering_analysis.assumption_verification import VerificationResult

        result = VerificationResult(
            model_name="fake-model",
            concept="sentiment",
            steering_layers_tested=[2, 3],
            per_layer_results={},
        )
        assert result.model_name == "fake-model"
        assert result.concept == "sentiment"
        assert result.steering_layers_tested == [2, 3]
        assert result.per_layer_results == {}


# ---------------------------------------------------------------------------
# get_all_layer_activations tests
# ---------------------------------------------------------------------------


class TestGetAllLayerActivations:
    def test_returns_dict_with_all_layers(self, mock_hooked_model):
        from steering_analysis.assumption_verification import get_all_layer_activations

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        result = get_all_layer_activations(hm, "hello world")
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_activation_shape(self, mock_hooked_model):
        from steering_analysis.assumption_verification import get_all_layer_activations

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        text = "hello"
        result = get_all_layer_activations(hm, text)
        seq_len = len(text)
        hidden_dim = 8
        for layer_idx, tensor in result.items():
            assert tensor.shape == (1, seq_len, hidden_dim), (
                f"Layer {layer_idx}: expected (1, {seq_len}, {hidden_dim}), got {tensor.shape}"
            )

    def test_hooks_cleaned_up(self, mock_hooked_model):
        from steering_analysis.assumption_verification import get_all_layer_activations

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        get_all_layer_activations(hm, "test")
        for i in range(4):
            hooks = list(mock_hooked_model.model.layers[i]._forward_hooks.keys())
            assert len(hooks) == 0, f"Layer {i} still has {len(hooks)} hooks"

    def test_pythia_architecture(self, mock_pythia_model):
        from steering_analysis.assumption_verification import get_all_layer_activations

        hm = HookedModel(ModelConfig(model_name="fake-pythia"))
        result = get_all_layer_activations(hm, "hi")
        assert set(result.keys()) == {0, 1, 2, 3}
        for tensor in result.values():
            assert tensor.shape[0] == 1
            assert tensor.shape[2] == 8


# ---------------------------------------------------------------------------
# compute_cosine_similarities tests
# ---------------------------------------------------------------------------


class TestComputeCosineSimilarities:
    def test_returns_correct_keys(self):
        from steering_analysis.assumption_verification import compute_cosine_similarities

        activations = _make_layer_activations(num_layers=3, seq_len=5, hidden_dim=8)
        concept = _make_concept_vector(hidden_dim=8)
        result = compute_cosine_similarities(activations, concept)
        assert set(result.keys()) == {0, 1, 2}

    def test_output_shape(self):
        from steering_analysis.assumption_verification import compute_cosine_similarities

        seq_len = 7
        activations = _make_layer_activations(num_layers=2, seq_len=seq_len, hidden_dim=8)
        concept = _make_concept_vector(hidden_dim=8)
        result = compute_cosine_similarities(activations, concept)
        for layer_idx, tensor in result.items():
            assert tensor.shape == (seq_len,), f"Layer {layer_idx}: expected ({seq_len},), got {tensor.shape}"

    def test_cosine_range(self):
        from steering_analysis.assumption_verification import compute_cosine_similarities

        activations = _make_layer_activations(num_layers=2, seq_len=5, hidden_dim=8)
        concept = _make_concept_vector(hidden_dim=8)
        result = compute_cosine_similarities(activations, concept)
        for layer_idx, tensor in result.items():
            assert (tensor >= -1.0 - 1e-6).all() and (tensor <= 1.0 + 1e-6).all(), (
                f"Layer {layer_idx}: cosine similarities out of range"
            )

    def test_identical_vector_cosine_is_one(self):
        from steering_analysis.assumption_verification import compute_cosine_similarities

        concept = torch.tensor([1.0, 0.0, 0.0, 0.0])
        activations = {0: concept.unsqueeze(0).unsqueeze(0).expand(1, 3, -1).clone()}
        result = compute_cosine_similarities(activations, concept)
        assert torch.allclose(result[0], torch.ones(3), atol=1e-6)


# ---------------------------------------------------------------------------
# compute_threshold_violations tests
# ---------------------------------------------------------------------------


class TestComputeThresholdViolations:
    def test_returns_correct_number_of_results(self):
        from steering_analysis.assumption_verification import LayerThresholdResult, compute_threshold_violations

        cos_matrix = torch.tensor([[0.1, 0.3, 0.5], [0.7, 0.9, 0.2]])
        results = compute_threshold_violations(cos_matrix, 0.5)
        assert len(results) == 2
        assert all(isinstance(r, LayerThresholdResult) for r in results)

    def test_fraction_above(self):
        from steering_analysis.assumption_verification import compute_threshold_violations

        cos_matrix = torch.tensor([[0.1, 0.6, 0.8]])
        results = compute_threshold_violations(cos_matrix, 0.5)
        assert len(results) == 1
        assert abs(results[0].fraction_above - 2.0 / 3.0) < 1e-6

    def test_all_above_true(self):
        from steering_analysis.assumption_verification import compute_threshold_violations

        cos_matrix = torch.tensor([[0.6, 0.7, 0.8]])
        results = compute_threshold_violations(cos_matrix, 0.5)
        assert results[0].all_above is True

    def test_all_above_false(self):
        from steering_analysis.assumption_verification import compute_threshold_violations

        cos_matrix = torch.tensor([[0.6, 0.3, 0.8]])
        results = compute_threshold_violations(cos_matrix, 0.5)
        assert results[0].all_above is False

    def test_max_and_mean(self):
        from steering_analysis.assumption_verification import compute_threshold_violations

        cos_matrix = torch.tensor([[0.2, 0.5, 0.8]])
        results = compute_threshold_violations(cos_matrix, 0.5)
        assert abs(results[0].max_cosine - 0.8) < 1e-6
        assert abs(results[0].mean_cosine - 0.5) < 1e-6

    def test_layer_idx_preserved(self):
        from steering_analysis.assumption_verification import compute_threshold_violations

        cos_matrix = torch.tensor([[0.1], [0.2], [0.3]])
        results = compute_threshold_violations(cos_matrix, 0.0)
        assert [r.layer_idx for r in results] == [0, 1, 2]


# ---------------------------------------------------------------------------
# compute_empirical_thresholds tests
# ---------------------------------------------------------------------------


class TestComputeEmpiricalThresholds:
    def test_returns_requested_percentiles(self):
        from steering_analysis.assumption_verification import compute_empirical_thresholds

        data = torch.randn(5, 10)
        percentiles = [50.0, 95.0, 99.0]
        result = compute_empirical_thresholds(data, percentiles)
        assert set(result.keys()) == {50.0, 95.0, 99.0}

    def test_median_near_zero_for_standard_normal(self):
        from steering_analysis.assumption_verification import compute_empirical_thresholds

        torch.manual_seed(42)
        data = torch.randn(100, 100)
        result = compute_empirical_thresholds(data, [50.0])
        assert abs(result[50.0]) < 0.1

    def test_percentiles_are_ordered(self):
        from steering_analysis.assumption_verification import compute_empirical_thresholds

        data = torch.randn(5, 10)
        result = compute_empirical_thresholds(data, [25.0, 50.0, 75.0])
        assert result[25.0] <= result[50.0] <= result[75.0]

    def test_flattens_input(self):
        from steering_analysis.assumption_verification import compute_empirical_thresholds

        data = torch.ones(3, 4)
        result = compute_empirical_thresholds(data, [50.0])
        assert abs(result[50.0] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# run_verification tests
# ---------------------------------------------------------------------------


class TestGetSteeredActivations:
    def test_returns_dict_with_all_layers(self, mock_hooked_model):
        from steering_analysis.assumption_verification import get_steered_activations

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        sv = torch.randn(8)
        result = get_steered_activations(hm, "hello world", layer_idx=2, steering_vector=sv, scale=1.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_steered_differs_from_clean(self, mock_hooked_model):
        from steering_analysis.assumption_verification import get_all_layer_activations, get_steered_activations

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        sv = torch.randn(8)
        text = "hello"
        clean = get_all_layer_activations(hm, text)
        steered = get_steered_activations(hm, text, layer_idx=2, steering_vector=sv, scale=10.0)
        assert not clean[2].allclose(steered[2]), "Steered layer 2 should differ from clean"

    def test_steered_layer_differs_more_than_others(self, mock_hooked_model):
        from steering_analysis.assumption_verification import get_all_layer_activations, get_steered_activations

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        sv = torch.randn(8)
        text = "hello"
        clean = get_all_layer_activations(hm, text)
        steered = get_steered_activations(hm, text, layer_idx=1, steering_vector=sv, scale=10.0)
        delta_steered = (clean[1] - steered[1]).abs().mean().item()
        delta_other = (clean[0] - steered[0]).abs().mean().item()
        assert delta_steered > delta_other, "Steered layer should change more than non-steered layers"

    def test_hooks_cleaned_up(self, mock_hooked_model):
        from steering_analysis.assumption_verification import get_steered_activations

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        sv = torch.randn(8)
        get_steered_activations(hm, "test", layer_idx=2, steering_vector=sv, scale=1.0)
        for i in range(4):
            hooks = list(mock_hooked_model.model.layers[i]._forward_hooks.keys())
            assert len(hooks) == 0, f"Layer {i} still has {len(hooks)} hooks"


# ---------------------------------------------------------------------------
# generate_orthogonal_vector tests
# ---------------------------------------------------------------------------


class TestGenerateOrthogonalVector:
    def test_returns_unit_vector(self):
        from steering_analysis.assumption_verification import generate_orthogonal_vector

        concept = torch.randn(512)
        result = generate_orthogonal_vector(concept)
        assert abs(result.norm().item() - 1.0) < 1e-5

    def test_is_orthogonal_to_concept(self):
        from steering_analysis.assumption_verification import generate_orthogonal_vector

        concept = torch.randn(512)
        concept = concept / concept.norm()
        result = generate_orthogonal_vector(concept)
        cos = torch.nn.functional.cosine_similarity(
            concept.unsqueeze(0), result.unsqueeze(0)
        )
        assert abs(cos.item()) < 0.01

    def test_different_seeds_give_different_vectors(self):
        from steering_analysis.assumption_verification import generate_orthogonal_vector

        concept = torch.randn(512)
        v1 = generate_orthogonal_vector(concept, seed=42)
        v2 = generate_orthogonal_vector(concept, seed=123)
        assert not torch.allclose(v1, v2)

    def test_deterministic_with_same_seed(self):
        from steering_analysis.assumption_verification import generate_orthogonal_vector

        concept = torch.randn(512)
        v1 = generate_orthogonal_vector(concept, seed=42)
        v2 = generate_orthogonal_vector(concept, seed=42)
        assert torch.allclose(v1, v2)


# ---------------------------------------------------------------------------
# SteeringLayerResult control fields tests
# ---------------------------------------------------------------------------


class TestSteeringLayerResultControls:
    def test_control_fields_default_to_none(self):
        from steering_analysis.assumption_verification import SteeringLayerResult

        lr = SteeringLayerResult(
            steering_layer=2,
            threshold_results=[],
            empirical_thresholds={},
            cos_matrix_steered=torch.randn(2, 5),
            cos_matrix_unsteered=torch.randn(2, 5),
        )
        assert lr.cos_matrix_random_control is None
        assert lr.cos_matrix_natural is None

    def test_control_fields_can_be_set(self):
        from steering_analysis.assumption_verification import SteeringLayerResult

        lr = SteeringLayerResult(
            steering_layer=2,
            threshold_results=[],
            empirical_thresholds={},
            cos_matrix_steered=torch.randn(2, 5),
            cos_matrix_unsteered=torch.randn(2, 5),
            cos_matrix_random_control=torch.randn(2, 5),
            cos_matrix_natural=torch.randn(2, 5),
        )
        assert lr.cos_matrix_random_control is not None
        assert lr.cos_matrix_natural is not None


# ---------------------------------------------------------------------------
# Natural alignment test (steered text without steering hook)
# ---------------------------------------------------------------------------


class TestNaturalAlignment:
    def test_get_all_layer_activations_on_steered_text(self, mock_hooked_model):
        from steering_analysis.assumption_verification import (
            get_all_layer_activations,
            get_steered_activations,
        )

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        sv = torch.randn(8)
        text = "hello"

        steered_act = get_steered_activations(hm, text, 2, sv, scale=5.0)
        natural_act = get_all_layer_activations(hm, text)

        assert not steered_act[2].allclose(natural_act[2])

    def test_natural_cosine_lower_than_steered(self, mock_hooked_model):
        from steering_analysis.assumption_verification import (
            compute_cosine_similarities,
            get_all_layer_activations,
            get_steered_activations,
        )

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        sv = torch.randn(8)
        text = "hello"

        steered_act = get_steered_activations(hm, text, 2, sv, scale=10.0)
        natural_act = get_all_layer_activations(hm, text)

        concept = sv
        steered_cos = compute_cosine_similarities(steered_act, concept)
        natural_cos = compute_cosine_similarities(natural_act, concept)

        # At steering layer, forced alignment should be higher than natural
        assert steered_cos[2].mean() > natural_cos[2].mean()


# ---------------------------------------------------------------------------
# run_verification with controls tests
# ---------------------------------------------------------------------------


class TestRunVerificationWithControls:
    def test_control_data_populated(self, mock_hooked_model):
        from steering_analysis.assumption_verification import run_verification

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        vc = VerificationConfig(
            thresholds=[0.3],
            empirical_percentiles=[95.0],
            extraction_layers=[0.5],
            extraction_method="mean",
            extraction_num_pairs=2,
            num_samples=2,
            max_new_tokens=5,
            run_controls=True,
        )
        pairs = [
            ContrastPair(
                positive="great",
                negative="terrible",
                metadata=ContrastPairMetadata(
                    concept="sentiment", dataset="test", source="test", pair_index=0
                ),
            ),
            ContrastPair(
                positive="wonderful",
                negative="awful",
                metadata=ContrastPairMetadata(
                    concept="sentiment", dataset="test", source="test", pair_index=1
                ),
            ),
        ]

        from unittest.mock import patch

        with patch(
            "steering_analysis.assumption_verification.load_contrast_pairs",
            return_value=pairs,
        ):
            result = run_verification(hm, "sentiment", vc)

        for lr in result.per_layer_results.values():
            assert lr.cos_matrix_random_control is not None, "Missing random control data"
            assert lr.cos_matrix_natural is not None, "Missing natural alignment data"

    def test_controls_not_run_when_disabled(self, mock_hooked_model):
        from steering_analysis.assumption_verification import run_verification

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        vc = VerificationConfig(
            thresholds=[0.3],
            empirical_percentiles=[95.0],
            extraction_layers=[0.5],
            extraction_method="mean",
            extraction_num_pairs=2,
            num_samples=2,
            max_new_tokens=5,
            run_controls=False,
        )
        pairs = [
            ContrastPair(
                positive="great",
                negative="terrible",
                metadata=ContrastPairMetadata(
                    concept="sentiment", dataset="test", source="test", pair_index=0
                ),
            ),
            ContrastPair(
                positive="wonderful",
                negative="awful",
                metadata=ContrastPairMetadata(
                    concept="sentiment", dataset="test", source="test", pair_index=1
                ),
            ),
        ]

        from unittest.mock import patch

        with patch(
            "steering_analysis.assumption_verification.load_contrast_pairs",
            return_value=pairs,
        ):
            result = run_verification(hm, "sentiment", vc)

        for lr in result.per_layer_results.values():
            assert lr.cos_matrix_random_control is None
            assert lr.cos_matrix_natural is None


class TestRunVerification:
    def test_returns_verification_result(self, mock_hooked_model):
        from steering_analysis.assumption_verification import VerificationResult, run_verification

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        vc = VerificationConfig(
            thresholds=[0.3],
            empirical_percentiles=[95.0],
            extraction_layers=[0.5],
            extraction_method="mean",
            extraction_num_pairs=2,
            num_samples=2,
            max_new_tokens=5,
        )
        pairs = [
            ContrastPair(
                positive="great movie",
                negative="terrible film",
                metadata=ContrastPairMetadata(concept="sentiment", dataset="test", source="test", pair_index=0),
            ),
            ContrastPair(
                positive="wonderful experience",
                negative="awful experience",
                metadata=ContrastPairMetadata(concept="sentiment", dataset="test", source="test", pair_index=1),
            ),
        ]

        from unittest.mock import patch

        with patch("steering_analysis.assumption_verification.load_contrast_pairs", return_value=pairs):
            result = run_verification(hm, "sentiment", vc)

        assert isinstance(result, VerificationResult)
        assert result.model_name == "fake-model"
        assert result.concept == "sentiment"
        assert len(result.steering_layers_tested) > 0
        assert len(result.per_layer_results) > 0

    def test_per_layer_results_populated(self, mock_hooked_model):
        from steering_analysis.assumption_verification import SteeringLayerResult, run_verification

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        vc = VerificationConfig(
            thresholds=[0.3, 0.7],
            extraction_layers=[0.5],
            extraction_method="mean",
            extraction_num_pairs=2,
            num_samples=2,
            max_new_tokens=5,
        )
        pairs = [
            ContrastPair(
                positive="great",
                negative="terrible",
                metadata=ContrastPairMetadata(concept="sentiment", dataset="test", source="test", pair_index=0),
            ),
            ContrastPair(
                positive="wonderful",
                negative="awful",
                metadata=ContrastPairMetadata(concept="sentiment", dataset="test", source="test", pair_index=1),
            ),
        ]

        from unittest.mock import patch

        with patch("steering_analysis.assumption_verification.load_contrast_pairs", return_value=pairs):
            result = run_verification(hm, "sentiment", vc)

        for s_layer, lr in result.per_layer_results.items():
            assert isinstance(lr, SteeringLayerResult)
            assert lr.steering_layer == s_layer
            assert lr.cos_matrix_steered.shape[0] > 0
            assert lr.cos_matrix_unsteered.shape[0] > 0

    def test_empirical_thresholds_populated(self, mock_hooked_model):
        from steering_analysis.assumption_verification import run_verification

        config = ModelConfig(model_name="fake-model")
        hm = HookedModel(config)
        vc = VerificationConfig(
            thresholds=[0.3],
            empirical_percentiles=[90.0, 95.0],
            extraction_layers=[0.5],
            extraction_method="mean",
            extraction_num_pairs=2,
            num_samples=2,
            max_new_tokens=5,
        )
        pairs = [
            ContrastPair(
                positive="great",
                negative="terrible",
                metadata=ContrastPairMetadata(concept="sentiment", dataset="test", source="test", pair_index=0),
            ),
            ContrastPair(
                positive="wonderful",
                negative="awful",
                metadata=ContrastPairMetadata(concept="sentiment", dataset="test", source="test", pair_index=1),
            ),
        ]

        from unittest.mock import patch

        with patch("steering_analysis.assumption_verification.load_contrast_pairs", return_value=pairs):
            result = run_verification(hm, "sentiment", vc)

        for lr in result.per_layer_results.values():
            assert 90.0 in lr.empirical_thresholds
            assert 95.0 in lr.empirical_thresholds


# ---------------------------------------------------------------------------
# save_results tests
# ---------------------------------------------------------------------------


class TestSaveResults:
    def _make_result(self):
        from steering_analysis.assumption_verification import (
            LayerThresholdResult,
            SteeringLayerResult,
            VerificationResult,
        )

        lr = SteeringLayerResult(
            steering_layer=2,
            threshold_results=[
                LayerThresholdResult(0, 0.5, False, 0.9, 0.4, 0.3),
                LayerThresholdResult(1, 1.0, True, 0.95, 0.7, 0.3),
            ],
            empirical_thresholds={95.0: 0.8},
            cos_matrix_steered=torch.randn(2, 5),
            cos_matrix_unsteered=torch.randn(2, 5),
        )
        return VerificationResult(
            model_name="test-model",
            concept="sentiment",
            steering_layers_tested=[2],
            per_layer_results={2: lr},
        )

    def test_creates_output_dir(self, tmp_path):
        from steering_analysis.assumption_verification import save_results

        result = self._make_result()
        output_dir = tmp_path / "nested" / "output"
        save_results(result, output_dir, "exp1")
        assert output_dir.exists()

    def test_saves_cosine_matrices_pt(self, tmp_path):
        from steering_analysis.assumption_verification import save_results

        result = self._make_result()
        save_results(result, tmp_path, "test")
        pt_file = tmp_path / "test_cosine_matrices.pt"
        assert pt_file.exists()
        loaded = torch.load(pt_file, weights_only=True)
        assert "steered_L2" in loaded
        assert "unsteered_L2" in loaded

    def test_saves_summary_json(self, tmp_path):
        from steering_analysis.assumption_verification import save_results

        result = self._make_result()
        save_results(result, tmp_path, "test")
        summary_file = tmp_path / "test_summary.json"
        assert summary_file.exists()
        with open(summary_file) as f:
            summary = json.load(f)
        assert summary["model_name"] == "test-model"
        assert summary["concept"] == "sentiment"
        assert "per_layer_summary" in summary
        assert "2" in summary["per_layer_summary"]

    def test_saves_full_results_json(self, tmp_path):
        from steering_analysis.assumption_verification import save_results

        result = self._make_result()
        save_results(result, tmp_path, "test")
        full_file = tmp_path / "test_full_results.json"
        assert full_file.exists()
        with open(full_file) as f:
            data = json.load(f)
        assert data["model_name"] == "test-model"
        assert "per_layer_results" in data
        assert "2" in data["per_layer_results"]


# ---------------------------------------------------------------------------
# Pythia architecture fixture (inline, same as test_models.py)
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
