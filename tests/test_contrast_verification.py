"""Tests for contrast pair verification experiment.

This experiment computes cosine similarity between the positive/negative text
hidden states (used to build the steering vector) and the steering vector itself,
then replicates Assumption 1's Exp1 (token-level) and Exp2 (layer-level).
"""

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor

from steering_analysis.config import ExtractionConfig, VerificationConfig
from steering_analysis.contrast_verification import (
    ContrastVerificationResult,
    compute_pair_cosine_matrices,
    run_contrast_verification,
    save_contrast_results,
)
from steering_analysis.extract import extract_steering_vector, load_contrast_pairs
from steering_analysis.models import HookedModel


# ---------------------------------------------------------------------------
# Unit tests for compute_pair_cosine_matrices
# ---------------------------------------------------------------------------


class TestComputePairCosineMatrices:
    """Test the core cosine matrix computation for contrast pairs."""

    def test_basic_shapes(self):
        """Returns (pos_cos, neg_cos) each shaped (num_layers, total_tokens)."""
        num_layers = 4
        seq_len = 5
        num_pairs = 3
        hidden_dim = 8

        # Fake activations: one dict per pair, each layer has (1, seq_len, hidden_dim)
        pos_activations = [
            {i: torch.randn(1, seq_len, hidden_dim) for i in range(num_layers)} for _ in range(num_pairs)
        ]
        neg_activations = [
            {i: torch.randn(1, seq_len, hidden_dim) for i in range(num_layers)} for _ in range(num_pairs)
        ]
        steering_vec = torch.randn(hidden_dim)

        pos_cos, neg_cos = compute_pair_cosine_matrices(pos_activations, neg_activations, steering_vec, num_layers)

        assert pos_cos.shape == (num_layers, num_pairs * seq_len)
        assert neg_cos.shape == (num_layers, num_pairs * seq_len)

    def test_cosine_range(self):
        """All cosine similarity values are in [-1, 1]."""
        num_layers = 3
        seq_len = 4
        num_pairs = 2
        hidden_dim = 16

        pos_activations = [
            {i: torch.randn(1, seq_len, hidden_dim) for i in range(num_layers)} for _ in range(num_pairs)
        ]
        neg_activations = [
            {i: torch.randn(1, seq_len, hidden_dim) for i in range(num_layers)} for _ in range(num_pairs)
        ]
        steering_vec = torch.randn(hidden_dim)

        pos_cos, neg_cos = compute_pair_cosine_matrices(pos_activations, neg_activations, steering_vec, num_layers)

        assert pos_cos.min() >= -1.0 and pos_cos.max() <= 1.0
        assert neg_cos.min() >= -1.0 and neg_cos.max() <= 1.0

    def test_perfect_alignment(self):
        """When activation equals steering vector, cosine similarity is 1.0."""
        num_layers = 2
        seq_len = 3
        num_pairs = 1
        hidden_dim = 8

        steering_vec = torch.randn(hidden_dim)
        # Make activations exactly equal to steering vector (broadcasted)
        pos_activations = [
            {
                i: steering_vec.unsqueeze(0).unsqueeze(0).expand(1, seq_len, hidden_dim).clone()
                for i in range(num_layers)
            }
        ]
        neg_activations = [{i: torch.randn(1, seq_len, hidden_dim) for i in range(num_layers)}]

        pos_cos, _ = compute_pair_cosine_matrices(pos_activations, neg_activations, steering_vec, num_layers)

        # All positive cosine values should be ~1.0
        assert torch.allclose(pos_cos, torch.ones_like(pos_cos), atol=1e-5)

    def test_orthogonal_vectors_near_zero(self):
        """When activations are orthogonal to steering vector, cosine ~0."""
        num_layers = 1
        seq_len = 2
        num_pairs = 1
        hidden_dim = 4

        steering_vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # Orthogonal: only non-zero in dimensions perpendicular to steering vec
        pos_activations = [{0: torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])}]
        neg_activations = [{0: torch.randn(1, seq_len, hidden_dim)}]

        pos_cos, _ = compute_pair_cosine_matrices(pos_activations, neg_activations, steering_vec, num_layers)

        assert torch.allclose(pos_cos, torch.zeros_like(pos_cos), atol=1e-5)

    def test_empty_activations_raises(self):
        """Empty activation list raises ValueError."""
        steering_vec = torch.randn(8)
        with pytest.raises(ValueError, match="No activations"):
            compute_pair_cosine_matrices([], [], steering_vec, 4)

    def test_mismatched_counts_raises(self):
        """Mismatched positive/negative activation counts raises ValueError."""
        num_layers = 2
        hidden_dim = 8
        pos_activations = [{i: torch.randn(1, 3, hidden_dim) for i in range(num_layers)}]
        neg_activations = [
            {i: torch.randn(1, 3, hidden_dim) for i in range(num_layers)},
            {i: torch.randn(1, 3, hidden_dim) for i in range(num_layers)},
        ]
        steering_vec = torch.randn(hidden_dim)
        with pytest.raises(ValueError, match="Mismatched"):
            compute_pair_cosine_matrices(pos_activations, neg_activations, steering_vec, num_layers)


# ---------------------------------------------------------------------------
# Integration tests with fake model
# ---------------------------------------------------------------------------


class TestRunContrastVerification:
    """Integration tests using the fake model from conftest."""

    def test_returns_result_with_correct_fields(self, mock_hooked_model, sample_contrast_pairs):
        """run_contrast_verification returns a ContrastVerificationResult with expected fields."""
        from steering_analysis.config import ModelConfig

        model = HookedModel(ModelConfig(model_name="fake/model"))
        config = VerificationConfig(
            thresholds=[0.3, 0.5],
            extraction_num_pairs=3,
            num_samples=3,
            seed=42,
        )

        result = run_contrast_verification(model, "sentiment", config)

        assert isinstance(result, ContrastVerificationResult)
        assert result.model_name == "fake/model"
        assert result.concept == "sentiment"
        assert len(result.steering_layers_tested) > 0
        assert len(result.per_layer_results) > 0

    def test_per_layer_has_exp1_and_exp2(self, mock_hooked_model):
        """Each layer result has experiment1 and experiment2 verdicts."""
        from steering_analysis.config import ModelConfig

        model = HookedModel(ModelConfig(model_name="fake/model"))
        config = VerificationConfig(
            thresholds=[0.3, 0.5],
            extraction_num_pairs=2,
            num_samples=2,
            seed=42,
        )

        result = run_contrast_verification(model, "sentiment", config)

        for s_layer, lr in result.per_layer_results.items():
            assert isinstance(lr.experiment1_verdicts, dict)
            assert isinstance(lr.experiment2_verdicts, dict)
            for t in config.thresholds:
                assert t in lr.experiment1_verdicts
                assert t in lr.experiment2_verdicts
                assert len(lr.experiment1_verdicts[t]) > 0

    def test_cosine_matrices_have_correct_shape(self, mock_hooked_model):
        """Cosine matrices have shape (num_layers, total_tokens)."""
        from steering_analysis.config import ModelConfig

        model = HookedModel(ModelConfig(model_name="fake/model"))
        num_pairs = 3
        config = VerificationConfig(
            thresholds=[0.5],
            extraction_num_pairs=num_pairs,
            num_samples=num_pairs,
            seed=42,
        )

        result = run_contrast_verification(model, "sentiment", config)

        for s_layer, lr in result.per_layer_results.items():
            assert lr.cos_matrix_positive.ndim == 2
            assert lr.cos_matrix_negative.ndim == 2
            # First dim = num_layers
            assert lr.cos_matrix_positive.shape[0] == model.num_layers
            assert lr.cos_matrix_negative.shape[0] == model.num_layers


# ---------------------------------------------------------------------------
# Tests for save_contrast_results
# ---------------------------------------------------------------------------


class TestSaveContrastResults:
    """Test result serialization."""

    def test_creates_files(self, tmp_path, mock_hooked_model):
        """save_contrast_results creates .pt, _summary.json, and _full_results.json."""
        from steering_analysis.config import ModelConfig
        from steering_analysis.contrast_verification import ContrastLayerResult, ContrastVerificationResult

        num_layers = 4
        num_tokens = 10

        layer_result = ContrastLayerResult(
            steering_layer=0,
            cos_matrix_positive=torch.randn(num_layers, num_tokens),
            cos_matrix_negative=torch.randn(num_layers, num_tokens),
            experiment1_verdicts={0.5: []},
            experiment2_verdicts={0.5: None},
            empirical_thresholds={95.0: 0.3},
        )
        # Add a mock experiment2 verdict
        from steering_analysis.assumption_verification import LayerExistenceVerdict

        layer_result.experiment2_verdicts[0.5] = LayerExistenceVerdict(
            threshold=0.5,
            exists_steered_layer=True,
            exists_unsteered_layer=False,
            steered_layers_above=[2],
            unsteered_layers_above=[],
            assumption_holds=True,
        )

        result = ContrastVerificationResult(
            model_name="fake/model",
            concept="sentiment",
            steering_layers_tested=[0],
            per_layer_results={0: layer_result},
        )

        save_contrast_results(result, tmp_path, "test_label")

        assert (tmp_path / "test_label_cosine_matrices.pt").exists()
        assert (tmp_path / "test_label_summary.json").exists()
        assert (tmp_path / "test_label_full_results.json").exists()

    def test_summary_json_parseable(self, tmp_path):
        """Summary JSON is valid and contains expected keys."""
        from steering_analysis.contrast_verification import (
            ContrastLayerResult,
            ContrastVerificationResult,
        )
        from steering_analysis.assumption_verification import LayerExistenceVerdict

        num_layers = 2
        num_tokens = 5
        layer_result = ContrastLayerResult(
            steering_layer=2,
            cos_matrix_positive=torch.randn(num_layers, num_tokens),
            cos_matrix_negative=torch.randn(num_layers, num_tokens),
            experiment1_verdicts={},
            experiment2_verdicts={
                0.5: LayerExistenceVerdict(
                    threshold=0.5,
                    exists_steered_layer=True,
                    exists_unsteered_layer=False,
                    steered_layers_above=[1],
                    unsteered_layers_above=[],
                    assumption_holds=True,
                )
            },
            empirical_thresholds={95.0: 0.42},
        )

        result = ContrastVerificationResult(
            model_name="test-model",
            concept="refusal",
            steering_layers_tested=[2],
            per_layer_results={2: layer_result},
        )

        save_contrast_results(result, tmp_path, "test")

        with open(tmp_path / "test_summary.json") as f:
            data = json.load(f)

        assert data["model_name"] == "test-model"
        assert data["concept"] == "refusal"
        assert "2" in data["per_layer_summary"]
