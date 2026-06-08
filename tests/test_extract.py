import pytest
import torch

from steering_analysis.config import ExtractionConfig, ModelConfig
from steering_analysis.extract import extract_steering_vector
from steering_analysis.models import HookedModel
from steering_analysis.types import ContrastPair, ContrastPairMetadata, SteeringVector


@pytest.fixture
def contrast_pairs():
    """Working contrast pairs with empty metadata (overrides broken conftest fixture)."""
    return [
        ContrastPair(positive="great movie", negative="terrible film", metadata=ContrastPairMetadata()),
        ContrastPair(positive="wonderful experience", negative="awful experience", metadata=ContrastPairMetadata()),
        ContrastPair(positive="highly recommended", negative="not recommended", metadata=ContrastPairMetadata()),
        ContrastPair(positive="excellent quality", negative="poor quality", metadata=ContrastPairMetadata()),
        ContrastPair(positive="love this product", negative="hate this product", metadata=ContrastPairMetadata()),
    ]


def _make_model(mock_hooked_model):
    config = ModelConfig(model_name="fake-model")
    return HookedModel(config)


class TestExtractSteeringVector:
    def test_returns_steering_vector(self, mock_hooked_model, contrast_pairs):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=8)
        result = extract_steering_vector(hm, contrast_pairs, ext_config)
        assert isinstance(result, SteeringVector)

    def test_has_all_requested_layers(self, mock_hooked_model, contrast_pairs):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=8)
        result = extract_steering_vector(hm, contrast_pairs, ext_config)
        expected_layers = hm.resolve_layers([0.5])
        assert list(result.layer_activations.keys()) == expected_layers

    def test_activations_shape(self, mock_hooked_model, contrast_pairs):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=8)
        result = extract_steering_vector(hm, contrast_pairs, ext_config)
        for layer_idx, activation in result.layer_activations.items():
            assert activation.shape == (8,), f"Layer {layer_idx} activation shape was {activation.shape}"

    def test_model_name_matches(self, mock_hooked_model, contrast_pairs):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=8)
        result = extract_steering_vector(hm, contrast_pairs, ext_config)
        assert result.model_name == "fake-model"

    def test_concept_is_unknown_when_metadata_empty(self, mock_hooked_model, contrast_pairs):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=8)
        result = extract_steering_vector(hm, contrast_pairs, ext_config)
        assert result.concept == "unknown"

    def test_method_matches_mean(self, mock_hooked_model, contrast_pairs):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=8)
        result = extract_steering_vector(hm, contrast_pairs, ext_config)
        assert result.method == "mean"

    def test_method_matches_pca(self, mock_hooked_model, contrast_pairs):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="pca", batch_size=8)
        result = extract_steering_vector(hm, contrast_pairs, ext_config)
        assert result.method == "pca"
        for layer_idx, activation in result.layer_activations.items():
            assert activation.shape == (8,), f"Layer {layer_idx} PCA activation shape was {activation.shape}"
            assert torch.isfinite(activation).all(), f"Layer {layer_idx} has non-finite values"

    def test_non_zero_vector(self, mock_hooked_model, contrast_pairs):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=8)
        result = extract_steering_vector(hm, contrast_pairs, ext_config)
        for layer_idx, activation in result.layer_activations.items():
            assert not torch.allclose(activation, torch.zeros_like(activation)), (
                f"Layer {layer_idx} steering vector is all zeros"
            )

    def test_empty_pairs_raises(self, mock_hooked_model):
        hm = _make_model(mock_hooked_model)
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=8)
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_steering_vector(hm, [], ext_config)

    def test_extract_uses_real_last_token_not_padded(self, mock_hooked_model):
        """When read_token_index=-1, extraction uses attention-masked real last token,
        not the padded column."""
        from unittest.mock import patch

        hm = _make_model(mock_hooked_model)
        pairs = [
            ContrastPair(positive="a", negative="b", metadata=ContrastPairMetadata()),
            ContrastPair(positive="abcde", negative="fghij", metadata=ContrastPairMetadata()),
        ]
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=2, read_token_index=-1)

        # "a"/"b" are 1 token padded to 5. Real last = position 0.
        # "abcde"/"fghij" are 5 tokens, no padding. Real last = position 4.
        # resolve_layers([0.5]) for a 4-layer model returns [2].
        torch.manual_seed(0)
        fake_pos = {2: torch.randn(2, 5, 8)}
        fake_neg = {2: torch.randn(2, 5, 8)}
        # Mark padded positions with opposite-sign large values so pos-neg won't cancel
        fake_pos[2][0, 1:, :] = 999.0
        fake_neg[2][0, 1:, :] = -999.0

        with patch.object(hm, "get_activations", side_effect=[fake_pos, fake_neg]):
            result = extract_steering_vector(hm, pairs, ext_config)

        vec = result.layer_activations[2]
        # With the bug: for "a" vs "b", pos-neg at padded position = 999 - (-999) = 1998.
        # With the fix: for "a" vs "b", pos-neg at real last (position 0) = small random diff.
        assert vec.abs().max() < 100, (
            f"Vector max abs value is {vec.abs().max():.1f}, likely reading padded positions instead of real last token"
        )

    def test_extract_negative_index_uses_relative_offset(self, mock_hooked_model):
        """read_token_index=-2 should select the second-to-last real token, not the last."""
        from unittest.mock import patch

        hm = _make_model(mock_hooked_model)
        # "abcde" has 5 real tokens (positions 0-4). read_token_index=-2 → position 3.
        pairs = [
            ContrastPair(positive="abcde", negative="fghij", metadata=ContrastPairMetadata()),
        ]
        ext_config = ExtractionConfig(layers=[0.5], method="mean", batch_size=1, read_token_index=-2)

        torch.manual_seed(0)
        fake_pos = {2: torch.randn(1, 5, 8)}
        fake_neg = {2: torch.randn(1, 5, 8)}
        # Mark positions we do NOT want selected with large values
        # Position 4 (last real token) should NOT be selected by -2
        fake_pos[2][0, 4, :] = 999.0
        fake_neg[2][0, 4, :] = -999.0

        with patch.object(hm, "get_activations", side_effect=[fake_pos, fake_neg]):
            result = extract_steering_vector(hm, pairs, ext_config)

        vec = result.layer_activations[2]
        # If -2 incorrectly selects position 4 (last token), pos-neg = 999-(-999) = 1998
        # If -2 correctly selects position 3 (second-to-last), pos-neg = small random diff
        assert vec.abs().max() < 100, (
            f"Vector max abs value is {vec.abs().max():.1f}, likely selecting last token instead of second-to-last for read_token_index=-2"
        )
