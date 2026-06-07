import pytest
import torch

from steering_analysis.extract import _resolve_aggregator, mean_aggregator, pca_aggregator


class TestMeanAggregator:
    def test_mean_aggregator_known_values(self):
        pos = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        neg = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        result = mean_aggregator(pos, neg)
        expected = torch.tensor([1.0, -1.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_mean_aggregator_shape(self):
        pos = torch.randn(3, 8)
        neg = torch.randn(3, 8)
        result = mean_aggregator(pos, neg)
        assert result.shape == (8,)

    def test_mean_aggregator_single_sample(self):
        pos = torch.randn(1, 4)
        neg = torch.randn(1, 4)
        result = mean_aggregator(pos, neg)
        assert result.shape == (4,)


class TestPCAAggregator:
    def test_pca_aggregator_shape(self):
        pos = torch.randn(5, 8)
        neg = torch.randn(5, 8)
        result = pca_aggregator(pos, neg)
        assert result.shape == (8,)

    def test_pca_aggregator_unit_norm(self):
        pos = torch.randn(5, 8)
        neg = torch.randn(5, 8)
        result = pca_aggregator(pos, neg)
        assert abs(torch.norm(result).item() - 1.0) < 1e-5

    def test_pca_aggregator_separation_direction(self):
        torch.manual_seed(42)
        shared_noise = torch.randn(20, 3) * 0.1
        pos = torch.cat([torch.randn(20, 1) + 3.0, shared_noise], dim=1)
        neg = torch.cat([torch.randn(20, 1) - 3.0, shared_noise], dim=1)
        result = pca_aggregator(pos, neg)
        assert abs(result[0].item()) > 0.8


class TestResolveAggregator:
    def test_resolve_aggregator_mean(self):
        fn = _resolve_aggregator("mean")
        assert fn is mean_aggregator

    def test_resolve_aggregator_pca(self):
        fn = _resolve_aggregator("pca")
        assert fn is pca_aggregator

    def test_resolve_aggregator_invalid(self):
        with pytest.raises(ValueError, match="Unsupported method"):
            _resolve_aggregator("invalid")
