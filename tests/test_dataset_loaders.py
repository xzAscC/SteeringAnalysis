from __future__ import annotations

from unittest.mock import patch

import pytest

from steering_analysis.types import ContrastPair

# ---------------------------------------------------------------------------
# Fakes that mimic HuggingFace Dataset / DatasetDict interfaces
# ---------------------------------------------------------------------------


class FakeDataset:
    """Minimal HuggingFace Dataset-like object (iterable, subscriptable)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


class FakeDatasetDict(dict):
    """Minimal HuggingFace DatasetDict (dict of split -> FakeDataset)."""

    pass


def _make_sentiment_rows(n_pos: int = 20, n_neg: int = 20) -> list[dict]:
    rows = [{"sentence": f"good sentence {i}", "label": 1} for i in range(n_pos)]
    rows += [{"sentence": f"bad sentence {i}", "label": 0} for i in range(n_neg)]
    return rows


def _make_refusal_rows(n: int = 20) -> list[dict]:
    return [{"prompt": f"prompt {i}"} for i in range(n)]


def _make_polite_rows(n_polite: int = 20, n_impolite: int = 20) -> list[dict]:
    rows = [{"text": f"kind text {i}", "label": "polite"} for i in range(n_polite)]
    rows += [{"text": f"rude text {i}", "label": "impolite"} for i in range(n_impolite)]
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("steering_analysis.extract.load_dataset")
def test_load_sentiment_pairs(mock_load_dataset):
    fake_ds = FakeDatasetDict({"train": FakeDataset(_make_sentiment_rows())})
    mock_load_dataset.return_value = fake_ds

    from steering_analysis.extract import load_contrast_pairs

    pairs = load_contrast_pairs("sentiment", num_pairs=5)

    assert len(pairs) == 5
    for pair in pairs:
        assert isinstance(pair, ContrastPair)
        # Positive should start with "good" (label==1), negative with "bad" (label==0)
        assert pair.positive.startswith("good sentence")
        assert pair.negative.startswith("bad sentence")
        assert pair.metadata["concept"] == "sentiment"


@patch("steering_analysis.extract.load_dataset")
def test_load_refusal_pairs(mock_load_dataset):
    benign_rows = _make_refusal_rows()
    harmful_rows = _make_refusal_rows()

    benign_ds = FakeDatasetDict({"train": FakeDataset(benign_rows)})
    harmful_ds = FakeDatasetDict({"train": FakeDataset(harmful_rows)})

    def side_effect(name, *args, **kwargs):
        if "benign" in name:
            return benign_ds
        if "harmful" in name:
            return harmful_ds
        raise ValueError(f"Unexpected dataset: {name}")

    mock_load_dataset.side_effect = side_effect

    from steering_analysis.extract import load_contrast_pairs

    pairs = load_contrast_pairs("refusal", num_pairs=5)

    assert len(pairs) == 5
    for pair in pairs:
        assert isinstance(pair, ContrastPair)
        assert pair.metadata["concept"] == "refusal"


@patch("steering_analysis.extract.load_dataset")
def test_load_polite_pairs(mock_load_dataset):
    fake_ds = FakeDataset(_make_polite_rows())
    mock_load_dataset.return_value = fake_ds

    from steering_analysis.extract import load_contrast_pairs

    pairs = load_contrast_pairs("polite", num_pairs=5)

    assert len(pairs) == 5
    for pair in pairs:
        assert isinstance(pair, ContrastPair)
        # Positive = polite, negative = impolite
        assert pair.positive.startswith("kind text")
        assert pair.negative.startswith("rude text")
        assert pair.metadata["concept"] == "polite"


def test_load_invalid_concept():
    from steering_analysis.extract import load_contrast_pairs

    with pytest.raises(ValueError, match="Invalid concept"):
        load_contrast_pairs("invalid", num_pairs=5)


@patch("steering_analysis.extract.load_dataset")
def test_pair_metadata_fields(mock_load_dataset):
    fake_ds = FakeDatasetDict({"train": FakeDataset(_make_sentiment_rows())})
    mock_load_dataset.return_value = fake_ds

    from steering_analysis.extract import load_contrast_pairs

    pairs = load_contrast_pairs("sentiment", num_pairs=3)

    required_keys = {"concept", "dataset", "source", "pair_index"}
    for pair in pairs:
        assert required_keys.issubset(pair.metadata.keys())


@patch("steering_analysis.extract.load_dataset")
def test_deterministic_loading(mock_load_dataset):
    fake_ds = FakeDatasetDict({"train": FakeDataset(_make_sentiment_rows())})
    mock_load_dataset.return_value = fake_ds

    from steering_analysis.extract import load_contrast_pairs

    run1 = load_contrast_pairs("sentiment", num_pairs=5, seed=42)
    run2 = load_contrast_pairs("sentiment", num_pairs=5, seed=42)

    assert len(run1) == len(run2)
    for p1, p2 in zip(run1, run2):
        assert p1.positive == p2.positive
        assert p1.negative == p2.negative
