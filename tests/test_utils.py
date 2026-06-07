from steering_analysis.utils import safe_model_name, sample_with_seed


def test_sample_with_seed_returns_k_items():
    result = sample_with_seed(list(range(100)), 5, seed=42)
    assert len(result) == 5


def test_sample_with_seed_deterministic():
    r1 = sample_with_seed(list(range(100)), 5, seed=42)
    r2 = sample_with_seed(list(range(100)), 5, seed=42)
    assert r1 == r2


def test_sample_with_seed_different_seed():
    r1 = sample_with_seed(list(range(100)), 5, seed=42)
    r2 = sample_with_seed(list(range(100)), 5, seed=99)
    assert r1 != r2


def test_sample_with_seed_k_exceeds_len():
    result = sample_with_seed([1, 2, 3], 10)
    assert len(result) == 3
    assert sorted(result) == [1, 2, 3]


def test_safe_model_name_replaces_slash():
    assert safe_model_name("Qwen/Qwen3-1.7B") == "Qwen_Qwen3-1.7B"


def test_safe_model_name_no_slash():
    assert safe_model_name("gpt2") == "gpt2"
