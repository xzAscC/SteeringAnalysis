from steering_analysis.config import (
    VALID_CONCEPTS,
    ConceptConfig,
    ExtractionConfig,
    ModelConfig,
    SteeringConfig,
)


def test_valid_concepts():
    assert VALID_CONCEPTS == ("refusal", "sentiment", "polite")


def test_model_config_defaults():
    cfg = ModelConfig(model_name="test")
    assert cfg.device == "auto"
    assert cfg.dtype == "float16"
    assert cfg.trust_remote_code is False


def test_model_config_allenai_trust():
    cfg = ModelConfig(model_name="allenai/something")
    assert cfg.trust_remote_code is True


def test_extraction_config_defaults():
    cfg = ExtractionConfig()
    assert cfg.method == "mean"
    assert cfg.batch_size == 8
    assert cfg.num_pairs == 50
    assert cfg.layers == [0.4, 0.5, 0.6, 0.7, 0.8]
    assert cfg.read_token_index == -1


def test_concept_config():
    cfg = ConceptConfig(concept_name="sentiment", dataset_name="sentiment", num_pairs=50)
    assert cfg.concept_name == "sentiment"
    assert cfg.dataset_name == "sentiment"
    assert cfg.num_pairs == 50


def test_default_num_pairs_is_50():
    assert ExtractionConfig().num_pairs == 50


def test_default_read_token_index_is_minus_1():
    assert ExtractionConfig().read_token_index == -1


def test_steering_config_defaults():
    cfg = SteeringConfig()
    assert cfg.multipliers == [0.01, 0.1, 1.0, 10.0]
    assert cfg.num_samples == 10
    assert cfg.seed == 42
    assert cfg.max_new_tokens == 100
    assert cfg.temperature == 0.0
    assert cfg.steer_tokens is None


def test_steering_config_custom_steer_tokens():
    cfg = SteeringConfig(steer_tokens=5)
    assert cfg.steer_tokens == 5


def test_steering_config_custom_multipliers():
    cfg = SteeringConfig(multipliers=[0.5, 2.0])
    assert cfg.multipliers == [0.5, 2.0]
