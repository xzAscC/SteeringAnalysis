from steering_analysis.config import (
    VALID_CONCEPTS,
    ConceptConfig,
    ExtractionConfig,
    ModelConfig,
    SteeringConfig,
    VerificationConfig,
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


def test_verification_config_defaults():
    cfg = VerificationConfig()
    assert cfg.thresholds == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert cfg.empirical_percentiles == [95.0, 99.0]
    assert cfg.extraction_layers == [0.4, 0.5, 0.6, 0.7, 0.8]
    assert cfg.extraction_method == "mean"
    assert cfg.extraction_num_pairs == 50
    assert cfg.steering_multiplier == 1.0
    assert cfg.max_new_tokens == 100
    assert cfg.temperature == 0.0
    assert cfg.num_samples == 5
    assert cfg.seed == 42


def test_verification_config_custom():
    cfg = VerificationConfig(
        thresholds=[0.5],
        empirical_percentiles=[90.0],
        steering_multiplier=2.0,
        num_samples=10,
    )
    assert cfg.thresholds == [0.5]
    assert cfg.empirical_percentiles == [90.0]
    assert cfg.steering_multiplier == 2.0
    assert cfg.num_samples == 10


# ---------------------------------------------------------------------------
# steering_method field tests
# ---------------------------------------------------------------------------


def test_steering_config_default_steering_method_is_additive():
    cfg = SteeringConfig()
    assert cfg.steering_method == "additive"


def test_steering_config_custom_steering_method():
    cfg = SteeringConfig(steering_method="angular")
    assert cfg.steering_method == "angular"


def test_verification_config_default_steering_method_is_additive():
    cfg = VerificationConfig()
    assert cfg.steering_method == "additive"


def test_verification_config_custom_steering_method():
    cfg = VerificationConfig(steering_method="angular")
    assert cfg.steering_method == "angular"


def test_verification_config_default_steer_tokens_is_none():
    cfg = VerificationConfig()
    assert cfg.steer_tokens is None


def test_verification_config_custom_steer_tokens():
    cfg = VerificationConfig(steer_tokens=10)
    assert cfg.steer_tokens == 10
