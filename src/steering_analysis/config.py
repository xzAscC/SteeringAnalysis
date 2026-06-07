from dataclasses import dataclass, field

VALID_CONCEPTS: tuple[str, ...] = ("refusal", "sentiment", "polite")
DEFAULT_NUM_PAIRS: int = 50


@dataclass
class ModelConfig:
    model_name: str
    device: str = "auto"
    dtype: str = "float16"
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.model_name.startswith("allenai/"):
            self.trust_remote_code = True


@dataclass
class ExtractionConfig:
    layers: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8])
    method: str = "mean"
    batch_size: int = 8
    read_token_index: int = -1
    num_pairs: int = 50
    seed: int = 42


@dataclass
class ConceptConfig:
    concept_name: str
    dataset_name: str
    num_pairs: int = 50


@dataclass
class SteeringConfig:
    multipliers: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    num_samples: int = 10
    seed: int = 42
    max_new_tokens: int = 100
    temperature: float = 0.0
    steer_tokens: int | None = None


@dataclass
class VerificationConfig:
    thresholds: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    empirical_percentiles: list[float] = field(default_factory=lambda: [95.0, 99.0])
    extraction_layers: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8])
    extraction_method: str = "mean"
    extraction_num_pairs: int = 50
    steering_multiplier: float = 1.0
    max_new_tokens: int = 100
    temperature: float = 0.0
    num_samples: int = 5
    seed: int = 42
