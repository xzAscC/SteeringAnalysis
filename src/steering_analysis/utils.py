import random


def sample_with_seed[T](items: list[T], k: int, seed: int = 42) -> list[T]:
    rng = random.Random(seed)
    return rng.sample(items, min(k, len(items)))


def safe_model_name(name: str) -> str:
    return name.replace("/", "_")
