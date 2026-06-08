from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


class FakeTokenizer:
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


class FakeLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return (self.linear(x),)


class FakeCausalLM(nn.Module):
    def __init__(self, hidden_dim: int = 8, num_layers: int = 4, vocab_size: int = 32):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer(hidden_dim) for _ in range(num_layers)])
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)[0]
        return SimpleNamespace(logits=x)

    def generate(self, input_ids=None, max_new_tokens=20, pad_token_id=None, **kwargs):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            x = self.model.embed_tokens(generated)
            for layer in self.model.layers:
                x = layer(x)[0]
            logits = x[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    @property
    def device(self):
        return next(self.parameters()).device


@pytest.fixture
def sample_contrast_pairs():
    from steering_analysis.types import ContrastPair, ContrastPairMetadata

    pairs = [
        ContrastPair(
            positive="great movie",
            negative="terrible film",
            metadata=ContrastPairMetadata(concept="sentiment", dataset="test", source="test", pair_index=i),
        )
        for i, (p, n) in enumerate(
            [
                ("great movie", "terrible film"),
                ("wonderful experience", "awful experience"),
                ("highly recommended", "not recommended"),
                ("excellent quality", "poor quality"),
                ("love this product", "hate this product"),
            ]
        )
    ]
    return pairs


@pytest.fixture
def mock_hooked_model(monkeypatch):
    import transformers

    model = FakeCausalLM(hidden_dim=8, num_layers=4)
    tokenizer = FakeTokenizer()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    return model
