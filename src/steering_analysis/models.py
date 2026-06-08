import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class HookedModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        dtype = _DTYPE_MAP.get(config.dtype, torch.float16)
        device_map = "auto" if config.device == "auto" else {"": config.device}
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, device_map=device_map, dtype=dtype, trust_remote_code=config.trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def num_layers(self) -> int:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return len(self.model.gpt_neox.layers)
        raise ValueError("Could not determine number of layers")

    def resolve_layers(self, relative_layers: list[float]) -> list[int]:
        n = self.num_layers
        return [math.floor(max(0.0, min(1.0, r)) * (n - 1) + 0.5) for r in relative_layers]

    def _get_layers_module(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        raise ValueError("Could not find layers")

    def get_activations(self, texts: list[str], layers: list[int]) -> dict[int, Tensor]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        activations: dict[int, Tensor] = {}
        handles = []
        model_layers = self._get_layers_module()

        def make_hook(layer_idx: int) -> Callable:
            def hook_fn(module: Any, input: Any, output: Any) -> None:
                tensor_output = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = tensor_output.detach().clone()

            return hook_fn

        for layer_idx in layers:
            handle = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        return activations

    def generate_with_steering(
        self,
        prompt: str,
        layer_idx: int,
        steering_vector: Tensor,
        scale: float,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        steer_tokens: int | None = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model_layers = self._get_layers_module()
        step_counter = [0]

        def steering_hook(module: Any, input: Any, output: Any) -> Any:
            step_counter[0] += 1
            if steer_tokens is not None and step_counter[0] > steer_tokens:
                return output
            tensor_output = output[0] if isinstance(output, tuple) else output
            steering = steering_vector.to(device=tensor_output.device, dtype=tensor_output.dtype)
            tensor_output = tensor_output + steering * scale
            if isinstance(output, tuple):
                return (tensor_output,) + output[1:]
            return tensor_output

        handle = model_layers[layer_idx].register_forward_hook(steering_hook)
        try:
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            if steer_tokens is not None:
                gen_kwargs["use_cache"] = True
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_text = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            assert isinstance(generated_text, str)
            return generated_text
        finally:
            handle.remove()
