from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DataType, DeviceType
from ..libllaisys.models import LlaisysQwen2Config

import ctypes
import json
from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with (model_path / "config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)

        native_config = LlaisysQwen2Config(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            eos_token_id=config["eos_token_id"],
            device_type=int(device),
            device_id=0,
        )
        self._model = LIB_LLAISYS.llaisysQwen2Create(ctypes.byref(native_config))
        if not self._model:
            raise RuntimeError("Failed to create the native Qwen2 model")

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as data:
                for name in data.keys():
                    tensor = data.get_tensor(name).contiguous()
                    if str(tensor.dtype) != "torch.bfloat16":
                        raise TypeError(f"Unsupported weight dtype for {name}: {tensor.dtype}")
                    shape = (ctypes.c_size_t * tensor.ndim)(*tensor.shape)
                    loaded = LIB_LLAISYS.llaisysQwen2LoadWeight(
                        self._model,
                        name.encode("utf-8"),
                        ctypes.c_void_p(tensor.data_ptr()),
                        shape,
                        tensor.ndim,
                        int(DataType.BF16),
                    )
                    if not loaded:
                        raise RuntimeError(f"Failed to load model weight: {name}")

        if not LIB_LLAISYS.llaisysQwen2Finalize(self._model):
            raise RuntimeError("Qwen2 model is missing one or more required weights")

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2Destroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        if max_new_tokens is None:
            max_new_tokens = 128
        input_ids = [int(token) for token in inputs]
        input_buffer = (ctypes.c_int64 * len(input_ids))(*input_ids)
        capacity = len(input_ids) + max_new_tokens
        output_buffer = (ctypes.c_int64 * capacity)()
        output_count = LIB_LLAISYS.llaisysQwen2Generate(
            self._model,
            input_buffer,
            len(input_ids),
            max_new_tokens,
            output_buffer,
            capacity,
        )
        if output_count == 0:
            raise RuntimeError("Native Qwen2 generation failed")
        return list(output_buffer[:output_count])
