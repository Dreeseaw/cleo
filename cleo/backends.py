"""Generation backends with lazy heavy imports."""
from __future__ import annotations

import importlib.util
import warnings

DEFAULT_REPO = "dreeseaw/cleo"
DEFAULT_GGUF = "cleo-Q8_0.gguf"   # legacy HF alias for llama-cpp-python compatible quant


def download_gguf(repo_id: str = DEFAULT_REPO, filename: str = DEFAULT_GGUF,
                  revision: str | None = None) -> str:
    """Download or reuse the cached default GGUF and return its local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, revision=revision)


class GGUFBackend:
    """llama-cpp-python backend for CPU or GPU use."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 8, n_gpu_layers: int = 0):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)

    def generate(self, prompt: str, max_new_tokens: int = 256, *, sample: bool = False,
                 temperature: float = 0.0, top_p: float = 0.95, seed: int | None = None) -> str:
        kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": float(temperature if sample else 0.0),
            "top_p": float(top_p),
            "stop": ["\n\n\n"],
        }
        if seed is not None:
            kwargs["seed"] = int(seed)
        out = self.llm(prompt, **kwargs)
        return out["choices"][0]["text"]


def _device_kind(device: str) -> str:
    return str(device).split(":", 1)[0]


def resolve_hf_device(torch, device: str | None = None) -> str:
    """Resolve an HF runtime device from the current PyTorch install."""
    if device not in (None, "auto"):
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _auto_hf_dtype(torch, device: str):
    kind = _device_kind(device)
    if kind == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if kind == "xpu":
        is_bf16_supported = getattr(getattr(torch, "xpu", None), "is_bf16_supported", None)
        return torch.bfloat16 if callable(is_bf16_supported) and is_bf16_supported() else torch.float16
    if kind == "mps":
        return torch.float16
    return torch.float32


def _flash_linear_attention_available() -> bool:
    try:
        from transformers.utils.import_utils import is_flash_linear_attention_available
    except Exception:
        return False
    try:
        return bool(is_flash_linear_attention_available())
    except Exception:
        return False


def _raise_if_unsafe_fla_device(device: str) -> None:
    kind = _device_kind(device)
    if kind == "cuda" or not _flash_linear_attention_available():
        return
    raise RuntimeError(
        "Cleo's current HF model uses Qwen3.5 linear-attention layers. This Python "
        "environment has the FLA/Triton fast path installed, but that fast path is "
        f"not safe on device={device!r}. Use a CUDA PyTorch runtime, remove FLA so "
        "Transformers can use its torch fallback, or explicitly use a known-good GGUF "
        "with Cleo.from_gguf(...)."
    )


def _bnb_quantization_config(quantization: str | None, device: str):
    if quantization in (None, "", "none", "fp16", "bf16"):
        return None
    if quantization != "int8":
        raise ValueError("quantization must be one of: None, 'none', or 'int8'")
    if _device_kind(device) != "cuda":
        raise ValueError("quantization='int8' currently requires a CUDA device")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "quantization='int8' requires bitsandbytes support. Install with "
            '`pip install "cleo-sql[hf,int8] @ git+https://github.com/Dreeseaw/cleo.git@master"`.'
        )
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise ImportError(
            "quantization='int8' requires bitsandbytes support. Install with "
            '`pip install "cleo-sql[hf,int8] @ git+https://github.com/Dreeseaw/cleo.git@master"`.'
        ) from exc
    return BitsAndBytesConfig(load_in_8bit=True)


class HFBackend:
    """Transformers backend with automatic device and dtype selection."""

    def __init__(self, model: str, device: str | None = None, quantization: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = resolve_hf_device(torch, device)
        self.device_kind = _device_kind(self.device)
        self.dtype = _auto_hf_dtype(torch, self.device)
        self.quantization = quantization or "none"
        _raise_if_unsafe_fla_device(self.device)
        qconfig = _bnb_quantization_config(quantization, self.device)
        if self.device_kind == "cpu":
            warnings.warn(
                "Cleo.from_hf() selected CPU. The current hardel is intended for accelerator-backed "
                "Transformers inference and may be slow on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
        kwargs = {"trust_remote_code": True}
        if qconfig is None:
            kwargs["dtype"] = self.dtype
        else:
            kwargs["quantization_config"] = qconfig
            kwargs["device_map"] = {"": self.device}
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if qconfig is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256, *, sample: bool = False,
                 temperature: float = 0.0, top_p: float = 0.95, seed: int | None = None) -> str:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        if seed is not None:
            self._torch.manual_seed(int(seed))
            if self.device_kind == "cuda":
                self._torch.cuda.manual_seed_all(int(seed))
            elif self.device_kind == "xpu":
                self._torch.xpu.manual_seed_all(int(seed))
            elif self.device_kind == "mps" and hasattr(self._torch, "mps"):
                self._torch.mps.manual_seed(int(seed))
        gen_kwargs = {
            "do_sample": bool(sample),
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tok.pad_token_id,
        }
        if sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
        with self._torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
