"""Generation backends with lazy heavy imports."""
from __future__ import annotations

DEFAULT_REPO = "dreeseaw/cleo"
DEFAULT_GGUF = "cleo-Q8_0.gguf"   # stable HF alias for llama-cpp-python compatible quant


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


class HFBackend:
    """transformers backend (bf16). Use a GPU for speed."""

    def __init__(self, model: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device).eval()

    def generate(self, prompt: str, max_new_tokens: int = 256, *, sample: bool = False,
                 temperature: float = 0.0, top_p: float = 0.95, seed: int | None = None) -> str:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        if seed is not None:
            self._torch.manual_seed(int(seed))
            if self.device == "cuda":
                self._torch.cuda.manual_seed_all(int(seed))
        gen_kwargs = {
            "do_sample": bool(sample),
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tok.pad_token_id,
        }
        if sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
        with self._torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
