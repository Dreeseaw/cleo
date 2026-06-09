"""Generation backends. Heavy deps (torch / llama-cpp-python) are imported lazily so `import cleo`
is cheap and you only pay for the backend you use."""
from __future__ import annotations


class GGUFBackend:
    """llama-cpp-python backend (CPU or GPU). The lightweight default for laptops / MCP servers."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 8, n_gpu_layers: int = 0):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        out = self.llm(prompt, max_tokens=max_new_tokens, temperature=0.0, stop=["\n\n\n"])
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

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
