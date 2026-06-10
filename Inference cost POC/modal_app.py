"""Self-hosted vLLM OpenAI-compatible server on Modal.

Pattern follows Modal's current official vLLM example (verified 2026-06; see BUDGET.md for the
source URL + lookup date):
  * Weights cached in a named modal.Volume mounted at /root/.cache/huggingface so they download
    ONCE and persist across cold starts. A second volume caches vLLM's compiled artifacts.
  * Scale-to-zero by default; `scaledown_window` kept LOW so we are not billed for idle GPU.
  * `vllm serve` launched as a subprocess inside a @modal.web_server function, exposing the
    standard OpenAI endpoints (/v1/chat/completions, /v1/completions, /v1/models).

Deploy (Phase 1+, real spend — gated):
    modal deploy modal_app.py
Then the harness talks to the printed https URL + "/v1".

Cost discipline:
  * Default GPU = L4 (cheapest that fits Qwen2.5-1.5B comfortably and 7B in bf16 within 24GB).
  * Pin vLLM. Don't leave containers warm. Don't download weights twice.
  * Qwen2.5-1.5B/7B-Instruct are ungated (Apache-2.0) — no HF token needed. If you later use a
    gated model, add an HF token via `modal secret create huggingface-secret HF_TOKEN=...` and
    attach it to the function.
"""
from __future__ import annotations

import os
import subprocess

import modal

# --- configuration ---------------------------------------------------------
# Override at deploy time with env vars MODEL_NAME / GPU / SCALEDOWN.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
GPU = os.environ.get("GPU", "L4")  # L4 cheapest fitting; A10G for 7B headroom
SCALEDOWN_WINDOW = int(os.environ.get("SCALEDOWN", "60"))  # seconds idle before scale-to-zero
VLLM_PORT = 8000
# Pin vLLM for reproducibility. Bump deliberately, not implicitly.
# 0.21.0 is the version the current official Modal vLLM example pins (verified 2026-06-05 via
# PyPI JSON: released 2026-05-15) — stable, supports Qwen2.5, not bleeding edge.
VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.21.0")
# CUDA base image to match the official, tested Modal example (bare debian_slim is NOT what the
# current example uses; the CUDA devel base avoids subtle kernel/flash-attn build issues).
CUDA_IMAGE = os.environ.get("CUDA_IMAGE", "nvidia/cuda:12.9.0-devel-ubuntu22.04")

# --- image -----------------------------------------------------------------
# IMPORTANT: runtime config consumed INSIDE the container (MODEL_NAME, MAX_MODEL_LEN,
# QUANTIZATION) must be baked into the image env. A bare `MODEL_NAME=... modal deploy` only sets
# the var in the LOCAL deploy shell; the remote container re-imports this module WITHOUT it and
# would fall back to the default model. Baking via .env() (evaluated at deploy time from the
# globals above) makes the container actually see the chosen model. Decorator args (gpu=GPU,
# scaledown_window=SCALEDOWN) are fine as plain globals — they're read at deploy time.
vllm_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]",
    )
    # Faster weight transfers into the volume on first download + runtime model config.
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "MODEL_NAME": MODEL_NAME,
            "MAX_MODEL_LEN": os.environ.get("MAX_MODEL_LEN", "8192"),
            "QUANTIZATION": os.environ.get("QUANTIZATION", ""),
        }
    )
)

# --- volumes (weight + compile caches) -------------------------------------
hf_cache_vol = modal.Volume.from_name("reliability-tax-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("reliability-tax-vllm-cache", create_if_missing=True)

app = modal.App("reliability-tax-vllm")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=20 * 60,
    volumes={
        # Default HF cache location; weights persist here across cold starts.
        "/root/.cache/huggingface": hf_cache_vol,
        # vLLM compile cache speeds later cold starts.
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=64)  # batch many concurrent requests onto one GPU
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * 60)
def serve():
    """Launch the vLLM OpenAI-compatible server as a subprocess.

    The function returns immediately after spawning vLLM; Modal exposes port 8000. vLLM accepts
    requests once it finishes downloading weights (first run only), loading, and compiling. The
    long startup_timeout covers the one-time cold start.
    """
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Keep memory modest so it fits the cheap GPU; tune if you see OOM.
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        os.environ.get("MAX_MODEL_LEN", "8192"),
        # Enable logprobs in responses (needed by the `abstain` strategy).
        # vLLM returns logprobs when the request asks for them; no special flag required,
        # but we keep served context modest to control KV-cache cost.
    ]
    # Optional quantization (AWQ/GPTQ) if a pre-quantized repo is used; pass via env.
    quant = os.environ.get("QUANTIZATION")
    if quant:
        cmd += ["--quantization", quant]

    print("Launching:", " ".join(cmd))
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main():
    """`modal run modal_app.py` prints the config without deploying a persistent endpoint."""
    print("reliability-tax vLLM server config:")
    print(f"  MODEL_NAME       = {MODEL_NAME}")
    print(f"  GPU              = {GPU}")
    print(f"  SCALEDOWN_WINDOW = {SCALEDOWN_WINDOW}s")
    print(f"  VLLM_VERSION     = {VLLM_VERSION}")
    print(f"  CUDA_IMAGE       = {CUDA_IMAGE}")
    print("Deploy with:  modal deploy modal_app.py")
    print("Then point the harness at the printed URL + '/v1'.")
