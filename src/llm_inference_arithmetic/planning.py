"""Reading a real model, and turning its numbers into a capacity plan.

Everything in the base module answers a question the articles pose in the abstract.
This module is for pointing the same arithmetic at a model you actually run.

Deliberately imports nothing from the rest of the package: it duck-types the shape
object (anything with .layers, .kv_heads, .head_dim and .full_attention_layers will
do) so there is no import cycle and no ordering to get wrong.

Standard library only, like the rest of it.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass

GIB = 1024 ** 3


# ── reading a real model ─────────────────────────────────────────────────────

def fetch_config(model_id: str, timeout: float = 15.0) -> dict:
    """Pull a model's config.json straight off the Hugging Face Hub.

    The series' central instruction is "pull the values out of config.json and
    multiply", so the package should be able to do that rather than make you
    retype five numbers off a web page.

    Gated repos return 401, and that is not a bug to work around. It is the reason
    the articles work through gpt-oss-120b rather than Llama-3: a number the reader
    cannot look up is a citation, not a result.
    """
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise PermissionError(
                f"{model_id} is gated: its config.json needs an account. "
                "If your readers cannot check it, pick a model they can."
            ) from e
        raise


def describe_config(cfg: dict) -> dict:
    """The five numbers that decide a KV cache, pulled out of a raw config.

    Handles the trap the articles spend a section on: `layer_types`. If a config
    lists it, some layers are sliding-window and their cache stops growing. Reading
    `num_hidden_layers` and stopping there overstates the cache by the ratio of
    total layers to full-attention ones, which is 2x on gpt-oss-120b.
    """
    layers = cfg["num_hidden_layers"]
    kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    types = cfg.get("layer_types")
    growing = types.count("full_attention") if types else layers
    return {
        "layers": layers,
        "growing_layers": growing,
        "sliding_layers": layers - growing,
        "sliding_window": cfg.get("sliding_window"),
        "kv_heads": kv_heads,
        "query_heads": cfg.get("num_attention_heads"),
        "head_dim": head_dim,
        "experts_total": cfg.get("num_local_experts") or cfg.get("num_experts"),
        "experts_fired": cfg.get("num_experts_per_tok"),
    }


# ── mixture of experts ───────────────────────────────────────────────────────

def expert_intensity(batch: int, experts_fired: int, experts_total: int) -> float:
    """Arithmetic intensity of the expert layers: batch x fired/total.

    Fired over total. NOT active parameters over total parameters, which counts
    always-on attention and embeddings in both halves and comes out roughly three
    times too kind. Part 1 of the series shipped exactly that mistake, describing
    Qwen3.6-35B-A3B as 3/35 = 0.086 when it routes 8 of 256 = 0.031.
    """
    if not 0 < experts_fired <= experts_total:
        raise ValueError("experts_fired must be in (0, experts_total]")
    return batch * (experts_fired / experts_total)


def batch_for_ridge(ridge: float, experts_fired: int, experts_total: int) -> int:
    """Batch at which the expert layers stop being bandwidth-bound.

    Large enough to be theoretical: 9,472 for a 4-of-128 router and for an 8-of-256
    one alike, against an H100's ridge of 296.
    """
    return math.ceil(ridge / (experts_fired / experts_total))


# ── the capacity plan ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Plan:
    """What one line of a capacity plan should actually contain."""

    budget_gib: float
    per_sequence_gib: float
    fits: int
    target: int
    ridge: float
    verdict: str

    def __str__(self) -> str:
        rows = [
            ("KV budget", f"{self.budget_gib:.1f} GiB"),
            ("per conversation", f"{self.per_sequence_gib:.3f} GiB"),
            ("conversations that fit", f"{self.fits}"),
            ("you asked for", f"{self.target}"),
            ("ridge point", f"{self.ridge:.0f}"),
        ]
        body = "\n".join(f"  {k:<24}{v}" for k, v in rows)
        return f"{body}\n\n  {self.verdict}"


def kv_bytes_per_token(shape, dtype_bytes: int = 2) -> int:
    """2 (K and V) x growing layers x kv heads x head dim x bytes."""
    growing = getattr(shape, "full_attention_layers", None) or shape.layers
    return 2 * growing * shape.kv_heads * shape.head_dim * dtype_bytes


def plan(
    shape,
    context: int,
    target_concurrency: int,
    total_gb: float,
    weights_gb: float,
    ridge: float,
    utilisation: float = 0.92,
    overhead_gib: float = 5.0,
    dtype_bytes: int = 2,
    budget_gib: float | None = None,
) -> Plan:
    """Total memory, minus weights, divided by the per-conversation cost.

    The series says this "belongs in every capacity plan and is almost never in
    one". Defaults match vLLM: `gpu_memory_utilization` is 0.92, and the engine
    still wants activations and CUDA graph buffers on top of that.

    `total_gb` and `weights_gb` are GB (10^9), because that is what spec sheets and
    safetensors listings report, while the answer is GiB. Mixing the two is how 4.5
    quietly becomes 4.8, so the conversion happens here, once, where it can be read.

    Pass `budget_gib` to skip the derivation. Worth knowing why you might: the default
    lands on 74.7 GiB for the series' example and the articles round that to 75, which
    moves two of the four published counts by one. Rounding a budget and then flooring
    a division does that. Better said out loud than discovered by a reader.
    """
    if budget_gib is None:
        usable_gib = (total_gb * utilisation - weights_gb) * 1e9 / GIB
        budget_gib = max(0.0, usable_gib - overhead_gib)
    per = kv_bytes_per_token(shape, dtype_bytes) * context / GIB
    fits = int(budget_gib // per) if per else 0

    if fits < target_concurrency:
        verdict = (
            f"SHORT by {target_concurrency - fits}. No engine tuning rescues this: "
            "more memory, a smaller cache, or a shorter context."
        )
    elif fits < ridge:
        verdict = (
            f"FITS, but {fits} is under the ridge of {ridge:.0f}. You will be "
            "bandwidth-bound, paying for arithmetic you cannot reach."
        )
    else:
        verdict = f"FITS, and {fits} clears the ridge of {ridge:.0f}."
    return Plan(budget_gib, per, fits, target_concurrency, ridge, verdict)


# ── attention traffic, and prefix reuse ──────────────────────────────────────

def score_matrix_bytes(seq_len: int, dtype_bytes: int = 2) -> int:
    """One head, one layer: the S x S score matrix if you materialise it.

    Grows with the square of the sequence, which is the number that turned this
    layer into a research field and the reason FlashAttention never writes it out.
    """
    return seq_len * seq_len * dtype_bytes


def prefix_reuse(a: list[int], b: list[int], block_size: int = 16) -> dict:
    """How much of prompt `a` an engine can actually reuse when serving `b`.

    Takes token ids rather than strings so the package stays dependency-free: bring
    your own tokenizer, and the answer will be right for the one you actually run.

    Reports the token-level match and the block-level one, because only the second
    is real. vLLM matches on block boundaries, so a six-token match is zero blocks
    and no reuse at all — a distinction that turns "13% reuse" into "none".
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    blocks = n // block_size
    return {
        "matching_tokens": n,
        "reusable_blocks": blocks,
        "reusable_tokens": blocks * block_size,
        "token_fraction": n / len(b) if b else 0.0,
        "real_fraction": (blocks * block_size) / len(b) if b else 0.0,
    }


__all__ = [
    "fetch_config", "describe_config",
    "expert_intensity", "batch_for_ridge",
    "Plan", "plan", "kv_bytes_per_token",
    "score_matrix_bytes", "prefix_reuse",
]
