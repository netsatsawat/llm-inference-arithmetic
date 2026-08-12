"""
The four calculations that decide LLM serving performance.

None of this is hard. All of it is skipped, and every one of the four answers a
question people usually answer with a vendor's number instead:

    roofline    Am I compute-bound or memory-bound, and at what batch size?
    kvcache     How much memory is one conversation, and how many fit?
    amdahl      What is that kernel speedup actually worth end to end?
    evalstats   Can my benchmark resolve the difference I just reported?

Pure Python, no dependencies. Every function is short enough to read, because the
point is that you check it rather than trust it.

Companion to the series "LLM Inference, Measured" — https://satsawat.ai
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__version__ = "0.1.0"

GIB = 1024 ** 3
TERA = 1e12

# ──────────────────────────────────────────────────────────────────────────────
# Reference hardware. Dense (non-sparse) tensor-core throughput and HBM bandwidth.
# Vendors usually quote the sparsity-doubled FLOP number; these are the dense ones,
# because you are not running structured sparsity.
# ──────────────────────────────────────────────────────────────────────────────
GPUS = {
    #                 dense BF16 TFLOP/s,  HBM TB/s,  memory GiB
    "h100-sxm":       (990.0,              3.35,      80),
    "h100-pcie":      (756.0,              2.00,      80),
    "h200":           (990.0,              4.80,      141),
    "a100-80":        (312.0,              2.04,      80),
    "l40s":           (362.0,              0.864,     48),
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ROOFLINE
# ══════════════════════════════════════════════════════════════════════════════

def ridge_point(tflops: float, tb_per_s: float) -> float:
    """FLOP per byte at which a kernel stops being memory-bound.

    Below this, the tensor cores idle waiting for HBM and faster arithmetic buys
    nothing. H100 SXM: 990 / 3.35 = 295.5.
    """
    return (tflops * TERA) / (tb_per_s * TERA)


def achievable_tflops(intensity: float, tflops: float, tb_per_s: float) -> float:
    """Where an operation of this arithmetic intensity actually lands."""
    return min(tflops, tb_per_s * intensity)


def decode_intensity(batch_size: int) -> float:
    """Arithmetic intensity of the WEIGHT MATRICES during decode.

    Each weight is read once and used for every sequence in the batch, so
    intensity == batch size. This is the whole reason batching works.
    """
    return float(batch_size)


def attention_decode_intensity() -> float:
    """Arithmetic intensity of ATTENTION during decode: about 1, always.

    Every sequence reads its own KV cache, so nothing amortises across the batch.
    Batching lifts the GEMMs off the memory roof and cannot lift attention. This
    asymmetry is why long-context decode stays bandwidth-bound however you batch.
    """
    return 1.0


def prefill_intensity(seq_len: int) -> float:
    """Prefill multiplies each loaded weight by the whole prompt: intensity ~= S."""
    return float(seq_len)


# ══════════════════════════════════════════════════════════════════════════════
# 2. KV CACHE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelShape:
    """The numbers from config.json that decide your KV cache.

    ``growing_layers`` exists because not every layer's cache grows with the
    conversation. Models that interleave sliding-window attention with full
    attention — gpt-oss-120b alternates them across all 36 of its layers — pay an
    unbounded per-token cost only on the full-attention half. The sliding half
    holds ``sliding_window`` tokens and then stops, forever.

    Reading ``layers`` where you meant ``growing_layers`` overstates the cache by
    exactly the ratio of the two, which for gpt-oss-120b is a factor of two.
    """
    layers: int
    kv_heads: int          # num_key_value_heads — NOT num_attention_heads
    head_dim: int
    name: str = "model"
    growing_layers: int | None = None      # defaults to `layers`
    sliding_window: int | None = None

    @property
    def full_attention_layers(self) -> int:
        return self.layers if self.growing_layers is None else self.growing_layers

    @property
    def sliding_layers(self) -> int:
        return self.layers - self.full_attention_layers

    @classmethod
    def from_config(cls, cfg: dict, name: str = "model") -> "ModelShape":
        """Build from a HuggingFace config dict, handling the usual key drift."""
        layers = cfg["num_hidden_layers"]
        kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        # Count only the layers whose cache actually grows. A config with layer_types
        # is telling you that some of them do not.
        types = cfg.get("layer_types")
        growing = types.count("full_attention") if types else None
        return cls(
            layers=layers, kv_heads=kv_heads, head_dim=head_dim, name=name,
            growing_layers=growing, sliding_window=cfg.get("sliding_window"),
        )


LLAMA3_70B = ModelShape(layers=80, kv_heads=8, head_dim=128, name="Llama-3-70B")
# openai/gpt-oss-120b. Public, ungated config.json — which is why the article works
# through this one: a reader can check every number without an account.
GPT_OSS_120B = ModelShape(
    layers=36, kv_heads=8, head_dim=64, name="gpt-oss-120b",
    growing_layers=18, sliding_window=128,
)
LLAMA3_8B = ModelShape(layers=32, kv_heads=8, head_dim=128, name="Llama-3-8B")


def kv_bytes_per_token(shape: ModelShape, dtype_bytes: int = 2) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x dtype_bytes.

    Counts only the layers whose cache grows. gpt-oss-120b at fp16 is
    2 * 18 * 8 * 64 * 2 = 36,864 bytes = 36 KiB per token, not the 72 KiB you get
    from all 36 layers. Llama-3-70B has no sliding layers, so it is all 80:
    2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KiB per token.
    """
    return 2 * shape.full_attention_layers * shape.kv_heads * shape.head_dim * dtype_bytes


def sliding_window_bytes(shape: ModelShape, dtype_bytes: int = 2) -> int:
    """The fixed cost of the sliding-window layers. It does not grow with context.

    Zero for a model without them.
    """
    if not shape.sliding_window or shape.sliding_layers <= 0:
        return 0
    return (2 * shape.sliding_layers * shape.kv_heads * shape.head_dim
            * dtype_bytes * shape.sliding_window)


def kv_gib_per_sequence(shape: ModelShape, context: int, dtype_bytes: int = 2) -> float:
    """GiB of KV cache for one sequence at this context length.

    GiB, not GB. Dividing a KiB figure by 1e6 is how 40 becomes 41.9.
    """
    return kv_bytes_per_token(shape, dtype_bytes) * context / GIB


def sequences_that_fit(
    shape: ModelShape,
    context: int,
    total_gib: float,
    weights_gib: float,
    dtype_bytes: int = 2,
) -> int:
    """How many concurrent sequences the KV budget actually allows.

    This — not the scheduler, not --max-num-seqs — is what caps your batch size,
    and therefore your arithmetic intensity, and therefore whether you are
    anywhere near the ridge point.
    """
    free = total_gib - weights_gib
    if free <= 0:
        return 0
    return int(free // kv_gib_per_sequence(shape, context, dtype_bytes))


# ══════════════════════════════════════════════════════════════════════════════
# 3. AMDAHL
# ══════════════════════════════════════════════════════════════════════════════

def end_to_end_speedup(p: float, s: float) -> float:
    """1 / ((1 - p) + p/s).

    p = fraction of total time in the part you optimised.
    s = how much faster you made that part.

    A 2x attention kernel at p=0.18 gives 1.10x. At p=0.85 it gives 1.74x. Same
    kernel; the difference is a workload parameter, not the code.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a fraction between 0 and 1")
    if s <= 0:
        raise ValueError("s must be positive")
    return 1.0 / ((1.0 - p) + p / s)


def speedup_ceiling(p: float) -> float:
    """1 / (1 - p) — the best you can do if the optimised part becomes free.

    Compute this before starting work. If it is 1.22x, no kernel will save you and
    the effort belongs at a different layer.
    """
    if p >= 1.0:
        return math.inf
    return 1.0 / (1.0 - p)


def required_kernel_speedup(p: float, target: float) -> float:
    """How fast the part has to get to reach a target end-to-end speedup.

    Returns math.inf when the target is above the ceiling, which is the useful
    answer: it means stop.
    """
    denom = p - (1.0 - (1.0 - p) * target) * 0.0 - (target * (1.0 - p))
    # target = 1/((1-p) + p/s)  =>  p/s = 1/target - (1-p)
    rhs = (1.0 / target) - (1.0 - p)
    if rhs <= 0:
        return math.inf
    return p / rhs


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVAL STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def _norm_sf(z: float) -> float:
    """Two-tailed tail probability for a standard normal."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def recover_count(pct: float, n: int) -> int:
    """Turn a reported percentage back into the number of items.

    Do this before believing any delta. HumanEval reports 39.02% on 164 problems,
    which is 64 problems. An "8-point drop" is then 13 problems, and 13 out of 164
    is a very different claim from "8 points".
    """
    return round(pct / 100.0 * n)


def wilson_interval(pct: float, n: int, z: float = 1.959964) -> tuple:
    """95% Wilson score interval, in percentage points.

    Correct at small n, and unlike the normal approximation it cannot run past 0
    or 100.

    The percentage is snapped back to a whole number of items first. A benchmark
    score is a count divided by n, so 39.02% on 164 is 64 items and the interval
    belongs to 64/164, not to 0.3902. The difference is about half a hundredth of a
    point here — invisible at the two decimals anyone prints, and worth doing anyway
    because recovering the count is the habit the whole section is arguing for.
    """
    p = recover_count(pct, n) / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / d * 100.0, (centre + margin) / d * 100.0)


def two_proportion_test(pct_a: float, pct_b: float, n: int) -> tuple:
    """Unpaired two-proportion z-test. Returns (z, p_value).

    This is usually all a published evaluation gives you enough information for —
    and it is the wrong test when both models saw the same items. See mcnemar_exact.
    """
    a, b = recover_count(pct_a, n), recover_count(pct_b, n)
    p1, p2 = a / n, b / n
    pool = (a + b) / (2 * n)
    se = math.sqrt(pool * (1 - pool) * (2 / n))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    return (z, _norm_sf(z))


def min_n_for_difference(pct_a: float, pct_b: float, z: float = 1.959964) -> int:
    """Items per arm needed to call this gap significant, unpaired.

    A 7.9-point gap around 35% needs about 279. HumanEval has 164.
    """
    p1, p2 = pct_a / 100.0, pct_b / 100.0
    if p1 == p2:
        return 0
    pool = (p1 + p2) / 2.0
    return math.ceil(2 * pool * (1 - pool) * (z / (p1 - p2)) ** 2)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-tailed McNemar p-value for paired binary outcomes.

    b = items A got right and B got wrong; c = the reverse. The CORRECT test when
    both models were scored on the same items — and the one you usually cannot run,
    because published percentages preserve only b - c.

    At b=15, c=2 the answer is p=0.002. At b=30, c=17 — same net difference of 13 —
    it is p=0.079. Nobody can tell those apart from a percentage.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cdf)

# Reading a real model, and planning capacity from it. Kept in its own module so it
# imports nothing from here and there is no cycle to reason about.
from .planning import (  # noqa: E402
    fetch_config, describe_config, expert_intensity, batch_for_ridge,
    Plan, plan, score_matrix_bytes, prefix_reuse,
)

__all__ = [
    "GPUS", "GIB", "ModelShape", "LLAMA3_70B", "LLAMA3_8B", "GPT_OSS_120B",
    "ridge_point", "achievable_tflops", "decode_intensity",
    "attention_decode_intensity", "prefill_intensity",
    "kv_bytes_per_token", "kv_gib_per_sequence", "sequences_that_fit",
    "sliding_window_bytes",
    "end_to_end_speedup", "speedup_ceiling", "required_kernel_speedup",
    "recover_count", "wilson_interval", "two_proportion_test",
    "min_n_for_difference", "mcnemar_exact",
    "fetch_config", "describe_config", "expert_intensity", "batch_for_ridge",
    "Plan", "plan", "score_matrix_bytes", "prefix_reuse",
]
