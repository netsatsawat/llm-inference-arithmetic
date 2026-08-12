"""
Tests that assert the published claims, not just the code.

Every figure quoted in the series "LLM Inference, Measured" is pinned here. If a
number in an article changes and this suite does not, one of them is wrong — which
is the same contract tsfm-bakeoff's data generator uses: fail rather than let the
prose and the arithmetic drift apart.

    python3 -m pytest tests/ -q          # or: python3 tests/test_arithmetic.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference_arithmetic import (  # noqa: E402
    GPUS, LLAMA3_70B, ModelShape,
    ridge_point, achievable_tflops, decode_intensity, prefill_intensity,
    kv_bytes_per_token, kv_gib_per_sequence, sequences_that_fit,
    sliding_window_bytes, GPT_OSS_120B, ModelShape as _MS,
    describe_config, expert_intensity, batch_for_ridge, plan,
    score_matrix_bytes, prefix_reuse,
    end_to_end_speedup, speedup_ceiling, required_kernel_speedup,
    recover_count, wilson_interval, two_proportion_test,
    min_n_for_difference, mcnemar_exact,
)


# ── roofline ─────────────────────────────────────────────────────────────────

def test_h100_ridge_point_is_295():
    tflops, bw, _ = GPUS["h100-sxm"]
    assert round(ridge_point(tflops, bw), 1) == 295.5


def test_decode_intensity_is_the_batch_size():
    assert decode_intensity(32) == 32.0
    assert decode_intensity(256) == 256.0


def test_batch_below_ridge_is_memory_bound():
    tflops, bw, _ = GPUS["h100-sxm"]
    at_32 = achievable_tflops(decode_intensity(32), tflops, bw)
    assert at_32 < tflops * 0.15          # batch 32 reaches under 15% of peak
    at_4k = achievable_tflops(prefill_intensity(4096), tflops, bw)
    assert at_4k == tflops                 # prefill is pinned to the compute roof


# ── kv cache ─────────────────────────────────────────────────────────────────

def test_llama3_70b_is_320_kib_per_token():
    assert kv_bytes_per_token(LLAMA3_70B) == 327_680
    assert kv_bytes_per_token(LLAMA3_70B) / 1024 == 320


def test_published_kv_table():
    """Part 2's table, which is now gpt-oss-120b. GiB, not GB.

    The article switched off Llama-3 because meta-llama's config.json is gated: it
    returns 401 to a reader without an account, and the series' rule is that a
    number the reader cannot recompute is a citation, not a result. gpt-oss's is open.
    """
    assert kv_bytes_per_token(GPT_OSS_120B) == 36_864
    assert kv_bytes_per_token(GPT_OSS_120B) / 1024 == 36
    assert round(kv_gib_per_sequence(GPT_OSS_120B, 8192) * 1024, 0) == 288    # MiB
    # exact, deliberately. 1.125 rounds to 1.12 in Python and 1.13 in JavaScript,
    # so the article prints the exact value rather than one language's convention.
    assert kv_gib_per_sequence(GPT_OSS_120B, 32768) == 1.125
    assert round(kv_gib_per_sequence(GPT_OSS_120B, 131072), 1) == 4.5


def test_only_the_full_attention_layers_grow():
    """Half of gpt-oss-120b's layers hold 128 tokens and then stop, forever.

    Counting all 36 overstates the per-token cache by exactly 2x, which is the
    mistake the layer_types key exists to prevent.
    """
    naive = _MS(layers=36, kv_heads=8, head_dim=64)
    assert kv_bytes_per_token(naive) / 1024 == 72
    assert kv_bytes_per_token(GPT_OSS_120B) / 1024 == 36
    assert sliding_window_bytes(GPT_OSS_120B) / 1024 ** 2 == 4.5              # MiB, fixed
    assert sliding_window_bytes(LLAMA3_70B) == 0


def test_from_config_reads_layer_types():
    """A config that lists layer_types is telling you which layers grow."""
    cfg = {
        "num_hidden_layers": 4, "num_attention_heads": 64, "num_key_value_heads": 8,
        "head_dim": 64, "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention",
                        "sliding_attention", "full_attention"],
    }
    shape = _MS.from_config(cfg, name="toy")
    assert shape.layers == 4 and shape.full_attention_layers == 2
    assert shape.sliding_layers == 2


def test_dense_contrast_still_holds():
    """Part 2 keeps the older dense shape as the contrast. Same formula, 9x apart."""
    assert kv_bytes_per_token(LLAMA3_70B) / 1024 == 320
    assert round(kv_gib_per_sequence(LLAMA3_70B, 131072), 1) == 40.0
    assert round(40.0 / 4.5, 0) == 9


def test_gqa_is_load_bearing():
    """Without GQA gpt-oss-120b at 128K is 36 GiB rather than 4.5 — an 8x cut."""
    no_gqa = _MS(layers=36, kv_heads=64, head_dim=64, growing_layers=18)
    assert round(kv_gib_per_sequence(no_gqa, 131072), 0) == 36
    assert 64 // 8 == 8


def test_context_length_sets_the_batch():
    """Part 2's published capacity table, on the budget the article now derives.

    160 GB across two H100s, x0.92 (vLLM's gpu_memory_utilization default), minus
    61.6 GB of gpt-oss-120b weights = 85 GB = 80 GiB, minus activations and CUDA
    graphs. The article uses 75 GiB. An earlier version used 90, which was a leftover
    from the Llama-3 example and was never derived from anything.
    """
    fits = lambda ctx: int(75 // (kv_gib_per_sequence(GPT_OSS_120B, ctx)))
    assert fits(4096) == 533
    assert fits(8192) == 266
    assert fits(32768) == 66
    assert fits(131072) == 16
    # the crossover: 4K clears part 1's dense ridge of 296, 8K falls just short
    assert fits(4096) > 296 > fits(8192)


def test_expert_intensity_is_experts_not_parameters():
    """Expert-layer intensity is experts-fired / experts-total.

    Part 1 used active-params / total-params (3/35 = 0.086 for Qwen3.6-35B-A3B),
    which counts always-on attention and embeddings in both halves and put the two
    articles on different definitions of the same quantity. Qwen3.6 routes 8 of 256;
    gpt-oss-120b routes 4 of 128. Both are 0.031, and both need a batch near 9,500.
    """
    qwen, gpt_oss = 8 / 256, 4 / 128
    assert round(qwen, 3) == round(gpt_oss, 3) == 0.031
    assert round(296 / qwen / 100) * 100 == 9500
    assert round(3 / 35, 3) == 0.086          # the wrong number, kept so it stays wrong


def test_decode_attention_intensity_is_the_group_size():
    """Not 1. G query heads share each stored KV pair, so the same bytes feed G heads.

    gpt-oss-120b and Llama-3-70B both have 64 query heads over 8 KV heads, so decode
    attention sits at intensity 8, not 1. Still far below the ridge, so the argument
    is unchanged — but 8 is the honest number, and it is the same 8x that GQA cuts
    the cache by.
    """
    assert 64 // GPT_OSS_120B.kv_heads == 8
    assert 8 < 296                             # still bandwidth-bound, which was the point


def _unused_old_llama_budget():
    fits = lambda ctx: sequences_that_fit(LLAMA3_70B, ctx, total_gib=160, weights_gib=70)
    assert fits(4096) == 72
    assert fits(8192) == 36
    assert fits(131072) == 2
    # And the punchline: the roofline wanted ~295 and the cache allows two.
    assert fits(131072) < ridge_point(*GPUS["h100-sxm"][:2]) / 100


def test_from_config_prefers_kv_heads_over_attention_heads():
    cfg = {"num_hidden_layers": 80, "num_attention_heads": 64,
           "num_key_value_heads": 8, "hidden_size": 8192}
    assert ModelShape.from_config(cfg) == ModelShape(80, 8, 128)


# ── amdahl ───────────────────────────────────────────────────────────────────

def test_published_amdahl_table():
    """Part 5: a 2x attention kernel at the two published attention shares."""
    assert round(end_to_end_speedup(0.18, 2), 2) == 1.10
    assert round(end_to_end_speedup(0.85, 2), 2) == 1.74


def test_ceilings():
    assert round(speedup_ceiling(0.18), 2) == 1.22
    assert round(speedup_ceiling(0.85), 2) == 6.67
    assert speedup_ceiling(1.0) == math.inf


def test_infinite_kernel_speedup_converges_on_the_ceiling():
    assert round(end_to_end_speedup(0.85, 1e9), 2) == round(speedup_ceiling(0.85), 2)


def test_required_speedup_is_infinite_above_the_ceiling():
    assert required_kernel_speedup(0.18, 1.5) == math.inf     # ceiling is 1.22
    assert round(required_kernel_speedup(0.85, 1.74), 1) == 2.0


# ── eval statistics ──────────────────────────────────────────────────────────

def test_counts_recover_exactly_from_published_percentages():
    """Both benchmarks round-trip, which is what identifies n in the first place."""
    assert recover_count(39.02, 164) == 64
    assert recover_count(31.10, 164) == 51
    assert recover_count(70.24, 12032) == 8451
    assert recover_count(68.66, 12032) == 8261


def test_humaneval_gap_is_not_significant_unpaired():
    """The finding that redrew the part-3 chart: 13 problems out of 164."""
    z, p = two_proportion_test(39.02, 31.10, 164)
    assert round(z, 2) == 1.50
    assert round(p, 2) == 0.13
    assert p > 0.05


def test_mmlu_pro_gap_is_significant():
    """The difference the first draft dismissed is the one that is real."""
    z, p = two_proportion_test(70.24, 68.66, 12032)
    assert round(z, 2) == 2.66
    assert p < 0.01


def test_humaneval_is_too_small_an_instrument():
    assert min_n_for_difference(39.02, 31.10) == 279     # it has 164


def test_wilson_intervals_show_instrument_precision():
    """What the intervals actually say — including the bit that caught a bad caption.

    A first draft of the part-3 figure asserted "MMLU-Pro's intervals do not
    overlap". They do, by 0.07 points, and the difference is still significant at
    p = 0.008. Overlapping confidence intervals DO NOT imply non-significance: the
    eyeball test is strictly more conservative than the z-test. Non-overlap proves
    significance; overlap proves nothing.
    """
    lo_a, hi_a = wilson_interval(39.02, 164)
    lo_b, hi_b = wilson_interval(31.10, 164)
    assert hi_a - lo_a > 14                               # ~15 points wide
    assert lo_a < hi_b                                    # and heavily overlapping

    lo_c, hi_c = wilson_interval(70.24, 12032)
    lo_d, hi_d = wilson_interval(68.66, 12032)
    assert hi_c - lo_c < 2                                # ~1.6 points wide
    assert 0 < hi_d - lo_c < 0.1                          # they touch, barely
    assert two_proportion_test(70.24, 68.66, 12032)[1] < 0.01   # and still significant


def test_mcnemar_cannot_be_settled_from_percentages():
    """Same net difference of 13, opposite verdicts. This is why it matters."""
    assert mcnemar_exact(15, 2) < 0.05                    # p = 0.002
    assert mcnemar_exact(30, 17) > 0.05                   # p = 0.079
    assert mcnemar_exact(0, 0) == 1.0


# ── planning: reading a real model, and the plan that comes out of it ─────────

# gpt-oss-120b's config.json, trimmed to the keys that decide a cache. Kept as a
# fixture rather than fetched, so the suite runs on a plane.
GPT_OSS_CFG = {
    "num_hidden_layers": 36,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "sliding_window": 128,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "layer_types": ["sliding_attention", "full_attention"] * 18,
}


def test_describe_config_reads_layer_types():
    """Stopping at num_hidden_layers overstates this model's cache by exactly 2x."""
    d = describe_config(GPT_OSS_CFG)
    assert d["layers"] == 36 and d["growing_layers"] == 18 and d["sliding_layers"] == 18
    assert d["kv_heads"] == 8 and d["head_dim"] == 64 and d["query_heads"] == 64
    assert d["experts_fired"] == 4 and d["experts_total"] == 128


def test_describe_config_without_layer_types():
    """A model with no sliding layers grows on all of them."""
    d = describe_config({"num_hidden_layers": 80, "num_attention_heads": 64,
                         "num_key_value_heads": 8, "hidden_size": 8192})
    assert d["layers"] == d["growing_layers"] == 80
    assert d["head_dim"] == 128          # derived from hidden_size / heads


def test_expert_intensity_uses_experts_not_parameters():
    assert round(expert_intensity(1, 4, 128), 3) == 0.031
    assert round(expert_intensity(1, 8, 256), 3) == 0.031
    assert batch_for_ridge(296, 4, 128) == batch_for_ridge(296, 8, 256) == 9472


def test_plan_reproduces_the_published_table():
    """The capacity numbers part 2 prints, from the budget part 2 derives."""
    fits = lambda ctx: plan(GPT_OSS_120B, ctx, target_concurrency=1, total_gb=160,
                            weights_gb=61.6, ridge=296, budget_gib=75).fits
    assert (fits(4096), fits(8192), fits(32768), fits(131072)) == (533, 266, 66, 16)


def test_plan_derives_the_budget_it_is_given():
    """160 GB x 0.92, minus 61.6 GB of weights, minus overhead."""
    p = plan(GPT_OSS_120B, 131072, 32, total_gb=160, weights_gb=61.6, ridge=296)
    assert round(p.budget_gib, 1) == 74.7
    assert p.fits == 16 and "SHORT by 16" in p.verdict
    # the articles round 74.7 to 75, which moves two of the four counts by one
    assert plan(GPT_OSS_120B, 8192, 1, 160, 61.6, 296).fits == 265
    assert plan(GPT_OSS_120B, 8192, 1, 160, 61.6, 296, budget_gib=75).fits == 266


def test_plan_calls_it_short_before_it_calls_it_bandwidth_bound():
    """Not fitting is the louder problem, so it wins the verdict."""
    assert "SHORT" in plan(GPT_OSS_120B, 131072, 100, 160, 61.6, 296).verdict
    assert "under the ridge" in plan(GPT_OSS_120B, 131072, 4, 160, 61.6, 296).verdict


def test_score_matrix_is_part_3s_number():
    """8.6 GB for one head in one layer at 64K, which is why FlashAttention exists."""
    assert round(score_matrix_bytes(65536) / 1e9, 1) == 8.6


def test_prefix_reuse_is_block_granular():
    """Six matching tokens is zero blocks, which is no reuse at all.

    This is the gap between what a tokenizer diff shows you and what vLLM does.
    """
    a = list(range(6)) + [9] * 39
    b = list(range(6)) + [7] * 39
    r = prefix_reuse(a, b, block_size=16)
    assert r["matching_tokens"] == 6
    assert r["reusable_blocks"] == 0 and r["reusable_tokens"] == 0
    assert round(r["token_fraction"], 2) == 0.13     # what the diff suggests
    assert r["real_fraction"] == 0.0                 # what you actually get

    a2 = list(range(43)) + [9, 9]
    b2 = list(range(43)) + [7, 7]
    r2 = prefix_reuse(a2, b2, block_size=16)
    assert r2["matching_tokens"] == 43 and r2["reusable_blocks"] == 2
    assert r2["reusable_tokens"] == 32
    assert round(r2["real_fraction"], 2) == 0.71     # 96% at token level, 71% in blocks


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  pass  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}  {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
