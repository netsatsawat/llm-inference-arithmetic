"""
`lia`: the four calculations from the command line.

    lia roofline --gpu h100-sxm --batch 32
    lia kv --layers 80 --kv-heads 8 --head-dim 128 --context 131072 --total 160 --weights 70
    lia amdahl --p 0.18 --speedup 2
    lia eval --a 79.7 --b 80.5 --n 164

Deliberately verbose output: the intermediate values are the point. A tool that
prints only the answer teaches nothing and cannot be checked.
"""

from __future__ import annotations

import argparse
import math

from . import (
    GPUS, ModelShape,
    ridge_point, achievable_tflops, decode_intensity, prefill_intensity,
    kv_bytes_per_token, kv_gib_per_sequence, sequences_that_fit,
    end_to_end_speedup, speedup_ceiling,
    recover_count, wilson_interval, two_proportion_test,
    min_n_for_difference, mcnemar_exact,
    GIB, fetch_config, describe_config, expert_intensity, batch_for_ridge,
    plan, score_matrix_bytes, prefix_reuse,
)


def _roofline(a) -> None:
    tflops, bw, mem = GPUS[a.gpu]
    ridge = ridge_point(tflops, bw)
    print(f"\n{a.gpu}: {tflops:g} TFLOP/s dense BF16, {bw:g} TB/s HBM, {mem} GB\n")
    print(f"  ridge point            {ridge:.1f} FLOP/byte")
    print(f"  (peak FLOPs / bandwidth, below this you are moving bytes, not multiplying)\n")
    for label, intensity in [
        (f"decode GEMMs, batch {a.batch}", decode_intensity(a.batch)),
        ("decode attention", 1.0),
        (f"prefill, {a.seq_len} tokens", prefill_intensity(a.seq_len)),
    ]:
        got = achievable_tflops(intensity, tflops, bw)
        pct = 100 * got / tflops
        bound = "compute-bound" if intensity >= ridge else "MEMORY-BOUND"
        print(f"  {label:<28} intensity {intensity:>8.0f}  ->  {got:6.1f} TFLOP/s ({pct:4.1f}% of peak)  {bound}")
    print(f"\n  batch needed to reach the ridge: {math.ceil(ridge)}")
    print("  attention never gets there: each sequence reads its own KV cache.\n")


def _kv(a) -> None:
    shape = ModelShape(layers=a.layers, kv_heads=a.kv_heads, head_dim=a.head_dim)
    per_token = kv_bytes_per_token(shape, a.dtype_bytes)
    print(f"\n2 x {a.layers} layers x {a.kv_heads} kv-heads x {a.head_dim} head-dim x {a.dtype_bytes} bytes\n")
    print(f"  per token              {per_token:,} bytes  ({per_token / 1024:.0f} KiB)")
    print(f"  at {a.context:,} context     {kv_gib_per_sequence(shape, a.context, a.dtype_bytes):.1f} GiB per sequence\n")
    if a.total and a.weights:
        free = a.total - a.weights
        print(f"  {a.total} GiB total - {a.weights} GiB weights = {free} GiB for KV\n")
        print("  context      GiB/seq    sequences that fit")
        ctx = 4096
        while ctx <= a.context:
            n = sequences_that_fit(shape, ctx, a.total, a.weights, a.dtype_bytes)
            print(f"  {ctx // 1024:>4}K        {kv_gib_per_sequence(shape, ctx, a.dtype_bytes):6.1f}    {n:>6}")
            ctx *= 2
        print("\n  This, not the scheduler, is what caps your batch size.\n")


def _amdahl(a) -> None:
    print(f"\nend-to-end = 1 / ((1 - p) + p/s)   with p = {a.p}, s = {a.speedup}\n")
    print(f"  end-to-end speedup     {end_to_end_speedup(a.p, a.speedup):.2f}x")
    ceil = speedup_ceiling(a.p)
    print(f"  ceiling (s -> inf)     {ceil:.2f}x")
    print(f"  gain realised          {100 * (end_to_end_speedup(a.p, a.speedup) - 1):.1f}%")
    print(f"  gain available         {100 * (ceil - 1):.1f}%\n")
    if ceil < 1.25:
        print("  The ceiling is under 1.25x. No kernel will save this; the work belongs elsewhere.\n")


def _eval(a) -> None:
    ca, cb = recover_count(a.a, a.n), recover_count(a.b, a.n)
    print(f"\n{a.a}% and {a.b}% on n = {a.n}\n")
    print(f"  counts                 {ca} and {cb}  ->  a difference of {ca - cb} items")
    lo_a, hi_a = wilson_interval(a.a, a.n)
    lo_b, hi_b = wilson_interval(a.b, a.n)
    print(f"  95% Wilson (A)         [{lo_a:.2f}, {hi_a:.2f}]   width {hi_a - lo_a:.2f}")
    print(f"  95% Wilson (B)         [{lo_b:.2f}, {hi_b:.2f}]   width {hi_b - lo_b:.2f}")
    z, p = two_proportion_test(a.a, a.b, a.n)
    verdict = "SIGNIFICANT at 0.05" if p < 0.05 else "NOT significant at 0.05"
    print(f"\n  two-proportion z-test  z = {z:.2f}, p = {p:.4f}   {verdict}")
    need = min_n_for_difference(a.a, a.b)
    print(f"  n needed per arm       {need}   (you have {a.n})")
    d = abs(ca - cb)
    print("\n  Note: if both arms were scored on the SAME items this is paired data and")
    print("  McNemar is the correct test. It needs the discordant counts, which a")
    print("  published percentage does not preserve. It preserves only the gap of")
    print(f"  {d}, and the gap is not enough. Every row below has it, and the verdict")
    print("  moves anyway (the test is symmetric, so only the sign is yours to drop):")
    shown = set()
    for c in (0, d // 2, d):
        b = d + c
        if (b, c) in shown:
            continue
        shown.add((b, c))
        print(f"    b={b:<3} c={c:<3} -> exact p = {mcnemar_exact(b, c):.4f}")
    print("\n  Overlapping intervals do NOT imply non-significance. Non-overlap implies")
    print("  significance; overlap implies nothing.\n")



def _model(a):
    """Point it at a Hub model and get the five numbers that decide its cache."""
    cfg = fetch_config(a.model_id)
    d = describe_config(cfg)
    print(f"\n  {a.model_id}\n")
    print(f"    layers                  {d['layers']}")
    if d["sliding_layers"]:
        print(f"      full attention        {d['growing_layers']}   <- only these grow")
        print(f"      sliding window {str(d['sliding_window']):<7}{d['sliding_layers']}   <- these do not")
    print(f"    key-value heads         {d['kv_heads']}")
    if d["query_heads"]:
        print(f"    query heads             {d['query_heads']}   ({d['query_heads'] // d['kv_heads']}x sharing)")
    print(f"    head dimension          {d['head_dim']}")
    if d["experts_total"]:
        fired, total = d["experts_fired"], d["experts_total"]
        print(f"    experts                 {fired} of {total} fired  ({fired / total:.3f})")
        print(f"      batch to reach 296    {batch_for_ridge(296, fired, total):,}")

    shape = ModelShape(layers=d["layers"], kv_heads=d["kv_heads"], head_dim=d["head_dim"],
                       name=a.model_id, growing_layers=d["growing_layers"],
                       sliding_window=d["sliding_window"])
    per = kv_bytes_per_token(shape, a.dtype_bytes)
    print(f"\n    KV cache                {per:,} bytes/token = {per / 1024:.0f} KiB")
    if d["sliding_layers"]:
        naive = 2 * d["layers"] * d["kv_heads"] * d["head_dim"] * a.dtype_bytes
        print(f"      (all {d['layers']} layers would be {naive / 1024:.0f} KiB, {naive / per:.0f}x too high)")
    print()
    for ctx in (4096, 8192, 32768, 131072):
        g = per * ctx / GIB
        size = f"{g * 1024:.0f} MiB" if g < 1 else f"{g:.2f} GiB"
        print(f"      {ctx // 1024:>4}K  {size:>10} per conversation")
    print()


def _plan(a):
    """Total memory, minus weights, divided by the per-conversation cost."""
    if a.model_id:
        d = describe_config(fetch_config(a.model_id))
        shape = ModelShape(layers=d["layers"], kv_heads=d["kv_heads"], head_dim=d["head_dim"],
                           name=a.model_id, growing_layers=d["growing_layers"],
                           sliding_window=d["sliding_window"])
    else:
        # growing_layers, not layers. Passing 36 where 18 grow doubles the answer,
        # which is the mistake --model-id exists to stop you making.
        shape = ModelShape(layers=a.layers, kv_heads=a.kv_heads, head_dim=a.head_dim,
                           growing_layers=a.growing_layers or a.layers,
                           name=a.name or "model")
    tflops, bw, mem_gb = GPUS[a.gpu]
    ridge = ridge_point(tflops, bw)
    # GPUS holds spec-sheet GB (10^9), which is the unit plan() takes; it does the
    # GiB conversion itself, so nothing is converted twice on the way in.
    gpu_gb = a.gpu_gb or mem_gb
    print(f"\n  {shape.name} on {a.gpus} x {a.gpu}, {a.context // 1024}K context\n")
    print(plan(shape, a.context, a.target, total_gb=a.gpus * gpu_gb,
               weights_gb=a.weights, ridge=ridge, utilisation=a.utilisation,
               budget_gib=a.budget_gib))
    print()


def _moe(a):
    """Expert-layer intensity: fired over total, not active params over total params."""
    frac = a.fired / a.total
    tflops, bw, _ = GPUS[a.gpu]
    ridge = ridge_point(tflops, bw)
    print(f"\n  {a.fired} of {a.total} experts fire per token\n")
    print(f"    expert fraction         {frac:.4f}")
    print(f"    intensity at batch B    B x {frac:.3f}")
    print(f"    ridge on {a.gpu:<14} {ridge:.0f}")
    print(f"    batch to reach it       {batch_for_ridge(ridge, a.fired, a.total):,}")
    print(f"\n    At batch {a.batch}, the expert layers sit at {expert_intensity(a.batch, a.fired, a.total):.1f}"
          f", {'past' if expert_intensity(a.batch, a.fired, a.total) >= ridge else 'still short of'} the ridge.\n")


def _attention(a):
    """The S x S score matrix, which is why FlashAttention exists."""
    b = score_matrix_bytes(a.seq_len, a.dtype_bytes)
    print(f"\n  Sequence {a.seq_len:,}, one head, one layer\n")
    print(f"    materialised scores     {b / 1e9:.2f} GB")
    print(f"    x {a.heads} heads x {a.layers} layers  {b * a.heads * a.layers / 1e12:.1f} TB if you wrote it all out")
    print(f"\n    FlashAttention writes none of it. Same numbers, same precision, tiles that")
    print(f"    stay in shared memory.\n")


def _prefix(a):
    """What an engine can actually reuse between two prompts.

    Two caveats are printed rather than buried, because both change the answer:
    the block boundary, and the tokenizer. tiktoken only knows OpenAI's encodings,
    and a model outside that family will split the same string differently.
    """
    if a.tokens_a and a.tokens_b:
        ta = [int(x) for x in a.tokens_a.split(",")]
        tb = [int(x) for x in a.tokens_b.split(",")]
        source = "token ids you supplied"
    else:
        if not (a.text_a and a.text_b):
            print("\n  pass --text-a/--text-b, or --tokens-a/--tokens-b for your own tokenizer\n")
            return
        try:
            import tiktoken
        except ImportError:
            print("\n  --text needs tiktoken (pip install tiktoken).")
            print("  Or tokenize with your model's own tokenizer and pass --tokens-a/--tokens-b.\n")
            return
        enc = tiktoken.get_encoding(a.encoding)
        ta, tb = enc.encode(a.text_a), enc.encode(a.text_b)
        source = a.encoding

    r = prefix_reuse(ta, tb, a.block_size)
    print(f"\n  tokenized with: {source}")
    print(f"  A: {len(ta)} tokens    B: {len(tb)} tokens\n")
    print(f"    matching tokens         {r['matching_tokens']}")
    print(f"    at token granularity    {r['token_fraction'] * 100:.0f}%")
    print(f"    reusable blocks         {r['reusable_blocks']} x {a.block_size} = {r['reusable_tokens']} tokens")
    print(f"    what you ACTUALLY reuse {r['real_fraction'] * 100:.0f}%")
    if r["matching_tokens"] and not r["reusable_blocks"]:
        print("\n    A match shorter than one block is no reuse at all.")

    if a.compare and not (a.tokens_a and a.tokens_b):
        import tiktoken
        print("\n  the same two strings, under other encodings:\n")
        for name in ("o200k_base", "cl100k_base", "p50k_base"):
            e = tiktoken.get_encoding(name)
            x, y = e.encode(a.text_a), e.encode(a.text_b)
            m = prefix_reuse(x, y, a.block_size)
            print(f"    {name:<14} {len(x)} vs {len(y)} tokens, {m['matching_tokens']} shared")

    print("\n  NOTE: tokenization is not universal. tiktoken covers OpenAI's encodings")
    print("  only. Llama, Qwen, Mistral and Gemma each split text their own way, and")
    print("  the same two strings can share a prefix under one and share nothing under")
    print("  another. Use the tokenizer belonging to the model you actually serve, or")
    print("  this number is about somebody else's deployment.\n")



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lia", description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("roofline", help="am I memory-bound?")
    r.add_argument("--gpu", choices=sorted(GPUS), default="h100-sxm")
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--seq-len", type=int, default=4096)
    r.set_defaults(fn=_roofline)

    k = sub.add_parser("kv", help="KV cache size and how many sequences fit")
    k.add_argument("--layers", type=int, default=80)
    k.add_argument("--kv-heads", type=int, default=8)
    k.add_argument("--head-dim", type=int, default=128)
    k.add_argument("--context", type=int, default=131072)
    k.add_argument("--dtype-bytes", type=int, default=2, choices=[1, 2, 4])
    k.add_argument("--total", type=float, help="total GiB across the GPUs")
    k.add_argument("--weights", type=float, help="GiB taken by model weights")
    k.set_defaults(fn=_kv)

    m = sub.add_parser("amdahl", help="what a kernel speedup is worth end to end")
    m.add_argument("--p", type=float, required=True, help="fraction of time in the optimised part")
    m.add_argument("--speedup", type=float, default=2.0)
    m.set_defaults(fn=_amdahl)

    e = sub.add_parser("eval", help="can the benchmark resolve this difference?")
    e.add_argument("--a", type=float, required=True, help="baseline score, percent")
    e.add_argument("--b", type=float, required=True, help="comparison score, percent")
    e.add_argument("--n", type=int, required=True, help="items in the benchmark")
    e.set_defaults(fn=_eval)

    mo = sub.add_parser("model", help="read a model's config.json off the Hub")
    mo.add_argument("model_id", help="e.g. openai/gpt-oss-120b")
    mo.add_argument("--dtype-bytes", type=int, default=2, choices=[1, 2, 4])
    mo.set_defaults(fn=_model)

    pl = sub.add_parser("plan", help="does my target concurrency actually fit?")
    pl.add_argument("--model-id", help="read the shape off the Hub instead of passing dims")
    pl.add_argument("--layers", type=int, default=36)
    pl.add_argument("--growing-layers", type=int, default=18,
                    help="full-attention layers, the only ones whose cache grows")
    pl.add_argument("--name", default="gpt-oss-120b (defaults)")
    pl.add_argument("--kv-heads", type=int, default=8)
    pl.add_argument("--head-dim", type=int, default=64)
    pl.add_argument("--context", type=int, default=131072)
    pl.add_argument("--target", type=int, default=32, help="concurrency you need")
    pl.add_argument("--gpu", choices=sorted(GPUS), default="h100-sxm")
    pl.add_argument("--gpus", type=int, default=2)
    pl.add_argument("--gpu-gb", type=float, default=None, help="defaults to the card's own memory")
    pl.add_argument("--weights", type=float, default=61.6, help="GB of weights")
    pl.add_argument("--utilisation", type=float, default=0.92, help="vLLM gpu_memory_utilization")
    pl.add_argument("--budget-gib", type=float, default=None, help="skip the derivation, name the budget")
    pl.set_defaults(fn=_plan)

    mx = sub.add_parser("moe", help="expert-layer intensity and the batch it needs")
    mx.add_argument("--fired", type=int, default=4, help="experts per token")
    mx.add_argument("--total", type=int, default=128)
    mx.add_argument("--batch", type=int, default=64)
    mx.add_argument("--gpu", choices=sorted(GPUS), default="h100-sxm")
    mx.set_defaults(fn=_moe)

    at = sub.add_parser("attention", help="the score matrix FlashAttention refuses to write")
    at.add_argument("--seq-len", type=int, default=65536)
    at.add_argument("--heads", type=int, default=64)
    at.add_argument("--layers", type=int, default=18)
    at.add_argument("--dtype-bytes", type=int, default=2, choices=[1, 2, 4])
    at.set_defaults(fn=_attention)

    px = sub.add_parser("prefix", help="what an engine can really reuse between two prompts")
    px.add_argument("--text-a", help="needs tiktoken; OpenAI encodings only")
    px.add_argument("--text-b")
    px.add_argument("--tokens-a", help="comma-separated ids from YOUR model's tokenizer")
    px.add_argument("--tokens-b")
    px.add_argument("--compare", action="store_true",
                    help="show how the answer moves across encodings")
    px.add_argument("--encoding", default="o200k_base")
    px.add_argument("--block-size", type=int, default=16, help="vLLM's prefix match unit")
    px.set_defaults(fn=_prefix)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
