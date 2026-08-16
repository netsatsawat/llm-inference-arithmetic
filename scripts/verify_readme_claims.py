#!/usr/bin/env python3
"""No number in the README without a runnable path behind it.

Every figure README.md quotes is RECOMPUTED here (from the package, from the
CLI run in-process, or from a committed fixture) and then asserted to appear
verbatim in the README text. Nothing is written down twice: if the arithmetic
moves or the prose moves, the substring stops matching and this exits 1.

Offline by construction, and stdlib-only, like the package. `lia model` and
`lia plan` normally read config.json off the Hub; neither is fetched here. The
plan block is reproduced through the CLI's own defaults, which are gpt-oss-120b's
shape, and the model block from the config fixture committed in tests/.

It also polices the retraction. 39.02, 31.10, 70.24 and 68.66 were attributed to
a paper that does not contain them. They may appear in exactly two places: the
README paragraph that narrates the retraction, and the tests comment that records
it. A half-applied correction once left one of them standing in prose for weeks,
so their location is pinned by line number rather than trusted.
"""

import contextlib
import io
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from llm_inference_arithmetic import (                       # noqa: E402
    GPUS, LLAMA3_70B, GPT_OSS_120B, ModelShape,
    ridge_point, decode_intensity,
    kv_bytes_per_token, kv_gib_per_sequence, sequences_that_fit,
    end_to_end_speedup, speedup_ceiling,
    recover_count, wilson_interval, two_proportion_test,
    min_n_for_difference, mcnemar_exact,
    describe_config, batch_for_ridge,
)
from llm_inference_arithmetic.cli import main as lia          # noqa: E402
import test_arithmetic as suite                               # noqa: E402

# The figures that were retracted. Named once, here, and used both to recompute
# what the retraction paragraph says and to police where they are allowed to be.
RETRACTED = ("39.02", "31.10", "70.24", "68.66")

# Files that are allowed to contain them, and the only ones excluded from the sweep.
# scripts/ is excluded because this file has to name them to check for them.
SWEEP_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".impeccable",
                   ".venv", "dist", "build", "scripts"}
SWEEP_SUFFIXES = {".py", ".md", ".html", ".js", ".json", ".yml", ".yaml", ".toml", ".txt"}

problems = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}" +
          (f" ({detail})" if detail and not condition else ""))
    if not condition:
        problems.append(name)


def quotes(readme, label, text, detail=""):
    """Assert the README literally contains a string built from a computed value."""
    check(f"README quotes {label} ({text.strip()})", text in readme,
          detail or "not found in README.md")


def retracted_hits(text):
    """(line number, figure) for every retracted figure in `text`. 1-based."""
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for fig in RETRACTED:
            if re.search(r"(?<![\d.])" + re.escape(fig) + r"(?!\d)", line):
                hits.append((i, fig))
    return hits


def plan_cli(context, budget=None):
    """Run `lia plan` in-process and hand back its stdout. No --model-id, no network."""
    argv = ["plan", "--context", str(context), "--target", "32"]
    if budget is not None:
        argv += ["--budget-gib", str(budget)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lia(argv)
    return buf.getvalue()


def fits(out):
    m = re.search(r"conversations that fit\s+(\d+)", out)
    return int(m.group(1)) if m else None


def main() -> int:
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    print("1. the ridge point:")
    ridge = ridge_point(*GPUS["h100-sxm"][:2])
    quotes(readme, "the H100 ridge point", f"# {ridge:.1f} FLOP per byte")
    quotes(readme, "the batch that saturates it", f"batch near {int(ridge)} to saturate")
    quotes(readme, "the same ridge in the KV punchline", f"The roofline wants ~{int(ridge)};")
    quotes(readme, "decode intensity at batch 32",
           f"`decode_intensity(32) == {decode_intensity(32):.0f}`")

    print("2. the KV cache:")
    per_token = kv_bytes_per_token(LLAMA3_70B)
    quotes(readme, "Llama-3-70B per token", f"**{per_token // 1024} KiB per token**")
    one_seq = kv_gib_per_sequence(LLAMA3_70B, 131072)
    quotes(readme, "one 128K sequence", f"# {one_seq:.1f} GiB for one sequence")
    quotes(readme, "the same figure in prose",
           f"is {one_seq:.0f} GiB for a single conversation")
    long_ctx = sequences_that_fit(LLAMA3_70B, 131072, 160, 70)
    short_ctx = sequences_that_fit(LLAMA3_70B, 8192, 160, 70)
    quotes(readme, "sequences at 128K",
           f"sequences_that_fit(LLAMA3_70B, 131072, total_gib=160, weights_gib=70)   # {long_ctx}")
    quotes(readme, "sequences at 8K",
           f"sequences_that_fit(LLAMA3_70B, 8192,   total_gib=160, weights_gib=70)   # {short_ctx}")

    print("3. what the kernel is worth:")
    quotes(readme, "a 2x kernel at p=0.18",
           f"# {end_to_end_speedup(0.18, 2):.2f} when attention is 18% of time")
    quotes(readme, "a 2x kernel at p=0.85",
           f"# {end_to_end_speedup(0.85, 2):.2f} when attention is 85% of time")
    ceiling = speedup_ceiling(0.18)
    quotes(readme, "the ceiling at p=0.18", f"# {ceiling:.2f} if attention became free")
    quotes(readme, "the ceiling again, in prose", f"if it is {ceiling:.2f}×")

    print("4. can the benchmark prove the claim:")
    # Inputs come from the test suite's own constants, so the README, the tests and
    # the arithmetic are all pinned to one set of numbers from arXiv:2411.02355.
    a_he, b_he, n_he = suite.BF16_HE, suite.INT4_HE, suite.N_HE
    a_mp, b_mp, n_mp = suite.BF16_MMLU, suite.INT4_MMLU, suite.N_MMLU
    quotes(readme, "the HumanEval row", f"HumanEval, {n_he} problems: BF16 {a_he} -> W4A16 {b_he}")
    quotes(readme, "the MMLU-Pro row",
           f"MMLU-Pro, {n_mp:,} questions: BF16 {a_mp} -> W4A16 {b_mp}")

    counts = (recover_count(a_he, n_he), recover_count(b_he, n_he))
    quotes(readme, "the recovered counts", f"# ({counts[0]}, {counts[1]}): a ONE-problem gap")
    check(f"the recovered gap really is one problem ({counts[0]} - {counts[1]})",
          counts[0] - counts[1] == 1)

    z_he, p_he = two_proportion_test(a_he, b_he, n_he)
    quotes(readme, "the HumanEval test", f"# z={z_he:.2f}, p={p_he:.2f}: nowhere near")
    z_mp, p_mp = two_proportion_test(a_mp, b_mp, n_mp)
    quotes(readme, "the MMLU-Pro test", f"# z={z_mp:.2f}, p={p_mp:.2f}: also not")
    need = min_n_for_difference(a_mp, b_mp)
    quotes(readme, "the items MMLU-Pro would need", f"# {need:,} needed; it has {n_mp:,}")

    # The retraction paragraph's own arithmetic, recomputed from the retracted figures.
    stale_gap = recover_count(float(RETRACTED[0]), n_he) - recover_count(float(RETRACTED[1]), n_he)
    check(f"the retracted table really was a {stale_gap}-problem gap on {n_he}",
          stale_gap == 13 and f"thirteen problems out of {n_he}" in readme)

    print("5. McNemar, which needs what nobody publishes:")
    # Read b and c out of the README's own calls rather than restating them here.
    # Hardcoding both pairs made "the two gaps are equal" constant-true, and left
    # the README free to change its arguments without the script noticing.
    pairs = [(int(b), int(c)) for b, c in
             re.findall(r"mcnemar_exact\(b=(\d+), c=(\d+)\)", readme)]
    check("the README calls mcnemar_exact with two pairs", len(pairs) == 2,
          f"found {len(pairs)}")
    if len(pairs) == 2:
        (b1, c1), (b2, c2) = pairs
        quotes(readme, "the significant pair",
               f"# {mcnemar_exact(b1, c1):.4f}, significant")
        quotes(readme, "the non-significant pair",
               f"# {mcnemar_exact(b2, c2):.4f}, not significant")
        check(f"the README's own two pairs carry the same net difference "
              f"({b1 - c1})", b1 - c1 == b2 - c2,
              f"{b1}-{c1}={b1 - c1} but {b2}-{c2}={b2 - c2}; the whole point of "
              f"the paragraph is that these are equal and the verdicts are not")
        check("one pair is significant and the other is not",
              (mcnemar_exact(b1, c1) < 0.05) != (mcnemar_exact(b2, c2) < 0.05),
              f"{mcnemar_exact(b1, c1):.4f} and {mcnemar_exact(b2, c2):.4f}")
        quotes(readme, "that net difference", f"Same net difference of {b1 - c1}.")

    print("6. overlapping intervals:")
    lo_a, hi_a = wilson_interval(50.0, n_mp)
    lo_b, hi_b = wilson_interval(48.7, n_mp)
    quotes(readme, "the 50.0% Wilson interval", f"[{lo_a:.2f}, {hi_a:.2f}]")
    quotes(readme, "the 48.7% Wilson interval", f"[{lo_b:.2f}, {hi_b:.2f}]")
    overlap = min(hi_a, hi_b) - max(lo_a, lo_b)
    check(f"the two intervals overlap by half a point ({overlap:.2f})",
          round(overlap, 1) == 0.5 and "overlapping by half a point" in readme)
    quotes(readme, "the p-value for that overlapping pair",
           f"p = {two_proportion_test(50.0, 48.7, n_mp)[1]:.3f}")

    # How far past the eyeball test significance reaches: walk the score down from
    # 50.0% until the intervals stop overlapping, and read off the p-value there.
    touch_p = None
    for count in range(n_mp // 2, n_mp // 2 - 400, -1):
        pct = 100 * count / n_mp
        if wilson_interval(pct, n_mp)[1] <= lo_a:
            touch_p = two_proportion_test(50.0, pct, n_mp)[1]
            break
    check("the point where the intervals separate is findable", touch_p is not None)
    if touch_p is not None:
        quotes(readme, "the p-value at that boundary", f"about p = {touch_p:.3f}")

    print("the retracted figures, and nowhere but where they are retracted:")
    lines = readme.splitlines()
    start = next((i for i, ln in enumerate(lines, 1)
                  if "This one exists because of a mistake" in ln), None)
    end = next((i for i, ln in enumerate(lines, 1) if "any other I could find." in ln), None)
    check("the README still carries the retraction paragraph",
          start is not None and end is not None and start < end, f"{start}..{end}")
    hits = retracted_hits(readme)
    check("the README names the retracted figures exactly twice",
          len(hits) == 2, f"{len(hits)} occurrence(s): {hits}")
    if start and end:
        stray = [h for h in hits if not start <= h[0] <= end]
        check(f"...and only inside that paragraph (lines {start}-{end})",
              not stray, f"also at {stray}")

    tests_src = (REPO / "tests" / "test_arithmetic.py").read_text(encoding="utf-8")
    t_hits = retracted_hits(tests_src)
    check("the tests name all four retracted figures, on a single line",
          len(t_hits) == len(RETRACTED) and len({ln for ln, _ in t_hits}) == 1,
          f"{t_hits}")
    if t_hits:
        ln_no = t_hits[0][0]
        t_lines = tests_src.splitlines()
        check(f"tests/test_arithmetic.py:{ln_no} is a comment, not an assertion",
              t_lines[ln_no - 1].lstrip().startswith("#"), t_lines[ln_no - 1].strip())
        window = "\n".join(t_lines[ln_no - 2:ln_no + 2])
        check("...and it is the comment that retracts them",
              "not in that paper" in window, window)

    scanned, offenders = 0, []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or path.suffix not in SWEEP_SUFFIXES:
            continue
        if SWEEP_SKIP_DIRS & set(path.relative_to(REPO).parts):
            continue
        if path.name == "README.md" or path.name == "test_arithmetic.py":
            continue
        scanned += 1
        if retracted_hits(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(REPO)))
    check(f"no retracted figure survives anywhere else ({scanned} files swept)",
          not offenders, f"still in {offenders}")

    print("the `lia model` block, recomputed from the committed config fixture:")
    d = describe_config(suite.GPT_OSS_CFG)
    shape = ModelShape(layers=d["layers"], kv_heads=d["kv_heads"], head_dim=d["head_dim"],
                       growing_layers=d["growing_layers"], sliding_window=d["sliding_window"])
    quotes(readme, "the layer count", f"layers                  {d['layers']}")
    quotes(readme, "the growing half", f"full attention        {d['growing_layers']}")
    quotes(readme, "the sliding half",
           f"sliding window {str(d['sliding_window']):<7}{d['sliding_layers']}")
    quotes(readme, "the KV heads", f"key-value heads         {d['kv_heads']}")
    quotes(readme, "the query heads and their sharing",
           f"query heads             {d['query_heads']}   "
           f"({d['query_heads'] // d['kv_heads']}x sharing)")
    quotes(readme, "the head dimension", f"head dimension          {d['head_dim']}")
    fired, total = d["experts_fired"], d["experts_total"]
    quotes(readme, "the expert fraction", f"{fired} of {total} fired  ({fired / total:.3f})")
    quotes(readme, "the batch the experts need",
           f"batch to reach {ridge:.0f}    {batch_for_ridge(int(round(ridge)), fired, total):,}")
    cache = kv_bytes_per_token(shape)
    naive = 2 * d["layers"] * d["kv_heads"] * d["head_dim"] * 2
    quotes(readme, "the per-token cache", f"{cache:,} bytes/token = {cache / 1024:.0f} KiB")
    quotes(readme, "the mistake it prevents",
           f"(all {d['layers']} layers would be {naive / 1024:.0f} KiB"
           f", {naive / cache:.0f}x too high)")

    print("the `lia plan` block, from the CLI run in-process:")
    out = plan_cli(131072)
    for label in ("KV budget", "per conversation", "conversations that fit",
                  "you asked for", "ridge point"):
        row = next((ln for ln in out.splitlines() if ln.startswith(f"  {label} ")), None)
        check(f"lia plan still prints a {label!r} row", row is not None)
        if row:
            quotes(readme, f"the {label!r} row", row)
    # The CLI writes the verdict on one line and the README wraps it, so only the
    # sentence carrying the number is asserted.
    verdict = next((ln.strip() for ln in out.splitlines() if ln.strip().startswith("SHORT")), "")
    check("lia plan still returns a SHORT verdict", bool(verdict))
    if verdict:
        quotes(readme, "the shortfall", verdict.split(".")[0] + ".")

    print("the rounding wrinkle, both budgets run through the CLI:")
    budget = re.search(r"KV budget\s+([\d.]+) GiB", out)
    check("lia plan still prints a derived budget", budget is not None)
    if budget:
        quotes(readme, "the derived budget", f"lands on {budget.group(1)} GiB")
    derived = {ctx: fits(plan_cli(ctx)) for ctx in (4096, 8192, 32768, 131072)}
    rounded = {ctx: fits(plan_cli(ctx, budget=75)) for ctx in derived}
    check("every context yields a count under both budgets",
          all(v is not None for v in list(derived.values()) + list(rounded.values())),
          f"{derived} {rounded}")
    moved = [ctx for ctx in derived if derived[ctx] != rounded[ctx]]
    check(f"rounding 74.7 to 75 moves two of the four counts ({len(moved)} moved)",
          len(moved) == 2 and "two of the four published counts by one" in readme)
    quotes(readme, "both pairs it moves",
           f"({rounded[4096]} vs {derived[4096]}, {rounded[8192]} vs {derived[8192]})")

    print("the repo's own furniture:")
    n_checks = sum(1 for name, fn in vars(suite).items()
                   if name.startswith("test_") and callable(fn))
    said = re.search(r"#\s*(\d+)\s+checks", readme)
    quotes(readme, "the size of the test suite", f"# {n_checks} checks",
           f"tests/test_arithmetic.py defines {n_checks} test functions, "
           f"README says {said.group(1) if said else 'nothing'}")
    pens = sorted(p.name for p in (REPO / "docs" / "pens").glob("*.html"))
    listed = sorted(re.findall(r"^\| `(part-[\w.-]+\.html)`", readme, re.M))
    check(f"the pens table lists every file in docs/pens/ ({len(pens)} of them)",
          pens == listed and bool(pens), f"on disk {pens}, in README {listed}")

    if problems:
        print(f"\n{len(problems)} README claim(s) drifted: {problems}")
        return 1
    print("\nevery quoted README number matches its artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
