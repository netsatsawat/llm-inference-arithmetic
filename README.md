# llm-inference-arithmetic

**The four calculations that decide LLM serving performance.** No dependencies, no
model downloads, no GPU. Every function is short enough to read, so you can check
it rather than trust it.

**[Run the calculators in your browser →](https://netsatsawat.github.io/llm-inference-arithmetic/)**
Same arithmetic, no install, nothing sent anywhere.

Companion to the series *LLM Inference, Measured* at [satsawat.ai](https://satsawat.ai).

```bash
pip install git+https://github.com/netsatsawat/llm-inference-arithmetic
lia model openai/gpt-oss-120b
```

Or from a clone, which is the same thing plus the tests:

```bash
git clone https://github.com/netsatsawat/llm-inference-arithmetic
cd llm-inference-arithmetic
pip install .
python3 tests/test_arithmetic.py        # 32 checks, no install needed for this one
```

Python 3.9 and up. `pip install -e .` needs **pip 21.3 or newer**. The build backend is
hatchling, and editable installs for non-setuptools backends are PEP 660, which older pip
does not implement. A plain `pip install .` works on any pip. Verified on 3.9.6.

## The four

| | Answers |
|---|---|
| **roofline** | Am I compute-bound or memory-bound, and at what batch size? |
| **kvcache** | How much memory is one conversation, and how many fit? |
| **amdahl** | What is that kernel speedup actually worth end to end? |
| **evalstats** | Can my benchmark resolve the difference I just reported? |

## 1. The ridge point

```python
from llm_inference_arithmetic import GPUS, ridge_point, decode_intensity

tflops, bandwidth, _ = GPUS["h100-sxm"]
ridge_point(tflops, bandwidth)        # 295.5 FLOP per byte
```

Below that, tensor cores idle waiting for HBM and faster arithmetic buys nothing.

**In decode, arithmetic intensity *is* the batch size.** Each weight is read once
and used for every sequence. So `decode_intensity(32) == 32`, and you would need a
batch near 295 to saturate an H100.

**Attention is the exception, and it is the important one.** Every sequence reads
its own KV cache, so nothing amortises across the batch and attention's intensity
stays around 1 however you batch. Batching lifts the GEMMs off the memory roof; it
cannot lift attention.

## 2. Your KV cache

```python
from llm_inference_arithmetic import LLAMA3_70B, kv_gib_per_sequence, sequences_that_fit

kv_gib_per_sequence(LLAMA3_70B, 131072)                    # 40.0 GiB for one sequence
sequences_that_fit(LLAMA3_70B, 131072, total_gib=160, weights_gib=70)   # 2
sequences_that_fit(LLAMA3_70B, 8192,   total_gib=160, weights_gib=70)   # 36
```

`2 × layers × kv_heads × head_dim × dtype_bytes` per token. For Llama-3-70B that is
**320 KiB per token**, which at 128K context is 40 GiB for a single conversation,
re-read in full on every decode step.

**Your context length sets your batch size.** Not your scheduler, and not
`--max-num-seqs`. The roofline wants ~295; the cache allows two.

Works on any model straight from its config:

```python
import json, urllib.request
from llm_inference_arithmetic import ModelShape, kv_gib_per_sequence

cfg = json.load(open("config.json"))
shape = ModelShape.from_config(cfg)     # handles num_key_value_heads vs num_attention_heads
kv_gib_per_sequence(shape, 32768)
```

Everything is **GiB**, deliberately. Dividing a KiB figure by 10⁶ is how 40 becomes
41.9, and that mistake shipped in a draft of the article before this library existed.

## 3. What the kernel is worth

```python
from llm_inference_arithmetic import end_to_end_speedup, speedup_ceiling

end_to_end_speedup(p=0.18, s=2)    # 1.10 when attention is 18% of time
end_to_end_speedup(p=0.85, s=2)    # 1.74 when attention is 85% of time
speedup_ceiling(0.18)              # 1.22 if attention became free
```

Same 2× kernel. Seven times the value, decided by a workload parameter rather than
by the code. **Compute the ceiling before starting work**: if it is 1.22×, no kernel
will save that service and the effort belongs at another layer.

## 4. Can your benchmark prove your claim?

This one exists because of a mistake, and the mistake turned out to be worse than the
one it was built to catch. An article draft asserted that INT4 costs *"a fifth of your
coding ability"*, on a table showing HumanEval falling 39.02% → 31.10%. The power check
said the gap was thirteen problems out of 164 and did not reach significance. Correct,
and beside the point: those numbers are not in the paper they were attributed to, or in
any other I could find.

The real figures, from *Give Me BF16 or Give Me Death?* ([arXiv:2411.02355](https://arxiv.org/abs/2411.02355),
Table 3, Llama-3.1-70B-Instruct):

```python
from llm_inference_arithmetic import recover_count, two_proportion_test, min_n_for_difference

# HumanEval, 164 problems: BF16 57.0 -> W4A16 56.3
recover_count(57.0, 164), recover_count(56.3, 164)   # (93, 92): a ONE-problem gap
two_proportion_test(57.0, 56.3, 164)                 # z=0.11, p=0.91: nowhere near

# MMLU-Pro, 12,032 questions: BF16 48.1 -> W4A16 47.2
two_proportion_test(48.1, 47.2, 12032)               # z=1.39, p=0.16: also not
min_n_for_difference(48.1, 47.2)                     # 23,661 needed; it has 12,032
```

Neither benchmark separates its quantized model from 16-bit, and the paper agrees:
FP8 lossless, INT8 1-3%, INT4 *"more competitive than expected, rivaling 8-bit
quantization"*.

Two lessons, and the second one cost more. Convert percentages to items before
believing a delta. And check that the delta is in the source at all. A number with a
good story travels perfectly well without one.

**The correct test is one you usually cannot run.** Both models saw the same 164
problems, so this is paired data and McNemar applies. McNemar needs the
discordant pairs, of which a published percentage preserves only the difference:

```python
from llm_inference_arithmetic import mcnemar_exact
mcnemar_exact(b=15, c=2)     # 0.0023, significant
mcnemar_exact(b=30, c=17)    # 0.0789, not significant
```

Same net difference of 13. Opposite verdicts. Nobody can tell which world they are in
from the numbers as published. An evaluation report should carry discordant counts,
and almost never does.

**And overlapping intervals do not mean "no difference."** Neither MMLU-Pro pair above
demonstrates this, because neither is significant. So here is one that is, at the same
n = 12,032. Scores of 50.0% and 48.7% give Wilson intervals of [49.11, 50.89] and
[47.81, 49.60], overlapping by half a point, and a two-proportion test returns
p = 0.044. Non-overlap proves significance; overlap proves nothing. Reading two error
bars for a gap between them answers a different question than the one you asked, and
answers it conservatively. The band where the intervals touch but the difference is
real runs up to about p = 0.006.

## Embedding these

**satsawat.ai, or any site you control:** an iframe is enough.

```html
<iframe src="https://netsatsawat.github.io/llm-inference-arithmetic/pens/part-2-kv-cache.html"
        style="width:100%;height:620px;border:0" loading="lazy"
        title="What one conversation costs"></iframe>
```

**Medium: not this page.** Medium accepts no HTML, no iframes and no scripts. It unfurls pasted
URLs through Embed.ly, which knows about 300 providers: CodePen and Observable among them, a
GitHub Pages URL not among them. Paste a link to the page above into a Medium story and you get
a link card, not a calculator.

The route to a live calculator inside a Medium story is to host the same code somewhere Embed.ly
already trusts, then paste that URL on its own line. `docs/pens/` exists for that: four
self-contained files, one per article, each of which pastes straight into a CodePen HTML pane.

| file | article | what it does |
|---|---|---|
| `part-1-roofline.html` | one | ridge point, intensity, and whether you are memory-bound |
| `part-2-kv-cache.html` | two | cache per token, per conversation, and how many fit |
| `part-4-eval-stats.html` | four | counts, Wilson intervals, and whether the gap resolves |
| `part-7-amdahl.html` | seven | what a kernel speedup is worth end to end |

One per article rather than one big page, so the calculator sits next to the arithmetic it
belongs to instead of making the reader leave and come back.

They are generated by `node tools/make-pens.js` rather than hand-written, because four copies of
the same theme and the same number formatting is four places for them to drift.

## Two implementations, on purpose

`docs/index.html` is the same five calculations in JavaScript, written independently and
served as a static page. The redundancy pays for itself: the two were compared field by
field and disagreed on one, the Wilson interval, because the browser version snapped the
percentage back to a whole number of items and the Python did not. 57.0% of 164 is 93
items, and the interval belongs to 93/164 rather than to 0.570. The Python was changed
to match, and a test now pins it.

A single implementation cannot catch that class of mistake. Two can.

## Tests assert the published claims

Every figure quoted in the articles is pinned in `tests/`. If a number in an article
changes and the suite does not, one of them is wrong. That is the same contract
[tsfm-bakeoff](https://github.com/netsatsawat/tsfm-bakeoff) uses for its benchmark
data: fail rather than let the prose and the arithmetic drift apart.

```bash
python3 tests/test_arithmetic.py     # no pytest needed
python3 -m pytest tests/ -q          # if you have it
```

## What this is not

Not a profiler and not a benchmark harness. It computes the numbers you should know
*before* you rent a GPU, and the ones you should check *after* you read someone
else's result. For the actual measurement you still want `nsys` and a real workload.

MIT.

## Pointing it at a model you actually run

The series' central instruction is "pull the values out of `config.json` and multiply".
This does that, against the live Hub, so nobody has to retype five numbers:

```bash
lia model openai/gpt-oss-120b
```

```
    layers                  36
      full attention        18   <- only these grow
      sliding window 128    18   <- these do not
    key-value heads         8
    query heads             64   (8x sharing)
    head dimension          64
    experts                 4 of 128 fired  (0.031)
      batch to reach 296    9,472

    KV cache                36,864 bytes/token = 36 KiB
      (all 36 layers would be 72 KiB, 2x too high)
```

That last line is why the command exists. Reading `num_hidden_layers` and stopping
there overstates this model's cache by exactly 2x, because half its layers are
sliding-window and stop growing. `layer_types` is the key that tells you, and it is
easy to miss.

A gated model raises rather than guesses. The series moved off Llama-3 precisely
because its config returns 401 to a reader without an account.

## The capacity plan nobody writes

```bash
lia plan --model-id openai/gpt-oss-120b --context 131072 --target 32
```

```
  KV budget               74.7 GiB
  per conversation        4.500 GiB
  conversations that fit  16
  you asked for           32
  ridge point             296

  SHORT by 16. No engine tuning rescues this: more memory, a smaller
  cache, or a shorter context.
```

Defaults match vLLM: `gpu_memory_utilization` is 0.92, and the engine still wants
activations and CUDA graph buffers on top. The verdict distinguishes the two failures
that get confused, *does not fit* and *fits but cannot use the arithmetic*, because
only one of them is fixable by buying memory.

**One honest wrinkle.** The derivation lands on 74.7 GiB and the articles round it to
75, which moves two of the four published counts by one (533 vs 531, 266 vs 265).
Rounding a budget and then flooring a division does that. Pass `--budget-gib 75` to
reproduce the articles exactly. Both numbers are in the test suite.

**A second one, about units.** Section 2 above is GiB throughout, but `plan` takes its
card memory as GB (10⁹) and does the conversion itself, so the 80 in the GPU table is
read as 80 GB rather than the ~79.7 GiB an "80GB" H100 actually reports. That makes the
budget about 7% conservative. It is the convention the published example was computed
under, so it stays; `--budget-gib` bypasses the derivation entirely if you would rather
name the number yourself.

## The other three

```bash
lia moe --fired 8 --total 256 --batch 533     # expert intensity, and the batch it needs
lia attention --seq-len 65536                 # the score matrix FlashAttention refuses to write
lia prefix --text-a "..." --text-b "..."      # what an engine can really reuse
```

`lia moe` exists because its absence produced a published error. Expert-layer intensity
is experts-fired over experts-total; using active-parameters over total-parameters
counts always-on attention in both halves and comes out about three times too kind.

`lia prefix` reports two numbers and the second is the real one. vLLM matches prefixes
in blocks of sixteen tokens, so a six-token match is zero blocks and no reuse at all.
A token-level diff will tell you 13%; you will get nothing.

**Tokenization is not universal, and the answer moves with it.** `tiktoken` is the only
optional dependency here and it covers OpenAI's encodings only. Llama, Qwen, Mistral and
Gemma each split text their own way. The same two strings:

```
o200k_base    (gpt-oss)   5 vs 6 tokens,  0 shared   -> nothing reusable
cl100k_base   (GPT-4)     6 vs 6 tokens,  1 shared
```

`What's` is one token in the first and two in the second, so the prompts diverge at
position zero under one encoding and position one under the other. Run `--compare` to see
it. If your model is outside the OpenAI family, tokenize with its own tokenizer and pass
`--tokens-a/--tokens-b`. The package takes raw ids precisely so it never has to guess
which vocabulary you are on. A reuse figure computed with the wrong tokenizer is a fact
about somebody else's deployment.
