# llm-inference-arithmetic

**Before you rent GPUs to serve a language model, or believe a gap between two benchmark
scores, four small sums tell you whether the numbers hold.** A GPU is the chip that runs the
model. Language models read and write text. Benchmarks are fixed tests a model is scored on.
This package does those sums in Python short enough to read, so you can check it rather than
trust it. The four sums it is built around download nothing and need no GPU.

**[Run the calculators in your browser →](https://netsatsawat.github.io/llm-inference-arithmetic/)**
The browser version runs the same arithmetic, installs nothing, and keeps your inputs on
your own machine.

Companion to the series *LLM Inference, Measured* at [satsawat.ai](https://satsawat.ai).

New to these terms? Plain-words definitions of language model, benchmark, context and the
rest sit in [what you need to know first](#what-you-need-to-know-first) below.

```bash
pip install git+https://github.com/netsatsawat/llm-inference-arithmetic
lia plan --context 131072 --target 32
```

The second line runs offline. Its `--context 131072` sets how long each conversation can be,
here 131,072 tokens. It prints the block under "Run it" below: how many long conversations
fit on two GPUs, and why the answer is 16 when you asked for 32.

## What this is

A small Python package and a command-line tool called `lia`. Together they do the
arithmetic you should do before you rent GPUs to serve a language model, and the
arithmetic you should do before you believe a benchmark score. Four questions, each
answered by a few multiplications. How much memory does one conversation take, and how
many conversations fit on the card? Is the GPU waiting on memory or on maths? What is a
faster kernel worth to the whole job? Is the gap between two benchmark scores big enough
to mean anything?

It is for an engineer who is about to buy hardware or quote a number and wants to check
the vendor's figure first. It needs nothing beyond Python 3.9. The pip package is
`llm-inference-arithmetic`, the Python import is `llm_inference_arithmetic`, and the
command is `lia`.

A test in this repo pins every number the articles quote. An automated check runs on
every change and fails if a number in this README stops matching what the code
computes. I also got two of the numbers wrong myself on the way here, and section 4
says how.

## What you need to know first

The terms this file leans on, in plain words.

| term | in plain words |
|---|---|
| language model, or LLM | A program that reads text and writes text, one piece at a time. gpt-oss-120b, an open model from OpenAI, is the example most of this file uses. |
| token | The unit a model reads and writes, roughly a word piece. Different model families split the same text into different tokens. A prompt is the text you send in. |
| weights, layers, parameters | The weights are the stored numbers that make up a model, and they sit in GPU memory the whole time the model runs. A model is a stack of layers, each with its own weights. The parameter count is how many weights there are. |
| GPU, or card | The chip that runs the model. The examples here use the H100 SXM, an NVIDIA card with 80 GB of memory. |
| serving, or inference | Running an already trained model to answer requests. Two phases: prefill, where the model reads the prompt, and decode, where it produces one token at a time. |
| engine | The program that loads the model onto the GPU and shares it between requests. vLLM is the engine this file assumes. |
| context | How many tokens one conversation can hold, prompt and answer together. 128K context means 131,072 tokens. |
| attention, heads | Attention is the step where each new token looks back at every earlier token in the conversation. Every layer has one. Heads are parallel copies of that step inside a layer, and the head dimension is how wide each copy is. |
| KV cache | The memory a model keeps for each conversation so it does not recompute earlier tokens. It grows with the length of the conversation, and it is what limits how many conversations fit on a GPU. |
| memory-bound, compute-bound, ridge point | Memory-bound: the GPU spends its time moving bytes while its maths units sit idle. Compute-bound: the maths units are the bottleneck. The ridge point is the number of maths operations per byte loaded where one turns into the other. |
| batch size | How many conversations the GPU handles in one step. The code calls a conversation a sequence. This file says conversation, except where it quotes code output. |
| kernel | One GPU function, such as the attention step, that someone might make faster. |
| benchmark | A fixed set of test items a model is scored on. The score is the share it got right. |
| Amdahl's law | If one part of a job takes a fraction p of the time and you make that part s times faster, the whole job gets 1 / ((1 - p) + p/s) faster. If that part became free, the most you could get is 1 / (1 - p). |
| GiB and GB | GiB is 1024 cubed bytes, GB is 1000 cubed. Spec sheets quote GB. The cache maths comes out in GiB. Mixing them moves answers by a few percent. KiB and MiB are the same idea one and two steps down: 1024 bytes, and 1024 squared. |

## Install

```bash
pip install git+https://github.com/netsatsawat/llm-inference-arithmetic
```

Or from a clone, which is the same thing plus the tests:

```bash
git clone https://github.com/netsatsawat/llm-inference-arithmetic
cd llm-inference-arithmetic
pip install .
python3 tests/test_arithmetic.py        # 32 checks, no install needed for this one
```

Python 3.9 or newer. Verified on 3.9.6, and the automated checks run the suite on 3.9
and 3.12. The package itself has no dependencies. Pip still downloads the build tool,
hatchling, to install it, so the install step needs a network connection.
`pip install -e .` needs **pip 21.3 or newer**, because the build backend is hatchling,
and editable installs for non-setuptools backends use PEP 660, which older pip does not
implement. A plain `pip install .` works on any pip.

## Run it: the capacity plan nobody writes

A capacity plan is the sum that says how many conversations a machine can hold at once.
It is the smallest useful command here. It needs no network: the numbers it uses by
default are those of gpt-oss-120b running on two H100 SXM cards.

```bash
lia plan --context 131072 --target 32
```

```
  gpt-oss-120b (defaults) on 2 x h100-sxm, 128K context

  KV budget               74.7 GiB
  per conversation        4.500 GiB
  conversations that fit  16
  you asked for           32
  ridge point             296

  SHORT by 16. No engine tuning rescues this: more memory, a smaller cache, or a shorter context.
```

Read it top to bottom. The KV budget is the memory left for conversations once the
weights and the engine's own needs are taken out. It comes from the command's defaults.
Start with 160 GB across two cards times 0.92, which is 147.2 GB. That 0.92 is the share
of each card vLLM uses by default, its `gpu_memory_utilization` setting, read from vLLM
v0.27.1. Take off 61.6 GB for gpt-oss-120b's stored weights, the size its weight files
report and the command's `--weights` default. That leaves 85.6 GB. Do the whole
subtraction in GB, then convert once by dividing by 1.074, the GB-to-GiB factor:
85.6 / 1.074 = 79.7 GiB. Now take off 5 GiB of
scratch memory the engine needs to run. The budget is 74.7 GiB. That 5 GiB is the
command's default allowance, not a measured figure. To name the whole budget yourself,
pass `--budget-gib`. 128K context means
131,072 tokens. One conversation that long needs 4.5 GiB of cache: 36 KiB per token times 131,072
tokens, and the next section shows where the 36 KiB comes from. Sixteen fit. You asked
for 32, so you are short by 16, and no engine setting changes that. The ridge point row
reads 296. That figure is a fixed property of the card, worked out in section 1 as
990 / 3.35. It matters here as a target. During decode, the number of maths operations
per byte loaded equals the batch size, so you would need about 296 conversations running
at once before the H100 stops waiting on memory. Sixteen is nowhere near it.

The verdict tells apart two failures that get confused. *Does not fit* is fixed by
buying memory. *Fits but sits under the ridge point* is not fixed by buying memory:
there is room, but the GPU still spends its time waiting on bytes, because too few
conversations share each load of the weights. The way out of that one is more
conversations per step, and the plan prints a different message for that case. Two
caveats about how the budget is derived, one on rounding and one on GB against GiB, are
under "Two wrinkles in the plan" further down.

### Point it at a model you actually run

The series' central instruction is "pull the values out of `config.json` and multiply".
`config.json` is the small file published with every model on the Hugging Face Hub that
lists the model's dimensions. The `model` command reads that file off the Hub, so nobody
has to retype five numbers. It needs a network connection. So does `lia plan --model-id`,
which reads the same file and then prints the same plan as above with the shape read
live:

```bash
lia model openai/gpt-oss-120b
lia plan --model-id openai/gpt-oss-120b --context 131072 --target 32
```

Every other `lia` command runs offline, with one exception. `lia prefix --text-a` goes
through `tiktoken`, which is not part of this package and fetches its encoding tables the
first time it is used.

The first command prints:

```
  openai/gpt-oss-120b

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

         4K     144 MiB per conversation
         8K     288 MiB per conversation
        32K    1.12 GiB per conversation
       128K    4.50 GiB per conversation
```

I reproduced the block above from the copy of gpt-oss-120b's config committed in
`tests/test_arithmetic.py`. The checker recomputes the rows from `layers` down to the
72 KiB line from that same copy. The four per-conversation rows are 36 KiB times the
context, which you can redo by hand: 36 KiB times 4,096 tokens is 144 MiB. Nothing in
the tests or the checks touches the network.

The 72 KiB line is why the command exists. Reading `num_hidden_layers` and stopping
there overstates this model's cache by exactly 2x. Half of its layers use
sliding-window attention. A sliding-window layer looks back 128 tokens here and no
further, so its cache stops growing after 128 tokens. Only the full-attention layers
grow with the conversation. The key in `config.json` that tells you which is which is
`layer_types`, and it is easy to miss.

"8x sharing" is grouped-query attention: eight query heads share one stored key-value
pair, so the cache is eight times smaller than it would be without it. "4 of 128 fired"
is mixture-of-experts. The model holds 128 expert sub-networks, and only four run for any
one token. Each expert therefore fires for only 1/32 of tokens (4 of 128 experts fire for
each token), so its weights get only 1/32 of the batch. To give one expert 296 conversations'
worth of work you need 32 times as many: 296 × 32 = 9,472, which is the "batch to reach
296" row.

Some models are gated: their files need a logged-in Hugging Face account. For those the
command stops with an error instead of guessing. Fetch a gated config without an account
and you get HTTP 401. The series moved off Llama-3, its earlier example model, precisely
because its config returns 401 to a reader without an account.

## The four calculations

| | Answers |
|---|---|
| **roofline** | Am I compute-bound or memory-bound, and at what batch size? |
| **kvcache** | How much memory is one conversation, and how many fit? |
| **amdahl** | What is that kernel speedup actually worth end to end? |
| **evalstats** | Does my benchmark have enough items to tell two scores apart? |

### 1. The ridge point

```python
from llm_inference_arithmetic import GPUS, ridge_point, decode_intensity

tflops, bandwidth, _ = GPUS["h100-sxm"]
ridge_point(tflops, bandwidth)        # 295.5 FLOP per byte
```

```bash
lia roofline --gpu h100-sxm --batch 32
```

A GPU can only do so much maths per second and move so many bytes per second. The
`GPUS` table holds both figures for five cards. Its maths figure is dense BF16 TFLOP/s:
trillions of operations per second on 16-bit numbers, without the sparsity trick that
doubles the headline figure on vendor spec sheets. Its bytes figure is HBM bandwidth in
TB/s, how fast the card's own memory can be read. Divide the first by the second and you
have the ridge point. For an H100 SXM that is 990 / 3.35 = 295.5. The unit is FLOP per
byte: maths operations per byte loaded. Roofline is the name for this picture of a
GPU: two limits, memory and maths, and a ridge point where they meet. Below the ridge
point the tensor cores (the units that do matrix multiplies) idle waiting for memory,
and faster arithmetic buys nothing.

Arithmetic intensity is how many maths operations you get out of each byte you load.
**In decode, arithmetic intensity *is* the batch size.** Each weight is loaded from
memory once per step and then multiplied once for every conversation in the batch, so
operations per byte loaded equals the number of conversations. So
`decode_intensity(32) == 32`, and you would need a batch near 295 to saturate an H100.
`lia roofline` prints the consequence. At batch 32 the weight multiplies reach
32 operations per byte times 3.35 TB/s = 107.2 TFLOP/s, which is 10.8 percent of the
card's 990 peak.

**Attention is the exception, and it is the important one.** Attention reads the
conversation's KV cache. Every conversation reads its own. Nothing is shared
across the batch, so attention's intensity stays around 1 however you batch. Batching
lifts the weight multiplies (the matrix multiplies, or GEMMs) off the memory roof. It
cannot lift attention.
`lia roofline` prints 1 for attention. The test suite records a refinement: with
grouped-query attention, eight query heads read the same cached bytes, so the closer
figure is about 8 for these models. Either way it is nowhere near 296.

### 2. Your KV cache

```python
from llm_inference_arithmetic import LLAMA3_70B, kv_gib_per_sequence, sequences_that_fit

kv_gib_per_sequence(LLAMA3_70B, 131072)                    # 40.0 GiB for one sequence
sequences_that_fit(LLAMA3_70B, 131072, total_gib=160, weights_gib=70)   # 2
sequences_that_fit(LLAMA3_70B, 8192,   total_gib=160, weights_gib=70)   # 36
# The roofline wants ~295; the cache allows two.
```

```bash
lia kv --layers 80 --kv-heads 8 --head-dim 128 --context 131072 --total 160 --weights 70
```

The cache per token is `2 × layers × kv_heads × head_dim × dtype_bytes`. That 2 is for
the two arrays, K and V, kept in every layer. The other four numbers come out of the
model's `config.json`: how many layers, how many key-value heads, how wide each head is,
and how many bytes one number takes (2 for 16-bit). For Llama-3-70B that is
2 × 80 × 8 × 128 × 2 = 327,680 bytes, which is **320 KiB per token**. At 128K context
that is 40 GiB for a single conversation, re-read in full on every decode step. Now the
budget. With 160 GiB across two cards and 70 GiB of weights, 90 GiB is left for cache.
Two conversations fit. At 8K context, 36 fit. `lia kv` prints the whole table from 4K to
128K.

**Your context length sets your batch size.** Not the engine's settings, and not
`--max-num-seqs`, which is vLLM's cap on the batch. The roofline wants about 295. The
cache allows two.

Works on any model straight from its config:

```python
import json
from llm_inference_arithmetic import ModelShape, kv_gib_per_sequence

cfg = json.load(open("config.json"))
shape = ModelShape.from_config(cfg)     # handles num_key_value_heads vs num_attention_heads
kv_gib_per_sequence(shape, 32768)
```

Everything is **GiB**, deliberately. Multiply 320 KiB by 131,072 tokens and you get
41,943,040 KiB. Divide by 1,000,000 instead of 1,048,576 and 40 GiB turns into 41.9.
A draft of the article shipped that mistake. This library did not exist yet.

### 3. What the kernel is worth

```python
from llm_inference_arithmetic import end_to_end_speedup, speedup_ceiling

end_to_end_speedup(p=0.18, s=2)    # 1.10 when attention is 18% of time
end_to_end_speedup(p=0.85, s=2)    # 1.74 when attention is 85% of time
speedup_ceiling(0.18)              # 1.22 if attention became free
```

```bash
lia amdahl --p 0.18 --speedup 2
```

Amdahl's law says what a faster kernel is worth to the whole job. Here `p` is the share
of total time spent in the kernel and `s` is how much faster you made it. The same 2×
kernel gives a 10 percent gain when attention is 18 percent of the time and a 74 percent
gain when attention is 85 percent of the time. Seven times the value, decided by a
workload parameter rather than by the code. The ceiling is what you would get if the
kernel became free. **Compute the ceiling before starting work**: if it is 1.22×, no
kernel will save that service and the effort belongs at another layer. `lia amdahl` prints the
same figures and, when the ceiling is under 1.25×, says so in words.

### 4. Can your benchmark prove your claim?

Turn a percentage back into a count of items before you believe a gap. A benchmark
scores a model on a fixed number of items, so a score is a count divided by that number.
On 164 items, a gap of "0.8 points" is one item. It took me two mistakes to learn that.

The examples compare a model stored at 16 bits per weight (BF16) with the same model
squeezed to 4 bits per weight. Squeezing is called quantization. The 4-bit format is
INT4, written W4A16 when the activations, the values flowing between layers, stay at
16 bits. The two benchmarks are HumanEval, 164 coding problems scored by the share
solved on the first try (pass@1), and MMLU-Pro, 12,032 multiple-choice questions shown
with five worked examples first (5-shot).

This one exists because of two mistakes, and each was worse than the one before it.

The first: an article draft asserted that INT4 costs *"a fifth of your coding ability"*,
on a table showing HumanEval falling 39.02% → 31.10%. The power check asks whether the
benchmark has enough items to see a gap this size. It said the gap was
thirteen problems out of 164, too small to rule out chance. A gap is called significant
when the chance of seeing one at least that big, with no real difference behind it, is
under 5 percent, and this one was not. Correct, and beside the point, because those
numbers are not in the paper they were attributed to, or in any other I could find.

The second was found while checking the replacement. HumanEval 57.0% → 56.3% went in as
the corrected pair. Both numbers are real, both sit in exactly the rows claimed, and
both are the wrong column. Table 3 runs its columns in this order: MMLU-Pro, then
Arena-Hard Win-Rate, then HumanEval pass@1, then HumanEval+. The HumanEval pass@1 scores
for that row are 79.7 and 80.5. The column immediately to its left, Arena-Hard Win-Rate,
holds 57.0 and 56.3. Reading one column too far left lifts those instead. A fabricated number at least looks
unfamiliar. A number lifted from the neighbouring column looks exactly like
the number you wanted, because it is a real measurement of a real model.

The direction reverses with the correction. INT4 does not lose ground on HumanEval at
70B, it scores slightly higher.

The code below reports two statistics for each pair. z is how many times bigger the gap is
than the random wobble a benchmark this size produces. p is the chance of seeing a gap at
least this big if the two models were really equal.

The real figures, from *Give Me BF16 or Give Me Death?*
([arXiv:2411.02355v4](https://arxiv.org/abs/2411.02355), Table 3, Llama-3.1-70B-Instruct,
`HumanEval pass@1` and `MMLU-Pro 5-shot` columns):

```python
from llm_inference_arithmetic import recover_count, two_proportion_test, min_n_for_difference

# HumanEval pass@1, 164 problems: BF16 79.7 -> W4A16 80.5
recover_count(79.7, 164), recover_count(80.5, 164)   # (131, 132): a ONE-problem gap
two_proportion_test(79.7, 80.5, 164)                 # z=-0.14, p=0.89: nowhere near

# MMLU-Pro 5-shot, 12,032 questions: BF16 48.1 -> W4A16 47.2
two_proportion_test(48.1, 47.2, 12032)               # z=1.39, p=0.16: also not
min_n_for_difference(48.1, 47.2)                     # 23,661 needed; it has 12,032
```

```bash
lia eval --a 79.7 --b 80.5 --n 164
```

`recover_count` turns a percentage back into items: 79.7 percent of 164 problems is 131
and 80.5 percent is 132. The whole INT4 difference is one problem, and it is in INT4's
favour. `two_proportion_test` computes the z and p above. The usual bar for calling a gap
significant is p under 0.05. Neither gap gets close. `min_n_for_difference` says how many items the benchmark
would need before a gap this small could count: 23,661 for the MMLU-Pro pair, against
the 12,032 it has.

Benchmark sizes are from the benchmarks, not from this paper: 164 problems from the
Codex paper ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374), §2.2), and 12,032 test
questions from MMLU-Pro ([arXiv:2406.01574](https://arxiv.org/abs/2406.01574), §3.1).
Every figure above is Llama-3.1-70B-Instruct. The same benchmarks read very differently
at 8B and 405B, so a score quoted without its model cannot be checked by anyone.

Neither benchmark separates its quantized model from 16-bit, and the paper agrees. It
calls FP8 (8-bit floating point) *"effectively lossless across all model scales"*, INT8
(8-bit integer) a *"surprisingly low (1-3%)"* degradation, and INT4 *"more competitive
than expected, rivaling 8-bit quantization"*. Those three phrases are v2 and later. v1
words all three differently, which is why the citation above is pinned to a version.

Three lessons, from the cheapest to the most expensive. Convert percentages to items
before believing a delta. Check that the delta is in the source at all. And when it is,
check that you read it out of the right column, because that is the failure that
survives every sanity check you would think to run: the number is real, the row is
right, the arithmetic on it is sound, and the conclusion is still backwards.

**The correct test is one you usually cannot run.** Both models saw the same 164
problems, so the two sets of results are paired, and the right test for paired results
is McNemar's. It works from the discordant pairs, the items the two models disagree on:
`b` is the number of items model A got right and model B got wrong, and `c` is the
reverse. A published percentage keeps only the difference between them:

```python
from llm_inference_arithmetic import mcnemar_exact
mcnemar_exact(b=15, c=2)     # 0.0023, significant
mcnemar_exact(b=30, c=17)    # 0.0789, not significant
```

Same net difference of 13. Opposite verdicts. Nobody can tell which world they are in
from the numbers as published. An evaluation report should carry discordant counts,
and almost never does.

**And overlapping intervals do not mean "no difference."** A confidence interval is the
range of scores a benchmark of that size cannot tell apart from the one it reported.
The Wilson interval is the specific recipe used here to work out that range. It stays
reliable even when the benchmark has few items.
Neither pair above shows the trap, because neither gap is significant. So here is a
significant pair at the same n = 12,032. Scores of 50.0% and 48.7% give Wilson intervals
of [49.11, 50.89] and [47.81, 49.60], overlapping by half a point, and a two-proportion
test returns p = 0.044. The bars overlap, yet that gap is real. Two questions get mixed
up here. "Do these two error bars overlap?" is not the same question as "is the gap
between the two scores significant?" A gap can be significant while the bars still
plainly overlap. Non-overlap proves significance. Overlap proves nothing either way. The bars
for this pair keep overlapping until the gap reaches about p = 0.006, so a real
difference can hide under an overlap all the way from p = 0.05 down to there. Never read
significance off whether the error bars touch.

## The other commands

```bash
lia moe --fired 8 --total 256 --batch 533     # expert intensity, and the batch it needs
lia attention --seq-len 65536                 # the score matrix FlashAttention refuses to write
lia prefix --text-a "..." --text-b "..."      # what an engine can really reuse (needs tiktoken)
lia prefix --text-a "..." --text-b "..." --compare   # the same, under three OpenAI encodings
lia prefix --tokens-a 1,2,3 --tokens-b 1,2,4  # the same, from your own tokenizer's ids
```

`lia moe` exists because its absence produced a published error. Expert-layer intensity
is experts fired over experts total. Part 1 of the series divided active parameters by
total parameters instead. Take a mixture-of-experts model with 35 billion parameters in
total and 3 billion of them used for each token. Part 1 divided 3 by 35 and got 0.086. That
model routes each token to 8 of its 256 experts, so the right ratio is 8/256 = 0.031. The parameter ratio counts the attention weights, which every
token uses, in both the top and the bottom of the fraction, and comes out close to three
times too kind. One thing to know if you run both commands: `lia moe` prints a batch of
9,457 for 8 of 256, because it divides by the unrounded ridge point of 295.5. `lia model`
prints 9,472 because it rounds the ridge to 296 first.

`lia attention` prints the size of the score table attention would write if it did the
naive thing. For S tokens the table is S by S. At 65,536 tokens that is 65,536 × 65,536
entries at 2 bytes each, 8.59 GB, for one head in one layer. FlashAttention computes the
same result in tiles and never writes the table, which is why it exists.

`lia prefix` reports two numbers and the second is the real one. Prefix caching is what
an engine does when two prompts start the same way: it reuses the cached start. vLLM
matches prefixes in blocks of sixteen tokens, so a six-token match is zero blocks and no
reuse at all. The test suite pins the case:

```python
from llm_inference_arithmetic import prefix_reuse

a = list(range(6)) + [9] * 39
b = list(range(6)) + [7] * 39
prefix_reuse(a, b, block_size=16)
# {'matching_tokens': 6, 'reusable_blocks': 0, 'reusable_tokens': 0,
#  'token_fraction': 0.133..., 'real_fraction': 0.0}
```

A token-level diff will tell you 13 percent, 6 of 45 tokens. You will get nothing.

**Tokenization is not universal, and the answer moves with it.** A tokenizer is the
piece that splits text into tokens, and an encoding is one tokenizer's fixed vocabulary.
`tiktoken` is OpenAI's tokenizer library and the only optional import here.
`pyproject.toml` declares no extra for it, so install it yourself with
`pip install tiktoken`. It covers OpenAI's encodings only. Llama, Qwen, Mistral and Gemma
each split text their own way. When I ran `--compare` on two prompts of my own, two of its
three encodings split them like this. I have written the rows out by hand from what I saw,
so read them as my notes and not as the command's exact printout:

```
o200k_base    (gpt-oss)   5 vs 6 tokens,  0 shared   -> nothing reusable
cl100k_base   (GPT-4)     6 vs 6 tokens,  1 shared
```

`What's` was one token under o200k_base and two under cl100k_base, so the prompts
diverged at position zero under one encoding and at position one under the other. Those
two prompts are not in the repo and nothing here re-derives the counts, so treat the
block as an anecdote and run `--compare` on your own prompts. If your model is outside
the OpenAI family, tokenize with its own tokenizer and pass `--tokens-a/--tokens-b`. The
package takes raw ids on purpose. It never has to guess which vocabulary you are on. A
reuse figure computed with the wrong tokenizer is a fact about somebody else's
deployment.

## Two wrinkles in the plan

**One honest wrinkle.** The derivation lands on 74.7 GiB and the articles round it to
75, which moves two of the four published counts by one (533 vs 531, 266 vs 265). Those
are conversations that fit at 4K and at 8K context, the article's 75 GiB figure first
and the derived 74.7 GiB figure second. Rounding a budget and then flooring a division
does that. Pass `--budget-gib 75` to reproduce the articles exactly. Both budgets are in
the test suite.

**A second one, about units.** Section 2 above is GiB throughout, but `plan` takes its
card memory as GB (10⁹) and does the conversion itself. So the 80 in the GPU table is
read as 80 GB, which is 74.5 GiB. An "80GB" H100 actually reports about 79.7 GiB of total
memory per card. That per-card total is unrelated to the 79.7 GiB of free memory across
both cards in the "Run it" sum, which came from 85.6 GB and only happens to land on the
same number. Divide the real 79.7 by the assumed 74.5 and the card reads about 7 percent
low, and because the weights and the 5 GiB scratch come off the top before anything is
divided, the budget ends up about 13 percent conservative. Run
`lia plan --gpu-gb 85.58 --context 131072 --target 32`, which is 79.7 GiB per card
written in GB, and the budget is 84.3 GiB, so 18 conversations fit instead of 16. It is
the convention the published example was computed under, so it stays. `--budget-gib` bypasses the derivation
entirely if you would rather name the number yourself.

## How the numbers are checked

Every figure quoted in the articles is pinned in `tests/`. If a number in an article
changes and the suite does not, one of them is wrong.
[tsfm-bakeoff](https://github.com/netsatsawat/tsfm-bakeoff) uses the same contract for
its benchmark data: fail rather than let the prose and the arithmetic drift apart.

```bash
python3 tests/test_arithmetic.py     # no pytest needed
python3 -m pytest tests/ -q          # if you have it
```

`scripts/verify_readme_claims.py` goes one step further. It recomputes the figures in
this file's code blocks, its two `lia` output blocks and the rounding wrinkle. Its
sources are the package, the CLI run in-process (called as a Python function, no shell),
and the config copy in the tests. If any of those figures no longer appears in the text,
the automated check fails. The prose around the blocks shows working that the script
does not cover. I checked that working by hand. Every step can be redone from the
numbers on the page. The script also polices where the bad figures from section 4 may
appear: one paragraph here, and one comment in the tests.

### Two implementations, on purpose

`docs/index.html` is the same arithmetic in JavaScript, written independently and served
as a static page. It has five calculators: the four above plus the capacity plan. That
redundancy pays for itself. The two were compared field by field and disagreed on one,
the Wilson interval. In the browser, the percentage was snapped back to a whole number
of items, and in Python it was not. 79.7% of 164 is 131 items, and the interval
belongs to 131/164 rather than to 0.797. The Python was changed to match, and a test now
pins it. A single implementation cannot catch that class of mistake. Two can.

## What is in the repo

```
src/llm_inference_arithmetic/
  __init__.py      the four calculations, the GPUS table and the model shapes
  planning.py      reading a config.json, the capacity plan, moe, attention, prefix
  cli.py           the lia command
tests/test_arithmetic.py        32 checks that pin the published figures
scripts/verify_readme_claims.py the automated gate for this file
docs/index.html                 the five browser calculators
docs/pens/                      the four calculators as single files for CodePen
docs/vllm-tuning-cheatsheet.html  a one-page sheet of six vLLM settings (PDF alongside)
tools/make-pens.js              generates docs/pens/
```

The [vLLM tuning cheat sheet](docs/vllm-tuning-cheatsheet.html) is not part of the
package. It lists six vLLM settings, what each one is really a knob on, and which way to
move it. It is pinned to vLLM v0.27.1 because the defaults move between releases.

## What this is not

Not a profiler (a tool that measures where a running program spends its time) and not
a benchmark harness (a tool that runs the test items against a model). It computes the
numbers you should know *before* you rent a GPU, and the ones you should check *after*
you read someone else's result. For the actual measurement you still want `nsys`,
NVIDIA's profiler, and a real workload.

The `GPUS` table uses dense BF16 throughput, not the sparsity-doubled figure vendors
headline. It reads card memory as GB, as described above. `lia prefix --text-a` knows
OpenAI's encodings only. `lia model` needs the network, and on a gated model it stops
with an error rather than guessing.

## Embedding the calculators

**satsawat.ai, or any site you control:** an iframe is enough.

```html
<iframe src="https://netsatsawat.github.io/llm-inference-arithmetic/pens/part-2-kv-cache.html"
        style="width:100%;height:620px;border:0" loading="lazy"
        title="What one conversation costs"></iframe>
```

**Medium: not this page.** Medium accepts no HTML, no iframes and no scripts. It turns a
pasted URL into a preview card through Embed.ly, a service that knows about 300
providers: CodePen and Observable are on the list, a GitHub Pages URL is not. Paste a
link to the page above into a Medium story and you get a link card, not a calculator.

The route to a live calculator inside a Medium story is to host the same code somewhere
Embed.ly already trusts, then paste that URL on its own line. `docs/pens/` exists for
that: four self-contained files, one per calculator, each of which pastes straight into
a CodePen HTML pane. A pen is CodePen's name for one such page.

| file | what it does |
|---|---|
| `part-1-roofline.html` | ridge point, intensity, and whether you are memory-bound |
| `part-2-kv-cache.html` | cache per token, per conversation, and how many fit |
| `part-4-eval-stats.html` | counts, Wilson intervals, and whether the gap resolves |
| `part-7-amdahl.html` | what a kernel speedup is worth end to end |

One per article rather than one big page. That way the calculator sits next to the
arithmetic it belongs to instead of making the reader leave and come back. The part
numbers in the file names come from `tools/make-pens.js`, which generates all four. The
test docstrings number some of the same material differently, so read the file names as
labels and not as a map of the series. Four hand-written copies of the same theme and the
same number formatting would be four places for them to drift. So the files are
generated. Regenerate them after an edit with `node tools/make-pens.js`.

## Links

- The series *LLM Inference, Measured* at [satsawat.ai](https://satsawat.ai).
- [The calculators in your browser](https://netsatsawat.github.io/llm-inference-arithmetic/).
- [vLLM tuning cheat sheet](docs/vllm-tuning-cheatsheet.html), one page.
- License: MIT.
