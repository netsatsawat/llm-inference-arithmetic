/**
 * Standalone calculators, one per article.
 *
 * Medium does not accept HTML, iframes or scripts. It unfurls pasted URLs through
 * Embed.ly, which supports about 300 providers: CodePen and Observable among them,
 * a GitHub Pages URL not among them. So a link to docs/index.html becomes a link
 * card on Medium, and the only route to a *live* calculator inside a Medium story
 * is to host the same code somewhere Embed.ly already trusts.
 *
 * These files are that. Each is self-contained and pastes straight into a CodePen
 * HTML pane, or serves as-is. One per article, so the calculator sits next to the
 * arithmetic it belongs to rather than making the reader leave and come back.
 *
 * Generated rather than hand-written, because four copies of the same theme and the
 * same number formatting is four places for them to drift.
 *
 *   node tools/make-pens.js
 */
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, '..', 'docs', 'pens');

/* Same palette as docs/index.html and the article figures. */
const CSS = `
  :root{--cloth:#0B0C0E;--band:#171A1F;--well:#101318;--rule:rgba(237,234,227,.16);
        --ink:#EDEAE3;--ink2:#B9B5AC;--ink3:#8B8880;--gold:#D4A843;--gold-ink:#A88434;--cool:#6E9FD4;
        --sans:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;
        --mono:ui-monospace,Menlo,Consolas,monospace}
  *{box-sizing:border-box}
  body{margin:0;background:var(--cloth);color:var(--ink2);font-family:var(--sans);
       font-size:15px;line-height:1.6;padding:1.25rem}
  .k{font-family:var(--mono);font-size:.68rem;letter-spacing:.2em;color:var(--gold-ink);text-transform:uppercase}
  h1{font-size:1.15rem;color:var(--ink);margin:.3rem 0 .2rem;letter-spacing:-.01em}
  p.why{font-size:.88rem;color:var(--ink3);margin:0 0 1.1rem}
  .c{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:.9rem 1.1rem}
  label{display:block;font-family:var(--mono);font-size:.65rem;letter-spacing:.13em;
        text-transform:uppercase;color:var(--ink3);margin-bottom:.3rem}
  input[type=number],select{width:100%;background:var(--well);color:var(--ink);border:1px solid var(--rule);
        border-radius:4px;padding:.45rem .55rem;font-family:var(--mono);font-size:.9rem}
  input[type=range]{width:100%;accent-color:var(--gold)}
  .v{font-family:var(--mono);color:var(--gold)}
  .out{margin-top:1.1rem;border-top:1px solid var(--rule);padding-top:.9rem}
  dl{display:grid;grid-template-columns:1fr auto;gap:.3rem 1rem;margin:0;font-family:var(--mono);font-size:.9rem}
  dt{color:var(--ink3)} dd{margin:0;color:var(--ink);text-align:right;font-variant-numeric:tabular-nums}
  dd.b{color:var(--gold);font-size:1.05rem;font-weight:700}
  .verdict{margin-top:.9rem;padding:.7rem .85rem;border-radius:5px;font-size:.9rem;
           background:#1E2128;border-left:3px solid var(--ink3);color:var(--ink)}
  .verdict.good{border-left-color:var(--gold)} .verdict.bad{border-left-color:var(--cool)}
  svg{display:block;width:100%;height:auto}
  a{color:var(--gold)}
  footer{margin-top:1.2rem;font-size:.78rem;color:var(--ink3)}
`;

const HELPERS = `
  const $=id=>document.getElementById(id);
  const fmt=(x,d=0)=>x.toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
  const GIB=1024**3;
`;

const FOOT = (part) => `<footer>From <em>LLM Inference, Measured</em>, part ${part}.
  Same arithmetic as a Python package and a CLI:
  <a href="https://github.com/netsatsawat/llm-inference-arithmetic">llm-inference-arithmetic</a>.</footer>`;

/* `extra` appends pen-specific CSS. Only part four uses it, because only part four has
   to teach the statistics as well as compute them. The other three stay byte-identical. */
const page = (title, part, body, js, extra = '') => `<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title>
<style>${CSS}${extra}</style>
${body}
${FOOT(part)}
<script>
"use strict";${HELPERS}${js}
</script>
`;

/* ── part 1: the ridge point ───────────────────────────────────────────────── */
const roofline = page('Are you memory-bound?', 'one', `
<div class="k">llm inference · part one</div>
<h1>Are you memory-bound?</h1>
<p class="why">Peak arithmetic rate divided by memory bandwidth is the ridge point. Below it you
are paying to move bytes rather than to multiply them.</p>
<div class="c">
  <div><label for="g">GPU</label><select id="g"></select></div>
  <div><label for="b">Batch <span class="v" id="bv">32</span></label><input type="range" id="b" min="1" max="1024" value="32"></div>
  <div><label for="w">Bytes per weight</label><select id="w">
    <option value="2" selected>2 (BF16)</option><option value="1">1 (FP8)</option><option value="0.5">0.5 (INT4)</option></select></div>
</div>
<div class="out"><dl>
  <dt>ridge point</dt><dd id="r">·</dd>
  <dt>your intensity</dt><dd id="i">·</dd>
  <dt>achievable</dt><dd class="b" id="p">·</dd>
</dl><div class="verdict" id="v"></div></div>`, `
const G={"h100-sxm":[990,3.35,"H100 SXM"],"h200":[990,4.8,"H200"],"a100-80":[312,2.04,"A100 80GB SXM"],"l40s":[362,.864,"L40S"]};
for(const[k,g]of Object.entries(G)){const o=document.createElement("option");o.value=k;
  o.textContent=g[2]+": "+g[0]+" TFLOP/s, "+g[1]+" TB/s";$("g").appendChild(o);}
function go(){const[t,bw]=G[$("g").value],b=+$("b").value,by=+$("w").value;
  const ridge=t/bw,int=b*(2/by),pct=Math.min(100,int/ridge*100);
  $("bv").textContent=b;$("r").textContent=fmt(ridge,1)+" FLOP/byte";$("i").textContent=fmt(int);
  $("p").textContent=pct<1?"<1% of peak":fmt(pct,0)+"% of peak";
  const v=$("v");
  if(int>=ridge){v.className="verdict good";v.textContent="Compute-bound. The arithmetic is the limit, which is where you want to be.";}
  else{v.className="verdict bad";v.textContent="Memory-bound. Reaching the ridge would take a batch of "+fmt(Math.ceil(ridge/(2/by)))+
    ", and attention never gets there whatever you do.";}}
["g","b","w"].forEach(i=>$(i).addEventListener("input",go));go();`);

/* ── part 2: the cache, and what fits ──────────────────────────────────────── */
const kv = page('What one conversation costs', 'two', `
<div class="k">llm inference · part two</div>
<h1>What one conversation costs, and how many fit</h1>
<p class="why">Two, times the layers whose cache grows, times the key-value heads, times the head
dimension, times the bytes. Read <code>layer_types</code> first: layers that only see a sliding
window stop growing, and counting them doubles your answer.</p>
<div class="c">
  <div><label for="L">Layers that grow</label><input type="number" id="L" value="18" min="1"></div>
  <div><label for="H">Key-value heads</label><input type="number" id="H" value="8" min="1"></div>
  <div><label for="D">Head dimension</label><input type="number" id="D" value="64" min="1"></div>
  <div><label for="T">Cache precision</label><select id="T"><option value="2" selected>16-bit</option><option value="1">8-bit</option></select></div>
  <div><label for="C">Context <span class="v" id="cv">128K</span></label><input type="range" id="C" min="10" max="17" value="17"></div>
  <div><label for="B">KV budget, GiB</label><input type="number" id="B" value="75" min="1"></div>
</div>
<div class="out"><dl>
  <dt>per token</dt><dd id="pt">·</dd>
  <dt>per conversation</dt><dd id="pc">·</dd>
  <dt>conversations that fit</dt><dd class="b" id="f">·</dd>
</dl><div class="verdict" id="v"></div></div>`, `
function go(){const per=2*+$("L").value*+$("H").value*+$("D").value*+$("T").value;
  const ctx=2**+$("C").value,budget=+$("B").value;
  const g=per*ctx/GIB,fits=Math.floor(budget/g);
  $("cv").textContent=ctx>=1024?ctx/1024+"K":ctx;
  $("pt").textContent=fmt(per)+" bytes = "+fmt(per/1024,0)+" KiB";
  $("pc").textContent=g<1?fmt(g*1024,0)+" MiB":fmt(g,2)+" GiB";
  $("f").textContent=fmt(fits);
  const v=$("v");
  if(fits<296){v.className="verdict bad";v.textContent=fits+" is below part one's dense ridge of 296, so the cache "+
    "decided your batch size before the scheduler got a vote.";}
  else{v.className="verdict good";v.textContent=fits+" clears the dense ridge of 296. Note that a mixture-of-experts model "+
    "has a much further ridge on its expert layers.";}}
["L","H","D","T","C","B"].forEach(i=>$(i).addEventListener("input",go));go();`);

/* ── part 4: can the benchmark resolve it ──────────────────────────────────
   This one carries a long explanation on purpose. The other three pens compute a
   number the reader already understands. This one computes a number that is routinely
   misread, so the page has to teach the reading as well as do the arithmetic. Written
   for somebody who has never run a significance test and has just been handed one. */
const EVAL_CSS = `
  .lede{font-size:.92rem;color:var(--ink2);margin:0 0 1rem;max-width:64ch}
  .presets{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 1.1rem}
  .presets button{background:var(--band);color:var(--ink2);border:1px solid var(--rule);
    border-radius:999px;padding:.32rem .7rem;font-family:var(--mono);font-size:.72rem;cursor:pointer}
  .presets button:hover{border-color:var(--gold-ink);color:var(--ink)}
  .presets button:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
  .hint{display:block;font-family:var(--sans);font-size:.72rem;line-height:1.45;
    letter-spacing:0;text-transform:none;color:var(--ink3);margin:.3rem 0 0}
  dl{align-items:start}
  dt small{display:block;font-family:var(--sans);font-size:.72rem;line-height:1.45;
    letter-spacing:0;text-transform:none;color:#6F6C66;margin:.1rem 0 .35rem}
  h2{font-size:.95rem;color:var(--ink);margin:1.6rem 0 .5rem;letter-spacing:-.005em}
  details{border-top:1px solid var(--rule);padding:.6rem 0}
  details summary{cursor:pointer;color:var(--ink);font-size:.88rem;list-style:none}
  details summary::-webkit-details-marker{display:none}
  details summary::before{content:"+ ";color:var(--gold);font-family:var(--mono)}
  details[open] summary::before{content:"− "}
  details summary:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
  details .body{font-size:.86rem;color:var(--ink2);padding:.5rem 0 .2rem;max-width:66ch}
  details .body p{margin:.55rem 0}
  details .body code{font-family:var(--mono);font-size:.82rem;color:var(--gold);
    background:var(--well);padding:.05rem .3rem;border-radius:3px}
  .trap{margin-top:1.4rem;background:var(--well);border:1px solid var(--rule);
    border-left:3px solid var(--cool);border-radius:5px;padding:.85rem .95rem;max-width:66ch}
  .trap h3{margin:0 0 .4rem;font-size:.86rem;color:var(--ink)}
  .trap p{margin:.5rem 0;font-size:.85rem}
  .trap table{border-collapse:collapse;font-family:var(--mono);font-size:.74rem;margin:.6rem 0 .2rem}
  .trap td,.trap th{padding:.18rem .55rem;text-align:right;border-bottom:1px solid var(--rule);color:var(--ink2)}
  .trap th{color:var(--ink3);font-weight:400;text-align:right}
  .trap td:first-child,.trap th:first-child{text-align:left}
  .trap .hit{color:var(--gold);font-weight:700}
  .wrapx{overflow-x:auto}
`;

const evalStats = page('Can the benchmark prove its claim?', 'four', `
<div class="k">llm inference · part four</div>
<h1>Can the benchmark prove its own claim?</h1>
<p class="lede">Somebody hands you two benchmark scores and a conclusion. This works out
whether the benchmark was ever capable of telling those two scores apart. Most of the time
it was not, and the honest answer is "this cannot be resolved" rather than "there is no
difference". Start with a preset, then read <em>How to read this</em> underneath.</p>

<div class="presets" id="presets"></div>

<div class="c">
  <div><label for="a">Baseline, %</label><input type="number" id="a" value="79.7" step=".01">
    <span class="hint">The score you are comparing against. Usually the unquantized model.</span></div>
  <div><label for="b">Comparison, %</label><input type="number" id="b" value="80.5" step=".01">
    <span class="hint">The new one. Order does not matter; the test is two-sided.</span></div>
  <div><label for="n">Items</label><input type="number" id="n" value="164" min="2">
    <span class="hint">How many questions or problems the benchmark contains. Not the number
    of models, and not the number of runs. HumanEval is 164. MMLU-Pro is 12,032.</span></div>
  <div><label for="k">They disagree on <span class="v" id="kv">15%</span></label><input type="range" id="k" min="1" max="60" value="15">
    <span class="hint">An assumption, not data: on what share of items do the two models give
    different answers? Nobody publishes this. It decides the honest verdict.</span></div>
</div>

<div class="out"><dl>
  <dt>counts<small>The scores as whole items. This is the first thing to look at.</small></dt><dd id="c">·</dd>
  <dt>difference<small>How many items separate them. Percentages hide how small this is.</small></dt><dd id="d">·</dd>
  <dt>95% interval, baseline<small>The range the true score plausibly sits in.</small></dt><dd id="ia">·</dd>
  <dt>95% interval, comparison<small>Same, for the other model.</small></dt><dd id="ib">·</dd>
  <dt>items the unpaired test needs<small>How large the benchmark would have to be to resolve a gap this size.</small></dt><dd id="mn">·</dd>
  <dt>unpaired p, a ceiling<small>Small means probably real. Large does NOT mean "no difference".</small></dt><dd class="b" id="p">·</dd>
  <dt>paired p, if they disagree that often<small>The test you should be running, under the assumption in the slider.</small></dt><dd class="b" id="pm">·</dd>
</dl><div class="verdict" id="v"></div></div>

<h2>How to read this</h2>

<details open><summary>Step 1. Turn the percentage back into items</summary>
<div class="body"><p>A benchmark score is a count divided by the number of questions. "57.0%"
sounds continuous and precise. On a 164-problem benchmark it is 93 problems, and the next
achievable score up is 94, which is 57.32%. There is nothing in between.</p>
<p>So a "0.7 point drop" on that benchmark is <em>one problem</em>. One problem is not a
finding. Do this conversion before you form any opinion, because the percentage is designed
to make small differences look substantial.</p></div></details>

<details><summary>Step 2. What the p-value actually says</summary>
<div class="body"><p>The p-value answers one narrow question: <em>if the two models were truly
identical, how often would chance alone produce a gap at least this large?</em></p>
<p><code>p = 0.89</code> means: about 89% of the time. That is completely unremarkable, so
the data gives you no reason to believe the models differ.</p>
<p>The trap is reading a large p as proof they are the same. It is not. It means the
instrument could not tell. A bathroom scale that reads to the nearest kilogram cannot detect
a 200 gram change, and its silence is not evidence that you weigh the same.</p>
<p>Small p means the difference is probably real. Large p means you learned nothing. Those
are not opposites.</p></div></details>

<details><summary>Step 3. Why overlapping error bars prove nothing</summary>
<div class="body"><p>The two intervals above are where each true score plausibly lives. People
eyeball whether they overlap and call it a day. That rule is asymmetric, and only one half
of it works.</p>
<p>If the intervals <em>do not</em> overlap, the difference is significant. That direction is
sound. If they <em>do</em> overlap, you have learned nothing at all: two intervals can overlap
by nearly a third of their width and still differ at <code>p = 0.05</code>. The overlap rule is
equivalent to testing at about <code>p = 0.006</code>, roughly a tenth as permissive as the test
you meant to run.</p>
<p>Read the p-value. Use the picture to communicate, never to decide.</p></div></details>

<details><summary>Step 4. The unpaired test is probably the wrong test</summary>
<div class="body"><p>The p-value above assumes the two models were measured on two independent
samples. They were not. They answered <em>the same questions</em>. That is paired data, and the
correct test is McNemar's.</p>
<p>McNemar ignores every item both models got right and every item both got wrong. It looks
only at the ones they disagreed on: <code>b</code>, where the baseline won, and <code>c</code>,
where the comparison won. The statistic is <code>(b - c) / sqrt(b + c)</code>.</p>
<p>Here is the problem. A published table of percentages gives you <code>b - c</code>, because
that is the visible gap. It never gives you <code>b + c</code>. And <code>b + c</code> is what
decides the answer.</p>
<p>That is what the slider is for. Assume a disagreement rate and watch the verdict move. A
model and its own quantization agree on most items, so realistic rates are low, and at low
rates gaps that looked like noise stop looking like noise. Set the slider near 50% and the
paired result collapses back onto the unpaired one, because that is the case where the two
models are effectively unrelated.</p>
<p>Practical consequence: the unpaired p is a <em>ceiling</em> on the evidence. It is safe to
say "this gap is small". It is not safe to say "this gap is not significant".</p></div></details>

<details><summary>Step 5. What to do with the answer</summary>
<div class="body"><p>If the gap is a handful of items, stop. It is not a result, however
confidently it was printed, and no amount of restating it will make it one.</p>
<p>If you need to know, run your own evaluation on your own task, and check the size first.
The row above tells you how many items it would take. A 200-case internal suite cannot see a
one-point change. It can see a ten-point one.</p>
<p>And when you publish an evaluation yourself, publish the discordant counts. They are the
one number that makes a comparison checkable, they cost nothing to record, and almost nobody
includes them.</p></div></details>

<div class="trap">
<h3>The trap that caused this calculator</h3>
<p>An earlier version of this project quoted HumanEval as falling from 57.0% to 56.3% under
INT4. Both numbers are real. Both sit in the correct row of the source table, for the correct
model. Both are the wrong column.</p>
<div class="wrapx"><table>
<tr><th>Llama-3.1-70B-Instruct</th><th>MMLU-Pro</th><th>Arena-Hard</th><th>HumanEval</th></tr>
<tr><td>BF16</td><td>48.1</td><td class="hit">57.0</td><td>79.7</td></tr>
<tr><td>W4A16-INT</td><td>47.2</td><td class="hit">56.3</td><td>80.5</td></tr>
</table></div>
<p>Arena-Hard sits immediately left of HumanEval, so reading one column short returns 57.0
and 56.3 intact. The real HumanEval figures are 79.7 and 80.5, which reverses the finding:
the quantized model scored slightly <em>higher</em>.</p>
<p>This is worse than an invented number. An invented number looks unfamiliar. A number from
the neighbouring column is a genuine measurement of a genuine model, it survives every sanity
check you would think to run, and it still gives you a backwards conclusion. No calculator can
catch it. Read the cell against its own column header, from the paper rather than from a
summary of the paper.</p>
</div>`, `
function erf(x){const s=Math.sign(x);x=Math.abs(x);const t=1/(1+.3275911*x);
  return s*(1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x));}
const cdf=z=>.5*(1+erf(z/Math.SQRT2));
function wil(x,n,z=1.959964){const p=x/n,d=1+z*z/n,c=(p+z*z/(2*n))/d,
  h=z/d*Math.sqrt(p*(1-p)/n+z*z/(4*n*n));return[Math.max(0,c-h)*100,Math.min(1,c+h)*100];}
/* Same closed form as min_n_for_difference in the Python package. */
function minN(a,b,z=1.959964){const p1=a/100,p2=b/100;if(p1===p2)return null;
  const pool=(p1+p2)/2;return Math.ceil(2*pool*(1-pool)*Math.pow(z/(p1-p2),2));}

/* Real cases, so a newcomer starts from something that happened rather than from
   numbers they invented. The last one is the only gap here that survives a test. */
const PRESETS=[
  ["HumanEval, INT4 vs BF16",79.7,80.5,164,15],
  ["MMLU-Pro, INT4 vs BF16",48.1,47.2,12032,15],
  ["A gap that is real",50.0,48.7,12032,15],
  ["Your 200-case internal suite",92.0,91.0,200,15]
];
(function(){const box=$("presets");PRESETS.forEach(function(pr){
  const btn=document.createElement("button");btn.type="button";btn.textContent=pr[0];
  btn.addEventListener("click",function(){$("a").value=pr[1];$("b").value=pr[2];
    $("n").value=pr[3];$("k").value=pr[4];go();});box.appendChild(btn);});})();

function go(){const n=Math.max(2,+$("n").value),xa=Math.round(+$("a").value/100*n),xb=Math.round(+$("b").value/100*n);
  const gap=Math.abs(xa-xb);
  $("c").textContent=xa+" and "+xb+" of "+fmt(n);
  $("d").textContent=fmt(gap)+(gap===1?" item":" items");
  const A=wil(xa,n),B=wil(xb,n);
  $("ia").textContent="["+fmt(A[0],2)+", "+fmt(A[1],2)+"]";$("ib").textContent="["+fmt(B[0],2)+", "+fmt(B[1],2)+"]";
  const need=minN(+$("a").value,+$("b").value);
  $("mn").textContent=need===null?"no gap to resolve":fmt(need)+(need>n?"  (has "+fmt(n)+")":"");
  const pool=(xa+xb)/(2*n),se=Math.sqrt(pool*(1-pool)*(2/n)),z=se>0?(xa/n-xb/n)/se:0,pv=2*(1-cdf(Math.abs(z)));
  $("p").textContent=pv<.001?"< 0.001":fmt(pv,3);
  /* McNemar on the discordant pairs: z = (b-c)/sqrt(b+c). The net difference b-c is
     recoverable from the published percentages; b+c is not, and it decides the answer. */
  const k=+$("k").value;$("kv").textContent=k+"%";
  const m=Math.max(gap,Math.min(n,Math.round(k/100*n)));
  const pm=gap>0?2*(1-cdf(gap/Math.sqrt(m))):1;
  $("pm").textContent=(pm<.001?"< 0.001":fmt(pm,3))+"  on "+fmt(m)+" pairs";
  const ov=!(A[1]<B[0]||B[1]<A[0]),v=$("v");
  if(pv<.05){v.className="verdict good";v.textContent="Real on the conservative reading. Even the "+
    "test that understates the evidence clears 0.05"+
    (ov?", and the intervals still overlap, which is exactly why overlap proves nothing.":".");}
  else if(pm<.05){v.className="verdict bad";v.textContent="Undecidable from what was published. "+
    "A gap of "+fmt(gap)+(gap===1?" item":" items")+" looks like noise against all "+fmt(n)+
    " items, but against the "+fmt(m)+" the two models actually answer differently it does not. "+
    "The count that settles it is the one the table left out, so the honest report is \\"cannot be "+
    "resolved from these figures\\", not \\"no difference\\".";}
  else{v.className="verdict bad";v.textContent="Nothing established either way. A gap of "+fmt(gap)+
    (gap===1?" item":" items")+" on "+fmt(n)+" stays inside what chance produces even when only "+
    fmt(m)+" items are in play"+(need&&need>n?", and resolving it would take about "+fmt(need)+
    " items rather than "+fmt(n)+".":".");}}
["a","b","n","k"].forEach(i=>$(i).addEventListener("input",go));go();`, EVAL_CSS);

/* ── part 7: what a kernel speedup is worth ────────────────────────────────── */
const amdahl = page('What is that speedup worth?', 'seven', `
<div class="k">llm inference · part seven</div>
<h1>What is that kernel speedup actually worth?</h1>
<p class="why">A kernel that runs twice as fast helps in proportion to how much of your time was
spent in it. The number you were quoted was measured on somebody else's workload.</p>
<div class="c">
  <div><label for="p">Share of time there <span class="v" id="pv">18%</span></label><input type="range" id="p" min="1" max="99" value="18"></div>
  <div><label for="s">Kernel speedup <span class="v" id="sv">2.0×</span></label><input type="range" id="s" min="11" max="100" value="20"></div>
</div>
<div class="out"><dl>
  <dt>end to end</dt><dd class="b" id="e">·</dd>
  <dt>ceiling, if it became free</dt><dd id="c">·</dd>
  <dt>needed for 1.5× overall</dt><dd id="n">·</dd>
</dl><div class="verdict" id="v"></div></div>`, `
function go(){const p=+$("p").value/100,s=+$("s").value/10;
  $("pv").textContent=fmt(p*100,0)+"%";$("sv").textContent=fmt(s,1)+"×";
  const e=1/((1-p)+p/s),c=1/(1-p);
  $("e").textContent=fmt(e,2)+"×";$("c").textContent=fmt(c,2)+"×";
  const need=c<=1.5?null:p/(1/1.5-(1-p));
  $("n").textContent=need===null?"impossible: the ceiling is "+fmt(c,2)+"×":fmt(need,1)+"×";
  const v=$("v");v.className=e>=1.25?"verdict good":"verdict bad";
  v.textContent="A "+fmt(s,1)+"× kernel on "+fmt(p*100,0)+"% of your time is "+fmt(e,2)+"× end to end. "+
    (e<1.15?"Close to nothing, which is why speedups do not transfer.":"Worth having, but that is your p, not theirs.");}
["p","s"].forEach(i=>$(i).addEventListener("input",go));go();`);

fs.mkdirSync(OUT, { recursive: true });
const pens = {
  'part-1-roofline.html': roofline,
  'part-2-kv-cache.html': kv,
  'part-4-eval-stats.html': evalStats,
  'part-7-amdahl.html': amdahl,
};
console.log(`pens → ${OUT}`);
for (const [name, html] of Object.entries(pens)) {
  fs.writeFileSync(path.join(OUT, name), html);
  console.log(`  ${name.padEnd(26)} ${(html.length / 1024).toFixed(1)} KB`);
}
