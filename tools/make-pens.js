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

const page = (title, part, body, js) => `<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title>
<style>${CSS}</style>
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

/* ── part 4: can the benchmark resolve it ──────────────────────────────────── */
const evalStats = page('Can the benchmark prove its claim?', 'four', `
<div class="k">llm inference · part four</div>
<h1>Can the benchmark prove its own claim?</h1>
<p class="why">Convert the percentages back into items before believing anything. This is the
unpaired test. If both models were scored on the same items the correct test is McNemar's,
which needs per-item results, and this p-value is then an upper bound.</p>
<div class="c">
  <div><label for="a">Baseline, %</label><input type="number" id="a" value="79.7" step=".01"></div>
  <div><label for="b">Comparison, %</label><input type="number" id="b" value="80.5" step=".01"></div>
  <div><label for="n">Items</label><input type="number" id="n" value="164" min="2"></div>
</div>
<div class="out"><dl>
  <dt>counts</dt><dd id="c">·</dd>
  <dt>difference</dt><dd id="d">·</dd>
  <dt>95% interval, baseline</dt><dd id="ia">·</dd>
  <dt>95% interval, comparison</dt><dd id="ib">·</dd>
  <dt>unpaired p</dt><dd class="b" id="p">·</dd>
</dl><div class="verdict" id="v"></div></div>`, `
function erf(x){const s=Math.sign(x);x=Math.abs(x);const t=1/(1+.3275911*x);
  return s*(1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x));}
const cdf=z=>.5*(1+erf(z/Math.SQRT2));
function wil(x,n,z=1.959964){const p=x/n,d=1+z*z/n,c=(p+z*z/(2*n))/d,
  h=z/d*Math.sqrt(p*(1-p)/n+z*z/(4*n*n));return[Math.max(0,c-h)*100,Math.min(1,c+h)*100];}
function go(){const n=Math.max(2,+$("n").value),xa=Math.round(+$("a").value/100*n),xb=Math.round(+$("b").value/100*n);
  $("c").textContent=xa+" and "+xb+" of "+fmt(n);$("d").textContent=fmt(Math.abs(xa-xb))+" items";
  const A=wil(xa,n),B=wil(xb,n);
  $("ia").textContent="["+fmt(A[0],2)+", "+fmt(A[1],2)+"]";$("ib").textContent="["+fmt(B[0],2)+", "+fmt(B[1],2)+"]";
  const pool=(xa+xb)/(2*n),se=Math.sqrt(pool*(1-pool)*(2/n)),z=se>0?(xa/n-xb/n)/se:0,pv=2*(1-cdf(Math.abs(z)));
  $("p").textContent=pv<.001?"< 0.001":fmt(pv,3);
  const ov=!(A[1]<B[0]||B[1]<A[0]),v=$("v");
  if(pv<.05){v.className="verdict good";v.textContent="Clears a conventional test"+
    (ov?", and the intervals still overlap, which is why overlap proves nothing.":".");}
  else{v.className="verdict bad";v.textContent="Not established by this test. A gap of "+fmt(Math.abs(xa-xb))+
    " items on "+fmt(n)+" is inside what chance produces, but the unpaired test is conservative for paired data, "+
    "so treat this as a ceiling on the evidence, not a verdict.";}}
["a","b","n"].forEach(i=>$(i).addEventListener("input",go));go();`);

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
