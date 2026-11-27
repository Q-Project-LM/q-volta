# Degenerate repetition ("loops") in Inkling-2-Mini at T=1.0 (workstream p28/loops, 2026-11-27)

Problem: through `qv_server` (Inkling q3a5, int4 KV, MTP on) the model "sometimes loops". Live log: 2 of ~1100 generations ran to the 16384-token cap on 44.5K / 55K token
agent prompts. The model card (README section 7) lists it as a known limitation: "Repetitive reasoning. In internal testing the model sometimes became stuck in long reasoning loops, or
repeatedly re-checked its own work without making progress, particularly when debugging code. We recommend setting limits on output tokens and agent turns." It gives no sampling
recommendation; `generation_config.json` has `temperature 1.0` only.

## What the server applies by default (read from the code, confirmed with /props)
`q3a5.qv` carries no embedded generation_config, so the server uses its llama-server defaults: temperature 0.8 (the client's value wins; owner traffic sends 1.0), top_k 40, top_p 0.95,
min_p 0.05, no repeat/presence/frequency penalty, no DRY, random seed per request. With `--mtp` and T > 0 the Inkling engine has NO speculative path (`has_speculative_decode()` is
false, only the Qwen engine implements it), so `spec_ok=0 use_chunk=0` and tokens are drawn one at a time by the host `Sampler`.

## Tools (all committed)
* `tools/loop_bench.py` - loop benchmark client (HTTP, streaming). `run` / `report` / `selftest`; `tools/loop_bench_server.sh` and `tools/loop_bench_driver.sh` start a private server under
  `flock /tmp/qv_bench.lock` in short batches and resume. Prompt cells: `short_chat` (20 prompts), `reason_math` (12), `reason_code` (12), `agent_8k/32k/64k` (tool-using coding-agent transcript built from
  repository files, 6 debugging tasks; the final user message asks to answer without more tool calls). Generation i of a cell uses prompt `i mod len` and seed `seed0 + i`, so runs are comparable.
  Only `temperature`, `seed`, `max_tokens` (8192) are sent unless `--param k=v` says otherwise. Reasoning is on (template default).
* Detector: an n-gram of >= 16 tokens occurring >= 4 times (non-overlapping) in the last 512 tokens = `loop`. Checking n = 16 suffices. A run that reaches the cap without EOS and without that
  repetition is `cap_other` ("runaway": long reasoning without degeneration). Because ordinary text can contain 4 copies of a 16-token phrase (a regex explained piece by piece, repeated SQL), every detected loop
  is followed for 400 more tokens and is called `strong` if >= 50% of the final 512 tokens are covered by 16-grams that occur >= 3 times. Wilson 95% intervals are reported for both.
* `tools/ink_nll.cpp` (`qv_ink_nll`) - teacher-forced NLL by position on the GPU engine: batched prefill to the window start (KV in the chosen format), then NLL of the next 512 tokens through the decode
  path (`step()`), resuming the prefill after each window. `qv_ink_cpu --nll --nll-out F` writes the same per-position array for the independent CPU engine. `tools/nll_curve.py` bins/compares.
* `tests/test_sampler_dist.cpp` - chi-square of the host Sampler, `build_dist_dense` and one lossless speculative step (accept / residual) against the analytic distribution.
* `tests/test_loop_bench.py` - detector and mock-backend end-to-end test.

## Results (T=1.0, server-default top_k 40 / top_p 0.95 / min_p 0.05, reasoning on, max 8192 new tokens, int4 KV, `--mtp`, dev build e0c600b unless noted)
Honest sample sizes: short_chat and reason_code n=50; reason_math and agent_* n=30 (the brief asked n>=50; GPU time on the shared V100 did not allow it for the long cells); padcode and
fp16 n=24; v_topk20 n=24. Intervals are Wilson 95%. `loop` = detector hit (includes ordinary text that repeats a 16-token phrase 4x), `strong` = still >= 50% repeated after 400 more tokens.

| cell | n | loop | Wilson | strong | cap without repetition (runaway) | median tokens |
|---|---:|---:|---|---:|---:|---:|
| short_chat | 50 | 4.0% | 1.1-13.5% | 1 | 0 | 391 |
| reason_math | 30 | 0.0% | 0.0-11.4% | 0 | 1 | 760 |
| reason_code | 50 | 6.0% | 2.1-16.2% | 0 | 6 | 1961 |
| agent_8k | 30 | 6.7% | 1.8-21.3% | 0 | 17 | 8118 |
| agent_32k | 30 | 10.0% | 3.5-25.6% | 0 | 24 | 8139 |
| agent_64k | 30 | 16.7% | 7.3-33.6% | 1 | 25 | 8150 |

Reading: at 8K-64K agent contexts most generations (57-83%) reach the 8192 cap while still reasoning ("runaway"), only 0-1 in 30 are strong loops. Strict repetition is rare; the
dominant failure is reasoning that does not terminate, growing with context length. Counterexample strong loops seen: a digit-by-digit `` `3` = 3. `9` = 9. ``enumeration (agent_64k) and a regex
self-correction cycle (padcode_1k, 1 of 24).

### Hypotheses
* L1 long-context numerical error: REFUTED. Teacher-forced NLL per 512-token window on 131K tokens of real text is flat with position (e.g. 1.12 at 0, 0.72 at 32K, 0.59 at 65K, 0.51 at 98K, 0.30 at 130K,
  se 0.04-0.08 each) and the GPU engine agrees with the independent CPU engine (qv_ink_cpu) at 32K and 64K.
* L2 int4 KV: REFUTED for quality. int4 vs fp16 KV NLL differs by <= 0.012 nats in every window up to 49K (within se). Loop rate padcode_64k int4 12.5% (3/24, 4.3-31.0%) vs fp16 8.3% (2/24, 2.3-25.8%): not distinguishable.
* L3 sampler distribution: host Sampler, `build_dist_dense` and the lossless speculative step match the analytic distribution (chi-square, tests/test_sampler_dist.cpp). Merged-tree check: agent_8k with `--mtp`
  on the merged build (adec248 + p28) gives the same token counts and finish kind as the dev build on all 18 seeded generations, loop 1/18 (5.6%, 1.0-25.8%). No sign of a distortion.
* L4 q3a5 vs higher-bit: NOT TESTED (no higher-precision model fits the 15 GB disk).
* L5 unbounded reasoning / sampling: partly supported. Context length, not sampling, is the visible driver: padcode (same prompts) 4.2% at 1K filler (1/24) vs 12.5% at 64K (3/24), intervals overlap, so task
  difficulty versus length is not separated at this n. Sampling variants on agent_8k (n=30 unless noted; base 6.7%, 17 runaway): top_k 20 / top_p 0.9 / min_p 0.1 20.8% (n=24, 9.2-40.5%, worse, not better);
  T=0.7 33.3% (19.2-51.2%, 3 strong; lower temperature loops MORE); `--reasoning-budget 4096` 6.7% loop but runaway drops 17 -> 5 and eos rises 11 -> 23 of 30, median 5547 tokens.

### Conclusion
No engine bug was found behind loops: the numerics, KV format and samplers check out. The behaviour is the documented model property (long reasoning that does not converge), made worse by long
agent contexts and by LOWER temperature or tighter truncation. Mitigation that measurably helps: a reasoning budget (`--reasoning-budget N`, 4096 on these prompts) plus client token and turn limits,
as the model card advises. We did not change any default: the 4096 budget cuts runaway generations by about 70% but also truncates legitimate long reasoning, which is a product decision.
Do not lower the temperature or tighten top_k as a loop fix.

## Related bug found while investigating: slow tool-call turns (fixed)
Symptom: a tool-call turn with `--mtp` at T=0 ran at 3.3 tok/s (32K) and 1.4 tok/s (66K) while the same turn without tools ran at 245-285 tok/s. Cause: (1) the Inkling grammar demanded required argument keys first,
while the chat template writes arguments in `dictsort` (byte) order, so the grammar rejected the model's own `{"path":..,"pattern":..}` call; (2) a reject inside a greedy MTP lookahead triggers a rollback
and Inkling has no `state_save`, so the rollback is a full re-prefill. Fix: arguments are declared in byte order for the Inkling format (`src/grammar/tool_grammar.cpp`), and a lazy-grammar trigger inside a
greedy lookahead whose remaining tokens the grammar accepts drains the lookahead instead of rolling back (`src/server/app_gen.cpp`; `QV_NO_DRAIN=1` restores the old path; `QV_MTP_DEBUG=1` logs
the rejected token). Tool-turn decode on the merged tree, 33K / 66K prompts (tokens/s):

| config | 32K tools | 64K tools |
|---|---:|---:|
| `--mtp` T=0, before (`QV_NO_DRAIN=1`: key order fixed, drain off) | 3.7 | 1.6 |
| `--mtp` T=0, after | 241 | 224 |
| `--mtp --mtp-grammar` T=0 | 206 | 254 |
| `--mtp` T=1.0 | 100 | 94 |
| `--mtp --mtp-grammar` T=1.0 | 242 | 254 |
| no `--mtp` T=1.0 | 99 | 95 |

(Both parts are needed: with drain off, a lazy trigger inside a lookahead always rolls back, even for a valid call; the key-order fix makes the model's own call grammar-valid so the drain applies. Pre-fix tree: 3.3 / 1.4 tok/s.) Prefill time (19-26 s at 33K-66K) is unchanged and is not part of this issue.
