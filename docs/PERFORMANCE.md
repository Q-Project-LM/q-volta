# Q-Volta — Performance report (Tesla V100-SXM2-16GB, sm_70)

Status: Stage A (GPU) in progress. All numbers below were measured on this machine; nothing is projected.
Hardware/software: V100 SXM2 16 GB, SM clock 1447-1530 MHz, CUDA 12.9, driver 580.178.04. Model: Qwen3.8-27B (BF16 rev 37561816), quantized by our converter (E0: RTN g64, MLP 3-bit, GDN/attn 4-bit, embed 4-bit, lm_head 6-bit; 12.63 GiB).
Competitor: NInfer (Apache-2.0), model Qwen3.8-27B-GSQ-RCO-IQ3_S (13.99 GiB file, ~3.5 bpw average, mixed 1..6-bit per tensor). Raw logs: benchmarks/baselines/, logs/.

## Decode (greedy, 128 new tokens, thinking off, 3 repeats, deterministic)
| engine | mode | code tok/s | prose tok/s |
|---|---|---:|---:|
| NInfer | plain | 45.3 | 45.3 |
| NInfer | MTP-3 | 106.5 (282/294 drafts) | 69.0 (228/459) |
| Q-Volta E0 | plain | 52.2 | 52.1 |
| Q-Volta E0 | MTP | see the 2026-11-20 GPU MTP section below | |
Bytes-normalized plain decode: Q-Volta 52.2 x 12.63 = 659 GiB/s effective vs NInfer 45.3 x 13.99 = 634 GiB/s (+4%). Most of the raw +15% comes from reading fewer bytes (coarser quantization), not from a faster kernel. NInfer artifact may contain tensors not read per token, so treat +4% as an upper-bound-ish estimate until verified.

## Prefill (cold, no prefix cache, identical wikitext prompts)
| prompt tokens | Q-Volta (chunk 2048) | Q-Volta (chunk 512) | NInfer (chunk 1024) | Q-Volta chunk-2048 vs NInfer |
|---:|---:|---:|---:|---:|
| 512 | 1013 (one chunk) | 1013 | 837 | +21% |
| 2048 | 1205 | 971 | 1010 | +19% |
| 4088 | 1143 | 917 | 1013 | +13% |
Path: dequantize 32-MiB weight slabs to fp16 -> cuBLAS tensor-core HGEMM; GDN recurrence in-kernel with register-resident state; fp32 CUDA-core flash-style attention. Reps agree within ~1%.
Why chunk size matters: weights are re-dequantized once per chunk, so chunk 512 paid for 8 dequant passes on a 4K prompt. Profile at 4088 tokens (chunk 512): GEMM 61% (~77 TFLOP/s, near the cuBLAS ceiling), dequant 15%, attention 10%, GDN 10%. Chunk 2048 costs +0.6 GiB VRAM (14.9 GiB total at max_ctx 8192).
Multi-chunk correctness: golden passes with chunks of 16 and 23 tokens (GDN/conv/KV carry across chunks), same error level as single-chunk.
Not yet measured: prompts > 4K (attention is CUDA-core fp32 and grows quadratically; expect the lead to shrink), VRAM at long context (fp32 KV).
History: step-by-step 51 -> batched 680 -> aligned slabs 1014 -> chunk 2048 1205 tok/s. (An earlier "25.6 tok/s" figure was a benchmark-timer artifact in --graph mode; fixed.)

## Quality (the co-metric; speed wins at worse quality are not wins)
- wikitext-2 test, 4088 tokens: BF16 ppl 8.468 -> E0 9.508 (+12%), KL(ref||q) 0.122, top-1 agreement 84.6%.
- p10/quant (docs/QUANTIZATION.md): GPTQ g64 +6.1% / KL 0.080 (12.63 GiB), GPTQ g128 +7.3% (11.83), e1 all-3-bit g128 meta8 +9.6% / KL 0.112 at 10.44 GiB.
- NInfer README (different protocol, 2048-token windows): ppl 7.071 vs BF16 7.05 (+0.3%). Only the ratio is comparable: we are far behind on quality.
- KL is heavy-tailed (median 0): a few confident flips on structural tokens (JSON keys) dominate the mean. Report median/p99/flip-rate next.
- Engine correctness: golden vs simulated-quant reference PASS on real weights. Step path (int8 activations): layer rel err 2.0e-2, KL 5.7e-4. Prefill path (fp16 activations): layer rel err 6.3e-4, KL 7.3e-5.

## pi adapter (harness-hub/Qwen3.8-27B-pi-adapted, LoRA r=32 alpha=64)
Merged into BF16 before quantization (256/256 modules). Engine golden PASS. Adapter's logit shift (rms 0.08 wikitext / 0.23 agentic) is 8-18x smaller than 3-bit RTN noise (0.63 / 4.08); cosine between BF16 shift and quantized shift 0.25-0.44 => behaviour is partly blurred. Plan: runtime LoRA over a once-quantized base (L1). Not yet verified on real agent trajectories (the trajectory bucket is not publicly readable).

## Context length
Default engine path: fp32 KV (128 KiB/token). Measured max_ctx that builds and decodes (V100, plain decode, prefill chunk 512): fp32 18,432 (20,480 OOM), fp16 38,912 (40,960 OOM), int4 139,264 (143,360 OOM); see the int4 KV section below. 256K needs 4-bit KV AND weights+scratch <= ~11.3 GiB (now ~13.3 incl. CUDA context): not reachable on this card yet; no claim made beyond the lengths tested in the int4 KV section.

## Known limitations / not done
split-K long-context decode attention is done (see below); int4 KV is done (see the int4 KV section; GPU only, no host tier yet); tensor-core attention, quality improvements (GPTQ/bit allocation), tokenizer/server/CLI, CPU backend (gated until the GPU completion gate).

## Decode vs context (split-K attention, 2026-11-20)
| context at decode start | before | after |
|---:|---:|---:|
| ~128 | 52 tok/s | 52.3 |
| 4088 | 36 | 50.2 |
Kernel-only (per layer, us, old -> new): pos 511 52->19.5; 4095 573->69; 16383 2278->189; 32767 4557->348. Short contexts (pos<=63) are ~4 us slower per layer (second launch). Not yet measured end-to-end at >4K; no NIAH/quality check at long context yet.

## GPU MTP speculative decoding (2026-11-20, branch p8/gpumtp)
Exact greedy speculative decoding with the checkpoint's own MTP head (same semantics as the CPU path, docs/CPU_OPTIMIZATION.md). Output is token-for-token identical to plain greedy (and the final logits are bit-identical, i.e. the rolled-back engine state equals plain decode state): test_cuda_mtp (depth 1, 2, 3; 4 prompts; 97 tokens) and every one of the 60 benchmark runs below (ids compared against the plain run).
Mechanism: verify [cur, d1..dD] in ONE pass over the weights (`gemv_q8_multi`: weight planes unpacked once, per-token arithmetic and warp split identical to the 1-token dp4a kernel, test_gemv_multi checks bit equality incl. both warp configs); rmsnorm / conv / GDN / split-K attention / f16 dual GEMV run all rows in one launch with the exact 1-token per-row arithmetic; GDN state is carried in registers across the verify rows and a snapshot after each row (and the conv state) is written for O(copy) rollback; KV is rolled back by position. Draft: mtp.fc + one attention+MLP layer + shared lm_head, chained for depth > 1 (MTP output hidden fed back). MTP KV for accepted positions is rewritten from the true main hidden after every cycle. `--mtp-depth 1..3`, `EngineOptions::mtp_depth`.
Protocol: qv_run --bench --prefill --graph, 129 tokens (128 timed), max_ctx 4608, V100 (device 0) otherwise idle, 3 interleaved runs per cell (spread = min..max), plain row re-measured with the same build. Chat prompts code/prose are the NInfer-protocol prompts (37/47 tokens); 512/2048/4088 are wikitext prefixes (real text, base-model continuation, not chat).
| prompt (ctx) | plain tok/s | MTP depth 1 | depth 2 | depth 3 | draft acceptance d1 / d2 / d3 | ids identical |
|---|---:|---:|---:|---:|---|---|
| code, chat (37) | 52.22 [52.22-52.23] | 87.16 [86.94-87.27] (1.67x) | 116.68 [116.63-116.76] (2.23x) | 129.46 [129.44-129.49] (2.48x) | 1.000 / 0.955 / 0.912 | yes |
| prose, chat (47) | 52.22 [52.21-52.22] | 79.63 [79.62-79.66] (1.52x) | 83.94 [83.93-83.96] (1.61x) | 84.33 [84.25-84.37] (1.61x) | 0.829 / 0.558 / 0.472 | yes |
| wikitext (512) | 52.05 [52.04-52.05] | 85.75 [85.67-85.82] (1.65x) | 115.24 [115.15-115.35] (2.21x) | 134.48 [134.37-134.57] (2.58x) | 0.984 / 0.955 / 0.960 | yes |
| wikitext (2048) | 50.78 [50.76-50.80] | 69.93 [69.83-70.05] (1.38x) | 76.69 [76.42-76.86] (1.51x) | 72.61 [72.59-72.64] (1.43x) | 0.693 / 0.524 / 0.409 | yes |
| wikitext (4088) | 48.58 [47.68-49.98] | 69.77 [68.84-70.83] (1.44x) | 82.35 [81.61-83.73] (1.70x) | 79.90 [79.16-80.95] (1.64x) | 0.753 / 0.670 / 0.549 | yes |
(acceptance = accepted drafts / drafted tokens. Runs are deterministic, hence the tiny spread except at 4088 where plain itself varies 47.7-50.0.)
Versus NInfer MTP-3 on the same chat prompts and V100 (NInfer numbers are from the earlier run in this document, NOT re-run today; NInfer was measured through its server timings, ours with qv_run's decode window, so the measurement paths differ): code 129.5 vs 106.5 tok/s (+22%), prose 84.3 vs 69.0 (+22%; depth 2 gives 83.9). Plain: ours 52.2 vs 45.3. Caveats: NInfer's IQ3_S file is 13.99 GiB with ppl +0.3% vs our E0 12.63 GiB at +12% ppl, so our speed is partly bought with lower quality; MTP does not change quality (output equals our plain greedy).
Cycle breakdown (QV_MTP_PROF=1, one run each, ms per cycle): verify 21.3 (depth 1, ctx 39) / 21.8 (depth 2) / 23.9 (depth 3) versus a plain step of 19.1 ms: verify costs 1.11x / 1.14x / 1.25x of a 1-token step for 2 / 3 / 4 tokens (hypothesis was ~1.1x for 2 tokens: confirmed). Draft (+MTP KV update) costs 1.6 / 3.2 / 4.7 ms per cycle, about 1.6 ms per chained step, 1.2 ms of it the 6-bit lm_head (0.95 GB) over the full vocabulary. At ctx 4088: verify 23.0 / 24.3 / 27.3 ms, plain step 20.6 ms. Tokens per cycle = 1 + accepted drafts per cycle.
Depth 1 hypothesis (acceptance ~0.8 -> ~1.5x, 52 -> ~75 tok/s): prose 1.52x (79.6) and code 1.67x (87.2) at depth 1, confirmed; it does not beat NInfer MTP-3 by itself, depth 3 does on code.
Which depth: it depends on acceptance. Code / predictable text: depth 3 (2.5x). Chat prose: depth 2 and 3 are equal (84 tok/s) because the third draft is accepted only 47% of the time. At 2048-4088 tokens of non-chat wikitext depth 2 is best and depth 3 is worse than depth 2. No adaptive depth policy is implemented (a fixed depth is chosen at engine construction); the numbers above show a per-workload choice is worth up to ~13% over a fixed depth 3.
Graphs: the verify pass and draft steps are CUDA-graph captured, but the decode is GPU-bound: graph on/off gives +0.3% (code d3 129.5 vs 129.2, prose d1 79.65 vs 79.37). The big win over the first working version (verify 22.8 ms -> 21.3 ms at depth 1, 28.7 -> 23.9 ms at depth 3) came from launching all verify rows in one kernel; the remaining gap to a plain step is the multi-token GEMV (~9% slower per call than the single-token kernel for 3-bit) and per-row argmax.
VRAM at max_ctx 4608: plain 14.51 GiB, MTP depth 1/2/3: 14.74 / 14.89 / 15.04 GiB (of 15.77): GDN+conv snapshots cost 0.15 GiB per depth step, plus verify scratch and the MTP KV slot (fp32 KV: 128 KiB/token for the 16 main attention layers, +8 KiB/token for the MTP layer). Depth 3 leaves ~0.73 GiB at 4608, i.e. by arithmetic roughly 5K more tokens of KV (max_ctx ~10K, estimate, not tested) until KV is quantized.
Prefill cost of MTP: the MTP layer also processes the prompt: prefill tok/s 1011 -> ~995 (512), 1208 -> 1187 (2048), 1131 -> 1108 (4088) = 1.5-2.0% slower.
Not done / limits: lm_head restricted to a candidate set for the draft (would cut ~1.2 ms per draft step), adaptive depth, depth > 3 (the multi-token GEMV is compiled for 2, 3, 4, 8 rows), sampling (temperature) speculation (only exact greedy is implemented), MTP with the pi LoRA merged model (acceptance with an unadapted head not measured), server integration (the GPU server path does not call decode_greedy yet), contexts > ~8K (VRAM).
Tests (this machine, 2026-11-20): ctest all pass (gemv, gemv_multi, dequant, attn_seq, gdn_seq, attn_decode_split, cuda_mtp and the CPU tests); model golden (tests/check_golden.py, 64 tokens) PASS: worst layer rel 0.033, final logits rel 0.012, KL(qsim||engine) 6.5e-4, top1 agree vs qsim 0.984. Raw data: benchmarks/results/gpu_2026-11-20.json.
Plain-decode baseline re-measured today before touching the code (build of 1946b36, graph decode, 3 runs, tok/s): ctx 37: 52.30-52.33, ctx 47: 52.28-52.32, 512: 52.10-52.14, 2048: 50.99-51.02, 4088: 48.80-50.19; prefill 512/2048/4088: 1010 / 1207 / 1136 tok/s (mean of 3). The plain rows in the table are from the MTP build and agree (the GDN kernel now has a token loop; plain arithmetic is unchanged). See the json for every run.

## int4 KV cache (2026-11-20, branch p9/kv4; hypothesis H-A of PLAN.md Gen 6)
`--kv fp32|fp16|int4` (qv_run and qv_server; `EngineOptions::kv_mode` 0/1/2, `kv4_kmode`). fp32 is the unchanged default path. fp16 = the same code path as int4 with a full-length fp16 ring and no quantization (64 KiB/token; used as the exact-ish baseline where fp32 does not fit). Design and byte layout: src/backends/cuda/attention/kv4.h.
Design: KV of the 16 attention layers (4 kv heads x head_dim 256) is stored in self-contained 64-token block records of 18,432 B (K int4 + K scale/zero + V int4 + V scale/zero; scale/zero fp16 half2, asymmetric min/max, 16 levels) = 4.5 bit/element = 288 B/token/head = 18 KiB/token for all 16 layers (fp32: 128 KiB, fp16: 64 KiB). The most recent 128..191 tokens stay un-quantized in an fp16 ring (256 slots per head); block b is quantized when position 64(b+3)-1 is written, so every path (prefill, 1-token decode, MTP verify) applies the same rule "keys [0, 64*floor((n-128)/64)) quantized, the rest fp16" and sees the same bits. K quantization has two modes (`--kv4-kmode channel|token`): per-channel (KIVI style: scale/zero per channel over the 64 tokens of a block) and per-token (groups of 64 channels per token); V is always per-token groups of 64 channels. All block addressing goes through `kv4_rec()` and a record is a contiguous 18 KiB unit, so a later host-RAM tier can move cold blocks without touching the kernels. Kernels: kv4.cu (block quantize, ring commit), attn_kv4.cu (split-K decode: dequantizes in registers while streaming, same online-softmax/combine structure as the fp32 kernel, splits own whole 64-key blocks), attn_seq.cu (`attn_seq_kv4_kernel`, prefill: dequantizes tiles into shared memory, handles windows that straddle a tile with a per-row mask); the prefill chunk's post-RoPE K/V are staged in fp16 (2 MiB each at chunk 1024) and quantized after the chunk's prep.
Correctness: test_attn_kv4 (new, ctest) - int4 prefill kernel vs fp32 path (random data with outlier K channels: rel L2 0.2e-3..2.5e-2 per-channel, up to 3.7e-2 per-token; fp16 mode 1e-4), decode kernel vs prefill kernel on the same int4 cache 3e-7..1e-6, multi-chunk prefill (chunks of 193 / 64 / rest) identical to single chunk (0.0), nt=3 MTP-verify launch bit-identical to three 1-row launches. test_cuda_mtp with int4 KV (depth 1, 3; contexts 14, 37, 120 and 4088 tokens, 96 generated tokens each, crossing block-quantization boundaries): ids AND final logits identical to plain greedy on the same int4 cache in every case; also per-token-K mode (depth 3). The fp32 path is unchanged (ctest attn_seq, attn_decode_split, gdn_seq, dequant, gemv_multi pass).
### Quality (wikitext-2 test text, first 4088 / 8192 real tokens, prefill chunk 2048, E0 model; reference = same engine with fp32 KV)
| context | KV | ppl | ppl vs fp32 KV | KL(fp32-KV || KV) over top-1024 + tail | top-1 agree |
|---|---|---:|---:|---:|---:|
| 4088 | fp32 | 6.2567 | - | - | - |
| 4088 | fp16 | 6.2569 | +0.003% | 8e-6 | 99.88% |
| 4088 | int4, K per-channel | 6.2537 | -0.048% | 8.3e-4 | 98.65% |
| 4088 | int4, K per-token | 6.2605 | +0.060% | 1.22e-3 | 98.48% |
| 8192 | fp32 | 6.8793 | - | - | - |
| 8192 | fp16 | 6.8792 | -0.002% | 8e-6 | 99.82% |
| 8192 | int4, K per-channel | 6.8807 | +0.020% | 7.9e-4 | 98.75% |
| 8192 | int4, K per-token | 6.8853 | +0.087% | 1.31e-3 | 98.39% |
(Single run each, deterministic. KL: mean over positions of KL(P_ref||P_kv) computed on the reference's top-1024 tokens plus one lumped tail bucket (mean reference tail mass 0.8%), so it is a close approximation, not the full-vocabulary KL. ppl differences of +-0.05% are within what a different but equally valid quantization noise realisation would give; the robust signal is KL ~8e-4 (100x the fp16 floor) and 1.2-1.5% top-1 flips. Per-channel K is better than per-token K on both metrics (KL 8e-4 vs 1.2-1.3e-3) and is the default.) The ppl of 6.26/6.88 is for these text prefixes and is not comparable to the 9.5 of the 4088-token eval above (different text window; this eval is also teacher-forced with the KV of the *quantized-weights* model on both sides, so it isolates the KV effect only). H-A falsification threshold (ppl +>3%): not met (max +0.09%).
### Passkey retrieval (tools/niah.py; real token lengths via qv_tok; chat prompt, thinking off, greedy 16 tokens; 5-digit key hidden in wikitext filler at depth 0.1/0.5/0.9, 2 random keys each = 6 prompts per length)
| length (tokens) | fp32 KV | fp16 KV | int4 KV |
|---|---:|---:|---:|
| 4,096 | 6/6 | 6/6 | 6/6 |
| 8,192 | 6/6 | 6/6 | 6/6 |
| 16,384 | 6/6 | 6/6 | 6/6 |
| 32,768 | does not fit | 6/6 | 6/6 |
Harder variant (tools/niah.py --multi 6: six "The secret code for NAME is NNNNN." needles spread over the document, ask for one of them, so the answer must be bound to the right name; 6 prompts per length, the asked needle rotates through the document): int4 6/6 at 8K, 16K and 32K; fp16 6/6 at 8K, 16K and 32K (fp32 not run on this variant).
Caveats: single-passkey retrieval is an easy task (saturated at 100% for all three KV formats, so it can only detect a gross failure, not mild degradation); an earlier version of the generator that asked for a completion without the chat template produced <think> openings that were scored as misses for fp32 as well as int4 (format artefact, replaced by the chat prompt above). Only lengths <= 32K are tested; nothing is claimed above that. H-A falsification (NIAH drops at <= 32K): not met.
### Speed (qv_run --decode-bench: prefill the real wikitext prefix, untimed 1-token warm-up that captures the CUDA graph, then 129 timed graph-decode tokens; V100 device 0, `flock` + idle check per run, prefill chunk 1024)
| context at decode start | fp32 KV tok/s | fp16 KV tok/s | int4 KV tok/s | int4 vs fp32 | VRAM fp32 / fp16 / int4 (GiB, nvidia-smi incl. CUDA context) |
|---:|---:|---:|---:|---:|---|
| 512 | 52.03 [51.89-52.10] | 51.88 [51.87-51.88] | 51.70 [51.69-51.71] | -0.6% | 13.63 / 13.59 / 13.57 |
| 4,096 | 50.01 [50.00-50.01] | 50.36 [50.34-50.37] | 50.50 [50.37-50.59] | +1.0% | 14.06 / 13.80 / 13.63 |
| 8,192 | 46.81 [45.85-48.17] | 47.46 [45.77-49.02] | 47.87 [46.14-49.26] | +2.3% | 14.56 / 14.05 / 13.70 |
| 16,384 | 39.66 [36.45-42.87] | 42.63 [39.35-45.91] | 42.54 [38.41-46.67] | +7% | 15.56 / 14.55 / 13.84 |
| 32,768 | does not fit | 35.98 [33.56-38.41] | 38.35 [36.63-40.07] | (+6.6% vs fp16) | - / 15.55 / 14.12 |
(mean [min-max] of 3 runs at <=8K and 2 runs at 16K/32K; the GPU/CPU were shared with other jobs (all timed runs held the lock, but non-locking load exists) and the spread at >=8K is mostly drift between rounds (all three formats drop together), not between formats: in all 10 rounds at >=4K int4 was >= fp32, and int4 >= fp16 in 9 of 10 (not at 16K round 2: 38.41 vs 39.35). Differences of 1-3% are weak evidence; the robust statement is "int4 is not slower than fp32/fp16 at any context (512: -0.6%) and uses 0.5-1.4 GiB less VRAM at 16K".) Why the speed-up is small although int4 reads 3.6-7x fewer bytes: the split-K attention kernel appears to be instruction/latency bound rather than bandwidth bound. This is inferred, not profiled: the end-to-end cost of 32K context over 512 context is ~5.6 ms/token for int4 (24.96 vs 19.34 ms), i.e. ~350 us per attention layer, the same as the 348 us measured for the fp32 kernel at 32K (test_attn_decode_split), while the int4 data of one layer at 32K (37.7 MB) would take ~47 us at 800 GB/s. A kernel with a cheaper per-key reduction (the 6-head shuffle reduce) could therefore turn the byte saving into speed; not done.
Prefill: 8K prefill 9.3 s int4 vs 9.0 s fp32 (+3%, single runs, chunk 2048); in the NIAH runs (chunk 1024, mean of 6) 4K 4.5 / 4.2 / 4.8 s, 8K 10.4 / 10.0 / 10.8, 16K 25.3 / 24.8 / 25.8, 32K 67.1 / 67.6 / - s for int4 / fp16 / fp32 (i.e. 32K prefill ~490 tok/s; the attention share is not profiled here, CUDA-core attention is the suspected cause and tensor-core attention is hypothesis H-D).
MTP depth 3 with int4 KV (wikitext prefixes, base-model continuation, 129 tokens, 1 run each): 512: 133.1 tok/s (acceptance 0.962) vs fp32 129.2 (0.926); 4,096: 89.8 (0.608) vs 85.1 (0.589); 16,384: 71.4 (0.563); 32,768: 44.6 (0.375). Output ids identical to the plain int4 decode of the same prompt in all four cases (compared over the 122 tokens both runs cover); fp32 MTP at 16K does not fit (snapshots + KV). Acceptance differs slightly between KV formats because the model's own outputs differ slightly.
### Max context and VRAM
Measured (qv_run --decode-bench --synth-ctx: engine built with max_ctx N, 512-token prefill at chunk 512, then 33 graph-decode tokens at position ~N-300 over UNWRITTEN KV, so speed/VRAM are measured but the content is not meaningful; plain decode, no MTP):
| KV | largest max_ctx that fits | first size that OOMs | decode tok/s at that position | nvidia-smi used |
|---|---:|---:|---:|---:|
| fp32 | 18,432 | 20,480 | 44.85 | 15.58 GiB |
| fp16 | 38,912 | 40,960 | 41.94 | 15.71 GiB |
| int4 | 139,264 | 143,360 (147,456 and 155,648 also) | 30.28 | 15.74 GiB |
(int4 also: 65,536 -> 38.9 tok/s, 98,304 -> 34.4, 131,072 -> 31.2; sizes tested in 2K-8K steps, so the true limit lies inside the gap; MTP needs 0.15 GiB more per depth step.) Quality was verified only up to 32K (above); lengths 33K-139K fit but are NOT validated. The int4 limit is 7.6x the fp32 one and 3.6x the fp16 one (expected ~100K+; measured 139K).
256K requirement (arithmetic, not a measurement): int4 KV = 18,432 B/token x 262,144 = 4.5 GiB (+0.28 GiB for the MTP layer's KV), fp16 KV would be 16 GiB. The V100 exposes 15.77 GiB; everything except KV currently takes ~13.35 GiB (weights 12.63 + scratch/prefill buffers + CUDA context, taken from the 65,536-token probe: 14.48 used - 1.13 KV), so 256K needs ~17.85 GiB: ~2.1 GiB too much. Options: weights+scratch <= ~11.3 GiB (about -2.1 GiB of weights), or keep cold int4 blocks in host RAM (record layout is ready for that).
Reproduce: `qv_run --model M --max-ctx N --eval wiki.i32 --eval-mode write|cmp --eval-ref ref.bin --kv fp32|fp16|int4 [--kv4-kmode channel|token]` (ppl/KL), `python3 -I tools/niah.py gen ...` + `qv_run --gen --prompt a.i32,b.i32 --n 16 --kv ...` + `niah.py score` (passkey), `qv_run --decode-bench --prompt P.i32 [--synth-ctx N] [--mtp --mtp-depth 3] --kv ...` (speed / VRAM / max ctx), `QV_TEST_KV=1 QV_TEST_MAXCTX=4608 test_cuda_mtp MODEL 96 1,3 prompts...` (MTP exactness on the int4 cache). Raw data: benchmarks/results/gpu_kv4_2026-11-20.json.
### int4 KV on the tensor-core prefill attention (merge of p9/kv4 with p9/tcattn, 2026-11-20)
`attn_seq_tc` now reads int4 KV through the same fetch/store source (`KvSrcF32` got an int4 mode: dequantize from block records or read the fp16 window = chunk staging for key >= pos0, ring otherwise). The window rule is the same as for decode; a 64-key tile that some rows read quantized and others from the fp16 window is processed as two consecutive steps (quantized first) with the non-owning rows masked, so the online softmax sees an empty tile for them. The ring commit runs after the kernel. fp32/fp16/int4 all work with `QV_ATTN=fp32` (CUDA-core kernels) and with the default tensor-core kernel.
Re-verified after the merge: ctest 13/13 (incl. attn_seq, cuda_mtp on fp32 KV), model golden PASS, test_attn_kv4 PASS with both prefill kernels (tensor-core: decode-kernel vs prefill-kernel rel L2 5e-5..1.5e-4 = fp16-P rounding, multi-chunk identical to single chunk; QV_ATTN=fp32: 1e-6 as before), test_cuda_mtp with int4 KV (depth 1 and 3, K per-channel; depth 3, K per-token; prompts 14/37/120/4088 tokens): ids and final logits identical to plain greedy in all cases. wikitext 4088 ppl, int4 vs fp32 KV (both with the tensor-core kernel): 6.2538 vs 6.2566 (-0.05%), KL 8.2e-4, top-1 98.85% (unchanged from the CUDA-core kernel numbers above).
Prefill with int4 KV (qv_run --decode-bench, prefill chunk 2048, one run each, shared GPU): 4,096 tok 3.62 s = 1,131 tok/s; 8,192 tok 7.82 s = 1,048 tok/s (CUDA-core attention, same KV: 8.69 s = 943 tok/s; fp16 KV with tensor cores 1,077 tok/s; fp32 KV with tensor cores 1,148 tok/s, i.e. the int4 fetch costs ~9% of prefill at 8K vs fp32); 16,384 tok 18.2 s = 902 tok/s; 32,768 tok 47.8 s = 686 tok/s (was ~488 tok/s with the CUDA-core kernel in the NIAH runs above, chunk 1024). Decode after the 32K prefill: 40.8 tok/s (unchanged within noise: 38.4 [36.6-40.1] before; 8K 49.3).
Long-context end-to-end sanity check, int4 KV, passkey in wikitext filler (chat prompt, greedy): 65,536 tokens: prefill 145.7 s / 147.6 s (about 450 tok/s), key retrieved 2/2 (depth 0.5 and 0.9); 131,072 tokens (max_ctx 131,136, prefill chunk 512): prefill 529.8 s / 528.1 s (about 248 tok/s), key retrieved 2/2. Prefill time at 128K is therefore ~9 minutes (the O(T^2) attention dominates); decode at ~139K position was measured at 30.3 tok/s earlier on synthetic KV. Only 2 prompts per length (single passkey task), so this shows the pipeline works end to end at 64K/128K, not a quality bound.

## Tensor-core prefill attention (H-D, p9/tcattn, 2026-11-20)
Kernel: `src/backends/cuda/attention/attn_seq_tc.cu` (WMMA 16x16x16 = Volta HMMA, fp16 inputs, fp32 accumulate and fp32 online softmax). One block = 8 queries x the 6 q heads of one kv head (48 rows) x 64-key tiles, K/V tile shared by all 6 heads and converted fp32 -> fp16 in smem (global loads register-prefetched one phase ahead), causal mask + `pos0` offset, gate/sigmoid epilogue and fp16 clamp unchanged. KV reads sit behind a `KvSrcF32` fetch/store pair so an int4-block source can replace it. `attn_seq()` dispatches to it; `QV_ATTN=fp32` selects the old CUDA-core kernel. No split over keys: with chunk 2048 the grid is 256 x 4 blocks, all equally long, enough for 80 SMs.
Attention-only, one layer, chunk T=2048, synthetic data (`tests/bench_attn_seq.cu`; TFLOP/s counts causal-useful FLOPs):
| pos0 (keys before chunk) | fp32 ms | TC ms | speedup | TC TFLOP/s |
|---:|---:|---:|---:|---:|
| 0 | 5.83 | 2.06 | 2.8x | 25.1 |
| 6144 | 40.2 | 12.4 | 3.2x | 29.0 |
| 14336 | 87.4 | 26.2 | 3.3x | 29.5 |
| 30720 | 189 | 54.1 | 3.5x | 29.5 |
| 63488 | 431 | 110 | 3.9x | 29.5 |
| 129024 | 834 | 240 | 3.5x | 27.3 |
| 260096 | 1676 | 498 | 3.4x | 26.4 |
Prefill attention for a whole prompt (16 layers, chunk 2048; per-chunk time interpolated between the measured points above, so extrapolated for 32K+ prompts, attention only): 16K 1.8 s (fp32 5.9 s), 32K 7.2 s (24.4 s), 64K 28.6 s (105.7 s), 128K 119 s (433 s), 256K 499 s = 8.3 min (fp32 1725 s = 28.7 min). That is 3.5x, not the ~5x/4 min the plan expected: the kernel reaches 27-29.5 TFLOP/s, about 23% of the 125 TFLOP/s HMMA peak.
End-to-end prefill tok/s (E0 model, wikitext prefix, 2 reps, GPU/CPU shared with other agents so +-3% noise): 4096 (chunk 2048) fp32 1101/1075 -> TC 1191/1171 (+9%); 8192 (chunk 2048) 909/913 -> 1040/1088 (+15%); 16384 (chunk 1024) 692/684 -> 888/870 (+28%). Chunk 2048 OOMs at 16K (fp32 KV 2.1 GiB + scratch) and 32K does not fit (fp32 KV 4 GiB) until int4 KV lands, so 32K end-to-end is NOT measured.
Correctness: `test_attn_seq` TC vs golden decode loop rel err 2.1e-4..4.1e-4 (fp16 out 2.9e-4..4.6e-4; tolerance 3e-3), TC vs fp32 kernel 4.2e-4 at pos0 63488 (synthetic). Full-model golden: step path KL 6.5e-4 (unchanged); prefill path KL 7.29e-5 (fp32 kernel 7.27e-5), chunk-23 prefill 7.46e-5 (fp32 7.23e-5), top-1 vs qsim 0.984 / 0.952.
What limits it (measured): removing the global K/V loads from the kernel (diagnostic build, zeros instead) gave 40 TFLOP/s vs 27, so ~30% is the fp32 KV read + convert through L2 (an fp16/int4 cache cuts this); the rest is the serial S -> softmax -> PV phases with one 8-warp block per SM (93 KB smem) and smem-bandwidth-bound WMMA fragment loads. Tried and slower: 12 warps one-tile-per-warp S (11.1 ms vs 7.95 at pos0 8192, T 1024), 96-row tile with 32-key tiles (8.7 ms, register spills). Not tried: raw mma.sync with register-resident S/P (FlashAttention-2 style) which should avoid the smem round trip.
Raw data: benchmarks/results/gpu_tcattn_2026-11-20.json.

## Decode regression after the p9/p10 merges, and the fix (p11/perf, 2026-11-21)
Honest note: after merging p9/tcattn, p9/kv4 and p10/quant onto the GPU-MTP commit 324ea1e, E0 decode on the same V100 got slower. Re-measured back to back in separate build dirs, 3 interleaved runs, `benchmarks/prompts/code.i32`, `--max-ctx 4608 --bench --prefill --graph --n 129`: 324ea1e plain 52.24/52.20/52.24, MTP-3 129.45/129.65/129.59; 9531c85 plain 48.66/48.67/48.67 (-6.8%), MTP-3 103.79/104.35/104.42 (-19.5%). Machine drift ruled out.
Bisect (E0, MTP-3, 3 runs): 2786d85 (tensor-core attention) 131.3-132.0, bf2ce47 (int4 KV merge) 130.9-131.6 (both slightly faster than 324ea1e), 28dcefc (p10/quant) 104.4-104.6. So the whole regression comes from p10/quant; attention, int4 KV and CMake flags are innocent (the int4-KV refactor costs nothing with the default fp32 KV).
Cause: p10/quant generalized `load_sm()` (runtime branch on `W.sm8`, runtime group shift `W.gshift`) inside the hot `gemv_q_kernel`, `gemv_q8_kernel` and `gemv_q8_multi_kernel`; with the extra QMat fields and the non-constant shift/branch the compiler no longer scheduled the metadata loads the way it did for the fixed group-64 layout. (No per-kernel before/after timing was taken; the cause was established by bisect and by the end-to-end recovery.)
Fix: the three kernels are templated on `MODE` (bit0 = group 128, bit1 = meta8) and the host dispatches on `qmode(W)`; MODE 0 (group 64, fp16 pairs) is the pre-p10 code. For meta8 the per-row factor `rs/255` is computed once per row outside the chunk loop (bit-identical value; it was a full fp32 division per 32-weight chunk). Arithmetic is unchanged: ids identical before/after on E0 and E1 (code, prose, plain), ctest 13/13 including gemv, gemv_multi (g64/g128/meta8 cases), attn_kv4, and test_cuda_mtp (depth 1,2,3, ids and final logits identical) on both E0 and E1.
After (E0, code): plain 52.20/52.18/52.15, MTP-3 131.76/131.62/130.93 (back to and slightly above the 324ea1e report). E1 (10.44 GiB, g128+meta8 GPTQ), code: plain 57.54/57.54/57.60 (was 55.4), MTP-3 137.9-138.5 (was 114.7).
Per byte: E1 is faster in tok/s (+10% plain vs E0) but reads fewer bytes, so its effective rate is lower: 57.55 tok/s x 11.21e9 B (file) ~ 645 GB/s vs E0 52.2 x 13.56e9 ~ 708 GB/s (file sizes, upper-bound proxy for bytes read per token). The meta8 decode ALU and 128-group layout cost ~9% of bandwidth efficiency; not further optimized.
E1 matrix (3 runs, [min-max] tok/s; all ids identical to plain; accept rate in parentheses):
| prompt | plain | MTP-1 | MTP-2 | MTP-3 |
|---|---:|---:|---:|---:|
| code | 57.6 [57.54-57.60] | 96.5 [96.41-96.58] (1.00) | 127.5 [127.22-127.92] (0.977) | 138.2 [137.87-138.50] (0.94) |
| prose | 57.6 [57.56-57.57] | 84.2 [84.16-84.35] (0.753) | 95.6 [95.53-95.68] (0.609) | 88.2 [88.09-88.35] (0.478) |
| wikitext 512 | 57.3 [57.25-57.28] | 92.9 [92.74-92.98] (0.939) | 122.1 [121.82-122.23] (0.922) | 132.2 [132.10-132.22] (0.894) |
| wikitext 2048 | 55.7 [55.69-55.74] | 76.8 [76.70-76.84] (0.684) | 83.6 [83.38-83.77] (0.537) | 76.4 [76.26-76.54] (0.420) |
| wikitext 4088 | 54.6 [54.18-54.79] | 78.1 [77.51-79.04] (0.789) | 94.1 [92.93-96.21] (0.731) | 81.4 [80.16-83.31] (0.527) |
Note: during this session another job intermittently used the GPU/lock; runs were serialized with flock and retried when the GPU was occupied, and the spreads above are tight. Raw data: benchmarks/results/gpu_perf_2026-11-21.json.

## CPU Inkling-2-Mini (Q3A5, p14/ink-cpu, 2026-11-21)
Details and correctness: docs/INKLING_CPU.md; raw: benchmarks/results/inkling_cpu_20261121.json. 32 pinned threads, AVX2, one session on a shared box.
| ctx | decode tok/s | GB/s (2.592 GB/token) | of 54 GB/s | prefill tok/s (chunk 256) |
|---:|---:|---:|---:|---:|
| 128 | 11.3-11.7 | 29-30 | 54-56 % | 69-70 |
| 512 | 11.2 | 29 | 54 % | 77-83 |
| 2048 | 11.0 | 28.4 | 53 % | 66-68 |
Same-session `bench_cpu_gemv` streamed 36-41 GB/s (not 54), so against the measured ceiling the engine is at ~75 %. Design estimate 17-20 tok/s NOT reached.
MoE decode region 39 GB/s (at the box ceiling); the rest is dense GEMV, lm_head, loop gates and ~100 barriers/token. Prefill is compute bound at ~0.5 TMAC/s;
chunk size (128-1024) and register tile (4x2, 2x4, 4x4, 3x3) change it by < 10 %. Quality: val NLL 1.465-1.466 vs reference 1.4643. No llama.cpp comparison.
## Inkling-2-Mini CUDA engine (p14/ink-gpu, 2026-11-21)
Q3A5 on one V100 (CUDA_VISIBLE_DEVICES=0). Details: docs/INKLING_ENGINE.md; raw: benchmarks/results/inkling_gpu_20261121.json.
Correctness: greedy 32/32 vs the Q3A5 reference goldens on all four prompts (step decode, batched prefill + decode, int4 KV on p4); vs the fp goldens 32/32 on p1-p3 and a
4-token prefix on p4 (Q3A5 quantisation drift, same as the CPU engine). Last-position KL to the quantised reference 1e-12..6e-9; vs fp golden top-20 mean abs error 0.77-1.87
(0.19-0.52 vs the quantised reference).
| ctx | KV | decode tok/s |
|---|---|---|
| 25 | fp16 | 122.4 |
| 4K (real, after prefill) | fp16 | 113.8 |
| 4K (synthetic KV) | int4 / fp16 | 113.5 / 112.9 |
| 32K (synthetic) | int4 / fp16 | 101.1 / 97.3 |
| 128K (synthetic) | int4 / fp16 | 75.9 / 68.1 |
| 256K (synthetic) | int4 | 57.4 |
Prefill (4K prompt): chunk 2048 2118 tok/s, chunk 4096 2764 tok/s (target 2500 met at chunk 4096). 8K 2005 tok/s, 32K 1830, 128K 1164, 256K 766 (int4, chunk 2048, includes quadratic attention; 256K = 334 s).
Passkey (chat template via HF tokenizers, key at 50 % depth, greedy 200 tokens): found at 8K, 32K, 128K and 256K with int4 KV; also 32K fp16.
Decode is ~33 % of the HBM bound: latency-bound small kernels (see INKLING_ENGINE.md).

### Inkling decode latency work (p26/ink-fuse, 2026-11-24)
Same-session A/B (baseline f0645bd vs p26/ink-fuse, int4 KV, `qv_ink --bench --n 256 --runs 3`, under `flock /tmp/qv_bench.lock`, spread < 0.1 %):
| ctx | before | after |
|---|---:|---:|
| 25 | 118.65 | 135.4 |
| 4K (real) | 111.4 | 126.6 |
| 32K (synthetic KV) | 99.75 | 111.6 |
MTP-3 (qv_ink --mtp-depth 3, 256 tokens): code 135.3 -> 137.7, prose 94.3 -> 96.4 tok/s (ids identical to plain). Per-step history (ctx 25, tok/s): fold finish 120.4; unroll lane loops 123.0;
attn_res load hoisting 124.3; 1 row per warp GEMV 127.7; gate_down blocks 128.3; combine in smem 129.5; router rank top-8 130.3; reduction barriers 130.6; prep one reduction 131.5; k_norm 134.1; gate loads 135.4.
All steps bit-exact (per-step logits hash + ids vs the old binary; test_ink_mtp, test_multiseq pass). Details and negative results: docs/INKLING_ENGINE.md "Decode latency work". "Synthetic" = decode over unwritten KV at that position (speed only).

## Comparison with llama.cpp on the same V100 (p18/compare, 2026-11-21..23)
Protocol, commands, caveats and the full table: benchmarks/compare/README.md; data: benchmarks/compare/results.json and raw/. llama.cpp 9d6f04f CUDA sm_70, `-fa 1 -ngl 99`, 3 interleaved rounds under /tmp/qv_bench.lock, SM clock 1530 MHz. Quality reference: llama.cpp Q8_0 GGUF (proxy: ppl 8.467 vs BF16 8.468 on 8x512), same 4088-token wikitext window for ppl/KL on both engines.
| model | GiB | ppl 8x512 | ppl 4088 | KL to Q8_0 | tg128 / decode tok/s | pp2048 | max ctx |
|---|---:|---:|---:|---:|---:|---:|---|
| llama.cpp Q3_K_M | 12.77 | 8.905 | 6.047 | 0.0535 | 42.3 | 931 | 39.9K f16 / 74.8K q8_0 / 132.1K q4_0 |
| Q-Volta e0 | 12.63 | 9.511 | 6.257 | 0.1018 | 52.2 | 1250 | 38.9K fp16 / 140.3K int4 |
| llama.cpp IQ3_XXS + imatrix | 10.89 | 8.992 | 6.102 | 0.0751 | 38.7 | 779 | 70.7K / 129.0K / 219.1K |
| Q-Volta e1 | 10.44 | 9.278 | 6.281 | 0.0923 | 57.6 | 1250 | 74.8K fp16 / 267.3K int4 |
MTP (llama.cpp `--spec-type draft-mtp`, supported for qwen35): code 74.8 (Q3_K_M) / 68.8 (IQ3_XXS) vs Q-Volta 131.7 (e0) / 135.9 (e1); prose 56.9 / 52.6 vs 86.5 / 93.8. Verdict: Q-Volta is faster (decode +23%/+49%, prefill +34%/+57% at 2048) and fits more int4 KV; llama.cpp is better in quality at equal bytes. llama.cpp's MTP output equality with plain greedy was not checked.

## Inkling vision (2026-11-23, p19/ink-vision)
Encoder latency (median of 5, V100 device 0 / E5-2683 v4 with 32 OpenMP threads on a loaded shared host; `qv_vision_run --synthetic WxH --bench N`, includes H2D, kernels, D2H, excludes decode/preprocess):

| image | tokens | CUDA fp16 | CPU fp32 |
|---|---:|---:|---:|
| 224x224 | 36 | 5.5 ms | 143 ms |
| 512x512 | 169 | 22.9 ms | 749 ms |
| 1024x1024 | 676 | 186 ms | 7.77 s |
| 2048x2048 | 2704 | 2.14 s | not run |

Attention is quadratic and not tuned (two-pass WMMA); the CPU attention is a simple blocked dot-product kernel (the 512 -> 1024 jump is attention). Preprocess (CPU, single thread): 0.7 / 3.6 / 14.8 ms for 224 / 512 / 1024.
Through `qv_server` (CUDA, Q3A5, int4 KV, `cache_prompt` off, max_tokens 1, 3 requests each, includes PNG decode + encode + prefill): TTFT 0.22 s for 224x224 (59 prompt tokens), 0.32 s for 512x512 (192), 0.71 s for 1024x1024 (699); text-only 18-token prompt 0.15 s. Decode with image context 119.5-119.9 tok/s (text-only 120). Parity numbers: docs/INKLING_VISION.md section 9; raw data `benchmarks/results/inkling_vision_20261123.json`.
Sanity answers (temperature 0, Q3A5): red square on blue -> "A red square on a blue background." ; rendered "HELLO" -> "The text in the image is **HELLO**." ; gradient -> "smooth abstract gradient ... top-left dark black/purple, magenta/pink top-right, vivid green bottom-left, pale gray/white bottom-right" (the true image: R grows with x, G with y, B with x+y, so top-left is black, top-right red-magenta, bottom-left green, bottom-right light: correct up to the colour names). The model emits its reasoning in `content` and, at temperature 0, falls into repeating its final sentence (greedy repetition, also seen on text prompts; not an image-path effect).
## Inkling MTP (DSpark drafter, exact greedy), V100, int4 KV, 2026-11-23
Plain 118.6 tok/s at ctx 40. MTP depth 3/7 at ctx 40: 149-152 tok/s (1.25-1.28x) on code and prose; ctx 512 code depth 3 1.25x; ctx 4K code depth 3 1.10x (depth 7 0.96x); prose 512 depth 7 0.81x; prose 4K depth 7 1.73x; ctx 32K code depth 7 1.69x, prose depth 7 0.55x. Depth 1 is never faster than plain. Ids identical to plain in all 60+ runs. Cost per cycle (ctx 512): drafter 3.2 ms, verify 13.3/15.7/18.3/28.4 ms at 2/3/4/8 rows against 8.6 ms for a plain step, so the gain is capped near 1.25x at typical acceptance. Data: benchmarks/results/inkling_mtp_20261123.json, docs/INKLING_MTP.md.

## Concurrent requests in qv_server (p24/multiseq, 2026-11-24)
Inkling-2-Mini Q3A5, int4 KV, V100 (device 0, `flock /tmp/qv_bench.lock`), `benchmarks/bench_concurrent.py` (streaming /completion, greedy, ignore_eos, cache_prompt off, different prompt per client, 3 runs, spread < 3 %).
Design: N engine sequences (`--parallel N`) over one copy of the weights; every engine op (prefill chunk / decode step + argmax fetch) goes through a FIFO lock, so the sequences are INTERLEAVED, not batched into one forward.
Old = commit 98324e0 (one sequence, FIFO queue); new = `--parallel 4`, `--max-ctx 163840` (40960 per slot), prefill chunk 2048, 1024 while another slot runs.

| case | clients | old agg tok/s | new agg tok/s | new per-request tok/s | old / new mean TTFT |
|---|---|---|---|---|---|
| 100-token prompt, 128 new | 1 | 81.5 | 81.2 | 92.8 | 0.21 s / 0.21 s |
| | 2 | 82.0 | 97.5 (+19 %) | 55-57 | 0.99 s / 0.37 s |
| | 4 | 82.1 | 97.6 (+19 %) | 27.9 | 2.54 s / 0.66 s |
| 4K prompt, 128 new | 1 | 44.6 | 44.7 | 89.4 | 1.45 s / 1.45 s |
| | 2 | 44.5 | 43.1 (-3 %) | 47-54 | 2.88 s / 3.2 s |
| | 4 | 44.7 | 42.9 (-4 %) | 24.7 | 5.74 s / 6.55 s |

Aggregate = generated tokens / wall time (prefill included). Per-request = decoded tokens / (last token - first token). A single stream decodes 93 tok/s through the server (host sync per token); with 2-4 streams the GPU always has a step queued while the host samples and writes SSE, so aggregate decode rises to about 110 tok/s (+19 % end to end), but each request gets 1/N of that: there is no batching, so no weight-read sharing. With 4K prompts the smaller shared chunk (1024 instead of 2048 while others run) costs 3-4 % aggregate (chunk 512: -20 %, 2048: 0 %).
TTFT of a short (~100-token) request that arrives 1 s into a 30K-token prompt of another client: **12.4 s old (waits for the whole prefill) -> 1.65 s new** (shared chunk 1024; trade-off: shared chunk 512 gives TTFT 1.35-1.6 s and long prefill 16.5 s, 1024 gives 1.65-1.9 s / 15.7 s, 2048 gives 1.7-2.2 s / 13.5 s). The 30K prompt itself takes 15.8 s instead of 13.2 s alone-time because the short request's decode steps interleave.
Memory per extra slot: 0.19 GiB fixed (local-attention rings 0.16 GiB + conv states, logits, token buffers) + 9216 B per context token (int4 KV): Inkling at 40960 tokens/slot = 0.19 + 0.35 = 0.54 GiB; the whole 4-slot server at 163840 total tokens used 13.16 GiB on the V100 (production single sequence at 131072: 13.41 GiB incl. vision).
Exactness (tests/test_multiseq.cpp, ids + logits at EVERY step): 4 prompts (37 / 150 / 2600 / 64 tokens) decoded interleaved on one n_seq=4 engine with a late-joining sequence (round 10) and an aborted + re-used slot (step 15), greedy and fixed-seed sampling: every sequence is bit-identical to the same prompt run alone on a single-sequence engine. Inkling int4 KV (48 steps), Inkling fp16 KV (24 steps), Qwen3.8-27B e1 int4 KV (40 steps): all PASS. CPU engines (Inkling, Qwen): the multi-sequence code compiles and an 18-step interleaved test on Inkling CPU was bit-identical, but a 6-step variant showed one sequence differing from the alone run and was NOT resolved (cause not found: possibly leftover state of the alone-reference engine rather than the sequence switch; a fresh-engine reference run was aborted by the owner). CPU multi-sequence is therefore UNVERIFIED; CPU Qwen was not run at all.
Not done (phase 2, true batched decode): one forward over B sequences needs per-row attention against each sequence's own KV (split-K with a sequence table), union-of-experts MoE GEMV with per-row routing, per-sequence loop/sconv state and graphs per B. Estimated upside is bounded by the MoE: at B=4 up to 32 experts per layer are read instead of 8, the dense part (attention projections, router, head) is shared; not measured, so no number is claimed. llama.cpp (continuous batching) reaches higher aggregate throughput by design; no llama.cpp comparison was measured here (no GGUF of this model, models deleted).

## Long context (p30/longctx, 2026-11-27)
V100 shared with other agents (kernel numbers have a few % noise), Q3A5, int4 KV. Dev = e0c600b. Raw logs: benchmarks/results/longctx_20261127/.
**Agent-turn replay** (`benchmarks/longctx_replay.py replay`: two interleaved sessions, 60K-token initial document, 3 turns each, tool-call shaped turns of 50-500 new tokens, reasoning on, greedy, 96 new tokens, `--parallel 1`, single run each):
| build | hits (turns after the first) | turn wall time | prefill per turn |
|---|---|---|---|
| dev, `--mtp` | 0/4 | 41.8-42.9 s (one 84.8 s) | 60.6K tokens at 1.46-1.50K tok/s |
| p30, `--mtp` | 4/4 | median 4.06 s (3.3-4.15; park 1.0-1.7 s + restore 0.2 s + 1.0-1.3 s prefill) | 0.9-1.6K tokens |
| p30, `--mtp`, one session | 4/4 | 1.7-2.1 s (earlier build, N-16 cut) | |
| dev, no `--mtp`, `--parallel 1` | 4/4 | median 2.8 s | |
| dev / p30, no `--mtp`, `--parallel 2`, 40K | 4/4 / 4/4 | median 1.7 s / 2.2 s (grid-aligned cut) | |
Root cause: `has_checkpoint()` was `!mtp_`. Cached == uncached replies (verify mode, 7K tokens, 2 sessions x 3 turns): identical 6/6 with and without `--mtp` (needs `--checkpoint-align`; with the N-16 cut 1 of 6 differed).
**Decode tok/s versus context, engine level** (`qv_ink --bench --synth-ctx`, plain greedy, 3 runs each, spread <0.2 %; dev -> p30): 4K 126.4 -> 125.7, 16K 119.3 -> 120.9, 32K 111.5 -> 115.2, 64K 98.5 -> 104.3, 85K 92.5 -> 99.1, 131K 81.9 -> 89.8, 200K 69.4 -> 76.5, 262K 60.4 -> 69.1
(bit-identical ids/logits to dev: `tests/test_ink_exact.cpp`, 6 prompts up to 85K).
**Server level, `--mtp`, real prompts, reasoning on, 128 new tokens, single run** (dev -> p30): decode 4K 126.5 -> 126.4, 16K 125.4 -> 128.6, 32K 108.3 -> 113.5, 64K 104.1 -> 112.5, 85K 74.1 -> 81.4 tok/s (content dependent: acceptance);
prefill 4K 1651 -> 1655, 16K 1867 -> 1899, 32K 1706 -> 1755, 64K 1439 -> 1525, 85K 1301 -> 1399 tok/s (chunk 2048). At 85K plain decode (99 engine) is still faster than MTP (81): the verify rows kernel `attn_part_rows` costs
about 7.3 ms per R=4 cycle at 85K (microbench, tests/bench_attn_long.cu) because it is instruction bound (per-row dequant + softmax), not bandwidth bound; sharing the dequant between rows needs >255 registers (open).
**Kernels** (microbench, us per layer call at 85K, bit-identical output): decode attention 1 row 394 -> 300, R=4 rows 1151 -> 919; prefill attention (chunk 2048 at 63K) 111.3 -> 88.3 ms, at 83K 145.7 -> 114.6 ms.
Rejected (exact but slower): q in shared memory, forced 4-5 blocks/SM, magic-number nibble convert, skipping the softmax rescale. Profile (nsys, 85K prompt): k_attn_pf = 48 % of prefill kernel time, per-chunk weight dequant 9.6 %.
**Prefill chunk** (qv_ink, 32768 tokens): 2048 17.81 s, 4096 14.06 s (-21 %), 8192 11.78 s (-34 %) with the shared-tile kernel (1-head kernel: 18.11 / 15.07 / 12.93 s). Numerics differ between chunk sizes (not bit-identical to 2048); `--prefill-chunk 8192`
ran out of memory in the server at `--mtp --ctx-size 131072`. Server replay with `--prefill-chunk 4096` (--mtp, 2 x 60K): cold prefill 31 s instead of 39 s (1.96-1.99K tok/s), but the grid-aligned checkpoint re-prefills up to 4096 tokens per turn, so the median turn is 5.1 s instead of 4.1 s: a bigger chunk helps cold prompts, not turn latency. Default left at 2048.
**C4 (max usable context)**: engine decode at position 262,024 over synthetic KV fits (12.61 / 15.77 GiB, 69.1 tok/s); 131K 11.49 GiB. Real prompts were prefilled only up to 85K (prefill 60.7 s through the server); no NIAH/ppl at 128K-256K was run here, so no quality claim above 85K.
## Inkling speculative sampling at T = 1.0 (p29/spec-speed, 2026-11-27)
qv_server, defaults top_k 40 / top_p 0.95 / min_p 0.05, ctx 25, 256 tokens, 3 runs, decode tok/s, dev (e0c600b) vs p29 binary measured back to back (V100, int4 KV):

| class | dev --mtp | p29 --mtp | dev no MTP | p29 no MTP |
|---|---|---|---|---|
| code | 104.5 | 130.3 | 106.4 | 107.0 |
| prose | 104.0 | 130.6 | 104.7 | 105.2 |
| tool (p29 with --mtp-grammar) | 108.9 | 171.0 (146-188) | 110.5 | 111.0 |

Distance to 200: code 1.5x, prose 1.5x. A verify cycle is about 17 ms (MoE union floor about 8.7 ms) and yields 2.4-2.6 tokens; a sampled plain step is about 7.7 ms (130 tok/s).
