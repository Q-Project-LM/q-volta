# Inkling-2-Mini CUDA engine (V100, sm_70)

Sources: `src/runtime/cuda_inkling_engine.cu` (engine, schedule, graph capture, prefill driver), `src/backends/cuda/inkling/ink_ops.{h,cu}` (decode kernels),
`src/backends/cuda/inkling/ink_prefill.{h,cu}` (batched-prefill kernels), `tools/ink_run.cpp` (`qv_ink`), `tools/ink_check.py`, `tools/ink_pfcheck.py`.
`make_cuda_engine` dispatches on the model arch (`inkling2_loop`). Model: Q3A5 (`inkling2-mini-q3a5.qv`, 9.886 GiB, all of it resident).

## Decode (step 1)
One CUDA graph per token (the loop schedule of 48 layer applications is unrolled at capture; position and token live in device memory).
Per application: attention-residual mix (+norm, int8 quantisation) -> one multi-matrix GEMV launch (q/k/v/gate dp4a GEMV plus the small fp16 rel/sink/dw projections)
-> prep (short convs, l2 clips, tau, relative-bias table) -> split-K flash-decode (8 lanes per key, 64 splits on global layers, 8 on local) -> combine with sink and
sigmoid gate -> o-proj -> post-attention conv+residual -> router (device-side top-8 with a last-block ticket, graph capturable) -> grouped-expert w13 / w2 GEMV (8 experts
in one launch) -> conv + residual. Loop passes >0 add the message gates (cat/down/forget/gain). Local layers keep a 1024-slot ring; global layers keep the full cache.
Logits: gemv_q8 on the unembedding, softcap 30, argmax fused in the graph (`--bench` replays it).

Decode result before p26 (Q3A5, V100, ctx ~25, fp16 KV): 122.4 tok/s. At 4K real context: 113.8 tok/s. After p26 (int4 KV): see below. That is about 33 % of the HBM2 bound at the stored precision, so the engine is NOT
bandwidth bound: nsys (`--cuda-graph-trace=node`) shows many short latency-bound kernels (mean 7-33 us each, 20-25 launches per application). Largest decode kernels per token:
multi-GEMV 1.04 ms, w13 0.93 ms, attention-residual 0.63 ms, w2 0.56 ms, router 0.39 ms, attention split 0.36 ms (ctx 4K), gate-down 0.29 ms, prep 0.24 ms, norm 0.23 ms, finish 0.22 ms.
Remaining headroom is kernel fusion / fewer launches (e.g. fuse norm into the preceding reduction, persistent MoE), not a faster GEMV.
Profile: `benchmarks/results/inkling_gpu_20261121.json` (summary); raw report `/tmp/ink/prof_decode.nsys-rep` (not committed).

### Decode latency work (p26/ink-fuse, 2026-11-24)
Plain decode, int4 KV, same session, 3 runs each: ctx 25 118.65 -> 135.4 tok/s (+14.1 %), ctx 4K (real) 111.4 -> 126.7 (+13.7 %), ctx 32K (synthetic KV) 99.8 -> 111.6 (+11.9 %).
Every step is bit-exact (logits hash of every step and ids identical to the f0645bd binary on the four golden prompts incl. the 4K one, 64 steps, via `tests/test_ink_exact.cpp`;
`test_ink_mtp` depth 3/7 PASS, `test_multiseq` int4 + fp16 PASS), because no reduction order was changed: only loads were issued earlier, loops unrolled, work split into more
(blocks / warps) and barriers shared. The profile found that the inter-kernel gap inside the graph is only ~0.47 us (3 % of the step); the time is each small kernel's own
dependent latency chain, so the wins came from shortening those chains, not from fewer launches (the one launch removal, folding k_finish into the next attn_res, gave +1.9 %).
What worked: finish folded into the next attention-input attn_res; GEMV rows one per warp and one warp per block (more, smaller blocks beat 4 warps x 2 rows: +2.7 %); all independent
global loads of attn_res / gate_down / gate_apply / norm issued before the first store or barrier (the compiler cannot hoist loads over stores that may alias); the 128-byte local copy of
the argument struct in k_norm removed (+2.0 %); router top-8 by rank instead of 8 serial warp argmax rounds; attn_combine stages the split maxima in shared memory; the five l2 norms
of attn_prep share one reduction round; the two gate_down sums share one barrier set.
What did not (measured, reverted): merging attn_part+combine or gate_apply+norm with a last-block ticket (-3 %: the serial tail in one block costs more than the launch), a side-stream
graph branch for wg/sink (-6 %: fork/join nodes cost ~10 us each on this driver), `prefetch.global.L2` of the next kernel's weights (-12 %), forcing occupancy with `__launch_bounds__` min blocks
(spills, -3..-47 %), unrolling the quantised GEMV lane loop (-3 %), more rows per warp in w13, moving finish into the w2 epilogue (-1.3 % vs folding it into attn_res).
Not possible on sm_70: programmatic dependent launch (needs sm_90). Cooperative persistent kernels were not tried: a grid barrier (~2-4 us) is about the cost of a graph kernel boundary (0.47 us gap + the kernel's own tail).
Remaining time per token (nsys, ctx 25, 9.06 ms under the profiler vs 7.4 ms without): w13 1.58 ms (33 us x 48, 0.5 TB/s), multi-GEMV 1.60, attn_res 1.26 (99 calls, one block each, ~4.9 us of
loads per call: one SM cannot stream the ~150 KB it reads), w2 0.92, attn_part 0.66, router 0.55, prep 0.42, gate 0.82, combine 0.33, norm 0.28, lm_head 0.31: still ~35-40 % of the HBM2 bound.
MTP verify (`QV_MTP_PROF=1`, ctx ~35): plain step 8.34 -> 7.30 ms, verify R=2/3/4/8 13.48/15.58/18.27/28.79 -> 12.82/15.01/17.80/28.53 ms, so the MTP-3 gain over plain shrank (code 1.14x -> 1.02x, absolute 135.3 -> 137.7 tok/s; prose 94.3 -> 96.4): the rows kernels (w13_u / w2_u / part_rows) still have the old structure and are the next target.

## Batched prefill (step 2)
Per chunk (default 2048 tokens, token-parallel): attention-residual and norms emit fp16 rows; projections = dequantise the weight to an fp16 slab + cuBLAS tensor-op GEMM
(fp32 accumulate); relative bias table `rl[t,h,d] = rel[t,h,:] . proj[:,d]` is one GEMM; flash attention is a WMMA two-pass kernel (pass 1 row max incl. sink, pass 2 exp and PV)
with window / causal masking and the sigmoid gate fused; the sliding-window layers stage the 1024-key ring + the chunk (`ring_to_stage`/`stage_to_ring`) so chunks
chain exactly; conv states carry across chunks. MoE: sigmoid top-8 per token, counting sort by expert, gather, per-expert dequant + cuBLAS GEMM (w13 then w2), act kernel
(softcaps 4/25), weighted combine. The router logits use an fp32 GEMM (an fp16 router flipped near-tie expert choices; found by comparing per-application router ids).
Decode graph and prefill share all state, so prefill -> decode is seamless. `QV_INK_STEP_PREFILL=1` falls back to token-by-token.

## int4 KV (step 3)
Global layers (8 applications) store K and V as int4 per token and kv head: 128 channels -> 64 B of nibbles, plus an fp16 (scale, min) pair per 64-channel group, for K and V
(144 B per head-token, 1152 B per token per application, **9216 B/token** for the 8 applications). Quantisation is asymmetric min/max per token, so there are no 64-token blocks and
no fp16 recent window (a simplification of the generic kv4 block records; chosen because the Inkling K is l2-clipped and the test below shows no measurable damage).
Local layers keep fp16 1024-slot rings (a fixed cost). Decode dequantises in registers; the prefill tile loader
dequantises into the WMMA tile. `--kv int4|fp16`.

## Limits
- Max context with int4 KV: 580K tokens fit (chunk 512, 15.68 / 15.77 GiB used); 620K does not. The design estimate was 478K; the difference is that the weights (9.89 GiB) plus
  scratch leave ~5.2 GiB and 9216 B/token. 256K uses 2.4 GiB of KV.
- fp16 KV: 131K needs 14.4 GiB of 15.77 (tight); 256K does not fit.
- Prefill is quadratic: attention dominates beyond ~16K (see table in PERFORMANCE.md).
- Decode attention at long context is 2-3x above its own HBM bound (256K: 14.5 ms/token after p30, 17.4 before, vs 8.2 ms at 4K; the int4 KV read is 2.4 GB = 2.7 ms at 900 GB/s).
- The pi (harness-hub) LoRA is not implemented in this engine; the plan (merge before quantisation) needs no engine change. Stage B on ARM is not addressed here.

## Long context (p30/longctx, 2026-11-27)
- Checkpoints with MTP: `save_checkpoint` appends the drafter's local-layer context rings (K,V fp16 x 1024 slots, 2 layers) and the recurrent-head feature; `save_kv_prefix` appends the drafter's global-layer
  context K/V prefix (NKV x n x 128 fp16 x 2). Tested by `tests/test_ink_ckpt.cpp` (ids, logits and acceptance counts identical after L1 / L2 restore).
- Decode attention (`k_attn_part`): the int4 loader fetches the next key before consuming the current one; the MTP rows launch is grid (NKV*rows, nsplit) so the rows of one split are adjacent (share K/V through L2).
  The kernel is instruction bound at long context (~340 instructions per key and lane, 128 of them nibble dequant): 85K single row 300 us/layer, R=4 919 us/layer. Bit-identical to the previous kernel.
- Prefill attention: `k_attn_pf3` shares each K/V tile between the 3 query heads of a kv head (12 warps per block, 97 KB shared memory): 111 -> 88 ms per 2048-token chunk at 63K, bit-identical (`QV_PF1=1` = old kernel).
  Attention is 48 % of the 85K prefill; the rest is dominated by per-chunk expert dequant + GEMM, which a larger `--prefill-chunk` amortises (32K prompt: 17.8 s at 2048, 11.8 s at 8192).
- Chunk-boundary dependence: prefilling a prompt whole vs in two pieces gives different logits (cuBLAS shapes), hence the grid-aligned checkpoints in docs/SERVER.md.

## Correctness
See PERFORMANCE.md. Quantised-reference goldens (`/tmp/ink/qg`, from the CPU Q3A5 reference) and the fp goldens in `tests/golden/inkling`.

## Sequences (p24/multiseq)
`EngineOptions::n_seq = N` allocates N copies of the per-sequence state: local-attention rings, global KV (fp16 or int4), conv states of k/v/attn/mlp, logits, token/position buffer, argmax history and the CUDA graphs (captured lazily per sequence, they bake in the sequence's pointers). Weights, scratch, prefill buffers and cuBLAS state are shared: every step / prefill chunk fully rewrites the scratch it reads, so a sequence switch (`select_seq`) is pointer bookkeeping only (no copy, no sync) and results are bit-identical to running alone (tests/test_multiseq.cpp). Per extra sequence: 0.19 GiB + 9216 B x max_ctx (int4 KV) (see docs/PERFORMANCE.md). MTP keeps single-sequence drafter state (n_seq > 1 with MTP throws).

## Speculative verify (p29/spec-speed)
- `decode_speculative` (T > 0, lossless) and `decode_speculative_constrained` are implemented in `cuda_inkling_mtp.inc`; see INKLING_MTP.md section 11.
- Verify rows kernels: `k_w2_g` is expert-grouped over the union of routed experts (4 rows per warp, per-row arithmetic order unchanged), `k_w13_u` and `k_w2_g` use `warp_reduce4` (6 shuffles for 4 rows,
  xor-tree order 16,8,4,2,1 kept). Bit-identical to the 1-row kernels. Verify R = 4 is bandwidth-bound on the union (about 28-32 experts, about 4.4 MB per expert per application), floor about 8.7 ms for MoE.
