# Q-Volta — Progress log (newest last)
- 2026-11-19: environment audit (V100 free, CUDA 12.9; CPU = Xeon E5-2683 v4, AVX2 only, no AVX-512). Prior sessions: engine skeleton, dp4a GEMV at 81-99% of HBM peak, CUDA-graph decode.
- 2026-11-19: real E0 model converted (12.63 GiB); golden PASS on real weights; decode 51.1-52.2 tok/s.
- 2026-11-19: requirements added: pi LoRA adapter (converter --lora, merge before quant), ARM/Snapdragon design constraint for Stage B (no ARM tooling here), prefill must beat competitors, 256K context.
- 2026-11-20: B0 baseline: NInfer re-run on this V100 (plain 45.3, MTP-3 106.5 code / 69.0 prose). Bytes-normalized comparison shows most of our raw decode lead is fewer bytes.
- 2026-11-20: adapter analysis: RTN noise >> adapter shift; runtime LoRA planned.
- 2026-11-20: research fan-out (decode kernels, MTP, prefill, quantization quality) -> PLAN.md (hypotheses with mechanism/expected/falsification).
- 2026-11-20: P1 batched prefill: dequant_rows_f16 (751 GB/s), gdn_seq (2.15 us/token/layer), attn_seq (8.4 TFLOP/s), cuBLAS slab GEMM; golden PASS; prefill 51 -> 1013 tok/s @512 (NInfer 837), 917 @4088 (NInfer 1013).
- 2026-11-20: prefill chunk 512 -> 2048 (fewer dequant passes): 1205 / 1143 tok/s @2048/4088 vs NInfer 1010/1013 (+19%/+13%); multi-chunk golden PASS.
- 2026-11-20: split-K (flash-decoding) decode attention wired in (13x faster kernel at 32K ctx, 8x at 4K; kernel test vs old kernel rel err <=6.4e-6, q8 bit-identical). End-to-end decode after a 4088-token prompt: 36 -> 50.2 tok/s; short context 52.3 (unchanged). Scratch scores[24][max_ctx] replaced by fixed 1.9 MiB.
- 2026-11-20: CPU backend v1 (AVX2 GEMV + ops + engine): 3.93 tok/s decode @32 threads (~49-54 GB/s of RAM, bandwidth-bound), golden PASS (thin margin). GPU work paused at user request.
- 2026-11-20: CPU batched prefill (gemm_q, 256-token chunks): ~21-24 tok/s @32 threads vs 3.2 (step loop) under heavy host load (lower bound); golden PASS on prefill path (top1 0.9524, KL 5.1e-3).
- 2026-11-20: OpenAI-compatible server (/health, /v1/models, /v1/completions, /v1/chat/completions, SSE streaming), native Qwen BPE tokenizer (509/509 vs tokenizers), chat template (31/31 vs HF), sampler, prefix reuse. test_server.py 45/45 on CPU backend; verified by curl. CPU TTFT 0.86 s (19 tok) / 4.6 s (136 tok), ~2.7 tok/s streaming under load.
Next: tensor-core attention + dequant/GEMM overlap (prefill at long prompts), runtime LoRA, MTP, KV quantization + split-K decode attention, quantization quality (clip search, bit allocation, GPTQ), NIAH/ppl at 32K-256K.
- 2026-11-20: CPU MTP speculative decoding (p7/mtp): exact (ids identical to plain greedy, 6 prompts x 97 tokens + ctest), bit-exact 2-token verify (gemv_multi_q), GDN state snapshot/rollback; decode 2.78 -> 4.46 tok/s (1.60x) on a code prompt, 1.46-1.55x on text at 128-1536 ctx; acceptance 0.79-0.94; verify pass costs 1.10x a 1-token step. See docs/CPU_OPTIMIZATION.md.
- 2026-11-20: GPU MTP speculative decoding (p8/gpumtp, `qv_run --mtp --mtp-depth 1..3`): exact (ids and final logits bit-identical to plain greedy in 60 benchmark runs + test_cuda_mtp), bit-exact multi-token dp4a GEMV verify, all-rows-per-launch norm/conv/GDN/attention kernels, GDN/conv snapshot rollback, MTP prefill. V100 decode 52.2 plain -> 87.2 / 116.7 / 129.5 tok/s (depth 1/2/3) on the code chat prompt and 79.6 / 83.9 / 84.3 on the prose chat prompt; real text 69.8-134.5 depending on context and depth (1.4-2.6x). Beats NInfer MTP-3 (106.5 / 69.0, earlier run, different measurement path) by 22% on both chat prompts at depth 3. Verify = 1.11x a plain step at 2 tokens. Graphs only +0.3% (GPU-bound). Limits: fixed depth, 0.15 GiB snapshot per depth step, no server integration, draft lm_head is 1.2 ms of each ~1.6 ms draft step. See docs/PERFORMANCE.md.
- 2026-11-20: int4 KV cache for the CUDA backend (p9/kv4, `--kv fp32|fp16|int4`, hypothesis H-A): 64-token block records (K int4 per-channel scale/zero or per-token groups, V int4 per-token groups, 4.5 bit/element, 18 KiB/token for the 16 attention layers vs 128 KiB fp32) + a 128..191-token fp16 window; prefill KV write, split-K decode attention, prefill attention and the MTP verify path all use it. Exact: MTP depth 1/3 ids and final logits identical to plain greedy on the same int4 cache; nt-row verify launch bit-identical to 1-row launches. Quality (E0, wikitext 4088 / 8192 tokens): ppl +-0.09% vs fp32 KV (per-channel K -0.05% / +0.02%), KL ~8e-4 (fp16 KV floor 8e-6), top-1 agreement 98.7%; passkey retrieval 6/6 at 4K, 8K, 16K, 32K (same as fp16 KV). Decode: no slower than fp32 KV at any context (512: 51.7 vs 52.0 tok/s; 4K 50.5 vs 50.0; 16K 42.5 vs 39.7; 32K int4 38.4 vs fp16 36.0, fp32 does not fit); MTP-3 still exact (133 tok/s @512, 89.8 @4K, 44.6 @32K). Max max_ctx measured: fp32 18,432, fp16 38,912, int4 139,264 (143,360 OOM). 256K by arithmetic: 4.5 GiB int4 KV but ~13.35 GiB of non-KV memory today vs 15.77 GiB available (~2.1 GiB short). Limits: GPU only, lengths > 32K fit but are unvalidated, decode attention appears to be instruction-bound (inferred, not profiled) so the byte saving is barely turned into speed, no host-RAM tier yet. H-A falsification (ppl +>3% or NIAH drop at <=32K) not met. See docs/PERFORMANCE.md, benchmarks/results/gpu_kv4_2026-11-20.json.
- 2026-11-20: H-D tensor-core prefill attention (p9/tcattn): WMMA HMMA flash kernel (48 rows = 8 queries x 6 GQA heads, 64-key tiles, fp32 online softmax, fp32->fp16 KV tiles in smem behind a KvSrcF32 load abstraction). Attention-only 3.0-3.9x faster than the fp32 CUDA-core kernel from 4K to 256K ctx (27-29.5 TFLOP/s); extrapolated 256K prefill attention 499 s vs 1725 s. End-to-end prefill +9% / +15% / +28% at 4K / 8K / 16K (1191 / 1088 / 888 tok/s); golden KL unchanged (step 6.5e-4, prefill 7.3e-5). H-D not falsified (>2x) but below the 5x hoped for. 32K end-to-end not measured (fp32 KV does not fit).
- 2026-11-20: Gen 6 host-RAM KV tiering study (p9/tier, docs/LONG_CONTEXT.md, benchmarks/results/tier_20261120.json): real K/V/queries of the 16 attention layers dumped at 32K (CPU engine, wikitext + needle prompt). H-B (sparse top-k 64-token blocks, Quest bound) mass recall with sink + 1K recent + 4K fetched tokens: 0.91 at 16K, 0.83 at 32K (0/16 layers >= 0.95 at 32K; oracle block selector 0.90) -> the >=0.95 criterion FAILS; end-to-end at 32K it is benign (KL 0.008, top-1 95%, NIAH found with 1K fetched). Zero-copy gather kernel (small persistent grid) 11-12.2 GB/s for scattered 18 KiB units (cudaMemcpyAsync per unit only 4 GB/s), overlapped prefetch costs +3/+6/+11% of a step at 2K/4K/8K fetched (stand-in pipeline); CPU int4 attention over all cold KV at 256K = 10.4 ms/layer (166 ms/token, H-C refuted as a decode path, 185 ms/5.4 tok/s with the GPU step). 256K extrapolations are order-of-magnitude only; nothing > 32K measured.
- 2026-11-20: merged p9/tcattn into p9/kv4: the tensor-core prefill attention reads int4 KV (window rule preserved, mixed tiles as two masked steps). int4 prefill 1,131 / 1,048 / 902 / 686 tok/s at 4K / 8K / 16K / 32K (32K was ~488 with CUDA-core attention); ctest 13/13, golden PASS, MTP int4 ids+logits identical; passkey 2/2 at 64K (prefill 146 s) and 2/2 at 128K (prefill 530 s, max_ctx 131,136). See docs/PERFORMANCE.md.

## p10/quant (2026-11-21)
GPTQ + clip + group128 + meta8 quantizer and kernels; profile e1 10.44 GiB +9.6% ppl (KL 0.112), 55.4 tok/s decode; <=10.4 GiB at +5% not reached. Details and falsified hypotheses: docs/QUANTIZATION.md; data benchmarks/results/quant_2026-11-21.json. ctest 13/13 on the merge with main.

## p12/ink-design (2026-11-21)
- Inkling-2-Mini design + feasibility study (docs/INKLING_DESIGN.md, tools/ink_model.py; pure analysis, nothing run on GPU/CPU). Exact budget from the safetensors headers (text 18.89 GB: experts NVFP4 layers 0-19 + MXFP8 layers 20-23 = 15.67 GB, attention 1.29 GB BF16, untied lm_head 0.93 GB; MTP is a separate 0.64 GB block drafter file). Q3 profile 9.7 GiB resident; decode ceiling V100 392 tok/s (bandwidth) / ~206 (kernel-overhead model), CPU 23.8 tok/s; max context on V100 ~478K (per-pass KV) / ~860K (shared KV), 1M only with host KV tier or hybrid expert residency; MTP on MoE 1.3-1.7x (alpha 0.75). One new measurement: requantizing NVFP4 experts to uniform int adds 10% rel. RMS noise at 4-bit g128 and 19-22% at 3-bit.
## p12/ink-spec (2026-11-21) - Gen 7 H-I1: Inkling-2-Mini spec + reference
Weights downloaded (/workspace/q-volta-models/Inkling-2-Mini, 19 GB; set HF_HUB_DISABLE_XET=1, xet stalled via the proxy).
Finding: installed transformers 5.17 has no `inkling2_loop`; PyPI 5.23 has only Inkling2 Small (no loop); the loop model is in the
open PR huggingface/transformers#51050 (read, not executed - permission denied). docs/INKLING_SPEC.md = exact spec (all ops, checkpoint
tensor table, NVFP4/MXFP8 dequant, tokenizer/template, MTP inventory). tools/inkling_ref.py = own torch reference (CPU fp32);
goldens in tests/golden/inkling (4 prompts, 32 greedy tokens each, 48 layer-application hiddens, router ids; 8.9 MB).
HF logit parity NOT verified (H-I1 partially: spec + behaviourally validated reference, HF comparison pending). Decode bytes/token at
stored precision 5.4 GiB (68 % dense bf16) -> ~130 tok/s V100 ceiling unless dense weights are quantised.
## p11/perf (2026-11-21)
Fixed the decode regression introduced by p10/quant (bisected: only 28dcefc; tensor-core attention and int4 KV are neutral). Hot GEMV kernels now compile per (group, meta8) mode, so group-64 E0 runs the pre-p10 code; meta8 row scale hoisted. E0 code: plain 48.7 -> 52.2, MTP-3 104.4 -> 131.5 (324ea1e: 52.2 / 129.6); E1: plain 55.4 -> 57.6, MTP-3 114.7 -> 138.2. Bit-exact (ids identical, ctest 13/13, test_cuda_mtp E0+E1). E1 is faster in tok/s but ~9% lower in effective bytes/s than E0. See docs/PERFORMANCE.md, benchmarks/results/gpu_perf_2026-11-21.json.
## p14/ink-cpu (2026-11-21)
CPU engine for Inkling-2-Mini (src/runtime/cpu_inkling_engine.cpp, ink_ops, docs/INKLING_CPU.md): decode (one parallel region per MoE layer application, experts x row chunks),
batched prefill with token-sorted expert GEMM, fp16 KV (local ring + global per-pass slots). Q3A5: decode 11.0-11.7 tok/s at ctx 128-2048 (54 % of the 54 GB/s bound, ~75 % of the
39 GB/s this box streamed during the runs), prefill 66-83 tok/s; 32/32 greedy ids on 3 of 4 goldens vs the Q3A5 reference (p1 flips at a 0.056-logit near tie), val NLL +0.0005..0.002.
Router ids are not identical (int8 activation noise amplified by the loop). New goldens tests/golden/inkling_q3a5 (tools/inkling_golden_q.py).

## p14/ink-gpu (2026-11-21)
Inkling-2-Mini CUDA engine on V100: decode 122 tok/s (graph, ~33 % of HBM bound, latency bound), batched prefill 2.1-2.8K tok/s at 4K, int4 KV (9216 B/token), max ctx ~580K,
passkey 32K/128K/256K passed. pi LoRA and ARM Stage B not covered. docs/INKLING_ENGINE.md.

## p15/srv2 (2026-11-21) - llama-server compatible qv_server
- qv_server rewritten to be 1:1 with llama-server (llama.cpp 9d6f04f): all llama flag spellings + LLAMA_ARG_* env (tools/gen_llama_flags.py -> src/server/llama_flags.inc), endpoints /completion /infill /v1/completions /v1/chat/completions /tokenize /detokenize /apply-template /props /slots (+save/restore/erase) /metrics /models /health (503 while loading), llama error JSON, CORS, SSE framing, llama sampler chain (mt19937 draw), n_probs/logprobs, n>1, reasoning_content, timings, cache_n. Matrix: docs/SERVER_COMPAT.md.
- Tests: tests/test_server_compat.py = stand-alone checks (mock engine) + structural-diff cases vs the real llama-server (Qwen2.5-0.5B GGUF): status, content-type, header names, JSON key/type shape, SSE event shapes; documented exceptions whitelisted. tests/test_server.py (real e0 model, CPU, 32 threads): 47/47, stream 3.0 tok/s at load 11.7 (no regression vs ~2.7-3.9).
- Open: tools/grammar/custom Jinja templates wait for p15/jinja (src/jinja, src/grammar not on the main branch yet); one decoding sequence (queued slots); embeddings/vision/runtime LoRA unsupported; DRY approximate; CUDA server path not run.

## p15/jinja (2026-11-21) - Jinja2 interpreter + chat helpers (milestone A)
src/jinja: standalone Jinja2-compatible engine + chat rendering + tool-call / reasoning output parser (docs/JINJA.md). Differential test against
Python jinja2 3.1.6 (oracle env = HF settings): 82/82 templates (80 llama.cpp `models/templates` + Qwen3.8-27B + Inkling-2-Mini) byte-exact on 33
generated conversations each (2345 exact renders + 361 cases where both engines raise, 130 of them `raise_exception` with identical message), 0 mismatches;
94/94 feature snippets match; ctest `jinja` 90 checks (incl. golden renders of 11 templates and cross-check against the hand-written
src/tokenizer/chat_template.cpp). Not done: include/import/extends, autoescape, Python Markup semantics (deliberately follows llama.cpp).

## p15/jinja (2026-11-21) - GBNF grammar + JSON schema + tool-call grammars (milestone B)
src/grammar (docs/GRAMMAR.md): GBNF parser, stack-set matcher with interned states, trie masks with per-state cache, JSON-schema -> GBNF, json_object grammar,
tool-call grammars (Hermes/QwenXml/Inkling/Mistral/Generic, forced/parallel/lazy-trigger), `qv::LogitMask` hook in the sampler. ctest `grammar`: 807 checks pass,
including mask == brute-force per-token acceptance on the real Qwen3.8 tokenizer (248077 tokens; at sampled steps of 5 grammars, all allowed tokens + 1/37 of the rest) and on a small UTF-8 vocab.
Mask latency, single thread, Qwen3.8 vocab (first-visit = state mask computed, warm = cached lookup; mask_logits = applying a cached mask to 248K logits):
```
[qwen] vocab=248077 trie_nodes=591809 max_piece=128 eog=2 build=169.3 ms
[qwen] json_object     steps=41 cold avg=1.635 ms max=10.893 ms | warm lookup avg=0.000 ms | mask_logits avg=0.050 ms | states=448 masks=25
[qwen] schema_obj      steps=25 cold avg=1.260 ms max=10.617 ms | warm lookup avg=0.000 ms | mask_logits avg=0.052 ms | states=205 masks=20
[qwen] enum+int_range  steps=14 cold avg=0.059 ms max=0.181 ms | warm lookup avg=0.000 ms | mask_logits avg=0.072 ms | states=120 masks=15
[qwen] qwen_tool       steps=45 cold avg=0.969 ms max=10.940 ms | warm lookup avg=0.000 ms | mask_logits avg=0.060 ms | states=127 masks=43
[qwen] utf8_class      steps=3 cold avg=7.340 ms max=13.444 ms | warm lookup avg=0.000 ms | mask_logits avg=0.255 ms | states=4 masks=3
```
First-visit masks cost up to ~11 ms (states inside a string where most of the vocab is allowed), ~1-1.6 ms averaged over short 25-45 token samples; cached lookups are ~0.001 ms and
mask_logits 0.05-0.07 ms (0.26 ms with only 3 allowed CJK-range tokens path). Long generations amortise well below 1 ms/token; short ones with many distinct states do not (not measured on a long run).
Not supported: `<[id]>` token elements; non-integer minimum/maximum are not enforced.

## p16/srv3 (2026-11-21) - Jinja + grammar integrated into qv_server
- Chat templates via qv::chat::ChatTemplate (model template, --chat-template, --chat-template-file; native Qwen3.8 renderer only as fallback, byte-equal on the tested conversations). tools/tool_choice/parallel_tool_calls -> OpenAI tool_calls (non-stream + streaming deltas), reasoning_content via the chat parser, /apply-template with tools.
- grammar / json_schema / response_format / forced tool_choice / lazy tool triggers / --grammar / --json-schema via src/grammar on host logits (argmax + MTP paths skipped when constrained).
- tests/test_server_compat.py: 121 passed (48 stand-alone incl. tools, tool streaming, parallel calls, history, grammar, json modes, template flags; 73 structural-diff cases vs llama-server incl. tools/json/grammar). tests/test_server.py real e0 CPU: 50/50 incl. real-model tool call, required tool_choice, json_object, grammar; stream 2.9 tok/s at load 12.5.
- 2026-11-22: H-I1 closure (p17/hfparity): HF PR #51050 (a10ebc3 model code) run in an isolated venv, CPU bf16, vs tests/golden/inkling: 4 prompts, greedy ids 32/32, 32/32, 32/32, 31/32 (p4 near-tie flip at step 17, 0.016 logit gap), last-position top-20 max abs 0.22/0.24/1.10/2.05, KL 1e-11..1e-10. No semantic difference found; reference and goldens unchanged. HF dequant bit-identical to ours. tau != 1 verified only at attention-module level (fp32, rel err <=2e-7, real floor T=8800); no end-to-end 16K run (HF CPU 2.6 tok/s). PR has no MTP code. See docs/INKLING_SPEC.md section 15, docs/results/inkling_hf_*.

## p18/compare (2026-11-23) - fair comparison stand vs llama.cpp on the V100
benchmarks/compare/: scripts, `llama_eval.cpp` (libllama harness writing/reading the qv_run eval format), results.json, raw/, README with protocol and verdict. Measured: Q8_0 reference (ppl 8.467), Q3_K_M, IQ3_XXS+imatrix (Q-Volta GPTQ calibration text) vs e0/e1: llama.cpp better quality at equal bytes, Q-Volta faster (decode, prefill, MTP) and longer int4 KV. ExLlamaV2/V3, vLLM, SGLang not run on sm_70 (reasons in the compare README). Missing: IQ3_S, no-imatrix IQ3, BF16-based KL.

## p19/ink-vision (2026-11-23) - image input for Inkling-2-Mini
Spec docs/INKLING_VISION.md (preprocessing = pad to 40 px tiles with no resize, ViT 23 layers / 2-D axial RoPE / bilinear position table, 2x2 merge projector, one text token per 40x40 tile, rows replace post-embed_norm embeddings, no special attention). `tools/convert_inkling_vision.py` -> sidecar `inkling2-mini-vision.qv` (0.639 GB fp16; int8 0.323 GB, 2-4 % error, not default).
`src/vision/` (stb_image decode, preprocessing, CPU AVX2 and CUDA encoders), `Engine::prefill_embeds` (CUDA + CPU Inkling engines, +60 lines), server `image_url` (data URLs + `--image-root` files, http(s) only with `--allow-remote-images`, SSRF-guarded), `/props` vision, content-hashed pseudo ids for the prompt cache.
Validated against the HF PR tower on 6 images: pixels equal to 1.2e-7, embeddings rel L2 2e-6..1e-5 (CPU) / 6e-4..4e-3 (CUDA) vs 1.3-5 % for HF's own bf16; e2e with the Q3A5 engines top-1 equal, 9/32 (CUDA) and 31/32 (CPU) greedy ids vs the fp32 reference (quantisation noise). Encoder 5.5 / 23 / 186 ms (CUDA) and 143 ms / 0.75 s / 7.8 s (CPU) for 224 / 512 / 1024 px; TTFT through the server 0.22 / 0.32 / 0.71 s. Real answers (red square, "HELLO", gradient) correct. Tests: tests/test_server_vision.py (8, mock), tests/test_server_compat.py 121 pass unchanged.
## p20/webui (2026-11-23) - llama.cpp web UI embedded in qv_server
- UI built from source at llama.cpp 9d6f04f (`ui/apps/chat`), vendored gzip (src/server/webui, MIT, THIRD_PARTY.md), embedded through a generated source (cmake/qv_embed_webui.cmake); routes, gzip/415, ETag/304, Cache-Control, COEP/COOP, --no-webui, --path, --api-key, --api-prefix diffed against llama-server.
- Driven in headless Chromium (Playwright) against the mock backend: loads, streams markdown + reasoning, settings dialog, client-side tool call, Stop. Mismatches found and fixed: /tools and /cors-proxy 403, /v1/streams/lookup + resumable streams, prompt_progress, timings_per_token, SSE pings, /v1/chat/completions/control.
- Added /v1/messages, /v1/messages/count_tokens, /v1/responses (event sequences diffed vs llama-server), --sleep-idle-seconds, --sse-ping-interval, --ui-config(-file).
- Not implemented: router mode, /tools, /cors-proxy, embeddings, rerank, audio; log format differs; one decoding sequence.
## p19/ink-mtp + p22/ink-mtp2 (2026-11-23) - Inkling MTP
DSpark drafter graph reconstructed from vLLM and validated with a torch reference (mean accepted 3.5/7 on code); CUDA implementation with bit-exact batched verify, sidecar `inkling2-mini-mtp.qv` (258 MiB), server flags. Exact; speedup 0.55-1.73x depending on text, about 1.25x typical. Verify (2.5 ms per extra row) is the bottleneck; not optimised. See docs/INKLING_MTP.md.

## p23/embed (2026-11-23) - self-contained .qv (tokenizer, chat template, generation defaults, meta embedded)
- Container format_version 2: optional `blobs` (64-byte aligned after the tensors) + `meta` in the JSON index; old files and old runtimes unaffected (docs/QV_FORMAT.md).
- `tools/qv_embed.py` (embed / --list / --extract / in place, streamed, sha256-verified, disk check), hooks in convert.py --emit-qv, convert_inkling.py, qv_pack.py (`--no-embed` to skip).
- Runtime: `ModelFile::blob()/meta()`, `Tokenizer::from_memory`, qv_server uses embedded tokenizer/template/eos ids/sampling defaults (flags win, `--no-embedded-defaults`), qv_tok takes a .qv.
- Verified on real files: qwen38-27b-e1-st.qv and e0-st built, tensor-area sha256 identical to the originals, CPU-backend qv_server with no --tokenizer: /tokenize identical to the --tokenizer path and the same greedy chat answer. GPU was occupied by another job, so no CUDA run. Inkling-st not built (disk).
- Overhead: +12.8 MB (tokenizer.json 12.8 MB, template 9 KB, configs < 1 KB) = 0.11 % of e1.

## p21/quant2 (2026-11-23) - quantization quality at equal bytes (interim)
Proxy ablation on held-out text (see docs/QUANTIZATION.md "p21/quant2"): act-order+damping per tensor class cuts held-out layer error 6-12%, 128 calibration sequences beat 32 by 12-25%, g64 costs less per unit error than 4-bit upgrades, down_proj GPTQ overfits. Tools committed; full e1b1/e0b1 runs started, no model-level number measured yet, README claims unchanged. Branch p21/quant2.

Update: e1b1 (10.40 GiB) measured: ppl 8x512 9.112, KL to Q8_0 0.0750, top-1 89.3%, decode 57.5 / MTP-3 137.2 (code); see docs/QUANTIZATION.md.

## p24/multiseq (2026-11-24) - concurrent requests in qv_server (phase 1: interleaved sequences)
- Engine API: `EngineOptions::n_seq`, `Engine::n_seq()/select_seq(i)`; CUDA Inkling, CUDA Qwen, CPU Inkling, CPU Qwen and the mock engine hold N sequences (KV incl. int4 KV, GDN/conv/sconv/loop state, logits, token/position, CUDA graphs per sequence) over one copy of the weights; MTP stays single-sequence (the server then runs requests one at a time).
- Server: slot i == sequence i; FIFO admission (slot / exclusive / model-user waiters), FIFO engine lock per engine op, host sampling outside the lock, per-slot `cache_prompt`, slot choice by longest cached prefix, `--ctx-size` split per slot, prefill chunk 1024 while others run, `/slots` and `/metrics` real; `--reasoning-budget N` + `--reasoning-budget-message`; length inside the thinking block documented (same as llama-server: content "" + reasoning_content).
- Measured: aggregate +19 % (2-4 clients, short prompts), -3..4 % with 4K prompts, TTFT of a short request during a 30K prefill 12.4 s -> 1.65 s; bit-exact vs running alone (Inkling int4/fp16 KV, Qwen int4 KV, CPU). docs/PERFORMANCE.md "Concurrent requests".
- Phase 2 (batched decode over several sequences) NOT done; no speedup claimed beyond the above.
- Tests: tests/test_multiseq.cpp (exactness), tests/test_server_compat.py (+concurrent interleave / per-slot cache / reasoning budget; ctx expectations changed to the per-slot value), tests/test_server_concurrent_real.py (4 simultaneous real chats equal to sequential greedy). Known pre-existing flake: test_models_preset_routing (first request of a freshly loaded mock preset sometimes returns garbage tokens; also on 98324e0: 4 of 40 runs).
- NOT RUN / open: CPU multi-sequence tests (Inkling CPU: 18-step interleaved case bit-identical, 6-step case with a late-joining 104-token prompt differed from the alone run, unresolved; CPU Qwen untested); Qwen server-level concurrency smoke; llama.cpp comparison. Treat CPU `--parallel N` concurrency as experimental.

## p26/ink-fuse (2026-11-24) - Inkling decode latency (branch p26/ink-fuse, not merged)
- Plain decode (int4 KV, V100): ctx 25 118.65 -> 135.4 tok/s (+14.1 %), 4K 111.4 -> 126.6, 32K synthetic 99.75 -> 111.6; all steps bit-exact (per-step logits hash + ids, 4 golden prompts x 64 steps incl. 4K, test_ink_mtp depth 3/7, test_multiseq int4/fp16). The modeled 206 tok/s was NOT reached (~65 %).
- Finding: inter-kernel gaps in the graph are 0.47 us (3 %); the time is per-kernel dependent latency chains. Wins: load hoisting, finer GEMV decomposition, removing a hidden local copy of a kernel argument struct, shared barriers, rank-based top-8, finish folded into attn_res. Ticket-merged kernels, side-stream branches, L2 prefetch and forced occupancy were slower (docs/INKLING_ENGINE.md).
- MTP: absolute speed +1.7 % (code) but the gain over plain shrank 1.14x -> 1.02x because the verify rows kernels were not restructured (open item).
- New: tests/test_ink_exact.cpp (per-step logits hash regression).

## p29/spec-speed (2026-11-27) - Inkling speculative sampling at T > 0
- Finding: Inkling had no `decode_speculative`, so `qv_server --mtp` at the default T = 1.0 ran the plain host-sampled path (about 105 tok/s, host sampling about 2 ms/token). Implemented lossless
  speculative sampling (point-mass drafter, `spec_round_point`), GPU candidate extraction, adaptive width / plain-step policy, constrained MTP (commits 6876acc, 34ddc51, 6da51d9).
- Verify kernels: expert-grouped w2 (`k_w2_g`) and 6-shuffle `warp_reduce4` reductions, bit-identical, about 1.1 ms off the R = 4 verify (17.8 -> 16.7 ms). Verify is bandwidth-bound on the expert union
  (about 28-32 experts, w13_u about 0.87 TB/s; MoE floor about 8.7 ms).
- Result through qv_server, T = 1.0, ctx 25, decode tok/s: code 104 -> 130, prose 104 -> 131, tool (--mtp-grammar) 109 -> 171 (see PERFORMANCE.md). 200 is not reached for code or prose.
- Gates: T = 0 spec == plain greedy, T = 1 state replay bit-identical, CPU chi-square of the algebra (tests/test_spec_point), test_ink_mtp depth 3/7, test_multiseq, test_ink_exact.
