// CUDA backend: Inkling-2-Mini (inkling2_loop) engine. Decode step = one CUDA graph (no host sync, expert ids stay on the device).
// Spec: docs/INKLING_SPEC.md. Design/measurements: docs/INKLING_ENGINE.md.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../backends/cuda/common.cuh"
#include <cublas_v2.h>

#include "../backends/cuda/inkling/ink_ops.h"
#include "../backends/cuda/inkling/ink_prefill.h"
#include "../backends/cuda/kernels/dequant.h"
#include "../backends/cuda/kernels/elementwise.h"
#include "../backends/cuda/kernels/gemv.h"
#include "qv/engine.h"

#define QV_CUBLAS(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); std::abort(); } } while (0)

namespace qv {
using namespace qv::cuda;
namespace ik = qv::cuda::ink;

namespace {
static bool g_pfdump = std::getenv("QV_INK_PF_DUMP") != nullptr;
static int g_dbgapp = std::getenv("QV_INK_DBGAPP") ? std::atoi(std::getenv("QV_INK_DBGAPP")) : -1;
static bool g_trace = std::getenv("QV_INK_TRACE") != nullptr;
#define STAGE(name) do { if (g_trace) { cudaError_t e_ = cudaStreamSynchronize(stream_); std::fprintf(stderr, "app %d L%d p%d %s -> %s\n", A.idx, A.L, A.p, name, cudaGetErrorString(e_)); } } while (0)

struct Ctrl { const __half *down = nullptr, *forget = nullptr, *gain = nullptr; };
struct LayerW {
    bool glob = false;
    QMat wq, wk, wv, wo, wg;
    DMat16 wr, sink, kproj, vproj;
    const float *qscale, *relscale, *relproj, *kw, *vw;
    const float *nw[3];  // local: attn_norm in [0]; global: q/k/v_input_norm
    const float *mlp_norm, *attn_sconv, *mlp_sconv;
    DMat16 router;
    const float* router_bias;
    ik::ExpertW w13, w2;
    const float *attn_q, *mlp_q;
    Ctrl ctrl[4];  // attn(q), k, v, mlp
};
struct App { int L, p, idx, kvslot; };

class InklingEngine final : public Engine {
   public:
    InklingEngine(const ModelFile& mf, const EngineOptions& opt) : opt_(opt) {
        kv4_ = opt.kv_mode == 1;
        cfg_ = mf.cfg;
        const InklingConfig& k = cfg_.ink;
        if (cfg_.hidden != ik::H || cfg_.n_heads != ik::NH || cfg_.n_kv_heads != ik::NKV || cfg_.head_dim != ik::HD || k.n_experts != ik::NE ||
            k.topk != ik::TOPK || k.expert_ff != ik::FF || k.window != 1024 || k.rel_extent != 1024 || k.d_rel != 16 || k.sconv_kernel != 4 ||
            k.attn_res_block != 6 || k.n_shared_experts != 0 || !k.w13_interleaved)
            throw std::runtime_error("unsupported inkling configuration");
        if (opt_.n_seq < 1) opt_.n_seq = 1;
        if (opt_.n_seq > 1 && opt.mtp) throw std::runtime_error("MTP supports one sequence (use n_seq = 1)");
        QV_CUDA_CHECK(cudaStreamCreate(&stream_));
        load_weights(mf);
        bind(mf);
        alloc_state();
        if (opt.mtp && !opt.mtp_file.empty()) {
            load_mtp(opt.mtp_file);
            mtp_ = true;
            alloc_mtp();
            int ones[16];
            for (int i = 0; i < 16; ++i) ones[i] = i + 1;
            d_dummy_i_ = dalloc<int>(16);
            QV_CUDA_CHECK(cudaMemcpy(d_dummy_i_, ones, sizeof ones, cudaMemcpyHostToDevice));
        }
        for (int i = opt_.n_seq - 1; i >= 0; --i) { select_seq(i); reset(); }  // ends on sequence 0
    }
    ~InklingEngine() override {
        if (cublas_) cublasDestroy(cublas_);
        seqs_[cur_seq_] = snapshot();
        for (SeqSt& q : seqs_) {
            if (q.g_fwd) cudaGraphExecDestroy(q.g_fwd);
            if (q.g_dec) cudaGraphExecDestroy(q.g_dec);
        }
        g_fwd_ = g_dec_ = nullptr;
        if (h_tp_) cudaFreeHost(h_tp_);
        if (h_stage_) cudaFreeHost(h_stage_);
        for (int i = 0; i < 9; ++i) if (g_mtp_[i]) cudaGraphExecDestroy(g_mtp_[i]);
        for (void* p : allocs_) cudaFree(p);
        cudaFree(weights_);
        if (mw_) cudaFree(mw_);
        cudaStreamDestroy(stream_);
    }

    void step(int token, int pos) override {
        if (pos >= opt_.max_ctx) throw std::runtime_error("context overflow");
        if (tp_n_ == TP_SLOTS) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); tp_n_ = 0; }
        int* tp = h_tp_ + 2 * tp_n_++;
        tp[0] = token;
        tp[1] = pos;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, tp, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        if (opt_.dump_hidden) {
            forward();
            collect_dump();
        } else {
            if (!g_fwd_) g_fwd_ = capture(false);
            QV_CUDA_CHECK(cudaGraphLaunch(g_fwd_, stream_));
            if (mtp_) mtp_ingest(featv_, 1, 0, d_pos_, nullptr, nullptr);
        }
    }

    void prefill(const int* tokens, int T, int pos0) override { prefill_impl(tokens, nullptr, nullptr, T, pos0); }
    // Vision: rows with mask[t] != 0 take their residual-stream input from `emb` (packed, one H-row per masked position, in order)
    // instead of embed + embed_norm (docs/INKLING_VISION.md section 7).
    void prefill_embeds(const int* tokens, const float* emb, const unsigned char* mask, int T, int pos0) override { prefill_impl(tokens, emb, mask, T, pos0); }
    void prefill_impl(const int* tokens, const float* emb, const unsigned char* mask, int T, int pos0) {
        const bool batched = !opt_.dump_hidden && std::getenv("QV_INK_STEP_PREFILL") == nullptr;
        if (pos0 + T > opt_.max_ctx) throw std::runtime_error("context overflow");
        int off = 0;
        const float* e = emb;
        while (off < T) {
            const int n = std::min(T - off, opt_.prefill_chunk);
            const unsigned char* m = mask ? mask + off : nullptr;
            if (!batched || n < 4) {
                for (int i = 0; i < n; ++i) {
                    if (m && m[i]) { step_override(tokens[off + i], pos0 + off + i, e); e += ik::H; }
                    else step(tokens[off + i], pos0 + off + i);
                }
            } else {
                prefill_chunk(tokens + off, n, pos0 + off, m, e);
                if (m) for (int i = 0; i < n; ++i) if (m[i]) e += ik::H;
            }
            off += n;
        }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    // single token whose embedding row is injected (uncaptured forward; used for <4-token chunks / debug paths)
    void step_override(int token, int pos, const float* row) {
        if (pos >= opt_.max_ctx) throw std::runtime_error("context overflow");
        if (tp_n_ == TP_SLOTS) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); tp_n_ = 0; }
        int* tp = h_tp_ + 2 * tp_n_++;
        tp[0] = token;
        tp[1] = pos;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, tp, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        if (!d_ovr_) { QV_CUDA_CHECK(cudaMalloc(&d_ovr_, ik::H * 4)); }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        QV_CUDA_CHECK(cudaMemcpy(d_ovr_, row, ik::H * 4, cudaMemcpyHostToDevice));
        ovr_on_ = true;
        forward();
        ovr_on_ = false;
        if (opt_.dump_hidden) collect_dump();
    }

    void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool use_graph) override {
        if (start_pos + n > opt_.max_ctx) throw std::runtime_error("context overflow");
        if (tp_n_ == TP_SLOTS) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); tp_n_ = 0; }
        int* tp = h_tp_ + 2 * tp_n_++;
        tp[0] = first;
        tp[1] = start_pos;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, tp, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        if (mtp_ && opt_.mtp) {
            QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
            mtp_decode(first, start_pos, n, out);
            return;
        }
        if (use_graph && !g_dec_) g_dec_ = capture(true);
        for (int i = 0; i < n; ++i) {
            if (use_graph) {
                QV_CUDA_CHECK(cudaGraphLaunch(g_dec_, stream_));
            } else {
                forward();
                argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_tok_, d_pos_, d_gen_, stream_);
            }
        }
        out.resize(n);
        QV_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_gen_ + start_pos, n * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    int argmax() override {
        argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_arg_, nullptr, nullptr, stream_);
        int r;
        QV_CUDA_CHECK(cudaMemcpyAsync(&r, d_arg_, 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return r;
    }
    void logits(std::vector<float>& out) override {
        out.resize(cfg_.ink.unpadded_vocab);
        QV_CUDA_CHECK(cudaMemcpyAsync(out.data(), logits_, out.size() * 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    void reset() override {
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        QV_CUDA_CHECK(cudaMemset(cst_k_, 0, 48 * 3 * 1024 * 4));
        QV_CUDA_CHECK(cudaMemset(cst_v_, 0, 48 * 3 * 1024 * 4));
        QV_CUDA_CHECK(cudaMemset(cst_a_, 0, (size_t)48 * 3 * ik::H * 4));
        QV_CUDA_CHECK(cudaMemset(cst_m_, 0, (size_t)48 * 3 * ik::H * 4));
        QV_CUDA_CHECK(cudaMemset(ticket_, 0, 4));
        QV_CUDA_CHECK(cudaDeviceSynchronize());
    }
    bool has_checkpoint() const override { return true; }
    // MTP drafter state that is NOT append-only: the two local-layer context rings (K and V, fp16, 1024 slots) and the recurrent head feature.
    // The drafter's global-layer context K/V is append-only like the main global KV (kept in VRAM for L1, saved by save_kv_prefix for L2).
    size_t mtp_ring_bytes() const { return mtp_ ? 4 * (size_t)ik::NKV * 1024 * ik::HD * sizeof(__half) + (size_t)ik::H * sizeof(float) : 0; }
    size_t mtp_gkv_bytes(int n_tokens) const { return mtp_ ? 2 * (size_t)ik::NKV * (size_t)n_tokens * ik::HD * sizeof(__half) : 0; }
    bool save_checkpoint(std::vector<uint8_t>& out) override {
        struct InkCkptHdr {
            uint32_t magic = 0x494B4331u;
            uint64_t lkv_b = 0, cstk_b = 0, csta_b = 0, logits_b = 0;
        };
        flush_fin();
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        InkCkptHdr h;
        h.lkv_b = (size_t)n_lslots_ * ik::NKV * 1024 * ik::HD * sizeof(__half);
        h.cstk_b = (size_t)48 * 3 * 1024 * sizeof(float);
        h.csta_b = (size_t)48 * 3 * ik::H * sizeof(float);
        h.logits_b = (size_t)cfg_.vocab * sizeof(float);
        const size_t tot = sizeof(InkCkptHdr) + 2 * h.lkv_b + 2 * h.cstk_b + 2 * h.csta_b + h.logits_b + mtp_ring_bytes();
        out.resize(tot);
        uint8_t* p = out.data();
        std::memcpy(p, &h, sizeof(h));
        p += sizeof(h);
        auto get = [&](const void* src, size_t b) {
            if (b > 0) QV_CUDA_CHECK(cudaMemcpy(p, src, b, cudaMemcpyDeviceToHost));
            p += b;
        };
        get(lk_, h.lkv_b);
        get(lv_, h.lkv_b);
        get(cst_k_, h.cstk_b);
        get(cst_v_, h.cstk_b);
        get(cst_a_, h.csta_b);
        get(cst_m_, h.csta_b);
        get(logits_, h.logits_b);
        if (mtp_) {
            const size_t rb = (size_t)ik::NKV * 1024 * ik::HD * sizeof(__half);
            get(dkc_[0], rb); get(dvc_[0], rb); get(dkc_[2], rb); get(dvc_[2], rb);
            get(f_rnn_, (size_t)ik::H * sizeof(float));
        }
        return true;
    }
    bool restore_checkpoint(const std::vector<uint8_t>& in) override {
        struct InkCkptHdr {
            uint32_t magic = 0x494B4331u;
            uint64_t lkv_b = 0, cstk_b = 0, csta_b = 0, logits_b = 0;
        };
        if (in.size() < sizeof(InkCkptHdr)) return false;
        InkCkptHdr h;
        std::memcpy(&h, in.data(), sizeof(h));
        if (h.magic != 0x494B4331u) return false;
        const size_t lkv_b = (size_t)n_lslots_ * ik::NKV * 1024 * ik::HD * sizeof(__half);
        const size_t cstk_b = (size_t)48 * 3 * 1024 * sizeof(float);
        const size_t csta_b = (size_t)48 * 3 * ik::H * sizeof(float);
        const size_t logits_b = (size_t)cfg_.vocab * sizeof(float);
        if (h.lkv_b != lkv_b || h.cstk_b != cstk_b || h.csta_b != csta_b || h.logits_b != logits_b) return false;
        const size_t tot = sizeof(InkCkptHdr) + 2 * lkv_b + 2 * cstk_b + 2 * csta_b + logits_b + mtp_ring_bytes();
        if (in.size() != tot) return false;
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        pend_.on = false;
        const uint8_t* p = in.data() + sizeof(InkCkptHdr);
        auto put = [&](void* dst, size_t b) {
            if (b > 0) QV_CUDA_CHECK(cudaMemcpy(dst, p, b, cudaMemcpyHostToDevice));
            p += b;
        };
        put(lk_, lkv_b);
        put(lv_, lkv_b);
        put(cst_k_, cstk_b);
        put(cst_v_, cstk_b);
        put(cst_a_, csta_b);
        put(cst_m_, csta_b);
        put(logits_, logits_b);
        if (mtp_) {
            const size_t rb = (size_t)ik::NKV * 1024 * ik::HD * sizeof(__half);
            put(dkc_[0], rb); put(dvc_[0], rb); put(dkc_[2], rb); put(dvc_[2], rb);
            put(f_rnn_, (size_t)ik::H * sizeof(float));
        }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }
    struct InkKvHdr {
        uint32_t magic = 0x494B4B31u;  // "IKK1"
        int32_t kv4 = 0, n_gslots = 0, n_tokens = 0;
        uint64_t payload_bytes = 0;
    };
    size_t checkpoint_bytes() const override {
        const size_t lkv_b = (size_t)n_lslots_ * ik::NKV * 1024 * ik::HD * sizeof(__half);
        const size_t cstk_b = (size_t)48 * 3 * 1024 * sizeof(float);
        const size_t csta_b = (size_t)48 * 3 * ik::H * sizeof(float);
        const size_t logits_b = (size_t)cfg_.vocab * sizeof(float);
        return 40 + 2 * lkv_b + 2 * cstk_b + 2 * csta_b + logits_b + mtp_ring_bytes();  // sizeof(InkCkptHdr) == 40
    }
    bool has_kv_checkpoint() const override { return true; }
    size_t kv_checkpoint_bytes(int n_tokens) const override {
        if (n_tokens < 0 || n_tokens > opt_.max_ctx) return 0;
        const size_t rows = (size_t)n_gslots_ * ik::NKV;
        const size_t payload = kv4_ ? rows * (size_t)n_tokens * (64 + 64 + 2 * sizeof(__half2) + 2 * sizeof(__half2))
                                    : 2 * rows * (size_t)n_tokens * ik::HD * sizeof(__half);
        return sizeof(InkKvHdr) + payload + mtp_gkv_bytes(n_tokens);
    }
    bool save_kv_prefix(std::vector<uint8_t>& out, int n_tokens) override {
        if (n_tokens < 0 || n_tokens > opt_.max_ctx) return false;
        flush_fin();
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        InkKvHdr h;
        h.kv4 = kv4_ ? 1 : 0;
        h.n_gslots = n_gslots_;
        h.n_tokens = n_tokens;
        const size_t rows = (size_t)n_gslots_ * ik::NKV;
        h.payload_bytes = kv4_ ? rows * (size_t)n_tokens * (64 + 64 + 2 * sizeof(__half2) + 2 * sizeof(__half2))
                               : 2 * rows * (size_t)n_tokens * ik::HD * sizeof(__half);
        h.payload_bytes += mtp_gkv_bytes(n_tokens);  // drafter global-layer context K/V prefix after the main KV
        out.resize(sizeof(InkKvHdr) + h.payload_bytes);
        std::memcpy(out.data(), &h, sizeof(h));
        // device -> pinned staging buffer (a copy straight into the pageable vector ran at ~1 GB/s: page faults + the driver's bounce buffer), then one memcpy
        if (stage_bytes_ < h.payload_bytes) {
            if (h_stage_) cudaFreeHost(h_stage_);
            h_stage_ = nullptr;
            QV_CUDA_CHECK(cudaMallocHost(&h_stage_, h.payload_bytes));
            stage_bytes_ = h.payload_bytes;
        }
        uint8_t* const dst = out.data() + sizeof(h);
        uint8_t* p = h_stage_;
        if (n_tokens > 0 && rows > 0) {
            if (kv4_) {
                const size_t q_w = (size_t)n_tokens * 64, q_sp = (size_t)opt_.max_ctx * 64, q_b = rows * q_w;
                const size_t s_w = (size_t)n_tokens * 2 * sizeof(__half2), s_sp = (size_t)opt_.max_ctx * 2 * sizeof(__half2), s_b = rows * s_w;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p, q_w, gkq_, q_sp, q_w, rows, cudaMemcpyDeviceToHost, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p + q_b, q_w, gvq_, q_sp, q_w, rows, cudaMemcpyDeviceToHost, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p + 2 * q_b, s_w, gks_, s_sp, s_w, rows, cudaMemcpyDeviceToHost, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p + 2 * q_b + s_b, s_w, gvs_, s_sp, s_w, rows, cudaMemcpyDeviceToHost, stream_));
            } else {
                const size_t w = (size_t)n_tokens * ik::HD * sizeof(__half), sp = (size_t)opt_.max_ctx * ik::HD * sizeof(__half), half_b = rows * w;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p, w, gk_, sp, w, rows, cudaMemcpyDeviceToHost, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p + half_b, w, gv_, sp, w, rows, cudaMemcpyDeviceToHost, stream_));
            }
        }
        if (mtp_ && n_tokens > 0) {  // drafter global-layer context K/V (layer 1), [NKV][max_ctx][HD] fp16, first n_tokens rows
            const size_t main_b = h.payload_bytes - mtp_gkv_bytes(n_tokens);
            const size_t w = (size_t)n_tokens * ik::HD * sizeof(__half), sp = (size_t)opt_.max_ctx * ik::HD * sizeof(__half);
            QV_CUDA_CHECK(cudaMemcpy2DAsync(p + main_b, w, dkc_[1], sp, w, ik::NKV, cudaMemcpyDeviceToHost, stream_));
            QV_CUDA_CHECK(cudaMemcpy2DAsync(p + main_b + (size_t)ik::NKV * w, w, dvc_[1], sp, w, ik::NKV, cudaMemcpyDeviceToHost, stream_));
        }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (h.payload_bytes) std::memcpy(dst, h_stage_, h.payload_bytes);
        return true;
    }
    uint8_t* h_stage_ = nullptr;
    size_t stage_bytes_ = 0;
    bool restore_kv_prefix(const std::vector<uint8_t>& in) override {
        if (in.size() < sizeof(InkKvHdr)) return false;
        InkKvHdr h;
        std::memcpy(&h, in.data(), sizeof(h));
        if (h.magic != 0x494B4B31u) return false;
        if (h.kv4 != (kv4_ ? 1 : 0) || h.n_gslots != n_gslots_ || h.n_tokens < 0 || h.n_tokens > opt_.max_ctx) return false;
        const size_t rows = (size_t)n_gslots_ * ik::NKV;
        const size_t expected_payload = kv4_ ? rows * (size_t)h.n_tokens * (64 + 64 + 2 * sizeof(__half2) + 2 * sizeof(__half2))
                                             : 2 * rows * (size_t)h.n_tokens * ik::HD * sizeof(__half);
        const size_t extra = mtp_gkv_bytes(h.n_tokens);
        if (h.payload_bytes != expected_payload + extra || in.size() != sizeof(InkKvHdr) + expected_payload + extra) return false;
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        pend_.on = false;
        const uint8_t* p = in.data() + sizeof(InkKvHdr);
        if (h.n_tokens > 0 && rows > 0) {
            if (kv4_) {
                const size_t q_w = (size_t)h.n_tokens * 64, q_dp = (size_t)opt_.max_ctx * 64, q_b = rows * q_w;
                const size_t s_w = (size_t)h.n_tokens * 2 * sizeof(__half2), s_dp = (size_t)opt_.max_ctx * 2 * sizeof(__half2), s_b = rows * s_w;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(gkq_, q_dp, p, q_w, q_w, rows, cudaMemcpyHostToDevice, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(gvq_, q_dp, p + q_b, q_w, q_w, rows, cudaMemcpyHostToDevice, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(gks_, s_dp, p + 2 * q_b, s_w, s_w, rows, cudaMemcpyHostToDevice, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(gvs_, s_dp, p + 2 * q_b + s_b, s_w, s_w, rows, cudaMemcpyHostToDevice, stream_));
            } else {
                const size_t w = (size_t)h.n_tokens * ik::HD * sizeof(__half), dp = (size_t)opt_.max_ctx * ik::HD * sizeof(__half), half_b = rows * w;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(gk_, dp, p, w, w, rows, cudaMemcpyHostToDevice, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(gv_, dp, p + half_b, w, w, rows, cudaMemcpyHostToDevice, stream_));
            }
        }
        if (mtp_ && h.n_tokens > 0) {
            const size_t w = (size_t)h.n_tokens * ik::HD * sizeof(__half), dp = (size_t)opt_.max_ctx * ik::HD * sizeof(__half);
            const uint8_t* q = in.data() + sizeof(InkKvHdr) + expected_payload;
            QV_CUDA_CHECK(cudaMemcpy2DAsync(dkc_[1], dp, q, w, w, ik::NKV, cudaMemcpyHostToDevice, stream_));
            QV_CUDA_CHECK(cudaMemcpy2DAsync(dvc_[1], dp, q + (size_t)ik::NKV * w, w, w, ik::NKV, cudaMemcpyHostToDevice, stream_));
        }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }
    const std::vector<float>& hidden_dump() const override { return hidden_; }
    const std::vector<float>& logits_dump() const override { return logits_dump_; }
    const std::vector<int>& aux_dump() const override { return rids_; }
    void set_mtp_enabled(bool e) override { opt_.mtp = e; }
    void mtp_stats(MtpStats& st) const override { st = mstats_; }
    void sync() override { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    size_t device_bytes_weights() const override { return weights_bytes_; }
    size_t device_bytes_state() const override { return state_bytes_; }

    int n_seq() const override { return opt_.n_seq; }
    // Sequence switch: the members below (KV caches, conv states, token/position/logits buffers, CUDA graphs) are views onto the current
    // sequence; everything else (weights, scratch, prefill buffers) is shared and fully rewritten by every step/chunk.
    void select_seq(int i) override {
        if (i < 0 || i >= opt_.n_seq) throw std::runtime_error("sequence index out of range");
        if (i == cur_seq_) return;
        seqs_[cur_seq_] = snapshot();
        const SeqSt& q = seqs_[i];
        lk_ = q.lk; lv_ = q.lv; gk_ = q.gk; gv_ = q.gv; gkq_ = q.gkq; gvq_ = q.gvq; gks_ = q.gks; gvs_ = q.gvs;
        cst_k_ = q.cst_k; cst_v_ = q.cst_v; cst_a_ = q.cst_a; cst_m_ = q.cst_m;
        d_tok_ = q.d_tok; d_pos_ = q.d_tok + 1; d_gen_ = q.d_gen; logits_ = q.logits; g_fwd_ = q.g_fwd; g_dec_ = q.g_dec;
        cur_seq_ = i;
    }

   private:
    struct SeqSt {
        __half *lk = nullptr, *lv = nullptr, *gk = nullptr, *gv = nullptr;
        uint8_t *gkq = nullptr, *gvq = nullptr;
        __half2 *gks = nullptr, *gvs = nullptr;
        float *cst_k = nullptr, *cst_v = nullptr, *cst_a = nullptr, *cst_m = nullptr, *logits = nullptr;
        int *d_tok = nullptr, *d_gen = nullptr;
        cudaGraphExec_t g_fwd = nullptr, g_dec = nullptr;
    };
    std::vector<SeqSt> seqs_;
    int cur_seq_ = 0;
    size_t seq_bytes_ = 0;
    SeqSt snapshot() const {
        SeqSt q;
        q.lk = lk_; q.lv = lv_; q.gk = gk_; q.gv = gv_; q.gkq = gkq_; q.gvq = gvq_; q.gks = gks_; q.gvs = gvs_;
        q.cst_k = cst_k_; q.cst_v = cst_v_; q.cst_a = cst_a_; q.cst_m = cst_m_; q.logits = logits_;
        q.d_tok = d_tok_; q.d_gen = d_gen_; q.g_fwd = g_fwd_; q.g_dec = g_dec_;
        return q;
    }
    static constexpr int TP_SLOTS = 4096;
    static constexpr int NSPLIT_L = 8, NSPLIT_G = 64;
    EngineOptions opt_;
    cudaStream_t stream_{};
    uint8_t* weights_ = nullptr;
    size_t weights_bytes_ = 0, state_bytes_ = 0;
    std::vector<void*> allocs_;
    cudaGraphExec_t g_fwd_ = nullptr, g_dec_ = nullptr;
    int *h_tp_ = nullptr, tp_n_ = 0;
    std::vector<LayerW> lw_;
    std::vector<App> apps_;
    QMat embed_, unembed_;
    float* d_ovr_ = nullptr;  // vision: injected embedding row for step_override
    bool ovr_on_ = false;
    const float *embed_norm_, *final_norm_, *out_query_, *msgq_ = nullptr;
    // state
    __half *lk_, *lv_, *gk_ = nullptr, *gv_ = nullptr;
    uint8_t *gkq_ = nullptr, *gvq_ = nullptr; __half2 *gks_ = nullptr, *gvs_ = nullptr;
    bool kv4_ = false;
    size_t grow(const App& A) const { return (size_t)A.kvslot * ik::NKV * opt_.max_ctx; }
    float *cst_k_, *cst_v_, *cst_a_, *cst_m_;
    // scratch
    float *h_, *pre_h_, *blocks_, *res_, *araw_, *ymoe_, *proj_, *xn_f_, *u_, *gs_, *gg_, *xm_f_, *mraw_, *mg_, *msgs_, *rlog_, *wts_, *qs_, *rl_,
        *part_, *logits_;
    int *ids_, *ticket_, *d_tok_, *d_pos_, *d_gen_, *d_arg_;
    Q8Buf q8x_, q8m_, q8act_, q8o_;
    std::vector<float> hidden_, logits_dump_;
    std::vector<int> rids_;
    float *dump_d_ = nullptr;
    int* rid_d_ = nullptr;

    template <class T> T* dalloc(size_t n) {
        void* p = nullptr;
        QV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        QV_CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
        allocs_.push_back(p);
        state_bytes_ += n * sizeof(T);
        return (T*)p;
    }
    Q8Buf q8alloc(int chunks) {
        Q8Buf b;
        b.q = dalloc<int>((size_t)chunks * 8);
        b.d = dalloc<float>(chunks);
        b.sum = dalloc<float>(chunks);
        b.qsum = dalloc<int>(chunks);
        return b;
    }

    void load_weights(const ModelFile& mf) {
        weights_bytes_ = mf.bin_size;
        QV_CUDA_CHECK(cudaMalloc(&weights_, weights_bytes_));
        FILE* f = std::fopen(mf.bin_path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + mf.bin_path);
        const size_t CH = 256u << 20;
        void* pinned;
        QV_CUDA_CHECK(cudaMallocHost(&pinned, CH));
        size_t off = 0;
        while (off < weights_bytes_) {
            size_t n = std::fread(pinned, 1, std::min(CH, weights_bytes_ - off), f);
            if (n == 0) throw std::runtime_error("short read on weights");
            QV_CUDA_CHECK(cudaMemcpy(weights_ + off, pinned, n, cudaMemcpyHostToDevice));
            off += n;
        }
        cudaFreeHost(pinned);
        std::fclose(f);
    }
    QMat qmat(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::Quant || !t.meta8) throw std::runtime_error(n + " is not a meta8 quantized tensor");
        if (t.group != 64 && t.group != 128) throw std::runtime_error("group must be 64 or 128");
        QMat m;
        m.sm8 = (const uint16_t*)(weights_ + t.off_sm);
        m.rs = (const __half*)(weights_ + t.off_rs);
        m.q = (const uint32_t*)(weights_ + t.off_q);
        m.zlo = -0.25f * (float)((1 << t.bits) - 1);
        m.zstep = (float)(1.5 * (double)((1 << t.bits) - 1) / 255.0);
        m.N = (int)t.shape[t.shape.size() - 2];
        m.K = (int)t.shape[t.shape.size() - 1];
        m.bits = t.bits;
        m.gshift = t.group == 128 ? 7 : 6;
        return m;
    }
    ik::ExpertW expert(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::Quant || !t.meta8 || !t.stacked()) throw std::runtime_error(n + " is not a stacked meta8 tensor");
        ik::ExpertW e;
        e.q = (const uint32_t*)(weights_ + t.off_q);
        e.sm8 = (const uint16_t*)(weights_ + t.off_sm);
        e.rs = (const __half*)(weights_ + t.off_rs);
        e.N = (int)t.shape[1];
        e.K = (int)t.shape[2];
        e.bits = t.bits;
        e.gshift = t.group == 128 ? 7 : 6;
        e.zlo = -0.25f * (float)((1 << t.bits) - 1);
        e.zstep = (float)(1.5 * (double)((1 << t.bits) - 1) / 255.0);
        return e;
    }
    const float* f32(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F32) throw std::runtime_error(n + " is not f32");
        return (const float*)(weights_ + t.off);
    }
    DMat16 f16(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F16) throw std::runtime_error(n + " is not f16");
        return DMat16{(const __half*)(weights_ + t.off), (int)t.shape[0], (int)t.shape[1]};
    }

    void bind(const ModelFile& mf) {
        const InklingConfig& k = cfg_.ink;
        const std::string P = "model.llm.";
        embed_ = qmat(mf, P + "embed.weight");
        unembed_ = qmat(mf, P + "unembed.weight");
        embed_norm_ = f32(mf, P + "embed_norm.weight");
        final_norm_ = f32(mf, P + "norm.weight");
        out_query_ = f32(mf, P + "attn_res.output_query");
        for (int L = 0; L < cfg_.n_layers; ++L) {
            const std::string l = P + "layers." + std::to_string(L) + ".", a = l + "attn.";
            LayerW W;
            W.glob = k.is_global(L);
            W.wq = qmat(mf, a + "wq_du.weight");
            W.wk = qmat(mf, a + "wk_dv.weight");
            W.wv = qmat(mf, a + "wv_dv.weight");
            W.wo = qmat(mf, a + "wo_ud.weight");
            W.wg = qmat(mf, a + "gate_attn.weight");
            W.wr = f16(mf, a + "wr_du.weight");
            W.sink = f16(mf, a + "attn_sink_proj.weight");
            W.qscale = f32(mf, a + "q_l2_clip.scale");
            W.relscale = f32(mf, a + "rel_l2_clip.scale");
            W.relproj = f32(mf, a + "rel_logits_proj.proj");
            if (W.glob) {
                W.kw = f32(mf, a + "k_sconv.static_w");
                W.vw = f32(mf, a + "v_sconv.static_w");
                W.kproj = f16(mf, a + "k_sconv.weight_proj");
                W.vproj = f16(mf, a + "v_sconv.weight_proj");
                W.nw[0] = f32(mf, l + "q_input_norm.weight");
                W.nw[1] = f32(mf, l + "k_input_norm.weight");
                W.nw[2] = f32(mf, l + "v_input_norm.weight");
            } else {
                W.kw = f32(mf, a + "k_sconv.weight");
                W.vw = f32(mf, a + "v_sconv.weight");
                W.nw[0] = W.nw[1] = W.nw[2] = f32(mf, l + "attn_norm.weight");
            }
            W.mlp_norm = f32(mf, l + "mlp_norm.weight");
            W.attn_sconv = f32(mf, l + "attn_sconv.weight");
            W.mlp_sconv = f32(mf, l + "mlp_sconv.weight");
            W.router = f16(mf, l + "mlp.gate.weight");
            W.router_bias = f32(mf, l + "mlp.gate.bias");
            W.w13 = expert(mf, l + "mlp.experts.w13_weight");
            W.w2 = expert(mf, l + "mlp.experts.w2_weight");
            W.attn_q = L >= 1 ? f32(mf, P + "attn_res.attn_queries." + std::to_string(L - 1)) : nullptr;
            W.mlp_q = f32(mf, P + "attn_res.mlp_queries." + std::to_string(L));
            if (k.in_loop(L)) {
                static const char* nm[4] = {"attn", "k", "v", "mlp"};
                for (int s = 0; s < 4; ++s) {
                    const std::string c = l + "controllers." + nm[s] + ".";
                    W.ctrl[s].down = f16(mf, c + "down_weight").w;
                    W.ctrl[s].forget = f16(mf, c + "forget_weight").w;
                    W.ctrl[s].gain = f16(mf, c + "gain_weight").w;
                }
            }
            lw_.push_back(W);
        }
        // schedule: 0..5, loop x3, 18..23 ; KV slots
        int app = 0, lslot = 0, gslot = 0;
        std::vector<std::vector<int>> slots(cfg_.n_layers);
        auto add = [&](int L, int p) {
            int ks;
            if (k.is_global(L)) ks = gslot++; else ks = lslot++;
            apps_.push_back(App{L, p, app++, ks});
        };
        for (int L = 0; L < k.loop_start; ++L) add(L, 0);
        for (int p = 0; p < k.loop_passes; ++p)
            for (int L = k.loop_start; L < k.loop_end; ++L) add(L, p);
        for (int L = k.loop_end; L < cfg_.n_layers; ++L) add(L, 0);
        n_apps_ = app;
        n_lslots_ = lslot;
        n_gslots_ = gslot;
        if (n_apps_ != 48 || n_lslots_ != 40 || n_gslots_ != 8) throw std::runtime_error("unexpected application count");
        // merged message queries [12 loop layers][4 = query,key,value,mlp][H]
        std::vector<float> mq((size_t)12 * 4 * ik::H);
        std::vector<float> a(12 * 2 * ik::H), b(12 * 2 * ik::H);
        QV_CUDA_CHECK(cudaMemcpy(a.data(), weights_ + mf.at(P + "middle_cycle_cross_pass_queries").off, a.size() * 4, cudaMemcpyDeviceToHost));
        QV_CUDA_CHECK(cudaMemcpy(b.data(), weights_ + mf.at(P + "middle_cycle_cross_pass_kv_queries").off, b.size() * 4, cudaMemcpyDeviceToHost));
        for (int li = 0; li < 12; ++li) {
            auto cp = [&](int dst, const float* src) { std::memcpy(&mq[((size_t)li * 4 + dst) * ik::H], src, ik::H * 4); };
            cp(0, &a[((size_t)li * 2 + 0) * ik::H]);
            cp(1, &b[((size_t)li * 2 + 0) * ik::H]);
            cp(2, &b[((size_t)li * 2 + 1) * ik::H]);
            cp(3, &a[((size_t)li * 2 + 1) * ik::H]);
        }
        float* d = dalloc<float>(mq.size());
        QV_CUDA_CHECK(cudaMemcpy(d, mq.data(), mq.size() * 4, cudaMemcpyHostToDevice));
        msgq_ = d;
    }
    int n_apps_ = 0, n_lslots_ = 0, n_gslots_ = 0;

    void alloc_state() {
        const int H = ik::H;
        seqs_.assign(opt_.n_seq, SeqSt());
        for (int i = 0; i < opt_.n_seq; ++i) {  // per-sequence state
            const size_t b0 = state_bytes_;
            lk_ = dalloc<__half>((size_t)n_lslots_ * ik::NKV * 1024 * ik::HD);
            lv_ = dalloc<__half>((size_t)n_lslots_ * ik::NKV * 1024 * ik::HD);
            if (kv4_) {
                const size_t rows = (size_t)n_gslots_ * ik::NKV * opt_.max_ctx;
                gkq_ = dalloc<uint8_t>(rows * 64); gvq_ = dalloc<uint8_t>(rows * 64);
                gks_ = dalloc<__half2>(rows * 2); gvs_ = dalloc<__half2>(rows * 2);
            } else {
                gk_ = dalloc<__half>((size_t)n_gslots_ * ik::NKV * opt_.max_ctx * ik::HD);
                gv_ = dalloc<__half>((size_t)n_gslots_ * ik::NKV * opt_.max_ctx * ik::HD);
            }
            cst_k_ = dalloc<float>(48 * 3 * 1024);
            cst_v_ = dalloc<float>(48 * 3 * 1024);
            cst_a_ = dalloc<float>((size_t)48 * 3 * H);
            cst_m_ = dalloc<float>((size_t)48 * 3 * H);
            logits_ = dalloc<float>(cfg_.vocab);
            d_tok_ = dalloc<int>(2);
            d_pos_ = d_tok_ + 1;
            d_gen_ = dalloc<int>(opt_.max_ctx + 1);
            g_fwd_ = g_dec_ = nullptr;
            seqs_[i] = snapshot();
            if (i == 0) seq_bytes_ = state_bytes_ - b0;
        }
        cur_seq_ = opt_.n_seq - 1;
        select_seq(0);
        h_ = dalloc<float>(H);
        pre_h_ = dalloc<float>(H);
        blocks_ = dalloc<float>(4 * H);
        res_ = dalloc<float>(H);
        araw_ = dalloc<float>(H);
        ymoe_ = dalloc<float>(H);
        proj_ = dalloc<float>(8704);
        xn_f_ = dalloc<float>(3 * H);
        u_ = dalloc<float>(3 * H);
        gs_ = dalloc<float>(3 * H);
        gg_ = dalloc<float>(4 * 64);
        xm_f_ = dalloc<float>(H);
        mraw_ = dalloc<float>(H);
        mg_ = dalloc<float>(H);
        msgs_ = dalloc<float>((size_t)12 * 4 * H);
        rlog_ = dalloc<float>(128);
        wts_ = dalloc<float>(8);
        ids_ = dalloc<int>(8);
        ticket_ = dalloc<int>(1);
        qs_ = dalloc<float>(ik::NH * ik::HD);
        rl_ = dalloc<float>(ik::NH * 1024);
        part_ = dalloc<float>((size_t)ik::NKV * std::max(NSPLIT_L, NSPLIT_G) * 3 * ik::PART_STRIDE);
        q8x_ = q8alloc(3 * ik::HC);
        q8m_ = q8alloc(ik::HC);
        q8act_ = q8alloc(8 * 36);
        q8o_ = q8alloc(96);
        d_arg_ = dalloc<int>(1);
        QV_CUDA_CHECK(cudaMallocHost(&h_tp_, 2 * TP_SLOTS * sizeof(int)));
        if (opt_.dump_hidden) {
            dump_d_ = dalloc<float>((size_t)50 * H);
            rid_d_ = dalloc<int>(48 * 8);
        }
    }


    // ======================================== batched prefill ========================================
    cublasHandle_t cublas_ = nullptr;
    int Tp_ = 0;  // allocated rows (multiple of 64)
    float *hT_, *preT_, *blocksT_, *resT_, *arawT_, *ymoeT_, *mrawT_, *prevhT_, *prevb2T_, *uT_, *gsT_, *msgsT_, *PB_, *xmf_, *router32_, *rlogT_, *wtsT_, *g32_, *fT_, *gT_;
    __half *xhT_, *xmh_, *qh_, *relh_, *rlT_, *o16_, *stK_, *stV_, *Xs_, *Y13_, *Ah_, *Y2_, *xcat_, *g16_, *slab_, *relproj_h_;
    int *idsT_, *permT_, *invT_, *cntT_, *offsT_, *fillT_, *d_ptok_, *h_offs_ = nullptr;
    size_t slab_bytes_ = (size_t)32 << 20;

    template <class T> T* palloc(size_t n) { return dalloc<T>(n); }
    void alloc_prefill() {
        Tp_ = (opt_.prefill_chunk + 63) / 64 * 64;
        const size_t T = Tp_, H = ik::H;
        QV_CUBLAS(cublasCreate(&cublas_));
        QV_CUBLAS(cublasSetStream(cublas_, stream_));
        QV_CUBLAS(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));
        hT_ = palloc<float>(T * H); preT_ = palloc<float>(T * H); blocksT_ = palloc<float>(4 * T * H); resT_ = palloc<float>(T * H);
        arawT_ = palloc<float>(T * H); ymoeT_ = palloc<float>(T * H); mrawT_ = palloc<float>(T * H); prevhT_ = palloc<float>(T * H);
        prevb2T_ = palloc<float>(T * H); uT_ = palloc<float>(3 * T * H); gsT_ = palloc<float>(3 * T * H); msgsT_ = palloc<float>(4 * T * H);
        PB_ = palloc<float>(T * ik::PS); xmf_ = palloc<float>(T * H); router32_ = palloc<float>((size_t)24 * 128 * H); rlogT_ = palloc<float>(T * 128); wtsT_ = palloc<float>(T * 8);
        if (mtp_) { featT_ = palloc<float>(T * 4 * H); finT_ = palloc<float>(T * H); }
        g32_ = palloc<float>(T * 64); fT_ = palloc<float>(T * H); gT_ = palloc<float>(T * H);
        xhT_ = palloc<__half>(3 * T * H); xmh_ = palloc<__half>(T * H); qh_ = palloc<__half>(T * 3072); relh_ = palloc<__half>(T * 24 * 16);
        rlT_ = palloc<__half>(T * 24 * 1024); o16_ = palloc<__half>(T * 3072);
        stK_ = palloc<__half>((size_t)8 * (1024 + T) * 128); stV_ = palloc<__half>((size_t)8 * (1024 + T) * 128);
        Xs_ = palloc<__half>(T * 8 * H); Y13_ = palloc<__half>(T * 8 * 2 * 1152); Ah_ = palloc<__half>(T * 8 * 1152); Y2_ = palloc<__half>(T * 8 * H);
        xcat_ = palloc<__half>(T * 2 * H); g16_ = palloc<__half>(T * 64);
        slab_ = (__half*)((void*)palloc<uint8_t>(slab_bytes_));
        idsT_ = palloc<int>(T * 8); permT_ = palloc<int>(T * 8); invT_ = palloc<int>(T * 8);
        cntT_ = palloc<int>(130); offsT_ = palloc<int>(130); fillT_ = palloc<int>(130); d_ptok_ = palloc<int>(T);
        QV_CUDA_CHECK(cudaMallocHost(&h_offs_, 130 * 4));
        // fp16 copies of rel_logits_proj [16][1024] for the bias-table GEMM
        relproj_h_ = palloc<__half>((size_t)24 * 16 * 1024);
        for (int L = 0; L < 24; ++L) ik::f32_to_f16(lw_[L].relproj, 16 * 1024, relproj_h_ + (size_t)L * 16 * 1024, stream_);
        for (int L = 0; L < 24; ++L) ik::f16_to_f32(lw_[L].router.w, 128 * H, router32_ + (size_t)L * 128 * H, stream_);
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // C[M][N] (row-major, ld ldc) = A[M][K] * B[N][K]^T ; fp16 inputs, fp32 accumulate
    void gemm_nt(const __half* A, int M, int K, const __half* B, int N, void* C, int ldc, bool c16) {
        const float alpha = 1.f, beta = 0.f;
        QV_CUBLAS(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C,
                               c16 ? CUDA_R_16F : CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    void qgemm(const QMat& W, const __half* A, int M, float* C, int ldc) {
        dequant_rows_f16(W, 0, W.N, slab_, stream_);
        gemm_nt(A, M, W.K, slab_, W.N, C, ldc, false);
    }
    void dgemm(const DMat16& W, const __half* A, int M, float* C, int ldc) { gemm_nt(A, M, W.K, W.w, W.N, C, ldc, false); }
    QMat expert_view(const ik::ExpertW& E, int e) {
        QMat m;
        const size_t C = E.K >> 5, G = E.K >> E.gshift;
        m.q = E.q + (size_t)e * E.N * C * E.bits;
        m.sm8 = E.sm8 + (size_t)e * E.N * G;
        m.rs = E.rs + (size_t)e * E.N;
        m.N = E.N; m.K = E.K; m.bits = E.bits; m.gshift = E.gshift; m.zlo = E.zlo; m.zstep = E.zstep;
        return m;
    }
    void gate_t(const float* h, const float* msg, const Ctrl& c, float* out, int T) {
        ik::gate_cat(h, msg, T, xcat_, stream_);
        gemm_nt(xcat_, T, 2 * ik::H, c.down, 64, g32_, 64, false);
        ik::f32_to_f16(g32_, (long)T * 64, g16_, stream_);
        gemm_nt(g16_, T, 64, c.forget, ik::H, fT_, ik::H, false);
        gemm_nt(g16_, T, 64, c.gain, ik::H, gT_, ik::H, false);
        ik::gate_apply_t(fT_, gT_, h, msg, (long)T * ik::H, out, stream_);
    }

    void prefill_chunk(const int* tokens, int T, int pos0, const unsigned char* mask = nullptr, const float* emb = nullptr) {
        if (!cublas_) alloc_prefill();
        if (T > Tp_) throw std::runtime_error("prefill chunk larger than allocation");
        cudaStream_t s = stream_;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_ptok_, tokens, T * 4, cudaMemcpyHostToDevice, s));  // caller keeps `tokens` alive until sync: copy via pinned below
        QV_CUDA_CHECK(cudaStreamSynchronize(s));
        ik::embed_norm(embed_, d_ptok_, embed_norm_, hT_, blocksT_, s, T);
        if (mask) {  // vision: overwrite masked rows (runs of consecutive rows -> one copy each) in the hidden state and block 0
            for (int i = 0; i < T;) {
                if (!mask[i]) { ++i; continue; }
                int j = i;
                while (j < T && mask[j]) ++j;
                QV_CUDA_CHECK(cudaMemcpyAsync(hT_ + (size_t)i * ik::H, emb, (size_t)(j - i) * ik::H * 4, cudaMemcpyHostToDevice, s));
                QV_CUDA_CHECK(cudaMemcpyAsync(blocksT_ + (size_t)i * ik::H, emb, (size_t)(j - i) * ik::H * 4, cudaMemcpyHostToDevice, s));
                QV_CUDA_CHECK(cudaStreamSynchronize(s));  // `emb` is pageable host memory: complete before it can change
                emb += (size_t)(j - i) * ik::H;
                i = j;
            }
        }
        const long TH = (long)Tp_ * ik::H;
        const InklingConfig& K = cfg_.ink;
        for (const App& A : apps_) {
            prefill_app(A, T, pos0);
            if (mtp_ && (A.idx == 4 || A.idx == 22 || A.idx == 40))
                QV_CUDA_CHECK(cudaMemcpy2DAsync(featT_ + (size_t)(A.idx == 4 ? 0 : A.idx == 22 ? 1 : 2) * ik::H, 4 * ik::H * 4, hT_, ik::H * 4, ik::H * 4, T,
                                                cudaMemcpyDeviceToDevice, s));
            if (g_pfdump) {
                QV_CUDA_CHECK(cudaStreamSynchronize(s));
                hidden_.resize((size_t)50 * ik::H);
                const float* ho = (A.L == K.loop_start - 1) ? preT_ : hT_;
                QV_CUDA_CHECK(cudaMemcpy(hidden_.data() + (size_t)(A.idx + 1) * ik::H, ho + (size_t)(T - 1) * ik::H, ik::H * 4, cudaMemcpyDeviceToHost));
            }
            if (A.L == K.loop_end - 1 && A.p < K.loop_passes - 1) {
                ik::copy_f32(hT_, prevhT_, TH, s);
                ik::copy_f32(blocksT_ + 2 * TH, prevb2T_, TH, s);
            }
        }
        if (mtp_) {  // final-norm features of all rows, then the drafter's context K/V of the chunk
            ik::ArArgs f2;
            f2.mode = 0; f2.B = 4; f2.Q = 1; f2.mix = 1; f2.epi = 2; f2.ntok = Tp_; f2.allpar = 1;
            f2.h = hT_; f2.blocks = blocksT_; f2.queries = out_query_; f2.nw[0] = final_norm_; f2.xh_out = xhT_; f2.xn_out = finT_;
            ik::attn_res(f2, T, s);
            QV_CUDA_CHECK(cudaMemcpy2DAsync(featT_ + (size_t)3 * ik::H, 4 * ik::H * 4, finT_, ik::H * 4, ik::H * 4, T, cudaMemcpyDeviceToDevice, s));
            for (int g = 0; g < T; g += RR) mtp_ingest(featT_ + (size_t)g * 4 * ik::H, std::min(RR, T - g), pos0 + g, nullptr, nullptr, nullptr);
        }
        // last token: final mix + norm + head
        ik::ArArgs f;
        f.mode = 0; f.B = 4; f.Q = 1; f.mix = 1; f.epi = 0;
        f.h = hT_; f.blocks = blocksT_; f.queries = out_query_; f.ntok = Tp_; f.tok0 = T - 1; f.raw_out = uT_;
        ik::attn_res(f, 1, s);
        ik::NormArgs n;
        n.S = 1; n.in[0] = uT_ + (size_t)(T - 1) * ik::H; n.nw[0] = final_norm_; n.xn_out = xn_f_; n.q8 = q8x_;
        ik::norm_q8(n, s);
        gemv_q8(unembed_, q8x_, logits_, nullptr, s);
        ik::softcap_inplace(logits_, cfg_.ink.unpadded_vocab, cfg_.ink.final_logit_softcap, s);
    }

    void prefill_app(const App& A, int T, int pos0) {
        const InklingConfig& K = cfg_.ink;
        const LayerW& W = lw_[A.L];
        cudaStream_t s = stream_;
        const int H = ik::H, L = A.L, p = A.p, li = A.L - K.loop_start;
        const long TH = (long)Tp_ * H;
        const bool glob = W.glob, msg = p > 0;
        const float* hin = (L == K.loop_start) ? preT_ : hT_;
        const int Battn = (L + 5) / 6;
        const int Qa = glob ? 3 : 1;
        // ---- 1. attention input
        if (!msg) {
            ik::ArArgs a;
            a.mode = 0; a.B = Battn; a.Q = Qa; a.mix = L > 0; a.epi = 2; a.ntok = Tp_; a.allpar = 1;
            a.h = hin; a.blocks = blocksT_; a.queries = W.attn_q;
            for (int q = 0; q < Qa; ++q) a.nw[q] = W.nw[q];
            a.xh_out = xhT_;
            ik::attn_res(a, T, s);
        } else {
            ik::ArArgs m;
            m.mode = 0; m.B = 1; m.Q = 4; m.mix = 1; m.epi = 0; m.ntok = Tp_; m.allpar = 1;
            m.h = prevhT_; m.blocks = prevb2T_; m.queries = msgq_ + (size_t)li * 4 * H; m.raw_out = msgsT_;
            ik::attn_res(m, T, s);
            ik::ArArgs a;
            a.mode = 0; a.B = Battn; a.Q = Qa; a.mix = 1; a.epi = 0; a.ntok = Tp_; a.allpar = 1;
            a.h = hin; a.blocks = blocksT_; a.queries = W.attn_q; a.raw_out = uT_;
            ik::attn_res(a, T, s);
            for (int q = 0; q < 3; ++q) gate_t(glob ? uT_ + q * TH : uT_, msgsT_ + q * TH, W.ctrl[q], gsT_ + q * TH, T);
            ik::NormArgs n;
            n.S = 3; n.T = Tp_; n.Tg = T; n.xh_out = xhT_;
            for (int q = 0; q < 3; ++q) { n.in[q] = gsT_ + q * TH; n.nw[q] = W.nw[q]; }
            ik::norm_q8(n, s);
        }
        const int sk = (glob || msg) ? 1 : 0, sv = (glob || msg) ? 2 : 0;
        // ---- 2. projections
        qgemm(W.wq, xhT_, T, PB_ + 0, ik::PS);
        qgemm(W.wk, xhT_ + sk * TH, T, PB_ + 3072, ik::PS);
        qgemm(W.wv, xhT_ + sv * TH, T, PB_ + 4096, ik::PS);
        qgemm(W.wg, xhT_, T, PB_ + 5120, ik::PS);
        dgemm(W.wr, xhT_, T, PB_ + 8192, ik::PS);
        dgemm(W.sink, xhT_, T, PB_ + 8576, ik::PS);
        if (glob) {
            dgemm(W.kproj, xhT_ + sk * TH, T, PB_ + 8600, ik::PS);
            dgemm(W.vproj, xhT_ + sv * TH, T, PB_ + 8632, ik::PS);
        }
        // ---- 3. prep (+ staging of the local window)
        const int SL = 1024 + Tp_;
        __half *kc, *vc;
        int cap;
        long kbase;
        __half *ringk = nullptr, *ringv = nullptr;
        if (glob) {
            kc = kv4_ ? nullptr : gk_ + (size_t)A.kvslot * ik::NKV * opt_.max_ctx * ik::HD;
            vc = kv4_ ? nullptr : gv_ + (size_t)A.kvslot * ik::NKV * opt_.max_ctx * ik::HD;
            cap = opt_.max_ctx; kbase = 0;
        } else {
            ringk = lk_ + (size_t)A.kvslot * ik::NKV * 1024 * ik::HD;
            ringv = lv_ + (size_t)A.kvslot * ik::NKV * 1024 * ik::HD;
            ik::ring_to_stage(ringk, stK_, SL, pos0, s);
            ik::ring_to_stage(ringv, stV_, SL, pos0, s);
            kc = stK_; vc = stV_; cap = SL; kbase = (long)pos0 - 1024;
        }
        {
            ik::PrepTArgs P{};
            P.proj = PB_; P.qscale = W.qscale; P.relscale = W.relscale; P.relproj = W.relproj; P.kw = W.kw; P.vw = W.vw;
            P.kst = cst_k_ + (size_t)A.idx * 3 * 1024; P.vst = cst_v_ + (size_t)A.idx * 3 * 1024;
            P.glob = glob; P.T = T; P.pos0 = pos0; P.qh = qh_; P.relh = relh_; P.kc = kc; P.vc = vc; P.cap = cap;
            if (glob && kv4_) { P.kq = gkq_ + grow(A) * 64; P.vq = gvq_ + grow(A) * 64; P.ks = gks_ + grow(A) * 2; P.vs = gvs_ + grow(A) * 2; }
            ik::attn_prep_t(P, s);
            ik::conv_commit(PB_ + 3072, ik::PS, T, 1024, cst_k_ + (size_t)A.idx * 3 * 1024, s);
            ik::conv_commit(PB_ + 4096, ik::PS, T, 1024, cst_v_ + (size_t)A.idx * 3 * 1024, s);
        }
        {  // relative-bias table: rl[t*24+h][d] = rel[t,h,:] . proj[:, d]
            const float alpha = 1.f, beta = 0.f;
            QV_CUBLAS(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, 1024, T * 24, 16, &alpha, relproj_h_ + (size_t)A.L * 16 * 1024, CUDA_R_16F, 1024,
                                   relh_, CUDA_R_16F, 16, &beta, rlT_, CUDA_R_16F, 1024, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        // ---- 4. attention
        {
            ik::PfAttnArgs P{};
            P.qh = qh_; P.kc = kc; P.vc = vc; P.kbase = kbase; P.cap = cap; P.rl = rlT_; P.proj = PB_; P.T = T; P.pos0 = pos0;
            P.window = glob ? 0 : K.window; P.out = o16_;
            if (glob && kv4_) { P.kq = gkq_ + grow(A) * 64; P.vq = gvq_ + grow(A) * 64; P.ks = gks_ + grow(A) * 2; P.vs = gvs_ + grow(A) * 2; }
            ik::attn_prefill(P, s);
            if (!glob) {
                ik::stage_to_ring(stK_, ringk, SL, pos0, T, s);
                ik::stage_to_ring(stV_, ringv, SL, pos0, T, s);
            }
        }
        // ---- 5. output projection
        qgemm(W.wo, o16_, T, arawT_, H);
        // ---- 6. post-attention conv + residual, MLP-input mix
        {
            ik::ArArgs b;
            b.mode = 1; b.B = Battn; b.Q = 1; b.mix = 1; b.boundary = (L % 6 == 0); b.ntok = Tp_; b.allpar = 1;
            b.h = hin; b.araw = arawT_; b.cw = W.attn_sconv; b.cst = cst_a_ + (size_t)A.idx * 3 * H; b.res_out = resT_; b.blocks = blocksT_;
            b.queries = W.mlp_q;
            if (!msg) {
                b.epi = 2; b.nw[0] = W.mlp_norm; b.xh_out = xmh_; b.xn_out = xmf_;
                ik::attn_res(b, T, s);
            } else {
                b.epi = 0; b.raw_out = mrawT_;
                ik::attn_res(b, T, s);
                gate_t(mrawT_, msgsT_ + 3 * TH, W.ctrl[3], gsT_, T);
                ik::NormArgs n;
                n.S = 1; n.T = Tp_; n.Tg = T; n.in[0] = gsT_; n.nw[0] = W.mlp_norm; n.xh_out = xmh_; n.xn_out = xmf_;
                ik::norm_q8(n, s);
            }
            ik::conv_commit(arawT_, H, T, H, cst_a_ + (size_t)A.idx * 3 * H, s);
        }
        // ---- 7. router
        {
            const float alpha = 1.f, beta = 0.f;
            QV_CUBLAS(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 128, T, H, &alpha, router32_ + (size_t)A.L * 128 * H, H, xmf_, H, &beta, rlogT_, 128));
        }
        ik::topk_t(rlogT_, W.router_bias, idsT_, wtsT_, T, K.route_scale, s);
        // ---- 8. experts: token-sorted grouped GEMM
        ik::moe_sort(idsT_, T, cntT_, offsT_, fillT_, permT_, invT_, s);
        QV_CUDA_CHECK(cudaMemcpyAsync(h_offs_, offsT_, 129 * 4, cudaMemcpyDeviceToHost, s));
        QV_CUDA_CHECK(cudaStreamSynchronize(s));
        ik::moe_gather(xmh_, permT_, T * 8, Xs_, s);
        for (int e = 0; e < ik::NE; ++e) {
            const int n = h_offs_[e + 1] - h_offs_[e];
            if (n <= 0) continue;
            dequant_rows_f16(expert_view(W.w13, e), 0, 2304, slab_, s);
            gemm_nt(Xs_ + (size_t)h_offs_[e] * H, n, H, slab_, 2 * ik::FF, Y13_ + (size_t)h_offs_[e] * 2 * ik::FF, 2 * ik::FF, true);
        }
        ik::moe_act(Y13_, T * 8, K.gate_softcap, K.up_softcap, Ah_, s);
        for (int e = 0; e < ik::NE; ++e) {
            const int n = h_offs_[e + 1] - h_offs_[e];
            if (n <= 0) continue;
            dequant_rows_f16(expert_view(W.w2, e), 0, 2304, slab_, s);
            gemm_nt(Ah_ + (size_t)h_offs_[e] * ik::FF, n, ik::FF, slab_, H, Y2_ + (size_t)h_offs_[e] * H, H, true);
        }
        ik::moe_combine(Y2_, invT_, wtsT_, T, ymoeT_, s);
        // ---- 9. MLP conv + residual
        float* hout = (L == K.loop_start - 1) ? preT_ : hT_;
        ik::finish(resT_, ymoeT_, W.mlp_sconv, cst_m_ + (size_t)A.idx * 3 * H, hout, s, Tp_);
        ik::conv_commit(ymoeT_, H, T, H, cst_m_ + (size_t)A.idx * 3 * H, s);
        if (g_dbgapp == A.idx) dbg_write("/tmp/ink/dbg_pf.bin", arawT_, resT_, xmf_, ymoeT_, idsT_, wtsT_, T - 1);
    }

    cudaGraphExec_t capture(bool with_argmax) {
        // warm-up outside capture (lazy one-time allocations of the shared argmax scratch)
        argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_arg_, nullptr, nullptr, stream_);
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        forward();
        if (with_argmax) argmax_advance(logits_, cfg_.ink.unpadded_vocab, d_tok_, d_pos_, d_gen_, stream_);
        cudaGraph_t g;
        QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
        cudaGraphExec_t ge;
        QV_CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        QV_CUDA_CHECK(cudaGraphDestroy(g));
        return ge;
    }

    int dbg_n_ = 0;
    void dbg_write(const char* path, const float* a, const float* b, const float* c, const float* d, const int* ids, const float* w, int row) {
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        std::vector<float> buf(4 * ik::H + 16);
        const float* src[4] = {a, b, c, d};
        for (int i = 0; i < 4; ++i) QV_CUDA_CHECK(cudaMemcpy(&buf[(size_t)i * ik::H], src[i] + (size_t)row * ik::H, ik::H * 4, cudaMemcpyDeviceToHost));
        QV_CUDA_CHECK(cudaMemcpy(&buf[4 * ik::H], ids + (size_t)row * 8, 32, cudaMemcpyDeviceToHost));
        QV_CUDA_CHECK(cudaMemcpy(&buf[4 * ik::H + 8], w + (size_t)row * 8, 32, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen(path, "wb");
        std::fwrite(buf.data(), 4, buf.size(), f);
        std::fclose(f);
    }
    void collect_dump() {
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        hidden_.resize((size_t)50 * ik::H);
        rids_.resize(48 * 8);
        QV_CUDA_CHECK(cudaMemcpy(hidden_.data(), dump_d_, hidden_.size() * 4, cudaMemcpyDeviceToHost));
        QV_CUDA_CHECK(cudaMemcpy(rids_.data(), rid_d_, rids_.size() * 4, cudaMemcpyDeviceToHost));
    }

    static ik::GD gd_q(const QMat& W, const Q8Buf& x, int stream, float* y) {
        ik::GD g;
        g.kind = 0; g.N = W.N; g.K = W.K; g.gshift = W.gshift; g.q = W.q; g.sm8 = W.sm8; g.rs = W.rs; g.zlo = W.zlo; g.zstep = W.zstep;
        g.xq = Q8Vec{x.q + stream * (ik::H / 4), x.d + stream * ik::HC, x.sum + stream * ik::HC, x.qsum + stream * ik::HC};
        g.y = y;
        return g;
    }
    static ik::GD gd_h(const DMat16& W, const float* x, float* y) {
        ik::GD g;
        g.kind = 1; g.N = W.N; g.K = W.K; g.w16 = W.w; g.xf = x; g.y = y;
        return g;
    }

    // F1 (p26): the MLP conv + residual (k_finish) of an application is folded into the next attention-input attn_res when that kernel reads the
    // finish output as its h (one block, one token). pend_ holds the deferred finish; flush_fin() launches it as a separate kernel.
    struct PendFin { bool on = false; const float* res; const float* y; const float* cw; float* cst; float* hout; } pend_;
    bool fuse_fin_ = std::getenv("QV_INK_NOFUSE_FIN") == nullptr;
    void flush_fin() {
        if (!pend_.on) return;
        ik::finish(pend_.res, pend_.y, pend_.cw, pend_.cst, pend_.hout, stream_);
        pend_.on = false;
    }
    // use the pending finish as the h of `a` (a.h must be its output) or launch it now
    void attach_fin(ik::ArArgs& a) {
        if (pend_.on && a.h == pend_.hout) {
            a.fin_res = pend_.res; a.fin_y = pend_.y; a.fin_cw = pend_.cw; a.fin_cst = pend_.cst; a.fin_h_out = pend_.hout;
            pend_.on = false;
        } else {
            flush_fin();
        }
    }

    void forward() {
        cudaStream_t s = stream_;
        pend_.on = false;
        ik::embed_norm(embed_, d_tok_, embed_norm_, h_, blocks_, s);
        if (ovr_on_) {
            QV_CUDA_CHECK(cudaMemcpyAsync(h_, d_ovr_, ik::H * 4, cudaMemcpyDeviceToDevice, s));
            QV_CUDA_CHECK(cudaMemcpyAsync(blocks_, d_ovr_, ik::H * 4, cudaMemcpyDeviceToDevice, s));
        }
        if (dump_d_) QV_CUDA_CHECK(cudaMemcpyAsync(dump_d_, h_, ik::H * 4, cudaMemcpyDeviceToDevice, s));
        for (const App& A : apps_) {
            run_app(A);
            if (mtp_ && (A.idx == 4 || A.idx == 22 || A.idx == 40)) flush_fin();
            if (mtp_ && (A.idx == 4 || A.idx == 22 || A.idx == 40))
                QV_CUDA_CHECK(cudaMemcpyAsync(featv_ + (size_t)(A.idx == 4 ? 0 : A.idx == 22 ? 1 : 2) * ik::H, h_, ik::H * 4, cudaMemcpyDeviceToDevice, s));
            if (A.L == cfg_.ink.loop_end - 1 && A.p < cfg_.ink.loop_passes - 1) {  // messages for the next pass
                ik::ArArgs m;
                flush_fin();
                m.mode = 0; m.B = 1; m.Q = 4; m.mix = 1; m.epi = 0;
                m.h = h_; m.blocks = blocks_ + 2 * ik::H;  // the state that entered layer 12 in this pass
                m.queries = msgq_; m.qbs = 4 * ik::H; m.raw_out = msgs_; m.rbs = 4 * ik::H;
                ik::attn_res(m, 12, s);
            }
        }
        ik::ArArgs f;
        f.mode = 0; f.B = 4; f.Q = 1; f.mix = 1; f.epi = 1;
        f.h = h_; f.blocks = blocks_; f.queries = out_query_; f.nw[0] = final_norm_; f.xn_out = xn_f_; f.q8 = q8x_;
        attach_fin(f);
        ik::attn_res(f, 1, s);
        if (dump_d_) QV_CUDA_CHECK(cudaMemcpyAsync(dump_d_ + 49 * ik::H, xn_f_, ik::H * 4, cudaMemcpyDeviceToDevice, s));
        if (mtp_) QV_CUDA_CHECK(cudaMemcpyAsync(featv_ + (size_t)3 * ik::H, xn_f_, ik::H * 4, cudaMemcpyDeviceToDevice, s));
        gemv_q8(unembed_, q8x_, logits_, nullptr, s);
        ik::softcap_inplace(logits_, cfg_.ink.unpadded_vocab, cfg_.ink.final_logit_softcap, s);
    }

    void run_app(const App& A) {
        const InklingConfig& K = cfg_.ink;
        const LayerW& W = lw_[A.L];
        cudaStream_t s = stream_;
        const int H = ik::H, L = A.L, p = A.p, li = A.L - K.loop_start;
        const bool glob = W.glob, msg = p > 0;
        const float* hin = (L == K.loop_start) ? pre_h_ : h_;
        const int Battn = (L + 5) / 6;
        // ---- 1. attention-input mix(es), norms, int8 quantization
        const int Qa = glob ? 3 : 1;
        if (!msg) {
            ik::ArArgs a;
            a.mode = 0; a.B = Battn; a.Q = Qa; a.mix = L > 0; a.epi = 1;
            a.h = hin; a.blocks = blocks_; a.queries = W.attn_q;
            for (int q = 0; q < Qa; ++q) a.nw[q] = W.nw[q];
            a.xn_out = xn_f_; a.q8 = q8x_;
            attach_fin(a);
            ik::attn_res(a, 1, s);
        } else {
            ik::ArArgs a;
            a.mode = 0; a.B = Battn; a.Q = Qa; a.mix = 1; a.epi = 0;
            a.h = hin; a.blocks = blocks_; a.queries = W.attn_q; a.raw_out = u_;
            attach_fin(a);
            ik::attn_res(a, 1, s);
            ik::GateArgs g;
            g.S = 3; g.g = gg_;
            for (int q = 0; q < 3; ++q) {
                g.h[q] = glob ? u_ + q * H : u_;
                g.msg[q] = msgs_ + ((size_t)li * 4 + q) * H;
                g.down[q] = W.ctrl[q].down; g.forget[q] = W.ctrl[q].forget; g.gain[q] = W.ctrl[q].gain;
                g.out[q] = gs_ + q * H;
            }
            ik::gate(g, s);
            ik::NormArgs n;
            n.S = 3;
            for (int q = 0; q < 3; ++q) { n.in[q] = gs_ + q * H; n.nw[q] = W.nw[q]; }
            n.xn_out = xn_f_; n.q8 = q8x_;
            ik::norm_q8(n, s);
        }
        STAGE("attnin");
        const int sk = (glob || msg) ? 1 : 0, sv = (glob || msg) ? 2 : 0;
        // ---- 2. projections: q, k, v, output gate, rel, sink (+ dynamic conv taps for global layers)
        {
            ik::GDs G;
            G.d[G.n++] = gd_q(W.wq, q8x_, 0, proj_ + 0);
            G.d[G.n++] = gd_q(W.wk, q8x_, sk, proj_ + 3072);
            G.d[G.n++] = gd_q(W.wv, q8x_, sv, proj_ + 4096);
            G.d[G.n++] = gd_q(W.wg, q8x_, 0, proj_ + 5120);
            G.d[G.n++] = gd_h(W.wr, xn_f_, proj_ + 8192);
            G.d[G.n++] = gd_h(W.sink, xn_f_, proj_ + 8576);
            if (glob) {
                G.d[G.n++] = gd_h(W.kproj, xn_f_ + (size_t)sk * H, proj_ + 8600);
                G.d[G.n++] = gd_h(W.vproj, xn_f_ + (size_t)sv * H, proj_ + 8632);
            }
            ik::mgemv(G, W.wq.bits, s);
        }
        STAGE("proj");
        // ---- 3. prep (q/k/v postprocessing, convs, KV write, rel bias)
        const bool ring = !glob;
        const int cap = glob ? opt_.max_ctx : 1024;
        __half* kc = glob ? (kv4_ ? nullptr : gk_ + (size_t)A.kvslot * ik::NKV * opt_.max_ctx * ik::HD) : lk_ + (size_t)A.kvslot * ik::NKV * 1024 * ik::HD;
        __half* vc = glob ? (kv4_ ? nullptr : gv_ + (size_t)A.kvslot * ik::NKV * opt_.max_ctx * ik::HD) : lv_ + (size_t)A.kvslot * ik::NKV * 1024 * ik::HD;
        {
            ik::PrepArgs P;
            P.qraw = proj_; P.kraw = proj_ + 3072; P.vraw = proj_ + 4096; P.relraw = proj_ + 8192; P.dwk = proj_ + 8600; P.dwv = proj_ + 8632;
            P.qscale = W.qscale; P.relscale = W.relscale; P.relproj = W.relproj; P.kw = W.kw; P.vw = W.vw;
            P.kst = cst_k_ + (size_t)A.idx * 3 * 1024; P.vst = cst_v_ + (size_t)A.idx * 3 * 1024;
            P.glob = glob; P.pos = d_pos_; P.qs = qs_; P.rl = rl_; P.kc = kc; P.vc = vc; P.cap = cap; P.ring = ring;
            if (glob && kv4_) { P.kq = gkq_ + grow(A) * 64; P.vq = gvq_ + grow(A) * 64; P.ks = gks_ + grow(A) * 2; P.vs = gvs_ + grow(A) * 2; }
            ik::attn_prep(P, s);
        }
        STAGE("prep");
        // ---- 4. attention + combine (sink, output gate) -> int8
        {
            ik::AttnArgs T;
            T.qs = qs_; T.rl = rl_; T.kc = kc; T.vc = vc; T.cap = cap; T.ring = ring; T.window = glob ? 0 : K.window;
            if (glob && kv4_) { T.kq = gkq_ + grow(A) * 64; T.vq = gvq_ + grow(A) * 64; T.ks = gks_ + grow(A) * 2; T.vs = gvs_ + grow(A) * 2; }
            T.pos = d_pos_; T.part = part_; T.nsplit = glob ? NSPLIT_G : NSPLIT_L;
            T.sink = proj_ + 8576; T.gate = proj_ + 5120; T.out = q8o_;
            ik::attn_part(T, s);
            STAGE("part");
            ik::attn_combine(T, s);
        }
        STAGE("attn");
        // ---- 5. output projection
        {
            ik::GDs G;
            ik::GD g;
            g.kind = 0; g.N = W.wo.N; g.K = W.wo.K; g.gshift = W.wo.gshift; g.q = W.wo.q; g.sm8 = W.wo.sm8; g.rs = W.wo.rs;
            g.zlo = W.wo.zlo; g.zstep = W.wo.zstep; g.xq = Q8Vec(q8o_); g.y = araw_;
            G.d[G.n++] = g;
            ik::mgemv(G, W.wo.bits, s);
        }
        STAGE("wo");
        // ---- 6. post-attention conv + residual (+ boundary reset), MLP-input mix, norm
        {
            ik::ArArgs b;
            b.mode = 1; b.B = Battn; b.Q = 1; b.mix = 1; b.boundary = (L % 6 == 0);
            b.h = hin; b.araw = araw_; b.cw = W.attn_sconv; b.cst = cst_a_ + (size_t)A.idx * 3 * H; b.res_out = res_; b.blocks = blocks_;
            b.queries = W.mlp_q;
            if (!msg) {
                b.epi = 1; b.nw[0] = W.mlp_norm; b.xn_out = xm_f_; b.q8 = q8m_;
                ik::attn_res(b, 1, s);
            } else {
                b.epi = 0; b.raw_out = mraw_;
                ik::attn_res(b, 1, s);
                ik::GateArgs g;
                g.S = 1; g.g = gg_;
                g.h[0] = mraw_; g.msg[0] = msgs_ + ((size_t)li * 4 + 3) * H;
                g.down[0] = W.ctrl[3].down; g.forget[0] = W.ctrl[3].forget; g.gain[0] = W.ctrl[3].gain; g.out[0] = mg_;
                ik::gate(g, s);
                ik::NormArgs n;
                n.S = 1; n.in[0] = mg_; n.nw[0] = W.mlp_norm; n.xn_out = xm_f_; n.q8 = q8m_;
                ik::norm_q8(n, s);
            }
        }
        STAGE("postattn");
        // ---- 7-8. router + experts
        ik::router(W.router.w, W.router_bias, xm_f_, rlog_, ids_, wts_, ticket_, K.route_scale, s);
        STAGE("router");
        ik::moe_w13(W.w13, Q8Vec(q8m_), ids_, q8act_, K.gate_softcap, K.up_softcap, s);
        STAGE("w13");
        ik::moe_w2(W.w2, Q8Vec(q8act_), ids_, wts_, ymoe_, s);
        STAGE("w2");
        // ---- 9. MLP conv + residual
        float* hout = (L == K.loop_start - 1) ? pre_h_ : h_;
        if (fuse_fin_ && !dump_d_ && g_dbgapp < 0) {
            pend_.on = true; pend_.res = res_; pend_.y = ymoe_; pend_.cw = W.mlp_sconv; pend_.cst = cst_m_ + (size_t)A.idx * 3 * H; pend_.hout = hout;
        } else {
            ik::finish(res_, ymoe_, W.mlp_sconv, cst_m_ + (size_t)A.idx * 3 * H, hout, s);
        }
        if (g_dbgapp == A.idx) dbg_write((std::string("/tmp/ink/dbg_step") + std::to_string(dbg_n_++) + ".bin").c_str(), araw_, res_, xm_f_, ymoe_, ids_, wts_, 0);
        if (dump_d_) {
            QV_CUDA_CHECK(cudaMemcpyAsync(dump_d_ + (size_t)(A.idx + 1) * H, hout, H * 4, cudaMemcpyDeviceToDevice, s));
            QV_CUDA_CHECK(cudaMemcpyAsync(rid_d_ + A.idx * 8, ids_, 32, cudaMemcpyDeviceToDevice, s));
        }
    }

#include "cuda_inkling_mtp.inc"
};

}  // namespace

std::unique_ptr<Engine> make_cuda_inkling_engine(const ModelFile& mf, const EngineOptions& opt) {
    return std::make_unique<InklingEngine>(mf, opt);
}

}  // namespace qv
