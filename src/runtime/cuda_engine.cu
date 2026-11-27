// CUDA backend: decode-step executor for Qwen3.5-family hybrid (GDN + gated attention) models.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../backends/cuda/attention/attn.h"
#include "../backends/cuda/attention/attn_seq.h"
#include "../backends/cuda/kernels/elementwise_seq.h"
#include "../backends/cuda/kernels/linear_prefill.h"
#include "../backends/cuda/recurrent/gdn_seq.h"
#include "../backends/cuda/common.cuh"
#include "../backends/cuda/kernels/elementwise.h"
#include "../backends/cuda/kernels/gemv.h"
#include "../backends/cuda/recurrent/gdn.h"
#include "qv/engine.h"

namespace qv {
using namespace qv::cuda;

namespace {

struct GdnLayer {
    QMat qkv, z, out;
    DMat16 a, b;
    const float *A_neg, *dt_bias, *conv_w, *norm_w;
};
struct AttnLayer {
    QMat q, k, v, o;
    const float *q_norm, *k_norm;
};
struct Layer {
    bool attn;
    int idx_kind;  // index into gdn/attn state arrays
    const float *in_norm, *post_norm;
    QMat gate, up, down;
    GdnLayer g;
    AttnLayer at;
};

class CudaEngine final : public Engine {
   public:
    CudaEngine(const ModelFile& mf, const EngineOptions& opt) : opt_(opt) {
        cfg_ = mf.cfg;
        if (opt_.mtp_depth < 1 || opt_.mtp_depth > 3) throw std::runtime_error("mtp_depth must be 1..3");
        if (opt_.n_seq < 1) opt_.n_seq = 1;
        if (opt_.n_seq > 1 && opt_.mtp) throw std::runtime_error("MTP supports one sequence (use n_seq = 1)");
        QV_CUDA_CHECK(cudaStreamCreate(&stream_));
        load_weights(mf);
        bind(mf);
        alloc_state();
        for (int i = opt_.n_seq - 1; i >= 0; --i) { select_seq(i); reset(); }  // ends on sequence 0
    }
    ~CudaEngine() override {
        seqs_[cur_seq_] = snapshot();
        for (SeqSt& q : seqs_) if (q.graph_exec) cudaGraphExecDestroy(q.graph_exec);
        graph_exec_ = nullptr;
        lp_.destroy();
        if (h_tp_) cudaFreeHost(h_tp_);
        if (opt_.mtp) {
            if (h_a_) cudaFreeHost(h_a_);
            if (h_b_) cudaFreeHost(h_b_);
            if (h_res_) cudaFreeHost(h_res_);
            if (h_smeta_) cudaFreeHost(h_smeta_);
            if (h_scand_id_) cudaFreeHost(h_scand_id_);
            if (h_scand_val_) cudaFreeHost(h_scand_val_);
            if (h_sfull_p_) cudaFreeHost(h_sfull_p_);
            if (h_dq_id_) cudaFreeHost(h_dq_id_);
            if (h_dq_prob_) cudaFreeHost(h_dq_prob_);
            if (h_dq_cnt_) cudaFreeHost(h_dq_cnt_);
        }
        for (auto g : vgraph_) if (g) cudaGraphExecDestroy(g);
        for (auto g : vgraph_spec_) if (g) cudaGraphExecDestroy(g);
        for (auto g : mgraph_) if (g) cudaGraphExecDestroy(g);
        for (auto g : mgraph_spec_) if (g) cudaGraphExecDestroy(g);
        for (void* p : allocs_) cudaFree(p);
        cudaFree(weights_);
        cudaStreamDestroy(stream_);
    }

    void step(int token, int pos) override {
        if (pos >= opt_.max_ctx) throw std::runtime_error("context overflow");
        mtp_ready_ = -1;
        h_tp_[0] = token;
        h_tp_[1] = pos;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, h_tp_, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        forward();
    }

    // Greedy decode of n tokens after the context [0, start_pos) has been processed with step().
    // first = argmax of the current logits (token at start_pos). With use_graph the whole step (forward + argmax +
    // feedback) is one CUDA graph and no host synchronisation happens until the end.
    void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool use_graph) override {
        if (start_pos + n > opt_.max_ctx) throw std::runtime_error("context overflow");
        if (opt_.mtp && mtp_on_ && n >= 2 && mtp_ready_ == start_pos) { decode_mtp(first, start_pos, n, out); return; }
        h_tp_[0] = first;
        h_tp_[1] = start_pos;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, h_tp_, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        if (use_graph && !graph_exec_) {
            QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            forward();
            argmax_advance(logits_, cfg_.vocab, d_tok_, d_pos_, d_gen_, stream_);
            cudaGraph_t g;
            QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
            QV_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, g, 0));
            QV_CUDA_CHECK(cudaGraphDestroy(g));
        }
        for (int i = 0; i < n; ++i) {
            if (use_graph) {
                QV_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
            } else {
                forward();
                argmax_advance(logits_, cfg_.vocab, d_tok_, d_pos_, d_gen_, stream_);
            }
        }
        out.resize(n);
        QV_CUDA_CHECK(cudaMemcpyAsync(out.data(), d_gen_ + start_pos, n * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // Enqueue one full forward for the token/position held in device memory (d_tok_, d_pos_).
    void forward() {
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        embed(embed_, d_tok_, x_, s);
        if (opt_.dump_hidden) dump();
        for (const Layer& L : layers_) {
            rmsnorm(x_, L.in_norm, xn_, xs_, c.hidden, c.eps, s, q8_ ? &xq_ : nullptr);
            if (!L.attn) {
                const GdnLayer& g = L.g;
                mv(g.qkv, xn_, qkv_, nullptr);
                mv(g.z, xn_, z_, nullptr);
                gemv_f16_dual(g.a, g.b, xn_, alpha_, beta_, s);
                const int C = g.qkv.N;
                conv_step(qkv_, g.conv_w, conv_state_ + (size_t)L.idx_kind * C * 3, conv_out_, C, s);
                GdnStepArgs a{conv_out_, alpha_, beta_, g.A_neg, g.dt_bias, z_, g.norm_w,
                              gdn_state_ + (size_t)L.idx_kind * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim, mix_,
                              c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim, c.eps, q8_ ? xq_ : Q8Buf{}};
                gdn_step(a, s);
                if (!q8_) prep(mix_, g.out.K, false);
                mv(g.out, mix_, x_, x_);  // residual add in place
            } else {
                const AttnLayer& at = L.at;
                mv(at.q, xn_, qg_, nullptr);
                mv(at.k, xn_, kb_, nullptr);
                mv(at.v, xn_, vb_, nullptr);
                const size_t kv_layer = (size_t)c.n_kv_heads * opt_.max_ctx * c.head_dim;
                float* kc = kv4_on_ ? nullptr : k_cache_ + L.idx_kind * kv_layer;
                float* vc = kv4_on_ ? nullptr : v_cache_ + L.idx_kind * kv_layer;
                AttnPrepArgs p{qg_, kb_, vb_, at.q_norm, at.k_norm, inv_freq_, qr_, kc, vc,
                               c.n_heads, c.n_kv_heads, c.head_dim, c.rope_dim, opt_.max_ctx, d_pos_, c.eps};
                p.kv4 = kv4_of(L.idx_kind);
                attn_prep(p, s);
                AttnDecodeArgs d{qr_, qg_, kc, vc, nullptr, mix_, c.n_heads, c.n_kv_heads, opt_.max_ctx, d_pos_,
                                 q8_ ? xq_ : Q8Buf{}};
                d.part = part_;
                d.kv4 = p.kv4;
                attn_decode(d, s);
                if (!q8_) prep(mix_, at.o.K, false);
                mv(at.o, mix_, x_, x_);
            }
            rmsnorm(x_, L.post_norm, xn_, xs_, c.hidden, c.eps, s, q8_ ? &xq_ : nullptr);
            mv(L.gate, xn_, g_, nullptr);
            mv(L.up, xn_, u_, nullptr);
            silu_mul(g_, u_, act_, xs_, c.inter, s, q8_ ? &xq_ : nullptr);
            mv(L.down, act_, x_, x_);
            if (opt_.dump_hidden) dump();
        }
        rmsnorm(x_, final_norm_, xn_, xs_, c.hidden, c.eps, s, q8_ ? &xq_ : nullptr);
        mv(head_, xn_, logits_, nullptr);
    }


    // ---------------------------------------------------------------- batched prefill (P1)
    void prefill(const int* tokens, int T, int pos0) override {
        if (pos0 + T > opt_.max_ctx) throw std::runtime_error("context overflow");
        ensure_prefill_buffers();
        mtp_pf_ = opt_.mtp && mtp_on_ && !opt_.dump_hidden && (pos0 == 0 || pos0 == mtp_ready_);
        if (mtp_pf_ && pos0 > 0 && T > 0) mtp_draft(1, tokens, hlast_, pos0 - 1, 1);
        mtp_ready_ = -1;
        mtp_pf_T_ = T;
        const int L = (int)layers_.size(), H = cfg_.hidden;
        size_t hbase = 0;
        if (opt_.dump_hidden) {
            hbase = hidden_.size();
            hidden_.resize(hbase + (size_t)T * (L + 1) * H);
        }
        int last_n = 0;
        for (int t0 = 0; t0 < T; t0 += pf_chunk_) {
            const int n = std::min(pf_chunk_, T - t0);
            prefill_chunk(tokens + t0, n, pos0 + t0, hbase, t0);
            last_n = n;
        }
        // logits of the last token through the single-token tail (final norm + lm_head)
        QV_CUDA_CHECK(cudaMemcpyAsync(x_, pf_x_ + (size_t)(last_n - 1) * H, H * 4, cudaMemcpyDeviceToDevice, stream_));
        rmsnorm(x_, final_norm_, xn_, xs_, H, cfg_.eps, stream_, q8_ ? &xq_ : nullptr);
        mv(head_, xn_, logits_, nullptr);
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (mtp_pf_) mtp_ready_ = pos0 + T;
    }

    void pf_dump(int layer_slot, int n, int t0, size_t hbase) {
        if (!opt_.dump_hidden) return;
        const int L = (int)layers_.size(), H = cfg_.hidden;
        QV_CUDA_CHECK(cudaMemcpy2DAsync(hidden_.data() + hbase + ((size_t)t0 * (L + 1) + layer_slot) * H,
                                        (size_t)(L + 1) * H * 4, pf_x_, (size_t)H * 4, (size_t)H * 4, n,
                                        cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // one decoder layer over n prompt rows (residual stream X [n][H], updated in place)
    void prefill_layer(const Layer& L, int n, int pos0, float* X) {
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        const int H = c.hidden;
        rmsnorm_rows_f16(X, L.in_norm, pf_xh_, H, n, c.eps, s);
        if (!L.attn) {
            const GdnLayer& g = L.g;
            const int C = g.qkv.N, VD = c.lin_v_heads * c.lin_v_dim;
            linear_seq(lp_, g.qkv, pf_xh_, n, pf_qkv_, s);
            linear_seq(lp_, g.z, pf_xh_, n, pf_z_, s);
            linear_seq_dense(lp_, g.a, pf_xh_, n, pf_alpha_, s);
            linear_seq_dense(lp_, g.b, pf_xh_, n, pf_beta_, s);
            conv_seq(pf_qkv_, g.conv_w, conv_state_ + (size_t)L.idx_kind * C * 3, pf_conv_, C, n, s);
            GdnSeqArgs a{};
            a.conv_out = pf_conv_; a.alpha = pf_alpha_; a.beta_logit = pf_beta_; a.A_neg = g.A_neg;
            a.dt_bias = g.dt_bias; a.z = pf_z_; a.norm_w = g.norm_w;
            a.state = gdn_state_ + (size_t)L.idx_kind * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim;
            a.out_h = pf_mixh_; a.out_f = nullptr;
            a.n_k_heads = c.lin_k_heads; a.n_v_heads = c.lin_v_heads; a.dk = c.lin_k_dim; a.dv = c.lin_v_dim;
            a.T = n; a.eps = c.eps;
            gdn_seq(a, s);
            (void)VD;
            linear_seq(lp_, g.out, pf_mixh_, n, pf_y_, s);
        } else {
            const AttnLayer& at = L.at;
            linear_seq(lp_, at.q, pf_xh_, n, pf_qg_, s);
            linear_seq(lp_, at.k, pf_xh_, n, pf_k_, s);
            linear_seq(lp_, at.v, pf_xh_, n, pf_v_, s);
            const size_t kv_layer = (size_t)c.n_kv_heads * opt_.max_ctx * c.head_dim;
            float* kc = kv4_on_ ? nullptr : k_cache_ + L.idx_kind * kv_layer;
            float* vc = kv4_on_ ? nullptr : v_cache_ + L.idx_kind * kv_layer;
            AttnSeqPrepArgs p{};
            p.kv4 = kv4_of(L.idx_kind); p.stage_k = pf_sk_; p.stage_v = pf_sv_;
            p.qg = pf_qg_; p.k = pf_k_; p.v = pf_v_; p.q_norm = at.q_norm; p.k_norm = at.k_norm;
            p.inv_freq = inv_freq_; p.q_out = pf_qr_; p.k_cache = kc; p.v_cache = vc;
            p.n_heads = c.n_heads; p.n_kv_heads = c.n_kv_heads; p.head_dim = c.head_dim; p.rope_dim = c.rope_dim;
            p.max_ctx = opt_.max_ctx; p.pos0 = pos0; p.T = n; p.eps = c.eps;
            attn_prep_seq(p, s);
            AttnSeqArgs d{};
            d.q = pf_qr_; d.qg = pf_qg_; d.k_cache = kc; d.v_cache = vc; d.out_h = pf_mixh_; d.out_f = nullptr;
            d.n_heads = c.n_heads; d.n_kv_heads = c.n_kv_heads; d.max_ctx = opt_.max_ctx; d.pos0 = pos0; d.T = n;
            d.kv4 = p.kv4; d.stage_k = pf_sk_; d.stage_v = pf_sv_;
            attn_seq(d, s);
            linear_seq(lp_, at.o, pf_mixh_, n, pf_y_, s);
        }
        add_f32(X, pf_y_, (size_t)n * H, s);
        rmsnorm_rows_f16(X, L.post_norm, pf_xh_, H, n, c.eps, s);
        linear_seq(lp_, L.gate, pf_xh_, n, pf_g_, s);
        linear_seq(lp_, L.up, pf_xh_, n, pf_u_, s);
        silu_mul_f16(pf_g_, pf_u_, pf_acth_, (size_t)n * c.inter, s);
        linear_seq(lp_, L.down, pf_acth_, n, pf_y_, s);
        add_f32(X, pf_y_, (size_t)n * H, s);
    }

    void prefill_chunk(const int* tokens, int n, int pos0, size_t hbase, int t0) {
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        const int H = c.hidden;
        QV_CUDA_CHECK(cudaMemcpyAsync(pf_tok_, tokens, n * sizeof(int), cudaMemcpyHostToDevice, s));
        QV_CUDA_CHECK(cudaStreamSynchronize(s));
        embed_rows(embed_, pf_tok_, n, pf_x_, s);
        pf_dump(0, n, t0, hbase);
        int li = 0;
        for (const Layer& L : layers_) {
            prefill_layer(L, n, pos0, pf_x_);
            pf_dump(++li, n, t0, hbase);
        }
        if (mtp_pf_) mtp_prefill_chunk(tokens, n, pos0, mtp_pf_last_ && t0 + n >= mtp_pf_T_);
        if (opt_.row_logits) {  // eval mode: logits of every row to the callback
            std::vector<float> lg(c.vocab);
            for (int tt = 0; tt < n; ++tt) {
                QV_CUDA_CHECK(cudaMemcpyAsync(x_, pf_x_ + (size_t)tt * H, H * 4, cudaMemcpyDeviceToDevice, s));
                rmsnorm(x_, final_norm_, xn_, xs_, H, c.eps, s, q8_ ? &xq_ : nullptr);
                mv(head_, xn_, logits_, nullptr);
                QV_CUDA_CHECK(cudaMemcpyAsync(lg.data(), logits_, (size_t)c.vocab * 4, cudaMemcpyDeviceToHost, s));
                QV_CUDA_CHECK(cudaStreamSynchronize(s));
                opt_.row_logits(pos0 + tt, lg.data());
            }
        }
        if (opt_.dump_hidden) {  // test mode: logits for every token of the chunk (final norm + head per row)
            for (int tt = 0; tt < n; ++tt) {
                QV_CUDA_CHECK(cudaMemcpyAsync(x_, pf_x_ + (size_t)tt * H, H * 4, cudaMemcpyDeviceToDevice, s));
                rmsnorm(x_, final_norm_, xn_, xs_, H, c.eps, s, q8_ ? &xq_ : nullptr);
                mv(head_, xn_, logits_, nullptr);
                const size_t o = logits_all_.size();
                logits_all_.resize(o + c.vocab);
                QV_CUDA_CHECK(cudaMemcpyAsync(logits_all_.data() + o, logits_, (size_t)c.vocab * 4, cudaMemcpyDeviceToHost, s));
                QV_CUDA_CHECK(cudaStreamSynchronize(s));
            }
        }
    }

    // prefill scratch (allocated on first use; counted in state_bytes_)
    LinearPrefill lp_;
    int pf_chunk_ = 0;
    float *pf_x_ = nullptr, *pf_y_ = nullptr, *pf_qkv_ = nullptr, *pf_conv_ = nullptr, *pf_z_ = nullptr,
          *pf_alpha_ = nullptr, *pf_beta_ = nullptr, *pf_qg_ = nullptr, *pf_k_ = nullptr, *pf_v_ = nullptr,
          *pf_qr_ = nullptr, *pf_g_ = nullptr, *pf_u_ = nullptr;
    __half *pf_xh_ = nullptr, *pf_mixh_ = nullptr, *pf_acth_ = nullptr, *pf_sk_ = nullptr, *pf_sv_ = nullptr;
    int* pf_tok_ = nullptr;
    float* pf_m_ = nullptr;

    void ensure_prefill_buffers() {
        if (pf_chunk_) return;
        const Config& c = cfg_;
        const int Tc = pf_chunk_ = opt_.prefill_chunk;
        const int C = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim;
        const int VD = std::max(c.lin_v_heads * c.lin_v_dim, c.n_heads * c.head_dim);
        lp_.init(stream_);
        state_bytes_ += lp_.slab_elems * sizeof(__half);
        pf_x_ = dalloc<float>((size_t)Tc * c.hidden);
        pf_y_ = dalloc<float>((size_t)Tc * c.hidden);
        pf_xh_ = dalloc<__half>((size_t)Tc * c.hidden);
        pf_qkv_ = dalloc<float>((size_t)Tc * C);
        pf_conv_ = dalloc<float>((size_t)Tc * C);
        pf_z_ = dalloc<float>((size_t)Tc * c.lin_v_heads * c.lin_v_dim);
        pf_alpha_ = dalloc<float>((size_t)Tc * c.lin_v_heads);
        pf_beta_ = dalloc<float>((size_t)Tc * c.lin_v_heads);
        pf_mixh_ = dalloc<__half>((size_t)Tc * VD);
        pf_qg_ = dalloc<float>((size_t)Tc * 2 * c.n_heads * c.head_dim);
        pf_k_ = dalloc<float>((size_t)Tc * c.n_kv_heads * c.head_dim);
        pf_v_ = dalloc<float>((size_t)Tc * c.n_kv_heads * c.head_dim);
        pf_qr_ = dalloc<float>((size_t)Tc * c.n_heads * c.head_dim);
        pf_g_ = dalloc<float>((size_t)Tc * c.inter);
        pf_u_ = dalloc<float>((size_t)Tc * c.inter);
        pf_acth_ = dalloc<__half>((size_t)Tc * c.inter);
        pf_tok_ = dalloc<int>(Tc);
        if (kv4_on_) {  // fp16 staging of the chunk's post-RoPE K / V (shared by all attention layers)
            pf_sk_ = dalloc<__half>((size_t)Tc * c.n_kv_heads * c.head_dim);
            pf_sv_ = dalloc<__half>((size_t)Tc * c.n_kv_heads * c.head_dim);
        }
        if (opt_.mtp) pf_m_ = dalloc<float>((size_t)Tc * c.hidden);
    }

    int argmax() override {
        qv::cuda::argmax(logits_, cfg_.vocab, d_arg_, stream_);
        int r;
        QV_CUDA_CHECK(cudaMemcpyAsync(&r, d_arg_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return r;
    }
    void logits(std::vector<float>& out) override {
        out.resize(cfg_.vocab);
        QV_CUDA_CHECK(cudaMemcpyAsync(out.data(), logits_, cfg_.vocab * 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    void reset() override {
        QV_CUDA_CHECK(cudaMemsetAsync(conv_state_, 0, conv_bytes_, stream_));
        QV_CUDA_CHECK(cudaMemsetAsync(gdn_state_, 0, gdn_bytes_, stream_));
        hidden_.clear();
        mtp_ready_ = -1;
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bool has_state_save() const override { return true; }
    void state_save() override {
        if (!save_conv_ && conv_bytes_ > 0) save_conv_ = dalloc<float>(conv_bytes_ / 4);
        if (!save_gdn_ && gdn_bytes_ > 0) save_gdn_ = dalloc<float>(gdn_bytes_ / 4);
        if (conv_bytes_ > 0) QV_CUDA_CHECK(cudaMemcpyAsync(save_conv_, conv_state_, conv_bytes_, cudaMemcpyDeviceToDevice, stream_));
        if (gdn_bytes_ > 0) QV_CUDA_CHECK(cudaMemcpyAsync(save_gdn_, gdn_state_, gdn_bytes_, cudaMemcpyDeviceToDevice, stream_));
    }
    void state_restore() override {
        if (save_conv_ && conv_bytes_ > 0) QV_CUDA_CHECK(cudaMemcpyAsync(conv_state_, save_conv_, conv_bytes_, cudaMemcpyDeviceToDevice, stream_));
        if (save_gdn_ && gdn_bytes_ > 0) QV_CUDA_CHECK(cudaMemcpyAsync(gdn_state_, save_gdn_, gdn_bytes_, cudaMemcpyDeviceToDevice, stream_));
        mtp_ready_ = -1;
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bool has_checkpoint() const override { return true; }
    bool save_checkpoint(std::vector<uint8_t>& out) override {
        struct CkptHdr {
            uint32_t magic = 0x51564331u;
            int32_t mtp_ready = -1;
            uint64_t conv_b = 0, gdn_b = 0, ring_b = 0, hlast_b = 0, logits_b = 0;
        };
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        CkptHdr h;
        h.mtp_ready = mtp_ready_;
        h.conv_b = conv_bytes_;
        h.gdn_b = gdn_bytes_;
        const size_t n_kv_layers = (size_t)n_attn_ + (opt_.mtp ? 1 : 0);
        if (kv4_on_ && !kv4_.empty() && !kv4_[0].exact && kv4_ring_k_ && kv4_ring_v_) {
            h.ring_b = n_kv_layers * (size_t)cfg_.n_kv_heads * (size_t)KV4_RING * (size_t)KV4_D * sizeof(__half);
        }
        h.hlast_b = (opt_.mtp && hlast_) ? (size_t)cfg_.hidden * sizeof(float) : 0;
        h.logits_b = (size_t)cfg_.vocab * sizeof(float);
        const size_t tot = sizeof(CkptHdr) + h.conv_b + h.gdn_b + 2 * h.ring_b + h.hlast_b + h.logits_b;
        out.resize(tot);
        uint8_t* p = out.data();
        std::memcpy(p, &h, sizeof(h));
        p += sizeof(h);
        auto get = [&](const void* src, size_t b) {
            if (b > 0) QV_CUDA_CHECK(cudaMemcpy(p, src, b, cudaMemcpyDeviceToHost));
            p += b;
        };
        get(conv_state_, h.conv_b);
        get(gdn_state_, h.gdn_b);
        get(kv4_ring_k_, h.ring_b);
        get(kv4_ring_v_, h.ring_b);
        get(hlast_, h.hlast_b);
        get(logits_, h.logits_b);
        return true;
    }
    bool restore_checkpoint(const std::vector<uint8_t>& in) override {
        struct CkptHdr {
            uint32_t magic = 0x51564331u;
            int32_t mtp_ready = -1;
            uint64_t conv_b = 0, gdn_b = 0, ring_b = 0, hlast_b = 0, logits_b = 0;
        };
        if (in.size() < sizeof(CkptHdr)) return false;
        CkptHdr h;
        std::memcpy(&h, in.data(), sizeof(h));
        if (h.magic != 0x51564331u) return false;
        const size_t n_kv_layers = (size_t)n_attn_ + (opt_.mtp ? 1 : 0);
        const size_t ring_b = (kv4_on_ && !kv4_.empty() && !kv4_[0].exact && kv4_ring_k_ && kv4_ring_v_)
                                  ? n_kv_layers * (size_t)cfg_.n_kv_heads * (size_t)KV4_RING * (size_t)KV4_D * sizeof(__half)
                                  : 0;
        const size_t hlast_b = (opt_.mtp && hlast_) ? (size_t)cfg_.hidden * sizeof(float) : 0;
        const size_t logits_b = (size_t)cfg_.vocab * sizeof(float);
        if (h.conv_b != conv_bytes_ || h.gdn_b != gdn_bytes_ || h.ring_b != ring_b ||
            h.hlast_b != hlast_b || h.logits_b != logits_b) return false;
        const size_t tot = sizeof(CkptHdr) + conv_bytes_ + gdn_bytes_ + 2 * ring_b + hlast_b + logits_b;
        if (in.size() != tot) return false;
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        const uint8_t* p = in.data() + sizeof(CkptHdr);
        auto put = [&](void* dst, size_t b) {
            if (b > 0) QV_CUDA_CHECK(cudaMemcpy(dst, p, b, cudaMemcpyHostToDevice));
            p += b;
        };
        put(conv_state_, conv_bytes_);
        put(gdn_state_, gdn_bytes_);
        put(kv4_ring_k_, ring_b);
        put(kv4_ring_v_, ring_b);
        put(hlast_, hlast_b);
        put(logits_, logits_b);
        mtp_ready_ = h.mtp_ready;
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }
    struct KvCkptHdr {
        uint32_t magic = 0x51564B31u;  // "QVK1"
        int32_t kv_mode = 0;           // 0=fp32, 1=int4(kv4 blocks), 2=fp16(kv4 exact)
        int32_t kv4_kmode = 0;
        int32_t n_kv_layers = 0;
        int32_t n_kv_heads = 0;
        int32_t head_dim = 0;
        int32_t n_tokens = 0;
        int32_t used_blocks = 0;
        uint64_t payload_bytes = 0;
    };
    int kv_ckpt_mode() const {
        if (!kv4_on_) return 0;
        return (!kv4_.empty() && kv4_[0].exact) ? 2 : 1;
    }
    size_t checkpoint_bytes() const override {
        const size_t n_kv_layers = (size_t)n_attn_ + (opt_.mtp ? 1 : 0);
        const size_t ring_b = (kv4_on_ && !kv4_.empty() && !kv4_[0].exact && kv4_ring_k_ && kv4_ring_v_)
                                  ? n_kv_layers * (size_t)cfg_.n_kv_heads * (size_t)KV4_RING * (size_t)KV4_D * sizeof(__half)
                                  : 0;
        const size_t hlast_b = (opt_.mtp && hlast_) ? (size_t)cfg_.hidden * sizeof(float) : 0;
        const size_t logits_b = (size_t)cfg_.vocab * sizeof(float);
        return 48 + conv_bytes_ + gdn_bytes_ + 2 * ring_b + hlast_b + logits_b;  // sizeof(CkptHdr) == 48
    }
    bool has_kv_checkpoint() const override { return true; }
    size_t kv_checkpoint_bytes(int n_tokens) const override {
        if (n_tokens < 0 || n_tokens > opt_.max_ctx) return 0;
        const int n_kv_layers = n_attn_ + (opt_.mtp ? 1 : 0);
        const size_t rows = (size_t)n_kv_layers * (size_t)cfg_.n_kv_heads;
        const int mode = kv_ckpt_mode();
        size_t payload = 0;
        if (mode == 1) {
            if (kv4_.empty()) return 0;
            const int used_blk = std::min(kv4_[0].nblocks, kv4_q(kv4_[0], n_tokens) / KV4_BLK);
            payload = rows * (size_t)used_blk * KV4_REC_BYTES;
        } else if (mode == 2) {
            payload = 2 * rows * (size_t)n_tokens * KV4_D * sizeof(__half);
        } else {
            payload = 2 * rows * (size_t)n_tokens * (size_t)cfg_.head_dim * sizeof(float);
        }
        return sizeof(KvCkptHdr) + payload;
    }
    bool save_kv_prefix(std::vector<uint8_t>& out, int n_tokens) override {
        if (n_tokens < 0 || n_tokens > opt_.max_ctx) return false;
        const int n_kv_layers = n_attn_ + (opt_.mtp ? 1 : 0);
        const size_t rows = (size_t)n_kv_layers * (size_t)cfg_.n_kv_heads;
        const int mode = kv_ckpt_mode();
        KvCkptHdr h;
        h.kv_mode = mode;
        h.kv4_kmode = opt_.kv4_kmode;
        h.n_kv_layers = n_kv_layers;
        h.n_kv_heads = cfg_.n_kv_heads;
        h.head_dim = cfg_.head_dim;
        h.n_tokens = n_tokens;
        if (mode == 1) {
            if (kv4_.empty()) return false;
            h.used_blocks = std::min(kv4_[0].nblocks, kv4_q(kv4_[0], n_tokens) / KV4_BLK);
            h.payload_bytes = rows * (size_t)h.used_blocks * KV4_REC_BYTES;
        } else if (mode == 2) {
            h.payload_bytes = 2 * rows * (size_t)n_tokens * KV4_D * sizeof(__half);
        } else {
            h.payload_bytes = 2 * rows * (size_t)n_tokens * (size_t)cfg_.head_dim * sizeof(float);
        }
        out.resize(sizeof(KvCkptHdr) + h.payload_bytes);
        std::memcpy(out.data(), &h, sizeof(h));
        uint8_t* p = out.data() + sizeof(h);
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (mode == 1) {
            if (h.used_blocks > 0 && rows > 0) {
                const size_t row_w = (size_t)h.used_blocks * KV4_REC_BYTES;
                const size_t spitch = (size_t)kv4_[0].nblocks * KV4_REC_BYTES;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p, row_w, kv4_blocks_, spitch, row_w, rows, cudaMemcpyDeviceToHost, stream_));
            }
        } else if (mode == 2) {
            if (n_tokens > 0 && rows > 0) {
                const size_t row_w = (size_t)n_tokens * KV4_D * sizeof(__half);
                const size_t spitch = (size_t)opt_.max_ctx * KV4_D * sizeof(__half);
                const size_t half_b = rows * row_w;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p, row_w, kv4_ring_k_, spitch, row_w, rows, cudaMemcpyDeviceToHost, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p + half_b, row_w, kv4_ring_v_, spitch, row_w, rows, cudaMemcpyDeviceToHost, stream_));
            }
        } else {
            if (n_tokens > 0 && rows > 0) {
                const size_t row_w = (size_t)n_tokens * (size_t)cfg_.head_dim * sizeof(float);
                const size_t spitch = (size_t)opt_.max_ctx * (size_t)cfg_.head_dim * sizeof(float);
                const size_t half_b = rows * row_w;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p, row_w, k_cache_, spitch, row_w, rows, cudaMemcpyDeviceToHost, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(p + half_b, row_w, v_cache_, spitch, row_w, rows, cudaMemcpyDeviceToHost, stream_));
            }
        }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }
    bool restore_kv_prefix(const std::vector<uint8_t>& in) override {
        if (in.size() < sizeof(KvCkptHdr)) return false;
        KvCkptHdr h;
        std::memcpy(&h, in.data(), sizeof(h));
        if (h.magic != 0x51564B31u) return false;
        const int n_kv_layers = n_attn_ + (opt_.mtp ? 1 : 0);
        const int mode = kv_ckpt_mode();
        if (h.kv_mode != mode || h.kv4_kmode != opt_.kv4_kmode || h.n_kv_layers != n_kv_layers ||
            h.n_kv_heads != cfg_.n_kv_heads || h.head_dim != cfg_.head_dim ||
            h.n_tokens < 0 || h.n_tokens > opt_.max_ctx) return false;
        const size_t rows = (size_t)n_kv_layers * (size_t)cfg_.n_kv_heads;
        size_t expected_payload = 0;
        if (mode == 1) {
            if (kv4_.empty()) return false;
            const int expected_blk = std::min(kv4_[0].nblocks, kv4_q(kv4_[0], h.n_tokens) / KV4_BLK);
            if (h.used_blocks != expected_blk) return false;
            expected_payload = rows * (size_t)expected_blk * KV4_REC_BYTES;
        } else if (mode == 2) {
            expected_payload = 2 * rows * (size_t)h.n_tokens * KV4_D * sizeof(__half);
        } else {
            expected_payload = 2 * rows * (size_t)h.n_tokens * (size_t)cfg_.head_dim * sizeof(float);
        }
        if (h.payload_bytes != expected_payload || in.size() != sizeof(KvCkptHdr) + expected_payload) return false;
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        const uint8_t* p = in.data() + sizeof(KvCkptHdr);
        if (mode == 1) {
            if (h.used_blocks > 0 && rows > 0) {
                const size_t row_w = (size_t)h.used_blocks * KV4_REC_BYTES;
                const size_t dpitch = (size_t)kv4_[0].nblocks * KV4_REC_BYTES;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(kv4_blocks_, dpitch, p, row_w, row_w, rows, cudaMemcpyHostToDevice, stream_));
            }
        } else if (mode == 2) {
            if (h.n_tokens > 0 && rows > 0) {
                const size_t row_w = (size_t)h.n_tokens * KV4_D * sizeof(__half);
                const size_t dpitch = (size_t)opt_.max_ctx * KV4_D * sizeof(__half);
                const size_t half_b = rows * row_w;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(kv4_ring_k_, dpitch, p, row_w, row_w, rows, cudaMemcpyHostToDevice, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(kv4_ring_v_, dpitch, p + half_b, row_w, row_w, rows, cudaMemcpyHostToDevice, stream_));
            }
        } else {
            if (h.n_tokens > 0 && rows > 0) {
                const size_t row_w = (size_t)h.n_tokens * (size_t)cfg_.head_dim * sizeof(float);
                const size_t dpitch = (size_t)opt_.max_ctx * (size_t)cfg_.head_dim * sizeof(float);
                const size_t half_b = rows * row_w;
                QV_CUDA_CHECK(cudaMemcpy2DAsync(k_cache_, dpitch, p, row_w, row_w, rows, cudaMemcpyHostToDevice, stream_));
                QV_CUDA_CHECK(cudaMemcpy2DAsync(v_cache_, dpitch, p + half_b, row_w, row_w, rows, cudaMemcpyHostToDevice, stream_));
            }
        }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }
    void mtp_stats(MtpStats& s) const override { s = mtp_st_; }
    void set_mtp_enabled(bool on) override { mtp_on_ = on && opt_.mtp; mtp_ready_ = -1; }
    const std::vector<float>& hidden_dump() const override { return hidden_; }
    const std::vector<float>& logits_dump() const override { return logits_all_; }
    std::vector<float> logits_all_;
    void sync() override { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    size_t device_bytes_weights() const override { return weights_bytes_; }
    size_t device_bytes_state() const override { return state_bytes_; }

    // =============================================================================================================
    // MTP speculative decoding (exact greedy). Head semantics as in the CPU engine (docs/CPU_OPTIMIZATION.md): for
    // position q with final-normed main hidden h_q and next token t_{q+1}: x = fc([pre_norm_e(embed(t_{q+1})) |
    // pre_norm_h(h_q)]); one attention+MLP decoder layer at RoPE position q (own KV slot); mtp.norm; lm_head -> draft
    // for t_{q+2}. Cycle: verify [cur, d1..dD] with ONE weight pass (gemv_q8_multi, bit-identical per token to the
    // 1-token kernel; attention/conv/GDN/norm kernels are the 1-token kernels run per token in order), accept the
    // longest prefix of drafts equal to the model's own argmax, roll the recurrent state back from per-token snapshots.
    // The MTP head itself need not be bit exact (only proposes), so its batch/chain use the multi-token GEMV freely.
    // =============================================================================================================
    static constexpr int NTM_MAX = 4;
    Layer mtp_L_{};
    QMat mtp_fc_{};
    const float *mtp_norm_ = nullptr, *mtp_pre_e_ = nullptr, *mtp_pre_h_ = nullptr;
    int mtp_ready_ = -1;  // MTP KV + hlast_ valid for context [0, mtp_ready_)
    bool mtp_on_ = true, mtp_pf_ = false;
    int mtp_pf_T_ = 0;
    bool mtp_pf_last_ = true;
    bool mtp_prof_ = false;
    MtpStats mtp_st_;
    int NTM_ = 2;
    // verify buffers (token-major, token t at +t*stride)
    float *vx_, *vxn_, *vqkv_, *vconv_, *vz_, *valpha_, *vbeta_, *vmix_, *vqg_, *vkb_, *vvb_, *vqr_, *vg_, *vu_, *vact_,
        *vlogits_;
    Q8Buf vxq_{};
    float *me_, *mcat_, *mhid_, *mlogits_, *hlast_, *vpart_;
    int *d_v_, *d_m_, *d_draft_, *d_varg_, *h_a_, *h_b_, *h_res_;  // d_v_: [tok NTM | pos NTM]; d_m_: [tok | pos | chain pos]
    float *snap_gdn_ = nullptr, *snap_conv_ = nullptr;
    cudaGraphExec_t vgraph_[NTM_MAX + 1] = {};
    cudaGraphExec_t vgraph_spec_[NTM_MAX + 1] = {};
    static constexpr int SPEC_MAX_CAND = 4096;
    static constexpr int SPEC_FAST_SLICE = 256;
    int* d_scand_id_ = nullptr;
    float* d_scand_val_ = nullptr;
    CandMeta* d_smeta_ = nullptr;
    CandMeta* h_smeta_ = nullptr;
    int* h_scand_id_ = nullptr;
    float* h_scand_val_ = nullptr;
    float* h_sfull_p_ = nullptr;
    std::vector<float> h_sfull_;
    int* d_dq_id_ = nullptr;
    float* d_dq_prob_ = nullptr;
    int* d_dq_cnt_ = nullptr;
    int* h_dq_id_ = nullptr;
    float* h_dq_prob_ = nullptr;
    int* h_dq_cnt_ = nullptr;

    void alloc_mtp() {
        const Config& c = cfg_;
        const int N = NTM_ = opt_.mtp_depth + 1, H = c.hidden, V = c.vocab;
        const int C = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim;
        const int VD = c.lin_v_heads * c.lin_v_dim;
        const int MIXS = std::max(VD, c.n_heads * c.head_dim);
        vx_ = dalloc<float>((size_t)N * H);
        vxn_ = dalloc<float>((size_t)N * H);
        vqkv_ = dalloc<float>((size_t)N * C);
        vconv_ = dalloc<float>((size_t)N * C);
        vz_ = dalloc<float>((size_t)N * VD);
        valpha_ = dalloc<float>((size_t)N * c.lin_v_heads);
        vbeta_ = dalloc<float>((size_t)N * c.lin_v_heads);
        vmix_ = dalloc<float>((size_t)N * MIXS);
        vqg_ = dalloc<float>((size_t)N * 2 * c.n_heads * c.head_dim);
        vkb_ = dalloc<float>((size_t)N * c.n_kv_heads * c.head_dim);
        vvb_ = dalloc<float>((size_t)N * c.n_kv_heads * c.head_dim);
        vqr_ = dalloc<float>((size_t)N * c.n_heads * c.head_dim);
        vg_ = dalloc<float>((size_t)N * c.inter);
        vu_ = dalloc<float>((size_t)N * c.inter);
        vact_ = dalloc<float>((size_t)N * c.inter);
        vlogits_ = dalloc<float>((size_t)N * V);
        const int Kmax = std::max(c.inter, 2 * c.n_heads * c.head_dim);
        vxq_.q = dalloc<int>((size_t)N * Kmax / 4);
        vxq_.d = dalloc<float>((size_t)N * Kmax / 32);
        vxq_.sum = dalloc<float>((size_t)N * Kmax / 32);
        vxq_.qsum = dalloc<int>((size_t)N * Kmax / 32);
        vpart_ = dalloc<float>((size_t)N * c.n_heads * ATTN_MAX_SPLITS * (c.head_dim + 2));
        me_ = dalloc<float>((size_t)N * H);
        mcat_ = dalloc<float>((size_t)N * 2 * H);
        mhid_ = dalloc<float>(H);
        mlogits_ = dalloc<float>(V);
        hlast_ = dalloc<float>(H);
        d_v_ = dalloc<int>(2 * N);
        d_m_ = dalloc<int>(3 * N);
        d_draft_ = dalloc<int>(N);
        d_varg_ = dalloc<int>(N);
        QV_CUDA_CHECK(cudaMallocHost(&h_a_, 2 * N * sizeof(int)));
        QV_CUDA_CHECK(cudaMallocHost(&h_b_, 3 * N * sizeof(int)));
        QV_CUDA_CHECK(cudaMallocHost(&h_res_, 2 * N * sizeof(int)));
        snap_gdn_ = dalloc<float>((size_t)(N - 1) * gdn_bytes_ / 4);
        snap_conv_ = dalloc<float>((size_t)(N - 1) * conv_bytes_ / 4);
        d_scand_id_ = dalloc<int>((size_t)N * SPEC_MAX_CAND);
        d_scand_val_ = dalloc<float>((size_t)N * SPEC_MAX_CAND);
        d_smeta_ = dalloc<CandMeta>(N);
        QV_CUDA_CHECK(cudaMallocHost(&h_smeta_, N * sizeof(CandMeta)));
        QV_CUDA_CHECK(cudaMallocHost(&h_scand_id_, (size_t)N * SPEC_MAX_CAND * sizeof(int)));
        QV_CUDA_CHECK(cudaMallocHost(&h_scand_val_, (size_t)N * SPEC_MAX_CAND * sizeof(float)));
        QV_CUDA_CHECK(cudaMallocHost(&h_sfull_p_, (size_t)V * sizeof(float)));
        d_dq_id_ = dalloc<int>((size_t)N * DRAFT_MAX_CAND);
        d_dq_prob_ = dalloc<float>((size_t)N * DRAFT_MAX_CAND);
        d_dq_cnt_ = dalloc<int>(N);
        QV_CUDA_CHECK(cudaMallocHost(&h_dq_id_, (size_t)N * DRAFT_MAX_CAND * sizeof(int)));
        QV_CUDA_CHECK(cudaMallocHost(&h_dq_prob_, (size_t)N * DRAFT_MAX_CAND * sizeof(float)));
        QV_CUDA_CHECK(cudaMallocHost(&h_dq_cnt_, (size_t)N * sizeof(int)));
        const char* e = std::getenv("QV_MTP_PROF");
        mtp_prof_ = e && e[0] == '1';
        const char* g = std::getenv("QV_MTP_GRAPH");
        mtp_graph_ = !(g && g[0] == '0');
        argmax_advance(logits_, V, d_varg_, nullptr, nullptr, stream_);  // allocates the argmax scratch before any graph capture
        extract_top_candidates_batch(vlogits_, V, N, 10.0f, SPEC_MAX_CAND, d_scand_id_, d_scand_val_, d_smeta_, stream_);
        {
            GpuDraftParams wgp;
            wgp.max_extract = SPEC_MAX_CAND;
            sample_draft_gpu(mlogits_, V, wgp, 1ull, d_scand_id_, d_scand_val_, d_smeta_,
                             d_dq_id_, d_dq_prob_, d_dq_cnt_, d_draft_, stream_);
        }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    static Q8Buf q8at(const Q8Buf& b, int K, int t) {
        Q8Buf r;
        r.q = b.q + (size_t)t * (K >> 2);
        r.d = b.d + (size_t)t * (K >> 5);
        r.sum = b.sum + (size_t)t * (K >> 5);
        r.qsum = b.qsum + (size_t)t * (K >> 5);
        return r;
    }

    // one decoder layer for NT rows held in vx_ (residual, in place); row t sits at position dpos[t]. Every elementwise /
    // recurrent / attention kernel runs all NT rows in one launch with the exact per-row arithmetic of the 1-token path.
    void layer_multi(const Layer& L, int NT, const int* dpos, bool snap) {
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        const int H = c.hidden, I = c.inter;
        rmsnorm(vx_, L.in_norm, vxn_, xs_, H, c.eps, s, &vxq_, NT);
        if (!L.attn) {
            const GdnLayer& g = L.g;
            const int C = g.qkv.N, VD = g.z.N, Hv = c.lin_v_heads;
            const int MIXS = std::max(c.lin_v_heads * c.lin_v_dim, c.n_heads * c.head_dim);
            gemv_q8_multi(g.qkv, vxq_, NT, vqkv_, nullptr, s);
            gemv_q8_multi(g.z, vxq_, NT, vz_, nullptr, s);
            gemv_f16_dual(g.a, g.b, vxn_, valpha_, vbeta_, s, NT);
            float* cs = conv_state_ + (size_t)L.idx_kind * C * 3;
            float* gs = gdn_state_ + (size_t)L.idx_kind * Hv * c.lin_k_dim * c.lin_v_dim;
            const size_t conv_snap_off = (size_t)L.idx_kind * C * 3;
            const size_t gdn_snap_off = (size_t)L.idx_kind * Hv * c.lin_k_dim * c.lin_v_dim;
            conv_step_multi(vqkv_, g.conv_w, cs, vconv_, C, NT, snap ? snap_conv_ + conv_snap_off : nullptr, conv_bytes_ / 4, s);
            GdnStepArgs a{vconv_, valpha_, vbeta_, g.A_neg, g.dt_bias, vz_, g.norm_w, gs, vmix_,
                          c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim, c.eps, vxq_};
            if (g.out.K != VD || VD != Hv * c.lin_v_dim) throw std::runtime_error("gdn out_proj shape");
            a.nt = NT; a.conv_stride = C; a.out_stride = MIXS; a.snap_stride = gdn_bytes_ / 4;
            if (snap) a.snap = snap_gdn_ + gdn_snap_off;
            gdn_step(a, s);
            gemv_q8_multi(g.out, vxq_, NT, vx_, vx_, s);
        } else {
            const AttnLayer& at = L.at;
            const int MIXS = std::max(c.lin_v_heads * c.lin_v_dim, c.n_heads * c.head_dim);
            const int QGN = 2 * c.n_heads * c.head_dim, KN = c.n_kv_heads * c.head_dim, QRN = c.n_heads * c.head_dim;
            gemv_q8_multi(at.q, vxq_, NT, vqg_, nullptr, s);
            gemv_q8_multi(at.k, vxq_, NT, vkb_, nullptr, s);
            gemv_q8_multi(at.v, vxq_, NT, vvb_, nullptr, s);
            const size_t kv_layer = (size_t)c.n_kv_heads * opt_.max_ctx * c.head_dim;
            float* kc = kv4_on_ ? nullptr : k_cache_ + L.idx_kind * kv_layer;
            float* vc = kv4_on_ ? nullptr : v_cache_ + L.idx_kind * kv_layer;
            AttnPrepArgs p{vqg_, vkb_, vvb_, at.q_norm, at.k_norm, inv_freq_, vqr_, kc, vc,
                           c.n_heads, c.n_kv_heads, c.head_dim, c.rope_dim, opt_.max_ctx, dpos, c.eps};
            p.nt = NT; p.qg_stride = QGN; p.k_stride = KN; p.q_stride = QRN;
            p.kv4 = kv4_of(L.idx_kind);
            attn_prep(p, s);
            AttnDecodeArgs d{vqr_, vqg_, kc, vc, nullptr, vmix_, c.n_heads, c.n_kv_heads, opt_.max_ctx, dpos, vxq_};
            d.part = vpart_;
            d.kv4 = p.kv4;
            d.nt = NT; d.q_stride = QRN; d.qg_stride = QGN; d.out_stride = MIXS; d.q8_k = at.o.K;
            d.part_stride = (size_t)c.n_heads * ATTN_MAX_SPLITS * (c.head_dim + 2);
            attn_decode(d, s);
            gemv_q8_multi(at.o, vxq_, NT, vx_, vx_, s);
        }
        rmsnorm(vx_, L.post_norm, vxn_, xs_, H, c.eps, s, &vxq_, NT);
        gemv_q8_multi(L.gate, vxq_, NT, vg_, nullptr, s);
        gemv_q8_multi(L.up, vxq_, NT, vu_, nullptr, s);
        silu_mul(vg_, vu_, vact_, xs_, NT * I, s, &vxq_);  // rows are contiguous; chunk numbering matches q8 row stride I/32
        gemv_q8_multi(L.down, vxq_, NT, vx_, vx_, s);
    }

    // verify: tokens/positions already in d_v_ ([tok NT | pos NT]); logits rows -> vlogits_, per-row argmax -> d_varg_,
    // final-normed hidden rows -> vxn_, GDN/conv snapshots after tokens 0..NT-2 -> snap_*.
    void verify_forward(int NT, bool with_argmax = true) {
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        const int H = c.hidden, V = c.vocab;
        const int* dpos = d_v_ + NTM_;
        embed_rows(embed_, d_v_, NT, vx_, s);
        for (const Layer& L : layers_) layer_multi(L, NT, dpos, true);
        rmsnorm(vx_, final_norm_, vxn_, xs_, H, c.eps, s, &vxq_, NT);
        gemv_q8_multi(head_, vxq_, NT, vlogits_, nullptr, s);
        if (with_argmax) for (int t = 0; t < NT; ++t) argmax_advance(vlogits_ + (size_t)t * V, V, d_varg_ + t, nullptr, nullptr, s);
    }

    // MTP head over nt rows: next tokens at dnext[0..nt) (device), main hidden rows hid (normed, stride H), RoPE/KV
    // positions dpos[0..nt). Writes the MTP KV of those positions; draft of the last row -> dout, its normed MTP output
    // -> mhid_ (input of a chained draft).
    void mtp_run(int nt, const int* dnext, const float* hid, const int* dpos, int* dout, bool with_argmax = true) {
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        const int H = c.hidden, V = c.vocab;
        embed_rows(embed_, dnext, nt, me_, s);
        mtp_cat_rows_f32(me_, hid, mtp_pre_e_, mtp_pre_h_, mcat_, H, nt, c.eps, s);
        quantize_q8(mcat_, nt * 2 * H, vxq_, s);
        gemv_q8_multi(mtp_fc_, vxq_, nt, vx_, nullptr, s);
        layer_multi(mtp_L_, nt, dpos, false);
        rmsnorm(vx_ + (size_t)(nt - 1) * H, mtp_norm_, mhid_, xs_, H, c.eps, s, &xq_);
        gemv_q8(head_, xq_, mlogits_, nullptr, s);
        if (with_argmax) argmax_advance(mlogits_, V, dout, nullptr, nullptr, s);
    }

    // update MTP KV for rows (positions q0..q0+nt-1, next tokens nxt[], hidden rows hid) and draft Dd tokens -> d_draft_
    void mtp_draft(int nt, const int* nxt, const float* hid, int q0, int Dd) {
        for (int t = 0; t < nt; ++t) { h_b_[t] = nxt[t]; h_b_[NTM_ + t] = q0 + t; }
        for (int j = 1; j < Dd; ++j) h_b_[2 * NTM_ + j - 1] = q0 + nt - 1 + j;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_m_, h_b_, 3 * NTM_ * sizeof(int), cudaMemcpyHostToDevice, stream_));
        // all arguments are device-resident and fixed per call kind -> each kind is captured once as a graph
        const int kind = hid == hlast_ ? 0 : nt;  // 0: first draft of a decode call, 1..NTM: update from verify rows
        graph_run(kind, [&] { mtp_run(nt, d_m_, hid, d_m_ + NTM_, d_draft_); });
        for (int j = 1; j < Dd; ++j)
            graph_run(NTM_ + j, [&] { mtp_run(1, d_draft_ + j - 1, mhid_, d_m_ + 2 * NTM_ + j - 1, d_draft_ + j); });
    }

    cudaGraphExec_t mgraph_[2 * NTM_MAX + 2] = {};
    cudaGraphExec_t mgraph_spec_[2 * NTM_MAX + 2] = {};
    template <class F>
    void graph_run(int id, F&& f) {
        if (!mtp_graph_) { f(); return; }
        if (!mgraph_[id]) {
            QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            f();
            cudaGraph_t g;
            QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
            QV_CUDA_CHECK(cudaGraphInstantiate(&mgraph_[id], g, 0));
            QV_CUDA_CHECK(cudaGraphDestroy(g));
        }
        QV_CUDA_CHECK(cudaGraphLaunch(mgraph_[id], stream_));
    }
    template <class F>
    void graph_run_spec(int id, F&& f) {
        if (!mtp_graph_) { f(); return; }
        if (!mgraph_spec_[id]) {
            QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            f();
            cudaGraph_t g;
            QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
            QV_CUDA_CHECK(cudaGraphInstantiate(&mgraph_spec_[id], g, 0));
            QV_CUDA_CHECK(cudaGraphDestroy(g));
        }
        QV_CUDA_CHECK(cudaGraphLaunch(mgraph_spec_[id], stream_));
    }

    // after the final-norm rows of a prefill chunk: fill the MTP KV for the chunk's positions
    void mtp_prefill_chunk(const int* tokens, int n, int pos0, bool last) {
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        const int H = c.hidden;
        rmsnorm_rows_f32(pf_x_, final_norm_, pf_y_, H, n, c.eps, s);
        if (last) QV_CUDA_CHECK(cudaMemcpyAsync(hlast_, pf_y_ + (size_t)(n - 1) * H, H * 4, cudaMemcpyDeviceToDevice, s));
        const int nf = last ? n - 1 : n;  // the last prompt token's successor is not known yet
        if (nf <= 0) return;
        QV_CUDA_CHECK(cudaMemcpyAsync(pf_tok_, tokens + 1, nf * sizeof(int), cudaMemcpyHostToDevice, s));
        QV_CUDA_CHECK(cudaStreamSynchronize(s));
        embed_rows(embed_, pf_tok_, nf, pf_m_, s);
        mtp_cat_rows_f16(pf_m_, pf_y_, mtp_pre_e_, mtp_pre_h_, pf_acth_, H, nf, c.eps, s);
        linear_seq(lp_, mtp_fc_, pf_acth_, nf, pf_m_, s);
        prefill_layer(mtp_L_, nf, pos0, pf_m_);
    }

    int masked_argmax_from_gpu(const float* d_logits_row, TokenConstraint* tc) {
        const int V = cfg_.vocab;
        float* buf = h_sfull_p_;
        if (!buf) {
            if ((int)h_sfull_.size() < V) h_sfull_.resize(V);
            buf = h_sfull_.data();
        }
        QV_CUDA_CHECK(cudaMemcpyAsync(buf, d_logits_row, (size_t)V * 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (tc && !tc->awaiting()) tc->mask_logits(buf, (size_t)V);
        int best = 0;
        float bv = buf[0];
        for (int i = 1; i < V; ++i) {
            if (buf[i] > bv) { bv = buf[i]; best = i; }
        }
        return best;
    }

    void decode_mtp(int first, int start_pos, int n, std::vector<int>& out, TokenConstraint* tc = nullptr) {
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const Config& c = cfg_;
        const int H = c.hidden, V = c.vocab, D = opt_.mtp_depth;
        out.resize(n);
        int cur = first, p = start_pos, produced = 0;
        auto t0 = clk::now();
        int Dd = std::min(D, n - 1);
        mtp_draft(1, &cur, hlast_, p - 1, Dd);
        if (mtp_prof_) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); mtp_st_.draft_ms += ms(t0, clk::now()); }
        while (produced < n) {
            const int NT = std::min(D + 1, n - produced);
            if (NT == 1 && produced > 0 && p + 2 * D + 3 < opt_.max_ctx) break;
            if (NT == 1 || p + 2 * D + 3 >= opt_.max_ctx) {  // tail / context end: plain step (state stays identical to plain decode)
                t0 = clk::now();
                h_tp_[0] = cur;
                h_tp_[1] = p;
                QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, h_tp_, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
                forward();
                QV_CUDA_CHECK(cudaMemcpyAsync(hlast_, xn_, H * 4, cudaMemcpyDeviceToDevice, stream_));
                cur = argmax();
                if (tc && !tc->try_accept(cur)) {
                    cur = masked_argmax_from_gpu(logits_, tc);
                    (void)tc->try_accept(cur);
                }
                out[produced++] = cur;
                p += 1;
                mtp_st_.plain_steps++;
                mtp_st_.plain_ms += ms(t0, clk::now());
                if (tc && (tc->is_eog(cur) || tc->is_dead())) break;
                continue;
            }
            t0 = clk::now();
            for (int t = 0; t < NTM_; ++t) h_a_[NTM_ + t] = p + t;
            h_a_[0] = cur;
            QV_CUDA_CHECK(cudaMemcpyAsync(d_v_, h_a_, 2 * NTM_ * sizeof(int), cudaMemcpyHostToDevice, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(d_v_ + 1, d_draft_, (NT - 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(h_res_ + NTM_, d_draft_, (NT - 1) * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            run_verify(NT);
            QV_CUDA_CHECK(cudaMemcpyAsync(h_res_, d_varg_, NT * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto t1 = clk::now();
            int m = 0;
            int n_acc = 0;
            bool stop_chunk = false;
            if (!tc) {
                while (m < NT - 1 && h_res_[m] == h_res_[NTM_ + m]) ++m;
                n_acc = m;
            } else {
                for (int j = 0; j < NT; ++j) {
                    int target = h_res_[j];
                    if (!tc->try_accept(target)) {
                        target = masked_argmax_from_gpu(vlogits_ + (size_t)j * V, tc);
                        (void)tc->try_accept(target);
                    }
                    h_res_[j] = target;
                    const bool matched_draft = (j < NT - 1 && target == h_res_[NTM_ + j]);
                    if (matched_draft) n_acc = j + 1;
                    if (tc->is_eog(target) || tc->is_dead()) { m = j; stop_chunk = true; break; }
                    if (!matched_draft) { m = j; break; }
                }
            }
            mtp_st_.verify_ms += ms(t0, t1);
            mtp_st_.cycles++;
            mtp_st_.drafted += NT - 1;
            mtp_st_.accepted += n_acc;
            for (int j = 0; j <= m; ++j) out[produced++] = h_res_[j];
            QV_CUDA_CHECK(cudaMemcpyAsync(logits_, vlogits_ + (size_t)m * V, (size_t)V * 4, cudaMemcpyDeviceToDevice, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(hlast_, vxn_ + (size_t)m * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, stream_));
            if (m < NT - 1) {  // roll the recurrent state back to "after token m"
                QV_CUDA_CHECK(cudaMemcpyAsync(conv_state_, snap_conv_ + (size_t)m * (conv_bytes_ / 4), conv_bytes_, cudaMemcpyDeviceToDevice, stream_));
                QV_CUDA_CHECK(cudaMemcpyAsync(gdn_state_, snap_gdn_ + (size_t)m * (gdn_bytes_ / 4), gdn_bytes_, cudaMemcpyDeviceToDevice, stream_));
            }
            cur = h_res_[m];
            if (!stop_chunk && produced < n) {
                Dd = std::min(D, n - produced - 1);
                if (Dd > 0) mtp_draft(m + 1, h_res_, vxn_, p, Dd);
            }
            p += m + 1;
            if (mtp_prof_) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); mtp_st_.draft_ms += ms(t1, clk::now()); }
            if (stop_chunk) break;
        }
        out.resize(produced);
        mtp_ready_ = p;
    }

    void extract_gpu_rows(const float* d_logits_base, int n_rows, float margin) {
        const int V = cfg_.vocab;
        extract_top_candidates_batch(d_logits_base, V, n_rows, margin, SPEC_MAX_CAND,
                                     d_scand_id_, d_scand_val_, d_smeta_, stream_);
        QV_CUDA_CHECK(cudaMemcpyAsync(h_smeta_, d_smeta_, (size_t)n_rows * sizeof(CandMeta), cudaMemcpyDeviceToHost, stream_));
        for (int t = 0; t < n_rows; ++t) {
            QV_CUDA_CHECK(cudaMemcpyAsync(h_scand_id_ + (size_t)t * SPEC_MAX_CAND,
                                          d_scand_id_ + (size_t)t * SPEC_MAX_CAND,
                                          (size_t)SPEC_FAST_SLICE * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(h_scand_val_ + (size_t)t * SPEC_MAX_CAND,
                                          d_scand_val_ + (size_t)t * SPEC_MAX_CAND,
                                          (size_t)SPEC_FAST_SLICE * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        }
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        bool any_tail = false;
        for (int t = 0; t < n_rows; ++t) {
            const int cnt = h_smeta_[t].count;
            if (cnt > SPEC_FAST_SLICE && cnt <= SPEC_MAX_CAND) {
                const int rem = cnt - SPEC_FAST_SLICE;
                QV_CUDA_CHECK(cudaMemcpyAsync(h_scand_id_ + (size_t)t * SPEC_MAX_CAND + SPEC_FAST_SLICE,
                                              d_scand_id_ + (size_t)t * SPEC_MAX_CAND + SPEC_FAST_SLICE,
                                              (size_t)rem * sizeof(int), cudaMemcpyDeviceToHost, stream_));
                QV_CUDA_CHECK(cudaMemcpyAsync(h_scand_val_ + (size_t)t * SPEC_MAX_CAND + SPEC_FAST_SLICE,
                                              d_scand_val_ + (size_t)t * SPEC_MAX_CAND + SPEC_FAST_SLICE,
                                              (size_t)rem * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                any_tail = true;
            }
        }
        if (any_tail) QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void build_dist_for_row(const float* d_logits_base, int row, float margin,
                            const SpecSamplingParams& sp, TokenConstraint* tc, SpecDist& out_dist) {
        const int V = cfg_.vocab;
        bool ok = false;
        if (margin >= 0.0f) {
            const CandMeta& md = h_smeta_[row];
            ok = md.count > 0 && md.count <= SPEC_MAX_CAND &&
                 build_dist_from_candidates(h_scand_id_ + (size_t)row * SPEC_MAX_CAND,
                                            h_scand_val_ + (size_t)row * SPEC_MAX_CAND,
                                            md.count, md.mx, (double)md.sum_exp, sp, out_dist, tc, margin);
        }
        if (!ok) {
            float* buf = h_sfull_p_;
            if (!buf) {
                if ((int)h_sfull_.size() < V) h_sfull_.resize(V);
                buf = h_sfull_.data();
            }
            QV_CUDA_CHECK(cudaMemcpyAsync(buf, d_logits_base + (size_t)row * V, (size_t)V * 4, cudaMemcpyDeviceToHost, stream_));
            QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
            build_dist_dense(buf, V, sp, out_dist, tc);
        }
    }

    void build_dists_from_gpu(const float* d_logits_base, int n_rows, const SpecSamplingParams& sp,
                              SpecDist* out_dists, TokenConstraint* tc = nullptr) {
        const float margin = spec_candidate_margin(sp);
        if (margin >= 0.0f) extract_gpu_rows(d_logits_base, n_rows, margin);
        for (int t = 0; t < n_rows; ++t) build_dist_for_row(d_logits_base, t, margin, sp, tc, out_dists[t]);
    }

    void mtp_draft_gpu(int nt, const int* nxt, const float* hid, int q0, int Dd,
                       const SpecSamplingParams& sp, std::mt19937_64& rng) {
        for (int t = 0; t < nt; ++t) { h_b_[t] = nxt[t]; h_b_[NTM_ + t] = q0 + t; }
        for (int j = 1; j < Dd; ++j) h_b_[2 * NTM_ + j - 1] = q0 + nt - 1 + j;
        QV_CUDA_CHECK(cudaMemcpyAsync(d_m_, h_b_, 3 * NTM_ * sizeof(int), cudaMemcpyHostToDevice, stream_));
        GpuDraftParams gp;
        if (sp.min_p > 0.0f && sp.min_keep <= 1) {
            gp.margin = std::max(3.5f, -std::log(sp.min_p) + 0.1f);
        } else {
            gp.margin = std::clamp(7.5f * sp.temperature, 4.0f, 9.0f);
        }
        gp.temperature = sp.temperature;
        gp.top_k = sp.top_k;
        gp.top_p = sp.top_p;
        gp.min_p = sp.min_p;
        gp.min_keep = sp.min_keep;
        gp.max_extract = SPEC_MAX_CAND;
        const int kind = hid == hlast_ ? 0 : nt;
        graph_run_spec(kind, [&] { mtp_run(nt, d_m_, hid, d_m_ + NTM_, d_draft_, false); });
        sample_draft_gpu(mlogits_, cfg_.vocab, gp, rng(), d_scand_id_, d_scand_val_, d_smeta_,
                         d_dq_id_, d_dq_prob_, d_dq_cnt_, d_draft_, stream_);
        for (int j = 1; j < Dd; ++j) {
            graph_run_spec(NTM_ + j, [&] { mtp_run(1, d_draft_ + j - 1, mhid_, d_m_ + 2 * NTM_ + j - 1, d_draft_ + j, false); });
            sample_draft_gpu(mlogits_, cfg_.vocab, gp, rng(), d_scand_id_, d_scand_val_, d_smeta_,
                             d_dq_id_ + (size_t)j * DRAFT_MAX_CAND, d_dq_prob_ + (size_t)j * DRAFT_MAX_CAND,
                             d_dq_cnt_ + j, d_draft_ + j, stream_);
        }
    }

    void decode_mtp_speculative(int first, int start_pos, int n, const SpecSamplingParams& sp, std::vector<int>& out,
                                TokenConstraint* tc = nullptr) {
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const Config& c = cfg_;
        const int H = c.hidden, V = c.vocab, D = opt_.mtp_depth;
        const float margin = spec_candidate_margin(sp);
        std::mt19937_64 rng(sp.seed);
        out.resize(n);
        int cur = first, p = start_pos, produced = 0;
        int drafts[NTM_MAX] = {};
        SpecDist q_dists[NTM_MAX];
        auto t0 = clk::now();
        int Dd = std::min(D, n - 1);
        if (Dd > 0) mtp_draft_gpu(1, &cur, hlast_, p - 1, Dd, sp, rng);
        if (mtp_prof_) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); mtp_st_.draft_ms += ms(t0, clk::now()); }
        while (produced < n) {
            const int NT = std::min(D + 1, n - produced);
            if (NT == 1 && produced > 0 && p + 2 * D + 3 < opt_.max_ctx) break;
            if (NT == 1 || p + 2 * D + 3 >= opt_.max_ctx) {
                t0 = clk::now();
                h_tp_[0] = cur;
                h_tp_[1] = p;
                QV_CUDA_CHECK(cudaMemcpyAsync(d_tok_, h_tp_, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
                forward();
                QV_CUDA_CHECK(cudaMemcpyAsync(hlast_, xn_, H * 4, cudaMemcpyDeviceToDevice, stream_));
                SpecDist tail_p;
                build_dists_from_gpu(logits_, 1, sp, &tail_p, tc);
                cur = sample_dist(tail_p, rng);
                if (tc) (void)tc->try_accept(cur);
                out[produced++] = cur;
                p += 1;
                mtp_st_.plain_steps++;
                mtp_st_.plain_ms += ms(t0, clk::now());
                if (tc && (tc->is_eog(cur) || tc->is_dead())) break;
                continue;
            }
            t0 = clk::now();
            h_a_[0] = cur;
            for (int t = 0; t < NTM_; ++t) h_a_[NTM_ + t] = p + t;
            QV_CUDA_CHECK(cudaMemcpyAsync(d_v_, h_a_, 2 * NTM_ * sizeof(int), cudaMemcpyHostToDevice, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(d_v_ + 1, d_draft_, (NT - 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(h_res_ + NTM_, d_draft_, (NT - 1) * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(h_dq_cnt_, d_dq_cnt_, (NT - 1) * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(h_dq_id_, d_dq_id_, (size_t)(NT - 1) * DRAFT_MAX_CAND * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(h_dq_prob_, d_dq_prob_, (size_t)(NT - 1) * DRAFT_MAX_CAND * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            run_verify_spec(NT);
            if (margin >= 0.0f) extract_gpu_rows(vlogits_, NT, margin);
            else QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto t1 = clk::now();
            for (int j = 0; j < NT - 1; ++j) {
                drafts[j] = h_res_[NTM_ + j];
                const int qc = std::clamp(h_dq_cnt_[j], 0, DRAFT_MAX_CAND);
                q_dists[j].dense = false;
                q_dists[j].dn.clear();
                q_dists[j].sp.resize((size_t)qc);
                const int* qids = h_dq_id_ + (size_t)j * DRAFT_MAX_CAND;
                const float* qprs = h_dq_prob_ + (size_t)j * DRAFT_MAX_CAND;
                for (int i = 0; i < qc; ++i) q_dists[j].sp[(size_t)i] = {qids[i], qprs[i]};
                std::sort(q_dists[j].sp.begin(), q_dists[j].sp.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            int m = 0;
            int n_acc = 0;
            bool stop_chunk = false;
            int accepted_seq[NTM_MAX];
            for (int j = 0; j < NT; ++j) {
                SpecDist pj;
                build_dist_for_row(vlogits_, j, margin, sp, tc, pj);
                if (j < NT - 1 && spec_accept(pj, q_dists[j], drafts[j], rng)) {
                    accepted_seq[j] = drafts[j];
                    n_acc = j + 1;
                    if (tc) {
                        (void)tc->try_accept(drafts[j]);
                        if (tc->is_eog(drafts[j]) || tc->is_dead()) { m = j; stop_chunk = true; break; }
                    }
                    m = j + 1;
                } else {
                    const int final_tok = (j < NT - 1) ? sample_residual(pj, q_dists[j], rng) : sample_dist(pj, rng);
                    accepted_seq[j] = final_tok;
                    if (tc) {
                        (void)tc->try_accept(final_tok);
                        if (tc->is_eog(final_tok) || tc->is_dead()) stop_chunk = true;
                    }
                    m = j;
                    break;
                }
            }
            mtp_st_.verify_ms += ms(t0, t1);
            mtp_st_.cycles++;
            mtp_st_.drafted += NT - 1;
            mtp_st_.accepted += n_acc;
            for (int j = 0; j <= m; ++j) out[produced++] = accepted_seq[j];
            QV_CUDA_CHECK(cudaMemcpyAsync(logits_, vlogits_ + (size_t)m * V, (size_t)V * 4, cudaMemcpyDeviceToDevice, stream_));
            QV_CUDA_CHECK(cudaMemcpyAsync(hlast_, vxn_ + (size_t)m * H, (size_t)H * 4, cudaMemcpyDeviceToDevice, stream_));
            if (m < NT - 1) {
                QV_CUDA_CHECK(cudaMemcpyAsync(conv_state_, snap_conv_ + (size_t)m * (conv_bytes_ / 4), conv_bytes_, cudaMemcpyDeviceToDevice, stream_));
                QV_CUDA_CHECK(cudaMemcpyAsync(gdn_state_, snap_gdn_ + (size_t)m * (gdn_bytes_ / 4), gdn_bytes_, cudaMemcpyDeviceToDevice, stream_));
            }
            cur = accepted_seq[m];
            if (!stop_chunk && produced < n) {
                Dd = std::min(D, n - produced - 1);
                if (Dd > 0) mtp_draft_gpu(m + 1, accepted_seq, vxn_, p, Dd, sp, rng);
            }
            p += m + 1;
            if (mtp_prof_) { QV_CUDA_CHECK(cudaStreamSynchronize(stream_)); mtp_st_.draft_ms += ms(t1, clk::now()); }
            if (stop_chunk) break;
        }
        out.resize(produced);
        mtp_ready_ = p;
    }

    bool has_speculative_decode() const override { return opt_.mtp && mtp_on_; }
    bool has_constrained_mtp() const override { return opt_.mtp && mtp_on_; }
    void decode_speculative(int first, int start_pos, int n, const SpecSamplingParams& sp, std::vector<int>& out) override {
        decode_speculative_constrained(first, start_pos, n, sp, out, nullptr);
    }
    void decode_greedy_constrained(int first, int start_pos, int n, std::vector<int>& out, TokenConstraint* tc) override {
        if (start_pos + n > opt_.max_ctx) throw std::runtime_error("context overflow");
        if (opt_.mtp && mtp_on_ && n >= 2 && mtp_ready_ == start_pos) {
            decode_mtp(first, start_pos, n, out, tc);
            return;
        }
        out.clear();
        int tok = first;
        for (int i = 0; i < n; ++i) {
            step(tok, start_pos + i);
            int cand = argmax();
            if (tc && !tc->try_accept(cand)) {
                cand = masked_argmax_from_gpu(logits_, tc);
                (void)tc->try_accept(cand);
            }
            out.push_back(cand);
            tok = cand;
            if (tc && (tc->is_eog(cand) || tc->is_dead())) break;
        }
    }
    void decode_speculative_constrained(int first, int start_pos, int n, const SpecSamplingParams& sp,
                                        std::vector<int>& out, TokenConstraint* tc) override {
        if (start_pos + n > opt_.max_ctx) throw std::runtime_error("context overflow");
        if (opt_.mtp && mtp_on_ && n >= 2 && mtp_ready_ == start_pos) {
            decode_mtp_speculative(first, start_pos, n, sp, out, tc);
            return;
        }
        std::mt19937_64 rng(sp.seed);
        SpecDist dist;
        out.clear();
        int tok = first;
        for (int i = 0; i < n; ++i) {
            step(tok, start_pos + i);
            logits(h_sfull_);
            build_dist_dense(h_sfull_.data(), cfg_.vocab, sp, dist, tc);
            tok = sample_dist(dist, rng);
            if (tc) (void)tc->try_accept(tok);
            out.push_back(tok);
            if (tc && (tc->is_eog(tok) || tc->is_dead())) break;
        }
    }

    // The verify pass reads tokens/positions from device memory only, so it is captured once per row count and replayed.
    bool mtp_graph_ = true;
    void run_verify(int NT) {
        if (!mtp_graph_) { verify_forward(NT); return; }
        if (!vgraph_[NT]) {
            QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            verify_forward(NT);
            cudaGraph_t g;
            QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
            QV_CUDA_CHECK(cudaGraphInstantiate(&vgraph_[NT], g, 0));
            QV_CUDA_CHECK(cudaGraphDestroy(g));
        }
        QV_CUDA_CHECK(cudaGraphLaunch(vgraph_[NT], stream_));
    }
    void run_verify_spec(int NT) {
        if (!mtp_graph_) { verify_forward(NT, false); return; }
        if (!vgraph_spec_[NT]) {
            QV_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            verify_forward(NT, false);
            cudaGraph_t g;
            QV_CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
            QV_CUDA_CHECK(cudaGraphInstantiate(&vgraph_spec_[NT], g, 0));
            QV_CUDA_CHECK(cudaGraphDestroy(g));
        }
        QV_CUDA_CHECK(cudaGraphLaunch(vgraph_spec_[NT], stream_));
    }

    int n_seq() const override { return opt_.n_seq; }
    // views onto the current sequence: recurrent / KV state, logits, token+position buffer, decode graph (weights and scratch are shared)
    void select_seq(int i) override {
        if (i < 0 || i >= opt_.n_seq) throw std::runtime_error("sequence index out of range");
        if (i == cur_seq_) return;
        seqs_[cur_seq_] = snapshot();
        const SeqSt& q = seqs_[i];
        conv_state_ = q.conv_state; gdn_state_ = q.gdn_state; k_cache_ = q.k_cache; v_cache_ = q.v_cache; kv4_blocks_ = q.kv4_blocks;
        kv4_ring_k_ = q.kv4_ring_k; kv4_ring_v_ = q.kv4_ring_v; kv4_ = q.kv4; logits_ = q.logits;
        d_tok_ = q.d_tok; d_pos_ = q.d_tok + 1; d_gen_ = q.d_gen; graph_exec_ = q.graph_exec;
        cur_seq_ = i;
    }

   private:
    struct SeqSt {
        float *conv_state = nullptr, *gdn_state = nullptr, *k_cache = nullptr, *v_cache = nullptr, *logits = nullptr;
        uint8_t* kv4_blocks = nullptr;
        __half *kv4_ring_k = nullptr, *kv4_ring_v = nullptr;
        std::vector<Kv4Layer> kv4;
        int *d_tok = nullptr, *d_gen = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
    };
    std::vector<SeqSt> seqs_;
    int cur_seq_ = 0;
    SeqSt snapshot() const {
        SeqSt q;
        q.conv_state = conv_state_; q.gdn_state = gdn_state_; q.k_cache = k_cache_; q.v_cache = v_cache_; q.kv4_blocks = kv4_blocks_;
        q.kv4_ring_k = kv4_ring_k_; q.kv4_ring_v = kv4_ring_v_; q.kv4 = kv4_; q.logits = logits_; q.d_tok = d_tok_; q.d_gen = d_gen_;
        q.graph_exec = graph_exec_;
        return q;
    }
    EngineOptions opt_;
    cudaStream_t stream_{};
    char* weights_ = nullptr;
    size_t weights_bytes_ = 0, state_bytes_ = 0;
    std::vector<void*> allocs_;
    std::vector<Layer> layers_;
    QMat embed_, head_;
    const float* final_norm_ = nullptr;
    // state
    float *conv_state_ = nullptr, *gdn_state_ = nullptr, *save_conv_ = nullptr, *save_gdn_ = nullptr, *k_cache_ = nullptr, *v_cache_ = nullptr;
    size_t conv_bytes_ = 0, gdn_bytes_ = 0;
    // int4 KV (EngineOptions::kv_int4)
    bool kv4_on_ = false;
    uint8_t* kv4_blocks_ = nullptr;
    __half *kv4_ring_k_ = nullptr, *kv4_ring_v_ = nullptr;
    std::vector<Kv4Layer> kv4_;
    Kv4Layer kv4_of(int idx) const { return kv4_on_ ? kv4_[idx] : Kv4Layer{}; }
    // scratch
    float *x_, *xn_, *xs_, *qkv_, *z_, *alpha_, *beta_, *conv_out_, *mix_, *qg_, *kb_, *vb_, *qr_, *part_, *g_, *u_,
        *act_, *logits_, *inv_freq_;
    int *d_tok_, *d_arg_, *d_pos_, *d_gen_, *h_tp_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::vector<float> hidden_;

    template <class T>
    T* dalloc(size_t n) {
        void* p;
        QV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        QV_CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
        allocs_.push_back(p);
        state_bytes_ += n * sizeof(T);
        return (T*)p;
    }

    // GEMV input preparation + dispatch (H2: int8/dp4a path by default; QV_GEMV=fp32 selects the fp32 path)
    bool q8_ = true;
    Q8Buf xq_{};
    void prep(const float* x, int K, bool have_xsum) {
        if (q8_) quantize_q8(x, K, xq_, stream_);
        else if (!have_xsum) xsum32(x, xs_, K, stream_);
    }
    void mv(const QMat& W, const float* x, float* y, const float* res) {
        if (q8_) gemv_q8(W, xq_, y, res, stream_);
        else gemv_q(W, x, xs_, y, res, stream_);
    }

    void dump() {
        size_t o = hidden_.size();
        hidden_.resize(o + cfg_.hidden);
        QV_CUDA_CHECK(cudaMemcpyAsync(hidden_.data() + o, x_, cfg_.hidden * 4, cudaMemcpyDeviceToHost, stream_));
        QV_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
        if (t.kind != Kind::Quant) throw std::runtime_error(n + " is not quantized");
        if (t.group != 64 && t.group != 128) throw std::runtime_error("group must be 64 or 128");
        QMat m;
        m.q = (const uint32_t*)(weights_ + t.off_q);
        m.sm = (const __half2*)(weights_ + t.off_sm);
        if (t.meta8) {
            m.sm = nullptr;
            m.sm8 = (const uint16_t*)(weights_ + t.off_sm);
            m.rs = (const __half*)(weights_ + t.off_rs);
            m.zlo = -0.25f * (float)((1 << t.bits) - 1);
            m.zstep = (float)(1.5 * (double)((1 << t.bits) - 1) / 255.0);
        }
        m.N = (int)t.shape[0];
        m.K = (int)t.shape[1];
        m.bits = t.bits;
        m.gshift = t.group == 128 ? 7 : 6;
        return m;
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
        const std::string LM = "model.language_model.";
        embed_ = qmat(mf, LM + "embed_tokens.weight");
        head_ = qmat(mf, "lm_head.weight");
        final_norm_ = f32(mf, LM + "norm.weight");
        int ng = 0, na = 0;
        for (int i = 0; i < cfg_.n_layers; ++i) {
            std::string p = LM + "layers." + std::to_string(i) + ".";
            Layer L{};
            L.attn = cfg_.is_attn[i];
            L.idx_kind = L.attn ? na++ : ng++;
            L.in_norm = f32(mf, p + "input_layernorm.weight");
            L.post_norm = f32(mf, p + "post_attention_layernorm.weight");
            L.gate = qmat(mf, p + "mlp.gate_proj.weight");
            L.up = qmat(mf, p + "mlp.up_proj.weight");
            L.down = qmat(mf, p + "mlp.down_proj.weight");
            if (!L.attn) {
                std::string q = p + "linear_attn.";
                L.g.qkv = qmat(mf, q + "in_proj_qkv.weight");
                L.g.z = qmat(mf, q + "in_proj_z.weight");
                L.g.out = qmat(mf, q + "out_proj.weight");
                L.g.a = f16(mf, q + "in_proj_a.weight");
                L.g.b = f16(mf, q + "in_proj_b.weight");
                L.g.A_neg = f32(mf, q + "A_log");
                L.g.dt_bias = f32(mf, q + "dt_bias");
                L.g.conv_w = f32(mf, q + "conv1d.weight");
                L.g.norm_w = f32(mf, q + "norm.weight");
            } else {
                std::string q = p + "self_attn.";
                L.at.q = qmat(mf, q + "q_proj.weight");
                L.at.k = qmat(mf, q + "k_proj.weight");
                L.at.v = qmat(mf, q + "v_proj.weight");
                L.at.o = qmat(mf, q + "o_proj.weight");
                L.at.q_norm = f32(mf, q + "q_norm.weight");
                L.at.k_norm = f32(mf, q + "k_norm.weight");
            }
            layers_.push_back(L);
        }
        n_gdn_ = ng;
        n_attn_ = na;
        if (opt_.mtp) {
            const std::string p = "mtp.layers.0.";
            mtp_fc_ = qmat(mf, "mtp.fc.weight");
            mtp_norm_ = f32(mf, "mtp.norm.weight");
            mtp_pre_e_ = f32(mf, "mtp.pre_fc_norm_embedding.weight");
            mtp_pre_h_ = f32(mf, "mtp.pre_fc_norm_hidden.weight");
            Layer L{};
            L.attn = true;
            L.idx_kind = na;  // own KV cache slot after the main attention layers
            L.in_norm = f32(mf, p + "input_layernorm.weight");
            L.post_norm = f32(mf, p + "post_attention_layernorm.weight");
            L.gate = qmat(mf, p + "mlp.gate_proj.weight");
            L.up = qmat(mf, p + "mlp.up_proj.weight");
            L.down = qmat(mf, p + "mlp.down_proj.weight");
            L.at.q = qmat(mf, p + "self_attn.q_proj.weight");
            L.at.k = qmat(mf, p + "self_attn.k_proj.weight");
            L.at.v = qmat(mf, p + "self_attn.v_proj.weight");
            L.at.o = qmat(mf, p + "self_attn.o_proj.weight");
            L.at.q_norm = f32(mf, p + "self_attn.q_norm.weight");
            L.at.k_norm = f32(mf, p + "self_attn.k_norm.weight");
            mtp_L_ = L;
            if (mtp_fc_.K != 2 * cfg_.hidden || mtp_fc_.N != cfg_.hidden) throw std::runtime_error("unexpected mtp.fc shape");
        }
    }
    int n_gdn_ = 0, n_attn_ = 0;

    void alloc_state() {
        const Config& c = cfg_;
        const int conv_C = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim;
        conv_bytes_ = (size_t)n_gdn_ * conv_C * 3 * 4;
        gdn_bytes_ = (size_t)n_gdn_ * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim * 4;
        seqs_.assign(opt_.n_seq, SeqSt());
        for (int sq = 0; sq < opt_.n_seq; ++sq) {  // per-sequence state
            kv4_.clear();
            conv_state_ = dalloc<float>(conv_bytes_ / 4);
            gdn_state_ = dalloc<float>(gdn_bytes_ / 4);
            kv4_on_ = opt_.kv_mode != 0;
            const int n_kv_layers = n_attn_ + (opt_.mtp ? 1 : 0);
            if (kv4_on_) {
                if (c.head_dim != KV4_D || c.n_kv_heads * 6 != c.n_heads) throw std::runtime_error("int4/fp16 KV needs head_dim 256, 6 q heads per kv head");
                const bool exact = opt_.kv_mode == 2;  // fp16 KV: full-length fp16 ring, no quantization
                const int nblk = exact ? 1 : (opt_.max_ctx + KV4_BLK - 1) / KV4_BLK;
                const int ring_n = exact ? opt_.max_ctx : KV4_RING;
                kv4_blocks_ = dalloc<uint8_t>((size_t)n_kv_layers * c.n_kv_heads * nblk * KV4_REC_BYTES);
                kv4_ring_k_ = dalloc<__half>((size_t)n_kv_layers * c.n_kv_heads * ring_n * KV4_D);
                kv4_ring_v_ = dalloc<__half>((size_t)n_kv_layers * c.n_kv_heads * ring_n * KV4_D);
                for (int i = 0; i < n_kv_layers; ++i) {
                    Kv4Layer l;
                    l.blocks = kv4_blocks_ + (size_t)i * c.n_kv_heads * nblk * KV4_REC_BYTES;
                    l.ring_k = kv4_ring_k_ + (size_t)i * c.n_kv_heads * ring_n * KV4_D;
                    l.ring_v = kv4_ring_v_ + (size_t)i * c.n_kv_heads * ring_n * KV4_D;
                    l.nblocks = nblk;
                    l.kmode = opt_.kv4_kmode;
                    l.ring_n = ring_n;
                    l.exact = exact;
                    kv4_.push_back(l);
                }
            } else {
                size_t kv = (size_t)n_kv_layers * c.n_kv_heads * opt_.max_ctx * c.head_dim;
                k_cache_ = dalloc<float>(kv);
                v_cache_ = dalloc<float>(kv);
            }
            logits_ = dalloc<float>(c.vocab);
            d_tok_ = dalloc<int>(2);  // [token, pos] contiguous: one H2D copy per step
            d_pos_ = d_tok_ + 1;
            d_gen_ = dalloc<int>(opt_.max_ctx + 1);
            graph_exec_ = nullptr;
            seqs_[sq] = snapshot();
        }
        cur_seq_ = opt_.n_seq - 1;
        select_seq(0);
        x_ = dalloc<float>(c.hidden);
        xn_ = dalloc<float>(c.hidden);
        xs_ = dalloc<float>(std::max(c.inter, 2 * c.n_heads * c.head_dim) / 32 + 32);
        qkv_ = dalloc<float>(conv_C);
        conv_out_ = dalloc<float>(conv_C);
        z_ = dalloc<float>(c.lin_v_heads * c.lin_v_dim);
        alpha_ = dalloc<float>(c.lin_v_heads);
        beta_ = dalloc<float>(c.lin_v_heads);
        mix_ = dalloc<float>(std::max(c.lin_v_heads * c.lin_v_dim, c.n_heads * c.head_dim));
        qg_ = dalloc<float>(2 * c.n_heads * c.head_dim);
        kb_ = dalloc<float>(c.n_kv_heads * c.head_dim);
        vb_ = dalloc<float>(c.n_kv_heads * c.head_dim);
        qr_ = dalloc<float>(c.n_heads * c.head_dim);
        part_ = dalloc<float>((size_t)c.n_heads * ATTN_MAX_SPLITS * (c.head_dim + 2));
        g_ = dalloc<float>(c.inter);
        u_ = dalloc<float>(c.inter);
        act_ = dalloc<float>(c.inter);
        QV_CUDA_CHECK(cudaMallocHost(&h_tp_, 2 * sizeof(int)));
        d_arg_ = dalloc<int>(1);
        {
            const int Kmax = std::max(c.inter, 2 * c.n_heads * c.head_dim);
            xq_.q = dalloc<int>(Kmax / 4);
            xq_.d = dalloc<float>(Kmax / 32);
            xq_.sum = dalloc<float>(Kmax / 32);
            xq_.qsum = dalloc<int>(Kmax / 32);
            const char* e = std::getenv("QV_GEMV");
            q8_ = !(e && std::string(e) == "fp32");
        }
        if (opt_.mtp) alloc_mtp();
        // inv_freq computed in fp32 exactly like torch: 1 / base ** (arange(0, dim, 2) / dim)
        std::vector<float> inv(c.rope_dim / 2);
        for (int i = 0; i < c.rope_dim / 2; ++i)
            inv[i] = 1.0f / std::pow((float)c.rope_theta, (float)(2 * i) / (float)c.rope_dim);
        inv_freq_ = dalloc<float>(inv.size());
        QV_CUDA_CHECK(cudaMemcpy(inv_freq_, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
    }
};

}  // namespace

std::unique_ptr<Engine> make_cuda_engine(const ModelFile& mf, const EngineOptions& opt) {
    if (mf.cfg.is_inkling()) return make_cuda_inkling_engine(mf, opt);
    return std::make_unique<CudaEngine>(mf, opt);
}

}  // namespace qv
