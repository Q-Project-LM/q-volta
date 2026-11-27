// CPU backend: decode-step executor for Qwen3.5-family hybrid (GDN + gated attention) models.
// Mirrors the per-layer dataflow of cuda_engine.cu on top of the AVX2 kernels in src/backends/cpu.
//
// Weights are mmap'ed read-only from the .qv.bin and QMatC views point straight into the mapping. Each token runs
// as a short sequence of parallel regions on a spin-barrier pool; the projections of one region are flattened into a
// single list of row chunks (dynamic claiming) and small elementwise/quantize/conv/residual work is fused into the
// chunk epilogues so a GDN layer needs 5 dispatches and an attention layer 7.
#include <deque>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_gemv.h"
#include "ops.h"
#include "qv/engine.h"

namespace qv {
using namespace qv::cpu;

namespace {

// ---- optional per-phase timers (build with -DQV_CPU_PROF) -------------------------------------------------------------
enum Phase { P_SERIAL, P_PROJ_IN, P_GDN, P_ATTN, P_PROJ_OUT, P_GATEUP, P_DOWN, P_HEAD, P_ARGMAX, P_N };
const char* kPhaseName[P_N] = {"serial(embed,norm,quant,prep)", "in-proj gemv (+conv,a,b / q,k,v)", "gdn step",
                               "attn decode+merge", "out-proj gemv (+res)", "gate/up gemv (+silu,quant)",
                               "down gemv (+res)", "lm_head gemv", "argmax"};
#ifdef QV_CPU_PROF
struct Prof {
    double ns[P_N] = {};
    long steps = 0;
};
struct Scope {
    Prof& p; int ph; std::chrono::steady_clock::time_point t0;
    Scope(Prof& pp, int ph_) : p(pp), ph(ph_), t0(std::chrono::steady_clock::now()) {}
    ~Scope() { p.ns[ph] += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count(); }
};
#define QV_PHASE(ph) Scope scope_##__LINE__(prof_, ph)
#else
struct Prof { double ns[P_N] = {}; long steps = 0; };
#define QV_PHASE(ph) ((void)0)
#endif

// batched-prefill phase timers (wall time of each parallel region; always on, ~50 clock reads per layer-chunk)
enum PPhase { PP_EMBED, PP_NORMQ, PP_QKV, PP_ZPROJ, PP_AB, PP_CONV, PP_GDN, PP_ATT_PREP, PP_ATT, PP_QMIX, PP_OUT, PP_GATE, PP_UP,
              PP_ACT, PP_DOWN, PP_RESID, PP_HEAD, PP_N };
const char* kPPName[PP_N] = {"embed+rope tables", "rmsnorm+quantize", "gemm qkv / q,k,v", "gemm z", "alpha/beta dense", "conv1d",
                             "gdn recurrence", "attn prep (norm,rope,kv)", "attn scores+softmax+pv", "attn merge+quant",
                             "gemm out-proj", "gemm gate", "gemm up", "silu*mul+quantize", "gemm down", "residual add", "head"};

// fp16 dense row dot (F16C + FMA), K % 8 == 0
float dot_f16(const uint16_t* w, const float* x, int K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + k))), _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + k + 8))), _mm256_loadu_ps(x + k + 8), a1);
    }
    for (; k < K; k += 8)
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + k))), _mm256_loadu_ps(x + k), a0);
    a0 = _mm256_add_ps(a0, a1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// ---- one parallel region = list of row-chunk jobs ---------------------------------------------------------------------
enum class Epi { None, Conv, GateUp, Residual };
struct Job {
    const QMatC* W = nullptr;       // quantized matrix (null: dense fp16 rows)
    const QMatC* W2 = nullptr;      // GateUp: the "up" matrix (W is "gate")
    const uint16_t* dw = nullptr;   // dense fp16 [N][K]
    int N = 0, K = 0;
    const Q8ActC* x = nullptr;      // quantized activation (quantized matrices)
    const float* xf = nullptr;      // fp32 activation (dense rows)
    float* y = nullptr;             // output (GateUp: gate)
    float* y2 = nullptr;            // GateUp: up
    Epi epi = Epi::None;
    int rows_per_chunk = 16;
    // epilogue operands
    const float* conv_w = nullptr; float* conv_state = nullptr; float* conv_out = nullptr;  // Conv
    float* act = nullptr; Q8ActC* xq_act = nullptr;                                          // GateUp
    float* resid = nullptr;                                                                  // Residual: resid += y
    size_t first_chunk = 0, nchunks = 0;
};

struct GdnLayer {
    QMatC qkv, z, out;
    const uint16_t *a, *b;
    const float *A_neg, *dt_bias, *conv_w, *norm_w;
    int a_rows, b_rows;
};
struct AttnLayer {
    QMatC q, k, v, o;
    const float *q_norm, *k_norm;
};
struct Layer {
    bool attn;
    int idx_kind;
    const float *in_norm, *post_norm;
    QMatC gate, up, down;
    GdnLayer g;
    AttnLayer at;
};

class CpuEngine final : public Engine {
   public:
    CpuEngine(const ModelFile& mf, const EngineOptions& opt)
        : opt_(opt), pool_(opt.threads > 0 ? opt.threads : 32, pool_cpus()) {
        cfg_ = mf.cfg;
        map_weights(mf);
        bind(mf);
        alloc_state();
        research_init();
        for (int i = opt_.n_seq - 1; i >= 0; --i) { select_seq(i); reset(); }
    }
    ~CpuEngine() override {
#ifdef QV_CPU_PROF
        if (prof_.steps) {
            double tot = 0;
            for (double v : prof_.ns) tot += v;
            std::fprintf(stderr, "[prof] %ld forward passes, %.2f ms/token total in timed phases\n", prof_.steps,
                         tot / prof_.steps / 1e6);
            for (int i = 0; i < P_N; ++i)
                std::fprintf(stderr, "[prof]   %-34s %8.3f ms/token  (%4.1f%%)\n", kPhaseName[i],
                             prof_.ns[i] / prof_.steps / 1e6, 100.0 * prof_.ns[i] / tot);
        }
#endif
        if (pp_steps_ && std::getenv("QV_PREFILL_PROF")) {
            double tot = 0;
            for (double v : pp_ns_) tot += v;
            std::fprintf(stderr, "[prefill-prof] %ld tokens in %.1f ms (timed phases)\n", pp_steps_, tot / 1e6);
            for (int i = 0; i < PP_N; ++i)
                std::fprintf(stderr, "[prefill-prof]   %-30s %9.1f ms  (%4.1f%%)\n", kPPName[i], pp_ns_[i] / 1e6, 100.0 * pp_ns_[i] / tot);
        }
        if (map_) { if (owned_) std::free(map_); else munmap(map_, map_len_); }
        std::free(k_cache_);
        std::free(v_cache_);
        for (SeqSt& q : seqs_) { std::free(q.k_cache); std::free(q.v_cache); }
    }

    // ---- sequences: the current one lives in the members; the others are parked in seqs_ (the holder of the current sequence is empty)
    int n_seq() const override { return opt_.n_seq; }
    void select_seq(int i) override {
        if (i < 0 || i >= opt_.n_seq) throw std::runtime_error("sequence index out of range");
        if (i == cur_seq_) return;
        swap_state(seqs_[cur_seq_]);
        swap_state(seqs_[i]);
        cur_seq_ = i;
    }

    void step(int token, int pos) override {
        if (pos >= opt_.max_ctx) throw std::runtime_error("context overflow");
        mtp_ready_ = -1;
        forward(token, pos, true);
    }

    void prefill(const int* tokens, int T, int pos0) override {
        if (pos0 + T > opt_.max_ctx) throw std::runtime_error("context overflow");
        if (T <= 0) return;
        mtp_prefill_ = opt_.mtp && mtp_on_ && !opt_.dump_hidden && (pos0 == 0 || pos0 == mtp_ready_);
        mtp_ready_ = -1;
        if (std::getenv("QV_PREFILL_STEP")) {  // reference path: token-by-token (identical to step())
            for (int t = 0; t < T; ++t) {
                const bool need = opt_.dump_hidden || t == T - 1;
                forward(tokens[t], pos0 + t, need);
                if (opt_.dump_hidden) logits_all_.insert(logits_all_.end(), logits_.begin(), logits_.end());
            }
            return;
        }
        prefill_batched(tokens, T, pos0);
        research_dump(pos0 + T);
        if (mtp_prefill_) mtp_ready_ = pos0 + T;
    }

    void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool) override {
        if (start_pos + n > opt_.max_ctx) throw std::runtime_error("context overflow");
        out.resize(n);
        if (opt_.mtp && mtp_on_ && n >= 2 && mtp_ready_ == start_pos) { decode_mtp(first, start_pos, n, out); return; }
        int tok = first;
        for (int i = 0; i < n; ++i) {
            forward(tok, start_pos + i, true);
            tok = argmax();
            out[i] = tok;
        }
    }

    int argmax() override { return argmax_of(logits_.data()); }
    int argmax_of(const float* lg) {
        QV_PHASE(P_ARGMAX);
        const int V = cfg_.vocab, nt = pool_.size();
        std::vector<float>& bv = arg_val_;
        std::vector<int>& bi = arg_idx_;
        bv.assign(nt, -INFINITY);
        bi.assign(nt, 0x7fffffff);
        pool_.parallel_for((size_t)V, [&](size_t b, size_t e, int t) { argmax_range(lg, (int)b, (int)e, &bv[t], &bi[t]); }, 64);
        int r;
        argmax_merge(bv.data(), bi.data(), nt, &r);
        return r;
    }
    void logits(std::vector<float>& out) override { out = logits_; }
    void reset() override {
        std::memset(conv_state_.data(), 0, conv_state_.size() * 4);
        std::memset(gdn_state_.data(), 0, gdn_state_.size() * 4);
        hidden_.clear();
        logits_all_.clear();
        mtp_ready_ = -1;
    }
    bool has_state_save() const override { return true; }
    void state_save() override { conv_save_ = conv_state_; gdn_save_ = gdn_state_; }
    void state_restore() override { conv_state_ = conv_save_; gdn_state_ = gdn_save_; mtp_ready_ = -1; }
    bool has_checkpoint() const override { return true; }
    bool save_checkpoint(std::vector<uint8_t>& out) override {
        struct CpuCkptHdr {
            uint32_t magic = 0x43504331u;
            int32_t mtp_ready = -1;
            uint64_t conv_b = 0, gdn_b = 0, logits_b = 0, hlast_b = 0;
        } h;
        h.mtp_ready = mtp_ready_;
        h.conv_b = conv_state_.size() * sizeof(float);
        h.gdn_b = gdn_state_.size() * sizeof(float);
        h.logits_b = logits_.size() * sizeof(float);
        h.hlast_b = mtp_hid_last_.size() * sizeof(float);
        out.resize(sizeof(CpuCkptHdr) + h.conv_b + h.gdn_b + h.logits_b + h.hlast_b);
        uint8_t* p = out.data();
        std::memcpy(p, &h, sizeof(h));
        p += sizeof(h);
        auto put = [&](const void* src, size_t b) {
            if (b > 0) std::memcpy(p, src, b);
            p += b;
        };
        put(conv_state_.data(), h.conv_b);
        put(gdn_state_.data(), h.gdn_b);
        put(logits_.data(), h.logits_b);
        put(mtp_hid_last_.data(), h.hlast_b);
        return true;
    }
    bool restore_checkpoint(const std::vector<uint8_t>& in) override {
        struct CpuCkptHdr {
            uint32_t magic = 0x43504331u;
            int32_t mtp_ready = -1;
            uint64_t conv_b = 0, gdn_b = 0, logits_b = 0, hlast_b = 0;
        };
        if (in.size() < sizeof(CpuCkptHdr)) return false;
        CpuCkptHdr h;
        std::memcpy(&h, in.data(), sizeof(h));
        if (h.magic != 0x43504331u) return false;
        const size_t conv_b = conv_state_.size() * sizeof(float);
        const size_t gdn_b = gdn_state_.size() * sizeof(float);
        const size_t logits_b = logits_.size() * sizeof(float);
        const size_t hlast_b = mtp_hid_last_.size() * sizeof(float);
        if (h.conv_b != conv_b || h.gdn_b != gdn_b || h.logits_b != logits_b || h.hlast_b != hlast_b) return false;
        if (in.size() != sizeof(CpuCkptHdr) + conv_b + gdn_b + logits_b + hlast_b) return false;
        const uint8_t* p = in.data() + sizeof(CpuCkptHdr);
        auto get = [&](void* dst, size_t b) {
            if (b > 0) std::memcpy(dst, p, b);
            p += b;
        };
        get(conv_state_.data(), conv_b);
        get(gdn_state_.data(), gdn_b);
        get(logits_.data(), logits_b);
        get(mtp_hid_last_.data(), hlast_b);
        mtp_ready_ = h.mtp_ready;
        return true;
    }
    struct CpuKvHdr {
        uint32_t magic = 0x43504B31u;  // "CPK1"
        int32_t n_kv_layers = 0, n_kv_heads = 0, head_dim = 0, n_tokens = 0;
        uint64_t payload_bytes = 0;
    };
    size_t checkpoint_bytes() const override {
        return 40 + (conv_state_.size() + gdn_state_.size() + logits_.size() + mtp_hid_last_.size()) * sizeof(float);  // sizeof(CpuCkptHdr) == 40
    }
    bool has_kv_checkpoint() const override { return true; }
    size_t kv_checkpoint_bytes(int n_tokens) const override {
        if (n_tokens < 0 || n_tokens > opt_.max_ctx) return 0;
        const size_t rows = (size_t)(n_attn_ + (opt_.mtp ? 1 : 0)) * (size_t)cfg_.n_kv_heads;
        return sizeof(CpuKvHdr) + 2 * rows * (size_t)n_tokens * (size_t)cfg_.head_dim * sizeof(float);
    }
    bool save_kv_prefix(std::vector<uint8_t>& out, int n_tokens) override {
        if (n_tokens < 0 || n_tokens > opt_.max_ctx) return false;
        const int n_kv_layers = n_attn_ + (opt_.mtp ? 1 : 0);
        const size_t rows = (size_t)n_kv_layers * (size_t)cfg_.n_kv_heads;
        const size_t row_w = (size_t)n_tokens * (size_t)cfg_.head_dim * sizeof(float);
        const size_t spitch = (size_t)opt_.max_ctx * (size_t)cfg_.head_dim * sizeof(float);
        CpuKvHdr h;
        h.n_kv_layers = n_kv_layers;
        h.n_kv_heads = cfg_.n_kv_heads;
        h.head_dim = cfg_.head_dim;
        h.n_tokens = n_tokens;
        h.payload_bytes = 2 * rows * row_w;
        out.resize(sizeof(CpuKvHdr) + h.payload_bytes);
        std::memcpy(out.data(), &h, sizeof(h));
        uint8_t* p = out.data() + sizeof(h);
        if (row_w > 0 && rows > 0) {
            const uint8_t* ks = reinterpret_cast<const uint8_t*>(k_cache_);
            const uint8_t* vs = reinterpret_cast<const uint8_t*>(v_cache_);
            uint8_t* kp = p;
            uint8_t* vp = p + rows * row_w;
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(kp + r * row_w, ks + r * spitch, row_w);
                std::memcpy(vp + r * row_w, vs + r * spitch, row_w);
            }
        }
        return true;
    }
    bool restore_kv_prefix(const std::vector<uint8_t>& in) override {
        if (in.size() < sizeof(CpuKvHdr)) return false;
        CpuKvHdr h;
        std::memcpy(&h, in.data(), sizeof(h));
        if (h.magic != 0x43504B31u) return false;
        const int n_kv_layers = n_attn_ + (opt_.mtp ? 1 : 0);
        if (h.n_kv_layers != n_kv_layers || h.n_kv_heads != cfg_.n_kv_heads || h.head_dim != cfg_.head_dim ||
            h.n_tokens < 0 || h.n_tokens > opt_.max_ctx) return false;
        const size_t rows = (size_t)n_kv_layers * (size_t)cfg_.n_kv_heads;
        const size_t row_w = (size_t)h.n_tokens * (size_t)cfg_.head_dim * sizeof(float);
        const size_t expected_payload = 2 * rows * row_w;
        if (h.payload_bytes != expected_payload || in.size() != sizeof(CpuKvHdr) + expected_payload) return false;
        if (row_w > 0 && rows > 0) {
            const size_t dpitch = (size_t)opt_.max_ctx * (size_t)cfg_.head_dim * sizeof(float);
            const uint8_t* kp = in.data() + sizeof(CpuKvHdr);
            const uint8_t* vp = kp + rows * row_w;
            uint8_t* kd = reinterpret_cast<uint8_t*>(k_cache_);
            uint8_t* vd = reinterpret_cast<uint8_t*>(v_cache_);
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(kd + r * dpitch, kp + r * row_w, row_w);
                std::memcpy(vd + r * dpitch, vp + r * row_w, row_w);
            }
        }
        return true;
    }
    void set_sparse(const std::string& v) override {
        sp_ = SparseCfg();
        if (v == "off" || v.empty()) return;
        int a[5] = {64, 1024, 64, 16, 0};
        const char* p = v.c_str();
        for (int i = 0; i < 5 && *p; ++i) { a[i] = std::atoi(p); while (*p && *p != ',') ++p; if (*p) ++p; }
        sp_.on = true; sp_.sink = a[0]; sp_.recent = a[1]; sp_.bs = a[2]; sp_.k = a[3]; sp_.agg = a[4];
    }
    void mtp_stats(MtpStats& s) const override { s = mtp_st_; }
    void set_mtp_enabled(bool on) override { mtp_on_ = on && opt_.mtp; mtp_ready_ = -1; }
    const std::vector<float>& hidden_dump() const override { return hidden_; }
    const std::vector<float>& logits_dump() const override { return logits_all_; }
    void sync() override {}
    size_t device_bytes_weights() const override { return map_len_; }
    size_t device_bytes_state() const override { return state_bytes_; }

   private:
    EngineOptions opt_;
    ThreadPool pool_;
    Prof prof_;
    double pp_ns_[PP_N] = {};
    long pp_steps_ = 0;
    void* map_ = nullptr;
    size_t map_len_ = 0;
    const char* base_ = nullptr;
    std::vector<Layer> layers_;
    QMatC head_;
    QRows embed_{};
    const float* final_norm_ = nullptr;
    int n_gdn_ = 0, n_attn_ = 0;
    // state
    std::vector<float> conv_state_, gdn_state_;
    struct SeqSt { std::vector<float> conv, gdn, logits; float *k_cache = nullptr, *v_cache = nullptr; };
    std::vector<SeqSt> seqs_;
    int cur_seq_ = 0;
    void swap_state(SeqSt& q) { std::swap(conv_state_, q.conv); std::swap(gdn_state_, q.gdn); std::swap(logits_, q.logits); std::swap(k_cache_, q.k_cache); std::swap(v_cache_, q.v_cache); }
    float *k_cache_ = nullptr, *v_cache_ = nullptr;
    size_t state_bytes_ = 0;
    // scratch
    std::vector<float> x_, xn_, qkv_, conv_out_, z_, alpha_, beta_, mix_, qg_, kb_, vb_, qr_, part_, g_, u_, act_, tmp_,
        logits_, inv_freq_, rope_cs_, arg_val_;
    std::vector<int> arg_idx_;
    Q8ActC xq_, xq_mix_, xq_act_;
    std::vector<float> hidden_, logits_all_;
    // ---- MTP speculative decoding
    Layer mtp_L_{};
    QMatC mtp_fc_{};
    const float *mtp_norm_ = nullptr, *mtp_pre_e_ = nullptr, *mtp_pre_h_ = nullptr;
    int mtp_ready_ = -1;        // MTP KV + mtp_hid_last_ valid for context [0, mtp_ready_)
    bool mtp_prefill_ = false;
    bool mtp_on_ = true;
    bool mtp_hid_pre_ = false;  // QV_MTP_HIDDEN=pre: feed the pre-final-norm residual instead of the normed hidden
    std::vector<float> mtp_hid_last_, mtp_logits_, vlog_, mh_, mhc_, gdn_alt_, conv_alt_;
    bool spec_snap_ = false;    // pb_gdn: keep the recurrent state after token 0 of a 2-token verify in *_alt_
    MtpStats mtp_st_;
    int splits_ = 64;
    std::vector<Job> jobs_;

    // ---- research hooks (branch p9/tier; debug only, both off unless the env var is set)
    // QV_KVDUMP=dir[,qstride]: after prefill() write fp16 K/V of every attention layer (k_<i>.f16 / v_<i>.f16, [Hkv][n][D]),
    //   the post-norm/rope queries of every qstride-th position (default 64, from pos 1023) plus the last 32 positions
    //   (q_<i>.f16 [nq][Hq][D], qpos.i32) and meta.txt.
    // QV_SPARSE=sink,recent,block,k[,agg]: single-token attention (step/decode path only, not prefill) attends to
    //   sink + recent window + the top-k `block`-token blocks of the middle ranked per kv head by the Quest bound
    //   sum_d max(q_d*kmin_d, q_d*kmax_d) (agg 0: sum over the q heads of the group, 1: max). Teacher-forced quality probe.
    std::string kvdump_dir_;
    int kvdump_qstride_ = 64;
    std::vector<std::vector<float>> qsamp_, qlast_;
    std::vector<int> qsamp_pos_, qlast_pos_ = std::vector<int>(32, -1);
    struct SparseCfg { bool on = false; int sink = 64, recent = 1024, bs = 64, k = 16, agg = 0; } sp_;
    std::vector<float> kcomp_, vcomp_, sp_score_, conv_save_, gdn_save_;
    void research_init() {
        if (const char* e = std::getenv("QV_KVDUMP")) {
            std::string v = e;
            const size_t c = v.find(',');
            kvdump_dir_ = v.substr(0, c);
            if (c != std::string::npos) kvdump_qstride_ = std::atoi(v.c_str() + c + 1);
            qsamp_.assign(n_attn_, {});
            qlast_.assign(n_attn_, std::vector<float>((size_t)32 * cfg_.n_heads * cfg_.head_dim, 0.f));
        }
        if (const char* e = std::getenv("QV_SPARSE")) {
            int a[5] = {64, 1024, 64, 16, 0};
            const char* p = e;
            for (int i = 0; i < 5 && *p; ++i) { a[i] = std::atoi(p); while (*p && *p != ',') ++p; if (*p) ++p; }
            sp_.on = true; sp_.sink = a[0]; sp_.recent = a[1]; sp_.bs = a[2]; sp_.k = a[3]; sp_.agg = a[4];
        }
    }
    void research_capture_q(int layer_idx, int Tn, int p0) {
        if (kvdump_dir_.empty()) return;
        const size_t QD = (size_t)cfg_.n_heads * cfg_.head_dim;
        for (int t = 0; t < Tn; ++t) {
            const int pos = p0 + t;
            const float* q = pb_.qr.data() + (size_t)t * QD;
            if (pos >= 1023 && pos % kvdump_qstride_ == kvdump_qstride_ - 1) {
                if (layer_idx == 0) qsamp_pos_.push_back(pos);
                qsamp_[layer_idx].insert(qsamp_[layer_idx].end(), q, q + QD);
            }
            std::memcpy(qlast_[layer_idx].data() + (size_t)(pos % 32) * QD, q, QD * 4);
            if (layer_idx == 0) qlast_pos_[pos % 32] = pos;
        }
    }
    static void write_f16(FILE* f, const float* x, size_t n) {
        std::vector<uint16_t> b(n);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) _mm_storeu_si128((__m128i*)&b[i], _mm256_cvtps_ph(_mm256_loadu_ps(x + i), 0));
        for (; i < n; ++i) b[i] = _cvtss_sh(x[i], 0);
        std::fwrite(b.data(), 2, n, f);
    }
    void research_dump(int n) {
        if (kvdump_dir_.empty()) return;
        const Config& c = cfg_;
        const size_t D = c.head_dim, QD = (size_t)c.n_heads * D, kv_layer = (size_t)c.n_kv_heads * opt_.max_ctx * D;
        std::vector<int> lastp;
        for (int p = std::max(0, n - 32); p < n; ++p) lastp.push_back(p);
        for (int l = 0; l < n_attn_; ++l) {
            for (int w = 0; w < 3; ++w) {
                char fn[512];
                std::snprintf(fn, sizeof fn, "%s/%c_%d.f16", kvdump_dir_.c_str(), "kvq"[w], l);
                FILE* f = std::fopen(fn, "wb");
                if (!f) throw std::runtime_error(std::string("cannot write ") + fn);
                if (w < 2) {
                    const float* base = (w == 0 ? k_cache_ : v_cache_) + l * kv_layer;
                    for (int g = 0; g < c.n_kv_heads; ++g) write_f16(f, base + (size_t)g * opt_.max_ctx * D, (size_t)n * D);
                } else {
                    write_f16(f, qsamp_[l].data(), qsamp_[l].size());
                    for (int p : lastp) write_f16(f, qlast_[l].data() + (size_t)(p % 32) * QD, QD);
                }
                std::fclose(f);
            }
        }
        std::vector<int> qp = qsamp_pos_;
        qp.insert(qp.end(), lastp.begin(), lastp.end());
        char fn[512];
        std::snprintf(fn, sizeof fn, "%s/qpos.i32", kvdump_dir_.c_str());
        FILE* f = std::fopen(fn, "wb"); std::fwrite(qp.data(), 4, qp.size(), f); std::fclose(f);
        std::snprintf(fn, sizeof fn, "%s/meta.txt", kvdump_dir_.c_str());
        f = std::fopen(fn, "w");
        std::fprintf(f, "n=%d layers=%d hkv=%d hq=%d d=%d nq=%zu\n", n, n_attn_, c.n_kv_heads, c.n_heads, c.head_dim, qp.size());
        std::fclose(f);
        std::fprintf(stderr, "[kvdump] %d tokens, %zu queries/layer -> %s\n", n, qp.size(), kvdump_dir_.c_str());
    }
    // Sparse selection for one token (see QV_SPARSE). Returns false when the context is too short (dense attention).
    bool sparse_select(const float* kc, const float* vc, int n, int& Ns) {
        const Config& c = cfg_;
        const int D = c.head_dim, Hkv = c.n_kv_heads, Hq = c.n_heads, G = Hq / Hkv, bs = sp_.bs;
        const int rstart = ((n - sp_.recent) / bs) * bs, b0 = sp_.sink / bs, nb = rstart / bs - b0;
        if (rstart <= sp_.sink || nb <= sp_.k) return false;
        sp_score_.assign((size_t)Hkv * nb, 0.f);
        pool_.parallel_for_dyn((size_t)Hkv * nb, [&](size_t b, size_t e, int) {
            for (size_t w = b; w < e; ++w) {
                const int g = (int)(w / nb), blk = b0 + (int)(w % nb);
                float mn[256], mx[256];
                const float* K = kc + ((size_t)g * opt_.max_ctx + (size_t)blk * bs) * D;
                for (int d = 0; d < D; ++d) mn[d] = mx[d] = K[d];
                for (int j = 1; j < bs; ++j)
                    for (int d = 0; d < D; ++d) { mn[d] = std::min(mn[d], K[(size_t)j * D + d]); mx[d] = std::max(mx[d], K[(size_t)j * D + d]); }
                float agg = sp_.agg ? -INFINITY : 0.f;
                for (int h = 0; h < G; ++h) {
                    const float* q = qr_.data() + (size_t)(g * G + h) * D;
                    float s = 0.f;
                    for (int d = 0; d < D; ++d) s += std::max(q[d] * mn[d], q[d] * mx[d]);
                    agg = sp_.agg ? std::max(agg, s) : agg + s;
                }
                sp_score_[w] = agg;
            }
        }, 4);
        Ns = sp_.sink + (n - rstart) + sp_.k * bs;
        kcomp_.resize((size_t)Hkv * Ns * D); vcomp_.resize((size_t)Hkv * Ns * D);
        std::vector<int> idx(nb);
        for (int g = 0; g < Hkv; ++g) {
            for (int i = 0; i < nb; ++i) idx[i] = i;
            const float* sc = sp_score_.data() + (size_t)g * nb;
            std::nth_element(idx.begin(), idx.begin() + sp_.k, idx.end(), [&](int a, int b) { return sc[a] > sc[b]; });
            std::sort(idx.begin(), idx.begin() + sp_.k);
            float* ko = kcomp_.data() + (size_t)g * Ns * D;
            float* vo = vcomp_.data() + (size_t)g * Ns * D;
            const float* ki = kc + (size_t)g * opt_.max_ctx * D;
            const float* vi = vc + (size_t)g * opt_.max_ctx * D;
            auto put = [&](int from, int cnt) {
                std::memcpy(ko, ki + (size_t)from * D, (size_t)cnt * D * 4);
                std::memcpy(vo, vi + (size_t)from * D, (size_t)cnt * D * 4);
                ko += (size_t)cnt * D; vo += (size_t)cnt * D;
            };
            put(0, sp_.sink);
            for (int i = 0; i < sp_.k; ++i) put((b0 + idx[i]) * bs, bs);
            put(rstart, n - rstart);
        }
        return true;
    }

    static std::vector<int> pool_cpus() {
        std::vector<int> v;
        if (const char* e = std::getenv("QV_CPU_LIST")) {  // comma separated cpu ids
            for (const char* p = e; *p;) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
        }
        return v;
    }

    // ---------------------------------------------------------------- weights
    // Default: weights are read once into a 2 MiB-aligned anonymous buffer backed by transparent huge pages (the decode
    // loop streams the whole 12.6 GiB per token, so 4 KiB-page TLB misses on the file mapping matter; see
    // QV_WEIGHTS=mmap for the plain read-only file mapping with MAP_POPULATE).
    void map_weights(const ModelFile& mf) {
        int fd = ::open(mf.bin_path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + mf.bin_path);
        const char* mode = std::getenv("QV_WEIGHTS");
        const bool use_mmap = mode && std::string(mode) == "mmap";
        map_len_ = mf.bin_size;
        if (use_mmap) {
            map_ = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
            if (map_ == MAP_FAILED) throw std::runtime_error("mmap failed");
            if (std::getenv("QV_HUGEPAGE")) madvise(map_, map_len_, MADV_HUGEPAGE);
        } else {
            const size_t al = 2u << 20, len = (map_len_ + al - 1) / al * al;
            map_ = std::aligned_alloc(al, len);
            if (!map_) throw std::runtime_error("weight buffer allocation failed");
            owned_ = true;
            madvise(map_, len, MADV_HUGEPAGE);
            const size_t CH = 64u << 20, nch = (map_len_ + CH - 1) / CH;
            std::atomic<bool> bad{false};
            pool_.parallel_for_dyn(nch, [&](size_t b, size_t e, int) {
                for (size_t c = b; c < e; ++c) {
                    size_t off = c * CH, end = std::min(map_len_, off + CH);
                    while (off < end) {
                        ssize_t n = ::pread(fd, (char*)map_ + off, end - off, off);
                        if (n <= 0) { bad = true; break; }
                        off += (size_t)n;
                    }
                }
            }, 1);
            if (bad) throw std::runtime_error("short read on weights");
        }
        ::close(fd);
        base_ = (const char*)map_;
    }
    bool owned_ = false;
    // group-128 tensors: the CPU kernels use 64-groups, so (scale,min) of each 128-group is duplicated into two 64-groups
    // at load (identical arithmetic to the GPU g128 path; costs 0.25 bit/weight of RAM, no kernel change).
    std::deque<std::vector<uint16_t>> sm_expanded_;
    QMatC qmat(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::Quant) throw std::runtime_error(n + " is not quantized");
        if (t.group != 64 && t.group != 128) throw std::runtime_error("group must be 64 or 128");
        QMatC m;
        m.q = (const uint32_t*)(base_ + t.off_q);
        m.sm = (const uint16_t*)(base_ + t.off_sm);
        if (t.group == 128 || t.meta8) {
            // CPU kernels want fp16 (scale,min) per 64-group: expand g128 and/or decode meta8 (same fp32 formula as the
            // CUDA load_sm), then round to fp16.
            const int G = (int)(t.shape[1] / t.group);
            const size_t rep = t.group / 64, rows = (size_t)t.shape[0];
            std::vector<uint16_t> ex(rows * G * rep * 2);
            const uint16_t* sm8 = (const uint16_t*)(base_ + t.off_sm);
            const uint16_t* rs = t.meta8 ? (const uint16_t*)(base_ + t.off_rs) : nullptr;
            const float zlo = -0.25f * (float)((1 << t.bits) - 1), zst = (float)(1.5 * (double)((1 << t.bits) - 1) / 255.0);
            for (size_t r = 0; r < rows; ++r)
                for (int g = 0; g < G; ++g) {
                    uint16_t sh, mh;
                    if (t.meta8) {
                        const unsigned v = sm8[r * G + g];
                        const float sc = (float)(v & 255u) * (_cvtsh_ss(rs[r]) / 255.0f);
                        const float z = (float)(v >> 8) * zst + zlo;
                        sh = _cvtss_sh(sc, 0);
                        mh = _cvtss_sh(-(sc * z), 0);
                    } else {
                        sh = m.sm[(r * G + g) * 2];
                        mh = m.sm[(r * G + g) * 2 + 1];
                    }
                    for (size_t d = 0; d < rep; ++d) {
                        ex[((r * G + g) * rep + d) * 2] = sh;
                        ex[((r * G + g) * rep + d) * 2 + 1] = mh;
                    }
                }
            sm_expanded_.push_back(std::move(ex));
            m.sm = sm_expanded_.back().data();
        }
        m.N = (int)t.shape[0];
        m.K = (int)t.shape[1];
        m.bits = t.bits;
        return m;
    }
    const float* f32(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F32) throw std::runtime_error(n + " is not f32");
        return (const float*)(base_ + t.off);
    }
    const uint16_t* f16(const ModelFile& mf, const std::string& n, int& rows) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F16) throw std::runtime_error(n + " is not f16");
        rows = (int)t.shape[0];
        return (const uint16_t*)(base_ + t.off);
    }

    void bind(const ModelFile& mf) {
        const std::string LM = "model.language_model.";
        {
            QMatC e = qmat(mf, LM + "embed_tokens.weight");
            embed_ = QRows{e.q, e.sm, e.K, e.bits};
        }
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
                L.g.a = f16(mf, q + "in_proj_a.weight", L.g.a_rows);
                L.g.b = f16(mf, q + "in_proj_b.weight", L.g.b_rows);
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
            const char* hm = std::getenv("QV_MTP_HIDDEN");
            mtp_hid_pre_ = hm && std::string(hm) == "pre";
        }
    }

    static void prep_q8(Q8ActC& a, int K) {
        a.K = K;
        a.q.assign((size_t)K, 0);
        a.ds.assign((size_t)(K / 64) * 4, 0.f);
    }

    void alloc_state() {
        const Config& c = cfg_;
        const int conv_C = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim;
        const int VD = c.lin_v_heads * c.lin_v_dim, QD = c.n_heads * c.head_dim;
        conv_state_.assign((size_t)n_gdn_ * conv_C * 3, 0.f);
        gdn_state_.assign((size_t)n_gdn_ * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim, 0.f);
        const size_t kv = (size_t)(n_attn_ + (opt_.mtp ? 1 : 0)) * c.n_kv_heads * opt_.max_ctx * c.head_dim;
        k_cache_ = (float*)std::calloc(kv, 4);
        v_cache_ = (float*)std::calloc(kv, 4);
        if (!k_cache_ || !v_cache_) throw std::runtime_error("KV cache allocation failed");
        state_bytes_ = (conv_state_.size() + gdn_state_.size()) * 4 + kv * 8;
        if (opt_.n_seq < 1) opt_.n_seq = 1;
        if (opt_.n_seq > 1 && opt_.mtp) throw std::runtime_error("MTP supports one sequence (use n_seq = 1)");
        seqs_.assign(opt_.n_seq, SeqSt());
        for (int i = 1; i < opt_.n_seq; ++i) {
            SeqSt& q = seqs_[i];
            q.conv.assign(conv_state_.size(), 0.f); q.gdn.assign(gdn_state_.size(), 0.f); q.logits.assign(c.vocab, 0.f);
            q.k_cache = (float*)std::calloc(kv, 4); q.v_cache = (float*)std::calloc(kv, 4);
            if (!q.k_cache || !q.v_cache) throw std::runtime_error("KV cache allocation failed");
            state_bytes_ += (conv_state_.size() + gdn_state_.size() + c.vocab) * 4 + kv * 8;
        }
        x_.assign(c.hidden, 0.f); xn_.assign(c.hidden, 0.f);
        qkv_.assign(conv_C, 0.f); conv_out_.assign(conv_C, 0.f);
        z_.assign(VD, 0.f); alpha_.assign(c.lin_v_heads, 0.f); beta_.assign(c.lin_v_heads, 0.f);
        if (opt_.mtp) {
            gdn_alt_.assign(gdn_state_.size(), 0.f);
            conv_alt_.assign(conv_state_.size(), 0.f);
            mtp_hid_last_.assign(c.hidden, 0.f);
            mtp_logits_.assign(c.vocab, 0.f);
            vlog_.assign((size_t)2 * c.vocab, 0.f);
            mh_.assign((size_t)2 * c.hidden, 0.f);
        }
        mix_.assign(std::max(VD, QD), 0.f);
        qg_.assign(2 * QD, 0.f); kb_.assign(c.n_kv_heads * c.head_dim, 0.f); vb_.assign(kb_.size(), 0.f);
        qr_.assign(QD, 0.f);
        part_.assign(attn_part_floats(c.n_heads, splits_, c.head_dim), 0.f);
        g_.assign(c.inter, 0.f); u_.assign(c.inter, 0.f); act_.assign(c.inter, 0.f);
        tmp_.assign(c.hidden, 0.f);
        logits_.assign(c.vocab, 0.f);
        prep_q8(xq_, c.hidden);
        prep_q8(xq_mix_, std::max(VD, QD));
        prep_q8(xq_act_, c.inter);
        inv_freq_.resize(c.rope_dim / 2);
        for (int i = 0; i < c.rope_dim / 2; ++i)  // fp32, exactly like torch / the CUDA engine
            inv_freq_[i] = 1.0f / std::pow((float)c.rope_theta, (float)(2 * i) / (float)c.rope_dim);
        rope_cs_.assign(c.rope_dim, 0.f);
        jobs_.reserve(8);
    }

    // ---------------------------------------------------------------- region runner
    void run_jobs() {
        size_t total = 0;
        for (Job& j : jobs_) {
            j.first_chunk = total;
            j.nchunks = (size_t)(j.N + j.rows_per_chunk - 1) / j.rows_per_chunk;
            total += j.nchunks;
        }
        auto body = [&](size_t cb, size_t ce, int) {
            for (size_t c = cb; c < ce; ++c) {
                const Job* jp = &jobs_[0];
                for (const Job& j : jobs_) if (c >= j.first_chunk) jp = &j;
                const Job& j = *jp;
                const int r0 = (int)(c - j.first_chunk) * j.rows_per_chunk;
                const int r1 = std::min(j.N, r0 + j.rows_per_chunk);
                chunk(j, r0, r1);
            }
        };
        if (total == 1) body(0, 1, 0);
        else pool_.parallel_for_dyn(total, body, 1);
        jobs_.clear();
    }

    void chunk(const Job& j, int r0, int r1) {
        if (!j.W) {  // dense fp16 rows
            for (int r = r0; r < r1; ++r) j.y[r] = dot_f16(j.dw + (size_t)r * j.K, j.xf, j.K);
            return;
        }
        if (diag_fp_ && j.xf) diag_gemv(*j.W, j.xf, j.y, r0, r1); else gemv_q(*j.W, *j.x, j.y, r0, r1);
        switch (j.epi) {
            case Epi::None: break;
            case Epi::Conv: conv_step_range(j.y, j.conv_w, j.conv_state, j.conv_out, r0, r1); break;
            case Epi::Residual: add_range(j.resid, j.y, j.resid, r0, r1); break;
            case Epi::GateUp:
                if (diag_fp_ && j.xf) diag_gemv(*j.W2, j.xf, j.y2, r0, r1); else gemv_q(*j.W2, *j.x, j.y2, r0, r1);
                silu_mul_range(j.y, j.y2, j.act, r0, r1);
                quantize_act_q8_range(j.act, r0 / 64, (r1 + 63) / 64, *j.xq_act);
                break;
        }
    }

    // Diagnostic (QV_DIAG_FP=1): fp32 activations against dequantized weights, scalar and slow. Isolates the error that
    // comes from int8 activation quantization.
    bool diag_fp_ = std::getenv("QV_DIAG_FP") != nullptr;
    void diag_gemv(const QMatC& W, const float* xf, float* y, int r0, int r1) {
        std::vector<float> row(W.K);
        for (int r = r0; r < r1; ++r) {
            embed_row(QRows{W.q, W.sm, W.K, W.bits}, r, row.data());
            double a = 0;
            for (int k = 0; k < W.K; ++k) a += (double)row[k] * xf[k];
            y[r] = (float)a;
        }
    }

    Job qjob(const QMatC& W, const Q8ActC& x, float* y, const float* xf = nullptr) {
        Job j;
        j.W = &W; j.x = &x; j.N = W.N; j.K = W.K; j.y = y; j.xf = xf;
        return j;
    }

    void norm_quant(const float* w) {
        rmsnorm(x_.data(), w, xn_.data(), cfg_.hidden, cfg_.eps);
        quantize_act_q8_range(xn_.data(), 0, cfg_.hidden / 64, xq_);
    }

    void dump() { hidden_.insert(hidden_.end(), x_.begin(), x_.end()); }

    // ---------------------------------------------------------------- batched prefill
    // Chunk of Tc tokens flows through the layers together: projections use gemm_q (each weight tile unpacked once, applied
    // to all tokens of the chunk), elementwise/quantize steps are parallel over tokens, the GDN recurrence is sequential over
    // tokens but parallel over v-heads, attention is causal over (cache + chunk) and parallel over (kv head, token).
    struct PB {  // chunk scratch
        int Tc = 0;
        std::vector<float> x, xn, qkv, conv, z, al, be, mix, qg, kb, vb, qr, part, g, u, tmp, rope;
        Q8Batch xq, xqmix, xqact, xqfc;
    } pb_;

    template <class F>
    void timed(PPhase ph, F&& f) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        pp_ns_[ph] += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    }
    template <class F>
    void par_tok(int Tn, size_t grain, F&& f) {  // f(t_begin, t_end)
        pool_.parallel_for_dyn((size_t)Tn, [&](size_t b, size_t e, int) { f((int)b, (int)e); }, grain);
    }
    bool diag_gemv_ = std::getenv("QV_DIAG_GEMV") != nullptr;  // debug: per-token gemv_q instead of gemm_q
    void gemm(const QMatC& W, const Q8Batch& xb, int Tn, float* y) {
        if (!diag_gemv_) {
            // <= 4 tokens (decode-time batches): bit-exact multi-token GEMV; larger: tiled GEMM
            if (Tn <= 4) gemv_multi_q_par(pool_, W, xb, Tn, y, (size_t)W.N);
            else gemm_q_par(pool_, W, xb, Tn, y, (size_t)W.N);
            return;
        }
        Q8ActC a; a.K = xb.K; a.q.resize(xb.K); a.ds.resize((size_t)xb.G * 4);
        std::vector<float> row(W.N);
        for (int t = 0; t < Tn; ++t) {
            std::memcpy(a.q.data(), xb.q.data() + (size_t)t * xb.K, xb.K);
            std::memcpy(a.ds.data(), xb.ds.data() + (size_t)t * xb.G * 4, (size_t)xb.G * 16);
            gemv_q_par(pool_, W, a, y + (size_t)t * W.N);
        }
    }

    void pb_alloc(int Tc) {
        const Config& c = cfg_;
        if (pb_.Tc >= Tc) return;
        const int conv_C = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim;
        const int VD = c.lin_v_heads * c.lin_v_dim, QD = c.n_heads * c.head_dim, KVD = c.n_kv_heads * c.head_dim;
        const size_t T = Tc;
        pb_.Tc = Tc;
        pb_.x.assign(T * c.hidden, 0.f); pb_.xn.assign(T * c.hidden, 0.f); pb_.tmp.assign(T * c.hidden, 0.f);
        pb_.qkv.assign(T * conv_C, 0.f); pb_.conv.assign(T * conv_C, 0.f); pb_.z.assign(T * VD, 0.f);
        pb_.al.assign(T * c.lin_v_heads, 0.f); pb_.be.assign(T * c.lin_v_heads, 0.f);
        pb_.mix.assign(T * std::max(VD, QD), 0.f);
        pb_.qg.assign(T * 2 * QD, 0.f); pb_.kb.assign(T * KVD, 0.f); pb_.vb.assign(T * KVD, 0.f); pb_.qr.assign(T * QD, 0.f);
        pb_.part.assign(T * attn_part_floats(c.n_heads, 1, c.head_dim), 0.f);
        pb_.g.assign(T * c.inter, 0.f); pb_.u.assign(T * c.inter, 0.f);
        pb_.rope.assign(T * c.rope_dim, 0.f);
        pb_.xq.resize(Tc, c.hidden);
        pb_.xqmix.resize(Tc, std::max(VD, QD));
        pb_.xqact.resize(Tc, c.inter);
        if (opt_.mtp) pb_.xqfc.resize(Tc, 2 * c.hidden);
    }

    void prefill_batched(const int* tokens, int T, int pos0) {
        const Config& c = cfg_;
        const int H = c.hidden, NL = (int)layers_.size();
        int Tc = 256;  // default when the caller did not set EngineOptions::prefill_chunk (qv_run: 256 for cpu)
        if (const char* e = std::getenv("QV_CPU_PREFILL_CHUNK")) Tc = std::max(1, std::atoi(e));
        else if (opt_.prefill_chunk > 0) Tc = opt_.prefill_chunk;
        Tc = std::min(Tc, T);
        pb_alloc(Tc);
        const size_t dump_tok0 = opt_.dump_hidden ? hidden_.size() / ((size_t)(NL + 1) * H) : 0;
        if (opt_.dump_hidden) {
            hidden_.resize((dump_tok0 + T) * (size_t)(NL + 1) * H);
            logits_all_.resize((dump_tok0 + T) * (size_t)c.vocab);
        }
        pp_steps_ += T;
        for (int c0 = 0; c0 < T; c0 += Tc) {
            const int Tn = std::min(Tc, T - c0);
            const int p0 = pos0 + c0;
            float* X = pb_.x.data();
            timed(PP_EMBED, [&] {
                par_tok(Tn, 4, [&](int b, int e) {
                    for (int t = b; t < e; ++t) {
                        embed_row(embed_, tokens[c0 + t], X + (size_t)t * H);
                        attn_rope_table(p0 + t, inv_freq_.data(), c.rope_dim, pb_.rope.data() + (size_t)t * c.rope_dim);
                        if (opt_.dump_hidden)
                            std::memcpy(hidden_.data() + ((dump_tok0 + c0 + t) * (size_t)(NL + 1)) * H, X + (size_t)t * H, H * 4);
                    }
                });
            });
            for (int li = 0; li < NL; ++li) {
                pb_layer(layers_[li], Tn, p0);
                if (opt_.dump_hidden)
                    for (int t = 0; t < Tn; ++t)
                        std::memcpy(hidden_.data() + ((dump_tok0 + c0 + t) * (size_t)(NL + 1) + li + 1) * H, X + (size_t)t * H, H * 4);
            }
            if (opt_.dump_hidden) {  // logits of every token
                pb_norm_quant(final_norm_, Tn);
                timed(PP_HEAD, [&] { gemm_q_par(pool_, head_, pb_.xq, Tn, logits_all_.data() + (dump_tok0 + c0) * (size_t)c.vocab, (size_t)c.vocab); });
                if (c0 + Tn == T)
                    std::memcpy(logits_.data(), logits_all_.data() + (dump_tok0 + T - 1) * (size_t)c.vocab, (size_t)c.vocab * 4);
            } else if (c0 + Tn == T) {  // logits of the last token only (GEMV path, bandwidth-bound but a single token)
                timed(PP_HEAD, [&] {
                    std::memcpy(x_.data(), X + (size_t)(Tn - 1) * H, H * 4);
                    norm_quant(final_norm_);
                    jobs_.push_back(qjob(head_, xq_, logits_.data(), xn_.data()));
                    run_jobs();
                });
            }
            if (mtp_prefill_) mtp_prefill_chunk(tokens, T, c0, Tn, p0);
        }
    }

    // one decoder layer over the chunk in pb_.x (rope tables in pb_.rope must match positions p0..)
    void pb_layer(const Layer& L, int Tn, int p0) {
        const Config& c = cfg_;
        pb_norm_quant(L.in_norm, Tn);
        if (!L.attn) pb_gdn(L, Tn); else pb_attn(L, Tn, p0);
        pb_norm_quant(L.post_norm, Tn);
        timed(PP_GATE, [&] { gemm(L.gate, pb_.xq, Tn, pb_.g.data()); });
        timed(PP_UP, [&] { gemm(L.up, pb_.xq, Tn, pb_.u.data()); });
        timed(PP_ACT, [&] {
            par_tok(Tn, 1, [&](int b, int e) {
                for (int t = b; t < e; ++t) {
                    float* g = pb_.g.data() + (size_t)t * c.inter;
                    silu_mul_range(g, pb_.u.data() + (size_t)t * c.inter, g, 0, c.inter);
                    pb_.xqact.quantize_row(t, g, c.inter);
                }
            });
        });
        timed(PP_DOWN, [&] { gemm(L.down, pb_.xqact, Tn, pb_.tmp.data()); });
        pb_residual(Tn);
    }

    void pb_norm_quant(const float* w, int Tn) {
        timed(PP_NORMQ, [&] {
            const int H = cfg_.hidden;
            par_tok(Tn, 1, [&](int b, int e) {
                for (int t = b; t < e; ++t) {
                    rmsnorm(pb_.x.data() + (size_t)t * H, w, pb_.xn.data() + (size_t)t * H, H, cfg_.eps);
                    pb_.xq.quantize_row(t, pb_.xn.data() + (size_t)t * H, H);
                }
            });
        });
    }
    void pb_residual(int Tn) {
        timed(PP_RESID, [&] {
            const int H = cfg_.hidden;
            par_tok(Tn, 4, [&](int b, int e) { add_range(pb_.x.data() + (size_t)b * H, pb_.tmp.data() + (size_t)b * H, pb_.x.data() + (size_t)b * H, 0, (e - b) * H); });
        });
    }

    void pb_gdn(const Layer& L, int Tn) {
        const Config& c = cfg_;
        const GdnLayer& g = L.g;
        const int H = c.hidden, C = g.qkv.N, Hv = c.lin_v_heads, dv = c.lin_v_dim, VD = Hv * dv;
        const int mld = pb_.xqmix.K;
        timed(PP_QKV, [&] { gemm(g.qkv, pb_.xq, Tn, pb_.qkv.data()); });
        timed(PP_ZPROJ, [&] { gemm(g.z, pb_.xq, Tn, pb_.z.data()); });
        timed(PP_AB, [&] {
            par_tok(Tn, 2, [&](int b, int e) {
                for (int t = b; t < e; ++t) {
                    const float* xn = pb_.xn.data() + (size_t)t * H;
                    for (int r = 0; r < g.a_rows; ++r) pb_.al[(size_t)t * Hv + r] = dot_f16(g.a + (size_t)r * H, xn, H);
                    for (int r = 0; r < g.b_rows; ++r) pb_.be[(size_t)t * Hv + r] = dot_f16(g.b + (size_t)r * H, xn, H);
                }
            });
        });
        float* cs = conv_state_.data() + (size_t)L.idx_kind * C * 3;
        timed(PP_CONV, [&] {  // causal conv: sequential over tokens, parallel over channel blocks
            pool_.parallel_for_dyn((size_t)C, [&](size_t b, size_t e, int) {
                for (int t = 0; t < Tn; ++t) {
                    conv_step_range(pb_.qkv.data() + (size_t)t * C, g.conv_w, cs, pb_.conv.data() + (size_t)t * C, (int)b, (int)e);
                    if (spec_snap_ && Tn == 2 && t == 0) {  // state after token 0 (rollback point of the speculative token)
                        float* ca = conv_alt_.data() + (size_t)L.idx_kind * C * 3;
                        std::memcpy(ca + b * 3, cs + b * 3, (e - b) * 3 * sizeof(float));
                    }
                }
            }, 128);
        });
        timed(PP_GDN, [&] {  // delta-rule recurrence: sequential over tokens, parallel over v-heads
            float* st = gdn_state_.data() + (size_t)L.idx_kind * Hv * c.lin_k_dim * dv;
            pool_.parallel_for_dyn((size_t)Hv, [&](size_t b, size_t e, int) {
                for (int t = 0; t < Tn; ++t) {
                    GdnArgs a{pb_.conv.data() + (size_t)t * C, pb_.al.data() + (size_t)t * Hv, pb_.be.data() + (size_t)t * Hv, g.A_neg,
                              g.dt_bias, pb_.z.data() + (size_t)t * VD, g.norm_w, st, pb_.mix.data() + (size_t)t * mld,
                              c.lin_k_heads, Hv, c.lin_k_dim, dv, c.eps};
                    gdn_step_range(a, (int)b, (int)e);
                    if (spec_snap_ && Tn == 2 && t == 0) {
                        const size_t hs = (size_t)c.lin_k_dim * dv;
                        float* ga = gdn_alt_.data() + (size_t)L.idx_kind * Hv * hs;
                        std::memcpy(ga + b * hs, st + b * hs, (e - b) * hs * sizeof(float));
                    }
                }
            }, 1);
        });
        pb_mix_out(g.out, Tn, VD);
    }

    // quantize mix rows (first kuse columns) -> out-projection -> residual
    void pb_mix_out(const QMatC& Wo, int Tn, int kuse) {
        const int mld = pb_.xqmix.K;
        timed(PP_QMIX, [&] {
            par_tok(Tn, 1, [&](int b, int e) { for (int t = b; t < e; ++t) pb_.xqmix.quantize_row(t, pb_.mix.data() + (size_t)t * mld, kuse); });
        });
        timed(PP_OUT, [&] { gemm(Wo, pb_.xqmix, Tn, pb_.tmp.data()); });
        pb_residual(Tn);
    }

    void pb_attn(const Layer& L, int Tn, int p0) {
        const Config& c = cfg_;
        const AttnLayer& at = L.at;
        const int Hq = c.n_heads, Hkv = c.n_kv_heads, D = c.head_dim, QD = Hq * D, KVD = Hkv * D, mld = pb_.xqmix.K;
        timed(PP_QKV, [&] {
            gemm(at.q, pb_.xq, Tn, pb_.qg.data());
            gemm(at.k, pb_.xq, Tn, pb_.kb.data());
            gemm(at.v, pb_.xq, Tn, pb_.vb.data());
        });
        const size_t kv_layer = (size_t)Hkv * opt_.max_ctx * D;
        float* kc = k_cache_ + L.idx_kind * kv_layer;
        float* vc = v_cache_ + L.idx_kind * kv_layer;
        timed(PP_ATT_PREP, [&] {
            const int per = Hq + Hkv;
            pool_.parallel_for_dyn((size_t)Tn * per, [&](size_t b, size_t e, int) {
                for (size_t w = b; w < e; ++w) {
                    const int t = (int)(w / per), it = (int)(w % per);
                    AttnPrepArgs p{pb_.qg.data() + (size_t)t * 2 * QD, pb_.kb.data() + (size_t)t * KVD, pb_.vb.data() + (size_t)t * KVD,
                                   at.q_norm, at.k_norm, pb_.rope.data() + (size_t)t * c.rope_dim, pb_.qr.data() + (size_t)t * QD, kc, vc,
                                   Hq, Hkv, D, c.rope_dim, opt_.max_ctx, p0 + t, c.eps};
                    attn_prep_range(p, it, it + 1);
                }
            }, 8);
        });
        research_capture_q(L.idx_kind, Tn, p0);
        const size_t pf = attn_part_floats(Hq, 1, D);
        auto targs = [&](int t) {
            return AttnDecodeArgs{pb_.qr.data() + (size_t)t * QD, pb_.qg.data() + (size_t)t * 2 * QD, kc, vc, pb_.part.data() + (size_t)t * pf,
                                  pb_.mix.data() + (size_t)t * mld, Hq, Hkv, D, opt_.max_ctx, p0 + t + 1, 1};
        };
        if (Tn <= 4) {
            // decode-time batch: per token the same split-K attention as the single-token path (bit-identical results)
            timed(PP_ATT, [&] {
                for (int t = 0; t < Tn; ++t) {
                    AttnDecodeArgs d{pb_.qr.data() + (size_t)t * QD, pb_.qg.data() + (size_t)t * 2 * QD, kc, vc, part_.data(),
                                     pb_.mix.data() + (size_t)t * mld, Hq, Hkv, D, opt_.max_ctx, p0 + t + 1, splits_};
                    const int ns = attn_active_splits(d.n_keys, splits_);
                    pool_.parallel_for_dyn((size_t)Hkv * ns, [&](size_t b, size_t e, int) {
                        for (size_t w = b; w < e; ++w) {
                            const int item = (int)(w / ns) * splits_ + (int)(w % ns);
                            attn_decode_range(d, item, item + 1);
                        }
                    }, 1);
                    pool_.parallel_for_dyn((size_t)Hq, [&](size_t b, size_t e, int) { attn_merge_range(d, (int)b, (int)e); }, 1);
                    pb_.xqmix.quantize_row(t, d.out, QD);
                }
            });
            timed(PP_OUT, [&] { gemm(at.o, pb_.xqmix, Tn, pb_.tmp.data()); });
            pb_residual(Tn);
            return;
        }
        timed(PP_ATT, [&] {  // item = (kv head, token); kv-head-major so one head's K/V stays cache-resident; causal keys per token
            pool_.parallel_for_dyn((size_t)Hkv * Tn, [&](size_t b, size_t e, int) {
                for (size_t w = b; w < e; ++w) {
                    const int g = (int)(w / Tn), t = (int)(w % Tn);
                    attn_decode_range(targs(t), g, g + 1);
                }
            }, 2);
        });
        timed(PP_QMIX, [&] {
            par_tok(Tn, 1, [&](int b, int e) {
                for (int t = b; t < e; ++t) {
                    attn_merge_range(targs(t), 0, Hq);
                    pb_.xqmix.quantize_row(t, pb_.mix.data() + (size_t)t * mld, QD);
                }
            });
        });
        timed(PP_OUT, [&] { gemm(at.o, pb_.xqmix, Tn, pb_.tmp.data()); });
        pb_residual(Tn);
    }

    // ---------------------------------------------------------------- MTP (multi-token prediction) speculative decoding
    // Head semantics (Qwen3-Next / Qwen3.5 MTP, as in vLLM qwen3_next_mtp.py): for position q with main-model hidden h_q and
    // the NEXT token t_{q+1}:  e = pre_fc_norm_embedding(embed(t_{q+1})), h = pre_fc_norm_hidden(h_q),
    // x = fc(concat[e, h]); one full-attention decoder layer (own KV cache, RoPE position q); out = mtp.norm(x);
    // logits = lm_head(out) predict t_{q+2}. h_q is the main model's final-norm output (QV_MTP_HIDDEN=pre: residual stream).
    // Runs Tn consecutive positions pos0.. (next_tok[t] follows position pos0+t, hid = Tn rows of H floats) and writes
    // their KV; if draft != nullptr also returns the greedy draft token of the last position.
    void mtp_batch(const int* next_tok, const float* hid, int Tn, int pos0, int* draft) {
        const Config& c = cfg_;
        const int H = c.hidden;
        pb_alloc(Tn);
        par_tok(Tn, 1, [&](int b, int e) {
            for (int t = b; t < e; ++t) {
                float* xr = pb_.x.data() + (size_t)t * H;
                float* cat = pb_.g.data() + (size_t)t * c.inter;
                embed_row(embed_, next_tok[t], xr);
                rmsnorm(xr, mtp_pre_e_, cat, H, c.eps);
                rmsnorm(hid + (size_t)t * H, mtp_pre_h_, cat + H, H, c.eps);
                pb_.xqfc.quantize_row(t, cat, 2 * H);
                attn_rope_table(pos0 + t, inv_freq_.data(), c.rope_dim, pb_.rope.data() + (size_t)t * c.rope_dim);
            }
        });
        gemm(mtp_fc_, pb_.xqfc, Tn, pb_.x.data());
        pb_layer(mtp_L_, Tn, pos0);
        if (draft) {
            std::memcpy(x_.data(), pb_.x.data() + (size_t)(Tn - 1) * H, H * 4);
            norm_quant(mtp_norm_);
            jobs_.push_back(qjob(head_, xq_, mtp_logits_.data(), xn_.data()));
            run_jobs();
            *draft = argmax_of(mtp_logits_.data());
        }
    }

    // after the main layers of prefill chunk [c0, c0+Tn): fill the MTP KV for the chunk's positions (needs hidden + next token)
    void mtp_prefill_chunk(const int* tokens, int T, int c0, int Tn, int p0) {
        const auto t0 = std::chrono::steady_clock::now();
        const int H = cfg_.hidden;
        mhc_.resize((size_t)Tn * H);
        par_tok(Tn, 1, [&](int b, int e) {
            for (int t = b; t < e; ++t) {
                const float* xr = pb_.x.data() + (size_t)t * H;
                if (mtp_hid_pre_) std::memcpy(mhc_.data() + (size_t)t * H, xr, H * 4);
                else rmsnorm(xr, final_norm_, mhc_.data() + (size_t)t * H, H, cfg_.eps);
            }
        });
        const bool last = c0 + Tn == T;
        if (last) std::memcpy(mtp_hid_last_.data(), mhc_.data() + (size_t)(Tn - 1) * H, H * 4);
        const int nf = last ? Tn - 1 : Tn;  // the last prompt token's next token is not known yet
        if (nf > 0) mtp_batch(tokens + c0 + 1, mhc_.data(), nf, p0, nullptr);
        mtp_st_.prefill_mtp_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // Main-model pass over [cur at p, draft at p+1] sharing one weight read. Logits of both rows -> vlog_, hidden rows -> mh_.
    // GDN/conv state after token 0 is kept in the *_alt_ buffers (swap them in to roll the draft back).
    void verify2(int cur, int draft, int p) {
        const Config& c = cfg_;
        const int H = c.hidden;
        pb_alloc(2);
        const int tk[2] = {cur, draft};
        for (int t = 0; t < 2; ++t) {
            embed_row(embed_, tk[t], pb_.x.data() + (size_t)t * H);
            attn_rope_table(p + t, inv_freq_.data(), c.rope_dim, pb_.rope.data() + (size_t)t * c.rope_dim);
        }
        spec_snap_ = true;
        for (const Layer& L : layers_) pb_layer(L, 2, p);
        spec_snap_ = false;
        if (mtp_hid_pre_) std::memcpy(mh_.data(), pb_.x.data(), (size_t)2 * H * 4);
        pb_norm_quant(final_norm_, 2);
        if (!mtp_hid_pre_) std::memcpy(mh_.data(), pb_.xn.data(), (size_t)2 * H * 4);
        gemm(head_, pb_.xq, 2, vlog_.data());
    }

    void decode_mtp(int first, int start_pos, int n, std::vector<int>& out) {
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const int V = cfg_.vocab;
        int cur = first, p = start_pos, produced = 0, draft = 0;
        auto t0 = clk::now();
        mtp_batch(&cur, mtp_hid_last_.data(), 1, p - 1, &draft);
        mtp_st_.draft_ms += ms(t0, clk::now());
        while (produced < n) {
            if (n - produced == 1) {  // last token: a plain step keeps engine state identical to plain decode
                t0 = clk::now();
                forward(cur, p, true);
                out[produced++] = argmax();
                mtp_st_.plain_steps++;
                mtp_st_.plain_ms += ms(t0, clk::now());
                break;
            }
            t0 = clk::now();
            verify2(cur, draft, p);
            const int g0 = argmax_of(vlog_.data());
            const bool ok = g0 == draft;
            int g1 = 0;
            if (ok) g1 = argmax_of(vlog_.data() + V);
            auto t1 = clk::now();
            mtp_st_.verify_ms += ms(t0, t1);
            mtp_st_.cycles++;
            mtp_st_.drafted++;
            if (ok) {
                mtp_st_.accepted++;
                out[produced++] = g0;
                out[produced++] = g1;
                std::memcpy(logits_.data(), vlog_.data() + V, (size_t)V * 4);
                const int nt[2] = {draft, g1};
                mtp_batch(nt, mh_.data(), 2, p, &draft);
                p += 2;
                cur = g1;
            } else {
                std::swap(gdn_state_, gdn_alt_);  // roll the recurrent state back to "after cur"; KV rows beyond p are overwritten later
                std::swap(conv_state_, conv_alt_);
                out[produced++] = g0;
                std::memcpy(logits_.data(), vlog_.data(), (size_t)V * 4);
                mtp_batch(&g0, mh_.data(), 1, p, &draft);
                p += 1;
                cur = g0;
            }
            mtp_st_.draft_ms += ms(t1, clk::now());
        }
    }

    // ---------------------------------------------------------------- one token
    void forward(int token, int pos, bool want_logits) {
        const Config& c = cfg_;
        const int H = c.hidden;
#ifdef QV_CPU_PROF
        ++prof_.steps;
#endif
        {
            QV_PHASE(P_SERIAL);
            embed_row(embed_, token, x_.data());
            attn_rope_table(pos, inv_freq_.data(), c.rope_dim, rope_cs_.data());
            if (opt_.dump_hidden) dump();
        }
        for (const Layer& L : layers_) {
            { QV_PHASE(P_SERIAL); norm_quant(L.in_norm); }
            if (!L.attn) gdn_layer(L); else attn_layer(L, pos);
            // MLP
            { QV_PHASE(P_SERIAL); norm_quant(L.post_norm); }
            {
                QV_PHASE(P_GATEUP);
                Job j = qjob(L.gate, xq_, g_.data(), xn_.data());
                j.W2 = &L.up; j.y2 = u_.data(); j.epi = Epi::GateUp; j.rows_per_chunk = 64;
                j.act = act_.data(); j.xq_act = &xq_act_;
                jobs_.push_back(j);
                run_jobs();
            }
            {
                QV_PHASE(P_DOWN);
                Job j = qjob(L.down, xq_act_, tmp_.data(), act_.data());
                j.epi = Epi::Residual; j.resid = x_.data();
                jobs_.push_back(j);
                run_jobs();
            }
            if (opt_.dump_hidden) dump();
        }
        if (want_logits) {
            { QV_PHASE(P_SERIAL); norm_quant(final_norm_); }
            QV_PHASE(P_HEAD);
            jobs_.push_back(qjob(head_, xq_, logits_.data(), xn_.data()));
            run_jobs();
        }
        (void)H;
    }

    void gdn_layer(const Layer& L) {
        const Config& c = cfg_;
        const GdnLayer& g = L.g;
        const int C = g.qkv.N;
        const int Hv = c.lin_v_heads, dv = c.lin_v_dim;
        {
            QV_PHASE(P_PROJ_IN);
            Job a = qjob(g.qkv, xq_, qkv_.data(), xn_.data());
            a.epi = Epi::Conv; a.conv_w = g.conv_w; a.conv_out = conv_out_.data();
            a.conv_state = conv_state_.data() + (size_t)L.idx_kind * C * 3;
            jobs_.push_back(a);
            jobs_.push_back(qjob(g.z, xq_, z_.data(), xn_.data()));
            Job b; b.dw = g.a; b.N = g.a_rows; b.K = c.hidden; b.xf = xn_.data(); b.y = alpha_.data(); jobs_.push_back(b);
            Job d; d.dw = g.b; d.N = g.b_rows; d.K = c.hidden; d.xf = xn_.data(); d.y = beta_.data(); jobs_.push_back(d);
            run_jobs();
        }
        {
            QV_PHASE(P_GDN);
            GdnArgs a{conv_out_.data(), alpha_.data(), beta_.data(), g.A_neg, g.dt_bias, z_.data(), g.norm_w,
                      gdn_state_.data() + (size_t)L.idx_kind * Hv * c.lin_k_dim * dv, mix_.data(),
                      c.lin_k_heads, Hv, c.lin_k_dim, dv, c.eps};
            pool_.parallel_for_dyn((size_t)Hv, [&](size_t b, size_t e, int) {
                gdn_step_range(a, (int)b, (int)e);
                quantize_act_q8_range(mix_.data(), (int)b * dv / 64, (int)e * dv / 64, xq_mix_);
            }, 1);
        }
        {
            QV_PHASE(P_PROJ_OUT);
            Job j = qjob(g.out, xq_mix_, tmp_.data(), mix_.data());
            j.epi = Epi::Residual; j.resid = x_.data();
            jobs_.push_back(j);
            run_jobs();
        }
    }

    void attn_layer(const Layer& L, int pos) {
        const Config& c = cfg_;
        const AttnLayer& at = L.at;
        const int Hq = c.n_heads, Hkv = c.n_kv_heads, D = c.head_dim;
        {
            QV_PHASE(P_PROJ_IN);
            jobs_.push_back(qjob(at.q, xq_, qg_.data(), xn_.data()));
            jobs_.push_back(qjob(at.k, xq_, kb_.data(), xn_.data()));
            jobs_.push_back(qjob(at.v, xq_, vb_.data(), xn_.data()));
            run_jobs();
        }
        const size_t kv_layer = (size_t)Hkv * opt_.max_ctx * D;
        float* kc = k_cache_ + L.idx_kind * kv_layer;
        float* vc = v_cache_ + L.idx_kind * kv_layer;
        {
            QV_PHASE(P_SERIAL);
            AttnPrepArgs p{qg_.data(), kb_.data(), vb_.data(), at.q_norm, at.k_norm, rope_cs_.data(), qr_.data(), kc, vc,
                           Hq, Hkv, D, c.rope_dim, opt_.max_ctx, pos, c.eps};
            attn_prep_range(p, 0, Hq + Hkv);
        }
        {
            QV_PHASE(P_ATTN);
            const float *Ku = kc, *Vu = vc;
            int mcx = opt_.max_ctx, nkeys = pos + 1, Ns = 0;
            if (sp_.on && sparse_select(kc, vc, pos + 1, Ns)) { Ku = kcomp_.data(); Vu = vcomp_.data(); mcx = Ns; nkeys = Ns; }
            AttnDecodeArgs d{qr_.data(), qg_.data(), Ku, Vu, part_.data(), mix_.data(), Hq, Hkv, D, mcx,
                             nkeys, splits_};
            const int ns = attn_active_splits(d.n_keys, splits_);
            pool_.parallel_for_dyn((size_t)Hkv * ns, [&](size_t b, size_t e, int) {
                for (size_t w = b; w < e; ++w) {
                    const int item = (int)(w / ns) * splits_ + (int)(w % ns);
                    attn_decode_range(d, item, item + 1);
                }
            }, 1);
            pool_.parallel_for_dyn((size_t)Hq, [&](size_t b, size_t e, int) {
                attn_merge_range(d, (int)b, (int)e);
                quantize_act_q8_range(mix_.data(), (int)b * D / 64, (int)e * D / 64, xq_mix_);
            }, 1);
        }
        {
            QV_PHASE(P_PROJ_OUT);
            Job j = qjob(at.o, xq_mix_, tmp_.data(), mix_.data());
            j.epi = Epi::Residual; j.resid = x_.data();
            jobs_.push_back(j);
            run_jobs();
        }
    }
};

}  // namespace

std::unique_ptr<Engine> make_cpu_inkling_engine(const ModelFile& mf, const EngineOptions& opt);  // cpu_inkling_engine.cpp

std::unique_ptr<Engine> make_cpu_engine(const ModelFile& mf, const EngineOptions& opt) {
    if (mf.cfg.is_inkling()) return make_cpu_inkling_engine(mf, opt);
    return std::make_unique<CpuEngine>(mf, opt);
}

}  // namespace qv
