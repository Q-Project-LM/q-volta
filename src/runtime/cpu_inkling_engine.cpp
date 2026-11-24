// CPU engine for Inkling-2-Mini (arch "inkling2_loop", docs/INKLING_SPEC.md, docs/INKLING_CPU.md).
//
// One code path for decode (T = 1) and batched prefill (T > 1 tokens per chunk): every op is written for a chunk of T tokens;
// only the matmul kernel family (gemv_q vs gemm_q), the attention split and the MoE schedule differ.
//   * dense matrices (q,k,v,o,gate_attn 5-bit g64; embed/unembed 4-bit g64): existing int8-activation kernels (cpu_gemv.cpp);
//     meta8 scale tables are decoded to fp16 (scale,min) pairs at load.
//   * stacked experts [E,N,K] (3-bit g128, meta8): (scale,min) pairs stored per 128-group (QMatC::gs = 1).
//   * decode MoE: ONE parallel region per layer application; work items = (expert, 64-row chunk) of w13, then of w2; the last
//     w13 chunk of an expert quantizes its activation vector and releases that expert's w2 chunks (spin on a per-expert flag).
//   * prefill MoE: (token, expert) pairs counting-sorted by expert; per expert gemm_q over its tokens (rows of the shared sorted
//     batch via the xoff parameter), items ordered by descending token count.
//   * attention: local layers use a ring of 2048 fp16 rows per (kv head, slot), global layers a linear fp16 cache per slot;
//     every layer application has its own slot (3 passes of a loop layer do not share KV or conv state).
// ARM/NEON: nothing here is x86 specific except the kernels it calls (cpu_gemv.cpp, ink_ops.cpp), see their port maps.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_gemv.h"
#include "cpu_relax.h"
#include "cpu_inkling_engine.h"
#include "ink_ops.h"
#include "ops.h"
#include "qv/engine.h"

namespace qv {
using namespace qv::cpu;

namespace {

constexpr int H = 2304, NH = 24, NKV = 8, HD = 128, DREL = 16, REXT = 1024, WIN = 1024, NE = 128, TOPK = 8, FF = 1152;
constexpr int QD = NH * HD, KD = NKV * HD;  // 3072, 1024
constexpr int RING = 2048;                   // local KV ring (>= window + largest prefill chunk)
constexpr int MAX_BLK = 4;
constexpr int NPH = 24;

struct Stk {  // stacked expert tensor view
    const uint32_t* q = nullptr;
    const uint16_t* sm = nullptr;
    int N = 0, K = 0, bits = 0, gs = 0;
    size_t qw = 0, smw = 0;  // words / u16 per expert
    QMatC view(int e) const {
        QMatC m;
        m.q = q + (size_t)e * qw;
        m.sm = sm + (size_t)e * smw;
        m.N = N; m.K = K; m.bits = bits; m.gs = gs;
        return m;
    }
    size_t expert_bytes() const { return qw * 4 + smw * 2; }
};

struct Ctrl { const uint16_t *down = nullptr, *forget = nullptr, *gain = nullptr; };

struct LayerW {
    bool glob = false;
    QMatC wq, wk, wv, wo, wg;
    const uint16_t *wr = nullptr, *sink = nullptr, *router = nullptr;
    const float *qscale = nullptr, *relscale = nullptr, *relproj = nullptr;
    const float *attn_norm = nullptr, *qn = nullptr, *kn = nullptr, *vn = nullptr, *mlp_norm = nullptr, *gate_bias = nullptr;
    const float *kstat = nullptr, *vstat = nullptr;      // global: static taps [4][1024]
    const uint16_t *kproj = nullptr, *vproj = nullptr;   // global: dynamic taps [32][2304]
    std::vector<float> kconv, vconv, aconv, mconv;       // [4][D] tap-major
    Stk w13, w2;
    Ctrl ctl[4];  // attn, k, v, mlp (loop layers)
};

// quantized activation, both layouts: single row (gemv_q) and batch (gemm_q)
struct ActQ {
    int K = 0;
    Q8ActC a1;
    Q8Batch b;
    void init(int k, int rows) {
        K = k;
        a1.K = k;
        a1.q.assign((size_t)k, 0);
        a1.ds.assign((size_t)(k / 64) * 4, 0.f);
        if (rows > 1) b.resize(rows, k);
    }
};

struct MJob {
    const QMatC* w = nullptr;  // quantized matrix, or null for fp16
    const ActQ* a = nullptr;
    const uint16_t* wf = nullptr;
    int N = 0, K = 0;
    const float* xf = nullptr;
    size_t ldx = 0;
    float* y = nullptr;
    size_t ldy = 0;
    int act_off = 0;  // batch row offset (gemm_q xoff)
};

struct AppInfo { int L, p, slot, app; };

}  // namespace

class InkCpuEngine final : public Engine, public InklingCpuInfo {
   public:
    InkCpuEngine(const ModelFile& mf, const EngineOptions& opt) : pool_(opt.threads, pool_cpus()), opt_(opt) {
        cfg_ = mf.cfg;
        if (!cfg_.is_inkling()) throw std::runtime_error("InkCpuEngine: not an inkling2_loop model");
        const auto& ik = cfg_.ink;
        if (cfg_.hidden != H || ik.n_experts != NE || ik.topk != TOPK || ik.expert_ff != FF || ik.window != WIN || ik.rel_extent != REXT)
            throw std::runtime_error("InkCpuEngine: unsupported Inkling shape");
        V_ = ik.unpadded_vocab;
        Tc_ = std::max(1, std::min(opt.prefill_chunk, 1024));
        prof_ = std::getenv("QV_INK_PROF") != nullptr;
        map_weights(mf);
        bind(mf);
        build_apps();
        alloc();
        reset();
    }
    ~InkCpuEngine() override {
        if (owned_) std::free(map_);
        else if (map_ && map_len_) munmap(map_, map_len_);
    }

    // ------------------------------------------------------------------------------------------------ Engine interface
    void step(int token, int pos) override {
        if (pos >= opt_.max_ctx) throw std::runtime_error("context overflow");
        prep_dump(1);
        forward(&token, 1, pos, 1);
    }
    void prefill(const int* tokens, int T, int pos0) override {
        if (pos0 + T > opt_.max_ctx) throw std::runtime_error("context overflow");
        prep_dump(T);
        for (int c0 = 0; c0 < T; c0 += Tc_) {
            const int n = std::min(Tc_, T - c0);
            forward(tokens + c0, n, pos0 + c0, c0 + n == T ? 1 : 0, c0);
        }
    }
    // Vision: masked rows take their input embedding from `emb` (packed rows, in order) instead of embed + embed_norm.
    void prefill_embeds(const int* tokens, const float* emb, const unsigned char* mask, int T, int pos0) override {
        if (pos0 + T > opt_.max_ctx) throw std::runtime_error("context overflow");
        prep_dump(T);
        const float* e = emb;
        for (int c0 = 0; c0 < T; c0 += Tc_) {
            const int n = std::min(Tc_, T - c0);
            forward(tokens + c0, n, pos0 + c0, c0 + n == T ? 1 : 0, c0, mask + c0, e);
            for (int i = 0; i < n; ++i) if (mask[c0 + i]) e += H;
        }
    }
    void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool) override {
        out.resize(n);
        int tok = first;
        for (int i = 0; i < n; ++i) {
            step(tok, start_pos + i);
            tok = argmax();
            out[i] = tok;
        }
    }
    int argmax() override {
        const int nt = pool_.size();
        std::vector<float> bv(nt, -INFINITY);
        std::vector<int> bi(nt, 0x7fffffff);
        pool_.parallel_for((size_t)V_, [&](size_t b, size_t e, int t) { argmax_range(logits_.data(), (int)b, (int)e, &bv[t], &bi[t]); }, 64);
        int r;
        argmax_merge(bv.data(), bi.data(), nt, &r);
        return r;
    }
    void logits(std::vector<float>& out) override { out = logits_; }
    void reset() override {
        std::fill(cstate_.begin(), cstate_.end(), 0.f);
    }
    const std::vector<float>& hidden_dump() const override { return hid_dump_; }
    const std::vector<float>& logits_dump() const override { return logits_dump_; }
    void sync() override {}
    size_t device_bytes_weights() const override { return map_len_ + extra_bytes_; }
    size_t device_bytes_state() const override { return state_bytes_; }

    // InklingCpuInfo
    double decode_bytes_per_token() const override { return decode_bytes_; }
    const std::vector<int8_t>& router_dump() const override { return router_dump_; }
    void phase_report(std::vector<const char*>& names, std::vector<double>& ms, std::vector<long>& calls) override {
        names.clear(); ms.clear(); calls.clear();
        for (int i = 0; i < NPH; ++i)
            if (ph_calls_[i]) { names.push_back(kPhaseNames[i]); ms.push_back(ph_ms_[i]); calls.push_back(ph_calls_[i]); }
        std::fill(ph_ms_, ph_ms_ + NPH, 0.0);
        std::fill(ph_calls_, ph_calls_ + NPH, 0L);
    }

   private:
    void prep_dump(int T) {
        dump_T_ = T;
        if (!opt_.dump_hidden) return;
        hid_dump_.assign((size_t)(2 + 48) * T * H, 0.f);
        router_dump_.assign((size_t)48 * T * TOPK, 0);
        logits_dump_.assign((size_t)T * V_, 0.f);
    }

    // ------------------------------------------------------------------------------------------------ profiling
    enum Phase { P_EMB, P_ATTNRES, P_GATE, P_NORM, P_PROJ, P_CONV, P_PREP, P_ATTN, P_MERGE, P_WO, P_RES, P_MLPIN, P_ROUTER,
                 P_MOE, P_MOE_SORT, P_MOE_W13, P_MOE_Q, P_MOE_W2, P_COMBINE, P_FINAL, P_HEAD, P_MSG, P_COPY, P_OTHER };
    static constexpr const char* kPhaseNames[NPH] = {"embed", "attnres", "gate", "norm+quant", "proj", "conv_kv", "qk_prep", "attention",
                                                     "attn_merge", "wo", "attn_conv+res", "mlp_in", "router", "moe_decode",
                                                     "moe_sort", "moe_w13", "moe_quant", "moe_w2", "moe_combine", "final", "lm_head",
                                                     "messages", "copy", "other"};
    struct Scope {
        InkCpuEngine* e; int ph; std::chrono::steady_clock::time_point t0;
        Scope(InkCpuEngine* e_, int p) : e(e_), ph(p) { if (e->prof_) t0 = std::chrono::steady_clock::now(); }
        ~Scope() {
            if (e->prof_) {
                e->ph_ms_[ph] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                e->ph_calls_[ph]++;
            }
        }
    };

    static std::vector<int> pool_cpus() {
        std::vector<int> v;
        if (const char* e = std::getenv("QV_CPU_LIST"))
            for (const char* p = e; *p;) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
        return v;
    }

    // ------------------------------------------------------------------------------------------------ weights
    void map_weights(const ModelFile& mf) {
        int fd = ::open(mf.bin_path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + mf.bin_path);
        map_len_ = mf.bin_size;
        // QV_WEIGHTS=mmap (default on Android, where anonymous memory is OOM-killed): file-backed read-only mapping, pages are
        // reclaimable instead of copied; QV_WEIGHTS=copy forces the pread-into-anonymous-buffer path.
        const char* wm = std::getenv("QV_WEIGHTS");
#ifdef __ANDROID__
        const bool use_mmap = !(wm && std::string(wm) == "copy");
#else
        const bool use_mmap = wm && std::string(wm) == "mmap";
#endif
        if (use_mmap) {
            map_ = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
            if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("mmap of weights failed"); }
            ::close(fd);
            base_ = (const char*)map_;
            return;
        }
        const size_t al = 2u << 20, len = (map_len_ + al - 1) / al * al;
        map_ = std::aligned_alloc(al, len);
        if (!map_) throw std::runtime_error("weight buffer allocation failed");
        owned_ = true;
#ifdef MADV_HUGEPAGE
        madvise(map_, len, MADV_HUGEPAGE);
#endif
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
        ::close(fd);
        if (bad) throw std::runtime_error("short read on weights");
        base_ = (const char*)map_;
    }

    // meta8 (s8 | z8 << 8, per-row fp16 scale) -> fp16 (scale, min) pairs, one pair per quantization group (64 or 128 columns)
    const uint16_t* expand_sm(const TensorInfo& t) {
        const int E = t.stacked() ? (int)t.shape[0] : 1;
        const size_t N = (size_t)t.shape[t.stacked() ? 1 : 0], K = (size_t)t.shape[t.stacked() ? 2 : 1], G = K / t.group;
        std::vector<uint16_t> ex((size_t)E * N * G * 2);
        const uint16_t* sm8 = (const uint16_t*)(base_ + t.off_sm);
        const uint16_t* rs = (const uint16_t*)(base_ + t.off_rs);
        const float zlo = -0.25f * (float)((1 << t.bits) - 1), zst = (float)(1.5 * (double)((1 << t.bits) - 1) / 255.0);
        pool_.parallel_for_dyn((size_t)E * N, [&](size_t b, size_t e, int) {
            for (size_t r = b; r < e; ++r) {
                const float rsf = _cvtsh_ss(rs[r]) / 255.0f;
                for (size_t g = 0; g < G; ++g) {
                    const unsigned v = sm8[r * G + g];
                    const float sc = (float)(v & 255u) * rsf;
                    const float z = (float)(v >> 8) * zst + zlo;
                    ex[(r * G + g) * 2] = _cvtss_sh(sc, 0);
                    ex[(r * G + g) * 2 + 1] = _cvtss_sh(-(sc * z), 0);
                }
            }
        }, 256);
        sm_store_.push_back(std::move(ex));
        extra_bytes_ += sm_store_.back().size() * 2;
        return sm_store_.back().data();
    }
    QMatC qmat(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::Quant || !t.meta8 || t.stacked() || (t.group != 64 && t.group != 128)) throw std::runtime_error(n + ": expected a meta8 2-D quant tensor");
        QMatC m;
        m.q = (const uint32_t*)(base_ + t.off_q);
        m.sm = expand_sm(t);
        m.N = (int)t.shape[0]; m.K = (int)t.shape[1]; m.bits = t.bits; m.gs = t.group == 128 ? 1 : 0;
        return m;
    }
    Stk stk(const ModelFile& mf, const std::string& n) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::Quant || !t.meta8 || !t.stacked() || t.group != 128) throw std::runtime_error(n + ": expected a stacked meta8 g128 tensor");
        Stk s;
        s.q = (const uint32_t*)(base_ + t.off_q);
        s.sm = expand_sm(t);
        s.N = (int)t.shape[1]; s.K = (int)t.shape[2]; s.bits = t.bits; s.gs = 1;
        s.qw = (size_t)s.N * (s.K / 32) * s.bits;
        s.smw = (size_t)s.N * (s.K / 128) * 2;
        if (s.qw * 4 * (size_t)t.shape[0] != t.nbytes_q) throw std::runtime_error(n + ": unexpected plane size");
        return s;
    }
    const float* f32(const ModelFile& mf, const std::string& n, size_t count = 0) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F32) throw std::runtime_error(n + " is not f32");
        if (count && t.nbytes != count * 4) throw std::runtime_error(n + ": unexpected size");
        return (const float*)(base_ + t.off);
    }
    const uint16_t* f16(const ModelFile& mf, const std::string& n, size_t count) {
        const TensorInfo& t = mf.at(n);
        if (t.kind != Kind::F16 || t.nbytes != count * 2) throw std::runtime_error(n + ": expected f16 of the right size");
        return (const uint16_t*)(base_ + t.off);
    }
    std::vector<float> tap_major(const float* w, int D) {  // [D][4] -> [4][D]
        std::vector<float> o((size_t)4 * D);
        for (int c = 0; c < D; ++c)
            for (int j = 0; j < 4; ++j) o[(size_t)j * D + c] = w[(size_t)c * 4 + j];
        return o;
    }

    void bind(const ModelFile& mf) {
        const std::string LM = "model.llm.";
        const auto& ik = cfg_.ink;
        {
            QMatC e = qmat(mf, LM + "embed.weight");
            embed_ = QRows{e.q, e.sm, e.K, e.bits};
        }
        head_ = qmat(mf, LM + "unembed.weight");
        if (head_.N != V_ || head_.K != H) throw std::runtime_error("unembed shape");
        embed_norm_ = f32(mf, LM + "embed_norm.weight", H);
        final_norm_ = f32(mf, LM + "norm.weight", H);
        out_query_ = f32(mf, LM + "attn_res.output_query", H);
        msg_q_ = f32(mf, LM + "middle_cycle_cross_pass_queries", 24 * H);
        msg_kv_ = f32(mf, LM + "middle_cycle_cross_pass_kv_queries", 24 * H);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            const bool g = ik.is_global(l);
            if (l > 0) attn_q_.push_back(f32(mf, LM + "attn_res.attn_queries." + std::to_string(l - 1), (g ? 3 : 1) * H));
            mlp_q_.push_back(f32(mf, LM + "attn_res.mlp_queries." + std::to_string(l), H));
        }
        lw_.resize(cfg_.n_layers);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            const std::string p = LM + "layers." + std::to_string(l) + ".";
            LayerW& w = lw_[l];
            w.glob = ik.is_global(l);
            w.wq = qmat(mf, p + "attn.wq_du.weight");
            w.wk = qmat(mf, p + "attn.wk_dv.weight");
            w.wv = qmat(mf, p + "attn.wv_dv.weight");
            w.wo = qmat(mf, p + "attn.wo_ud.weight");
            w.wg = qmat(mf, p + "attn.gate_attn.weight");
            w.wr = f16(mf, p + "attn.wr_du.weight", (size_t)NH * DREL * H);
            w.sink = f16(mf, p + "attn.attn_sink_proj.weight", (size_t)NH * H);
            w.router = f16(mf, p + "mlp.gate.weight", (size_t)NE * H);
            w.qscale = f32(mf, p + "attn.q_l2_clip.scale", (size_t)NH * HD);
            w.relscale = f32(mf, p + "attn.rel_l2_clip.scale", (size_t)NH * DREL);
            w.relproj = f32(mf, p + "attn.rel_logits_proj.proj", (size_t)DREL * REXT);
            w.mlp_norm = f32(mf, p + "mlp_norm.weight", H);
            w.gate_bias = f32(mf, p + "mlp.gate.bias", NE);
            w.aconv = tap_major(f32(mf, p + "attn_sconv.weight", (size_t)H * 4), H);
            w.mconv = tap_major(f32(mf, p + "mlp_sconv.weight", (size_t)H * 4), H);
            if (w.glob) {
                w.qn = f32(mf, p + "q_input_norm.weight", H);
                w.kn = f32(mf, p + "k_input_norm.weight", H);
                w.vn = f32(mf, p + "v_input_norm.weight", H);
                w.kstat = f32(mf, p + "attn.k_sconv.static_w", (size_t)4 * KD);
                w.vstat = f32(mf, p + "attn.v_sconv.static_w", (size_t)4 * KD);
                w.kproj = f16(mf, p + "attn.k_sconv.weight_proj", (size_t)32 * H);
                w.vproj = f16(mf, p + "attn.v_sconv.weight_proj", (size_t)32 * H);
            } else {
                w.attn_norm = f32(mf, p + "attn_norm.weight", H);
                w.kconv = tap_major(f32(mf, p + "attn.k_sconv.weight", (size_t)KD * 4), KD);
                w.vconv = tap_major(f32(mf, p + "attn.v_sconv.weight", (size_t)KD * 4), KD);
            }
            w.w13 = stk(mf, p + "mlp.experts.w13_weight");
            w.w2 = stk(mf, p + "mlp.experts.w2_weight");
            if (w.w13.N != 2 * FF || w.w13.K != H || w.w2.N != H || w.w2.K != FF) throw std::runtime_error("expert shapes");
            if (ik.in_loop(l)) {
                static const char* nm[4] = {"attn", "k", "v", "mlp"};
                for (int s = 0; s < 4; ++s) {
                    const std::string c = p + "controllers." + nm[s] + ".";
                    w.ctl[s].down = f16(mf, c + "down_weight", (size_t)64 * 2 * H);
                    w.ctl[s].forget = f16(mf, c + "forget_weight", (size_t)H * 64);
                    w.ctl[s].gain = f16(mf, c + "gain_weight", (size_t)H * 64);
                }
            }
        }
    }

    void build_apps() {
        const auto& ik = cfg_.ink;
        int nl = 0, ng = 0, a = 0;
        auto add = [&](int L, int p) {
            AppInfo ai{L, p, ik.is_global(L) ? ng++ : nl++, a++};
            apps_.push_back(ai);
        };
        for (int L = 0; L < ik.loop_start; ++L) add(L, 0);
        for (int p = 0; p < ik.loop_passes; ++p)
            for (int L = ik.loop_start; L < ik.loop_end; ++L) add(L, p);
        for (int L = ik.loop_end; L < cfg_.n_layers; ++L) add(L, 0);
        n_local_ = nl; n_global_ = ng;
        if ((int)apps_.size() != 48) throw std::runtime_error("expected 48 layer applications");
    }

    void alloc() {
        auto F = [](std::vector<float>& v, size_t n) { v.assign(n, 0.f); };
        const size_t T = Tc_;
        F(hid_, T * H); F(resid_, T * H); F(hm_, T * H); F(xm_, T * H); F(aout_, T * H);
        for (int s = 0; s < 3; ++s) { F(xin_[s], T * H); F(xn_[s], T * H); F(g64_[s], T * 64); }
        F(prevB_, T * H); F(prevH_, T * H); F(preH_, T * H); F(msgs_, T * 4 * H); F(blk_, T * MAX_BLK * H);
        F(qraw_, T * QD); F(gat_, T * QD); F(kraw_, T * KD); F(vraw_, T * KD); F(rraw_, T * NH * DREL); F(sraw_, T * NH);
        F(dwk_, T * 32); F(dwv_, T * 32); F(kc_, T * KD); F(vc_, T * KD); F(qp_, T * QD); F(relp_, T * NH * DREL);
        F(part_, std::max(T * 8 * 390, (size_t)8 * 16 * 390));
        F(ob_, T * QD); F(gcat_, T * 2 * H); F(rl_, T * NE); F(fin_, T * H);
        sel_.assign(T * TOPK, 0); rw_.assign(T * TOPK, 0.f); pairpos_.assign(T * TOPK, 0);
        const size_t P = T * TOPK;
        F(ypair_, P * H);
        F(gu8_, (size_t)8 * 2 * FF); F(act8_, (size_t)8 * FF);
        if (Tc_ > 1) { F(actf_, P * FF); }
        F(logits_, V_);
        for (int s = 0; s < 3; ++s) aq_[s].init(H, Tc_);
        aqo_.init(QD, Tc_);
        aqm_.init(H, Tc_);
        aqf_.init(H, Tc_);
        for (int e = 0; e < 8; ++e) aq8_[e].init(FF, 1);
        if (Tc_ > 1) { aqx_.init(H, (int)P); aqa_.init(FF, (int)P); }
        order_.assign(P, 0);
        cstride_ = (size_t)3 * (KD + KD + H + H);
        cstate_.assign(cstride_ * 48, 0.f);
        // KV caches (fp16): local ring per slot, global linear per slot (pages untouched until written)
        const size_t lsz = (size_t)NKV * RING * HD, gsz = (size_t)NKV * opt_.max_ctx * HD;
        kl_.resize(n_local_); vl_.resize(n_local_); kg_.resize(n_global_); vg_.resize(n_global_);
        for (auto& v : kl_) v.reset((uint16_t*)std::calloc(lsz, 2));
        for (auto& v : vl_) v.reset((uint16_t*)std::calloc(lsz, 2));
        { const char* kv = std::getenv("QV_KV"); kv8_ = kv && std::string(kv) == "int8"; }
        const size_t esz = kv8_ ? 1 : 2;  // int8 KV on global layers (the local ring is small): 1 B/elem + one fp32 scale per row
        for (auto& v : kg_) v.reset((uint16_t*)std::calloc(gsz, esz));
        for (auto& v : vg_) v.reset((uint16_t*)std::calloc(gsz, esz));
        if (kv8_) { ksc_.assign(n_global_, std::vector<float>((size_t)NKV * opt_.max_ctx, 0.f)); vsc_ = ksc_; }
        state_bytes_ = (size_t)n_local_ * lsz * 4 + (size_t)n_global_ * gsz * 2 * esz + (kv8_ ? (size_t)n_global_ * NKV * opt_.max_ctx * 8 : 0) + cstate_.size() * 4;
        // bytes touched per decode token
        double b = 0;
        for (const auto& ap : apps_) {
            const LayerW& w = lw_[ap.L];
            for (const QMatC* m : {&w.wq, &w.wk, &w.wv, &w.wo, &w.wg}) b += (double)m->bytes();
            b += (double)(NH * DREL + NH + NE) * H * 2;
            if (w.glob) b += 2.0 * 32 * H * 2;
            b += (double)TOPK * (w.w13.expert_bytes() + w.w2.expert_bytes());
            if (ap.p > 0) b += 4.0 * (64.0 * 2 * H + 2.0 * H * 64) * 2;
        }
        b += (double)head_.bytes();
        decode_bytes_ = b;
    }

    // ------------------------------------------------------------------------------------------------ parallel helpers
    template <class F> void par_t(int T, F&& f) {  // over tokens (inline for T == 1)
        if (T == 1) f((size_t)0, (size_t)1, 0);
        else pool_.parallel_for((size_t)T, f, 1);
    }

    void quant(ActQ& a, const float* x, size_t ldx, int T) {
        if (T == 1) quantize_act_q8_raw(x, 0, a.K / 64, a.a1.q.data(), a.a1.ds.data());
        else pool_.parallel_for((size_t)T, [&](size_t b, size_t e, int) { for (size_t t = b; t < e; ++t) a.b.quantize_row((int)t, x + t * ldx, a.K); }, 1);
    }

    std::vector<MJob> jobs_;
    void qjob(const QMatC& w, const ActQ& a, float* y, size_t ldy, int off = 0) {
        MJob j; j.w = &w; j.a = &a; j.N = w.N; j.K = w.K; j.y = y; j.ldy = ldy; j.act_off = off;
        jobs_.push_back(j);
    }
    void fjob(const uint16_t* w, int N, int K, const float* x, size_t ldx, float* y, size_t ldy) {
        MJob j; j.wf = w; j.N = N; j.K = K; j.xf = x; j.ldx = ldx; j.y = y; j.ldy = ldy;
        jobs_.push_back(j);
    }
    static size_t ntb(const MJob& j, int T) { return (j.w || T <= 64) ? 1 : (size_t)((T + 63) / 64); }
    void run_jobs(int T, int claim = 4) {
        constexpr int CH = 16;
        size_t total = 0;
        std::vector<size_t> pre(jobs_.size() + 1, 0);
        for (size_t i = 0; i < jobs_.size(); ++i) { pre[i] = total; total += (size_t)(jobs_[i].N + CH - 1) / CH * ntb(jobs_[i], T); }
        pre[jobs_.size()] = total;
        pool_.parallel_for_dyn(total, [&](size_t b, size_t e, int) {
            for (size_t i = b; i < e; ++i) {
                size_t j = 0;
                while (pre[j + 1] <= i) ++j;
                const MJob& jb = jobs_[j];
                const size_t nrc = (size_t)(jb.N + CH - 1) / CH, li = i - pre[j];
                const int r0 = (int)(li % nrc) * CH, r1 = std::min(jb.N, r0 + CH), tb = (int)(li / nrc);
                if (jb.w) {
                    if (T == 1) gemv_q(*jb.w, jb.a->a1, jb.y, r0, r1);
                    else gemm_q(*jb.w, jb.a->b, T, jb.y, jb.ldy, r0, r1, jb.act_off);
                } else {
                    const int t0 = tb * 64, tn = std::min(64, T - t0);
                    gemm_f16_rows(jb.wf, jb.K, jb.xf + (size_t)t0 * jb.ldx, jb.ldx, tn, jb.y + (size_t)t0 * jb.ldy, jb.ldy, r0, r1);
                }
            }
        }, (size_t)claim);
        jobs_.clear();
    }

    // ------------------------------------------------------------------------------------------------ conv cores
    template <class LoadX>
    static void conv_static_core(const float* w, float* st, int D, LoadX&& ldx, float* out, size_t ldo, const float* add, size_t ldadd,
                                 int T, int c0, int c1) {
        const __m256 inv = _mm256_set1_ps(1.0f / std::sqrt(5.0f));
        for (int c = c0; c < c1; c += 8) {
            __m256 s0 = _mm256_loadu_ps(st + c), s1 = _mm256_loadu_ps(st + D + c), s2 = _mm256_loadu_ps(st + 2 * D + c);
            const __m256 w0 = _mm256_loadu_ps(w + c), w1 = _mm256_loadu_ps(w + D + c), w2 = _mm256_loadu_ps(w + 2 * D + c),
                         w3 = _mm256_loadu_ps(w + 3 * D + c);
            for (int t = 0; t < T; ++t) {
                const __m256 x = ldx(t, c);
                __m256 y = _mm256_mul_ps(w0, s0);
                y = _mm256_fmadd_ps(w1, s1, y);
                y = _mm256_fmadd_ps(w2, s2, y);
                y = _mm256_fmadd_ps(w3, x, y);
                y = _mm256_mul_ps(_mm256_add_ps(y, x), inv);
                if (add) y = _mm256_add_ps(y, _mm256_loadu_ps(add + (size_t)t * ldadd + c));
                _mm256_storeu_ps(out + (size_t)t * ldo + c, y);
                s0 = s1; s1 = s2; s2 = x;
            }
            _mm256_storeu_ps(st + c, s0); _mm256_storeu_ps(st + D + c, s1); _mm256_storeu_ps(st + 2 * D + c, s2);
        }
    }
    // dynamic taps (global k/v): tap_l[t][c] = stat[l][c] + dw[t][c / 128 * 4 + l]; out = (sum_l tap_l x[t-l] + x) / 3
    static void conv_dynamic_core(const float* stat, const float* dw, float* st, const float* x, float* out, int T, int c0, int c1) {
        const __m256 inv = _mm256_set1_ps(1.0f / 3.0f);
        constexpr int D = KD;
        for (int c = c0; c < c1; c += 8) {
            const int g = c / HD;
            __m256 s0 = _mm256_loadu_ps(st + c), s1 = _mm256_loadu_ps(st + D + c), s2 = _mm256_loadu_ps(st + 2 * D + c);
            const __m256 a0 = _mm256_loadu_ps(stat + c), a1 = _mm256_loadu_ps(stat + D + c), a2 = _mm256_loadu_ps(stat + 2 * D + c),
                         a3 = _mm256_loadu_ps(stat + 3 * D + c);
            for (int t = 0; t < T; ++t) {
                const float* d = dw + (size_t)t * 32 + g * 4;
                const __m256 xv = _mm256_loadu_ps(x + (size_t)t * D + c);
                __m256 y = _mm256_mul_ps(_mm256_add_ps(a0, _mm256_set1_ps(d[0])), xv);
                y = _mm256_fmadd_ps(_mm256_add_ps(a1, _mm256_set1_ps(d[1])), s2, y);
                y = _mm256_fmadd_ps(_mm256_add_ps(a2, _mm256_set1_ps(d[2])), s1, y);
                y = _mm256_fmadd_ps(_mm256_add_ps(a3, _mm256_set1_ps(d[3])), s0, y);
                _mm256_storeu_ps(out + (size_t)t * D + c, _mm256_mul_ps(_mm256_add_ps(y, xv), inv));
                s0 = s1; s1 = s2; s2 = xv;
            }
            _mm256_storeu_ps(st + c, s0); _mm256_storeu_ps(st + D + c, s1); _mm256_storeu_ps(st + 2 * D + c, s2);
        }
    }

    // ------------------------------------------------------------------------------------------------ message gate
    // out[t] = (1 - tanh(F g)) * h[t] + tanh(G g) * msg[t],  g = down(concat(rms_u(h), rms_u(msg)))  (3 streams at once for attention)
    // hs[s], ms[s] are [T][H] (msg rows have stride msg_ld), outputs os[s] [T][H].
    void message_gates(const LayerW& w, int ns, const int* which, const float* const* hs, const float* const* ms, size_t msg_ld,
                       float* const* os, int T) {
        // gcat3_ layout: [s][T][2H]
        gcat3_.resize((size_t)3 * T * 2 * H);
        {
            Scope sc(this, P_GATE);
            par_t(T, [&](size_t b, size_t e, int) {
                for (size_t t = b; t < e; ++t)
                    for (int s = 0; s < ns; ++s) {
                        float* g = gcat3_.data() + ((size_t)s * T + t) * 2 * H;
                        rms_u(hs[s] + t * H, g, H, 1e-6f);
                        rms_u(ms[s] + t * msg_ld, g + H, H, 1e-6f);
                    }
            });
            for (int s = 0; s < ns; ++s) fjob(w.ctl[which[s]].down, 64, 2 * H, gcat3_.data() + (size_t)s * T * 2 * H, 2 * H, g64_[s].data(), 64);
            run_jobs(T, 1);
        }
        {
            Scope sc(this, P_GATE);
            const int CHR = 64, nch = H / CHR;
            pool_.parallel_for_dyn((size_t)ns * nch, [&](size_t b, size_t e, int) {
                for (size_t it = b; it < e; ++it) {
                    const int s = (int)(it / nch), c0 = (int)(it % nch) * CHR;
                    const Ctrl& ct = w.ctl[which[s]];
                    static thread_local std::vector<float> fgt, mg;
                    fgt.resize((size_t)T * 64); mg.resize((size_t)T * 64);
                    gemm_f16_rows(ct.forget, 64, g64_[s].data(), 64, T, fgt.data() - c0, 64, c0, c0 + CHR);
                    gemm_f16_rows(ct.gain, 64, g64_[s].data(), 64, T, mg.data() - c0, 64, c0, c0 + CHR);
                    for (int t = 0; t < T; ++t) {
                        float* f = fgt.data() + (size_t)t * 64;
                        float* m = mg.data() + (size_t)t * 64;
                        softcap_inplace(f, CHR, 1.0f);
                        softcap_inplace(m, CHR, 1.0f);
                        for (int i = 0; i < CHR; ++i) {
                            const size_t o = (size_t)t * H + c0 + i;
                            os[s][o] = (1.0f - f[i]) * hs[s][o] + m[i] * ms[s][(size_t)t * msg_ld + c0 + i];
                        }
                    }
                }
            }, 1);
        }
    }

    // ------------------------------------------------------------------------------------------------ one layer application
    void run_app(const AppInfo& ap, int T, int pos0, const float* msgs) {
        const LayerW& w = lw_[ap.L];
        const bool glob = w.glob, boundary = ap.L % 6 == 0;
        float* cs = cstate_.data() + cstride_ * ap.app;
        float *cs_k = cs, *cs_v = cs + 3 * KD, *cs_a = cs + 6 * KD, *cs_m = cs + 6 * KD + 3 * H;

        // ---- attention input streams
        const float* xs[3] = {nullptr, nullptr, nullptr};
        if (ap.L > 0) {
            Scope sc(this, P_ATTNRES);
            const int Q = glob ? 3 : 1;
            const float* qr = attn_q_[ap.L - 1];
            par_t(T, [&](size_t b, size_t e, int) {
                for (size_t t = b; t < e; ++t) {
                    const float* bl[MAX_BLK];
                    for (int i = 0; i < nblk_; ++i) bl[i] = blk_.data() + (t * MAX_BLK + i) * H;
                    float o[3 * H];
                    block_attn_res(bl, nblk_, hid_.data() + t * H, qr, Q, H, o);
                    for (int q = 0; q < Q; ++q) std::memcpy(xin_[q].data() + t * H, o + (size_t)q * H, H * 4);
                }
            });
            xs[0] = xin_[0].data();
            if (glob) { xs[1] = xin_[1].data(); xs[2] = xin_[2].data(); }
        } else {
            xs[0] = hid_.data();
        }
        int nstream = glob ? 3 : 1;
        const float* in3[3] = {xs[0], glob ? xs[1] : xs[0], glob ? xs[2] : xs[0]};
        if (msgs) {
            // gated streams (attention q / k / v paths); local layers gate the SAME input with three different messages
            static const int which[3] = {0, 1, 2};
            const float* ms[3] = {msgs + 0 * H, msgs + 1 * H, msgs + 2 * H};
            float* os[3] = {xin_[0].data(), xin_[1].data(), xin_[2].data()};
            // outputs must not alias inputs: use xn_ as the destination, then norm in place from there
            float* od[3] = {xn_[0].data(), xn_[1].data(), xn_[2].data()};
            (void)os;
            message_gates(w, 3, which, in3, ms, 4 * H, od, T);
            in3[0] = od[0]; in3[1] = od[1]; in3[2] = od[2];
            nstream = 3;
        }
        // ---- input norms + activation quantization (+ fp32 copy for the fp16 matrices)
        {
            Scope sc(this, P_NORM);
            const float* nw[3] = {glob ? w.qn : w.attn_norm, glob ? w.kn : w.attn_norm, glob ? w.vn : w.attn_norm};
            par_t(T, [&](size_t b, size_t e, int) {
                for (size_t t = b; t < e; ++t)
                    for (int s = 0; s < nstream; ++s) {
                        float* o = (msgs || glob) ? xn_[s].data() + t * H : xn_[0].data() + t * H;
                        float tmp[H];
                        rms_w(in3[s] + t * H, nw[s], tmp, H, 1e-6f);
                        std::memcpy(o, tmp, H * 4);
                    }
            });
            for (int s = 0; s < nstream; ++s) quant(aq_[s], xn_[s].data(), H, T);
        }
        const ActQ* aS[3] = {&aq_[0], nstream == 3 ? &aq_[1] : &aq_[0], nstream == 3 ? &aq_[2] : &aq_[0]};
        const float* xS[3] = {xn_[0].data(), nstream == 3 ? xn_[1].data() : xn_[0].data(), nstream == 3 ? xn_[2].data() : xn_[0].data()};
        // ---- projections
        {
            Scope sc(this, P_PROJ);
            qjob(w.wq, *aS[0], qraw_.data(), QD);
            qjob(w.wg, *aS[0], gat_.data(), QD);
            qjob(w.wk, *aS[1], kraw_.data(), KD);
            qjob(w.wv, *aS[2], vraw_.data(), KD);
            fjob(w.wr, NH * DREL, H, xS[0], H, rraw_.data(), NH * DREL);
            fjob(w.sink, NH, H, xS[0], H, sraw_.data(), NH);
            if (glob) {
                fjob(w.kproj, 32, H, xS[1], H, dwk_.data(), 32);
                fjob(w.vproj, 32, H, xS[2], H, dwv_.data(), 32);
            }
            run_jobs(T);
        }
        // ---- short convs on k / v (state per channel; tokens sequential inside a channel range)
        {
            Scope sc(this, P_CONV);
            pool_.parallel_for_dyn(2 * (KD / 64), [&](size_t b, size_t e, int) {
                for (size_t it = b; it < e; ++it) {
                    const bool isv = it >= (size_t)KD / 64;
                    const int c0 = (int)(it % (KD / 64)) * 64;
                    const float* x = isv ? vraw_.data() : kraw_.data();
                    float* out = isv ? vc_.data() : kc_.data();
                    float* st = isv ? cs_v : cs_k;
                    if (glob)
                        conv_dynamic_core(isv ? w.vstat : w.kstat, isv ? dwv_.data() : dwk_.data(), st, x, out, T, c0, c0 + 64);
                    else
                        conv_static_core(isv ? w.vconv.data() : w.kconv.data(), st, KD, [&](int t, int c) { return _mm256_loadu_ps(x + (size_t)t * KD + c); },
                                         out, KD, nullptr, 0, T, c0, c0 + 64);
                }
            }, 1);
        }
        // ---- q / rel prep, k clip + cache write, v norm + cache write: items (token, kv head)
        const int slot = ap.slot;
        uint16_t* kcache = glob ? kg_[slot].get() : kl_[slot].get();
        uint16_t* vcache = glob ? vg_[slot].get() : vl_[slot].get();
        const size_t head_stride = glob ? (size_t)opt_.max_ctx * HD : (size_t)RING * HD;
        {
            Scope sc(this, P_PREP);
            const float qkclip = cfg_.ink.qk_l2_clip, relclip = cfg_.ink.rel_l2_clip;
            pool_.parallel_for_dyn((size_t)T * NKV, [&](size_t b, size_t e, int) {
                for (size_t it = b; it < e; ++it) {
                    const int t = (int)(it / NKV), g = (int)(it % NKV);
                    const int p = pos0 + t;
                    float tau = 1.0f;
                    if (glob) tau = 1.0f + cfg_.ink.log_scaling_alpha * std::log(std::max((float)(p + 1) / (float)cfg_.ink.log_scaling_n_floor, 1.0f));
                    for (int i = 0; i < 3; ++i) {
                        const int h = 3 * g + i;
                        float q[HD], r[DREL];
                        const float* qs = w.qscale + h * HD;
                        const float* qr = qraw_.data() + (size_t)t * QD + h * HD;
                        for (int d = 0; d < HD; ++d) q[d] = qr[d] * qs[d];
                        float nq = std::sqrt(dot_f32(q, q, HD));
                        float f = std::min(1.0f, qkclip / (nq + 1e-6f)) * tau;
                        float* qo = qp_.data() + (size_t)t * QD + h * HD;
                        for (int d = 0; d < HD; ++d) qo[d] = q[d] * f;
                        const float* rr = rraw_.data() + (size_t)t * NH * DREL + h * DREL;
                        for (int d = 0; d < DREL; ++d) r[d] = rr[d] * w.relscale[h * DREL + d];
                        float nr = std::sqrt(dot_f32(r, r, DREL));
                        float fr = std::min(1.0f, relclip / (nr + 1e-6f)) * tau;
                        float* ro = relp_.data() + (size_t)t * NH * DREL + h * DREL;
                        for (int d = 0; d < DREL; ++d) ro[d] = r[d] * fr;
                    }
                    const size_t row = glob ? (size_t)p : (size_t)(p % RING);
                    {
                        float k[HD];
                        std::memcpy(k, kc_.data() + (size_t)t * KD + g * HD, HD * 4);
                        const float nk = std::sqrt(dot_f32(k, k, HD));
                        const float f = std::min(1.0f, qkclip / (nk + 1e-6f));
                        for (int d = 0; d < HD; ++d) k[d] *= f;
                        float v[HD];
                        rms_u(vc_.data() + (size_t)t * KD + g * HD, v, HD, 1e-6f);
                        if (glob && kv8_) {
                            const size_t o = g * head_stride + row * HD, so = (size_t)g * opt_.max_ctx + row;
                            store_q8(k, (int8_t*)kcache + o, &ksc_[slot][so]);
                            store_q8(v, (int8_t*)vcache + o, &vsc_[slot][so]);
                        } else {
                            store_f16(k, kcache + g * head_stride + row * HD, HD);
                            store_f16(v, vcache + g * head_stride + row * HD, HD);
                        }
                    }
                }
            }, 1);
        }
        // ---- attention (online softmax partials per (token, kv head, key split)), merge with sink, output gate
        {
            Scope sc(this, P_ATTN);
            attn_ns_ = 1;
            if (T == 1) {
                const int p = pos0;
                const int n = glob ? p + 1 : std::min(p + 1, WIN);
                attn_ns_ = std::max(1, std::min(16, n / 128));
            }
            const int ns = attn_ns_;
            pool_.parallel_for_dyn((size_t)T * NKV * ns, [&](size_t b, size_t e, int) {
                for (size_t it = b; it < e; ++it) {
                    const int split = (int)(it % ns), g = (int)((it / ns) % NKV), t = (int)(it / ((size_t)ns * NKV));
                    const int p = pos0 + t;
                    const int jlo = glob ? 0 : std::max(0, p - WIN + 1);
                    const int n = p + 1 - jlo;
                    InkAttnArgs a;
                    a.q = qp_.data() + (size_t)t * QD + g * 3 * HD;
                    a.rel = relp_.data() + (size_t)t * NH * DREL + g * 3 * DREL;
                    if (glob && kv8_) {
                        a.kq = (const int8_t*)kcache + g * head_stride; a.vq = (const int8_t*)vcache + g * head_stride;
                        a.ks = ksc_[slot].data() + (size_t)g * opt_.max_ctx; a.vs = vsc_[slot].data() + (size_t)g * opt_.max_ctx;
                    } else {
                        a.kc = kcache + g * head_stride;
                        a.vc = vcache + g * head_stride;
                    }
                    a.proj = w.relproj;
                    a.p = p; a.j0 = jlo + (int)((long)n * split / ns); a.j1 = jlo + (int)((long)n * (split + 1) / ns);
                    a.ring = glob ? 0 : RING; a.rel_extent = REXT; a.scale = 1.0f / std::sqrt((float)HD);
                    float* out = part_.data() + (((size_t)t * NKV + g) * ns + split) * 390;
                    ink_attn_partial(a, out, out + 3 * HD);
                }
            }, 1);
        }
        {
            Scope sc(this, P_MERGE);
            const int ns = attn_ns_;
            pool_.parallel_for_dyn((size_t)T * NH, [&](size_t b, size_t e, int) {
                for (size_t it = b; it < e; ++it) {
                    const int t = (int)(it / NH), h = (int)(it % NH), g = h / 3, i = h % 3;
                    const float* part = part_.data() + ((size_t)t * NKV + g) * ns * 390;
                    ink_attn_merge(part, ns, 390, i, sraw_[(size_t)t * NH + h], gat_.data() + (size_t)t * QD + h * HD, ob_.data() + (size_t)t * QD + h * HD);
                }
            }, 4);
            quant(aqo_, ob_.data(), QD, T);
        }
        {
            Scope sc(this, P_WO);
            qjob(w.wo, aqo_, aout_.data(), H);
            run_jobs(T);
        }
        // ---- post-attention conv, residual (reset at block boundaries), attention-residual blocks
        {
            Scope sc(this, P_RES);
            const float* add = boundary ? nullptr : hid_.data();
            pool_.parallel_for_dyn(H / 64, [&](size_t b, size_t e, int) {
                for (size_t it = b; it < e; ++it) {
                    const int c0 = (int)it * 64;
                    conv_static_core(w.aconv.data(), cs_a, H, [&](int t, int c) { return _mm256_loadu_ps(aout_.data() + (size_t)t * H + c); },
                                     resid_.data(), H, add, H, T, c0, c0 + 64);
                }
            }, 1);
            if (boundary) {
                for (int t = 0; t < T; ++t) std::memcpy(blk_.data() + ((size_t)t * MAX_BLK + nblk_) * H, hid_.data() + (size_t)t * H, H * 4);
                nblk_++;
            }
        }
        // ---- MLP input: attention-residual mix, loop message gate, norm
        {
            {
                Scope sc(this, P_MLPIN);
                par_t(T, [&](size_t b, size_t e, int) {
                    for (size_t t = b; t < e; ++t) {
                        const float* bl[MAX_BLK];
                        for (int i = 0; i < nblk_; ++i) bl[i] = blk_.data() + (t * MAX_BLK + i) * H;
                        block_attn_res(bl, nblk_, resid_.data() + t * H, mlp_q_[ap.L], 1, H, hm_.data() + t * H);
                    }
                });
            }
            if (msgs) {
                static const int which[1] = {3};
                const float* hs[1] = {hm_.data()};
                const float* ms[1] = {msgs + 3 * H};
                float* os[1] = {xn_[0].data()};
                message_gates(w, 1, which, hs, ms, 4 * H, os, T);
                std::memcpy(hm_.data(), xn_[0].data(), (size_t)T * H * 4);
            }
            Scope sc(this, P_MLPIN);
            par_t(T, [&](size_t b, size_t e, int) {
                for (size_t t = b; t < e; ++t) rms_w(hm_.data() + t * H, w.mlp_norm, xm_.data() + t * H, H, 1e-6f);
            });
            quant(aqm_, xm_.data(), H, T);
        }
        // ---- router
        {
            Scope sc(this, P_ROUTER);
            fjob(w.router, NE, H, xm_.data(), H, rl_.data(), NE);
            run_jobs(T, 2);
            par_t(T, [&](size_t b, size_t e, int) {
                for (size_t t = b; t < e; ++t) router_topk(rl_.data() + t * NE, w.gate_bias, NE, TOPK, cfg_.ink.route_scale, &sel_[t * TOPK], &rw_[t * TOPK]);
            });
            if (opt_.dump_hidden)
                for (int t = 0; t < T; ++t)
                    for (int k = 0; k < TOPK; ++k) router_dump_[((size_t)ap.app * dump_T_ + dump_off_ + t) * TOPK + k] = (int8_t)sel_[(size_t)t * TOPK + k];
        }
        // ---- experts
        if (T == 1) moe_decode(w);
        else moe_prefill(w, T);
        // ---- combine (weighted sum over the top-8), mlp conv, residual
        {
            Scope sc(this, P_COMBINE);
            pool_.parallel_for_dyn(H / 64, [&](size_t b, size_t e, int) {
                for (size_t it = b; it < e; ++it) {
                    const int c0 = (int)it * 64;
                    conv_static_core(w.mconv.data(), cs_m, H, [&](int t, int c) {
                        __m256 a = _mm256_setzero_ps();
                        for (int k = 0; k < TOPK; ++k)
                            a = _mm256_fmadd_ps(_mm256_set1_ps(rw_[(size_t)t * TOPK + k]), _mm256_loadu_ps(ypair_.data() + (size_t)pairpos_[(size_t)t * TOPK + k] * H + c), a);
                        return a;
                    }, hid_.data(), H, resid_.data(), H, T, c0, c0 + 64);
                }
            }, 1);
        }
        if (opt_.dump_hidden)
            std::memcpy(hid_dump_.data() + ((size_t)(1 + ap.app) * dump_T_ + dump_off_) * H, hid_.data(), (size_t)T * H * 4);
    }

    // ------------------------------------------------------------------------------------------------ MoE, decode (T = 1)
    void moe_decode(const LayerW& w) {
        Scope sc(this, P_MOE);
        constexpr int C13 = 2 * FF / 64, C2 = H / 64;
        for (int k = 0; k < 8; ++k) { cnt_[k].v.store(0, std::memory_order_relaxed); ready_[k].v.store(0, std::memory_order_relaxed); pairpos_[k] = k; }
        const float gcap = cfg_.ink.gate_softcap, ucap = cfg_.ink.up_softcap;
        pool_.parallel_for_dyn((size_t)8 * (C13 + C2), [&](size_t b, size_t e, int) {
            for (size_t i = b; i < e; ++i) {
                if (i < (size_t)8 * C13) {
                    const int k = (int)(i / C13), c = (int)(i % C13);
                    const QMatC W = w.w13.view(sel_[k]);
                    float* gu = gu8_.data() + (size_t)k * 2 * FF;
                    gemv_q(W, aqm_.a1, gu, c * 64, c * 64 + 64);
                    expert_act_interleaved(gu + c * 64, act8_.data() + (size_t)k * FF + c * 32, 32, gcap, ucap);
                    if (cnt_[k].v.fetch_add(1, std::memory_order_acq_rel) + 1 == C13) {
                        quantize_act_q8_raw(act8_.data() + (size_t)k * FF, 0, FF / 64, aq8_[k].a1.q.data(), aq8_[k].a1.ds.data());
                        ready_[k].v.store(1, std::memory_order_release);
                    }
                } else {
                    const size_t j = i - (size_t)8 * C13;
                    const int k = (int)(j / C2), c = (int)(j % C2);
                    while (!ready_[k].v.load(std::memory_order_acquire)) QV_CPU_RELAX();
                    gemv_q(w.w2.view(sel_[k]), aq8_[k].a1, ypair_.data() + (size_t)k * H, c * 64, c * 64 + 64);
                }
            }
        }, 1);
    }

    // ------------------------------------------------------------------------------------------------ MoE, prefill (T > 1)
    void moe_prefill(const LayerW& w, int T) {
        const int P = T * TOPK;
        {
            Scope sc(this, P_MOE_SORT);
            std::fill(ecount_, ecount_ + NE, 0);
            for (int i = 0; i < P; ++i) ecount_[sel_[i]]++;
            int o = 0;
            for (int e = 0; e < NE; ++e) { eoff_[e] = o; o += ecount_[e]; }
            int fill[NE] = {0};
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < TOPK; ++k) {
                    const int e = sel_[(size_t)t * TOPK + k], pos = eoff_[e] + fill[e]++;
                    order_[pos] = t;
                    pairpos_[(size_t)t * TOPK + k] = pos;
                }
            // gather the quantized activations of each (token, expert) pair into expert-sorted rows
            pool_.parallel_for((size_t)P, [&](size_t b, size_t e, int) {
                const Q8Batch& s = aqm_.b;
                Q8Batch& d = aqx_.b;
                for (size_t i = b; i < e; ++i) {
                    const size_t t = (size_t)order_[i];
                    std::memcpy(d.q.data() + i * H, s.q.data() + t * H, H);
                    std::memcpy(d.ds.data() + i * s.G * 4, s.ds.data() + t * s.G * 4, (size_t)s.G * 16);
                    std::memcpy(d.dv8.data() + i * s.G * 8, s.dv8.data() + t * s.G * 8, (size_t)s.G * 32);
                    std::memcpy(d.xs.data() + i * s.G, s.xs.data() + t * s.G, (size_t)s.G * 4);
                }
            }, 8);
            // experts by descending token count (largest items first)
            nact_ = 0;
            for (int e = 0; e < NE; ++e) if (ecount_[e]) eact_[nact_++] = e;
            std::sort(eact_, eact_ + nact_, [&](int a, int b) { return ecount_[a] != ecount_[b] ? ecount_[a] > ecount_[b] : a < b; });
        }
        const float gcap = cfg_.ink.gate_softcap, ucap = cfg_.ink.up_softcap;
        {
            Scope sc(this, P_MOE_W13);
            constexpr int C13 = 2 * FF / 64;
            pool_.parallel_for_dyn((size_t)nact_ * C13, [&](size_t b, size_t e, int) {
                static thread_local std::vector<float> tmp;
                for (size_t i = b; i < e; ++i) {
                    const int ex = eact_[i / C13], c = (int)(i % C13), n = ecount_[ex], off = eoff_[ex];
                    tmp.resize((size_t)n * 64);
                    const QMatC W = w.w13.view(ex);
                    gemm_q(W, aqx_.b, n, tmp.data() - (size_t)c * 64, 64, c * 64, c * 64 + 64, off);
                    for (int t = 0; t < n; ++t) expert_act_interleaved(tmp.data() + (size_t)t * 64, actf_.data() + (size_t)(off + t) * FF + c * 32, 32, gcap, ucap);
                }
            }, 1);
        }
        {
            Scope sc(this, P_MOE_Q);
            pool_.parallel_for((size_t)P, [&](size_t b, size_t e, int) {
                for (size_t i = b; i < e; ++i) aqa_.b.quantize_row((int)i, actf_.data() + i * FF, FF);
            }, 8);
        }
        {
            Scope sc(this, P_MOE_W2);
            constexpr int C2 = H / 64;
            pool_.parallel_for_dyn((size_t)nact_ * C2, [&](size_t b, size_t e, int) {
                for (size_t i = b; i < e; ++i) {
                    const int ex = eact_[i / C2], c = (int)(i % C2), n = ecount_[ex], off = eoff_[ex];
                    gemm_q(w.w2.view(ex), aqa_.b, n, ypair_.data() + (size_t)off * H, H, c * 64, c * 64 + 64, off);
                }
            }, 1);
        }
    }

    // ------------------------------------------------------------------------------------------------ whole forward
    void forward(const int* toks, int T, int pos0, int want_logits, int dump_off = 0, const unsigned char* mask = nullptr, const float* emb = nullptr) {
        dump_off_ = dump_off;
        if (pos0 == 0) reset();
        const auto& ik = cfg_.ink;
        {
            Scope sc(this, P_EMB);
            std::vector<int> eidx;
            if (mask) {
                eidx.resize(T);
                int c = 0;
                for (int t = 0; t < T; ++t) { eidx[t] = c; c += mask[t] ? 1 : 0; }
            }
            par_t(T, [&](size_t b, size_t e, int) {
                for (size_t t = b; t < e; ++t) {
                    if (mask && mask[t]) { std::memcpy(hid_.data() + t * H, emb + (size_t)eidx[t] * H, H * 4); continue; }
                    float x[H];
                    embed_row(embed_, toks[t], x);
                    rms_w(x, embed_norm_, hid_.data() + t * H, H, 1e-6f);
                }
            });
        }
        if (opt_.dump_hidden) std::memcpy(hid_dump_.data() + ((size_t)dump_off_) * H, hid_.data(), (size_t)T * H * 4);
        nblk_ = 0;
        int ai = 0;
        for (; ai < ik.loop_start; ++ai) run_app(apps_[ai], T, pos0, nullptr);
        {
            Scope sc(this, P_COPY);
            std::memcpy(preH_.data(), hid_.data(), (size_t)T * H * 4);
        }
        const int nblk_pre = nblk_;
        const int nloop = ik.loop_end - ik.loop_start;
        for (int p = 0; p < ik.loop_passes; ++p) {
            if (p > 0) {
                Scope sc(this, P_COPY);
                // previous pass: block state appended by layer loop_start+6 (index nblk_pre + 1) and the pass output
                for (int t = 0; t < T; ++t) std::memcpy(prevB_.data() + (size_t)t * H, blk_.data() + ((size_t)t * MAX_BLK + nblk_pre + 1) * H, H * 4);
                std::memcpy(prevH_.data(), hid_.data(), (size_t)T * H * 4);
                std::memcpy(hid_.data(), preH_.data(), (size_t)T * H * 4);
                nblk_ = nblk_pre;
            }
            for (int li = 0; li < nloop; ++li, ++ai) {
                const float* msgs = nullptr;
                if (p > 0) {
                    Scope sc(this, P_MSG);
                    float mq[4 * H];
                    std::memcpy(mq, msg_q_ + (size_t)(li * 2) * H, H * 4);          // query
                    std::memcpy(mq + H, msg_kv_ + (size_t)(li * 2) * H, H * 4);     // key
                    std::memcpy(mq + 2 * H, msg_kv_ + (size_t)(li * 2 + 1) * H, H * 4);  // value
                    std::memcpy(mq + 3 * H, msg_q_ + (size_t)(li * 2 + 1) * H, H * 4);  // mlp
                    par_t(T, [&](size_t b, size_t e, int) {
                        for (size_t t = b; t < e; ++t) {
                            const float* bl[1] = {prevB_.data() + t * H};
                            block_attn_res(bl, 1, prevH_.data() + t * H, mq, 4, H, msgs_.data() + t * 4 * H);
                        }
                    });
                    msgs = msgs_.data();
                }
                run_app(apps_[ai], T, pos0, msgs);
            }
        }
        for (; ai < (int)apps_.size(); ++ai) run_app(apps_[ai], T, pos0, nullptr);
        // ---- final attention-residual mix + norm + lm_head
        const bool all_rows = opt_.dump_hidden || (bool)opt_.row_logits;
        const int nfin = all_rows ? T : 1;
        const int t0 = T - nfin;
        {
            Scope sc(this, P_FINAL);
            par_t(nfin, [&](size_t b, size_t e, int) {
                for (size_t i = b; i < e; ++i) {
                    const size_t t = t0 + i;
                    const float* bl[MAX_BLK];
                    for (int j = 0; j < nblk_; ++j) bl[j] = blk_.data() + (t * MAX_BLK + j) * H;
                    float o[H];
                    block_attn_res(bl, nblk_, hid_.data() + t * H, out_query_, 1, H, o);
                    rms_w(o, final_norm_, fin_.data() + t * H, H, 1e-6f);
                }
            });
            if (opt_.dump_hidden) std::memcpy(hid_dump_.data() + ((size_t)(49) * dump_T_ + dump_off_) * H, fin_.data(), (size_t)T * H * 4);
        }
        if (!want_logits && !all_rows) return;
        {
            Scope sc(this, P_HEAD);
            const int nl = all_rows ? T : 1;
            lg_.resize((size_t)nl * V_);
            quant(aqf_, fin_.data() + (size_t)(T - nl) * H, H, nl);
            qjob(head_, aqf_, lg_.data(), (size_t)V_);
            run_jobs(nl, 16);
            const float cap = cfg_.ink.final_logit_softcap;
            pool_.parallel_for((size_t)nl * V_, [&](size_t b, size_t e, int) { softcap_inplace(lg_.data() + b, (int)(e - b), cap); }, 4096);
            if (opt_.dump_hidden) std::memcpy(logits_dump_.data() + (size_t)dump_off_ * V_, lg_.data(), (size_t)nl * V_ * 4);
            if (opt_.row_logits)
                for (int t = 0; t < nl; ++t) opt_.row_logits(pos0 + t, lg_.data() + (size_t)t * V_);
            std::memcpy(logits_.data(), lg_.data() + (size_t)(nl - 1) * V_, (size_t)V_ * 4);
        }
    }

    // ------------------------------------------------------------------------------------------------ state
    struct alignas(64) PaddedAtomic { std::atomic<int> v{0}; };

    ThreadPool pool_;
    EngineOptions opt_;
    int V_ = 0, Tc_ = 1;
    bool prof_ = false;
    void* map_ = nullptr;
    size_t map_len_ = 0, extra_bytes_ = 0, state_bytes_ = 0;
    bool owned_ = false;
    const char* base_ = nullptr;
    std::deque<std::vector<uint16_t>> sm_store_;
    QRows embed_{};
    QMatC head_{};
    const float *embed_norm_ = nullptr, *final_norm_ = nullptr, *out_query_ = nullptr, *msg_q_ = nullptr, *msg_kv_ = nullptr;
    std::vector<const float*> attn_q_, mlp_q_;
    std::vector<LayerW> lw_;
    std::vector<AppInfo> apps_;
    int n_local_ = 0, n_global_ = 0;
    bool kv8_ = false;
    std::vector<std::vector<float>> ksc_, vsc_;
    double decode_bytes_ = 0;
    // activations
    std::vector<float> hid_, resid_, hm_, xm_, aout_, xin_[3], xn_[3], g64_[3], gcat_, gcat3_, prevB_, prevH_, preH_, msgs_, blk_;
    std::vector<float> qraw_, gat_, kraw_, vraw_, rraw_, sraw_, dwk_, dwv_, kc_, vc_, qp_, relp_, part_, ob_, rl_, fin_, rw_, ypair_, gu8_, act8_, actf_,
        logits_, cstate_;
    std::vector<float> lg_;
    std::vector<int> sel_, pairpos_, order_;
    ActQ aq_[3], aqo_, aqm_, aqf_, aq8_[8], aqx_, aqa_;
    PaddedAtomic cnt_[8], ready_[8];
    int ecount_[NE], eoff_[NE], eact_[NE], nact_ = 0;
    int nblk_ = 0, attn_ns_ = 1, dump_off_ = 0, dump_T_ = 0;
    size_t cstride_ = 0;
    struct FreeDel { void operator()(uint16_t* p) const { std::free(p); } };
    std::vector<std::unique_ptr<uint16_t, FreeDel>> kl_, vl_, kg_, vg_;
    std::vector<float> hid_dump_, logits_dump_;
    std::vector<int8_t> router_dump_;
    double ph_ms_[NPH] = {};
    long ph_calls_[NPH] = {};
};

std::unique_ptr<Engine> make_cpu_inkling_engine(const ModelFile& mf, const EngineOptions& opt) {
    return std::make_unique<InkCpuEngine>(mf, opt);
}

}  // namespace qv
