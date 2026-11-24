// CPU Inkling vision encoder: fp32 activations, fp16-stored weights packed into 16-column panels, AVX2/FMA micro-kernel (6x16),
// OpenMP over output tiles. Exact same math as the HF module (docs/INKLING_VISION.md section 4-6).
#include "backends/cpu/cpu_relax.h"
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "vision.h"

namespace qv::vision {
namespace {

// Weight W[N][K] (row-major, K contiguous) -> panels [N/16][K][16] fp16 (N padded to 16)
struct PackedW {
    int N = 0, K = 0, NP = 0;
    std::vector<uint16_t> p;
    std::vector<float> bias;  // optional
};

PackedW pack_w(const std::vector<uint16_t>& w, int N, int K) {
    PackedW o;
    o.N = N; o.K = K; o.NP = (N + 15) / 16;
    o.p.assign((size_t)o.NP * K * 16, 0);
#pragma omp parallel for schedule(static)
    for (int pn = 0; pn < o.NP; ++pn)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < 16; ++j) {
                const int n = pn * 16 + j;
                if (n < N) o.p[((size_t)pn * K + k) * 16 + j] = w[(size_t)n * K + k];
            }
    return o;
}

// C[M][N] (ld ldc) = A[M][K] (ld lda) * W^T (+bias). 6x16 tile.
inline void tile6x16(const float* A, int lda, const uint16_t* B, int K, int mr, float acc[6][16]) {
    __m256 c[6][2];
    for (int i = 0; i < 6; ++i) c[i][0] = c[i][1] = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        const __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(B + (size_t)k * 16)));
        const __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(B + (size_t)k * 16 + 8)));
        for (int i = 0; i < mr; ++i) {
            const __m256 a = _mm256_broadcast_ss(A + (size_t)i * lda + k);
            c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
        }
    }
    for (int i = 0; i < mr; ++i) {
        _mm256_storeu_ps(acc[i], c[i][0]);
        _mm256_storeu_ps(acc[i] + 8, c[i][1]);
    }
}

void gemm(const float* A, int M, int lda, const PackedW& W, float* C, int ldc, bool gelu_act = false) {
    const int MB = 96;  // rows per task (keeps A tile in L2)
    const int mblocks = (M + MB - 1) / MB;
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int pn = 0; pn < W.NP; ++pn)
        for (int mb = 0; mb < mblocks; ++mb) {
            const uint16_t* B = W.p.data() + (size_t)pn * W.K * 16;
            const int m0 = mb * MB, m1 = std::min(M, m0 + MB);
            for (int m = m0; m < m1; m += 6) {
                const int mr = std::min(6, m1 - m);
                float acc[6][16];
                tile6x16(A + (size_t)m * lda, lda, B, W.K, mr, acc);
                for (int i = 0; i < mr; ++i)
                    for (int j = 0; j < 16; ++j) {
                        const int n = pn * 16 + j;
                        if (n >= W.N) break;
                        float v = acc[i][j] + (W.bias.empty() ? 0.f : W.bias[n]);
                        if (gelu_act) v = 0.5f * v * (1.f + std::erf(v * 0.70710678118654752f));
                        C[(size_t)(m + i) * ldc + n] = v;
                    }
            }
        }
}

void layernorm(const float* x, const float* g, const float* b, float* y, int rows, int D, float eps) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * D;
        double s = 0, s2 = 0;
        for (int i = 0; i < D; ++i) s += xr[i];
        const double mean = s / D;
        for (int i = 0; i < D; ++i) { const double d = xr[i] - mean; s2 += d * d; }
        const float inv = (float)(1.0 / std::sqrt(s2 / D + eps)), mf = (float)mean;
        for (int i = 0; i < D; ++i) y[(size_t)r * D + i] = (xr[i] - mf) * inv * g[i] + b[i];
    }
}

struct Layer {
    PackedW qkv, out, fc1, fc2;  // qkv: [3*hidden][hidden] with bias
    std::vector<float> ln1g, ln1b, ln2g, ln2b, lam1, lam2;
};

class CpuEncoder : public Encoder {
   public:
    explicit CpuEncoder(std::shared_ptr<VisionFile> f) : f_(f) {
        cfg_ = f->cfg;
        const int D = cfg_.hidden;
        auto P = [&](const std::string& n) { return "model.visual." + n; };
        patch_ = pack_w(f->half(P("backbone.embeddings.patch_embedding.weight")), D, 3 * cfg_.patch * cfg_.patch);
        pos_ = f->floats(P("backbone.embeddings.position_embedding"));
        auto cls = f->floats(P("backbone.embeddings.class_embedding")), clp = f->floats(P("backbone.embeddings.cls_position_embedding"));
        cls_.resize(D);
        for (int i = 0; i < D; ++i) cls_[i] = cls[i] + clp[i];
        pre_g_ = f->floats(P("backbone.ln_pre.weight")); pre_b_ = f->floats(P("backbone.ln_pre.bias"));
        layers_.resize(cfg_.layers);
        for (int l = 0; l < cfg_.layers; ++l) {
            Layer& L = layers_[l];
            const std::string b = "backbone.layers." + std::to_string(l) + ".";
            std::vector<uint16_t> qkv;
            std::vector<float> qb;
            for (const char* nm : {"q_proj", "k_proj", "v_proj"}) {
                auto w = f->half(P(b + "self_attn." + nm + ".weight"));
                qkv.insert(qkv.end(), w.begin(), w.end());
                auto bb = f->floats(P(b + "self_attn." + nm + ".bias"));
                qb.insert(qb.end(), bb.begin(), bb.end());
            }
            L.qkv = pack_w(qkv, 3 * D, D); L.qkv.bias = qb;
            L.out = pack_w(f->half(P(b + "self_attn.out_proj.weight")), D, D); L.out.bias = f->floats(P(b + "self_attn.out_proj.bias"));
            L.fc1 = pack_w(f->half(P(b + "mlp.c_fc.weight")), cfg_.mlp, D); L.fc1.bias = f->floats(P(b + "mlp.c_fc.bias"));
            L.fc2 = pack_w(f->half(P(b + "mlp.c_proj.weight")), D, cfg_.mlp); L.fc2.bias = f->floats(P(b + "mlp.c_proj.bias"));
            L.ln1g = f->floats(P(b + "ln_1.weight")); L.ln1b = f->floats(P(b + "ln_1.bias"));
            L.ln2g = f->floats(P(b + "ln_2.weight")); L.ln2b = f->floats(P(b + "ln_2.bias"));
            L.lam1 = f->floats(P(b + "ls_1.gamma")); L.lam2 = f->floats(P(b + "ls_2.gamma"));
        }
        m1_ = pack_w(f->half(P("proj_in.weight")), cfg_.proj, cfg_.merge * cfg_.merge * D);
        m2_ = pack_w(f->half(P("proj_out.weight")), cfg_.text_hidden, cfg_.proj);
    }
    const char* backend() const override { return "cpu"; }

    void encode(const Pixels& pix, float* out) override {
        const auto t0 = std::chrono::steady_clock::now();
        const int D = cfg_.hidden, T = pix.tokens(), P = 4 * T, N = P + 1, PS = cfg_.patch, HD = D / cfg_.heads, nh = cfg_.heads;
        const int rows = pix.rows * 2, cols = pix.cols * 2, TL = cfg_.tile();
        // patch vectors [P][1200]: token-major, sub-patch (sr,sc) order, inside a patch (y,x,c)
        std::vector<float> pv((size_t)P * 3 * PS * PS);
#pragma omp parallel for
        for (int tk = 0; tk < T; ++tk)
            for (int s = 0; s < 4; ++s) {
                const int sr = s >> 1, sc = s & 1;
                float* dst = pv.data() + ((size_t)tk * 4 + s) * 3 * PS * PS;
                const float* src = pix.data.data() + (size_t)tk * TL * TL * 3;
                for (int y = 0; y < PS; ++y)
                    std::memcpy(dst + (size_t)y * PS * 3, src + ((size_t)(sr * PS + y) * TL + sc * PS) * 3, PS * 3 * sizeof(float));
            }
        std::vector<float> x((size_t)N * D), xn((size_t)N * D), qkv((size_t)N * 3 * D), att((size_t)N * D), tmp((size_t)N * D), hid((size_t)N * cfg_.mlp);
        gemm(pv.data(), P, 3 * PS * PS, patch_, x.data() + D, D);
        std::memcpy(x.data(), cls_.data(), D * sizeof(float));
        // bilinear position embedding (align_corners=False, border clamp), per patch
        std::vector<int> r0(rows), r1(rows), c0(cols), c1(cols);
        std::vector<float> rw(rows), cw(cols);
        axis(rows, r0, r1, rw);
        axis(cols, c0, c1, cw);
#pragma omp parallel for
        for (int p = 0; p < P; ++p) {
            const int tk = p >> 2, s = p & 3, row = (tk / pix.cols) * 2 + (s >> 1), col = (tk % pix.cols) * 2 + (s & 1);
            float* xr = x.data() + (size_t)(p + 1) * D;
            const float* e00 = pos_.data() + ((size_t)r0[row] * cfg_.grid + c0[col]) * D;
            const float* e01 = pos_.data() + ((size_t)r0[row] * cfg_.grid + c1[col]) * D;
            const float* e10 = pos_.data() + ((size_t)r1[row] * cfg_.grid + c0[col]) * D;
            const float* e11 = pos_.data() + ((size_t)r1[row] * cfg_.grid + c1[col]) * D;
            const float a = 1.f - rw[row], b = rw[row], c = 1.f - cw[col], d = cw[col];
            for (int i = 0; i < D; ++i) xr[i] += a * c * e00[i] + a * d * e01[i] + b * c * e10[i] + b * d * e11[i];
        }
        layernorm(x.data(), pre_g_.data(), pre_b_.data(), x.data(), N, D, cfg_.eps);
        // rope tables per sequence index: cos/sin [N][64] (pair-interleaved, width freqs first then height)
        std::vector<float> cs((size_t)N * HD), sn((size_t)N * HD);
        {
            const int nf = HD / 4;  // 16 frequencies per axis
            std::vector<float> inv(nf);
            for (int j = 0; j < nf; ++j) inv[j] = 1.f / std::pow(cfg_.rope_theta, (float)(2 * j) / (float)(HD / 2));
            for (int s = 0; s < N; ++s) {
                float h = 0, w = 0;
                if (s > 0) {
                    const int p = s - 1, tk = p >> 2, sub = p & 3;
                    h = (float)((tk / pix.cols) * 2 + (sub >> 1) + 1);
                    w = (float)((tk % pix.cols) * 2 + (sub & 1) + 1);
                }
                for (int j = 0; j < nf; ++j) {
                    const float aw = w * inv[j], ah = h * inv[j];
                    cs[(size_t)s * HD + 2 * j] = cs[(size_t)s * HD + 2 * j + 1] = std::cos(aw);
                    sn[(size_t)s * HD + 2 * j] = sn[(size_t)s * HD + 2 * j + 1] = std::sin(aw);
                    cs[(size_t)s * HD + 2 * (nf + j)] = cs[(size_t)s * HD + 2 * (nf + j) + 1] = std::cos(ah);
                    sn[(size_t)s * HD + 2 * (nf + j)] = sn[(size_t)s * HD + 2 * (nf + j) + 1] = std::sin(ah);
                }
            }
        }
        for (int l = 0; l < cfg_.layers; ++l) {
            const Layer& L = layers_[l];
            layernorm(x.data(), L.ln1g.data(), L.ln1b.data(), xn.data(), N, D, cfg_.eps);
            gemm(xn.data(), N, D, L.qkv, qkv.data(), 3 * D);
            rope(qkv.data(), cs.data(), sn.data(), N, nh, HD, D);
            attention(qkv.data(), att.data(), N, nh, HD, D);
            gemm(att.data(), N, D, L.out, tmp.data(), D);
#pragma omp parallel for
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < D; ++j) x[(size_t)i * D + j] += L.lam1[j] * tmp[(size_t)i * D + j];
            layernorm(x.data(), L.ln2g.data(), L.ln2b.data(), xn.data(), N, D, cfg_.eps);
            gemm(xn.data(), N, D, L.fc1, hid.data(), cfg_.mlp, true);
            gemm(hid.data(), N, cfg_.mlp, L.fc2, tmp.data(), D);
#pragma omp parallel for
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < D; ++j) x[(size_t)i * D + j] += L.lam2[j] * tmp[(size_t)i * D + j];
        }
        // merger on patch states (skip the class token): [T][4*D] is a plain reshape of x[1..]
        std::vector<float> h1((size_t)T * cfg_.proj);
        gemm(x.data() + D, T, 4 * D, m1_, h1.data(), cfg_.proj, true);
        gemm(h1.data(), T, cfg_.proj, m2_, out, cfg_.text_hidden);
        last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

   private:
    static void axis(int size, std::vector<int>& i0, std::vector<int>& i1, std::vector<float>& w) {
        for (int i = 0; i < size; ++i) {
            const float src = ((float)i + 0.5f) * 32.f / (float)size - 0.5f;
            const float fl = std::floor(src);
            const int a = (int)fl, b = a + 1;
            const float wa = std::max(0.f, 1.f - std::fabs(src - fl)), wb = std::max(0.f, 1.f - std::fabs(src - fl - 1.f));
            i0[i] = std::min(std::max(a, 0), 31);
            i1[i] = std::min(std::max(b, 0), 31);
            w[i] = wb;  // weight of the second tap; first = 1 - wb (== wa when both taps are in range; border-clamped otherwise sums to 1)
            (void)wa;
        }
    }
    static void rope(float* qkv, const float* cs, const float* sn, int N, int nh, int HD, int D) {
#pragma omp parallel for
        for (int s = 0; s < N; ++s)
            for (int h = 0; h < nh; ++h)
                for (int t = 0; t < 2; ++t) {  // q then k
                    float* v = qkv + (size_t)s * 3 * D + t * D + h * HD;
                    const float* c = cs + (size_t)s * HD;
                    const float* sg = sn + (size_t)s * HD;
                    for (int j = 0; j < HD; j += 2) {
                        const float x1 = v[j], x2 = v[j + 1];
                        v[j] = x1 * c[j] - x2 * sg[j];
                        v[j + 1] = x2 * c[j + 1] + x1 * sg[j + 1];
                    }
                }
    }
    static void attention(const float* qkv, float* out, int N, int nh, int HD, int D) {
        const float scale = 1.f / std::sqrt((float)HD);
        const int QB = 4;
        const int nqb = (N + QB - 1) / QB;
#pragma omp parallel
        {
            std::vector<float> sc((size_t)QB * N);
#pragma omp for collapse(2) schedule(dynamic, 4)
            for (int h = 0; h < nh; ++h)
                for (int qb = 0; qb < nqb; ++qb) {
                    const int q0 = qb * QB, nq = std::min(QB, N - q0);
                    float mx[QB];
                    for (int i = 0; i < nq; ++i) mx[i] = -INFINITY;
                    for (int k = 0; k < N; ++k) {
                        const float* kp = qkv + (size_t)k * 3 * D + D + h * HD;
                        for (int i = 0; i < nq; ++i) {
                            const float* qp = qkv + (size_t)(q0 + i) * 3 * D + h * HD;
                            __m256 a = _mm256_setzero_ps();
                            for (int d = 0; d < HD; d += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(qp + d), _mm256_loadu_ps(kp + d), a);
                            __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
                            lo = _mm_hadd_ps(lo, lo);
                            lo = _mm_hadd_ps(lo, lo);
                            const float s = _mm_cvtss_f32(lo) * scale;
                            sc[(size_t)i * N + k] = s;
                            mx[i] = std::max(mx[i], s);
                        }
                    }
                    float sum[QB] = {0, 0, 0, 0};
                    for (int i = 0; i < nq; ++i)
                        for (int k = 0; k < N; ++k) {
                            const float e = std::exp(sc[(size_t)i * N + k] - mx[i]);
                            sc[(size_t)i * N + k] = e;
                            sum[i] += e;
                        }
                    // HD == 64: process two halves of 32 channels (4 ymm accumulators per query)
                    for (int half = 0; half < 2; ++half) {
                        __m256 acc[QB][4];
                        for (int i = 0; i < QB; ++i)
                            for (int j = 0; j < 4; ++j) acc[i][j] = _mm256_setzero_ps();
                        for (int k = 0; k < N; ++k) {
                            const float* vp = qkv + (size_t)k * 3 * D + 2 * D + h * HD + half * 32;
                            const __m256 v0 = _mm256_loadu_ps(vp), v1 = _mm256_loadu_ps(vp + 8), v2 = _mm256_loadu_ps(vp + 16), v3 = _mm256_loadu_ps(vp + 24);
                            for (int i = 0; i < nq; ++i) {
                                const __m256 p = _mm256_set1_ps(sc[(size_t)i * N + k]);
                                acc[i][0] = _mm256_fmadd_ps(p, v0, acc[i][0]);
                                acc[i][1] = _mm256_fmadd_ps(p, v1, acc[i][1]);
                                acc[i][2] = _mm256_fmadd_ps(p, v2, acc[i][2]);
                                acc[i][3] = _mm256_fmadd_ps(p, v3, acc[i][3]);
                            }
                        }
                        for (int i = 0; i < nq; ++i) {
                            const __m256 inv = _mm256_set1_ps(1.f / sum[i]);
                            float* op = out + (size_t)(q0 + i) * D + h * HD + half * 32;
                            for (int j = 0; j < 4; ++j) _mm256_storeu_ps(op + 8 * j, _mm256_mul_ps(acc[i][j], inv));
                        }
                    }
                }
        }
    }

    std::shared_ptr<VisionFile> f_;
    PackedW patch_, m1_, m2_;
    std::vector<float> pos_, cls_, pre_g_, pre_b_;
    std::vector<Layer> layers_;
};

}  // namespace

std::unique_ptr<Encoder> make_cpu_encoder(std::shared_ptr<VisionFile> f) { return std::make_unique<CpuEncoder>(f); }

}  // namespace qv::vision
