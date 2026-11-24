// Q-Volta CPU decode ops: AVX2 + FMA + F16C implementations. See ops.h for the contract.
#include "ops.h"

#include "cpu_relax.h"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace qv::cpu {
namespace {

[[noreturn]] void fatal(const char* msg) {
    std::fprintf(stderr, "qv_cpu_ops: %s\n", msg);
    std::abort();
}

// ---- vector exp: Cephes-style range reduction + degree-6 polynomial, 2^n applied through the exponent bits ---------
inline __m256 exp256(__m256 x) {
    const __m256 lo = _mm256_set1_ps(-87.3f), hi = _mm256_set1_ps(88.0f);
    const __m256 under = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);  // also true for -inf; NaN stays NaN via min/max order below
    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    __m256 y = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.f));
    const __m256i n = _mm256_cvtps_epi32(fx);
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    y = _mm256_mul_ps(y, _mm256_castsi256_ps(e));
    return _mm256_andnot_ps(under, y);
}

inline __m256 silu256(__m256 x) {
    const __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.f), e));
}

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

inline float sigmoid_f(float x) { return 1.f / (1.f + exp_fast(-x)); }

}  // namespace

float exp_fast(float x) {
    alignas(32) float t[8];
    _mm256_store_ps(t, exp256(_mm256_set1_ps(x)));
    return t[0];
}
float silu_f(float x) { return x / (1.f + exp_fast(-x)); }
float softplus_f(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }

// ---- elementwise ------------------------------------------------------------------------------------------------------
void silu_mul_range(const float* g, const float* u, float* out, int b, int e) {
    int i = b;
    for (; i + 8 <= e; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(silu256(_mm256_loadu_ps(g + i)), _mm256_loadu_ps(u + i)));
    for (; i < e; ++i) out[i] = silu_f(g[i]) * u[i];
}
void silu_range(const float* x, float* out, int b, int e) {
    int i = b;
    for (; i + 8 <= e; i += 8) _mm256_storeu_ps(out + i, silu256(_mm256_loadu_ps(x + i)));
    for (; i < e; ++i) out[i] = silu_f(x[i]);
}
void add_range(const float* a, const float* b, float* out, int s, int e) {
    int i = s;
    for (; i + 8 <= e; i += 8) _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < e; ++i) out[i] = a[i] + b[i];
}

// ---- rmsnorm ------------------------------------------------------------------------------------------------------------
double rmsnorm_sumsq_range(const float* x, int b, int e) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = b;
    for (; i + 16 <= e; i += 16) {
        const __m256 v0 = _mm256_loadu_ps(x + i), v1 = _mm256_loadu_ps(x + i + 8);
        a0 = _mm256_fmadd_ps(v0, v0, a0);
        a1 = _mm256_fmadd_ps(v1, v1, a1);
    }
    double s = (double)hsum(_mm256_add_ps(a0, a1));
    for (; i < e; ++i) s += (double)x[i] * x[i];
    return s;
}
float rmsnorm_scale(double sumsq, int n, float eps) { return (float)(1.0 / std::sqrt(sumsq / n + (double)eps)); }
void rmsnorm_apply_range(const float* x, const float* w, float* out, float r, int b, int e) {
    const __m256 rv = _mm256_set1_ps(r);
    int i = b;
    for (; i + 8 <= e; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), rv), _mm256_loadu_ps(w + i)));
    for (; i < e; ++i) out[i] = x[i] * r * w[i];
}
void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
    rmsnorm_apply_range(x, w, out, rmsnorm_scale(rmsnorm_sumsq_range(x, 0, n), n, eps), 0, n);
}

// ---- conv ----------------------------------------------------------------------------------------------------------------
void conv_step_range(const float* in, const float* w, float* state, float* out, int cb, int ce) {
    // state [C][3] and w [C][4] are array-of-structs: do the (cheap) scalar MACs, then SiLU in vector form.
    for (int c0 = cb; c0 < ce; c0 += 64) {
        const int c1 = c0 + 64 < ce ? c0 + 64 : ce;
        for (int c = c0; c < c1; ++c) {
            const float s0 = state[c * 3], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2], x = in[c];
            const float* wk = w + (size_t)c * 4;
            out[c] = wk[0] * s0 + wk[1] * s1 + wk[2] * s2 + wk[3] * x;
            state[c * 3] = s1;
            state[c * 3 + 1] = s2;
            state[c * 3 + 2] = x;
        }
        silu_range(out + c0, out + c0, 0, c1 - c0);
    }
}

// ---- Gated DeltaNet ------------------------------------------------------------------------------------------------------
void gated_rmsnorm_range(const float* o, const float* z, const float* w, float* out, int dv, float eps, int hb, int he) {
    for (int h = hb; h < he; ++h) {
        const float* oh = o + (size_t)h * dv;
        const double ss = rmsnorm_sumsq_range(oh, 0, dv);
        const float r = (float)(1.0 / std::sqrt(ss / dv + (double)eps));
        const __m256 rv = _mm256_set1_ps(r);
        int i = 0;
        for (; i + 8 <= dv; i += 8) {
            const __m256 y = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(oh + i), rv), _mm256_loadu_ps(w + i));
            _mm256_storeu_ps(out + (size_t)h * dv + i, _mm256_mul_ps(y, silu256(_mm256_loadu_ps(z + (size_t)h * dv + i))));
        }
        for (; i < dv; ++i) out[(size_t)h * dv + i] = (oh[i] * r) * w[i] * silu_f(z[(size_t)h * dv + i]);
    }
}

void gdn_step_range(const GdnArgs& a, int hb, int he) {
    constexpr int DK = 128, DV = 128, CB = 64, NV = CB / 8;
    if (a.dk != DK || a.dv != DV) fatal("gdn_step_range: only dk=dv=128 supported");
    alignas(32) float qs[DK], ks[DK], o[DV];
    const int ratio = a.n_v_heads / a.n_k_heads;
    for (int h = hb; h < he; ++h) {
        const int kh = h / ratio;
        const float* qin = a.conv_out + kh * DK;
        const float* kin = a.conv_out + a.n_k_heads * DK + kh * DK;
        const float* vin = a.conv_out + 2 * a.n_k_heads * DK + h * DV;
        // l2-normalise q and k (eps inside the sqrt), q additionally scaled by 1/sqrt(DK)
        const float qss = (float)rmsnorm_sumsq_range(qin, 0, DK), kss = (float)rmsnorm_sumsq_range(kin, 0, DK);
        const float qf = (1.f / std::sqrt(qss + 1e-6f)) * (1.f / std::sqrt((float)DK));
        const float kf = 1.f / std::sqrt(kss + 1e-6f);
        for (int i = 0; i < DK; ++i) {
            qs[i] = qin[i] * qf;
            ks[i] = kin[i] * kf;
        }
        const float g = a.A_neg[h] * softplus_f(a.alpha[h] + a.dt_bias[h]);
        const float decay = std::exp(g);
        const float beta = 1.f / (1.f + std::exp(-a.beta_logit[h]));
        const __m256 dv_ = _mm256_set1_ps(decay), bv_ = _mm256_set1_ps(beta);
        float* S = a.state + (size_t)h * DK * DV;
        // Two passes per 64-column block; the second pass re-reads rows that are still in L1/L2, so the state is
        // streamed from DRAM once and written back once.
        for (int cb = 0; cb < DV; cb += CB) {
            __m256 km[NV];
            for (int j = 0; j < NV; ++j) km[j] = _mm256_setzero_ps();
            for (int k = 0; k < DK; ++k) {
                const float* row = S + (size_t)k * DV + cb;
                const __m256 kv = _mm256_set1_ps(ks[k]);
                for (int j = 0; j < NV; ++j)
                    km[j] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_loadu_ps(row + 8 * j), dv_), kv, km[j]);
            }
            __m256 delta[NV], acc[NV];
            for (int j = 0; j < NV; ++j) {
                delta[j] = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(vin + cb + 8 * j), km[j]), bv_);
                acc[j] = _mm256_setzero_ps();
            }
            for (int k = 0; k < DK; ++k) {
                float* row = S + (size_t)k * DV + cb;
                const __m256 kv = _mm256_set1_ps(ks[k]), qv = _mm256_set1_ps(qs[k]);
                for (int j = 0; j < NV; ++j) {
                    const __m256 s = _mm256_fmadd_ps(kv, delta[j], _mm256_mul_ps(_mm256_loadu_ps(row + 8 * j), dv_));
                    _mm256_storeu_ps(row + 8 * j, s);
                    acc[j] = _mm256_fmadd_ps(s, qv, acc[j]);
                }
            }
            for (int j = 0; j < NV; ++j) _mm256_store_ps(o + cb + 8 * j, acc[j]);
        }
        gated_rmsnorm_range(o, a.z + (size_t)h * DV, a.norm_w, a.out + (size_t)h * DV, DV, a.eps, 0, 1);
    }
}

// ---- attention prep -------------------------------------------------------------------------------------------------------
void attn_rope_table(int pos, const float* inv_freq, int rope_dim, float* cs) {
    const int half = rope_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float f = (float)pos * inv_freq[i];
        cs[i] = std::cos(f);
        cs[half + i] = std::sin(f);
    }
}

void attn_prep_range(const AttnPrepArgs& a, int ib, int ie) {
    const int D = a.head_dim, half = a.rope_dim / 2;
    if (D > 256 || D % 8) fatal("attn_prep_range: head_dim must be a multiple of 8 and <= 256");
    alignas(32) float y[256];
    for (int it = ib; it < ie; ++it) {
        const bool is_q = it < a.n_heads;
        const int h = is_q ? it : it - a.n_heads;
        const float* x = is_q ? a.qg + (size_t)h * 2 * D : a.k + (size_t)h * D;
        const float* w = is_q ? a.q_norm : a.k_norm;
        const float r = rmsnorm_scale(rmsnorm_sumsq_range(x, 0, D), D, a.eps);
        rmsnorm_apply_range(x, w, y, r, 0, D);  // y = x * r * w (zero-centred norm weights already hold 1+w)
        float* dst;
        if (is_q) {
            dst = a.q_out + (size_t)h * D;
        } else {
            dst = a.k_cache + ((size_t)h * a.max_ctx + a.pos) * D;
            std::memcpy(a.v_cache + ((size_t)h * a.max_ctx + a.pos) * D, a.v + (size_t)h * D, D * sizeof(float));
        }
        const float* c = a.rope_cs;
        const float* s = a.rope_cs + half;
        for (int t = 0; t < a.rope_dim; ++t) {
            const int i = t < half ? t : t - half;
            dst[t] = t < half ? y[t] * c[i] - y[t + half] * s[i] : y[t] * c[i] + y[t - half] * s[i];
        }
        for (int t = a.rope_dim; t < D; ++t) dst[t] = y[t];
    }
}

// ---- attention decode (split-K / flash-decoding on CPU) ---------------------------------------------------------------------
int attn_chunk(int n, int S) {
    const int c = (n + S - 1) / S;
    return c < 32 ? 32 : c;
}
int attn_active_splits(int n, int S) {
    const int c = attn_chunk(n, S);
    const int ns = (n + c - 1) / c;
    return ns < S ? ns : S;
}

namespace {

constexpr int TK = 32;  // keys per online-softmax tile

// scores[h][j] = scale * dot(q_h, K[j]) for G heads and `cnt` keys (two keys per iteration for ILP).
template <int G>
inline void scores_tile(const float* q, const float* K, int D, int cnt, float scale, float (*sc)[TK]) {
    int j = 0;
    for (; j + 2 <= cnt; j += 2) {
        const float* k0 = K + (size_t)j * D;
        const float* k1 = k0 + D;
        __m256 a0[G], a1[G];
        for (int h = 0; h < G; ++h) a0[h] = a1[h] = _mm256_setzero_ps();
        for (int d = 0; d < D; d += 8) {
            const __m256 x0 = _mm256_loadu_ps(k0 + d), x1 = _mm256_loadu_ps(k1 + d);
            for (int h = 0; h < G; ++h) {
                const __m256 qv = _mm256_loadu_ps(q + (size_t)h * D + d);
                a0[h] = _mm256_fmadd_ps(qv, x0, a0[h]);
                a1[h] = _mm256_fmadd_ps(qv, x1, a1[h]);
            }
        }
        for (int h = 0; h < G; ++h) {
            sc[h][j] = hsum(a0[h]) * scale;
            sc[h][j + 1] = hsum(a1[h]) * scale;
        }
    }
    for (; j < cnt; ++j) {
        const float* k0 = K + (size_t)j * D;
        __m256 a0[G];
        for (int h = 0; h < G; ++h) a0[h] = _mm256_setzero_ps();
        for (int d = 0; d < D; d += 8) {
            const __m256 x0 = _mm256_loadu_ps(k0 + d);
            for (int h = 0; h < G; ++h) a0[h] = _mm256_fmadd_ps(_mm256_loadu_ps(q + (size_t)h * D + d), x0, a0[h]);
        }
        for (int h = 0; h < G; ++h) sc[h][j] = hsum(a0[h]) * scale;
    }
}

// acc[h][:] = acc[h][:] * alpha[h] + sum_j p[h][j] * V[j][:]
template <int G>
inline void pv_tile(float* acc, const float* V, int D, int cnt, const float (*p)[TK], const float* alpha) {
    constexpr int W = G <= 6 ? 2 : 1;  // ymm per head per column block (register budget: G*W accumulators)
    for (int d = 0; d < D; d += 8 * W) {
        __m256 a[G][W];
        for (int h = 0; h < G; ++h) {
            const __m256 al = _mm256_set1_ps(alpha[h]);
            for (int w = 0; w < W; ++w) a[h][w] = _mm256_mul_ps(_mm256_loadu_ps(acc + (size_t)h * D + d + 8 * w), al);
        }
        for (int j = 0; j < cnt; ++j) {
            const float* vr = V + (size_t)j * D + d;
            __m256 v[W];
            for (int w = 0; w < W; ++w) v[w] = _mm256_loadu_ps(vr + 8 * w);
            for (int h = 0; h < G; ++h) {
                const __m256 pv = _mm256_set1_ps(p[h][j]);
                for (int w = 0; w < W; ++w) a[h][w] = _mm256_fmadd_ps(pv, v[w], a[h][w]);
            }
        }
        for (int h = 0; h < G; ++h)
            for (int w = 0; w < W; ++w) _mm256_storeu_ps(acc + (size_t)h * D + d + 8 * w, a[h][w]);
    }
}

template <int G>
void decode_item(const AttnDecodeArgs& a, int g, int sp) {
    const int D = a.head_dim, S = a.n_splits, n = a.n_keys;
    const int chunk = attn_chunk(n, S);
    const int j0 = sp * chunk, j1 = n < j0 + chunk ? n : j0 + chunk;
    const int hb = g * G;
    alignas(32) float acc[G * 256];
    alignas(32) float sc[G][TK];
    float m[G], l[G], alpha[G];
    for (int h = 0; h < G; ++h) m[h] = -INFINITY, l[h] = 0.f;
    std::memset(acc, 0, sizeof(float) * G * D);
    const float* Kc = a.k_cache + (size_t)g * a.max_ctx * D;
    const float* Vc = a.v_cache + (size_t)g * a.max_ctx * D;
    const float* q = a.q + (size_t)hb * D;
    const float scale = 1.f / std::sqrt((float)D);
    for (int jt = j0; jt < j1; jt += TK) {
        const int cnt = j1 - jt < TK ? j1 - jt : TK;
        scores_tile<G>(q, Kc + (size_t)jt * D, D, cnt, scale, sc);
        for (int h = 0; h < G; ++h) {
            float tm = -INFINITY;
            for (int j = 0; j < cnt; ++j) tm = tm > sc[h][j] ? tm : sc[h][j];
            const float mn = m[h] > tm ? m[h] : tm;
            for (int j = cnt; j < TK; ++j) sc[h][j] = -INFINITY;  // pad: exp -> 0
            __m256 sv = _mm256_setzero_ps();
            const __m256 mv = _mm256_set1_ps(mn);
            for (int j = 0; j < TK; j += 8) {
                const __m256 p = exp256(_mm256_sub_ps(_mm256_loadu_ps(&sc[h][j]), mv));
                _mm256_storeu_ps(&sc[h][j], p);
                sv = _mm256_add_ps(sv, p);
            }
            alpha[h] = exp_fast(m[h] - mn);  // m = -inf on the first tile -> 0
            l[h] = l[h] * alpha[h] + hsum(sv);
            m[h] = mn;
        }
        pv_tile<G>(acc, Vc + (size_t)jt * D, D, cnt, sc, alpha);
    }
    float* pa = a.part;
    float* pml = a.part + (size_t)a.n_heads * S * D;
    for (int h = 0; h < G; ++h) {
        std::memcpy(pa + ((size_t)(hb + h) * S + sp) * D, acc + (size_t)h * D, D * sizeof(float));
        pml[((size_t)(hb + h) * S + sp) * 2] = m[h];
        pml[((size_t)(hb + h) * S + sp) * 2 + 1] = l[h];
    }
}

}  // namespace

void attn_decode_range(const AttnDecodeArgs& a, int ib, int ie) {
    if (a.head_dim > 256 || a.head_dim % 16) fatal("attn_decode_range: head_dim must be a multiple of 16, <= 256");
    const int G = a.n_heads / a.n_kv_heads;
    for (int it = ib; it < ie; ++it) {
        const int g = it / a.n_splits, sp = it % a.n_splits;
        switch (G) {
            case 1: decode_item<1>(a, g, sp); break;
            case 2: decode_item<2>(a, g, sp); break;
            case 3: decode_item<3>(a, g, sp); break;
            case 4: decode_item<4>(a, g, sp); break;
            case 6: decode_item<6>(a, g, sp); break;
            case 8: decode_item<8>(a, g, sp); break;
            default: fatal("attn_decode_range: unsupported GQA group size (1,2,3,4,6,8)");
        }
    }
}

void attn_merge_range(const AttnDecodeArgs& a, int hb, int he) {
    const int D = a.head_dim, S = a.n_splits;
    const int ns = attn_active_splits(a.n_keys, S);
    alignas(32) float acc[256];
    for (int h = hb; h < he; ++h) {
        const float* ml = a.part + (size_t)a.n_heads * S * D + (size_t)h * S * 2;
        const float* pa = a.part + (size_t)h * S * D;
        float M = -INFINITY;
        for (int s = 0; s < ns; ++s) M = M > ml[2 * s] ? M : ml[2 * s];
        float L = 0.f;
        std::memset(acc, 0, sizeof(float) * D);
        for (int s = 0; s < ns; ++s) {
            const float w = exp_fast(ml[2 * s] - M);
            L = std::fmaf(ml[2 * s + 1], w, L);
            const __m256 wv = _mm256_set1_ps(w);
            for (int d = 0; d < D; d += 8)
                _mm256_store_ps(acc + d, _mm256_fmadd_ps(_mm256_loadu_ps(pa + (size_t)s * D + d), wv, _mm256_load_ps(acc + d)));
        }
        const __m256 invL = _mm256_set1_ps(L);
        const float* gate = a.qg + (size_t)h * 2 * D + D;
        for (int d = 0; d < D; d += 8) {
            const __m256 o = _mm256_div_ps(_mm256_load_ps(acc + d), invL);
            const __m256 gt = _mm256_loadu_ps(gate + d);
            const __m256 sg = _mm256_div_ps(_mm256_set1_ps(1.f),
                                            _mm256_add_ps(_mm256_set1_ps(1.f), exp256(_mm256_sub_ps(_mm256_setzero_ps(), gt))));
            _mm256_storeu_ps(a.out + (size_t)h * D + d, _mm256_mul_ps(o, sg));
        }
    }
}

// ---- embedding ----------------------------------------------------------------------------------------------------------------
void embed_row(const QRows& E, int row, float* out) {
    const int C = E.K >> 5, G = E.K >> 6;
    if (E.K % 64) fatal("embed_row: K % 64 != 0");
    for (int c = 0; c < C; ++c) {
        const uint32_t* w = E.q + ((size_t)row * C + c) * E.bits;
        float scale, mn;
        {
            const uint16_t* sm = E.sm + ((size_t)row * G + (c >> 1)) * 2;
            alignas(16) uint16_t t[8] = {sm[0], sm[1], 0, 0, 0, 0, 0, 0};
            alignas(16) float f[4];
            _mm_store_ps(f, _mm_cvtph_ps(_mm_load_si128((const __m128i*)t)));
            scale = f[0];
            mn = f[1];
        }
        for (int i = 0; i < 32; ++i) {
            uint32_t q;
            switch (E.bits) {
                case 2: q = (w[i & 1] >> (2 * (i >> 1))) & 3u; break;
                case 3: q = ((w[i & 1] >> (2 * (i >> 1))) & 3u) | (((w[2] >> i) & 1u) << 2); break;
                case 4: q = (w[i & 3] >> (4 * (i >> 2))) & 15u; break;
                case 5: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4] >> i) & 1u) << 4); break;
                case 6: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 1)] >> (2 * (i >> 1))) & 3u) << 4); break;
                case 8: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 3)] >> (4 * (i >> 2))) & 15u) << 4); break;
                default: fatal("embed_row: unsupported bits");
            }
            out[c * 32 + i] = scale * (float)q + mn;
        }
    }
}

// ---- argmax ------------------------------------------------------------------------------------------------------------------------
void argmax_range(const float* x, int b, int e, float* best_val, int* best_idx) {
    __m256 bv = _mm256_set1_ps(-INFINITY);
    __m256i bi = _mm256_set1_epi32(INT_MAX);
    __m256i cur = _mm256_add_epi32(_mm256_set1_epi32(b), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    const __m256i step = _mm256_set1_epi32(8);
    int i = b;
    for (; i + 8 <= e; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 gt = _mm256_cmp_ps(v, bv, _CMP_GT_OQ);  // strict: earliest index wins inside a lane; NaN never wins
        bv = _mm256_blendv_ps(bv, v, gt);
        bi = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(bi), _mm256_castsi256_ps(cur), gt));
        cur = _mm256_add_epi32(cur, step);
    }
    alignas(32) float lv[8];
    alignas(32) int li[8];
    _mm256_store_ps(lv, bv);
    _mm256_store_si256((__m256i*)li, bi);
    float best = -INFINITY;
    int idx = INT_MAX;
    for (int k = 0; k < 8; ++k)
        if (lv[k] > best || (lv[k] == best && li[k] < idx)) best = lv[k], idx = li[k];
    for (; i < e; ++i)
        if (x[i] > best || (x[i] == best && i < idx)) best = x[i], idx = i;
    *best_val = best;
    *best_idx = idx;
}

void argmax_merge(const float* vals, const int* idxs, int m, int* out_idx) {
    float best = -INFINITY;
    int idx = INT_MAX;
    for (int k = 0; k < m; ++k)
        if (vals[k] > best || (vals[k] == best && idxs[k] < idx)) best = vals[k], idx = idxs[k];
    *out_idx = idx;
}

}  // namespace qv::cpu
