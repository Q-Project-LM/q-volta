#include "ink_ops.h"

#include "cpu_relax.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

namespace qv::cpu {
namespace {

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// Cephes-style exp (same as ops.cpp): range reduction + degree-6 polynomial
inline __m256 exp256(__m256 x) {
    const __m256 lo = _mm256_set1_ps(-87.3f), hi = _mm256_set1_ps(88.0f);
    const __m256 under = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);
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

// tanh(x) = 1 - 2 / (exp(2x) + 1); |x| clamped (tanh saturates to +-1 in fp32 beyond ~9)
inline __m256 tanh256(__m256 x) {
    const __m256 c = _mm256_set1_ps(10.f);
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_sub_ps(_mm256_setzero_ps(), c)), c);
    const __m256 e = exp256(_mm256_add_ps(x, x));
    return _mm256_sub_ps(_mm256_set1_ps(1.f), _mm256_div_ps(_mm256_set1_ps(2.f), _mm256_add_ps(e, _mm256_set1_ps(1.f))));
}

inline __m256 load_f16x8(const uint16_t* p) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)p)); }

}  // namespace

float dot_f32(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float t = hsum(_mm256_add_ps(s0, s1));
    for (; i < n; ++i) t += a[i] * b[i];
    return t;
}

void exp_vec(const float* x, float* y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, exp256(_mm256_loadu_ps(x + i)));
    for (; i < n; ++i) {
        float t[8] = {x[i], 0, 0, 0, 0, 0, 0, 0}, o[8];
        _mm256_storeu_ps(o, exp256(_mm256_loadu_ps(t)));
        y[i] = o[0];
    }
}

float rms_inv(const float* x, int n, float eps) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 a = _mm256_loadu_ps(x + i), b = _mm256_loadu_ps(x + i + 8);
        s0 = _mm256_fmadd_ps(a, a, s0);
        s1 = _mm256_fmadd_ps(b, b, s1);
    }
    for (; i + 8 <= n; i += 8) { const __m256 a = _mm256_loadu_ps(x + i); s0 = _mm256_fmadd_ps(a, a, s0); }
    float t = hsum(_mm256_add_ps(s0, s1));
    for (; i < n; ++i) t += x[i] * x[i];
    return 1.0f / std::sqrt(t / (float)n + eps);
}

void rms_w(const float* x, const float* w, float* out, int n, float eps) {
    const float r = rms_inv(x, n, eps);
    const __m256 rv = _mm256_set1_ps(r);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(w + i), _mm256_mul_ps(_mm256_loadu_ps(x + i), rv)));
    for (; i < n; ++i) out[i] = w[i] * (x[i] * r);
}

void rms_u(const float* x, float* out, int n, float eps) {
    const float r = rms_inv(x, n, eps);
    const __m256 rv = _mm256_set1_ps(r);
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), rv));
    for (; i < n; ++i) out[i] = x[i] * r;
}

void gemm_f16_rows(const uint16_t* W, int K, const float* X, size_t ldx, int T, float* Y, size_t ldy, int n0, int n1) {
    if (T == 1) {
        for (int n = n0; n < n1; ++n) {
            const uint16_t* w = W + (size_t)n * K;
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                s0 = _mm256_fmadd_ps(load_f16x8(w + k), _mm256_loadu_ps(X + k), s0);
                s1 = _mm256_fmadd_ps(load_f16x8(w + k + 8), _mm256_loadu_ps(X + k + 8), s1);
            }
            for (; k + 8 <= K; k += 8) s0 = _mm256_fmadd_ps(load_f16x8(w + k), _mm256_loadu_ps(X + k), s0);
            float t = hsum(_mm256_add_ps(s0, s1));
            for (; k < K; ++k) t += _cvtsh_ss(w[k]) * X[k];
            Y[n] = t;
        }
        return;
    }
    static thread_local std::vector<float> wf;
    wf.resize((size_t)K);
    for (int n = n0; n < n1; ++n) {
        const uint16_t* w = W + (size_t)n * K;
        int k = 0;
        for (; k + 8 <= K; k += 8) _mm256_storeu_ps(wf.data() + k, load_f16x8(w + k));
        for (; k < K; ++k) wf[k] = _cvtsh_ss(w[k]);
        int t = 0;
        for (; t + 4 <= T; t += 4) {
            const float* x0 = X + (size_t)t * ldx;
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            int k2 = 0;
            for (; k2 + 8 <= K; k2 += 8) {
                const __m256 wv = _mm256_loadu_ps(wf.data() + k2);
                a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + k2), a0);
                a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + ldx + k2), a1);
                a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + 2 * ldx + k2), a2);
                a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + 3 * ldx + k2), a3);
            }
            float r0 = hsum(a0), r1 = hsum(a1), r2 = hsum(a2), r3 = hsum(a3);
            for (; k2 < K; ++k2) { r0 += wf[k2] * x0[k2]; r1 += wf[k2] * x0[ldx + k2]; r2 += wf[k2] * x0[2 * ldx + k2]; r3 += wf[k2] * x0[3 * ldx + k2]; }
            Y[(size_t)t * ldy + n] = r0; Y[(size_t)(t + 1) * ldy + n] = r1; Y[(size_t)(t + 2) * ldy + n] = r2; Y[(size_t)(t + 3) * ldy + n] = r3;
        }
        for (; t < T; ++t) Y[(size_t)t * ldy + n] = dot_f32(wf.data(), X + (size_t)t * ldx, K);
    }
}

void block_attn_res(const float* const* blk, int nb, const float* partial, const float* queries, int Q, int H, float* out) {
    const float* st[8];
    float r[8];
    const int ns = nb + 1;
    for (int b = 0; b < nb; ++b) st[b] = blk[b];
    st[nb] = partial;
    for (int b = 0; b < ns; ++b) r[b] = rms_inv(st[b], H, 1e-6f);
    const float sc = 1.0f / std::sqrt((float)H);
    for (int q = 0; q < Q; ++q) {
        float s[8], m = -FLT_MAX, den = 0.f;
        for (int b = 0; b < ns; ++b) {
            s[b] = dot_f32(queries + (size_t)q * H, st[b], H) * r[b] * sc;
            m = std::max(m, s[b]);
        }
        for (int b = 0; b < ns; ++b) { s[b] = std::exp(s[b] - m); den += s[b]; }
        float c[8];
        for (int b = 0; b < ns; ++b) c[b] = s[b] / den * r[b];
        float* o = out + (size_t)q * H;
        int i = 0;
        for (; i + 8 <= H; i += 8) {
            __m256 a = _mm256_mul_ps(_mm256_set1_ps(c[0]), _mm256_loadu_ps(st[0] + i));
            for (int b = 1; b < ns; ++b) a = _mm256_fmadd_ps(_mm256_set1_ps(c[b]), _mm256_loadu_ps(st[b] + i), a);
            _mm256_storeu_ps(o + i, a);
        }
        for (; i < H; ++i) {
            float a = 0.f;
            for (int b = 0; b < ns; ++b) a += c[b] * st[b][i];
            o[i] = a;
        }
    }
}

void router_topk(const float* logits, const float* bias, int ne, int k, float route_scale, int* idx, float* w) {
    float s[256], sb[256];
    for (int e = 0; e < ne; ++e) {
        s[e] = 1.0f / (1.0f + std::exp(-logits[e]));
        sb[e] = s[e] + bias[e];
    }
    float sum = 0.f;
    for (int j = 0; j < k; ++j) {
        int best = -1;
        for (int e = 0; e < ne; ++e)
            if (sb[e] > -INFINITY && (best < 0 || sb[e] > sb[best])) best = e;
        idx[j] = best;
        w[j] = s[best];
        sum += s[best];
        sb[best] = -INFINITY;
    }
    for (int j = 0; j < k; ++j) w[j] = w[j] / (sum + 1e-20f) * route_scale;
}

void expert_act_interleaved(const float* gu, float* act, int n_pairs, float gcap, float ucap) {
    const __m256 gc = _mm256_set1_ps(gcap), gci = _mm256_set1_ps(1.0f / gcap), uc = _mm256_set1_ps(ucap),
                 uci = _mm256_set1_ps(1.0f / ucap), one = _mm256_set1_ps(1.f), zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n_pairs; i += 8) {
        const __m256 a = _mm256_loadu_ps(gu + 2 * i), b = _mm256_loadu_ps(gu + 2 * i + 8);
#ifdef __aarch64__
        const __m256 ge{vuzp1q_f32(a.lo, a.hi), vuzp1q_f32(b.lo, b.hi)}, ue{vuzp2q_f32(a.lo, a.hi), vuzp2q_f32(b.lo, b.hi)};
#else
        const __m256 ge = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(a, b, 0x88)), 0xD8));
        const __m256 ue = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8));
#endif
        const __m256 gcap_v = _mm256_mul_ps(gc, tanh256(_mm256_mul_ps(ge, gci)));
        const __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256(_mm256_sub_ps(zero, ge))));
        const __m256 ucap_v = _mm256_mul_ps(uc, tanh256(_mm256_mul_ps(ue, uci)));
        _mm256_storeu_ps(act + i, _mm256_mul_ps(_mm256_mul_ps(gcap_v, sig), ucap_v));
    }
    for (; i < n_pairs; ++i) {
        const float g = gu[2 * i], u = gu[2 * i + 1];
        act[i] = gcap * std::tanh(g / gcap) * (1.0f / (1.0f + std::exp(-g))) * (ucap * std::tanh(u / ucap));
    }
}

void softcap_inplace(float* x, int n, float cap) {
    const __m256 c = _mm256_set1_ps(cap), ci = _mm256_set1_ps(1.0f / cap);
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(c, tanh256(_mm256_mul_ps(_mm256_loadu_ps(x + i), ci))));
    for (; i < n; ++i) x[i] = cap * std::tanh(x[i] / cap);
}

void store_f16(const float* x, uint16_t* out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i*)(out + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    for (; i < n; ++i) out[i] = _cvtss_sh(x[i], 0);
}

void store_q8(const float* x, int8_t* out, float* scale) {
    float mx = 0.f;
    for (int i = 0; i < 128; ++i) mx = std::max(mx, std::fabs(x[i]));
    const float sc = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sc;
    for (int i = 0; i < 128; ++i) out[i] = (int8_t)std::max(-127.f, std::min(127.f, std::nearbyint(x[i] * inv)));
    *scale = sc;
}

void ink_attn_partial(const InkAttnArgs& a, float* acc, float* ml) {
    constexpr int D = 128, BK = 64;
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, l[3] = {0, 0, 0};
    std::memset(acc, 0, 3 * D * sizeof(float));
    alignas(32) float s[3][BK], rl[3][BK + 8];
    for (int jb = a.j0; jb < a.j1; jb += BK) {
        const int nk = std::min(BK, a.j1 - jb);
        // relative bias for distances d = p - j, j in [jb, jb+nk): d in [dmin, dmax]; rl[h][d - dmin]
        const int dmin = a.p - (jb + nk - 1), dmax = a.p - jb;
        for (int h = 0; h < 3; ++h) std::memset(rl[h], 0, sizeof(rl[h]));
        if (dmin < a.rel_extent) {
            const int dm = std::min(dmax, a.rel_extent - 1);
            for (int d0 = dmin; d0 <= dm; d0 += 8) {
                __m256 r0 = _mm256_setzero_ps(), r1 = _mm256_setzero_ps(), r2 = _mm256_setzero_ps();
                // d0..d0+7 may run past rel_extent when the block straddles it: guard with a scalar tail
                if (d0 + 8 <= a.rel_extent) {
                    for (int i = 0; i < 16; ++i) {
                        const __m256 pv = _mm256_loadu_ps(a.proj + (size_t)i * a.rel_extent + d0);
                        r0 = _mm256_fmadd_ps(_mm256_set1_ps(a.rel[i]), pv, r0);
                        r1 = _mm256_fmadd_ps(_mm256_set1_ps(a.rel[16 + i]), pv, r1);
                        r2 = _mm256_fmadd_ps(_mm256_set1_ps(a.rel[32 + i]), pv, r2);
                    }
                    _mm256_storeu_ps(&rl[0][d0 - dmin], r0);
                    _mm256_storeu_ps(&rl[1][d0 - dmin], r1);
                    _mm256_storeu_ps(&rl[2][d0 - dmin], r2);
                } else {
                    for (int d = d0; d <= dm; ++d)
                        for (int h = 0; h < 3; ++h) {
                            float t = 0.f;
                            for (int i = 0; i < 16; ++i) t += a.rel[16 * h + i] * a.proj[(size_t)i * a.rel_extent + d];
                            rl[h][d - dmin] = t;
                        }
                }
            }
            // the vector store above can write up to 7 entries past dm (distances >= rel_extent): clear them
            for (int d = std::max(dm + 1, dmin); d <= dmax; ++d)
                for (int h = 0; h < 3; ++h) rl[h][d - dmin] = 0.f;
        }
        float bm[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (int jj = 0; jj < nk; ++jj) {
            const int j = jb + jj;
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps();
            alignas(32) float kf[D];
            const uint16_t* k = nullptr;
            if (a.kq) { const int8_t* kr = a.kq + (size_t)j * D; const float sc = a.ks[j]; for (int d = 0; d < D; ++d) kf[d] = (float)kr[d] * sc; }
            else k = a.kc + (size_t)(a.ring ? j % a.ring : j) * D;
            for (int d = 0; d < D; d += 8) {
                const __m256 kv = a.kq ? _mm256_loadu_ps(kf + d) : load_f16x8(k + d);
                s0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(a.q + d), s0);
                s1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(a.q + D + d), s1);
                s2 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(a.q + 2 * D + d), s2);
            }
            const int di = a.p - j - dmin;  // index into rl
            s[0][jj] = hsum(s0) * a.scale + rl[0][di];
            s[1][jj] = hsum(s1) * a.scale + rl[1][di];
            s[2][jj] = hsum(s2) * a.scale + rl[2][di];
            for (int h = 0; h < 3; ++h) bm[h] = std::max(bm[h], s[h][jj]);
        }
        for (int h = 0; h < 3; ++h) {
            const float mn = std::max(m[h], bm[h]);
            const float alpha = std::exp(m[h] - mn);  // m = -inf initially -> 0
            l[h] *= alpha;
            float* ah = acc + h * D;
            if (alpha != 1.f) {
                const __m256 av = _mm256_set1_ps(alpha);
                for (int d = 0; d < D; d += 8) _mm256_storeu_ps(ah + d, _mm256_mul_ps(av, _mm256_loadu_ps(ah + d)));
            }
            for (int jj = 0; jj < nk; ++jj) s[h][jj] -= mn;
            exp_vec(s[h], s[h], nk);
            float lsum = 0.f;
            for (int jj = 0; jj < nk; ++jj) lsum += s[h][jj];
            l[h] += lsum;
            m[h] = mn;
        }
        for (int jj = 0; jj < nk; ++jj) {
            const int j = jb + jj;
            alignas(32) float vf[D];
            const uint16_t* v = nullptr;
            if (a.vq) { const int8_t* vr = a.vq + (size_t)j * D; const float sc = a.vs[j]; for (int d = 0; d < D; ++d) vf[d] = (float)vr[d] * sc; }
            else v = a.vc + (size_t)(a.ring ? j % a.ring : j) * D;
            const __m256 p0 = _mm256_set1_ps(s[0][jj]), p1 = _mm256_set1_ps(s[1][jj]), p2 = _mm256_set1_ps(s[2][jj]);
            for (int d = 0; d < D; d += 8) {
                const __m256 vv = a.vq ? _mm256_loadu_ps(vf + d) : load_f16x8(v + d);
                _mm256_storeu_ps(acc + d, _mm256_fmadd_ps(p0, vv, _mm256_loadu_ps(acc + d)));
                _mm256_storeu_ps(acc + D + d, _mm256_fmadd_ps(p1, vv, _mm256_loadu_ps(acc + D + d)));
                _mm256_storeu_ps(acc + 2 * D + d, _mm256_fmadd_ps(p2, vv, _mm256_loadu_ps(acc + 2 * D + d)));
            }
        }
    }
    for (int h = 0; h < 3; ++h) { ml[2 * h] = m[h]; ml[2 * h + 1] = l[h]; }
}

void ink_attn_merge(const float* part, int ns, size_t stride, int i, float sink, const float* gate, float* out) {
    constexpr int D = 128;
    float M = sink;
    for (int s = 0; s < ns; ++s) M = std::max(M, part[s * stride + 3 * D + 2 * i]);
    float w[16], L = std::exp(sink - M);
    for (int s = 0; s < ns; ++s) {
        const float m = part[s * stride + 3 * D + 2 * i];
        w[s] = m == -INFINITY ? 0.f : std::exp(m - M);
        L += part[s * stride + 3 * D + 2 * i + 1] * w[s];
    }
    const float inv = 1.0f / L;
    alignas(32) float sg[D];
    for (int d = 0; d < D; ++d) sg[d] = -gate[d];
    exp_vec(sg, sg, D);
    for (int d = 0; d < D; d += 8) {
        __m256 a = _mm256_setzero_ps();
        for (int s = 0; s < ns; ++s)
            a = _mm256_fmadd_ps(_mm256_set1_ps(w[s] * inv), _mm256_loadu_ps(part + s * stride + i * D + d), a);
        const __m256 sig = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_add_ps(_mm256_set1_ps(1.f), _mm256_loadu_ps(sg + d)));
        _mm256_storeu_ps(out + d, _mm256_mul_ps(a, sig));
    }
}

}  // namespace qv::cpu
