// ARM NEON (dotprod) implementation of the CPU quantized GEMV / GEMM. Same public API, weight layout and activation
// layout as cpu_gemv.cpp (AVX2); see the port map in cpu_gemv.h. Requires FEAT_DotProd (SDOT): every 64-weight group is
// processed as two 32-weight chunks (c = 0,1), each chunk is one 128-bit register pair (lo nibble-plane view, hi view).
#include <arm_neon.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cpu_gemv.h"

namespace qv::cpu {

namespace {
[[noreturn]] void throw_too_many_tokens() { throw std::invalid_argument("gemv_multi_q: T > 4"); }
inline float h2f(const uint16_t* p) { return (float)*reinterpret_cast<const __fp16*>(p); }
}  // namespace

void quantize_act_q8(const float* x, int K, Q8ActC& out) {
    out.K = K;
    out.q.assign((size_t)K, 0);
    out.ds.assign((size_t)(K / 64) * 4, 0.f);
    quantize_act_q8_range(x, 0, K / 64, out);
}

void quantize_act_q8_range(const float* x, int g0, int g1, Q8ActC& out) {
    quantize_act_q8_raw(x, g0, g1, out.q.data(), out.ds.data());
}

void quantize_act_q8_raw(const float* x, int g0, int g1, int8_t* qout, float* dsout) {
    // natural index n -> stored position (q8_perm), as a byte table for TBL scatter: stored[pos] = nat[inv[pos]]
    static uint8_t inv[64];
    static bool init = [] { for (int n = 0; n < 64; ++n) inv[q8_perm(n)] = (uint8_t)n; return true; }();
    (void)init;
    for (int g = g0; g < g1; ++g) {
        const float* xg = x + (size_t)g * 64;
        float32x4_t v[16];
        for (int i = 0; i < 16; ++i) v[i] = vld1q_f32(xg + 4 * i);
        float dsub[2], ssub[2];
        for (int c = 0; c < 2; ++c) {
            float32x4_t m = vabsq_f32(v[8 * c]), s = v[8 * c];
            for (int i = 1; i < 8; ++i) { m = vmaxq_f32(m, vabsq_f32(v[8 * c + i])); s = vaddq_f32(s, v[8 * c + i]); }
            dsub[c] = vmaxvq_f32(m) / 127.f;
            ssub[c] = vaddvq_f32(s);
        }
        dsout[4 * g] = dsub[0];
        dsout[4 * g + 1] = dsub[1];
        dsout[4 * g + 2] = ssub[0];
        dsout[4 * g + 3] = ssub[1];
        int8_t* o = qout + (size_t)g * 64;
        if (!(dsub[0] > 0.f) && !(dsub[1] > 0.f)) { std::memset(o, 0, 64); continue; }
        const float32x4_t d0 = vdupq_n_f32(dsub[0] > 0.f ? dsub[0] : 1.f), d1 = vdupq_n_f32(dsub[1] > 0.f ? dsub[1] : 1.f);
        int8x16_t nat[4];
        for (int i = 0; i < 4; ++i) {
            const float32x4_t dv = i < 2 ? d0 : d1;
            int16x8_t a = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vdivq_f32(v[4 * i], dv))),
                                       vqmovn_s32(vcvtnq_s32_f32(vdivq_f32(v[4 * i + 1], dv))));
            int16x8_t b = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vdivq_f32(v[4 * i + 2], dv))),
                                       vqmovn_s32(vcvtnq_s32_f32(vdivq_f32(v[4 * i + 3], dv))));
            nat[i] = vcombine_s8(vqmovn_s16(a), vqmovn_s16(b));
        }
        int8x16x4_t nt;
        nt.val[0] = nat[0]; nt.val[1] = nat[1]; nt.val[2] = nat[2]; nt.val[3] = nat[3];
        for (int i = 0; i < 4; ++i)
            vst1q_s8(o + 16 * i, vqtbl4q_s8(nt, vld1q_u8(inv + 16 * i)));
    }
}

namespace {

struct Pl { uint8x16_t lo[2], hi[2]; };  // per 32-chunk c: bytes pairing with xlo_c / xhi_c

template <int BITS>
inline void plane4(const uint32_t* p, int off, uint8x16_t& lo0, uint8x16_t& hi0, uint8x16_t& lo1, uint8x16_t& hi1) {
    const uint8x16_t m = vdupq_n_u8(0x0f);
    const uint8x16_t w0 = vld1q_u8((const uint8_t*)(p + off));
    const uint8x16_t w1 = vld1q_u8((const uint8_t*)(p + BITS + off));
    lo0 = vandq_u8(w0, m); hi0 = vshrq_n_u8(w0, 4);
    lo1 = vandq_u8(w1, m); hi1 = vshrq_n_u8(w1, 4);
}
template <int BITS, int SH>
inline void plane2(const uint32_t* p, int off, uint8x16_t& lo0, uint8x16_t& hi0, uint8x16_t& lo1, uint8x16_t& hi1) {
    static const int32_t sl[4] = {0, 0, -2, -2}, sh_[4] = {-4, -4, -6, -6};
    const int32x4_t sll = vld1q_s32(sl), shh = vld1q_s32(sh_);
    const uint8x16_t m = vdupq_n_u8(3);
    const uint32x4_t v0 = vcombine_u32(vld1_u32(p + off), vld1_u32(p + off));
    const uint32x4_t v1 = vcombine_u32(vld1_u32(p + BITS + off), vld1_u32(p + BITS + off));
    lo0 = vshlq_n_u8(vandq_u8(vreinterpretq_u8_u32(vshlq_u32(v0, sll)), m), SH);
    hi0 = vshlq_n_u8(vandq_u8(vreinterpretq_u8_u32(vshlq_u32(v0, shh)), m), SH);
    lo1 = vshlq_n_u8(vandq_u8(vreinterpretq_u8_u32(vshlq_u32(v1, sll)), m), SH);
    hi1 = vshlq_n_u8(vandq_u8(vreinterpretq_u8_u32(vshlq_u32(v1, shh)), m), SH);
}
template <int BITS, int SH>
inline void plane1(const uint32_t* p, int off, uint8x16_t& lo0, uint8x16_t& hi0, uint8x16_t& lo1, uint8x16_t& hi1) {
    static const int32_t sl[4] = {0, -1, -2, -3}, sh_[4] = {-4, -5, -6, -7};
    const int32x4_t sll = vld1q_s32(sl), shh = vld1q_s32(sh_);
    const uint8x16_t m = vdupq_n_u8(1);
    const uint32x4_t v0 = vdupq_n_u32(p[off]);
    const uint32x4_t v1 = vdupq_n_u32(p[BITS + off]);
    lo0 = vshlq_n_u8(vandq_u8(vreinterpretq_u8_u32(vshlq_u32(v0, sll)), m), SH);
    hi0 = vshlq_n_u8(vandq_u8(vreinterpretq_u8_u32(vshlq_u32(v0, shh)), m), SH);
    lo1 = vshlq_n_u8(vandq_u8(vreinterpretq_u8_u32(vshlq_u32(v1, sll)), m), SH);
    hi1 = vshlq_n_u8(vandq_u8(vreinterpretq_u8_u32(vshlq_u32(v1, shh)), m), SH);
}

// Rebuild the 64 q bytes of one group (BITS != 8): lo[c] pairs with xlo_c, hi[c] with xhi_c.
template <int BITS>
inline void unpack_group(const uint32_t* p, uint8x16_t* lo, uint8x16_t* hi) {
    if constexpr (BITS == 4) {
        plane4<4>(p, 0, lo[0], hi[0], lo[1], hi[1]);
    } else if constexpr (BITS == 6) {
        uint8x16_t l0, h0, l1, h1;
        plane4<6>(p, 0, lo[0], hi[0], lo[1], hi[1]);
        plane2<6, 4>(p, 4, l0, h0, l1, h1);
        lo[0] = vorrq_u8(lo[0], l0); hi[0] = vorrq_u8(hi[0], h0);
        lo[1] = vorrq_u8(lo[1], l1); hi[1] = vorrq_u8(hi[1], h1);
    } else if constexpr (BITS == 5) {
        uint8x16_t l0, h0, l1, h1;
        plane4<5>(p, 0, lo[0], hi[0], lo[1], hi[1]);
        plane1<5, 4>(p, 4, l0, h0, l1, h1);
        lo[0] = vorrq_u8(lo[0], l0); hi[0] = vorrq_u8(hi[0], h0);
        lo[1] = vorrq_u8(lo[1], l1); hi[1] = vorrq_u8(hi[1], h1);
    } else if constexpr (BITS == 3) {
        uint8x16_t l0, h0, l1, h1;
        plane2<3, 0>(p, 0, lo[0], hi[0], lo[1], hi[1]);
        plane1<3, 2>(p, 2, l0, h0, l1, h1);
        lo[0] = vorrq_u8(lo[0], l0); hi[0] = vorrq_u8(hi[0], h0);
        lo[1] = vorrq_u8(lo[1], l1); hi[1] = vorrq_u8(hi[1], h1);
    } else {  // 2
        plane2<2, 0>(p, 0, lo[0], hi[0], lo[1], hi[1]);
    }
}

inline int32x4_t sdot(int32x4_t a, uint8x16_t w, int8x16_t x) { return vdotq_s32(a, vreinterpretq_s8_u8(w), x); }

// int32x4 partial dots of one 64-weight group, one vector per 32-chunk: d[c].
template <int BITS>
inline void dot_group(const uint32_t* p, const int8x16_t* xlo, const int8x16_t* xhi, int32x4_t* d) {
    if constexpr (BITS == 8) {
        uint8x16_t al[2], ah[2], bl[2], bh[2];
        plane4<8>(p, 0, al[0], ah[0], al[1], ah[1]);
        plane4<8>(p, 4, bl[0], bh[0], bl[1], bh[1]);
        for (int c = 0; c < 2; ++c) {
            int32x4_t sa = sdot(sdot(vdupq_n_s32(0), al[c], xlo[c]), ah[c], xhi[c]);
            int32x4_t sb = sdot(sdot(vdupq_n_s32(0), bl[c], xlo[c]), bh[c], xhi[c]);
            d[c] = vaddq_s32(sa, vshlq_n_s32(sb, 4));
        }
    } else {
        uint8x16_t lo[2], hi[2];
        unpack_group<BITS>(p, lo, hi);
        for (int c = 0; c < 2; ++c) d[c] = sdot(sdot(vdupq_n_s32(0), lo[c], xlo[c]), hi[c], xhi[c]);
    }
}

template <int BITS, int R>
void rows_kernel(const QMatC& W, const Q8ActC& x, float* y, int r0) {
    const int G = W.K / 64;
    const size_t rw = W.row_words();
    const uint32_t* wp[R];
    const uint16_t* sp[R];
    float32x4_t acc[R][2];
    float accm[R];
    for (int r = 0; r < R; ++r) {
        wp[r] = W.q + (size_t)(r0 + r) * rw;
        sp[r] = W.sm + (size_t)(r0 + r) * (G >> W.gs) * 2;
        acc[r][0] = acc[r][1] = vdupq_n_f32(0.f);
        accm[r] = 0.f;
    }
    const int8_t* xq = x.q.data();
    const float* ds = x.ds.data();
    constexpr int PF = 64;
    for (int g = 0; g < G; ++g) {
        int8x16_t xlo[2], xhi[2];
        for (int c = 0; c < 2; ++c) {
            xlo[c] = vld1q_s8(xq + 64 * g + 16 * c);
            xhi[c] = vld1q_s8(xq + 64 * g + 32 + 16 * c);
        }
        const float d0 = ds[4 * g], d1 = ds[4 * g + 1], ssum = ds[4 * g + 2] + ds[4 * g + 3];
        for (int r = 0; r < R; ++r) {
            const uint32_t* p = wp[r] + (size_t)g * 2 * BITS;
            __builtin_prefetch(p + PF);
            int32x4_t d[2];
            dot_group<BITS>(p, xlo, xhi, d);
            const float sc = h2f(sp[r] + 2 * (g >> W.gs)), mn = h2f(sp[r] + 2 * (g >> W.gs) + 1);
            accm[r] += mn * ssum;
            acc[r][0] = vfmaq_n_f32(acc[r][0], vcvtq_f32_s32(d[0]), d0 * sc);
            acc[r][1] = vfmaq_n_f32(acc[r][1], vcvtq_f32_s32(d[1]), d1 * sc);
        }
    }
    for (int r = 0; r < R; ++r) y[r0 + r] = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1])) + accm[r];
}

template <int BITS, int R, int T>
void rows_kernel_multi(const QMatC& W, const Q8Batch& X, float* Y, size_t ldy, int r0) {
    const int G = W.K / 64;
    const size_t rw = W.row_words();
    const uint32_t* wp[R];
    const uint16_t* sp[R];
    float32x4_t acc[R][T][2];
    float accm[R][T];
    for (int r = 0; r < R; ++r) {
        wp[r] = W.q + (size_t)(r0 + r) * rw;
        sp[r] = W.sm + (size_t)(r0 + r) * (G >> W.gs) * 2;
        for (int t = 0; t < T; ++t) { acc[r][t][0] = acc[r][t][1] = vdupq_n_f32(0.f); accm[r][t] = 0.f; }
    }
    constexpr int PF = 64;
    for (int g = 0; g < G; ++g) {
        int8x16_t xlo[T][2], xhi[T][2];
        float d0[T], d1[T], ssum[T];
        for (int t = 0; t < T; ++t) {
            const int8_t* xq = X.q.data() + (size_t)t * X.K;
            const float* ds = X.ds.data() + (size_t)t * X.G * 4;
            for (int c = 0; c < 2; ++c) {
                xlo[t][c] = vld1q_s8(xq + 64 * g + 16 * c);
                xhi[t][c] = vld1q_s8(xq + 64 * g + 32 + 16 * c);
            }
            d0[t] = ds[4 * g]; d1[t] = ds[4 * g + 1]; ssum[t] = ds[4 * g + 2] + ds[4 * g + 3];
        }
        for (int r = 0; r < R; ++r) {
            const uint32_t* p = wp[r] + (size_t)g * 2 * BITS;
            __builtin_prefetch(p + PF);
            const float sc = h2f(sp[r] + 2 * (g >> W.gs)), mn = h2f(sp[r] + 2 * (g >> W.gs) + 1);
            uint8x16_t lo[2], hi[2];  // unpacked once, applied to all T tokens (BITS != 8 only)
            unpack_group<BITS>(p, lo, hi);
            for (int t = 0; t < T; ++t) {
                int32x4_t d[2];
                for (int c = 0; c < 2; ++c) d[c] = sdot(sdot(vdupq_n_s32(0), lo[c], xlo[t][c]), hi[c], xhi[t][c]);
                accm[r][t] += mn * ssum[t];
                acc[r][t][0] = vfmaq_n_f32(acc[r][t][0], vcvtq_f32_s32(d[0]), d0[t] * sc);
                acc[r][t][1] = vfmaq_n_f32(acc[r][t][1], vcvtq_f32_s32(d[1]), d1[t] * sc);
            }
        }
    }
    for (int r = 0; r < R; ++r)
        for (int t = 0; t < T; ++t)
            Y[(size_t)t * ldy + r0 + r] = vaddvq_f32(vaddq_f32(acc[r][t][0], acc[r][t][1])) + accm[r][t];
}

template <int BITS, int T>
void multi_bits_T(const QMatC& W, const Q8Batch& X, float* Y, size_t ldy, int row0, int row1) {
    int r = row0;
    for (; r + 4 <= row1; r += 4) rows_kernel_multi<BITS, 4, T>(W, X, Y, ldy, r);
    for (; r < row1; ++r) rows_kernel_multi<BITS, 1, T>(W, X, Y, ldy, r);
}
template <int BITS>
void multi_bits(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1) {
    switch (T) {
        case 1: multi_bits_T<BITS, 1>(W, X, Y, ldy, row0, row1); break;
        case 2: multi_bits_T<BITS, 2>(W, X, Y, ldy, row0, row1); break;
        case 3: multi_bits_T<BITS, 3>(W, X, Y, ldy, row0, row1); break;
        default: multi_bits_T<BITS, 4>(W, X, Y, ldy, row0, row1); break;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Batched GEMM (same loop nest as the AVX2 version): unpack RB rows x KCG groups -> int8 buffer, then register tiles.
#ifndef QV_GEMM_TB
#define QV_GEMM_TB 2
#endif
#ifndef QV_GEMM_RT
#define QV_GEMM_RT 4
#endif
constexpr int kRT = QV_GEMM_RT;
#ifndef QV_GEMM_RB
#define QV_GEMM_RB 16
#endif
#ifndef QV_GEMM_TP
#define QV_GEMM_TP 64
#endif
constexpr int kRB = QV_GEMM_RB;
constexpr int kTP = QV_GEMM_TP;
constexpr int kKCG = 16;
constexpr int kTB = QV_GEMM_TB;

template <int RT, int TB>
inline void tile_kernel(const int8_t* wb, size_t wrow, const float* sc, size_t scrow, const int8_t* xq, size_t xqrow,
                        const float* dv, size_t dvrow, float* acc, size_t accrow, int ng) {
    float32x4_t a[RT][TB][2];
    for (int r = 0; r < RT; ++r)
        for (int t = 0; t < TB; ++t) {
            a[r][t][0] = vld1q_f32(acc + r * accrow + t * 8);
            a[r][t][1] = vld1q_f32(acc + r * accrow + t * 8 + 4);
        }
    for (int g = 0; g < ng; ++g) {
        int8x16_t wlo[RT][2], whi[RT][2];
        float s[RT];
        for (int r = 0; r < RT; ++r) {
            const int8_t* w = wb + r * wrow + 64 * g;
            wlo[r][0] = vld1q_s8(w); wlo[r][1] = vld1q_s8(w + 16);
            whi[r][0] = vld1q_s8(w + 32); whi[r][1] = vld1q_s8(w + 48);
            s[r] = sc[r * scrow + g];
        }
        for (int t = 0; t < TB; ++t) {
            const int8_t* x = xq + t * xqrow + 64 * g;
            const int8x16_t xlo0 = vld1q_s8(x), xlo1 = vld1q_s8(x + 16), xhi0 = vld1q_s8(x + 32), xhi1 = vld1q_s8(x + 48);
            const float32x4_t dA = vld1q_f32(dv + t * dvrow + 8 * g), dB = vld1q_f32(dv + t * dvrow + 8 * g + 4);
            for (int r = 0; r < RT; ++r) {
                const int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wlo[r][0], xlo0), whi[r][0], xhi0);
                const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wlo[r][1], xlo1), whi[r][1], xhi1);
                a[r][t][0] = vfmaq_f32(a[r][t][0], vcvtq_f32_s32(p0), vmulq_n_f32(dA, s[r]));
                a[r][t][1] = vfmaq_f32(a[r][t][1], vcvtq_f32_s32(p1), vmulq_n_f32(dB, s[r]));
            }
        }
    }
    for (int r = 0; r < RT; ++r)
        for (int t = 0; t < TB; ++t) {
            vst1q_f32(acc + r * accrow + t * 8, a[r][t][0]);
            vst1q_f32(acc + r * accrow + t * 8 + 4, a[r][t][1]);
        }
}

template <int RT, int TB>
inline void tile_dispatch(int tn, const int8_t* wb, size_t wrow, const float* sc, size_t scrow, const int8_t* xq, size_t xqrow,
                          const float* dv, size_t dvrow, float* acc, size_t accrow, int ng) {
    if (tn == TB) tile_kernel<RT, TB>(wb, wrow, sc, scrow, xq, xqrow, dv, dvrow, acc, accrow, ng);
    else if constexpr (TB > 1) tile_dispatch<RT, TB - 1>(tn, wb, wrow, sc, scrow, xq, xqrow, dv, dvrow, acc, accrow, ng);
}

template <int RT>
inline void tile_tokens(const int8_t* wb, size_t wrow, const float* sc, size_t scrow, const Q8Batch& X, int t0, int tn,
                        int g0, int ng, float* acc, size_t accrow) {
    const size_t xqrow = (size_t)X.K, dvrow = (size_t)X.G * 8;
    const int8_t* xq = X.q.data() + (size_t)t0 * xqrow + 64 * (size_t)g0;
    const float* dv = X.dv8.data() + (size_t)t0 * dvrow + 8 * (size_t)g0;
    tile_dispatch<RT, kTB>(tn, wb, wrow, sc, scrow, xq, xqrow, dv, dvrow, acc, accrow, ng);
}

template <int BITS>
void gemm_block(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int r0, int nr, int xoff) {
    const int G = W.K / 64;
    const size_t rw = W.row_words();
    alignas(64) static thread_local int8_t wb[kRB * kKCG * 64];
    static thread_local std::vector<float> accv, scv, mnv;
    scv.resize((size_t)nr * G);
    mnv.resize((size_t)nr * G);
    for (int r = 0; r < nr; ++r) {
        const uint16_t* sp = W.sm + (size_t)(r0 + r) * (G >> W.gs) * 2;
        for (int g = 0; g < G; ++g) {
            scv[(size_t)r * G + g] = h2f(sp + 2 * (g >> W.gs));
            mnv[(size_t)r * G + g] = h2f(sp + 2 * (g >> W.gs) + 1);
        }
    }
    for (int tp0 = 0; tp0 < T; tp0 += kTP) {
        const int tpn = T - tp0 < kTP ? T - tp0 : kTP;
        const size_t accrow = (size_t)tpn * 8;
        accv.assign((size_t)nr * tpn * 8, 0.f);
        for (int g0 = 0; g0 < G; g0 += kKCG) {
            const int ng = G - g0 < kKCG ? G - g0 : kKCG;
            for (int r = 0; r < nr; ++r) {
                const uint32_t* p = W.q + (size_t)(r0 + r) * rw + (size_t)g0 * 2 * BITS;
                for (int g = 0; g < ng; ++g) {
                    uint8x16_t lo[2], hi[2];
                    unpack_group<BITS>(p + (size_t)g * 2 * BITS, lo, hi);
                    uint8_t* o = (uint8_t*)(wb + ((size_t)r * kKCG + g) * 64);
                    vst1q_u8(o, lo[0]); vst1q_u8(o + 16, lo[1]);
                    vst1q_u8(o + 32, hi[0]); vst1q_u8(o + 48, hi[1]);
                }
            }
            for (int tl = 0; tl < tpn; tl += kTB) {
                const int tn = tpn - tl < kTB ? tpn - tl : kTB;
                int r = 0;
                for (; r + kRT <= nr; r += kRT)
                    tile_tokens<kRT>(wb + (size_t)r * kKCG * 64, (size_t)kKCG * 64, scv.data() + (size_t)r * G + g0, G, X,
                                     xoff + tp0 + tl, tn, g0, ng, accv.data() + r * accrow + tl * 8, accrow);
                for (; r < nr; ++r)
                    tile_tokens<1>(wb + (size_t)r * kKCG * 64, (size_t)kKCG * 64, scv.data() + (size_t)r * G + g0, G, X,
                                   xoff + tp0 + tl, tn, g0, ng, accv.data() + r * accrow + tl * 8, accrow);
            }
        }
        for (int tl = 0; tl < tpn; ++tl) {
            const int t = tp0 + tl;
            const float* xs = X.xs.data() + (size_t)(xoff + t) * G;
            for (int r = 0; r < nr; ++r) {
                const float* mn = mnv.data() + (size_t)r * G;
                float32x4_t m = vdupq_n_f32(0.f);
                int g = 0;
                for (; g + 4 <= G; g += 4) m = vfmaq_f32(m, vld1q_f32(mn + g), vld1q_f32(xs + g));
                float tail = 0.f;
                for (; g < G; ++g) tail += mn[g] * xs[g];
                const float* a = accv.data() + r * accrow + (size_t)tl * 8;
                Y[(size_t)t * ldy + r0 + r] = vaddvq_f32(vaddq_f32(vld1q_f32(a), vld1q_f32(a + 4))) + vaddvq_f32(m) + tail;
            }
        }
    }
}

template <int BITS>
void gemm_bits(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1, int xoff) {
    for (int r = row0; r < row1; r += kRB) gemm_block<BITS>(W, X, T, Y, ldy, r, row1 - r < kRB ? row1 - r : kRB, xoff);
}

template <int BITS>
void gemv_bits(const QMatC& W, const Q8ActC& x, float* y, int row0, int row1) {
    int r = row0;
    for (; r + 4 <= row1; r += 4) rows_kernel<BITS, 4>(W, x, y, r);
    for (; r < row1; ++r) rows_kernel<BITS, 1>(W, x, y, r);
}

}  // namespace

void gemv_q(const QMatC& W, const Q8ActC& x, float* y, int row0, int row1) {
    switch (W.bits) {
        case 2: gemv_bits<2>(W, x, y, row0, row1); break;
        case 3: gemv_bits<3>(W, x, y, row0, row1); break;
        case 4: gemv_bits<4>(W, x, y, row0, row1); break;
        case 5: gemv_bits<5>(W, x, y, row0, row1); break;
        case 6: gemv_bits<6>(W, x, y, row0, row1); break;
        case 8: gemv_bits<8>(W, x, y, row0, row1); break;
        default: break;
    }
}

void gemv_q_par(ThreadPool& pool, const QMatC& W, const Q8ActC& x, float* y) {
    pool.parallel_for_dyn(
        (size_t)W.N, [&](size_t b, size_t e, int) { gemv_q(W, x, y, (int)b, (int)e); }, 16);
}

void Q8Batch::quantize_row(int t, const float* x, int kuse) {
    const int Gu = kuse / 64;
    quantize_act_q8_raw(x, 0, Gu, q.data() + (size_t)t * K, ds.data() + (size_t)t * G * 4);
    const float* d = ds.data() + (size_t)t * G * 4;
    float* o = dv8.data() + (size_t)t * G * 8;
    float* s = xs.data() + (size_t)t * G;
    for (int g = 0; g < Gu; ++g) {
        for (int i = 0; i < 4; ++i) { o[8 * g + i] = d[4 * g]; o[8 * g + 4 + i] = d[4 * g + 1]; }
        s[g] = d[4 * g + 2] + d[4 * g + 3];
    }
}

namespace {
void per_token_gemv(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1, int xoff) {
    Q8ActC a;
    a.K = X.K;
    a.q.resize((size_t)X.K);
    a.ds.resize((size_t)X.G * 4);
    std::vector<float> y((size_t)W.N);
    for (int t = 0; t < T; ++t) {
        std::memcpy(a.q.data(), X.q.data() + (size_t)(xoff + t) * X.K, X.K);
        std::memcpy(a.ds.data(), X.ds.data() + (size_t)(xoff + t) * X.G * 4, (size_t)X.G * 16);
        gemv_q(W, a, y.data(), row0, row1);
        for (int r = row0; r < row1; ++r) Y[(size_t)t * ldy + r] = y[r];
    }
}
}  // namespace

void gemm_q(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1, int xoff) {
    if (T <= 0) return;
    switch (W.bits) {
        case 2: gemm_bits<2>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 3: gemm_bits<3>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 4: gemm_bits<4>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 5: gemm_bits<5>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 6: gemm_bits<6>(W, X, T, Y, ldy, row0, row1, xoff); break;
        case 8: per_token_gemv(W, X, T, Y, ldy, row0, row1, xoff); break;
        default: break;
    }
}

void gemv_multi_q(const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy, int row0, int row1) {
    if (T <= 0) return;
    if (T > 4) throw_too_many_tokens();
    switch (W.bits) {
        case 2: multi_bits<2>(W, X, T, Y, ldy, row0, row1); break;
        case 3: multi_bits<3>(W, X, T, Y, ldy, row0, row1); break;
        case 4: multi_bits<4>(W, X, T, Y, ldy, row0, row1); break;
        case 5: multi_bits<5>(W, X, T, Y, ldy, row0, row1); break;
        case 6: multi_bits<6>(W, X, T, Y, ldy, row0, row1); break;
        default: per_token_gemv(W, X, T, Y, ldy, row0, row1, 0);
    }
}

void gemv_multi_q_par(ThreadPool& pool, const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy) {
    pool.parallel_for_dyn((size_t)W.N, [&](size_t b, size_t e, int) { gemv_multi_q(W, X, T, Y, ldy, (int)b, (int)e); }, 16);
}

void gemm_q_par(ThreadPool& pool, const QMatC& W, const Q8Batch& X, int T, float* Y, size_t ldy) {
    pool.parallel_for_dyn((size_t)W.N, [&](size_t b, size_t e, int) { gemm_q(W, X, T, Y, ldy, (int)b, (int)e); }, kRB);
}

}  // namespace qv::cpu
