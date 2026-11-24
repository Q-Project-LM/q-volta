// Minimal AVX2/F16C -> NEON shim: just the intrinsics the portable-C++ parts of the CPU backend use (layer ops, Inkling
// engine, Qwen engine). __m256 is a pair of float32x4_t; semantics follow Intel's (round-to-nearest-even, cmp masks,
// blendv on the sign bit). The hot GEMV/GEMM kernels do NOT use this; they have a native NEON file (cpu_gemv_neon.cpp).
#pragma once
#if defined(__aarch64__)
#include <arm_neon.h>
#include <cstdint>
#include <cstring>

struct __m256 { float32x4_t lo, hi; };
struct __m256i { int32x4_t lo, hi; };
typedef float32x4_t __m128;
typedef int32x4_t __m128i;

#define _CMP_LT_OQ 0x11
#define _CMP_GT_OQ 0x1E
#define _MM_FROUND_TO_NEAREST_INT 0x00
#define _MM_FROUND_NO_EXC 0x08

#define QV_AI static inline __attribute__((always_inline))
QV_AI __m256 _mm256_loadu_ps(const float* p) { return {vld1q_f32(p), vld1q_f32(p + 4)}; }
QV_AI __m256 _mm256_load_ps(const float* p) { return {vld1q_f32(p), vld1q_f32(p + 4)}; }
QV_AI void _mm256_storeu_ps(float* p, __m256 v) { vst1q_f32(p, v.lo); vst1q_f32(p + 4, v.hi); }
QV_AI void _mm256_store_ps(float* p, __m256 v) { vst1q_f32(p, v.lo); vst1q_f32(p + 4, v.hi); }
QV_AI __m256 _mm256_set1_ps(float x) { return {vdupq_n_f32(x), vdupq_n_f32(x)}; }
QV_AI __m256 _mm256_broadcast_ss(const float* p) { return {vld1q_dup_f32(p), vld1q_dup_f32(p)}; }
QV_AI __m128 _mm_hadd_ps(__m128 a, __m128 b) { return vpaddq_f32(a, b); }
QV_AI __m256 _mm256_setzero_ps() { return {vdupq_n_f32(0.f), vdupq_n_f32(0.f)}; }
QV_AI __m256 _mm256_add_ps(__m256 a, __m256 b) { return {vaddq_f32(a.lo, b.lo), vaddq_f32(a.hi, b.hi)}; }
QV_AI __m256 _mm256_sub_ps(__m256 a, __m256 b) { return {vsubq_f32(a.lo, b.lo), vsubq_f32(a.hi, b.hi)}; }
QV_AI __m256 _mm256_mul_ps(__m256 a, __m256 b) { return {vmulq_f32(a.lo, b.lo), vmulq_f32(a.hi, b.hi)}; }
QV_AI __m256 _mm256_div_ps(__m256 a, __m256 b) { return {vdivq_f32(a.lo, b.lo), vdivq_f32(a.hi, b.hi)}; }
QV_AI __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) { return {vfmaq_f32(c.lo, a.lo, b.lo), vfmaq_f32(c.hi, a.hi, b.hi)}; }
QV_AI __m256 _mm256_fnmadd_ps(__m256 a, __m256 b, __m256 c) { return {vfmsq_f32(c.lo, a.lo, b.lo), vfmsq_f32(c.hi, a.hi, b.hi)}; }
QV_AI __m256 _mm256_min_ps(__m256 a, __m256 b) { return {vminnmq_f32(a.lo, b.lo), vminnmq_f32(a.hi, b.hi)}; }
QV_AI __m256 _mm256_max_ps(__m256 a, __m256 b) { return {vmaxnmq_f32(a.lo, b.lo), vmaxnmq_f32(a.hi, b.hi)}; }

QV_AI __m256 _mm256_castsi256_ps(__m256i a) { return {vreinterpretq_f32_s32(a.lo), vreinterpretq_f32_s32(a.hi)}; }
QV_AI __m256i _mm256_castps_si256(__m256 a) { return {vreinterpretq_s32_f32(a.lo), vreinterpretq_s32_f32(a.hi)}; }
QV_AI __m256i _mm256_set1_epi32(int x) { return {vdupq_n_s32(x), vdupq_n_s32(x)}; }
QV_AI __m256i _mm256_setr_epi32(int a, int b, int c, int d, int e, int f, int g, int h) {
    const int32_t l[4] = {a, b, c, d}, u[4] = {e, f, g, h};
    return {vld1q_s32(l), vld1q_s32(u)};
}
QV_AI __m256i _mm256_add_epi32(__m256i a, __m256i b) { return {vaddq_s32(a.lo, b.lo), vaddq_s32(a.hi, b.hi)}; }
#define _mm256_slli_epi32(a, n) (__m256i{vshlq_n_s32((a).lo, n), vshlq_n_s32((a).hi, n)})
QV_AI __m256i _mm256_cvtps_epi32(__m256 a) { return {vcvtnq_s32_f32(a.lo), vcvtnq_s32_f32(a.hi)}; }
QV_AI void _mm256_store_si256(__m256i* p, __m256i v) { vst1q_s32((int32_t*)p, v.lo); vst1q_s32((int32_t*)p + 4, v.hi); }
QV_AI __m256 _mm256_andnot_ps(__m256 a, __m256 b) {  // ~a & b
    return {vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(b.lo), vreinterpretq_u32_f32(a.lo))),
            vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(b.hi), vreinterpretq_u32_f32(a.hi)))};
}
template <int OP> QV_AI __m256 qv_cmp_ps(__m256 a, __m256 b) {
    if constexpr (OP == _CMP_LT_OQ)
        return {vreinterpretq_f32_u32(vcltq_f32(a.lo, b.lo)), vreinterpretq_f32_u32(vcltq_f32(a.hi, b.hi))};
    else
        return {vreinterpretq_f32_u32(vcgtq_f32(a.lo, b.lo)), vreinterpretq_f32_u32(vcgtq_f32(a.hi, b.hi))};
}
#define _mm256_cmp_ps(a, b, op) qv_cmp_ps<(op)>((a), (b))
QV_AI __m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 m) {  // b where the sign bit of m is set
    const uint32x4_t ml = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(m.lo), 31));
    const uint32x4_t mh = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(m.hi), 31));
    return {vbslq_f32(ml, b.lo, a.lo), vbslq_f32(mh, b.hi, a.hi)};
}
QV_AI __m256 _mm256_round_ps(__m256 a, int) { return {vrndnq_f32(a.lo), vrndnq_f32(a.hi)}; }  // only TO_NEAREST_INT is used

// 128-bit helpers (horizontal-sum idioms)
QV_AI __m128 _mm256_castps256_ps128(__m256 a) { return a.lo; }
#define _mm256_extractf128_ps(a, i) ((i) ? (a).hi : (a).lo)
QV_AI __m128 _mm_add_ps(__m128 a, __m128 b) { return vaddq_f32(a, b); }
QV_AI __m128 _mm_movehl_ps(__m128 a, __m128 b) { return vcombine_f32(vget_high_f32(b), vget_high_f32(a)); }
QV_AI __m128 _mm_add_ss(__m128 a, __m128 b) { return vsetq_lane_f32(vgetq_lane_f32(a, 0) + vgetq_lane_f32(b, 0), a, 0); }
QV_AI float _mm_cvtss_f32(__m128 a) { return vgetq_lane_f32(a, 0); }
QV_AI void _mm_store_ps(float* p, __m128 v) { vst1q_f32(p, v); }
template <int IMM> QV_AI __m128 qv_shuffle_ps(__m128 a, __m128 b) {  // lanes 0,1 from a, lanes 2,3 from b
    float fa[4], fb[4], r[4];
    vst1q_f32(fa, a); vst1q_f32(fb, b);
    r[0] = fa[IMM & 3]; r[1] = fa[(IMM >> 2) & 3]; r[2] = fb[(IMM >> 4) & 3]; r[3] = fb[(IMM >> 6) & 3];
    return vld1q_f32(r);
}
#define _mm_shuffle_ps(a, b, imm) qv_shuffle_ps<(imm)>((a), (b))

// fp16 conversions (8 halves live in a 128-bit integer register)
QV_AI __m128i _mm_loadu_si128(const __m128i* p) { return vld1q_s32((const int32_t*)p); }
QV_AI __m128i _mm_load_si128(const __m128i* p) { return vld1q_s32((const int32_t*)p); }
QV_AI void _mm_storeu_si128(__m128i* p, __m128i v) { vst1q_s32((int32_t*)p, v); }
QV_AI __m256 _mm256_cvtph_ps(__m128i h) {
    const float16x8_t f = vreinterpretq_f16_s32(h);
    return {vcvt_f32_f16(vget_low_f16(f)), vcvt_f32_f16(vget_high_f16(f))};
}
QV_AI __m128 _mm_cvtph_ps(__m128i h) { return vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_s32(h))); }
QV_AI __m128i _mm256_cvtps_ph(__m256 a, int) {
    return vreinterpretq_s32_f16(vcombine_f16(vcvt_f16_f32(a.lo), vcvt_f16_f32(a.hi)));
}
QV_AI float _cvtsh_ss(uint16_t h) { __fp16 f; std::memcpy(&f, &h, 2); return (float)f; }
QV_AI uint16_t _cvtss_sh(float v, int) { __fp16 f = (__fp16)v; uint16_t h; std::memcpy(&h, &f, 2); return h; }
#undef QV_AI
#endif
