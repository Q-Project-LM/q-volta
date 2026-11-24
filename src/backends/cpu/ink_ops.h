// Inkling-2-Mini CPU ops (AVX2 + FMA + F16C): small kernels used by src/runtime/cpu_inkling_engine.cpp.
// Same contract as ops.h: single-threaded, range based, no allocation, no shared state.
//
// NEON port map: _mm256_fmadd_ps -> vfmaq_f32 (2 x 128-bit per 8 lanes), _mm256_cvtph_ps -> vcvt_f32_f16, exp256 -> a NEON
// polynomial of the same shape (vrndnq_f32 / vfmaq_f32 / vshlq_n_s32 for the exponent), shuffles in expert_act_interleaved
// -> vuzp1q_f32 / vuzp2q_f32 (de-interleave), fp16 KV stores -> vcvt_f16_f32.
#pragma once
#include <cstddef>
#include <cstdint>

namespace qv::cpu {

// ---- vector math ----------------------------------------------------------------------------------------------------
float dot_f32(const float* a, const float* b, int n);                 // n % 8 == 0 not required
void exp_vec(const float* x, float* y, int n);                        // y = exp(x), max rel err ~2e-7

// ---- RMS norms (fp32, eps added inside the sqrt as in the reference) -----------------------------------------------------
void rms_w(const float* x, const float* w, float* out, int n, float eps);   // out = w * x * rsqrt(mean(x^2) + eps)
void rms_u(const float* x, float* out, int n, float eps);                   // out = x * rsqrt(mean(x^2) + eps)
float rms_inv(const float* x, int n, float eps);                            // rsqrt(mean(x^2) + eps)

// ---- fp16 matrices (router, relative-bias input, sink, message controllers, dynamic-conv taps) ------------------------------
// Y[t*ldy + n] = sum_k W[n*K + k] * X[t*ldx + k] for rows n in [n0, n1), tokens t < T. W row-major fp16, fp32 accumulate.
void gemm_f16_rows(const uint16_t* W, int K, const float* X, size_t ldx, int T, float* Y, size_t ldy, int n0, int n1);

// ---- block attention residual -------------------------------------------------------------------------------------------
// States = nb block vectors (blk[b], each H floats) + `partial`; S = rms_u(state) (no gamma); a_q = softmax_b((S_b . query_q) * H^-0.5);
// out[q] = sum_b a_q[b] * S_b. queries [Q][H] fp32, out [Q][H]. Q <= 4, nb <= 4.
void block_attn_res(const float* const* blk, int nb, const float* partial, const float* queries, int Q, int H, float* out);

// ---- MoE ----------------------------------------------------------------------------------------------------------------
// router: logits[128] -> top-k of sigmoid(l) + bias (ties: lower index), weights = sigmoid(l)[idx] / (sum + 1e-20) * route_scale.
void router_topk(const float* logits, const float* bias, int n_experts, int k, float route_scale, int* idx, float* w);
// gu = [g0 u0 g1 u1 ...] (n_pairs pairs, interleaved) -> act[i] = softcap(g,gcap) * sigmoid(g) * softcap(u,ucap)
void expert_act_interleaved(const float* gu, float* act, int n_pairs, float gcap, float ucap);

// ---- relative-position attention (one kv head, 3 q heads, one query token) --------------------------------------------------
// Online-softmax partial over keys j in [j0, j1) (absolute positions, all <= p):
//   s[h][j] = (q[h].k_j) * scale + (p - j < rel_extent ? sum_i rel[h][i] * proj[i][p - j] : 0)
// k/v caches are fp16 rows of 128 at index (ring ? j % ring : j). acc [3][128] and ml [3][2] = (running max, running sum) are
// OVERWRITTEN (a fresh partial); the caller merges partials (and the sink logit) afterwards.
struct InkAttnArgs {
    const float* q;      // [3][128], clipped / tau-scaled
    const float* rel;    // [3][16], clipped / tau-scaled
    const uint16_t* kc;  // cache base of this kv head
    const uint16_t* vc;
    const float* proj;   // [16][rel_extent] fp32 (rel_logits_proj.proj)
    int p, j0, j1, ring, rel_extent;
    float scale;
    const int8_t *kq = nullptr, *vq = nullptr;  // int8 KV (global layers): rows of 128 + one fp32 scale per row; kc/vc unused when set
    const float *ks = nullptr, *vs = nullptr;
};
void ink_attn_partial(const InkAttnArgs& a, float* acc, float* ml);

// Merge the online-softmax partials of one q head over `ns` key splits (part + s*stride: acc[3][128] then ml[3][2]; head i in the kv
// group) together with the sink logit (value 0): out[d] = sum_s acc_s[d] e^{m_s - M} / (sum_s l_s e^{m_s - M} + e^{sink - M}) * sigmoid(gate[d]).
void ink_attn_merge(const float* part, int ns, size_t stride, int i, float sink, const float* gate, float* out);

// x = cap * tanh(x / cap), in place
void softcap_inplace(float* x, int n, float cap);

// fp32 vector -> fp16 row (round to nearest even), 128-element rows
void store_f16(const float* x, uint16_t* out, int n);

// fp32 128-vector -> int8 row + scale (symmetric, absmax/127)
void store_q8(const float* x, int8_t* out, float* scale);

}  // namespace qv::cpu
