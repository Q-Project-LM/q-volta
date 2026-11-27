#include <mma.h>

#include <cstdio>
#include <cstdlib>

#include "../kernels/q8.cuh"
#include "ink_prefill.h"

using namespace nvcuda;

namespace qv::cuda::ink {

// grid (8 kv heads, T tokens), 128 threads
__global__ void __launch_bounds__(128) k_prep_t(PrepTArgs a) {
    __shared__ float red[32], rv0[48];
    const int g = blockIdx.x, t = blockIdx.y, c = threadIdx.x, ch = g * HD + c;
    const int pos = a.pos0 + t;
    float tau = 1.f;
    if (a.glob) tau = 1.f + 0.1f * logf(fmaxf((float)(pos + 1) / 8192.f, 1.f));
    const float* pr = a.proj + (long)t * PS;
    auto rowv = [&](int off, int j /*lag*/, const float* st) -> float {  // value of raw input at token t-j for channel ch
        const int idx = t - j;
        return idx >= 0 ? a.proj[(long)idx * PS + off + ch] : st[(3 + idx) * 1024 + ch];
    };
    const float kx = pr[3072 + ch], vx = pr[4096 + ch];
    const float k1 = rowv(3072, 1, a.kst), k2 = rowv(3072, 2, a.kst), k3 = rowv(3072, 3, a.kst);
    const float v1 = rowv(4096, 1, a.vst), v2 = rowv(4096, 2, a.vst), v3 = rowv(4096, 3, a.vst);
    float ko, vo;
    if (a.glob) {
        const float* dwk = pr + 8600;
        const float* dwv = pr + 8632;
        const float t0 = a.kw[ch] + dwk[g * 4 + 0], t1 = a.kw[1024 + ch] + dwk[g * 4 + 1], t2 = a.kw[2048 + ch] + dwk[g * 4 + 2], t3 = a.kw[3072 + ch] + dwk[g * 4 + 3];
        ko = (t0 * kx + t1 * k1 + t2 * k2 + t3 * k3 + kx) * (1.f / 3.f);
        const float u0 = a.vw[ch] + dwv[g * 4 + 0], u1 = a.vw[1024 + ch] + dwv[g * 4 + 1], u2 = a.vw[2048 + ch] + dwv[g * 4 + 2], u3 = a.vw[3072 + ch] + dwv[g * 4 + 3];
        vo = (u0 * vx + u1 * v1 + u2 * v2 + u3 * v3 + vx) * (1.f / 3.f);
    } else {
        const float4 wk = *reinterpret_cast<const float4*>(a.kw + 4 * ch), wv = *reinterpret_cast<const float4*>(a.vw + 4 * ch);
        ko = (wk.x * k3 + wk.y * k2 + wk.z * k1 + wk.w * kx + kx) * 0.44721359549995794f;
        vo = (wv.x * v3 + wv.y * v2 + wv.z * v1 + wv.w * vx + vx) * 0.44721359549995794f;
    }
    {
        const float nk = block_sum(ko * ko, red);
        const float fk = fminf(1.f, 22.627416997969522f / (sqrtf(nk) + 1e-6f));
        const float nv = block_sum(vo * vo, red);
        const float fv = rsqrtf(nv / HD + 1e-6f);
        const long row = a.glob ? pos : (long)t + 1024;
        if (a.kq) {
            const size_t r = (size_t)g * a.cap + row;
            kv4_store(ko * fk, c, a.kq + r * 64, a.ks + r * 2, red);
            kv4_store(vo * fv, c, a.vq + r * 64, a.vs + r * 2, red);
        } else {
            a.kc[((size_t)g * a.cap + row) * HD + c] = __float2half(ko * fk);
            a.vc[((size_t)g * a.cap + row) * HD + c] = __float2half(vo * fv);
        }
    }
    for (int hh = 0; hh < 3; ++hh) {
        const int h = g * 3 + hh;
        const float qv = pr[h * HD + c] * a.qscale[h * HD + c];
        const float n2 = block_sum(qv * qv, red);
        const float f = fminf(1.f, 22.627416997969522f / (sqrtf(n2) + 1e-6f));
        a.qh[((long)t * NH + h) * HD + c] = __float2half(qv * f * tau * 0.08838834764831845f);
    }
    if (c < 48) rv0[c] = pr[8192 + (g * 3 + c / 16) * 16 + (c & 15)] * a.relscale[(g * 3 + c / 16) * 16 + (c & 15)];
    __syncthreads();
    if (c < 48) {
        float n2 = 0.f;
        for (int k = 0; k < 16; ++k) n2 += rv0[(c / 16) * 16 + k] * rv0[(c / 16) * 16 + k];
        a.relh[((long)t * NH + g * 3) * 16 + c] = __float2half(rv0[c] * fminf(1.f, 8.f / (sqrtf(n2) + 1e-6f)) * tau);
    }
}
void attn_prep_t(const PrepTArgs& a, cudaStream_t s) { k_prep_t<<<dim3(NKV, a.T), 128, 0, s>>>(a); }

// ------------------------------------------------------------------------------------------------------------------
// flash attention (WMMA fp16, two-pass softmax: pass 1 = row max incl. sink, pass 2 = exp / accumulate). grid (ceil(T/64), 24)
constexpr int KLD = 136, SLD = 68, PLD = 72, OLD = 132;
__global__ void __launch_bounds__(128) k_attn_pf(PfAttnArgs a) {
    extern __shared__ __align__(16) unsigned char smraw[];
    __half* Kt = reinterpret_cast<__half*>(smraw);
    float* Sb = reinterpret_cast<float*>(smraw + 64 * KLD * 2);
    __half* Pb = reinterpret_cast<__half*>(Sb + 4 * 16 * SLD);
    const int h = blockIdx.y, g = h / 3, t0 = blockIdx.x * 64, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float* Sw = Sb + warp * 16 * SLD;
    __half* Pw = Pb + warp * 16 * PLD;
    const int r = lane >> 1, hf = lane & 1;
    const int tq = t0 + warp * 16 + r;  // chunk-relative query token of this lane's row
    const bool rvalid = tq < a.T;
    const int iabs = a.pos0 + tq;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf[8];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) wmma::load_matrix_sync(qf[kk], a.qh + (long)(t0 + warp * 16) * 3072 + h * HD + kk * 16, 3072);
    const int imin = a.pos0 + t0, imax = a.pos0 + min(a.T - 1, t0 + 63);
    const int jlo = a.window > 0 ? max(0, imin - a.window + 1) : 0;
    const int jb0 = jlo & ~63;
    const __half* rlrow = a.rl + ((long)(rvalid ? tq : 0) * NH + h) * REXT;
    const float sink = rvalid ? a.proj[(long)tq * PS + 8576 + h] : 0.f;
    float m = sink;
    auto load_tile = [&](const __half* base, int jb, bool isv) {
        for (int i = tid; i < 64 * 16; i += 128) {
            const int rr = i >> 4, cc = (i & 15) * 8;
            const long j = jb + rr, row = j - a.kbase;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j >= 0 && j <= imax && row >= 0) {
                if (a.kq) {
                    const size_t r = (size_t)g * a.cap + row;
                    const uint32_t w = *reinterpret_cast<const uint32_t*>((isv ? a.vq : a.kq) + r * 64 + (cc >> 1));
                    const float2 sc = __half22float2((isv ? a.vs : a.ks)[r * 2 + (cc >> 6)]);
                    __half2* o = reinterpret_cast<__half2*>(&v);
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        o[e] = __floats2half2_rn(fmaf((float)((w >> (8 * e)) & 15u), sc.x, sc.y), fmaf((float)((w >> (8 * e + 4)) & 15u), sc.x, sc.y));
                } else {
                    v = *reinterpret_cast<const uint4*>(base + ((long)g * a.cap + row) * HD + cc);
                }
            }
            *reinterpret_cast<uint4*>(Kt + rr * KLD + cc) = v;
        }
    };
    auto qk = [&](int jb) {  // S tile of this warp into Sw
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, Kt + nt * 16 * KLD + kk * 16, KLD);
                wmma::mma_sync(acc, qf[kk], kf, acc);
            }
            wmma::store_matrix_sync(Sw + nt * 16, acc, SLD, wmma::mem_row_major);
        }
        __syncwarp();
    };
    auto score = [&](int jb, int col) -> float {
        const int j = jb + col;
        const int d = iabs - j;
        const bool ok = rvalid && d >= 0 && (a.window == 0 || d < a.window);
        if (!ok) return -INFINITY;
        return Sw[r * SLD + col] + (d < REXT ? __half2float(rlrow[d]) : 0.f);
    };
    // pass 1
    for (int jb = jb0; jb <= imax; jb += 64) {
        load_tile(a.kc, jb, false);
        __syncthreads();
        qk(jb);
        for (int c = 0; c < 32; ++c) m = fmaxf(m, score(jb, hf * 32 + c));
        __syncwarp();
        __syncthreads();
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    // pass 2
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) wmma::fill_fragment(of[nt], 0.f);
    float l = 0.f;
    for (int jb = jb0; jb <= imax; jb += 64) {
        load_tile(a.kc, jb, false);
        __syncthreads();
        qk(jb);
        for (int c = 0; c < 32; ++c) {
            const float s = score(jb, hf * 32 + c);
            const float p = __expf(s - m);  // exp(-inf) = 0
            l += p;
            Pw[r * PLD + hf * 32 + c] = __float2half(p);
        }
        __syncwarp();
        __syncthreads();
        load_tile(a.vc, jb, true);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
            wmma::load_matrix_sync(pf, Pw + kk * 16, PLD);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
                wmma::load_matrix_sync(vf, Kt + kk * 16 * KLD + nt * 16, KLD);
                wmma::mma_sync(of[nt], pf, vf, of[nt]);
            }
        }
        __syncthreads();
    }
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __expf(sink - m);
    float* Ow = reinterpret_cast<float*>(smraw) + warp * 16 * OLD;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) wmma::store_matrix_sync(Ow + nt * 16, of[nt], OLD, wmma::mem_row_major);
    __syncwarp();
    if (rvalid) {
        const float inv = 1.f / l;
        const float* gt = a.proj + (long)tq * PS + 5120 + h * HD;
        __half* o = a.out + (long)tq * 3072 + h * HD;
        for (int d = hf * 64; d < hf * 64 + 64; ++d) o[d] = __float2half(Ow[r * OLD + d] * inv * (1.f / (1.f + __expf(-gt[d]))));
    }
}
// Variant with the 3 query heads of one kv head in one block (12 warps): the K / V tile is loaded and int4-dequantised once for the 3 heads
// instead of three times. Every warp runs exactly the arithmetic of k_attn_pf for its (head, 16 rows): bit-identical output.
__global__ void __launch_bounds__(384, 1) k_attn_pf3(PfAttnArgs a) {
    extern __shared__ __align__(16) unsigned char smraw[];
    __half* Kt = reinterpret_cast<__half*>(smraw);
    float* Sb = reinterpret_cast<float*>(smraw + 64 * KLD * 2);
    __half* Pb = reinterpret_cast<__half*>(Sb + 12 * 16 * SLD);
    const int g = blockIdx.y, t0 = blockIdx.x * 64, tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int h = g * 3 + (wid >> 2), warp = wid & 3;  // warp: query sub-tile of this head, same meaning as in k_attn_pf
    const int wsm = wid;                               // shared-memory slot of this warp
    float* Sw = Sb + wsm * 16 * SLD;
    __half* Pw = Pb + wsm * 16 * PLD;
    const int r = lane >> 1, hf = lane & 1;
    const int tq = t0 + warp * 16 + r;  // chunk-relative query token of this lane's row
    const bool rvalid = tq < a.T;
    const int iabs = a.pos0 + tq;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf[8];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) wmma::load_matrix_sync(qf[kk], a.qh + (long)(t0 + warp * 16) * 3072 + h * HD + kk * 16, 3072);
    const int imin = a.pos0 + t0, imax = a.pos0 + min(a.T - 1, t0 + 63);
    const int jlo = a.window > 0 ? max(0, imin - a.window + 1) : 0;
    const int jb0 = jlo & ~63;
    const __half* rlrow = a.rl + ((long)(rvalid ? tq : 0) * NH + h) * REXT;
    const float sink = rvalid ? a.proj[(long)tq * PS + 8576 + h] : 0.f;
    float m = sink;
    auto load_tile = [&](const __half* base, int jb, bool isv) {
        for (int i = tid; i < 64 * 16; i += 384) {
            const int rr = i >> 4, cc = (i & 15) * 8;
            const long j = jb + rr, row = j - a.kbase;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j >= 0 && j <= imax && row >= 0) {
                if (a.kq) {
                    const size_t r = (size_t)g * a.cap + row;
                    const uint32_t w = *reinterpret_cast<const uint32_t*>((isv ? a.vq : a.kq) + r * 64 + (cc >> 1));
                    const float2 sc = __half22float2((isv ? a.vs : a.ks)[r * 2 + (cc >> 6)]);
                    __half2* o = reinterpret_cast<__half2*>(&v);
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        o[e] = __floats2half2_rn(fmaf((float)((w >> (8 * e)) & 15u), sc.x, sc.y), fmaf((float)((w >> (8 * e + 4)) & 15u), sc.x, sc.y));
                } else {
                    v = *reinterpret_cast<const uint4*>(base + ((long)g * a.cap + row) * HD + cc);
                }
            }
            *reinterpret_cast<uint4*>(Kt + rr * KLD + cc) = v;
        }
    };
    auto qk = [&](int jb) {  // S tile of this warp into Sw
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, Kt + nt * 16 * KLD + kk * 16, KLD);
                wmma::mma_sync(acc, qf[kk], kf, acc);
            }
            wmma::store_matrix_sync(Sw + nt * 16, acc, SLD, wmma::mem_row_major);
        }
        __syncwarp();
    };
    auto score = [&](int jb, int col) -> float {
        const int j = jb + col;
        const int d = iabs - j;
        const bool ok = rvalid && d >= 0 && (a.window == 0 || d < a.window);
        if (!ok) return -INFINITY;
        return Sw[r * SLD + col] + (d < REXT ? __half2float(rlrow[d]) : 0.f);
    };
    // pass 1
    for (int jb = jb0; jb <= imax; jb += 64) {
        load_tile(a.kc, jb, false);
        __syncthreads();
        qk(jb);
        for (int c = 0; c < 32; ++c) m = fmaxf(m, score(jb, hf * 32 + c));
        __syncwarp();
        __syncthreads();
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    // pass 2
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) wmma::fill_fragment(of[nt], 0.f);
    float l = 0.f;
    for (int jb = jb0; jb <= imax; jb += 64) {
        load_tile(a.kc, jb, false);
        __syncthreads();
        qk(jb);
        for (int c = 0; c < 32; ++c) {
            const float s = score(jb, hf * 32 + c);
            const float p = __expf(s - m);  // exp(-inf) = 0
            l += p;
            Pw[r * PLD + hf * 32 + c] = __float2half(p);
        }
        __syncwarp();
        __syncthreads();
        load_tile(a.vc, jb, true);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
            wmma::load_matrix_sync(pf, Pw + kk * 16, PLD);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
                wmma::load_matrix_sync(vf, Kt + kk * 16 * KLD + nt * 16, KLD);
                wmma::mma_sync(of[nt], pf, vf, of[nt]);
            }
        }
        __syncthreads();
    }
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __expf(sink - m);
    // epilogue in two halves (12 warps x 16 x 132 floats would not fit): lanes hf = 0 write dims 0..63 from the first half, hf = 1 dims 64..127 from the second
    constexpr int OLD2 = 68;
    float* Ow = reinterpret_cast<float*>(smraw) + wsm * 16 * OLD2;
    const float* gt = a.proj + (long)tq * PS + 5120 + h * HD;
    __half* o = a.out + (long)tq * 3072 + h * HD;
    const float inv = 1.f / l;
#pragma unroll
    for (int half_ = 0; half_ < 2; ++half_) {
        __syncthreads();  // everybody is done with the K/V tile region before it is reused
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) wmma::store_matrix_sync(Ow + nt * 16, of[half_ * 4 + nt], OLD2, wmma::mem_row_major);
        __syncwarp();
        if (rvalid && hf == half_)
            for (int d = 0; d < 64; ++d) o[half_ * 64 + d] = __float2half(Ow[r * OLD2 + d] * inv * (1.f / (1.f + __expf(-gt[half_ * 64 + d]))));
        __syncwarp();
    }
}

void attn_prefill(const PfAttnArgs& a, cudaStream_t s) {
    static const bool v3 = std::getenv("QV_PF1") == nullptr;  // QV_PF1=1: one head per block (reference kernel, same output bits)
    if (v3) {
        const size_t smem3 = 64 * KLD * 2 + 12 * 16 * SLD * 4 + 12 * 16 * PLD * 2;
        static bool init = (cudaFuncSetAttribute(k_attn_pf3, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3), true);
        (void)init;
        k_attn_pf3<<<dim3((a.T + 63) / 64, NKV), 384, smem3, s>>>(a);
        return;
    }
    const size_t smem = 64 * KLD * 2 + 4 * 16 * SLD * 4 + 4 * 16 * PLD * 2;
    k_attn_pf<<<dim3((a.T + 63) / 64, NH), 128, smem, s>>>(a);
}

// ------------------------------------------------------------------------------------------------------------------
__global__ void k_ring_to_stage(const __half* ring, __half* stage, int rows, int pos0) {
    const int g = blockIdx.x;
    for (int i = threadIdx.x; i < 1024 * 16; i += blockDim.x) {
        const int rr = i >> 4, cc = (i & 15) * 8;  // rr = stage row 0..1023 <-> position pos0 - 1024 + rr
        const int j = pos0 - 1024 + rr;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (j >= 0) v = *reinterpret_cast<const uint4*>(ring + ((long)g * 1024 + (j & 1023)) * HD + cc);
        *reinterpret_cast<uint4*>(stage + ((long)g * rows + rr) * HD + cc) = v;
    }
}
void ring_to_stage(const __half* ring, __half* stage, int rows, int pos0, cudaStream_t s) { k_ring_to_stage<<<NKV, 256, 0, s>>>(ring, stage, rows, pos0); }
__global__ void k_stage_to_ring(const __half* stage, __half* ring, int rows, int pos0, int T) {
    const int g = blockIdx.x;
    const int jlo = max(pos0, pos0 + T - 1024);
    for (int i = threadIdx.x; i < 1024 * 16; i += blockDim.x) {
        const int rr = i >> 4, cc = (i & 15) * 8;
        const int j = jlo + rr;
        if (j >= pos0 + T) continue;
        *reinterpret_cast<uint4*>(ring + ((long)g * 1024 + (j & 1023)) * HD + cc) =
            *reinterpret_cast<const uint4*>(stage + ((long)g * rows + (j - pos0 + 1024)) * HD + cc);
    }
}
void stage_to_ring(const __half* stage, __half* ring, int rows, int pos0, int T, cudaStream_t s) { k_stage_to_ring<<<NKV, 256, 0, s>>>(stage, ring, rows, pos0, T); }

// ------------------------------------------------------------------------------------------------------------------
// MoE
__global__ void k_topk_t(const float* logits, const float* bias, int* ids, float* wts, int T, float route_scale) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    float sv[4], sb[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float l = logits[(long)t * NE + lane + 32 * k];
        sv[k] = 1.f / (1.f + __expf(-l));
        sb[k] = sv[k] + bias[lane + 32 * k];
    }
    float wsel[TOPK], wsum = 0.f;
    int chosen_i = 0;
    for (int r = 0; r < TOPK; ++r) {
        float best = -1e30f;
        int bi = 1 << 30;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (sb[k] > best) { best = sb[k]; bi = lane + 32 * k; }
        float ob = best;
        int oi = bi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float vb = __shfl_xor_sync(0xffffffffu, ob, o);
            const int vi = __shfl_xor_sync(0xffffffffu, oi, o);
            if (vb > ob || (vb == ob && vi < oi)) { ob = vb; oi = vi; }
        }
        float w = 0.f;
        if (lane == (oi & 31)) {
#pragma unroll
            for (int k = 0; k < 4; ++k)
                if (k == (oi >> 5)) { w = sv[k]; sb[k] = -1e30f; }
        }
        w = __shfl_sync(0xffffffffu, w, oi & 31);
        wsel[r] = w;
        wsum += w;
        if (lane == r) chosen_i = oi;
    }
    if (lane < TOPK) {
        float w = 0.f;
#pragma unroll
        for (int r = 0; r < TOPK; ++r)
            if (r == lane) w = wsel[r];
        ids[(long)t * TOPK + lane] = chosen_i;
        wts[(long)t * TOPK + lane] = w / (wsum + 1e-20f) * route_scale;
    }
}
void topk_t(const float* logits, const float* bias, int* ids, float* wts, int T, float rs, cudaStream_t s) {
    k_topk_t<<<(T + 7) / 8, 256, 0, s>>>(logits, bias, ids, wts, T, rs);
}
__global__ void k_count(const int* ids, int n, int* cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&cnt[ids[i]], 1);
}
__global__ void k_scan(const int* cnt, int* offs, int* fill) {
    if (threadIdx.x == 0) {
        int s = 0;
        for (int e = 0; e < NE; ++e) { offs[e] = s; s += cnt[e]; fill[e] = 0; }
        offs[NE] = s;
    }
}
__global__ void k_scatter(const int* ids, int n, const int* offs, int* fill, int* perm, int* inv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = ids[i];
    const int p = offs[e] + atomicAdd(&fill[e], 1);
    perm[p] = i;
    inv[i] = p;
}
void moe_sort(const int* ids, int T, int* cnt, int* offs, int* fill, int* perm, int* inv, cudaStream_t s) {
    const int n = T * TOPK;
    cudaMemsetAsync(cnt, 0, NE * 4, s);
    k_count<<<(n + 255) / 256, 256, 0, s>>>(ids, n, cnt);
    k_scan<<<1, 32, 0, s>>>(cnt, offs, fill);
    k_scatter<<<(n + 255) / 256, 256, 0, s>>>(ids, n, offs, fill, perm, inv);
}
__global__ void k_gather(const __half* x, const int* perm, int n, __half* out) {
    const int p = blockIdx.x;
    if (p >= n) return;
    const uint4* src = reinterpret_cast<const uint4*>(x + (long)(perm[p] / TOPK) * H);
    uint4* dst = reinterpret_cast<uint4*>(out + (long)p * H);
    for (int i = threadIdx.x; i < H / 8; i += blockDim.x) dst[i] = src[i];
}
void moe_gather(const __half* x, const int* perm, int n, __half* out, cudaStream_t s) { k_gather<<<n, 288, 0, s>>>(x, perm, n, out); }
__global__ void k_act(const __half* y13, int n, float gcap, float ucap, __half* out) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n * FF) return;
    const long p = i / FF, c = i % FF;
    const float2 gu = __half22float2(*reinterpret_cast<const __half2*>(y13 + p * (2 * FF) + 2 * c));
    const float gc = gcap * tanhf(gu.x / gcap), uc = ucap * tanhf(gu.y / ucap);
    out[i] = __float2half(gc * (1.f / (1.f + __expf(-gu.x))) * uc);
}
void moe_act(const __half* y13, int n, float gcap, float ucap, __half* out, cudaStream_t s) {
    k_act<<<(unsigned)(((long)n * FF + 255) / 256), 256, 0, s>>>(y13, n, gcap, ucap, out);
}
__global__ void k_combine(const __half* y2, const int* inv, const float* wts, int T, float* out) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        float o = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) o = fmaf(wts[(long)t * TOPK + k], __half2float(y2[(long)inv[(long)t * TOPK + k] * H + c]), o);
        out[(long)t * H + c] = o;
    }
}
void moe_combine(const __half* y2, const int* inv, const float* wts, int T, float* out, cudaStream_t s) { k_combine<<<T, 256, 0, s>>>(y2, inv, wts, T, out); }

// ------------------------------------------------------------------------------------------------------------------
__global__ void k_gate_cat(const float* h, const float* msg, __half* xcat) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    float s1 = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) { s1 += h[(long)t * H + i] * h[(long)t * H + i]; s2 += msg[(long)t * H + i] * msg[(long)t * H + i]; }
    const float r1 = rsqrtf(block_sum(s1, red) / H + 1e-6f);
    const float r2 = rsqrtf(block_sum(s2, red) / H + 1e-6f);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        xcat[(long)t * 2 * H + i] = __float2half(h[(long)t * H + i] * r1);
        xcat[(long)t * 2 * H + H + i] = __float2half(msg[(long)t * H + i] * r2);
    }
}
void gate_cat(const float* h, const float* msg, int T, __half* xcat, cudaStream_t s) { k_gate_cat<<<T, 256, 0, s>>>(h, msg, xcat); }
__global__ void k_f2h(const float* x, long n, __half* y) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
__global__ void k_h2f(const __half* x, long n, float* y) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}
void f16_to_f32(const __half* x, long n, float* y, cudaStream_t s) { k_h2f<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, n, y); }
void f32_to_f16(const float* x, long n, __half* y, cudaStream_t s) { k_f2h<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, n, y); }
__global__ void k_gate_apply_t(const float* f, const float* g, const float* h, const float* msg, long n, float* out) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (1.f - tanhf(f[i])) * h[i] + tanhf(g[i]) * msg[i];
}
void gate_apply_t(const float* f, const float* g, const float* h, const float* msg, long n, float* out, cudaStream_t s) {
    k_gate_apply_t<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(f, g, h, msg, n, out);
}
__global__ void k_copy(const float* a, float* b, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}
void copy_f32(const float* src, float* dst, long n, cudaStream_t s) { k_copy<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, dst, n); }

}  // namespace qv::cuda::ink
