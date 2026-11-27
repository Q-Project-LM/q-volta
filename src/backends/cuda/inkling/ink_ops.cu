// Inkling-2-Mini decode kernels. Spec: docs/INKLING_SPEC.md; engine: src/runtime/cuda_inkling_engine.cu.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../kernels/elementwise.h"
#include "../kernels/q8.cuh"
#include "ink_ops.h"

namespace qv::cuda::ink {

namespace {
#ifndef MR_WPB
#define MR_WPB 1
#endif
#ifndef MR_R
#define MR_R 1
#endif
#ifndef GD_RPB
#define GD_RPB 4
#endif
#ifndef MG_WPB
#define MG_WPB 1
#endif
#define QV_STR_(x) #x
#define UNROLL_N(n) _Pragma(QV_STR_(unroll n))
#ifndef MB_MG
#define MB_MG 1
#endif
#ifndef MB_W13
#define MB_W13 1
#endif
#ifndef MB_W2
#define MB_W2 1
#endif
#ifndef UNG
#define UNG 6
#endif
#ifndef UNP
#define UNP 9
#endif
#ifndef UNQ
#define UNQ 1
#endif
#ifndef UNF
#define UNF 3
#endif
#ifndef UNW
#define UNW 1
#endif
#ifndef UNR
#define UNR 9
#endif
constexpr int NT = 768;      // threads of the hidden-size kernels: 3 elements per thread, element = t + 768 j
constexpr int NW = NT / 32;  // 24 warps = 24 chunks per j

template <int BITS>
__device__ __forceinline__ uint32_t qword(const uint32_t* w, int b) {
    if constexpr (BITS == 2) {
        return (w[b & 1] >> (2 * (b >> 1))) & 0x03030303u;
    } else if constexpr (BITS == 3) {
        return ((w[b & 1] >> (2 * (b >> 1))) & 0x03030303u) | (((w[2] >> b) & 0x01010101u) << 2);
    } else if constexpr (BITS == 4) {
        return (w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu;
    } else if constexpr (BITS == 5) {
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) | (((w[4] >> b) & 0x01010101u) << 4);
    } else if constexpr (BITS == 6) {
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) | (((w[4 + (b & 1)] >> (2 * (b >> 1))) & 0x03030303u) << 4);
    } else {
        static_assert(BITS == 8, "unsupported bits");
        return ((w[b & 3] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) | (((w[4 + (b & 3)] >> (4 * (b >> 2))) & 0x0F0F0F0Fu) << 4);
    }
}
__device__ __forceinline__ int dp4a_us(uint32_t a, int b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// one quantized chunk (32 weights of one row) against one int8 activation chunk
template <int BITS>
__device__ __forceinline__ float qchunk(const uint32_t* wp, const int (&xw)[8], float dx, float xs, int qs, float sc, float mn, float acc) {
    constexpr int ZP = 1 << (BITS - 1);
    uint32_t w[BITS];
#pragma unroll
    for (int b = 0; b < BITS; ++b) w[b] = __ldg(wp + b);
    int d = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) d = dp4a_us(qword<BITS>(w, b), xw[b], d);
    d -= ZP * qs;
    return fmaf(sc * dx, (float)d, fmaf(fmaf(sc, (float)ZP, mn), xs, acc));
}
__device__ __forceinline__ void load_x(const Q8Vec& xv, size_t c, int (&xw)[8], float& dx, float& xs, int& qs) {
    const int4* xp = reinterpret_cast<const int4*>(xv.q + c * 8);
    const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
    xw[0] = x0.x; xw[1] = x0.y; xw[2] = x0.z; xw[3] = x0.w; xw[4] = x1.x; xw[5] = x1.y; xw[6] = x1.z; xw[7] = x1.w;
    dx = __ldg(xv.d + c);
    xs = __ldg(xv.sum + c);
    qs = __ldg(xv.qsum + c);
}
// meta8 (scale, min) of group g of a row
__device__ __forceinline__ void sm8_decode(unsigned v, float rsc, float zlo, float zstep, float& sc, float& mn) {
    sc = (float)(v & 255u) * rsc;
    const float z = (float)(v >> 8) * zstep + zlo;
    mn = -(sc * z);
}

// reduce N (<= 32) values over the block of NT threads (all threads get the sums). sm >= NW*N + N floats, written once per call: successive calls in
// one kernel must use disjoint regions (no barrier protects reuse). Per value: warp xor tree, then the NW warp partials added in warp order 0..23
// by thread i; two barriers (no entry barrier: fresh smem).
template <int N>
__device__ __forceinline__ void bred(float (&v)[N], float* sm) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) sm[w * N + i] = v[i];
    }
    __syncthreads();
    if (threadIdx.x < N) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < NW; ++k) s += sm[k * N + threadIdx.x];
        sm[NW * N + threadIdx.x] = s;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = sm[NW * N + i];
}

__device__ __forceinline__ Q8Buf q8_stream(const Q8Buf& q, int s) {
    Q8Buf o;
    o.q = q.q + s * (H / 4);
    o.d = q.d + s * HC;
    o.sum = q.sum + s * HC;
    o.qsum = q.qsum + s * HC;
    return o;
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------------
// embedding (4-bit g64 meta8 row dequant) + embed_norm
template <int BITS>
__device__ __forceinline__ float dequant_elem(const QMat& E, size_t row, int c, int lane, size_t C) {
    const uint32_t* wp = E.q + (row * C + c) * BITS;
    uint32_t w[BITS];
#pragma unroll
    for (int b = 0; b < BITS; ++b) w[b] = wp[b];
    const int b = lane & 7, j = lane >> 3;
    const uint32_t code = (qword<BITS>(w, b) >> (8 * j)) & 0xFFu;
    const size_t G = E.K >> E.gshift;
    float sc, mn;
    sm8_decode(E.sm8[row * G + (c >> (E.gshift - 5))], __half2float(E.rs[row]) / 255.0f, E.zlo, E.zstep, sc, mn);
    return fmaf(sc, (float)code, mn);
}

__global__ void __launch_bounds__(NT) k_embed(QMat E, const int* tok, const float* w, float* h, float* blk0) {
    __shared__ float sm[NW];
    const size_t row = (size_t)tok[blockIdx.x];
    h += (size_t)blockIdx.x * H;
    blk0 += (size_t)blockIdx.x * H;
    const size_t C = E.K >> 5;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float v[3];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int c = warp + NW * j;
        switch (E.bits) {
            case 3: v[j] = dequant_elem<3>(E, row, c, lane, C); break;
            case 4: v[j] = dequant_elem<4>(E, row, c, lane, C); break;
            case 5: v[j] = dequant_elem<5>(E, row, c, lane, C); break;
            case 6: v[j] = dequant_elem<6>(E, row, c, lane, C); break;
            default: v[j] = dequant_elem<8>(E, row, c, lane, C); break;
        }
        ss += v[j] * v[j];
    }
    float r = block_sum(ss, sm);
    const float f = rsqrtf(r / H + 1e-6f);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = threadIdx.x + NT * j;
        const float o = w[i] * (v[j] * f);
        h[i] = o;
        blk0[i] = o;
    }
}
void embed_norm(const QMat& E, const int* tok, const float* w, float* h, float* blk0, cudaStream_t s, int T) {
    k_embed<<<T, NT, 0, s>>>(E, tok, w, h, blk0);
}

__global__ void k_softcap(float* x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
void softcap_inplace(float* x, int n, float cap, cudaStream_t s) { k_softcap<<<(n + 255) / 256, 256, 0, s>>>(x, n, cap); }

// ------------------------------------------------------------------------------------------------------------------
// attention residual
template <int QM>
__global__ void __launch_bounds__(NT) k_ar(ArArgs a) {
    __shared__ float sm[NW * 30 + 128];
    const int t = threadIdx.x;
    const float* queries = a.queries + (long)blockIdx.x * a.qbs;
    const long tok = a.rowy ? (long)blockIdx.y : a.allpar ? (long)blockIdx.x : (long)a.tok0, NTK = a.ntok;
    float v[5][3];
#pragma unroll
    for (int b = 0; b < 5; ++b)
#pragma unroll
        for (int j = 0; j < 3; ++j) v[b][j] = 0.f;
    // all global loads of the kernel that do not depend on earlier results are issued first (the stores below may alias them, which would make the
    // compiler serialize load -> store -> load chains: one memory round trip per j): block states, queries, norm weights, and the mode inputs.
    float qpre[QM][3], npre[QM][3];
#pragma unroll
    for (int q = 0; q < QM; ++q)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            qpre[q][j] = (a.mix && q < a.Q) ? queries[(long)q * H + t + NT * j] : 0.f;
            npre[q][j] = (a.epi != 0 && q < a.Q) ? a.nw[q][t + NT * j] : 0.f;
        }
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int j = 0; j < 3; ++j) v[b][j] = (b < a.B) ? a.blocks[((long)b * NTK + tok) * H + t + NT * j] : 0.f;
    int NB;
    if (a.mode == 0) {
        NB = a.B + 1;
        float hvv[3];
        if (a.fin_y) {  // fused k_finish (one token): identical expressions
            float x[3], c0[3], c1[3], c2[3], fr[3];
            float4 w[3];
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int c = t + NT * j;
                x[j] = a.fin_y[c]; c0[j] = a.fin_cst[c]; c1[j] = a.fin_cst[H + c]; c2[j] = a.fin_cst[2 * H + c];
                w[j] = *reinterpret_cast<const float4*>(a.fin_cw + 4 * c);
                fr[j] = a.fin_res[c];
            }
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int c = t + NT * j;
                const float yy = w[j].x * c0[j] + w[j].y * c1[j] + w[j].z * c2[j] + w[j].w * x[j];
                a.fin_cst[c] = c1[j];
                a.fin_cst[H + c] = c2[j];
                a.fin_cst[2 * H + c] = x[j];
                hvv[j] = fr[j] + (yy + x[j]) * 0.44721359549995794f;
                a.fin_h_out[c] = hvv[j];
                if (a.fin_copy) a.fin_copy[c] = hvv[j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < 3; ++j) hvv[j] = a.h[tok * H + t + NT * j];
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float hv = hvv[j];
            for (int b = 0; b < 5; ++b)
                if (b == a.B) v[b][j] = hv;
        }
    } else {
        float res[3], hin[3];
        float xr[3], q0[3], q1[3], q2[3];
        float4 w[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int c = t + NT * j;
            xr[j] = a.araw[tok * H + c];
            {
                const long i3 = tok - 3, i2 = tok - 2, i1 = tok - 1;
                q0[j] = i3 >= 0 ? a.araw[i3 * H + c] : a.cst[(3 + i3) * H + c];
                q1[j] = i2 >= 0 ? a.araw[i2 * H + c] : a.cst[(3 + i2) * H + c];
                q2[j] = i1 >= 0 ? a.araw[i1 * H + c] : a.cst[(3 + i1) * H + c];
            }
            w[j] = *reinterpret_cast<const float4*>(a.cw + 4 * c);
            hin[j] = a.h[tok * H + c];
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int c = t + NT * j;
            const float x = xr[j], s0 = q0[j], s1 = q1[j], s2 = q2[j];
            const float y = w[j].x * s0 + w[j].y * s1 + w[j].z * s2 + w[j].w * x;
            const float av = (y + x) * 0.44721359549995794f;  // 1/sqrt(5)
            if (NTK == 1) {
                a.cst[c] = s1;
                a.cst[H + c] = s2;
                a.cst[2 * H + c] = x;
            }
            res[j] = a.boundary ? av : av + hin[j];
            a.res_out[tok * H + c] = res[j];
        }
        NB = a.B + 1 + (a.boundary ? 1 : 0);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            for (int b = 0; b < 5; ++b) {
                if (a.boundary) {
                    if (b == a.B) v[b][j] = hin[j];
                    if (b == a.B + 1) v[b][j] = res[j];
                } else if (b == a.B) {
                    v[b][j] = res[j];
                }
            }
            if (a.boundary) a.blocks[((long)a.B * NTK + tok) * H + t + NT * j] = hin[j];
        }
    }
    float out[QM][3];
    if (!a.mix) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float pv = 0.f;
            for (int b = 0; b < 5; ++b)
                if (b == NB - 1) pv = v[b][j];
            out[0][j] = pv;
        }
    } else {
        float red[5 + 5 * QM];
#pragma unroll
        for (int b = 0; b < 5; ++b) red[b] = (b < NB) ? v[b][0] * v[b][0] + v[b][1] * v[b][1] + v[b][2] * v[b][2] : 0.f;
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            const float* qv = qpre[q];
#pragma unroll
            for (int b = 0; b < 5; ++b) red[5 + q * 5 + b] = (b < NB && q < a.Q) ? v[b][0] * qv[0] + v[b][1] * qv[1] + v[b][2] * qv[2] : 0.f;
        }
        bred<5 + 5 * QM>(red, sm);
        float rb[5];
#pragma unroll
        for (int b = 0; b < 5; ++b) rb[b] = rsqrtf(red[b] / H + 1e-6f);
        float coef[QM][5];
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            float mx = -1e30f;
#pragma unroll
            for (int b = 0; b < 5; ++b) {
                coef[q][b] = red[5 + q * 5 + b] * rb[b] * (1.0f / 48.0f);  // H^-0.5, H = 2304
                if (b < NB) mx = fmaxf(mx, coef[q][b]);
            }
            float sum = 0.f;
#pragma unroll
            for (int b = 0; b < 5; ++b) {
                coef[q][b] = (b < NB) ? __expf(coef[q][b] - mx) : 0.f;
                sum += coef[q][b];
            }
#pragma unroll
            for (int b = 0; b < 5; ++b) coef[q][b] = coef[q][b] / sum * rb[b];
        }
#pragma unroll
        for (int q = 0; q < QM; ++q)
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float o = 0.f;
#pragma unroll
                for (int b = 0; b < 5; ++b) o = fmaf(coef[q][b], v[b][j], o);
                out[q][j] = o;
            }
    }
    if (a.epi == 0) {
#pragma unroll
        for (int q = 0; q < QM; ++q)
#pragma unroll
            for (int j = 0; j < 3; ++j) a.raw_out[(long)blockIdx.x * a.rbs + ((long)q * NTK + tok) * H + t + NT * j] = out[q][j];
    } else if (a.epi == 2) {
        float ssq[QM];
#pragma unroll
        for (int q = 0; q < QM; ++q) ssq[q] = out[q][0] * out[q][0] + out[q][1] * out[q][1] + out[q][2] * out[q][2];
        bred<QM>(ssq, sm + NW * 30);
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            const float f = rsqrtf(ssq[q] / H + 1e-6f);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const float xv = npre[q][j] * (out[q][j] * f);
                a.xh_out[((long)q * NTK + tok) * H + t + NT * j] = __float2half(xv);
                if (a.xn_out) a.xn_out[((long)q * NTK + tok) * H + t + NT * j] = xv;
            }
        }
    } else {
        float ssq[QM];
#pragma unroll
        for (int q = 0; q < QM; ++q) ssq[q] = out[q][0] * out[q][0] + out[q][1] * out[q][1] + out[q][2] * out[q][2];
        bred<QM>(ssq, sm + NW * 30);
        const int warp = t >> 5;
#pragma unroll
        for (int q = 0; q < QM; ++q) {
            const float f = rsqrtf(ssq[q] / H + 1e-6f);
            const Q8Buf o = q8_stream(a.q8, (int)(q * NTK + tok));
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const float xn = npre[q][j] * (out[q][j] * f);
                a.xn_out[((long)q * NTK + tok) * H + t + NT * j] = xn;
                q8_store_chunk(xn, warp + NW * j, o);
            }
        }
    }
}
void attn_res(const ArArgs& a, int nblocks, cudaStream_t s, int rows) {
    const dim3 grid(nblocks, rows);
    if (a.Q == 1) k_ar<1><<<grid, NT, 0, s>>>(a);
    else if (a.Q == 3) k_ar<3><<<grid, NT, 0, s>>>(a);
    else k_ar<4><<<grid, NT, 0, s>>>(a);
}

__global__ void __launch_bounds__(NT) k_norm(NormArgs a) {
    __shared__ float sm[NW * 32];
    const int t = threadIdx.x, s = a.T > 1 ? blockIdx.y : blockIdx.x, warp = t >> 5, tk = a.T > 1 ? blockIdx.x : 0;
    // select the stream's pointers without a dynamically indexed kernel-parameter array (that forced a 128-byte local copy of the arguments)
    const float* inp = a.in[0];
    const float* nwp = a.nw[0];
    if (s == 1) { inp = a.in[1]; nwp = a.nw[1]; } else if (s == 2) { inp = a.in[2]; nwp = a.nw[2]; } else if (s == 3) { inp = a.in[3]; nwp = a.nw[3]; }
    float v[3], ss = 0.f, wv[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) wv[j] = nwp[t + NT * j];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        v[j] = inp[(long)tk * H + t + NT * j];
        ss += v[j] * v[j];
    }
    float r[1] = {ss};
    bred<1>(r, sm);
    const float f = rsqrtf(r[0] / H + 1e-6f);
    if (a.xh_out) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float xv = wv[j] * (v[j] * f);
            a.xh_out[((long)s * a.T + tk) * H + t + NT * j] = __float2half(xv);
            if (a.xn_out) a.xn_out[((long)s * a.T + tk) * H + t + NT * j] = xv;
        }
        return;
    }
    const Q8Buf o = q8_stream(a.q8, s * a.T + tk);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float xn = wv[j] * (v[j] * f);
        a.xn_out[((long)s * a.T + tk) * H + t + NT * j] = xn;
        q8_store_chunk(xn, warp + NW * j, o);
    }
}
void norm_q8(const NormArgs& a, cudaStream_t s) { k_norm<<<a.T > 1 ? dim3(a.Tg ? a.Tg : a.T, a.S) : dim3(a.S), NT, 0, s>>>(a); }

// ------------------------------------------------------------------------------------------------------------------
// loop message gates: g = down @ [rms_u(h), rms_u(msg)] ; out = (1 - tanh(F g)) h + tanh(G g) msg
__global__ void __launch_bounds__(256) k_gate_down(GateArgs a) {
    __shared__ float xs[2 * H];
    __shared__ float red[64];
    const int s = blockIdx.y / a.R, r = blockIdx.y % a.R, tid = threadIdx.x;
    const float* h = a.h[s] + r * a.hrs;
    const float* m = a.msg[s] + r * a.mrs;
    const int warp = tid >> 5, lane = tid & 31, row = blockIdx.x * GD_RPB + (warp % GD_RPB);
    // the weight row (4608 halves = 18 uint4 per lane) and both input vectors are loaded up front: nothing below waits for a second memory round trip
    uint4 wv[18];
    if (warp < GD_RPB) {
        const uint4* wp = reinterpret_cast<const uint4*>(a.down[s] + (size_t)row * 2 * H);
#pragma unroll
        for (int k = 0; k < 18; ++k) wv[k] = __ldg(wp + lane + 32 * k);
    }
    float hv[9], mv[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) { hv[j] = h[tid + 256 * j]; mv[j] = m[tid + 256 * j]; }
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        s1 += hv[j] * hv[j];
        s2 += mv[j] * mv[j];
    }
    // two block sums sharing one set of barriers (each is the same warp-tree + warp-0 tree as block_sum)
    {
        const int lane0 = tid & 31, w0 = tid >> 5;
        s1 = warp_sum(s1);
        s2 = warp_sum(s2);
        __syncthreads();
        if (lane0 == 0) { red[w0] = s1; red[32 + w0] = s2; }
        __syncthreads();
        float t1 = (tid < 8) ? red[tid] : 0.f, t2 = (tid < 8) ? red[32 + tid] : 0.f;
        if (w0 == 0) {
            t1 = warp_sum(t1);
            t2 = warp_sum(t2);
            if (lane0 == 0) { red[0] = t1; red[32] = t2; }
        }
        __syncthreads();
        s1 = red[0];
        s2 = red[32];
    }
    const float r1 = rsqrtf(s1 / H + 1e-6f);
    const float r2 = rsqrtf(s2 / H + 1e-6f);
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        xs[tid + 256 * j] = hv[j] * r1;
        xs[H + tid + 256 * j] = mv[j] * r2;
    }
    __syncthreads();
    if (warp >= GD_RPB) return;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 18; ++k) {
        const int i = lane + 32 * k;
        const __half2* hh = reinterpret_cast<const __half2*>(&wv[k]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __half22float2(hh[e]);
            acc = fmaf(f.x, xs[i * 8 + 2 * e], fmaf(f.y, xs[i * 8 + 2 * e + 1], acc));
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) a.g[(s * a.R + r) * 64 + row] = acc;
}
__global__ void __launch_bounds__(256) k_gate_apply(GateArgs a) {
    __shared__ float g[64];
    const int s = blockIdx.y / a.R, r = blockIdx.y % a.R;
    const int i = blockIdx.x * 256 + threadIdx.x;
    const bool act = i < H;
    // weights and residual inputs first (they do not depend on the gate vector)
    uint4 fu[8], gu[8];
    float hi = 0.f, mi = 0.f;
    if (act) {
        const uint4* fp = reinterpret_cast<const uint4*>(a.forget[s] + (size_t)i * 64);
        const uint4* gp = reinterpret_cast<const uint4*>(a.gain[s] + (size_t)i * 64);
#pragma unroll
        for (int k = 0; k < 8; ++k) { fu[k] = __ldg(fp + k); gu[k] = __ldg(gp + k); }
        hi = a.h[s][r * a.hrs + i];
        mi = a.msg[s][r * a.mrs + i];
    }
    if (threadIdx.x < 64) g[threadIdx.x] = a.g[(s * a.R + r) * 64 + threadIdx.x];
    __syncthreads();
    if (!act) return;
    float f = 0.f, gn = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const __half2* hu = reinterpret_cast<const __half2*>(&fu[k]);
        const __half2* hw = reinterpret_cast<const __half2*>(&gu[k]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 a1 = __half22float2(hu[e]), b1 = __half22float2(hw[e]);
            f = fmaf(a1.x, g[k * 8 + 2 * e], fmaf(a1.y, g[k * 8 + 2 * e + 1], f));
            gn = fmaf(b1.x, g[k * 8 + 2 * e], fmaf(b1.y, g[k * 8 + 2 * e + 1], gn));
        }
    }
    a.out[s][r * a.ors + i] = (1.f - tanhf(f)) * hi + tanhf(gn) * mi;
}
void gate(const GateArgs& a, cudaStream_t s) {
    k_gate_down<<<dim3(64 / GD_RPB, a.S * a.R), 256, 0, s>>>(a);
    k_gate_apply<<<dim3((H + 255) / 256, a.S * a.R), 256, 0, s>>>(a);
}

// ------------------------------------------------------------------------------------------------------------------
// multi-matrix GEMV: 4 warps per block, R rows per warp, each warp owns whole rows (lanes loop over 32-element chunks)
template <int BITS, int R>
__global__ void __launch_bounds__(128, MB_MG) k_mgemv(GDs D) {
    int b = blockIdx.x, di = 0;
#pragma unroll 1
    while (di < D.n - 1 && b >= D.d[di].nblk) {
        b -= D.d[di].nblk;
        ++di;
    }
    const GD& g = D.d[di];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (b * MG_WPB + warp) * R;
    if (row0 >= g.N) return;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    if (g.kind == 0) {
        const int C = g.K >> 5, GS = g.gshift - 5, G = g.K >> g.gshift;
        float rsc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rsc[r] = __half2float(g.rs[min(row0 + r, g.N - 1)]) / 255.0f;
UNROLL_N(UNQ)
        for (int c = lane; c < C; c += 32) {
            int xw[8], qs;
            float dx, xs;
            load_x(g.xq, c, xw, dx, xs, qs);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (row < g.N) {
                    float sc, mn;
                    sm8_decode(g.sm8[(size_t)row * G + (c >> GS)], rsc[r], g.zlo, g.zstep, sc, mn);
                    acc[r] = qchunk<BITS>(g.q + ((size_t)row * C + c) * BITS, xw, dx, xs, qs, sc, mn, acc[r]);
                }
            }
        }
    } else {
        const int K8 = g.K >> 3;
UNROLL_N(UNF)
        for (int i = lane; i < K8; i += 32) {
            const float4 x0 = *reinterpret_cast<const float4*>(g.xf + i * 8), x1 = *reinterpret_cast<const float4*>(g.xf + i * 8 + 4);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (row < g.N) {
                    const uint4 u = __ldg(reinterpret_cast<const uint4*>(g.w16 + (size_t)row * g.K) + i);
                    const __half2* hh = reinterpret_cast<const __half2*>(&u);
                    const float2 a0 = __half22float2(hh[0]), a1 = __half22float2(hh[1]), a2 = __half22float2(hh[2]), a3 = __half22float2(hh[3]);
                    acc[r] = fmaf(a0.x, x0.x, fmaf(a0.y, x0.y, fmaf(a1.x, x0.z, fmaf(a1.y, x0.w, fmaf(a2.x, x1.x, fmaf(a2.y, x1.y, fmaf(a3.x, x1.z, fmaf(a3.y, x1.w, acc[r]))))))));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0 && row0 + r < g.N) g.y[row0 + r] = v;
    }
}
void mgemv(GDs& g, int bits, cudaStream_t s) {
    constexpr int R = 1;
    int tot = 0;
    for (int i = 0; i < g.n; ++i) {
        g.d[i].nblk = (g.d[i].N + MG_WPB * R - 1) / (MG_WPB * R);
        tot += g.d[i].nblk;
    }
    switch (bits) {
        case 3: k_mgemv<3, R><<<tot, MG_WPB * 32, 0, s>>>(g); break;
        case 4: k_mgemv<4, R><<<tot, MG_WPB * 32, 0, s>>>(g); break;
        case 5: k_mgemv<5, R><<<tot, MG_WPB * 32, 0, s>>>(g); break;
        case 6: k_mgemv<6, R><<<tot, MG_WPB * 32, 0, s>>>(g); break;
        case 8: k_mgemv<8, R><<<tot, MG_WPB * 32, 0, s>>>(g); break;
        default: std::fprintf(stderr, "mgemv: unsupported bits %d\n", bits); std::abort();
    }
}

// K and V int4 stores of one token / kv head with one barrier (the per-64-channel-group min / max are order independent: same values as two kv4_store).
// sh: >= 16 floats, written once (fresh memory).
__device__ __forceinline__ void kv4_store2(float kval, float vval, int c, uint8_t* kq, __half2* ks, uint8_t* vq, __half2* vs, float* sh) {
    float mnk = kval, mxk = kval, mnv = vval, mxv = vval;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mnk = fminf(mnk, __shfl_xor_sync(0xffffffffu, mnk, o)); mxk = fmaxf(mxk, __shfl_xor_sync(0xffffffffu, mxk, o));
        mnv = fminf(mnv, __shfl_xor_sync(0xffffffffu, mnv, o)); mxv = fmaxf(mxv, __shfl_xor_sync(0xffffffffu, mxv, o));
    }
    const int w = c >> 5;
    if ((c & 31) == 0) { sh[w * 4] = mnk; sh[w * 4 + 1] = mxk; sh[w * 4 + 2] = mnv; sh[w * 4 + 3] = mxv; }
    __syncthreads();
    const int g = c >> 6;
    mnk = fminf(sh[g * 8], sh[g * 8 + 4]);
    mxk = fmaxf(sh[g * 8 + 1], sh[g * 8 + 5]);
    mnv = fminf(sh[g * 8 + 2], sh[g * 8 + 6]);
    mxv = fmaxf(sh[g * 8 + 3], sh[g * 8 + 7]);
    const __half shk = __float2half(fmaxf((mxk - mnk) * (1.f / 15.f), 1e-5f)), mhk = __float2half(mnk);
    const __half shv = __float2half(fmaxf((mxv - mnv) * (1.f / 15.f), 1e-5f)), mhv = __float2half(mnv);
    const float sck = __half2float(shk), mfk = __half2float(mhk), scv = __half2float(shv), mfv = __half2float(mhv);
    int qk = __float2int_rn((kval - mfk) / sck), qvv = __float2int_rn((vval - mfv) / scv);
    qk = max(0, min(15, qk));
    qvv = max(0, min(15, qvv));
    const int qkn = __shfl_xor_sync(0xffffffffu, qk, 1), qvn = __shfl_xor_sync(0xffffffffu, qvv, 1);
    if ((c & 1) == 0) { kq[c >> 1] = (uint8_t)(qk | (qkn << 4)); vq[c >> 1] = (uint8_t)(qvv | (qvn << 4)); }
    if ((c & 63) == 0) { ks[g] = __halves2half2(shk, mhk); vs[g] = __halves2half2(shv, mhv); }
}

// ------------------------------------------------------------------------------------------------------------------
// attention prep: q/k/v/rel post-processing, short convs, KV write, relative bias table. grid = 8 kv heads, 128 threads
__global__ void __launch_bounds__(128) k_prep(PrepArgs a) {
    __shared__ float rv[48], rv0[48];
    __shared__ float red5[48];
    const int g = blockIdx.x, c = threadIdx.x, ch = g * HD + c;
    const int row = blockIdx.z;
    const int pos0 = *a.pos, pos = pos0 + row;
    float tau = 1.f;
    if (a.glob) tau = 1.f + 0.1f * logf(fmaxf((float)(pos + 1) / 8192.f, 1.f));
    const int slot = a.ring ? (pos & (a.cap - 1)) : pos;
    if (blockIdx.y == 0) {
    // ---- k and v: short conv (state = 3 previous pre-conv inputs), then clip / value norm
    const float* kr = a.kraw + row * a.prs;
    const float* vr = a.vraw + row * a.prs;
    float kx = kr[ch], vx = vr[ch];
    float ko, vo;
    {
        // x_{t-3}, x_{t-2}, x_{t-1}: earlier rows of this verify batch, else the conv state
        float k0, k1, k2, v0, v1, v2;
        {
            const int b3 = row - 3, b2 = row - 2, b1 = row - 1;
            k0 = b3 >= 0 ? a.kraw[b3 * a.prs + ch] : a.kst[(3 + b3) * 1024 + ch];
            k1 = b2 >= 0 ? a.kraw[b2 * a.prs + ch] : a.kst[(3 + b2) * 1024 + ch];
            k2 = b1 >= 0 ? a.kraw[b1 * a.prs + ch] : a.kst[(3 + b1) * 1024 + ch];
            v0 = b3 >= 0 ? a.vraw[b3 * a.prs + ch] : a.vst[(3 + b3) * 1024 + ch];
            v1 = b2 >= 0 ? a.vraw[b2 * a.prs + ch] : a.vst[(3 + b2) * 1024 + ch];
            v2 = b1 >= 0 ? a.vraw[b1 * a.prs + ch] : a.vst[(3 + b1) * 1024 + ch];
        }
        if (a.glob) {
            // taps[l] = static_w[l][ch] + dw[g*4+l]; lag 0 = current
            const float* dwk = a.dwk + row * a.prs;
            const float* dwv = a.dwv + row * a.prs;
            const float t0 = a.kw[ch] + dwk[g * 4 + 0], t1 = a.kw[1024 + ch] + dwk[g * 4 + 1], t2 = a.kw[2048 + ch] + dwk[g * 4 + 2],
                        t3 = a.kw[3072 + ch] + dwk[g * 4 + 3];
            ko = (t0 * kx + t1 * k2 + t2 * k1 + t3 * k0 + kx) * (1.f / 3.f);
            const float u0 = a.vw[ch] + dwv[g * 4 + 0], u1 = a.vw[1024 + ch] + dwv[g * 4 + 1], u2 = a.vw[2048 + ch] + dwv[g * 4 + 2],
                        u3 = a.vw[3072 + ch] + dwv[g * 4 + 3];
            vo = (u0 * vx + u1 * v2 + u2 * v1 + u3 * v0 + vx) * (1.f / 3.f);
        } else {
            const float4 wk = *reinterpret_cast<const float4*>(a.kw + 4 * ch), wv = *reinterpret_cast<const float4*>(a.vw + 4 * ch);
            ko = (wk.x * k0 + wk.y * k1 + wk.z * k2 + wk.w * kx + kx) * 0.44721359549995794f;
            vo = (wv.x * v0 + wv.y * v1 + wv.z * v2 + wv.w * vx + vx) * 0.44721359549995794f;
        }
        if (a.rows == 1) {
            a.kst[ch] = k1; a.kst[1024 + ch] = k2; a.kst[2048 + ch] = kx;
            a.vst[ch] = v1; a.vst[1024 + ch] = v2; a.vst[2048 + ch] = vx;
        }
        if (a.khist) { a.khist[row * 1024 + ch] = kx; a.vhist[row * 1024 + ch] = vx; }
    }
    {
        // the five block sums (|k|^2, |v|^2, |q_h|^2 for the 3 heads) share one set of barriers; each value is reduced exactly like block_sum
        float qv[3];
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) qv[hh] = a.qraw[row * a.prs + (g * 3 + hh) * HD + c] * a.qscale[(g * 3 + hh) * HD + c];
        float r5[5] = {ko * ko, vo * vo, qv[0] * qv[0], qv[1] * qv[1], qv[2] * qv[2]};
        {
            const int lane = c & 31, w = c >> 5;
#pragma unroll
            for (int i = 0; i < 5; ++i) r5[i] = warp_sum(r5[i]);
            if (lane == 0) {
#pragma unroll
                for (int i = 0; i < 5; ++i) red5[i * 4 + w] = r5[i];
            }
            __syncthreads();
            if (w == 0) {
#pragma unroll
                for (int i = 0; i < 5; ++i) {
                    float t = lane < 4 ? red5[i * 4 + lane] : 0.f;
                    t = warp_sum(t);
                    if (lane == 0) red5[20 + i] = t;
                }
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 5; ++i) r5[i] = red5[20 + i];
        }
        const float fk = fminf(1.f, 22.627416997969522f / (sqrtf(r5[0]) + 1e-6f));
        const float fv = rsqrtf(r5[1] / HD + 1e-6f);
        if (a.kq) {
            const size_t r = (size_t)g * a.cap + slot;
            kv4_store2(ko * fk, vo * fv, c, a.kq + r * 64, a.ks + r * 2, a.vq + r * 64, a.vs + r * 2, red5 + 28);
        } else if (a.ktail) {
            a.ktail[((size_t)g * a.tail_rows + row) * HD + c] = __float2half(ko * fk);
            a.vtail[((size_t)g * a.tail_rows + row) * HD + c] = __float2half(vo * fv);
        } else {
            a.kc[((size_t)g * a.cap + slot) * HD + c] = __float2half(ko * fk);
            a.vc[((size_t)g * a.cap + slot) * HD + c] = __float2half(vo * fv);
        }
        // ---- q of the 3 heads sharing this kv head: l2 clip, tau, softmax scale
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            const float f = fminf(1.f, 22.627416997969522f / (sqrtf(r5[2 + hh]) + 1e-6f));
            a.qs[row * (NH * HD) + (g * 3 + hh) * HD + c] = qv[hh] * f * tau * 0.08838834764831845f;  // 128^-0.5
        }
    }
    }
    // ---- rel: scale, l2 clip (8.0), tau
    if (c < 48) rv0[c] = a.relraw[row * a.prs + (g * 3 + c / 16) * 16 + (c & 15)] * a.relscale[(g * 3 + c / 16) * 16 + (c & 15)];
    __syncthreads();
    if (c < 48) {
        float n2 = 0.f;
        for (int k = 0; k < 16; ++k) n2 += rv0[(c / 16) * 16 + k] * rv0[(c / 16) * 16 + k];
        rv[c] = rv0[c] * fminf(1.f, 8.f / (sqrtf(n2) + 1e-6f)) * tau;
    }
    __syncthreads();
    {
        const int d = blockIdx.y * 128 + c;
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            float b = 0.f;
#pragma unroll
            for (int k = 0; k < 16; ++k) b = fmaf(rv[hh * 16 + k], a.relproj[k * REXT + d], b);
            a.rl[(size_t)row * (NH * REXT) + (g * 3 + hh) * REXT + d] = b;
        }
    }
}
void attn_prep(const PrepArgs& a, cudaStream_t s) { k_prep<<<dim3(NKV, 8, 1), 128, 0, s>>>(a); }
void attn_prep_rows(const PrepArgs& a, cudaStream_t s) { k_prep<<<dim3(NKV, 8, a.rows), 128, 0, s>>>(a); }

// ------------------------------------------------------------------------------------------------------------------
// decode attention partials. grid (8 kv heads, nsplit); 128 threads = 4 warps x 4 groups of 8 lanes; a group owns one key at a time,
// lane `sub` holds 16 of the 128 dims. The 3 query heads of the kv head are processed together.
__global__ void __launch_bounds__(128) k_attn_part(AttnArgs a) {
    __shared__ float gacc[16][3][HD];
    __shared__ float gm[16][3], gl[16][3];
    const int g = blockIdx.x, sp = blockIdx.y, nsp = gridDim.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 3, sub = lane & 7;
    const int gi = warp * 4 + grp;
    const unsigned gmask = 0xFFu << (grp * 8);
    const int row = blockIdx.z;
    const int pos0 = *a.pos, pos = pos0 + (a.bidir ? 7 : row);
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int n = pos - lo + 1;
    const int chunk = (n + nsp - 1) / nsp;
    const int kb = lo + sp * chunk, ke = min(pos + 1, kb + chunk);
    float q[3][16];
#pragma unroll
    for (int hh = 0; hh < 3; ++hh)
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 v = *reinterpret_cast<const float4*>(a.qs + row * (NH * HD) + (g * 3 + hh) * HD + sub * 16 + i);
            q[hh][i] = v.x; q[hh][i + 1] = v.y; q[hh][i + 2] = v.z; q[hh][i + 3] = v.w;
        }
    float m[3] = {-1e30f, -1e30f, -1e30f}, l[3] = {0.f, 0.f, 0.f}, acc[3][16];
#pragma unroll
    for (int hh = 0; hh < 3; ++hh)
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[hh][i] = 0.f;
    for (int j = kb + gi; j < ke; j += 16) {
        const int slot = a.ring ? (j & (a.cap - 1)) : j;
        const bool tl = a.ktail && j >= pos0;
        const size_t base = tl ? ((size_t)g * a.tail_rows + (j - pos0)) * HD + sub * 16 : ((size_t)g * a.cap + slot) * HD + sub * 16;
        const __half* kcp = tl ? a.ktail : a.kc;
        const __half* vcp = tl ? a.vtail : a.vc;
        float kf[16], vf[16];
        if (a.kq) {
            const size_t r = (size_t)g * a.cap + slot;
            const uint2 kb8 = __ldg(reinterpret_cast<const uint2*>(a.kq + r * 64) + sub);
            const uint2 vb8 = __ldg(reinterpret_cast<const uint2*>(a.vq + r * 64) + sub);
            const float2 ksc = __half22float2(a.ks[r * 2 + (sub >> 2)]), vsc = __half22float2(a.vs[r * 2 + (sub >> 2)]);
            const uint32_t kw[2] = {kb8.x, kb8.y}, vw[2] = {vb8.x, vb8.y};
#pragma unroll
            for (int w = 0; w < 2; ++w)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    kf[w * 8 + i] = fmaf((float)((kw[w] >> (4 * i)) & 15u), ksc.x, ksc.y);
                    vf[w * 8 + i] = fmaf((float)((vw[w] >> (4 * i)) & 15u), vsc.x, vsc.y);
                }
        } else {
        const uint4 k0 = __ldg(reinterpret_cast<const uint4*>(kcp + base)), k1 = __ldg(reinterpret_cast<const uint4*>(kcp + base) + 1);
        const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(vcp + base)), v1 = __ldg(reinterpret_cast<const uint4*>(vcp + base) + 1);
        {
            const __half2* p0 = reinterpret_cast<const __half2*>(&k0);
            const __half2* p1 = reinterpret_cast<const __half2*>(&k1);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f0 = __half22float2(p0[i]), f1 = __half22float2(p1[i]);
                kf[2 * i] = f0.x; kf[2 * i + 1] = f0.y; kf[8 + 2 * i] = f1.x; kf[8 + 2 * i + 1] = f1.y;
            }
            const __half2* q0 = reinterpret_cast<const __half2*>(&v0);
            const __half2* q1 = reinterpret_cast<const __half2*>(&v1);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f0 = __half22float2(q0[i]), f1 = __half22float2(q1[i]);
                vf[2 * i] = f0.x; vf[2 * i + 1] = f0.y; vf[8 + 2 * i] = f1.x; vf[8 + 2 * i + 1] = f1.y;
            }
        }
        }
        float s[3];
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) d = fmaf(q[hh][i], kf[i], d);
            d += __shfl_xor_sync(gmask, d, 1);
            d += __shfl_xor_sync(gmask, d, 2);
            d += __shfl_xor_sync(gmask, d, 4);
            const int dist = pos - j;
            s[hh] = d + (dist < REXT ? a.rl[(size_t)row * (NH * REXT) + (g * 3 + hh) * REXT + dist] : 0.f);
        }
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            const float mn = fmaxf(m[hh], s[hh]);
            const float corr = __expf(m[hh] - mn), p = __expf(s[hh] - mn);
            l[hh] = l[hh] * corr + p;
            m[hh] = mn;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[hh][i] = fmaf(p, vf[i], acc[hh][i] * corr);
        }
    }
#pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
#pragma unroll
        for (int i = 0; i < 16; ++i) gacc[gi][hh][sub * 16 + i] = acc[hh][i];
        if (sub == 0) { gm[gi][hh] = m[hh]; gl[gi][hh] = l[hh]; }
    }
    __syncthreads();
    for (int hh = 0; hh < 3; ++hh) {
        float M = -1e30f;
        for (int k = 0; k < 16; ++k) M = fmaxf(M, gm[k][hh]);
        float L = 0.f, o = 0.f;
        for (int k = 0; k < 16; ++k) {
            const float e = gm[k][hh] > -1e29f ? __expf(gm[k][hh] - M) : 0.f;
            L += e * gl[k][hh];
            o += e * gacc[k][hh][tid];
        }
        float* out = a.part + (size_t)row * (NKV * nsp * 3 * PART_STRIDE) + ((size_t)(g * nsp + sp) * 3 + hh) * PART_STRIDE;
        out[2 + tid] = o;
        if (tid == 0) { out[0] = M; out[1] = L; }
    }
}
void attn_part(const AttnArgs& a, cudaStream_t s) { k_attn_part<<<dim3(NKV, a.nsplit, 1), 128, 0, s>>>(a); }
void attn_part_rows(const AttnArgs& a, cudaStream_t s) { k_attn_part<<<dim3(NKV, a.nsplit, a.rows), 128, 0, s>>>(a); }

// merge split partials + sink logit (value 0), multiply by sigmoid(gate), quantize to int8 for wo. grid 24 heads, 128 threads
__global__ void __launch_bounds__(128) k_attn_combine(AttnArgs a) {
    // the per-split maxima / sums are staged in shared memory by threads 0..nsp-1 (one parallel round of loads instead of a dependent load chain per
    // split); the arithmetic and the order of the split loop are unchanged
    __shared__ float sml[2 * 128];
    const int h = blockIdx.x, g = h / 3, hh = h % 3, d = threadIdx.x, nsp = a.nsplit, row = blockIdx.y;
    const float sink = a.sink[row * a.ss_rs + h];
    const float* partr = a.part + (size_t)row * (NKV * nsp * 3 * PART_STRIDE);
    if (d < nsp) {
        const float* p = partr + ((size_t)(g * nsp + d) * 3 + hh) * PART_STRIDE;
        sml[d] = p[0];
        sml[128 + d] = p[1];
    }
    const float gt = a.gate[row * a.gate_rs + h * HD + d];
    __syncthreads();
    float M = sink;
    for (int s = 0; s < nsp; ++s)
        if (sml[128 + s] > 0.f) M = fmaxf(M, sml[s]);
    float L = __expf(sink - M), o = 0.f;
    for (int s0 = 0; s0 < nsp; s0 += 8) {
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = (s0 + k < nsp) ? partr[((size_t)(g * nsp + s0 + k) * 3 + hh) * PART_STRIDE + 2 + d] : 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int s = s0 + k;
            if (s < nsp && sml[128 + s] > 0.f) {
                const float e = __expf(sml[s] - M);
                L += e * sml[128 + s];
                o += e * v[k];
            }
        }
    }
    const float v = o / L * (1.f / (1.f + __expf(-gt)));
    Q8Buf ob = a.out;
    ob.q += (size_t)row * a.q8_rs * 8; ob.d += row * a.q8_rs; ob.sum += row * a.q8_rs; ob.qsum += row * a.q8_rs;
    q8_store_chunk(v, h * 4 + (d >> 5), ob);
}
void attn_combine(const AttnArgs& a, cudaStream_t s) { k_attn_combine<<<dim3(NH, 1), 128, 0, s>>>(a); }
void attn_combine_rows(const AttnArgs& a, cudaStream_t s) { k_attn_combine<<<dim3(NH, a.rows), 128, 0, s>>>(a); }

// ------------------------------------------------------------------------------------------------------------------
// router
__global__ void __launch_bounds__(64) k_router(const __half* W, const float* bias, const float* x, float* logits, int* ids, float* wts,
                                                int* ticket, float route_scale) {
    __shared__ int s_last;
    {   // MTP verify: blockIdx.y = token row (decode: one row, offsets 0)
        const int r = blockIdx.y;
        x += (size_t)r * H; logits += r * NE; ids += r * TOPK; wts += r * TOPK; ticket += r;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row = blockIdx.x * 2 + warp;
    float acc = 0.f;
UNROLL_N(UNR)
    for (int i = lane; i < H / 8; i += 32) {
        const float4 x0 = *reinterpret_cast<const float4*>(x + i * 8), x1 = *reinterpret_cast<const float4*>(x + i * 8 + 4);
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(W + (size_t)row * H) + i);
        const __half2* hh = reinterpret_cast<const __half2*>(&u);
        const float2 a0 = __half22float2(hh[0]), a1 = __half22float2(hh[1]), a2 = __half22float2(hh[2]), a3 = __half22float2(hh[3]);
        acc = fmaf(a0.x, x0.x, fmaf(a0.y, x0.y, fmaf(a1.x, x0.z, fmaf(a1.y, x0.w, fmaf(a2.x, x1.x, fmaf(a2.y, x1.y, fmaf(a3.x, x1.z, fmaf(a3.y, x1.w, acc))))))));
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[row] = acc;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = (atomicAdd(ticket, 1) == (int)gridDim.x - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    // top-8 by (sigmoid + bias) with ties to the lower index, by rank: element i is selected at slot r iff exactly r elements precede it in the order
    // (score descending, index ascending) -- the same selection and the same order as 8 rounds of warp argmax. 64 threads, 2 elements each.
    __shared__ float s_sb[NE], s_sv[NE], s_w[TOPK];
    __shared__ int s_id[TOPK];
    {
        const int tid = threadIdx.x;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = tid + 64 * k;
            const float l = __ldcg(logits + i);
            const float sv = 1.f / (1.f + __expf(-l));
            s_sv[i] = sv;
            s_sb[i] = sv + bias[i];
        }
    }
    __syncthreads();
    {
        const int tid = threadIdx.x;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = tid + 64 * k;
            const float me = s_sb[i];
            int rank = 0;
#pragma unroll 16
            for (int j = 0; j < NE; ++j) {
                const float o = s_sb[j];
                rank += (o > me) || (o == me && j < i);
            }
            if (rank < TOPK) { s_id[rank] = i; s_w[rank] = s_sv[i]; }
        }
    }
    __syncthreads();
    if (threadIdx.x < TOPK) {
        float wsum = 0.f;
#pragma unroll
        for (int r = 0; r < TOPK; ++r) wsum += s_w[r];
        ids[threadIdx.x] = s_id[threadIdx.x];
        wts[threadIdx.x] = s_w[threadIdx.x] / (wsum + 1e-20f) * route_scale;
    }
    if (threadIdx.x == 0) *ticket = 0;
}
void router(const __half* W, const float* bias, const float* x, float* logits, int* ids, float* wts, int* ticket, float route_scale, cudaStream_t s) {
    k_router<<<NE / 2, 64, 0, s>>>(W, bias, x, logits, ids, wts, ticket, route_scale);
}
void router_rows(const __half* W, const float* bias, const float* x, float* logits, int* ids, float* wts, int* ticket, float route_scale, int rows,
                 cudaStream_t s) {
    k_router<<<dim3(NE / 2, rows), 64, 0, s>>>(W, bias, x, logits, ids, wts, ticket, route_scale);
}

// grouped expert GEMV 1: rows of w13 are interleaved (even = gate, odd = up). block = 64 rows = 32 activations, 16 warps x 4 rows.
template <int BITS>
__global__ void __launch_bounds__(512, MB_W13) k_w13(ExpertW W, Q8Vec x, const int* ids, Q8Buf act, float gcap, float ucap) {
    __shared__ float actv[32];
    const int rb = blockIdx.x, slot = blockIdx.y, e = ids[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = rb * 64 + warp * 4;
    const size_t C = W.K >> 5, G = W.K >> W.gshift;
    const int GS = W.gshift - 5;
    const uint32_t* qb = W.q + (size_t)e * W.N * C * BITS;
    const uint16_t* smb = W.sm8 + (size_t)e * W.N * G;
    const __half* rsb = W.rs + (size_t)e * W.N;
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, rsc[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) rsc[r] = __half2float(rsb[row0 + r]) / 255.0f;
UNROLL_N(UNW)
    for (size_t c = lane; c < C; c += 32) {
        int xw[8], qs;
        float dx, xs;
        load_x(x, c, xw, dx, xs, qs);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float sc, mn;
            sm8_decode(smb[(size_t)(row0 + r) * G + (c >> GS)], rsc[r], W.zlo, W.zstep, sc, mn);
            acc[r] = qchunk<BITS>(qb + ((size_t)(row0 + r) * C + c) * BITS, xw, dx, xs, qs, sc, mn, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[r] = warp_sum(acc[r]);
    if (lane == 0) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const float gv = acc[2 * p], uv = acc[2 * p + 1];
            const float gc = gcap * tanhf(gv / gcap), uc = ucap * tanhf(uv / ucap);
            actv[warp * 2 + p] = gc * (1.f / (1.f + __expf(-gv))) * uc;
        }
    }
    __syncthreads();
    if (warp == 0) q8_store_chunk(actv[lane], slot * (W.N / 64) + rb, act);
}
void moe_w13(const ExpertW& W, const Q8Vec& x, const int* ids, Q8Buf act, float gcap, float ucap, cudaStream_t s) {
    const dim3 grid(W.N / 64, TOPK);
    switch (W.bits) {
        case 2: break;
        case 3: k_w13<3><<<grid, 512, 0, s>>>(W, x, ids, act, gcap, ucap); return;
        case 4: k_w13<4><<<grid, 512, 0, s>>>(W, x, ids, act, gcap, ucap); return;
        case 5: k_w13<5><<<grid, 512, 0, s>>>(W, x, ids, act, gcap, ucap); return;
        default: break;
    }
    std::fprintf(stderr, "moe_w13: unsupported bits %d\n", W.bits);
    std::abort();
}

// grouped expert GEMV 2: block = R rows of w2 for all 8 expert slots (warp = slot); the 8 slots are summed in slot order.
template <int BITS, int R>
__global__ void __launch_bounds__(256, MB_W2) k_w2(ExpertW W, Q8Vec act, const int* ids, const float* wts, float* y) {
    __shared__ float ys[TOPK][R];
    const int slot = threadIdx.x >> 5, lane = threadIdx.x & 31, row0 = blockIdx.x * R, e = ids[slot];
    const size_t C = W.K >> 5, G = W.K >> W.gshift;
    const int GS = W.gshift - 5;
    const uint32_t* qb = W.q + (size_t)e * W.N * C * BITS;
    const uint16_t* smb = W.sm8 + (size_t)e * W.N * G;
    const __half* rsb = W.rs + (size_t)e * W.N;
    float acc[R], rsc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { acc[r] = 0.f; rsc[r] = __half2float(rsb[row0 + r]) / 255.0f; }
UNROLL_N(UNW)
    for (size_t c = lane; c < C; c += 32) {
        int xw[8], qs;
        float dx, xs;
        load_x(act, slot * C + c, xw, dx, xs, qs);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float sc, mn;
            sm8_decode(smb[(size_t)(row0 + r) * G + (c >> GS)], rsc[r], W.zlo, W.zstep, sc, mn);
            acc[r] = qchunk<BITS>(qb + ((size_t)(row0 + r) * C + c) * BITS, xw, dx, xs, qs, sc, mn, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) ys[slot][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < R) {
        float o = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) o = fmaf(wts[k], ys[k][threadIdx.x], o);
        y[row0 + threadIdx.x] = o;
    }
}
void moe_w2(const ExpertW& W, const Q8Vec& act, const int* ids, const float* wts, float* y, cudaStream_t s) {
    constexpr int R = 2;
    const int grid = W.N / R;
    switch (W.bits) {
        case 3: k_w2<3, R><<<grid, 256, 0, s>>>(W, act, ids, wts, y); return;
        case 4: k_w2<4, R><<<grid, 256, 0, s>>>(W, act, ids, wts, y); return;
        case 5: k_w2<5, R><<<grid, 256, 0, s>>>(W, act, ids, wts, y); return;
        default: break;
    }
    std::fprintf(stderr, "moe_w2: unsupported bits %d\n", W.bits);
    std::abort();
}

__global__ void __launch_bounds__(NT) k_finish(const float* res, const float* y, const float* cw, float* cst, float* h_out, int ntok) {
    const long tok = blockIdx.x;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int c = threadIdx.x + NT * j;
        const float x = y[tok * H + c];
        const long i3 = tok - 3, i2 = tok - 2, i1 = tok - 1;
        const float s0 = i3 >= 0 ? y[i3 * H + c] : cst[(3 + i3) * H + c];
        const float s1 = i2 >= 0 ? y[i2 * H + c] : cst[(3 + i2) * H + c];
        const float s2 = i1 >= 0 ? y[i1 * H + c] : cst[(3 + i1) * H + c];
        const float4 w = *reinterpret_cast<const float4*>(cw + 4 * c);
        const float yy = w.x * s0 + w.y * s1 + w.z * s2 + w.w * x;
        if (ntok == 1) {
            cst[c] = s1;
            cst[H + c] = s2;
            cst[2 * H + c] = x;
        }
        h_out[tok * H + c] = res[tok * H + c] + (yy + x) * 0.44721359549995794f;
    }
}
void finish(const float* res, const float* y, const float* cw, float* cst, float* h_out, cudaStream_t s, int T) {
    k_finish<<<T, NT, 0, s>>>(res, y, cw, cst, h_out, T);
}

// causal-conv state update after a chunk of T rows (row stride `rs` floats, D channels): state = last 3 of [state, rows]
__global__ void k_conv_commit(const float* rows, long rs, int T, int D, float* st) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float old[3] = {st[c], st[D + c], st[2 * D + c]}, nw[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = T + i;
        nw[i] = idx < 3 ? old[idx] : rows[(long)(idx - 3) * rs + c];
    }
    st[c] = nw[0]; st[D + c] = nw[1]; st[2 * D + c] = nw[2];
}
void conv_commit(const float* rows, long rs, int T, int D, float* st, cudaStream_t s) {
    k_conv_commit<<<(D + 255) / 256, 256, 0, s>>>(rows, rs, T, D, st);
}

// ======================================================================================================================
// MTP verify kernels: rows (<= 8) share each weight load; per row the arithmetic (and its order) equals the 1-token kernels above.
// ======================================================================================================================

template <int BITS, int R, int WPB>
__global__ void __launch_bounds__(WPB * 32) k_mgemv_rows(GDs D, int nr) {
    int b = blockIdx.x, di = 0;
#pragma unroll 1
    while (di < D.n - 1 && b >= D.d[di].nblk) {
        b -= D.d[di].nblk;
        ++di;
    }
    const GD& g = D.d[di];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (b * WPB + warp) * R;
    if (row0 >= g.N) return;
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[r][t] = 0.f;
    if (g.kind == 0) {
        constexpr int ZP = 1 << (BITS - 1);
        const int C = g.K >> 5, GS = g.gshift - 5, G = g.K >> g.gshift;
        float rsc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rsc[r] = __half2float(g.rs[min(row0 + r, g.N - 1)]) / 255.0f;
        for (int c = lane; c < C; c += 32) {
            uint32_t qw[R][BITS];
            float scs[R], mns[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = min(row0 + r, g.N - 1);
                const uint32_t* wp = g.q + ((size_t)row * C + c) * BITS;
#pragma unroll
                for (int bb = 0; bb < BITS; ++bb) qw[r][bb] = __ldg(wp + bb);
                sm8_decode(g.sm8[(size_t)row * G + (c >> GS)], rsc[r], g.zlo, g.zstep, scs[r], mns[r]);
            }
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t < nr) {
                    int xw[8], qs;
                    float dx, xs;
                    Q8Vec xv = g.xq;
                    xv.q += (size_t)t * g.xstride * 8; xv.d += (size_t)t * g.xstride; xv.sum += (size_t)t * g.xstride; xv.qsum += (size_t)t * g.xstride;
                    load_x(xv, c, xw, dx, xs, qs);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (row0 + r < g.N) {
                            int d = 0;
#pragma unroll
                            for (int bb = 0; bb < 8; ++bb) d = dp4a_us(qword<BITS>(qw[r], bb), xw[bb], d);
                            d -= ZP * qs;
                            acc[r][t] = fmaf(scs[r] * dx, (float)d, fmaf(fmaf(scs[r], (float)ZP, mns[r]), xs, acc[r][t]));
                        }
                    }
                }
            }
        }
    } else {
        const int K8 = g.K >> 3;
        for (int i = lane; i < K8; i += 32) {
            float4 x0[8], x1[8];
#pragma unroll
            for (int t = 0; t < 8; ++t)
                if (t < nr) {
                    x0[t] = *reinterpret_cast<const float4*>(g.xf + t * g.xfs + i * 8);
                    x1[t] = *reinterpret_cast<const float4*>(g.xf + t * g.xfs + i * 8 + 4);
                }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
                if (row < g.N) {
                    const uint4 u = __ldg(reinterpret_cast<const uint4*>(g.w16 + (size_t)row * g.K) + i);
                    const __half2* hh = reinterpret_cast<const __half2*>(&u);
                    const float2 a0 = __half22float2(hh[0]), a1 = __half22float2(hh[1]), a2 = __half22float2(hh[2]), a3 = __half22float2(hh[3]);
#pragma unroll
                    for (int t = 0; t < 8; ++t)
                        if (t < nr)
                            acc[r][t] = fmaf(a0.x, x0[t].x, fmaf(a0.y, x0[t].y, fmaf(a1.x, x0[t].z, fmaf(a1.y, x0[t].w, fmaf(a2.x, x1[t].x, fmaf(a2.y, x1[t].y, fmaf(a3.x, x1[t].z, fmaf(a3.y, x1[t].w, acc[r][t]))))))));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (t < nr) {
                const float v = warp_sum(acc[r][t]);
                if (lane == 0 && row0 + r < g.N) g.y[t * g.ystride + row0 + r] = v;
            }
}
template <int R, int WPB>
static void mgemv_rows_t(GDs& g, int bits, int nr, cudaStream_t s) {
    int tot = 0;
    for (int i = 0; i < g.n; ++i) {
        g.d[i].nblk = (g.d[i].N + WPB * R - 1) / (WPB * R);
        tot += g.d[i].nblk;
    }
    switch (bits) {
        case 3: k_mgemv_rows<3, R, WPB><<<tot, WPB * 32, 0, s>>>(g, nr); break;
        case 4: k_mgemv_rows<4, R, WPB><<<tot, WPB * 32, 0, s>>>(g, nr); break;
        case 5: k_mgemv_rows<5, R, WPB><<<tot, WPB * 32, 0, s>>>(g, nr); break;
        case 6: k_mgemv_rows<6, R, WPB><<<tot, WPB * 32, 0, s>>>(g, nr); break;
        case 8: k_mgemv_rows<8, R, WPB><<<tot, WPB * 32, 0, s>>>(g, nr); break;
        default: std::fprintf(stderr, "mgemv_rows: unsupported bits %d\n", bits); std::abort();
    }
}
void mgemv_rows(GDs& g, int bits, int nr, cudaStream_t s) {
    static const int var = std::getenv("QV_MGR") ? std::atoi(std::getenv("QV_MGR")) : 0;
    switch (var) {
        case 1: mgemv_rows_t<1, 2>(g, bits, nr, s); break;
        case 2: mgemv_rows_t<1, 4>(g, bits, nr, s); break;
        case 3: mgemv_rows_t<2, 1>(g, bits, nr, s); break;
        case 4: mgemv_rows_t<2, 2>(g, bits, nr, s); break;
        default: mgemv_rows_t<MR_R, MR_WPB>(g, bits, nr, s); break;
    }
}

// ---- union of the experts of all rows. ids [rows][8] -> uni_e[u], uni_cnt[u], uni_pair[u][8] (= row*8+slot), *uni_n
__global__ void k_moe_union(const int* ids, int rows, int* uni_e, int* uni_cnt, int* uni_pair, int* uni_n) {
    __shared__ int flag[NE], idx[NE];
    const int e = threadIdx.x;
    int cnt = 0, pr[8];
    for (int i = 0; i < rows * TOPK; ++i)
        if (ids[i] == e && cnt < 8) pr[cnt++] = i;
    flag[e] = cnt > 0;
    __syncthreads();
    if (e == 0) {
        int n = 0;
        for (int k = 0; k < NE; ++k) { idx[k] = n; n += flag[k]; }
        *uni_n = n;
    }
    __syncthreads();
    if (cnt > 0) {
        const int u = idx[e];
        uni_e[u] = e;
        uni_cnt[u] = cnt;
        for (int k = 0; k < cnt; ++k) uni_pair[u * 8 + k] = pr[k];
    }
}
void moe_union(const int* ids, int rows, int* uni_e, int* uni_cnt, int* uni_pair, int* uni_n, cudaStream_t s) {
    k_moe_union<<<1, NE, 0, s>>>(ids, rows, uni_e, uni_cnt, uni_pair, uni_n);
}


// Reduce 4 per-lane values over the warp: transposition for the first two xor stages (offsets 16, 8: 3 shuffles), plain xor stages for 4, 2, 1. Bit-identical to four
// warp_sum calls (same xor tree, same operand pairs; float addition is commutative) with 6 shuffles instead of 20. Lane l returns the full sum of a[2 * bit4 + bit3].
__device__ __forceinline__ float warp_reduce4(float a0, float a1, float a2, float a3) {
    const int lane = threadIdx.x & 31;
    const bool h4 = (lane & 16) != 0, h3 = (lane & 8) != 0;
    float k0 = h4 ? a2 : a0, k1 = h4 ? a3 : a1;
    const float s0 = h4 ? a0 : a2, s1 = h4 ? a1 : a3;
    k0 += __shfl_xor_sync(0xffffffffu, s0, 16);
    k1 += __shfl_xor_sync(0xffffffffu, s1, 16);
    float v = (h3 ? k1 : k0) + __shfl_xor_sync(0xffffffffu, h3 ? k0 : k1, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}
template <int BITS>
__device__ __forceinline__ float qchunk_wb(const uint32_t (&w)[BITS], const int (&xw)[8], float dx, float xs, int qs, float sc, float mn, float acc) {
    constexpr int ZP = 1 << (BITS - 1);
    int d = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) d = dp4a_us(qword<BITS>(w, b), xw[b], d);
    d -= ZP * qs;
    return fmaf(sc * dx, (float)d, fmaf(fmaf(sc, (float)ZP, mn), xs, acc));
}

// w13 over the union: block (row block of 64 rows, expert u); per chunk the 4 weight rows of a warp are loaded once and applied to every pair
template <int BITS>
__global__ void __launch_bounds__(512) k_w13_u(ExpertW W, Q8Vec x, const int* uni_e, const int* uni_cnt, const int* uni_pair, const int* uni_n, Q8Buf act,
                                                float gcap, float ucap) {
    __shared__ float actv[8][32];
    const int rb = blockIdx.x, u = blockIdx.y;
    if (u >= *uni_n) return;
    const int e = uni_e[u], cnt = uni_cnt[u];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = rb * 64 + warp * 4;
    const size_t C = W.K >> 5, G = W.K >> W.gshift;
    const int GS = W.gshift - 5;
    const uint32_t* qb = W.q + (size_t)e * W.N * C * BITS;
    const uint16_t* smb = W.sm8 + (size_t)e * W.N * G;
    const __half* rsb = W.rs + (size_t)e * W.N;
    float acc[4][8], rsc[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        rsc[r] = __half2float(rsb[row0 + r]) / 255.0f;
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[r][t] = 0.f;
    }
    int prow[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) prow[t] = t < cnt ? uni_pair[u * 8 + t] >> 3 : 0;
    for (size_t c = lane; c < C; c += 32) {
        uint32_t w[4][BITS];
        float sc[4], mn[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const uint32_t* wp = qb + ((size_t)(row0 + r) * C + c) * BITS;
#pragma unroll
            for (int b = 0; b < BITS; ++b) w[r][b] = __ldg(wp + b);
            sm8_decode(smb[(size_t)(row0 + r) * G + (c >> GS)], rsc[r], W.zlo, W.zstep, sc[r], mn[r]);
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < cnt) {
                int xw[8], qs;
                float dx, xs;
                Q8Vec xv = x;
                xv.q += (size_t)prow[t] * C * 8; xv.d += (size_t)prow[t] * C; xv.sum += (size_t)prow[t] * C; xv.qsum += (size_t)prow[t] * C;
                load_x(xv, c, xw, dx, xs, qs);
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[r][t] = qchunk_wb<BITS>(w[r], xw, dx, xs, qs, sc[r], mn[r], acc[r][t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float mine = warp_reduce4(acc[0][t], acc[1][t], acc[2][t], acc[3][t]);  // lane: row r = 2 * bit4 + bit3 (gate = even r, up = odd r)
        const float other = __shfl_xor_sync(0xffffffffu, mine, 8);
        if ((lane & 15) == 0 && t < cnt) {  // lanes 0 and 16: gate of pair p = lane >> 4
            const float gv = mine, uv = other;
            const float gc = gcap * tanhf(gv / gcap), uc = ucap * tanhf(uv / ucap);
            actv[t][warp * 2 + (lane >> 4)] = gc * (1.f / (1.f + __expf(-gv))) * uc;
        }
    }
    __syncthreads();
    if (warp < cnt) q8_store_chunk(actv[warp][lane], uni_pair[u * 8 + warp] * (W.N / 64) + rb, act);
}
void moe_w13_rows(const ExpertW& W, const Q8Vec& x, const int* uni_e, const int* uni_cnt, const int* uni_pair, const int* uni_n, int max_u, Q8Buf act,
                  float gcap, float ucap, cudaStream_t s) {
    const dim3 grid(W.N / 64, max_u);
    switch (W.bits) {
        case 3: k_w13_u<3><<<grid, 512, 0, s>>>(W, x, uni_e, uni_cnt, uni_pair, uni_n, act, gcap, ucap); return;
        case 4: k_w13_u<4><<<grid, 512, 0, s>>>(W, x, uni_e, uni_cnt, uni_pair, uni_n, act, gcap, ucap); return;
        case 5: k_w13_u<5><<<grid, 512, 0, s>>>(W, x, uni_e, uni_cnt, uni_pair, uni_n, act, gcap, ucap); return;
        default: break;
    }
    std::fprintf(stderr, "moe_w13_rows: unsupported bits %d\n", W.bits);
    std::abort();
}

// w2 over the union, expert-grouped: block (W2G_WPB warps x 4 output rows, expert u); each warp loads its 4 weight rows once per chunk and applies them to every
// pair (row, slot) that selected the expert; per-pair arithmetic is identical to the 1-row k_w2 (same chunk order, same fma chain). Writes ys[pair][N].
constexpr int W2G_WPB = 8;
template <int BITS>
__global__ void __launch_bounds__(W2G_WPB * 32) k_w2_g(ExpertW W, Q8Vec act, const int* uni_e, const int* uni_cnt, const int* uni_pair, const int* uni_n, float* ys) {
    const int u = blockIdx.y;
    if (u >= *uni_n) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * W2G_WPB + warp) * 4;
    if (row0 >= W.N) return;
    const int e = uni_e[u], cnt = uni_cnt[u];
    const size_t C = W.K >> 5, G = W.K >> W.gshift;
    const int GS = W.gshift - 5;
    const uint32_t* qb = W.q + (size_t)e * W.N * C * BITS;
    const uint16_t* smb = W.sm8 + (size_t)e * W.N * G;
    const __half* rsb = W.rs + (size_t)e * W.N;
    float acc[4][8], rsc[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        rsc[r] = __half2float(rsb[row0 + r]) / 255.0f;
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[r][t] = 0.f;
    }
    int pair[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) pair[t] = t < cnt ? uni_pair[u * 8 + t] : 0;
    for (size_t c = lane; c < C; c += 32) {
        uint32_t w[4][BITS];
        float sc[4], mn[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const uint32_t* wp = qb + ((size_t)(row0 + r) * C + c) * BITS;
#pragma unroll
            for (int b = 0; b < BITS; ++b) w[r][b] = __ldg(wp + b);
            sm8_decode(smb[(size_t)(row0 + r) * G + (c >> GS)], rsc[r], W.zlo, W.zstep, sc[r], mn[r]);
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < cnt) {
                int xw[8], qs;
                float dx, xs;
                load_x(act, (size_t)pair[t] * C + c, xw, dx, xs, qs);
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[r][t] = qchunk_wb<BITS>(w[r], xw, dx, xs, qs, sc[r], mn[r], acc[r][t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float mine = warp_reduce4(acc[0][t], acc[1][t], acc[2][t], acc[3][t]);
        if ((lane & 7) == 0 && t < cnt) ys[(size_t)pair[t] * W.N + row0 + (lane >> 3)] = mine;  // lane>>3 = 2 * bit4 + bit3
    }
}
__global__ void k_w2_combine(const float* ys, const float* wts, float* y, int N) {
    const int row = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) o = fmaf(wts[row * TOPK + k], ys[((size_t)row * TOPK + k) * N + i], o);
    y[(size_t)row * N + i] = o;
}
void moe_w2_rows(const ExpertW& W, const Q8Vec& act, const int* uni_e, const int* uni_cnt, const int* uni_pair, const int* uni_n, int max_u, float* ys,
                 const float* wts, float* y, int rows, cudaStream_t s) {
    const dim3 grid((W.N / 4 + W2G_WPB - 1) / W2G_WPB, max_u);
    switch (W.bits) {
        case 3: k_w2_g<3><<<grid, W2G_WPB * 32, 0, s>>>(W, act, uni_e, uni_cnt, uni_pair, uni_n, ys); break;
        case 4: k_w2_g<4><<<grid, W2G_WPB * 32, 0, s>>>(W, act, uni_e, uni_cnt, uni_pair, uni_n, ys); break;
        case 5: k_w2_g<5><<<grid, W2G_WPB * 32, 0, s>>>(W, act, uni_e, uni_cnt, uni_pair, uni_n, ys); break;
        default: std::fprintf(stderr, "moe_w2_rows: unsupported bits %d\n", W.bits); std::abort();
    }
    k_w2_combine<<<dim3((W.N + 255) / 256, rows), 256, 0, s>>>(ys, wts, y, W.N);
}

// ---- verify commit: nc = *n_commit rows are accepted
// conv state [apps][3][D] = last 3 of [old 3, hist rows 0..nc-1]; hist [apps][rs][D]
__global__ void k_conv_commit_n(const float* hist, int rs, float* st, int D, const int* n_commit) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, app = blockIdx.y;
    if (c >= D) return;
    const int T = *n_commit;
    float* s = st + (size_t)app * 3 * D;
    const float* hh = hist + (size_t)app * rs * D;
    float old[3] = {s[c], s[D + c], s[2 * D + c]}, nw[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = T + i;
        nw[i] = idx < 3 ? old[idx] : hh[(size_t)(idx - 3) * D + c];
    }
    s[c] = nw[0]; s[D + c] = nw[1]; s[2 * D + c] = nw[2];
}
void conv_commit_n(const float* hist, int rs, float* st, int D, int apps, const int* n_commit, cudaStream_t s) {
    k_conv_commit_n<<<dim3((D + 255) / 256, apps), 256, 0, s>>>(hist, rs, st, D, n_commit);
}
// ring commit: ring[(pos0 + r) & 1023] = tail[r] for r < nc, all local slots; tail [slots][8][tail_rows][128]
__global__ void k_ring_commit(__half* kring, __half* vring, const __half* ktail, const __half* vtail, int tail_rows, const int* pos0, const int* n_commit) {
    const int slot = blockIdx.y, g = blockIdx.x, r = blockIdx.z, c = threadIdx.x;
    if (r >= *n_commit) return;
    const size_t t = (((size_t)slot * NKV + g) * tail_rows + r) * HD + c;
    const size_t d = (((size_t)slot * NKV + g) * 1024 + ((*pos0 + r) & 1023)) * HD + c;
    kring[d] = ktail[t];
    vring[d] = vtail[t];
}
void ring_commit(__half* kring, __half* vring, const __half* ktail, const __half* vtail, int tail_rows, int nslots, int rows, const int* pos0,
                 const int* n_commit, cudaStream_t s) {
    k_ring_commit<<<dim3(NKV, nslots, rows), HD, 0, s>>>(kring, vring, ktail, vtail, tail_rows, pos0, n_commit);
}

// ---------------------------------------------------------------------------------------------------------------------------
// MTP drafter kernels
__global__ void __launch_bounds__(NT) k_mix_q8(const float* feat, const float* alpha, const float* nw, Q8Buf q8, float* out, const int* last_commit) {
    __shared__ float sm[32];
    const int t = threadIdx.x, warp = t >> 5;
    const int row = last_commit ? *last_commit - 1 : blockIdx.x;
    float v[3], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = t + NT * j;
        float m = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) m = fmaf(feat[((size_t)row * 4 + k) * H + i], alpha[k], m);
        v[j] = m;
        ss += m * m;
    }
    const float r = block_sum(ss, sm);
    const float f = rsqrtf(r / H + 1e-6f);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = t + NT * j;
        const float xn = nw[i] * (v[j] * f);
        if (out) out[(size_t)(last_commit ? 0 : blockIdx.x) * H + i] = xn;
        if (q8.q) q8_store_chunk(xn, warp + NW * j, q8_stream(q8, blockIdx.x));
    }
}
void mix_q8(const float* feat, int rows, const float* alpha, const float* nw, Q8Buf q8, float* out, const int* last_commit, cudaStream_t s) {
    k_mix_q8<<<last_commit ? 1 : rows, NT, 0, s>>>(feat, alpha, nw, q8, out, last_commit);
}

__device__ __forceinline__ void rope_pair(float& a, float& b, int c, double pos) {  // neox: (c, c+64), theta 1e6
    const double inv = exp(-13.815510557964274 * (2.0 * c) / 128.0);
    double sn, cs;
    sincos(pos * inv, &sn, &cs);
    const float x1 = a, x2 = b, cf = (float)cs, sf = (float)sn;
    a = x1 * cf - x2 * sf;
    b = x2 * cf + x1 * sf;
}
__global__ void __launch_bounds__(128) k_ctx_post(const float* kraw, const float* vraw, const float* knw, __half* kc, __half* vc, int cap, int ring, int pos0_val,
                                                   const int* pos0_ptr, const int* ncommit, int rows) {
    __shared__ float red[32], xs[128];
    const int row = blockIdx.x, g = blockIdx.y, c = threadIdx.x;
    if (ncommit && row >= *ncommit) return;
    const int pos = (pos0_ptr ? *pos0_ptr : pos0_val) + row;
    const float x = kraw[(size_t)row * 1024 + g * HD + c];
    const float ms = block_sum(x * x, red);
    xs[c] = x * rsqrtf(ms / HD + 1e-6f) * knw[c];
    __syncthreads();
    const int slot = ring ? (pos & (cap - 1)) : pos;
    const size_t base = ((size_t)g * cap + slot) * HD;
    if (c < 64) {
        float a = xs[c], b = xs[c + 64];
        rope_pair(a, b, c, (double)pos);
        kc[base + c] = __float2half(a);
        kc[base + c + 64] = __float2half(b);
    }
    vc[base + c] = __float2half(vraw[(size_t)row * 1024 + g * HD + c]);
}
void ctx_post(const float* kraw, const float* vraw, const float* knw, __half* kc, __half* vc, int cap, int ring, int pos0_val, const int* pos0_ptr,
              const int* ncommit, int rows, cudaStream_t s) {
    k_ctx_post<<<dim3(rows, NKV), 128, 0, s>>>(kraw, vraw, knw, kc, vc, cap, ring, pos0_val, pos0_ptr, ncommit, rows);
}

__global__ void __launch_bounds__(128) k_draft_qk(const float* qraw, const float* kraw, const float* vraw, const float* qnw, const float* knw, const int* pos0, float* qs,
                                                   __half* ktail, __half* vtail) {
    __shared__ float red[32], xs[128];
    const int row = blockIdx.x, hd = blockIdx.y, c = threadIdx.x;
    const int pos = *pos0 + row;
    if (hd < NH + NKV) {
        const bool isq = hd < NH;
        const float x = isq ? qraw[(size_t)row * (NH * HD) + hd * HD + c] : kraw[(size_t)row * 1024 + (hd - NH) * HD + c];
        const float ms = block_sum(x * x, red);
        xs[c] = x * rsqrtf(ms / HD + 1e-6f) * (isq ? qnw[c] : knw[c]);
        __syncthreads();
        if (c < 64) {
            float a = xs[c], b = xs[c + 64];
            rope_pair(a, b, c, (double)pos);
            if (isq) {
                const float sc = 0.08838834764831845f;
                qs[(size_t)row * (NH * HD) + hd * HD + c] = a * sc;
                qs[(size_t)row * (NH * HD) + hd * HD + c + 64] = b * sc;
            } else {
                ktail[((size_t)(hd - NH) * 8 + row) * HD + c] = __float2half(a);
                ktail[((size_t)(hd - NH) * 8 + row) * HD + c + 64] = __float2half(b);
            }
        }
    } else {
        const int g = hd - NH - NKV;
        vtail[((size_t)g * 8 + row) * HD + c] = __float2half(vraw[(size_t)row * 1024 + g * HD + c]);
    }
}
void draft_qk(const float* qraw, const float* kraw, const float* vraw, const float* qnw, const float* knw, const int* pos0, float* qs, __half* ktail, __half* vtail,
              cudaStream_t s) {
    k_draft_qk<<<dim3(8, NH + 2 * NKV), 128, 0, s>>>(qraw, kraw, vraw, qnw, knw, pos0, qs, ktail, vtail);
}

__global__ void __launch_bounds__(128) k_draft_combine(AttnArgs a) {
    const int h = blockIdx.x, g = h / 3, hh = h % 3, d = threadIdx.x, nsp = a.nsplit, row = blockIdx.y;
    const float* partr = a.part + (size_t)row * (NKV * nsp * 3 * PART_STRIDE);
    float M = -1e30f;
    for (int s = 0; s < nsp; ++s) {
        const float* p = partr + ((size_t)(g * nsp + s) * 3 + hh) * PART_STRIDE;
        if (p[1] > 0.f) M = fmaxf(M, p[0]);
    }
    float L = 0.f, o = 0.f;
    for (int s = 0; s < nsp; ++s) {
        const float* p = partr + ((size_t)(g * nsp + s) * 3 + hh) * PART_STRIDE;
        if (p[1] > 0.f) {
            const float e = __expf(p[0] - M);
            L += e * p[1];
            o += e * p[2 + d];
        }
    }
    Q8Buf ob = a.out;
    ob.q += (size_t)row * a.q8_rs * 8; ob.d += row * a.q8_rs; ob.sum += row * a.q8_rs; ob.qsum += row * a.q8_rs;
    q8_store_chunk(o / L, h * 4 + (d >> 5), ob);
}
void draft_combine(const AttnArgs& a, cudaStream_t s) { k_draft_combine<<<dim3(NH, a.rows), 128, 0, s>>>(a); }

__global__ void k_add_rows(float* x, const float* y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
void add_rows(float* x, const float* y, long n, cudaStream_t s) { k_add_rows<<<(n + 255) / 256, 256, 0, s>>>(x, y, n); }

__global__ void __launch_bounds__(1024) k_silu_mul_q8(const float* g, const float* u, int n, Q8Buf out) {
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = blockIdx.y * 32 + warp;
    const size_t i = (size_t)row * n + (size_t)c * 32 + lane;
    const float gv = g[i];
    const float v = gv / (1.f + expf(-gv)) * u[i];
    const int ch = n >> 5;
    Q8Buf o = out;
    o.q += (size_t)row * ch * 8; o.d += row * ch; o.sum += row * ch; o.qsum += row * ch;
    q8_store_chunk(v, c, o);
}
void silu_mul_q8(const float* g, const float* u, int rows, int n, Q8Buf out, cudaStream_t s) {
    k_silu_mul_q8<<<dim3(rows, (n / 32) / 32), 1024, 0, s>>>(g, u, n, out);
}
__global__ void k_fill_mask(float* x, const float* mask) {
    const int row = blockIdx.x + 1;
    for (int i = threadIdx.x; i < H; i += blockDim.x) x[(size_t)row * H + i] = mask[i];
}
void fill_mask_rows(float* x, const float* mask, int rows, cudaStream_t s) { k_fill_mask<<<rows - 1, 256, 0, s>>>(x, mask); }
__global__ void k_mk_inputs(const float* xn, const float* f, float* xcat) {
    const int j = blockIdx.x;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        xcat[(size_t)j * 2 * H + i] = xn[(size_t)(j + 1) * H + i];
        xcat[(size_t)j * 2 * H + H + i] = f[i];
    }
}
void mk_inputs(const float* xn, const float* f_rnn, float* xcat, int rows, cudaStream_t s) { k_mk_inputs<<<rows, 256, 0, s>>>(xn, f_rnn, xcat); }
__global__ void k_copy_f(const float* x, float* d, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = x[i];
}
void embed_row(const float* x, float* dst, int n, cudaStream_t s) { k_copy_f<<<(n + 255) / 256, 256, 0, s>>>(x, dst, n); }

template <int BITS>
__device__ __forceinline__ void w1_row(const QMat& E, size_t row, int warp, int lane, float* dst) {
    dst[warp * 32 + lane] = dequant_elem<BITS>(E, row, warp, lane, 8);
}
__global__ void __launch_bounds__(768) k_mk_step(QMat w1, const __half* wa, const float* ctxterm, const float* bias, const int* prev, float* state, Q8Buf q8lat,
                                                  int first) {
    __shared__ float in[512], raw[768], lat[256];
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    if (t < 256) in[t] = first ? 0.f : state[t];
    if (warp < 8) {
        const size_t row = (size_t)*prev;
        switch (w1.bits) {
            case 4: w1_row<4>(w1, row, warp, lane, in + 256); break;
            case 5: w1_row<5>(w1, row, warp, lane, in + 256); break;
            case 6: w1_row<6>(w1, row, warp, lane, in + 256); break;
            default: w1_row<8>(w1, row, warp, lane, in + 256); break;
        }
    }
    __syncthreads();
    {
        const uint4* wp = reinterpret_cast<const uint4*>(wa + (size_t)t * 512);
        float acc = 0.f;
        for (int i = 0; i < 64; ++i) {
            const uint4 u = __ldg(wp + i);
            const __half2* hh = reinterpret_cast<const __half2*>(&u);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 f = __half22float2(hh[k]);
                acc = fmaf(f.x, in[i * 8 + 2 * k], fmaf(f.y, in[i * 8 + 2 * k + 1], acc));
            }
        }
        raw[t] = acc + ctxterm[t] + bias[t];
    }
    __syncthreads();
    if (t < 256) {
        const float gt = 1.f / (1.f + expf(-raw[t]));
        const float st = gt * in[t] + (1.f - gt) * tanhf(raw[256 + t]);
        state[t] = st;
        lat[t] = tanhf(raw[512 + t]);
    }
    __syncthreads();
    if (warp < 8) q8_store_chunk(lat[warp * 32 + lane], warp, q8lat);
}
void mk_step(const QMat& w1, const __half* wa, const float* ctxterm_j, const float* bias, const int* prev, float* state, Q8Buf q8lat, int first, cudaStream_t s) {
    k_mk_step<<<1, 768, 0, s>>>(w1, wa, ctxterm_j, bias, prev, state, q8lat, first);
}

constexpr int ARGB = 196;
__global__ void __launch_bounds__(256) k_aa_part(const float* base, const float* bias, const float* fb, int n, float* sv, int* si) {
    float best = -3.0e38f;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += ARGB * 256) {
        const float v = base[i] + bias[i] + fb[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    __shared__ float bv[256];
    __shared__ int bidx[256];
    bv[threadIdx.x] = best; bidx[threadIdx.x] = bi;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            const float v2 = bv[threadIdx.x + o]; const int i2 = bidx[threadIdx.x + o];
            if (v2 > bv[threadIdx.x] || (v2 == bv[threadIdx.x] && i2 < bidx[threadIdx.x])) { bv[threadIdx.x] = v2; bidx[threadIdx.x] = i2; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { sv[blockIdx.x] = bv[0]; si[blockIdx.x] = bidx[0]; }
}
__global__ void __launch_bounds__(256) k_aa_final(const float* sv, const int* si, int* tok) {
    __shared__ float bv[256];
    __shared__ int bidx[256];
    float best = -3.0e38f; int bi = 0x7fffffff;
    if (threadIdx.x < ARGB) { best = sv[threadIdx.x]; bi = si[threadIdx.x]; }
    bv[threadIdx.x] = best; bidx[threadIdx.x] = bi;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            const float v2 = bv[threadIdx.x + o]; const int i2 = bidx[threadIdx.x + o];
            if (v2 > bv[threadIdx.x] || (v2 == bv[threadIdx.x] && i2 < bidx[threadIdx.x])) { bv[threadIdx.x] = v2; bidx[threadIdx.x] = i2; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *tok = bidx[0];
}
void add_argmax(const float* base, const float* bias, const float* fold_bias, int n, float* sv, int* si, int* tok, cudaStream_t s) {
    k_aa_part<<<ARGB, 256, 0, s>>>(base, bias, fold_bias, n, sv, si);
    k_aa_final<<<1, 256, 0, s>>>(sv, si, tok);
}

// st: [0]=anchor token, [1]=pos0, [2]=n_commit, [3]=n generated so far (index into gen), [4]=cycles, [5]=accepted drafts
__global__ void k_accept(const int* vtok, const int* vy, int R, int* st, int* gen, int* hist) {
    int n = 0;
    while (n < R - 1 && vtok[n + 1] == vy[n]) ++n;
    const int g0 = st[3];
    for (int i = 0; i < n; ++i) gen[g0 + i] = vtok[1 + i];
    gen[g0 + n] = vy[n];
    st[3] = g0 + n + 1;
    st[2] = n + 1;
    st[0] = vy[n];  // new anchor (not yet run through the target)
    st[4] += 1;
    st[5] += n;
    hist[n] += 1;
}
__global__ void k_advance(int* st) { st[1] += st[2]; }
void advance_pos(int* st, cudaStream_t s) { k_advance<<<1, 1, 0, s>>>(st); }
void accept_rows(const int* vtok, const int* vy, int R, int* st, int* gen, int* nacc_hist, cudaStream_t s) {
    k_accept<<<1, 1, 0, s>>>(vtok, vy, R, st, gen, nacc_hist);
}


// ---- sampled drafts (lossless speculative sampling at T > 0)
__global__ void k_draft_logits(const float* base, const float* bias, const float* fb, int n, float cap, float* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = cap * tanhf((base[i] + bias[i] + fb[i]) / cap);
}
void draft_logits(const float* base, const float* bias, const float* fold_bias, int n, float cap, float* out, cudaStream_t s) {
    k_draft_logits<<<(n + 255) / 256, 256, 0, s>>>(base, bias, fold_bias, n, cap, out);
}
// one block of 256 threads: sort the extracted candidates (bitonic, descending, ties by id), apply top_k / top_p / min_p / temperature exactly like
// qv::spec_sampling (build_dist_from_candidates), inverse-CDF draw with a counter-based hash of (*seed_ptr, step).
__global__ void __launch_bounds__(256) k_draft_sample(const int* cid, const float* cval, const CandMeta* meta, DraftSampleParams p, const unsigned long long* seed_ptr, int step,
                                                       int* q_id, float* q_prob, int* q_cnt, int* tok) {
    __shared__ int s_id[256];
    __shared__ float s_val[256];
    __shared__ float s_lo, s_hi;
    __shared__ int s_bcnt, s_out;
    const int tid = threadIdx.x;
    const int raw = meta->count;
    const int n_in = raw > 0 ? (raw < p.max_extract ? raw : p.max_extract) : 0;
    int cnt = n_in;
    if (n_in <= 256) {
        s_id[tid] = tid < n_in ? cid[tid] : 0x7fffffff;
        s_val[tid] = tid < n_in ? cval[tid] : -INFINITY;
    } else {  // more than 256 candidates: keep (about) the 256 largest by bisecting a value threshold
        if (tid == 0) { s_lo = meta->mx - p.margin; s_hi = meta->mx; s_out = 0; }
        s_id[tid] = 0x7fffffff;
        s_val[tid] = -INFINITY;
        __syncthreads();
        for (int it = 0; it < 10; ++it) {
            if (tid == 0) s_bcnt = 0;
            __syncthreads();
            const float mid = 0.5f * (s_lo + s_hi);
            int lc = 0;
            for (int i = tid; i < n_in; i += 256) if (cval[i] >= mid) ++lc;
            if (lc) atomicAdd(&s_bcnt, lc);
            __syncthreads();
            if (tid == 0) { if (s_bcnt > 256) s_lo = mid; else s_hi = mid; }
            __syncthreads();
        }
        const float thr = s_hi;
        for (int i = tid; i < n_in; i += 256) {
            const float v = cval[i];
            if (v >= thr) { const int pos = atomicAdd(&s_out, 1); if (pos < 256) { s_id[pos] = cid[i]; s_val[pos] = v; } }
        }
        __syncthreads();
        cnt = s_out < 256 ? s_out : 256;
    }
    __syncthreads();
    for (int k = 2; k <= 256; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const bool desc = ((tid & k) == 0);
                const float va = s_val[tid], vb = s_val[ixj];
                const int ia = s_id[tid], ib = s_id[ixj];
                const bool a_gt_b = (va > vb) || (va == vb && ia < ib);
                if (desc ? !a_gt_b : a_gt_b) { s_val[tid] = vb; s_val[ixj] = va; s_id[tid] = ib; s_id[ixj] = ia; }
            }
            __syncthreads();
        }
    }
    if (tid != 0) return;
    if (cnt == 0 || !isfinite(s_val[0])) { q_id[0] = 0; q_prob[0] = 1.0f; *q_cnt = 1; *tok = 0; return; }
    int keep = cnt;
    const int min_k = p.min_keep > 1 ? p.min_keep : 1;
    if (p.top_k > 0) { const int k = p.top_k > min_k ? p.top_k : min_k; if (k < keep) keep = k; }
    const double cmax = (double)s_val[0];
    if (p.top_p < 1.0f && keep > min_k) {
        double denom = 0.0;
        for (int i = 0; i < keep; ++i) denom += exp((double)s_val[i] - cmax);
        double cum = 0.0;
        for (int i = 0; i < keep; ++i) {
            cum += exp((double)s_val[i] - cmax) / denom;
            if (cum >= (double)p.top_p && i + 1 >= min_k) { keep = i + 1; break; }
        }
    }
    if (p.min_p > 0.0f) {
        int kp = 0;
        while (kp < keep && exp((double)s_val[kp] - cmax) >= (double)p.min_p) ++kp;
        const int mk = keep < min_k ? keep : min_k;
        keep = kp > mk ? kp : mk;
    }
    if (p.temperature <= 0.0f) { q_id[0] = s_id[0]; q_prob[0] = 1.0f; *q_cnt = 1; *tok = s_id[0]; return; }
    double sum_w = 0.0;
    const double inv_t = 1.0 / (double)p.temperature;
    double w[256];
    for (int i = 0; i < keep; ++i) { w[i] = exp(((double)s_val[i] - cmax) * inv_t); sum_w += w[i]; }
    unsigned long long z = *seed_ptr + 0x9e3779b97f4a7c15ull * (unsigned long long)(step + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z = z ^ (z >> 31);
    const double u = (double)(z >> 11) * (1.0 / 9007199254740992.0);
    double total = 0.0;
    for (int i = 0; i < keep; ++i) { const float pr = (float)(w[i] / sum_w); q_id[i] = s_id[i]; q_prob[i] = pr; total += (double)pr; }
    const double target = u * total;
    double cdf = 0.0;
    int chosen = s_id[keep - 1];
    for (int i = 0; i < keep; ++i) { cdf += (double)q_prob[i]; if (target < cdf) { chosen = s_id[i]; break; } }
    *q_cnt = keep;
    *tok = chosen;
}
void draft_sample(const float* logits, int n, const DraftSampleParams& p, const unsigned long long* seed_ptr, int step, int* cand_id, float* cand_val, void* cand_meta,
                  int* q_id, float* q_prob, int* q_cnt, int* tok, cudaStream_t s) {
    CandMeta* meta = (CandMeta*)cand_meta;
    extract_top_candidates_batch(logits, n, 1, p.margin, p.max_extract, cand_id, cand_val, meta, s);
    k_draft_sample<<<1, 256, 0, s>>>(cand_id, cand_val, meta, p, seed_ptr, step, q_id, q_prob, q_cnt, tok);
}

}  // namespace qv::cuda::ink
