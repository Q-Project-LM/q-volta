#include "../common.cuh"
#include "elementwise.h"
#include "q8.cuh"

namespace qv::cuda {

// out = x * rsqrt(mean(x^2)+eps) * w   (w already holds 1+w for zero-centred norms); also emits xsum32(out) or
// the fused int8 GEMV input. Multi-block: every block recomputes the (L2-resident) sum of squares, then handles its
// own slice of 32-chunks — trades redundant reads of n floats for parallelism (latency-bound single-block before).
constexpr int NORM_CHUNKS_PER_BLOCK = 8;
__global__ void __launch_bounds__(256) rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                      float* __restrict__ out, float* __restrict__ xs32, int n,
                                                      float eps, Q8Buf q8) {
    __shared__ float red[32];
    x += (size_t)blockIdx.y * n;  // token row (nt > 1: MTP verify); q8 chunks continue token-major
    out += (size_t)blockIdx.y * n;
    float ss = 0.f;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        float4 v = x4[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / n + eps);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int c = blockIdx.x * NORM_CHUNKS_PER_BLOCK + wid;  // one warp per chunk
    if (wid >= NORM_CHUNKS_PER_BLOCK || c >= n / 32) return;
    const int i = c * 32 + lane;
    const float v = x[i] * r * w[i];
    out[i] = v;
    if (q8.q) {
        q8_store_chunk(v, c + blockIdx.y * (n / 32), q8);
    } else {
        float s = warp_sum(v);
        if (xs32 && lane == 0) xs32[c] = s;
    }
}

void rmsnorm(const float* x, const float* w, float* out, float* xs32, int n, float eps, cudaStream_t s, const Q8Buf* q8,
             int nt) {
    const int blocks = (n / 32 + NORM_CHUNKS_PER_BLOCK - 1) / NORM_CHUNKS_PER_BLOCK;
    rmsnorm_kernel<<<dim3(blocks, nt), 32 * NORM_CHUNKS_PER_BLOCK, 0, s>>>(x, w, out, xs32, n, eps, q8 ? *q8 : Q8Buf{});
}

// Dequantize one embedding row (token id read from device memory) into out[K].
__global__ void embed_kernel(QMat E, const int* __restrict__ tok, float* __restrict__ out) {
    const int row = *tok;
    const int C = E.K >> 5, G = E.K >> E.gshift, GS = E.gshift - 5;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < E.K; k += gridDim.x * blockDim.x) {
        int c = k >> 5, i = k & 31;
        const uint32_t* w = E.q + ((size_t)row * C + c) * E.bits;
        uint32_t q;
        // generic (slow path is fine: 5120 elements once per token)
        switch (E.bits) {
            case 2: q = (w[i & 1] >> (2 * (i >> 1))) & 3u; break;
            case 3: q = ((w[i & 1] >> (2 * (i >> 1))) & 3u) | (((w[2] >> i) & 1u) << 2); break;
            case 4: q = (w[i & 3] >> (4 * (i >> 2))) & 15u; break;
            case 5: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4] >> i) & 1u) << 4); break;
            case 6: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 1)] >> (2 * (i >> 1))) & 3u) << 4); break;
            default: q = ((w[i & 3] >> (4 * (i >> 2))) & 15u) | (((w[4 + (i & 3)] >> (4 * (i >> 2))) & 15u) << 4);
        }
        float2 sm = load_sm(E, row, G, c >> GS);
        out[k] = sm.x * (float)q + sm.y;
    }
}

void embed(const QMat& E, const int* tok, float* out, cudaStream_t s) {
    embed_kernel<<<(E.K + 255) / 256, 256, 0, s>>>(E, tok, out);
}

// act = silu(g) * u ; also chunk sums for the following quantized GEMV.
__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out,
                                float* __restrict__ xs32, int n, Q8Buf q8) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // n % 32 == 0 → whole warps in range
    if (i >= n) return;
    float v = silu(g[i]) * u[i];
    out[i] = v;
    if (q8.q) { q8_store_chunk(v, i >> 5, q8); return; }
    float s = warp_sum(v);
    if ((threadIdx.x & 31) == 0) xs32[i >> 5] = s;
}

void silu_mul(const float* g, const float* u, float* out, float* xs32, int n, cudaStream_t s, const Q8Buf* q8) {
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(g, u, out, xs32, n, q8 ? *q8 : Q8Buf{});
}

// argmax over n floats, single block (first index wins on ties, like torch.argmax).
__global__ void argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        int nw = blockDim.x >> 5;
        best = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) *out = bi;
    }
}

// Greedy feedback for graph decode: tok = argmax(logits); gen[pos] = tok; pos += 1.
// Two stages: ARG_BLOCKS blocks reduce slices to (value, index) partials; one block finishes + feeds back.
constexpr int ARG_BLOCKS = 240;

__device__ __forceinline__ void argmax_merge(float& best, int& bi, float ov, int oi) {
    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
}

__device__ void block_argmax(float& best, int& bi) {
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1)
        argmax_merge(best, bi, __shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, bi, o));
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        int nw = blockDim.x >> 5;
        best = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1)
            argmax_merge(best, bi, __shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, bi, o));
    }
}

__global__ void argmax_partial_kernel(const float* __restrict__ x, int n, float* pv, int* pi) {
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) argmax_merge(best, bi, x[i], i);
    block_argmax(best, bi);
    if (threadIdx.x == 0) { pv[blockIdx.x] = best; pi[blockIdx.x] = bi; }
}

__global__ void argmax_final_kernel(const float* pv, const int* pi, int m, int* tok, int* pos, int* gen) {
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < m; i += blockDim.x) argmax_merge(best, bi, pv[i], pi[i]);
    block_argmax(best, bi);
    if (threadIdx.x == 0) {
        *tok = bi;
        if (pos) {
            int p = *pos;
            gen[p] = bi;
            *pos = p + 1;
        }
    }
}

static float* g_pv = nullptr;
static int* g_pi = nullptr;

void argmax_advance(const float* x, int n, int* tok, int* pos, int* gen, cudaStream_t s) {
    if (!g_pv) {  // one-time scratch (allocated before any graph capture: first call is eager)
        QV_CUDA_CHECK(cudaMalloc(&g_pv, ARG_BLOCKS * sizeof(float)));
        QV_CUDA_CHECK(cudaMalloc(&g_pi, ARG_BLOCKS * sizeof(int)));
    }
    argmax_partial_kernel<<<ARG_BLOCKS, 256, 0, s>>>(x, n, g_pv, g_pi);
    argmax_final_kernel<<<1, 256, 0, s>>>(g_pv, g_pi, ARG_BLOCKS, tok, pos, gen);
}

void argmax(const float* x, int n, int* out, cudaStream_t s) { argmax_advance(x, n, out, nullptr, nullptr, s); }

__global__ void argmax_partial_batch_kernel(const float* __restrict__ x, int n, float* pv, int* pi) {
    const int row = blockIdx.y;
    const float* xr = x + (size_t)row * n;
    float* pvr = pv + (size_t)row * ARG_BLOCKS;
    int* pir = pi + (size_t)row * ARG_BLOCKS;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) argmax_merge(best, bi, xr[i], i);
    block_argmax(best, bi);
    if (threadIdx.x == 0) { pvr[blockIdx.x] = best; pir[blockIdx.x] = bi; }
}

__global__ void cand_max_final_batch_kernel(const float* pv, const int* pi, int m, CandMeta* meta) {
    const int row = blockIdx.y;
    const float* pvr = pv + (size_t)row * m;
    const int* pir = pi + (size_t)row * m;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < m; i += blockDim.x) argmax_merge(best, bi, pvr[i], pir[i]);
    block_argmax(best, bi);
    if (threadIdx.x == 0) {
        meta[row].mx = best;
        meta[row].sum_exp = 0.0f;
        meta[row].count = 0;
    }
}

__global__ void extract_cand_batch_kernel(const float* __restrict__ x, int n, float margin, int max_cand,
                                          int* __restrict__ cid, float* __restrict__ cval, CandMeta* meta,
                                          float* __restrict__ ps) {
    constexpr int BBUF = 256;
    __shared__ float red[32];
    __shared__ int s_id[BBUF];
    __shared__ float s_val[BBUF];
    __shared__ int s_cnt;
    __shared__ int s_base;
    if (threadIdx.x == 0) { s_cnt = 0; s_base = 0; }
    __syncthreads();

    const int row = blockIdx.y;
    const float* xr = x + (size_t)row * n;
    int* cidr = cid + (size_t)row * max_cand;
    float* cvalr = cval + (size_t)row * max_cand;
    CandMeta* mr = meta + row;

    const float mx = mr->mx;
    const bool ok = isfinite(mx);
    const float thr = mx - margin;
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = xr[i];
        if (ok) {
            local_sum += expf(v - mx);
            if (v >= thr) {
                const int loc = atomicAdd(&s_cnt, 1);
                if (loc < BBUF) {
                    s_id[loc] = i;
                    s_val[loc] = v;
                } else {
                    const int gidx = atomicAdd(&mr->count, 1);
                    if (gidx < max_cand) {
                        cidr[gidx] = i;
                        cvalr[gidx] = v;
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && ok && s_cnt > 0) {
        const int n_loc = min(s_cnt, BBUF);
        s_base = atomicAdd(&mr->count, n_loc);
    }
    __syncthreads();
    if (ok) {
        const int n_loc = min(s_cnt, BBUF);
        if (threadIdx.x < n_loc) {
            const int gidx = s_base + threadIdx.x;
            if (gidx < max_cand) {
                cidr[gidx] = s_id[threadIdx.x];
                cvalr[gidx] = s_val[threadIdx.x];
            }
        }
    }
    local_sum = block_sum(local_sum, red);
    if (threadIdx.x == 0) ps[(size_t)row * gridDim.x + blockIdx.x] = ok ? local_sum : 0.0f;
}

__global__ void sum_exp_final_batch_kernel(const float* __restrict__ ps, int m, CandMeta* meta) {
    const int row = blockIdx.y;
    const float* psr = ps + (size_t)row * m;
    double tot = 0.0;
    for (int i = 0; i < m; ++i) tot += (double)psr[i];
    meta[row].sum_exp = (float)tot;
}

static float* g_pv_b = nullptr;
static int* g_pi_b = nullptr;
static int g_batch_cap = 0;

void extract_top_candidates_batch(const float* x, int n, int n_rows, float margin, int max_cand,
                                  int* cand_id, float* cand_val, CandMeta* meta, cudaStream_t s) {
    if (n_rows <= 0) return;
    if (n_rows > g_batch_cap) {
        if (g_pv_b) cudaFree(g_pv_b);
        if (g_pi_b) cudaFree(g_pi_b);
        QV_CUDA_CHECK(cudaMalloc(&g_pv_b, (size_t)n_rows * ARG_BLOCKS * sizeof(float)));
        QV_CUDA_CHECK(cudaMalloc(&g_pi_b, (size_t)n_rows * ARG_BLOCKS * sizeof(int)));
        g_batch_cap = n_rows;
    }
    argmax_partial_batch_kernel<<<dim3(ARG_BLOCKS, n_rows), 256, 0, s>>>(x, n, g_pv_b, g_pi_b);
    cand_max_final_batch_kernel<<<dim3(1, n_rows), 256, 0, s>>>(g_pv_b, g_pi_b, ARG_BLOCKS, meta);
    extract_cand_batch_kernel<<<dim3(ARG_BLOCKS, n_rows), 256, 0, s>>>(x, n, margin, max_cand, cand_id, cand_val, meta, g_pv_b);
    sum_exp_final_batch_kernel<<<dim3(1, n_rows), 1, 0, s>>>(g_pv_b, ARG_BLOCKS, meta);
}

void extract_top_candidates(const float* x, int n, float margin, int max_cand,
                            int* cand_id, float* cand_val, CandMeta* meta, cudaStream_t s) {
    extract_top_candidates_batch(x, n, 1, margin, max_cand, cand_id, cand_val, meta, s);
}

__global__ void draft_sample_kernel(const int* __restrict__ d_cand_id, const float* __restrict__ d_cand_val,
                                    const CandMeta* __restrict__ d_meta, GpuDraftParams p, uint64_t seed,
                                    int* __restrict__ d_q_id, float* __restrict__ d_q_prob,
                                    int* __restrict__ d_q_cnt, int* __restrict__ d_draft_out) {
    __shared__ int s_id[DRAFT_MAX_CAND];
    __shared__ float s_val[DRAFT_MAX_CAND];
    __shared__ double s_w[DRAFT_MAX_CAND];
    __shared__ int s_bcnt;
    __shared__ int s_out_cnt;
    __shared__ float s_lo, s_hi;

    const int tid = threadIdx.x;
    const int raw_cnt = d_meta->count;
    const int max_ext = p.max_extract > 0 ? p.max_extract : DRAFT_MAX_CAND;
    const int n_in = raw_cnt > 0 ? (raw_cnt < max_ext ? raw_cnt : max_ext) : 0;
    int cnt = n_in;

    if (n_in <= DRAFT_MAX_CAND) {
        s_id[tid] = tid < n_in ? d_cand_id[tid] : 0x7fffffff;
        s_val[tid] = tid < n_in ? d_cand_val[tid] : -INFINITY;
    } else {
        if (tid == 0) {
            s_lo = d_meta->mx - p.margin;
            s_hi = d_meta->mx;
            s_out_cnt = 0;
        }
        s_id[tid] = 0x7fffffff;
        s_val[tid] = -INFINITY;
        __syncthreads();
        for (int it = 0; it < 10; ++it) {
            if (tid == 0) s_bcnt = 0;
            __syncthreads();
            const float mid = 0.5f * (s_lo + s_hi);
            int local_c = 0;
            for (int i = tid; i < n_in; i += DRAFT_MAX_CAND) {
                if (d_cand_val[i] >= mid) ++local_c;
            }
            if (local_c) atomicAdd(&s_bcnt, local_c);
            __syncthreads();
            if (tid == 0) {
                if (s_bcnt > DRAFT_MAX_CAND) s_lo = mid;
                else s_hi = mid;
            }
            __syncthreads();
        }
        const float thr = s_hi;
        for (int i = tid; i < n_in; i += DRAFT_MAX_CAND) {
            const float v = d_cand_val[i];
            if (v >= thr) {
                const int pos = atomicAdd(&s_out_cnt, 1);
                if (pos < DRAFT_MAX_CAND) {
                    s_id[pos] = d_cand_id[i];
                    s_val[pos] = v;
                }
            }
        }
        __syncthreads();
        cnt = s_out_cnt < DRAFT_MAX_CAND ? s_out_cnt : DRAFT_MAX_CAND;
    }
    __syncthreads();

    for (int k = 2; k <= DRAFT_MAX_CAND; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const bool desc = ((tid & k) == 0);
                const float va = s_val[tid], vb = s_val[ixj];
                const int ia = s_id[tid], ib = s_id[ixj];
                const bool a_gt_b = (va > vb) || (va == vb && ia < ib);
                if (desc ? !a_gt_b : a_gt_b) {
                    s_val[tid] = vb;
                    s_val[ixj] = va;
                    s_id[tid] = ib;
                    s_id[ixj] = ia;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        if (cnt == 0 || !isfinite(s_val[0])) {
            d_q_id[0] = 0;
            d_q_prob[0] = 1.0f;
            *d_q_cnt = 1;
            *d_draft_out = 0;
            return;
        }
        int keep = cnt;
        const int min_k = p.min_keep > 1 ? p.min_keep : 1;
        bool trimmed = false;
        if (p.top_k > 0) {
            const int k = p.top_k > min_k ? p.top_k : min_k;
            if (k < keep) {
                keep = k;
                trimmed = true;
            }
        }
        const double cmax = (double)s_val[0];
        if (p.top_p < 1.0f && keep > min_k) {
            double sum_cur = 0.0;
            for (int i = 0; i < keep; ++i) sum_cur += exp((double)s_val[i] - cmax);
            const double denom = (trimmed || p.top_k > 0 || n_in > DRAFT_MAX_CAND || raw_cnt > max_ext)
                                     ? sum_cur
                                     : fmax((double)d_meta->sum_exp, sum_cur);
            if (denom > 0.0 && isfinite(denom)) {
                double cum = 0.0;
                for (int i = 0; i < keep; ++i) {
                    cum += exp((double)s_val[i] - cmax) / denom;
                    if (cum >= (double)p.top_p && i + 1 >= min_k) {
                        keep = i + 1;
                        break;
                    }
                }
            }
        }
        if (p.min_p > 0.0f) {
            int kp = 0;
            while (kp < keep && exp((double)s_val[kp] - cmax) >= (double)p.min_p) ++kp;
            const int mk = keep < min_k ? keep : min_k;
            keep = kp > mk ? kp : mk;
        }
        if (p.temperature <= 0.0f) {
            d_q_id[0] = s_id[0];
            d_q_prob[0] = 1.0f;
            *d_q_cnt = 1;
            *d_draft_out = s_id[0];
            return;
        }
        double sum_w = 0.0;
        const double inv_t = 1.0 / (double)p.temperature;
        for (int i = 0; i < keep; ++i) {
            const double w = exp(((double)s_val[i] - cmax) * inv_t);
            s_w[i] = w;
            sum_w += w;
        }
        if (!(sum_w > 0.0) || !isfinite(sum_w)) {
            d_q_id[0] = s_id[0];
            d_q_prob[0] = 1.0f;
            *d_q_cnt = 1;
            *d_draft_out = s_id[0];
            return;
        }
        uint64_t z = seed + 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z = z ^ (z >> 31);
        const double u = (double)(z >> 11) * (1.0 / 9007199254740992.0);
        const double inv_sum = 1.0 / sum_w;
        double total_f = 0.0;
        for (int i = 0; i < keep; ++i) {
            const float pr = (float)(s_w[i] * inv_sum);
            d_q_id[i] = s_id[i];
            d_q_prob[i] = pr;
            total_f += (double)pr;
        }
        const double target = u * total_f;
        double cdf = 0.0;
        int chosen = s_id[keep - 1];
        for (int i = 0; i < keep; ++i) {
            cdf += (double)d_q_prob[i];
            if (target < cdf) {
                chosen = s_id[i];
                break;
            }
        }
        *d_q_cnt = keep;
        *d_draft_out = chosen;
    }
}

void sample_draft_gpu(const float* logits, int n, const GpuDraftParams& p, uint64_t seed,
                      int* d_cand_id, float* d_cand_val, CandMeta* d_meta,
                      int* d_q_id, float* d_q_prob, int* d_q_cnt, int* d_draft_out, cudaStream_t s) {
    const int max_ext = p.max_extract > 0 ? p.max_extract : DRAFT_MAX_CAND;
    extract_top_candidates_batch(logits, n, 1, p.margin, max_ext, d_cand_id, d_cand_val, d_meta, s);
    draft_sample_kernel<<<1, DRAFT_MAX_CAND, 0, s>>>(d_cand_id, d_cand_val, d_meta, p, seed,
                                                     d_q_id, d_q_prob, d_q_cnt, d_draft_out);
}

}  // namespace qv::cuda
