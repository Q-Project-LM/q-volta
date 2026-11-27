// Prefill attention (global layer, int4 KV) for one chunk at a long position: time per call + hash of the output (bit-exactness between kernel variants).
// Usage: bench_attn_pf [pos0=63488] [T=2048] [reps=5]
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../src/backends/cuda/inkling/ink_prefill.h"

using namespace qv::cuda::ink;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    const int pos0 = argc > 1 ? std::atoi(argv[1]) : 63488, T = argc > 2 ? std::atoi(argv[2]) : 2048, reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const int cap = ((pos0 + T + 1023) / 1024) * 1024;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto up8 = [&](size_t n) { std::vector<uint8_t> h(n); for (auto& x : h) x = (uint8_t)rng(); void* d; CK(cudaMalloc(&d, n)); CK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice)); return (uint8_t*)d; };
    auto uph = [&](size_t n, float sd) { std::vector<__half> h(n); for (auto& x : h) x = __float2half(nd(rng) * sd); void* d; CK(cudaMalloc(&d, n * 2)); CK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice)); return (__half*)d; };
    PfAttnArgs P{};
    const size_t rows = (size_t)NKV * cap;
    P.kq = up8(rows * 64); P.vq = up8(rows * 64);
    {
        std::vector<__half2> h(rows * 2);
        for (auto& x : h) x = __floats2half2_rn(0.05f + 0.01f * std::fabs(nd(rng)), -0.4f);
        void *a, *b; CK(cudaMalloc(&a, h.size() * 4)); CK(cudaMalloc(&b, h.size() * 4));
        CK(cudaMemcpy(a, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(b, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        P.ks = (__half2*)a; P.vs = (__half2*)b;
    }
    const int Tp = (T + 63) / 64 * 64;
    P.qh = uph((size_t)Tp * 3072, 0.3f);
    P.rl = uph((size_t)T * NH * REXT, 0.5f);
    {
        std::vector<float> h((size_t)T * PS); for (auto& x : h) x = nd(rng);
        void* d; CK(cudaMalloc(&d, h.size() * 4)); CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); P.proj = (float*)d;
    }
    __half* out; CK(cudaMalloc(&out, (size_t)Tp * 3072 * 2)); CK(cudaMemset(out, 0, (size_t)Tp * 3072 * 2));
    P.out = out; P.kbase = 0; P.cap = cap; P.T = T; P.pos0 = pos0; P.window = 0;
    for (int i = 0; i < 2; ++i) attn_prefill(P, 0);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) attn_prefill(P, 0);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= reps;
    std::vector<__half> h((size_t)Tp * 3072);
    CK(cudaMemcpy(h.data(), out, h.size() * 2, cudaMemcpyDeviceToHost));
    uint64_t hash = 1469598103934665603ull;
    for (auto x : h) { hash ^= *(uint16_t*)&x; hash *= 1099511628211ull; }
    const double flop = (double)T * (pos0 + T / 2.0) * NH * 128 * 2 * 2;  // QK + PV (the two-pass kernel does QK twice)
    std::printf("pos0 %d T %d: %.2f ms/call (%.1f TFLOP/s useful), out hash %016llx\n", pos0, T, ms, flop / (ms * 1e-3) / 1e12, (unsigned long long)hash);
    return 0;
}
