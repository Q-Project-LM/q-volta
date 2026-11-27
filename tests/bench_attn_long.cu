// Decode attention partials at long context, int4 global-layer KV (the engine's AttnArgs for a global layer): time per call and a hash of the
// partial outputs (bit-exactness check between kernel variants). Usage: bench_attn_long [ctx=85000] [nsplit=64] [reps=200]
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../src/backends/cuda/inkling/ink_ops.h"

using namespace qv::cuda::ink;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    const int ctx = argc > 1 ? std::atoi(argv[1]) : 85000, nsplit = argc > 2 ? std::atoi(argv[2]) : 64, reps = argc > 3 ? std::atoi(argv[3]) : 200, R = argc > 4 ? std::atoi(argv[4]) : 1;  // R > 1: MTP verify rows (attn_part_rows)
    const int cap = ((ctx + 1023) / 1024) * 1024 + 1024;
    std::mt19937 rng(5);
    auto fill8 = [&](uint8_t* d, size_t n) { std::vector<uint8_t> h(n); for (auto& x : h) x = (uint8_t)rng(); CK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice)); };
    uint8_t *kq, *vq; __half2 *ks, *vs; float *qs, *rl, *part; int* dpos;
    const size_t rows = (size_t)NKV * cap;
    CK(cudaMalloc(&kq, rows * 64)); CK(cudaMalloc(&vq, rows * 64));
    CK(cudaMalloc(&ks, rows * 2 * sizeof(__half2))); CK(cudaMalloc(&vs, rows * 2 * sizeof(__half2)));
    fill8(kq, rows * 64); fill8(vq, rows * 64);
    {
        std::vector<__half2> h(rows * 2);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : h) x = __floats2half2_rn(0.05f + 0.01f * std::fabs(nd(rng)), -0.4f);
        CK(cudaMemcpy(ks, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(vs, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&qs, (size_t)R * NH * HD * 4)); CK(cudaMalloc(&rl, (size_t)R * NH * REXT * 4));
    {
        std::vector<float> h((size_t)R * NH * HD); std::normal_distribution<float> nd(0.f, 0.3f);
        for (auto& x : h) x = nd(rng);
        CK(cudaMemcpy(qs, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> r((size_t)R * NH * REXT); for (auto& x : r) x = nd(rng);
        CK(cudaMemcpy(rl, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
    }
    const size_t part_n = (size_t)R * NKV * nsplit * 3 * PART_STRIDE;
    CK(cudaMalloc(&part, part_n * 4)); CK(cudaMemset(part, 0, part_n * 4));
    CK(cudaMalloc(&dpos, 4));
    const int pos = ctx - R;  // row r is position pos + r
    CK(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice));
    AttnArgs T;
    T.qs = qs; T.rl = rl; T.kc = nullptr; T.vc = nullptr; T.cap = cap; T.ring = 0; T.window = 0;
    T.kq = kq; T.vq = vq; T.ks = ks; T.vs = vs; T.pos = dpos; T.part = part; T.nsplit = nsplit; T.rows = R;
    auto run = [&]() { if (R > 1) attn_part_rows(T, 0); else attn_part(T, 0); };
    for (int i = 0; i < 5; ++i) run();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) run();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    std::vector<uint32_t> h(part_n);
    CK(cudaMemcpy(h.data(), part, part_n * 4, cudaMemcpyDeviceToHost));
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t x : h) { hash ^= x; hash *= 1099511628211ull; }
    const double bytes = (double)NKV * ctx * (64 + 64 + 16);
    std::printf("ctx %d rows %d nsplit %d: %.1f us/call (x8 global layers = %.2f ms/token), %.0f GB/s on the int4 KV read, part hash %016llx\n", ctx, R, nsplit, ms * 1000 / reps,
                ms * 8 / reps, bytes / (ms / reps * 1e-3) / 1e9, (unsigned long long)hash);
    return 0;
}
