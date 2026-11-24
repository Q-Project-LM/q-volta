// Portable (ARM) GEMV bandwidth bench: rotates over >= --gib GiB of random quantized matrices, reports GB/s per thread count
// and a plain streaming-read ceiling. usage: bench_neon_gemv [--bits 3,4,5] [--threads 1,2,4,6,8,10] [--gib 1.5] [--N 4096] [--K 4096]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "cpu_gemv.h"
using namespace qv::cpu;
using clk = std::chrono::steady_clock;
static std::vector<int> parse(const std::string& s) {
    std::vector<int> v; size_t i = 0;
    while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); v.push_back(atoi(s.substr(i, j - i).c_str())); i = j + 1; }
    return v;
}
#ifdef __aarch64__
static __attribute__((noinline)) uint64_t stream_read(const void* p, size_t bytes) {
    const uint8_t* a = (const uint8_t*)p;
    uint8x16_t s0 = vdupq_n_u8(0), s1 = s0, s2 = s0, s3 = s0;
    for (size_t o = 0; o < bytes; o += 64) {
        s0 = veorq_u8(s0, vld1q_u8(a + o)); s1 = veorq_u8(s1, vld1q_u8(a + o + 16));
        s2 = veorq_u8(s2, vld1q_u8(a + o + 32)); s3 = veorq_u8(s3, vld1q_u8(a + o + 48));
    }
    s0 = veorq_u8(veorq_u8(s0, s1), veorq_u8(s2, s3));
    return vgetq_lane_u64(vreinterpretq_u64_u8(s0), 0) ^ vgetq_lane_u64(vreinterpretq_u64_u8(s0), 1);
}
#endif
int main(int argc, char** argv) {
    std::string bits_s = "3,4,5", thr_s = "1,2,4,6,8,10";
    double gib = 1.5; int N = 4096, K = 4096;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--bits") bits_s = argv[i + 1]; else if (a == "--threads") thr_s = argv[i + 1];
        else if (a == "--gib") gib = atof(argv[i + 1]); else if (a == "--N") N = atoi(argv[i + 1]); else if (a == "--K") K = atoi(argv[i + 1]);
    }
    std::mt19937_64 rng(1);
    std::vector<float> xf(K);
    for (auto& v : xf) v = (float)((int64_t)(rng() % 2001) - 1000) / 500.f;
    Q8ActC xa; quantize_act_q8(xf.data(), K, xa);
    std::vector<float> y(N);
    for (int bits : parse(bits_s)) {
        QMatC W{nullptr, nullptr, N, K, bits};
        const size_t one = W.bytes();
        const int nm = (int)((gib * (1u << 30)) / one) + 1;
        std::vector<std::vector<uint32_t>> qs(nm); std::vector<std::vector<uint16_t>> ss(nm);
        for (int m = 0; m < nm; ++m) {
            qs[m].resize((size_t)N * W.row_words()); ss[m].resize((size_t)N * (K / 64) * 2);
            for (auto& v : qs[m]) v = (uint32_t)rng();
            for (size_t i = 0; i < ss[m].size(); i += 2) { ss[m][i] = 0x2400; ss[m][i + 1] = 0x2000; }
        }
        printf("bits=%d N=%d K=%d  %d matrices x %.1f MiB\n", bits, N, K, nm, one / 1048576.0);
        for (int th : parse(thr_s)) {
            qv::cpu::ThreadPool pool(th);
            for (int rep = 0; rep < 2; ++rep) {
                int iters = 0; auto t0 = clk::now(); double el = 0;
                while (el < 1.0) {
                    QMatC w{qs[iters % nm].data(), ss[iters % nm].data(), N, K, bits};
                    gemv_q_par(pool, w, xa, y.data()); ++iters;
                    el = std::chrono::duration<double>(clk::now() - t0).count();
                }
                if (rep) printf("  threads=%2d  %7.1f GB/s  %6.1f us/matrix\n", th, (double)one * iters / el / 1e9, el / iters * 1e6);
            }
        }
#ifdef __aarch64__
        if (bits == 4) {
            qv::cpu::ThreadPool pool(10);
            for (int th : {4, 8, 10}) {
                qv::cpu::ThreadPool p2(th); volatile uint64_t sink = 0; int it = 0; auto t0 = clk::now(); double el = 0;
                while (el < 1.0) {
                    const size_t sz = qs[it % nm].size() * 4;
                    p2.parallel_for(sz / 64, [&](size_t b, size_t e, int) { sink ^= stream_read((const char*)qs[it % nm].data() + b * 64, (e - b) * 64); }, 64);
                    ++it; el = std::chrono::duration<double>(clk::now() - t0).count();
                }
                printf("  stream-read ceiling threads=%2d  %7.1f GB/s\n", th, (double)qs[0].size() * 4 * it / el / 1e9);
            }
        }
#endif
    }
}
