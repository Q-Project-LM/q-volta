// CPU GEMV correctness: packer vs python fixture, kernel vs scalar reference on the same int8 activations.
#include "../src/backends/cpu/cpu_relax.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "../src/backends/cpu/cpu_gemv.h"
#include "../src/backends/cpu/qv_pack.h"

using namespace qv::cpu;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

static float h2f(uint16_t h) { return qv_h2f(h); }

static void test_fixture(const std::string& path) {
    std::ifstream f(path);
    CHECK(f.good(), "cannot open fixture %s", path.c_str());
    std::string line;
    int nlines = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int bits; ss >> bits;
        std::vector<uint32_t> exp; std::string w;
        while (ss >> w) exp.push_back((uint32_t)std::stoul(w, nullptr, 16));
        const int N = 2, K = 64;
        std::vector<uint8_t> q(N * K);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) q[n * K + k] = (uint8_t)((n * 37 + k * 11 + k * k * 3 + k / 7) % (1 << bits));
        auto got = pack_q(q.data(), N, K, bits);
        CHECK(got == exp, "pack fixture mismatch bits=%d", bits);
        std::vector<uint8_t> back(N * K);
        unpack_q(got.data(), N, K, bits, back.data());
        CHECK(back == q, "unpack roundtrip bits=%d", bits);
        ++nlines;
    }
    CHECK(nlines == 6, "fixture lines %d", nlines);
    std::printf("packer fixture: %d bit widths ok\n", nlines);
}

static void run_case(int N, int K, int bits, std::mt19937_64& rng, ThreadPool* pool) {
    const int G = K / 64;
    std::vector<uint8_t> q((size_t)N * K);
    for (auto& v : q) v = (uint8_t)(rng() & ((1u << bits) - 1));
    auto words = pack_q(q.data(), N, K, bits);
    std::vector<uint16_t> sm((size_t)N * G * 2);
    std::uniform_real_distribution<float> us(0.002f, 0.03f), um(-0.2f, 0.05f);
    for (size_t i = 0; i < sm.size(); i += 2) { sm[i] = qv_f2h(us(rng)); sm[i + 1] = qv_f2h(um(rng)); }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(K);
    for (auto& v : x) v = nd(rng) * (1.f + (rng() % 7 == 0 ? 20.f : 0.f));

    Q8ActC a;
    quantize_act_q8(x.data(), K, a);
    // sanity of activation quantizer against scalar
    for (int g = 0; g < G; ++g) {
        for (int c = 0; c < 2; ++c) {
            float amax = 0, s = 0;
            for (int i = 0; i < 32; ++i) { amax = std::fmax(amax, std::fabs(x[g * 64 + 32 * c + i])); s += x[g * 64 + 32 * c + i]; }
            CHECK(a.d(2 * g + c) == amax / 127.f, "d mismatch g=%d c=%d", g, c);
            CHECK(std::fabs(a.sum(2 * g + c) - s) <= 1e-4f * (1.f + std::fabs(s)), "sum mismatch g=%d c=%d", g, c);
        }
        for (int i = 0; i < 64; ++i) {
            const float dx = a.d(2 * g + (i >> 5));
            const int e = dx > 0 ? (int)std::nearbyint(x[g * 64 + i] / dx) : 0;
            if (a.q[g * 64 + q8_perm(i)] != e) { CHECK(false, "xq mismatch g=%d i=%d", g, i); g = G; break; }
        }
    }

    // reference on dequantized weights, same int8 activations (double accumulation)
    std::vector<uint8_t> qu((size_t)N * K);
    unpack_q(words.data(), N, K, bits, qu.data());
    std::vector<double> ref(N);
    for (int n = 0; n < N; ++n) {
        double acc = 0;
        for (int g = 0; g < G; ++g) {
            const double sc = h2f(sm[((size_t)n * G + g) * 2]), mn = h2f(sm[((size_t)n * G + g) * 2 + 1]);
            for (int c = 0; c < 2; ++c) {
                long long dot = 0;
                for (int i = 32 * c; i < 32 * c + 32; ++i) dot += (long long)qu[(size_t)n * K + g * 64 + i] * a.q[g * 64 + q8_perm(i)];
                acc += sc * (double)a.d(2 * g + c) * (double)dot + mn * (double)a.sum(2 * g + c);
            }
        }
        ref[n] = acc;
    }
    QMatC W{words.data(), sm.data(), N, K, bits};
    std::vector<float> y(N, -1.f);
    gemv_q(W, a, y.data(), 0, N);
    double en = 0, rn = 0, maxabs = 0, maxref = 0;
    for (int n = 0; n < N; ++n) {
        en += (y[n] - ref[n]) * (y[n] - ref[n]);
        rn += ref[n] * ref[n];
        maxabs = std::fmax(maxabs, std::fabs(y[n] - ref[n]));
        maxref = std::fmax(maxref, std::fabs(ref[n]));
    }
    const double rel = std::sqrt(en / (rn > 0 ? rn : 1)), rel_max = maxabs / (maxref > 0 ? maxref : 1);
    std::printf("N=%5d K=%4d bits=%d  rel_l2=%.2e  maxabs/maxref=%.2e\n", N, K, bits, rel, rel_max);
    CHECK(rel < 1e-5 && rel_max < 1e-5, "accuracy N=%d K=%d bits=%d", N, K, bits);

    if (pool) {
        std::vector<float> yp(N, -2.f);
        gemv_q_par(*pool, W, a, yp.data());
        CHECK(std::memcmp(yp.data(), y.data(), N * sizeof(float)) == 0, "parallel != serial N=%d bits=%d", N, bits);
        // row sub-range
        if (N > 9) {
            std::vector<float> ys(N, 7.f);
            gemv_q(W, a, ys.data(), 3, N - 2);
            bool ok = ys[2] == 7.f && ys[N - 2] == 7.f;
            for (int n = 3; n < N - 2; ++n) ok &= ys[n] == y[n];
            CHECK(ok, "row range N=%d bits=%d", N, bits);
        }
    }
}

int main(int argc, char** argv) {
    test_fixture(argc > 1 ? argv[1] : "tests/data/pack_fixture.txt");
    std::mt19937_64 rng(1234);
    ThreadPool pool(8);
    for (int bits : {2, 3, 4, 5, 6, 8}) {
        run_case(7, 128, bits, rng, &pool);
        run_case(13, 64, bits, rng, &pool);
    }
    for (int bits : {3, 4, 6}) {
        run_case(6144, 5120, bits, rng, &pool);
        run_case(17408, 5120, bits, rng, &pool);
    }
    for (int bits : {2, 5, 8}) run_case(1024, 1024, bits, rng, &pool);
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
