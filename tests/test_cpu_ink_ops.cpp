// Unit tests for src/backends/cpu/ink_ops.cpp against scalar double references.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ink_ops.h"
#include "../src/backends/cpu/cpu_relax.h"

using namespace qv::cpu;
static int fails = 0;
static void check(bool ok, const char* what, double v) {
    if (!ok) { std::printf("FAIL %s (%g)\n", what, v); ++fails; }
}
static float rnd() { return (float)(std::rand() % 20001 - 10000) / 10000.f; }

int main() {
    std::srand(7);
    {  // expert activation
        const int n = 1152;
        std::vector<float> gu(2 * n), act(n);
        for (auto& x : gu) x = rnd() * 12.f;
        expert_act_interleaved(gu.data(), act.data(), n, 4.f, 25.f);
        double worst = 0;
        for (int i = 0; i < n; ++i) {
            const double g = gu[2 * i], u = gu[2 * i + 1];
            const double r = 4 * std::tanh(g / 4) * (1 / (1 + std::exp(-g))) * 25 * std::tanh(u / 25);
            worst = std::max(worst, std::fabs(r - act[i]) / (std::fabs(r) + 1e-3));
        }
        // tanh = 1 - 2/(e+1) cancels near 0 (abs err ~1e-6 at the worst element); the worst relative error depends on the libc rand() stream
        check(worst < 5e-4, "expert_act", worst);
    }
    {  // f16 gemm, T=1 and T=5
        const int N = 33, K = 2304;
        std::vector<uint16_t> W((size_t)N * K);
        std::vector<float> Wf((size_t)N * K), X(5 * K), Y(5 * N), Y1(N);
        for (size_t i = 0; i < W.size(); ++i) { W[i] = _cvtss_sh(rnd(), 0); Wf[i] = _cvtsh_ss(W[i]); }
        for (auto& x : X) x = rnd();
        gemm_f16_rows(W.data(), K, X.data(), K, 5, Y.data(), N, 0, N);
        gemm_f16_rows(W.data(), K, X.data() + 3 * K, K, 1, Y1.data(), N, 0, N);
        double worst = 0;
        for (int t = 0; t < 5; ++t)
            for (int n = 0; n < N; ++n) {
                double r = 0;
                for (int k = 0; k < K; ++k) r += (double)Wf[(size_t)n * K + k] * X[(size_t)t * K + k];
                worst = std::max(worst, std::fabs(r - Y[t * N + n]) / (std::fabs(r) + 1.0));
            }
        for (int n = 0; n < N; ++n) worst = std::max(worst, (double)std::fabs(Y1[n] - Y[3 * N + n]));
        check(worst < 5e-5, "gemm_f16_rows", worst);
    }
    {  // attention partial + merge vs naive (ring cache, window crossing the ring end, rel bias, sink)
        const int D = 128, ring = 64, rext = 40, p = 100, jlo = p - 49;
        std::vector<float> kf(ring * D), vf(ring * D), proj(16 * rext), q(3 * D), rel(48);
        std::vector<uint16_t> kc(ring * D), vc(ring * D);
        for (int i = 0; i < ring * D; ++i) { kc[i] = _cvtss_sh(rnd(), 0); vc[i] = _cvtss_sh(rnd(), 0); kf[i] = _cvtsh_ss(kc[i]); vf[i] = _cvtsh_ss(vc[i]); }
        for (auto& x : proj) x = rnd() * 0.3f;
        for (auto& x : q) x = rnd();
        for (auto& x : rel) x = rnd();
        const float scale = 1.0f / std::sqrt(128.f), sink = 0.7f;
        float part[2][390];
        for (int s = 0; s < 2; ++s) {
            InkAttnArgs a{q.data(), rel.data(), kc.data(), vc.data(), proj.data(), p, jlo + 25 * s, jlo + 25 * (s + 1), ring, rext, scale};
            ink_attn_partial(a, part[s], part[s] + 384);
        }
        std::vector<float> gate(D, 0.3f), out(D);
        double worst = 0;
        for (int h = 0; h < 3; ++h) {
            ink_attn_merge(&part[0][0], 2, 390, h, sink, gate.data(), out.data());
            std::vector<double> sc, o(D, 0);
            double m = sink;
            for (int j = jlo; j <= p; ++j) {
                double s = 0;
                for (int d = 0; d < D; ++d) s += (double)q[h * D + d] * kf[(j % ring) * D + d];
                s *= scale;
                const int dist = p - j;
                if (dist < rext) for (int i = 0; i < 16; ++i) s += (double)rel[16 * h + i] * proj[i * rext + dist];
                sc.push_back(s); m = std::max(m, s);
            }
            double den = std::exp(sink - m);
            for (double s : sc) den += std::exp(s - m);
            for (size_t jj = 0; jj < sc.size(); ++jj)
                for (int d = 0; d < D; ++d) o[d] += std::exp(sc[jj] - m) / den * vf[((jlo + jj) % ring) * D + d];
            for (int d = 0; d < D; ++d) worst = std::max(worst, std::fabs(o[d] / (1 + std::exp(-0.3)) - out[d]));
        }
        check(worst < 2e-5, "attn_partial+merge", worst);
    }
    {  // block attention residual
        const int Hh = 64;
        std::vector<float> b0(Hh), b1(Hh), part(Hh), qs(2 * Hh), out(2 * Hh);
        for (auto* v : {&b0, &b1, &part, &qs}) for (auto& x : *v) x = rnd() * 3;
        const float* bl[2] = {b0.data(), b1.data()};
        block_attn_res(bl, 2, part.data(), qs.data(), 2, Hh, out.data());
        double worst = 0;
        for (int q = 0; q < 2; ++q) {
            const std::vector<float>* st[3] = {&b0, &b1, &part};
            double S[3][Hh], sc[3], m = -1e30, den = 0;
            for (int b = 0; b < 3; ++b) {
                double ss = 0; for (int i = 0; i < Hh; ++i) ss += (double)(*st[b])[i] * (*st[b])[i];
                const double r = 1 / std::sqrt(ss / Hh + 1e-6);
                double dt = 0;
                for (int i = 0; i < Hh; ++i) { S[b][i] = (*st[b])[i] * r; dt += S[b][i] * qs[q * Hh + i]; }
                sc[b] = dt / std::sqrt((double)Hh); m = std::max(m, sc[b]);
            }
            for (int b = 0; b < 3; ++b) { sc[b] = std::exp(sc[b] - m); den += sc[b]; }
            for (int i = 0; i < Hh; ++i) {
                double o = 0; for (int b = 0; b < 3; ++b) o += sc[b] / den * S[b][i];
                worst = std::max(worst, std::fabs(o - out[q * Hh + i]));
            }
        }
        check(worst < 1e-5, "block_attn_res", worst);
    }
    {  // router
        std::vector<float> lg(128), bias(128);
        for (auto& x : lg) x = rnd() * 4;
        for (auto& x : bias) x = rnd() * 0.2f;
        int idx[8]; float w[8];
        router_topk(lg.data(), bias.data(), 128, 8, 8.0f, idx, w);
        double sum = 0; for (int j = 0; j < 8; ++j) sum += w[j];
        check(std::fabs(sum - 8.0) < 1e-4, "router weights sum", sum);
        float prev = 1e9; bool ok = true;
        for (int j = 0; j < 8; ++j) { const float s = 1 / (1 + std::exp(-lg[idx[j]])) + bias[idx[j]]; ok &= s <= prev; prev = s; }
        check(ok, "router order", 0);
    }
    if (!fails) std::printf("test_cpu_ink_ops: all passed\n");
    return fails ? 1 : 0;
}
