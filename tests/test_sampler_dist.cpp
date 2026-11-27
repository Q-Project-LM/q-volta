// Distribution tests of the host Sampler and of the speculative-sampling helpers against the analytic distribution
// (top_k -> top_p -> min_p -> temperature, llama.cpp semantics), chi-square with pooled bins (p28/loops, docs/LOOPS.md).
//   1. qv::Sampler draws vs analytic, server defaults (T=1, top_k 40, top_p 0.95, min_p 0.05) and T=0.7 / top_k 0 / min_p only.
//   2. build_dist_dense vs analytic (exact probabilities).
//   3. one lossless speculative step: draft x~q (a perturbed distribution), accept w.p. min(1,p/q), else residual; the emitted token is ~p.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "qv/spec_sampling.h"
#include "sampling/sampler.h"

using namespace qv;

static std::vector<double> analytic(const std::vector<float>& lg, int top_k, double top_p, double min_p, double T) {
    const int V = (int)lg.size();
    std::vector<int> ord(V);
    for (int i = 0; i < V; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    size_t keep = V;
    if (top_k > 0) keep = std::min<size_t>(keep, top_k);
    auto softmax = [&](size_t n, double t) {
        std::vector<double> p(n);
        double s = 0;
        for (size_t i = 0; i < n; ++i) { p[i] = std::exp((lg[ord[i]] - lg[ord[0]]) / t); s += p[i]; }
        for (auto& x : p) x /= s;
        return p;
    };
    if (top_p < 1.0) {
        auto p = softmax(keep, 1.0);
        double c = 0;
        size_t k = keep;
        for (size_t i = 0; i < keep; ++i) { c += p[i]; if (c >= top_p) { k = i + 1; break; } }
        keep = k;
    }
    if (min_p > 0) {
        size_t k = 0;
        while (k < keep && std::exp((double)lg[ord[k]] - lg[ord[0]]) >= min_p) ++k;
        keep = std::max<size_t>(k, 1);
    }
    auto p = softmax(keep, T);
    std::vector<double> out(V, 0.0);
    for (size_t i = 0; i < keep; ++i) out[ord[i]] = p[i];
    return out;
}

static double chi2_z(const std::map<int, long>& cnt, const std::vector<double>& p, long N, int* df_out) {
    // pool bins with expected < 10 into one; return (chi2 - df) / sqrt(2 df)
    double chi = 0, pool_p = 0, pool_o = 0;
    int bins = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const auto it = cnt.find((int)i);
        const double o = it == cnt.end() ? 0 : (double)it->second, e = p[i] * N;
        if (e == 0) { if (o > 0) { std::printf("  token %zu has probability 0 but %g draws\n", i, o); return 1e9; } continue; }
        if (e < 10) { pool_p += p[i]; pool_o += o; continue; }
        chi += (o - e) * (o - e) / e; ++bins;
    }
    if (pool_p * N >= 10) { const double e = pool_p * N; chi += (pool_o - e) * (pool_o - e) / e; ++bins; }
    const int df = std::max(1, bins - 1);
    *df_out = df;
    return (chi - df) / std::sqrt(2.0 * df);
}

int main() {
    const int V = 4000;
    std::mt19937 g(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    struct Cfg { int k; double p, mp, T; } cfgs[] = {{40, 0.95, 0.05, 1.0}, {40, 0.95, 0.05, 0.7}, {0, 1.0, 0.05, 1.0}, {20, 0.9, 0.0, 1.0}, {0, 0.95, 0.0, 1.3}};
    for (int scale = 0; scale < 2; ++scale)
        for (auto c : cfgs) {
            std::vector<float> lg(V);
            for (auto& x : lg) x = nd(g) * (scale ? 3.0f : 1.6f);
            for (int i = 0; i < 8; ++i) lg[i * 37] += 4.0f + i * 0.3f;  // a peaked head
            auto pa = analytic(lg, c.k, c.p, c.mp, c.T);
            const long N = 400000;
            SamplingParams sp;
            sp.temperature = (float)c.T; sp.top_k = c.k; sp.top_p = (float)c.p; sp.min_p = (float)c.mp; sp.has_seed = true; sp.seed = 7;
            Sampler s(sp, {});
            std::map<int, long> cnt;
            for (long i = 0; i < N; ++i) { std::vector<float> l = lg; ++cnt[s.sample(l)]; }
            int df;
            double z = chi2_z(cnt, pa, N, &df);
            SpecSamplingParams ssp;
            ssp.temperature = (float)c.T; ssp.top_k = c.k; ssp.top_p = (float)c.p; ssp.min_p = (float)c.mp;
            SpecDist d;
            build_dist_dense(lg.data(), V, ssp, d);
            double tvd = 0;
            for (int i = 0; i < V; ++i) tvd += std::fabs(d.prob_of(i) - pa[i]);
            tvd *= 0.5;
            // lossless speculative step
            std::vector<float> lq = lg;
            for (auto& x : lq) x += nd(g) * 0.7f;  // draft logits differ from the target
            SpecSamplingParams qsp = ssp;
            SpecDist q;
            build_dist_dense(lq.data(), V, qsp, q);
            std::mt19937_64 rng(99);
            std::map<int, long> cnt2;
            for (long i = 0; i < N; ++i) {
                int x = sample_dist(q, rng);
                int out = spec_accept(d, q, x, rng) ? x : sample_residual(d, q, rng);
                ++cnt2[out];
            }
            int df2;
            double z2 = chi2_z(cnt2, pa, N, &df2);
            const bool ok = std::fabs(z) < 5 && std::fabs(z2) < 5 && tvd < 2e-6;
            std::printf("scale %d k=%d p=%.2f minp=%.2f T=%.1f: sampler z=%+.2f (df %d)  spec-step z=%+.2f (df %d)  build_dist TVD=%.2e  %s\n", scale, c.k, c.p, c.mp, c.T, z, df, z2, df2, tvd, ok ? "ok" : "FAIL");
            fails += !ok;
        }
    std::printf(fails ? "FAILED\n" : "all ok\n");
    return fails != 0;
}
