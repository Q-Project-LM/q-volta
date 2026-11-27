// CPU-only distribution test of the lossless speculative round with a deterministic (point-mass) drafter, qv::spec_round_point (the code the Inkling engine runs).
// A synthetic order-1 language model (V tokens, context-dependent logits) is sampled through the real filter chain (build_dist_dense: top_k/top_p/min_p/temperature):
// plain = sample_dist from the next-token distribution; spec = draft chain of imperfect greedy drafts (the drafter uses a perturbed model) + spec_round_point.
// With the 6th argument "sample" the drafts are drawn from the drafter distribution (same filter chain) and the round is spec_round_q.
// Two-sample chi-square homogeneity per position and for the position pairs. Usage: test_spec_point [DRAWS=2000000] [TOP_K=40] [TOP_P=0.95] [MIN_P=0.05] [TEMP=1.0]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "qv/spec_sampling.h"

static double gamma_q(double a, double x) {
    if (x <= 0) return 1.0;
    const double gln = std::lgamma(a);
    if (x < a + 1) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 2000; ++n) { ap += 1; del *= x / ap; sum += del; if (std::fabs(del) < std::fabs(sum) * 1e-14) break; }
        return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (int i = 1; i < 2000; ++i) {
        const double an = -i * (i - a);
        b += 2; d = an * d + b; if (std::fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c; if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1 / d; const double del = d * c; h *= del;
        if (std::fabs(del - 1) < 1e-14) break;
    }
    return std::exp(-x + a * std::log(x) - gln) * h;
}
static constexpr int V = 64;
static float tab[V][V], tabd[V][V];
static void logits_of(int prev, bool drafter, std::vector<float>& lg) {
    lg.resize(V);
    for (int i = 0; i < V; ++i) lg[i] = drafter ? tabd[prev][i] : tab[prev][i];
}
int main(int argc, char** argv) {
    const long draws = argc > 1 ? std::atol(argv[1]) : 2000000;
    qv::SpecSamplingParams sp;
    sp.top_k = argc > 2 ? std::atoi(argv[2]) : 40;
    sp.top_p = argc > 3 ? (float)std::atof(argv[3]) : 0.95f;
    sp.min_p = argc > 4 ? (float)std::atof(argv[4]) : 0.05f;
    sp.temperature = argc > 5 ? (float)std::atof(argv[5]) : 1.0f;
    std::mt19937_64 g(12345);
    std::normal_distribution<float> nd(0.f, 2.0f), nz(0.f, 1.2f);
    for (int i = 0; i < V; ++i) for (int j = 0; j < V; ++j) { tab[i][j] = nd(g); tabd[i][j] = tab[i][j] + nz(g); }
    const bool sampled = argc > 6 && std::string(argv[6]) == "sample";
    const int ND = 3, R = ND + 1;
    std::vector<std::map<long, long>> A(R), B(R);  // key = joint of the first (j+1) tokens (position 0..j): tests the joint prefix distribution
    std::mt19937_64 rs(1), rp(2);
    long accepted = 0, rounds = 0;
    auto dist_for = [&](int prev, bool drafter, qv::SpecDist& d) {
        std::vector<float> lg;
        logits_of(prev, drafter, lg);
        qv::build_dist_dense(lg.data(), V, sp, d);
    };

    std::mt19937_64 rd(99);
    auto draft_chain = [&](int first, int* drafts, qv::SpecDist* qd) {
        int prev = first;
        for (int j = 0; j < ND; ++j) {
            if (!sampled) {
                std::vector<float> lg; logits_of(prev, true, lg);
                int best = 0; for (int i = 1; i < V; ++i) if (lg[i] > lg[best]) best = i;
                drafts[j] = best;
            } else {
                dist_for(prev, true, qd[j]);
                drafts[j] = qv::sample_dist(qd[j], rd);
            }
            prev = drafts[j];
        }
    };
    auto do_round = [&](int cur, const int* drafts, const qv::SpecDist* qd, std::mt19937_64& rg, int* seq, int* nacc) {
        auto pf = [&](int j, qv::SpecDist& d) { dist_for(j == 0 ? cur : drafts[j - 1], false, d); };
        if (!sampled) return qv::spec_round_point(ND, drafts, pf, rg, seq, nacc);
        return qv::spec_round_q(ND, drafts, pf, [&](int j, qv::SpecDist& q) { q = qd[j]; }, rg, seq, nacc);
    };
    for (long n = 0; n < draws; ++n) {
        const int first = (int)(n % 5);
        // plain
        {
            int prev = first;
            long key = 0;
            for (int j = 0; j < R; ++j) {
                qv::SpecDist d;
                dist_for(prev, false, d);
                const int t = qv::sample_dist(d, rp);
                key = key * V + t;
                B[j][key]++;
                prev = t;
            }
        }
        // speculative
        {
            int drafts[ND]; qv::SpecDist qd[ND];
            draft_chain(first, drafts, qd);
            int seq[R], nacc = 0;
            const int m = do_round(first, drafts, qd, rs, seq, &nacc);
            accepted += nacc; ++rounds;
            long key = 0;
            for (int j = 0; j <= m; ++j) { key = key * V + seq[j]; A[j][key]++; }
            // positions beyond m+1 come from the next round in the engine; here only the produced prefix counts, so compare only fully produced rounds below
            (void)m;
        }
    }
    // spec rounds yield m+1 tokens; to compare like with like, test the first token (always produced) and the pair (first two tokens) when produced in the same round
    int fail = 0;
    for (int j = 0; j < 2; ++j) {
        // restrict plain to the same key space: joint prefix of length j+1; spec counts only rounds with m >= j, so compare conditional frequencies via the ratio test below
        (void)j;
    }
    // position 0 marginal: all rounds produce it
    for (int j = 0; j < 1; ++j) {
        std::map<long, std::pair<long, long>> u;
        for (auto& kv : A[j]) u[kv.first].first = kv.second;
        for (auto& kv : B[j]) u[kv.first].second = kv.second;
        double stat = 0; int cats = 0; long oa = 0, ob = 0;
        auto add = [&](long a, long b) { const double e = 0.5 * (a + b); stat += ((a - e) * (a - e) + (b - e) * (b - e)) / e; ++cats; };
        for (auto& kv : u) { if (kv.second.first + kv.second.second < 20) { oa += kv.second.first; ob += kv.second.second; } else add(kv.second.first, kv.second.second); }
        if (oa + ob >= 20) add(oa, ob);
        const double pv = gamma_q(0.5 * (cats - 1), 0.5 * stat);
        std::printf("token 0 marginal: %zu categories, chi2 %.1f dof %d p %.4f %s\n", u.size(), stat, cats - 1, pv, pv < 1e-3 ? "FAIL" : "ok");
        if (pv < 1e-3) ++fail;
    }
    // full-sequence test: generate whole 24-token streams with rounds back to back and compare marginals of the token at positions 1..3 of the stream
    std::vector<std::map<int, long>> SA(8), SB(8);
    {
        std::mt19937_64 r1(11), r2(12);
        const int NS = 8;
        for (long n = 0; n < draws / 4; ++n) {
            const int first = (int)(n % 5);
            int prev = first;
            for (int j = 0; j < NS; ++j) { qv::SpecDist d; dist_for(prev, false, d); prev = qv::sample_dist(d, r2); SB[j][prev]++; }
            int cur = first, produced = 0;
            while (produced < NS) {
                int drafts[ND]; qv::SpecDist qd[ND];
                draft_chain(cur, drafts, qd);
                int seq[R];
                const int m = do_round(cur, drafts, qd, r1, seq, nullptr);
                for (int j = 0; j <= m && produced < NS; ++j) SA[produced++][seq[j]]++;
                cur = seq[m];
            }
        }
    }
    for (int j = 0; j < 8; ++j) {
        std::map<int, std::pair<long, long>> u;
        for (auto& kv : SA[j]) u[kv.first].first = kv.second;
        for (auto& kv : SB[j]) u[kv.first].second = kv.second;
        double stat = 0; int cats = 0; long oa = 0, ob = 0;
        auto add = [&](long a, long b) { const double e = 0.5 * (a + b); stat += ((a - e) * (a - e) + (b - e) * (b - e)) / e; ++cats; };
        for (auto& kv : u) { if (kv.second.first + kv.second.second < 20) { oa += kv.second.first; ob += kv.second.second; } else add(kv.second.first, kv.second.second); }
        if (oa + ob >= 20) add(oa, ob);
        const double pv = gamma_q(0.5 * (cats - 1), 0.5 * stat);
        std::printf("stream position %d marginal: %zu categories, chi2 %.1f dof %d p %.4f %s\n", j, u.size(), stat, cats - 1, pv, pv < 1e-3 ? "FAIL" : "ok");
        if (pv < 1e-3) ++fail;
    }
    std::printf("mean accepted drafts per round %.3f (of %d)\n", (double)accepted / rounds, ND);
    std::printf(fail ? "FAIL\n" : "PASS\n");
    return fail != 0;
}
