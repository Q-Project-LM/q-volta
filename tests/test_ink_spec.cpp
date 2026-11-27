// Lossless speculative sampling (T > 0) for the Inkling CUDA engine.
//  1. temperature 0: decode_speculative must give the plain greedy ids and the same next-step logits (exact).
//  2. N draws of the first R tokens after a fixed context, speculative (decode_speculative) vs plain (step + host Sampler): two-sample chi-square
//     homogeneity test of the marginal token distribution at each of the R positions (a lossless sampler matches the plain one at every position).
// Usage: test_ink_spec MODEL.qv MTP.qv DEPTH PROMPT.i32 N_DRAWS TOP_K TOP_P MIN_P [TEMP=1.0] [P_MIN=1e-4]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>

#include "qv/engine.h"
#include "../src/sampling/sampler.h"

static std::vector<int> read_i32(const char* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<int> v;
    int x;
    while (f.read((char*)&x, 4)) v.push_back(x);
    return v;
}
// regularized upper incomplete gamma Q(a, x) (chi-square survival with a = dof/2, x = stat/2)
static double gamma_q(double a, double x) {
    if (x <= 0) return 1.0;
    const double gln = std::lgamma(a);
    if (x < a + 1) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 1000; ++n) { ap += 1; del *= x / ap; sum += del; if (std::fabs(del) < std::fabs(sum) * 1e-14) break; }
        return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2; d = an * d + b; if (std::fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c; if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1 / d; const double del = d * c; h *= del;
        if (std::fabs(del - 1) < 1e-14) break;
    }
    return std::exp(-x + a * std::log(x) - gln) * h;
}
// toy constraint: only even token ids are allowed; the 11th accepted token is an "end of generation"
struct EvenConstraint : qv::TokenConstraint {
    int accepted = 0;
    bool awaiting() const override { return false; }
    bool is_dead() const override { return false; }
    bool is_eog(int) const override { return false; }
    bool probe_allowed(int tok) const override { return tok % 2 == 0; }
    bool try_accept(int tok) override { if (tok % 2) return false; ++accepted; return true; }
    void mask_logits(float* l, size_t n) override { for (size_t i = 1; i < n; i += 2) l[i] = -INFINITY; }
};
int main(int argc, char** argv) {
    if (argc < 9 || !std::ifstream(argv[1])) { std::printf("test_ink_spec: no model, skipped\n"); return 0; }
    const int depth = std::atoi(argv[3]);
    auto prompt = read_i32(argv[4]);
    const int draws = std::atoi(argv[5]);
    qv::SpecSamplingParams sp;
    sp.top_k = std::atoi(argv[6]); sp.top_p = (float)std::atof(argv[7]); sp.min_p = (float)std::atof(argv[8]);
    sp.temperature = argc > 9 ? (float)std::atof(argv[9]) : 1.0f;
    auto mf = qv::ModelFile::open(argv[1]);
    qv::EngineOptions opt;
    opt.max_ctx = 8192; opt.kv_mode = 1; opt.mtp = true; opt.mtp_depth = depth; opt.mtp_file = argv[2];
    auto eng = qv::make_cuda_engine(mf, opt);
    const int R = depth + 1, L = (int)prompt.size();
    int fail = 0;
    // 1. temperature 0
    {
        const int n = 24;
        auto run = [&](bool spec, std::vector<int>& ids, std::vector<float>& lg) {
            eng->reset();
            eng->set_mtp_enabled(spec);
            eng->prefill(prompt.data(), L, 0);
            const int first = eng->argmax();
            ids.clear();
            if (spec) { qv::SpecSamplingParams g; g.temperature = 0; eng->decode_speculative(first, L, n, g, ids); }
            else eng->decode_greedy(first, L, n, ids, false);
            eng->step(ids.back(), L + n);
            eng->logits(lg);
        };
        std::vector<int> a, b; std::vector<float> la, lb;
        run(false, a, la); run(true, b, lb);
        const bool ok = a == b && la == lb;
        std::printf("greedy equality (T=0, %d tokens): ids %d logits %d\n", n, (int)(a == b), (int)(la == lb));
        if (!ok) ++fail;
    }
    // 1a. constrained greedy (grammar path): masked argmax per token, equal to plain stepping with a host-side mask; state equality after the run
    {
        const int n = 30;
        eng->reset();
        eng->set_mtp_enabled(false);
        eng->prefill(prompt.data(), L, 0);
        std::vector<float> lg;
        eng->logits(lg);
        int tok0 = 0;
        { float b = -INFINITY; for (size_t i = 0; i < lg.size(); i += 2) if (lg[i] > b) { b = lg[i]; tok0 = (int)i; } }
        std::vector<int> ref{tok0};
        for (int j = 0; j < n; ++j) {
            eng->step(ref.back(), L + j);
            eng->logits(lg);
            int t = 0; float b = -INFINITY;
            for (size_t i = 0; i < lg.size(); i += 2) if (lg[i] > b) { b = lg[i]; t = (int)i; }
            ref.push_back(t);
        }
        eng->step(ref.back(), L + n);
        eng->logits(lg);
        std::vector<float> lref = lg;
        eng->reset();
        eng->set_mtp_enabled(true);
        eng->prefill(prompt.data(), L, 0);
        EvenConstraint c;
        std::vector<int> got;
        eng->decode_greedy_constrained(tok0, L, n, got, &c);
        eng->step(got.back(), L + n);
        std::vector<float> lgot;
        eng->logits(lgot);
        bool ok = (int)got.size() == n && lgot == lref;
        for (int j = 0; j < n && ok; ++j) ok = got[j] == ref[j + 1];
        std::printf("constrained greedy (even ids only, %d tokens): ids+state identical=%d (accepted by constraint %d)\n", n, (int)ok, c.accepted);
        if (!ok) ++fail;
    }
    // 1b. state consistency at T > 0: after speculative sampling of n tokens (any accept pattern) the engine state must equal plain decode's after the same tokens
    {
        const int n = 40, seeds = std::getenv("QV_SPEC_REPLAY") ? std::atoi(std::getenv("QV_SPEC_REPLAY")) : 6;
        int bad = 0;
        for (int sd = 0; sd < seeds; ++sd) {
            eng->reset();
            eng->set_mtp_enabled(true);
            eng->prefill(prompt.data(), L, 0);
            const int first = eng->argmax();
            std::vector<int> ids;
            qv::SpecSamplingParams g = sp;
            g.seed = 424242ull + sd;
            qv::MtpStats a0, a1;
            eng->mtp_stats(a0);
            eng->decode_speculative(first, L, n, g, ids);
            eng->mtp_stats(a1);
            eng->step(ids.back(), L + n);
            std::vector<float> la;
            eng->logits(la);
            eng->reset();
            eng->set_mtp_enabled(false);
            eng->prefill(prompt.data(), L, 0);
            eng->step(first, L);
            for (int j = 0; j < n; ++j) eng->step(ids[j], L + 1 + j);
            std::vector<float> lb;
            eng->logits(lb);
            const bool ok = (int)ids.size() == n && la == lb;
            std::printf("replay seed %d: %d tokens, cycles %ld accepted %ld plain_steps %ld, next-step logits identical=%d\n", sd, (int)ids.size(), a1.cycles - a0.cycles,
                        a1.accepted - a0.accepted, a1.plain_steps - a0.plain_steps, (int)ok);
            if (!ok) ++bad;
        }
        if (bad) ++fail;
    }
    if (draws <= 0) { std::printf(fail ? "FAIL\n" : "PASS\n"); return fail != 0; }
    // 2. distribution test
    std::vector<std::map<int, long>> cs(R), cp(R);
    qv::SamplingParams sampp;
    sampp.temperature = sp.temperature; sampp.top_k = sp.top_k; sampp.top_p = sp.top_p; sampp.min_p = sp.min_p; sampp.has_seed = true;
    qv::MtpStats s0, s1;
    eng->mtp_stats(s0);
    for (int d = 0; d < draws; ++d) {
        eng->reset();
        eng->set_mtp_enabled(true);
        eng->prefill(prompt.data(), L, 0);
        const int first = eng->argmax();
        std::vector<int> out;
        sp.seed = 1000003ull * (d + 1);
        eng->decode_speculative(first, L, R, sp, out);
        for (int j = 0; j < R; ++j) cs[j][out[j]]++;
    }
    eng->mtp_stats(s1);
    for (int d = 0; d < draws; ++d) {
        eng->reset();
        eng->set_mtp_enabled(false);
        eng->prefill(prompt.data(), L, 0);
        int tok = eng->argmax();
        sampp.seed = 7777777ull * (d + 1);
        qv::Sampler smp(sampp, prompt);
        std::vector<float> lg;
        for (int j = 0; j < R; ++j) {
            eng->step(tok, L + j);
            eng->logits(lg);
            tok = smp.sample(lg);
            smp.accept(tok);
            cp[j][tok]++;
        }
    }
    std::printf("spec stats: cycles %ld drafted %ld accepted %ld (accepted/cycle %.3f) plain_steps %ld\n", s1.cycles - s0.cycles, s1.drafted - s0.drafted,
                s1.accepted - s0.accepted, (double)(s1.accepted - s0.accepted) / std::max<long>(1, s1.cycles - s0.cycles), s1.plain_steps - s0.plain_steps);
    for (int j = 0; j < R; ++j) {
        std::map<int, std::pair<long, long>> u;
        for (auto& kv : cs[j]) u[kv.first].first = kv.second;
        for (auto& kv : cp[j]) u[kv.first].second = kv.second;
        double stat = 0; int cats = 0;
        long oa = 0, ob = 0;
        auto add = [&](long a, long b) {
            const double e = 0.5 * (a + b);
            stat += ((a - e) * (a - e) + (b - e) * (b - e)) / e;
            ++cats;
        };
        for (auto& kv : u) {
            const long a = kv.second.first, b = kv.second.second;
            if (a + b < 10) { oa += a; ob += b; } else add(a, b);
        }
        if (oa + ob >= 10) add(oa, ob); else if (cats > 0) { /* fold tiny remainder into nothing */ }
        const int dof = cats - 1;
        const double pv = dof > 0 ? gamma_q(0.5 * dof, 0.5 * stat) : 1.0;
        const double pmin = argc > 10 ? std::atof(argv[10]) : 1e-4;
        std::printf("position %d: support spec %zu plain %zu, chi2 %.2f dof %d p-value %.4f %s\n", j, cs[j].size(), cp[j].size(), stat, dof, pv, pv < pmin ? "FAIL" : "ok");
        if (pv < pmin) ++fail;
    }
    std::printf(fail ? "FAIL\n" : "PASS\n");
    return fail != 0;
}
