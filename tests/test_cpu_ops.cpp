// Tests for qv_cpu_ops: every op vs an independent scalar double-precision reference on random data.
#include "../src/backends/cpu/cpu_relax.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ops.h"

using namespace qv::cpu;
static std::mt19937 rng(1234);
static float rnd(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static std::vector<float> rvec(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto& x : v) x = rnd(lo, hi);
    return v;
}
static int fails = 0;
// error = max_i |a_i - b_i| / (|b_i| + 0.05 * max|b|): relative per element, floored so near-zero (cancelling) elements
// are judged against the vector scale, as fp32 accumulation noise is absolute there
static void check(const char* name, const float* a, const double* b, size_t n, double tol = 1e-5) {
    double mx = 0;
    for (size_t i = 0; i < n; ++i) mx = std::fmax(mx, std::fabs(b[i]));
    double e = 0;
    for (size_t i = 0; i < n; ++i) e = std::fmax(e, std::fabs(a[i] - b[i]) / (std::fabs(b[i]) + 0.05 * mx + 1e-30));
    const bool ok = e < tol;
    std::printf("%-28s max rel err %.3e  %s\n", name, e, ok ? "ok" : "FAIL");
    if (!ok) ++fails;
}
static double dsilu(double x) { return x / (1.0 + std::exp(-x)); }
static double dsp(double x) { return x > 20 ? x : std::log1p(std::exp(x)); }

static void test_math() {
    double me = 0, ms = 0, mp = 0;
    for (int i = 0; i < 2000000; ++i) {
        float x = rnd(-87.f, 88.f);
        me = std::fmax(me, std::fabs(exp_fast(x) - std::exp((double)x)) / std::exp((double)x));
        float y = rnd(-30.f, 30.f);
        ms = std::fmax(ms, std::fabs(silu_f(y) - dsilu(y)) / (std::fabs(dsilu(y)) + 1e-30));
        float z = rnd(-30.f, 30.f);
        mp = std::fmax(mp, std::fabs(softplus_f(z) - dsp(z)) / dsp(z));
    }
    std::printf("exp_fast  max rel err %.3e (x in [-87,88])\nsilu_f    max rel err %.3e (x in [-30,30])\nsoftplus  max rel err %.3e\n",
                me, ms, mp);
    if (me > 1e-6 || ms > 1e-6 || mp > 1e-6) ++fails;
    if (exp_fast(-INFINITY) != 0.f || exp_fast(-1000.f) != 0.f) { std::puts("exp underflow FAIL"); ++fails; }
}

static void test_elementwise() {
    const int n = 5120 + 3;
    auto g = rvec(n, -8, 8), u = rvec(n, -2, 2), a = rvec(n, -5, 5);
    std::vector<float> o(n);
    std::vector<double> r(n);
    silu_mul_range(g.data(), u.data(), o.data(), 0, n);
    for (int i = 0; i < n; ++i) r[i] = dsilu(g[i]) * u[i];
    check("silu_mul", o.data(), r.data(), n);
    // ranged in two pieces must equal whole
    std::vector<float> o2(n);
    silu_mul_range(g.data(), u.data(), o2.data(), 0, 1000);
    silu_mul_range(g.data(), u.data(), o2.data(), 1000, n);
    if (std::memcmp(o.data(), o2.data(), n * 4)) { std::puts("silu_mul ranged mismatch FAIL"); ++fails; }
    add_range(g.data(), u.data(), o.data(), 0, n);
    for (int i = 0; i < n; ++i) r[i] = (double)g[i] + u[i];
    check("add", o.data(), r.data(), n);
    auto w = rvec(n, 0.5f, 1.5f);
    rmsnorm(a.data(), w.data(), o.data(), n, 1e-6f);
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double)a[i] * a[i];
    for (int i = 0; i < n; ++i) r[i] = a[i] / std::sqrt(ss / n + 1e-6) * w[i];
    check("rmsnorm", o.data(), r.data(), n);
    // split sumsq
    double s1 = rmsnorm_sumsq_range(a.data(), 0, 2000) + rmsnorm_sumsq_range(a.data(), 2000, n);
    if (std::fabs(s1 - ss) / ss > 1e-6) { std::puts("sumsq split FAIL"); ++fails; }
}

static void test_conv() {
    const int C = 10240 + 5;
    auto in = rvec(C, -2, 2), w = rvec((size_t)C * 4, -1, 1), st = rvec((size_t)C * 3, -2, 2);
    std::vector<float> st2 = st, out(C);
    std::vector<double> r(C);
    std::vector<float> rs((size_t)C * 3);
    for (int c = 0; c < C; ++c) {
        double acc = 0;
        for (int k = 0; k < 3; ++k) acc += (double)w[c * 4 + k] * st[c * 3 + k];
        acc += (double)w[c * 4 + 3] * in[c];
        r[c] = dsilu(acc);
        rs[c * 3] = st[c * 3 + 1]; rs[c * 3 + 1] = st[c * 3 + 2]; rs[c * 3 + 2] = in[c];
    }
    conv_step_range(in.data(), w.data(), st2.data(), out.data(), 0, 7000);
    conv_step_range(in.data(), w.data(), st2.data(), out.data(), 7000, C);
    check("conv_step out", out.data(), r.data(), C);
    if (std::memcmp(rs.data(), st2.data(), rs.size() * 4)) { std::puts("conv state FAIL"); ++fails; }
}

// Straightforward scalar double implementation of the Gated DeltaNet recurrence (model description):
//   q,k <- l2norm (eps 1e-6 inside sqrt), q *= dk^-1/2 ; g = A_neg*softplus(a+dt_bias) ; S *= exp(g)
//   kv = S^T k ; d = beta*(v - kv) ; S += k d^T ; o = S^T q ; y = rmsnorm(o)*w * silu(z)
static void gdn_ref(const GdnArgs& a, std::vector<double>& S, std::vector<double>& y) {
    const int DK = 128, DV = 128, ratio = a.n_v_heads / a.n_k_heads;
    for (int h = 0; h < a.n_v_heads; ++h) {
        const int kh = h / ratio;
        std::vector<double> q(DK), k(DK), v(DV);
        double qs = 0, ks = 0;
        for (int i = 0; i < DK; ++i) {
            q[i] = a.conv_out[kh * DK + i];
            k[i] = a.conv_out[a.n_k_heads * DK + kh * DK + i];
            qs += q[i] * q[i]; ks += k[i] * k[i];
        }
        for (int i = 0; i < DK; ++i) { q[i] = q[i] / std::sqrt(qs + 1e-6) / std::sqrt((double)DK); k[i] /= std::sqrt(ks + 1e-6); }
        for (int j = 0; j < DV; ++j) v[j] = a.conv_out[2 * a.n_k_heads * DK + h * DV + j];
        const double g = (double)a.A_neg[h] * dsp((double)a.alpha[h] + a.dt_bias[h]);
        const double dec = std::exp(g), beta = 1.0 / (1.0 + std::exp(-(double)a.beta_logit[h]));
        double* Sh = &S[(size_t)h * DK * DV];
        std::vector<double> o(DV, 0.0);
        for (int j = 0; j < DV; ++j) {
            double kv = 0;
            for (int i = 0; i < DK; ++i) { Sh[i * DV + j] *= dec; kv += Sh[i * DV + j] * k[i]; }
            const double d = (v[j] - kv) * beta;
            for (int i = 0; i < DK; ++i) { Sh[i * DV + j] += k[i] * d; o[j] += Sh[i * DV + j] * q[i]; }
        }
        double ss = 0;
        for (int j = 0; j < DV; ++j) ss += o[j] * o[j];
        const double r = 1.0 / std::sqrt(ss / DV + a.eps);
        for (int j = 0; j < DV; ++j) y[h * DV + j] = o[j] * r * a.norm_w[j] * dsilu(a.z[h * DV + j]);
    }
}

static void test_gdn() {
    const int NSTEP = getenv("NSTEP") ? atoi(getenv("NSTEP")) : 3;
    const int Hk = 16, Hv = 48, DK = 128, DV = 128;
    auto conv = rvec((size_t)(2 * Hk * DK + Hv * DV), -2, 2);
    auto alpha = rvec(Hv, -3, 3), beta = rvec(Hv, -3, 3), dtb = rvec(Hv, -1, 1), z = rvec((size_t)Hv * DV, -4, 4);
    auto nw = rvec(DV, 0.5f, 1.5f);
    std::vector<float> Aneg(Hv);
    for (auto& x : Aneg) x = -std::exp(rnd(-3.f, 1.f));  // A_neg = -exp(A_log)
    auto S0 = rvec((size_t)Hv * DK * DV, -0.5f, 0.5f);
    std::vector<float> S = S0, out((size_t)Hv * DV);
    GdnArgs a{conv.data(), alpha.data(), beta.data(), Aneg.data(), dtb.data(), z.data(), nw.data(), S.data(), out.data(),
              Hk, Hv, DK, DV, 1e-6f};
    std::vector<double> Sr(S0.begin(), S0.end()), yr((size_t)Hv * DV);
    gdn_step_range(a, 0, 20);
    gdn_step_range(a, 20, Hv);
    for (int step = 0; step < NSTEP; ++step) {  // also checks recurrence over repeated steps (same inputs)
        gdn_ref(a, Sr, yr);
        if (step < NSTEP - 1) { gdn_step_range(a, 0, Hv); }
    }
    // after step 0 (done above) + 2 more kernel steps == 3 ref steps
    check("gdn_step out (3 steps)", out.data(), yr.data(), yr.size());
    std::vector<double> Sd(Sr.size());
    check("gdn_step state (3 steps)", S.data(), Sr.data(), Sr.size());
    // gated rmsnorm standalone
    auto o = rvec((size_t)Hv * DV, -3, 3);
    std::vector<float> go((size_t)Hv * DV);
    std::vector<double> gr(go.size());
    gated_rmsnorm_range(o.data(), z.data(), nw.data(), go.data(), DV, 1e-6f, 0, Hv);
    for (int h = 0; h < Hv; ++h) {
        double ss = 0;
        for (int j = 0; j < DV; ++j) ss += (double)o[h * DV + j] * o[h * DV + j];
        for (int j = 0; j < DV; ++j) gr[h * DV + j] = o[h * DV + j] / std::sqrt(ss / DV + 1e-6) * nw[j] * dsilu(z[h * DV + j]);
    }
    check("gated_rmsnorm", go.data(), gr.data(), gr.size());
}

static void test_attn(int n_keys, int S) {
    const int Hq = 24, Hkv = 4, D = 256, R = 64, max_ctx = n_keys + 8, pos = n_keys - 1;
    auto qg = rvec((size_t)Hq * 2 * D, -3, 3), k = rvec((size_t)Hkv * D, -3, 3), v = rvec((size_t)Hkv * D, -2, 2);
    auto qn = rvec(D, 0.2f, 2.f), kn = rvec(D, 0.2f, 2.f);
    std::vector<float> inv(R / 2), cs(R);
    for (int i = 0; i < R / 2; ++i) inv[i] = 1.0f / std::pow(1e7f, (float)(2 * i) / (float)R);
    attn_rope_table(pos, inv.data(), R, cs.data());
    std::vector<float> kc((size_t)Hkv * max_ctx * D), vc(kc.size()), qo((size_t)Hq * D);
    for (auto& x : kc) x = rnd(-2, 2);
    for (auto& x : vc) x = rnd(-2, 2);
    AttnPrepArgs p{qg.data(), k.data(), v.data(), qn.data(), kn.data(), cs.data(), qo.data(), kc.data(), vc.data(),
                   Hq, Hkv, D, R, max_ctx, pos, 1e-6f};
    attn_prep_range(p, 0, 10);
    attn_prep_range(p, 10, Hq + Hkv);
    // reference prep (double)
    auto prep_ref = [&](const float* x, const float* w, std::vector<double>& y) {
        double ss = 0;
        for (int t = 0; t < D; ++t) ss += (double)x[t] * x[t];
        std::vector<double> n(D);
        for (int t = 0; t < D; ++t) n[t] = x[t] / std::sqrt(ss / D + 1e-6) * w[t];
        y = n;
        for (int t = 0; t < R; ++t) {
            int i = t % (R / 2);
            double f = (double)((float)pos * inv[i]), c = std::cos(f), s = std::sin(f);
            y[t] = t < R / 2 ? n[t] * c - n[t + R / 2] * s : n[t] * c + n[t - R / 2] * s;
        }
    };
    std::vector<double> y, qref((size_t)Hq * D);
    for (int h = 0; h < Hq; ++h) { prep_ref(&qg[(size_t)h * 2 * D], qn.data(), y); std::copy(y.begin(), y.end(), &qref[(size_t)h * D]); }
    check("attn_prep q", qo.data(), qref.data(), qref.size(), 2e-5);
    std::vector<double> kref(D), kcw(D);
    double ek = 0;
    for (int h = 0; h < Hkv; ++h) {
        prep_ref(&k[(size_t)h * D], kn.data(), y);
        for (int t = 0; t < D; ++t) ek = std::fmax(ek, std::fabs(kc[((size_t)h * max_ctx + pos) * D + t] - y[t]) / (std::fabs(y[t]) + 1e-2));
        if (std::memcmp(&vc[((size_t)h * max_ctx + pos) * D], &v[(size_t)h * D], D * 4)) { std::puts("v copy FAIL"); ++fails; }
    }
    std::printf("%-28s max rel err %.3e  %s\n", "attn_prep k cache", ek, ek < 2e-5 ? "ok" : "FAIL");
    if (ek >= 2e-5) ++fails;
    // decode with moderate-magnitude q (peaky softmax would only measure fp32 conditioning of exp)
    for (auto& x : qo) x = rnd(-0.3f, 0.3f);
    std::vector<float> part(attn_part_floats(Hq, S, D)), out((size_t)Hq * D);
    AttnDecodeArgs d{qo.data(), qg.data(), kc.data(), vc.data(), part.data(), out.data(), Hq, Hkv, D, max_ctx, n_keys, S};
    const int items = Hkv * S;
    attn_decode_range(d, 0, items / 3);
    attn_decode_range(d, items / 3, items);
    attn_merge_range(d, 0, 7);
    attn_merge_range(d, 7, Hq);
    std::vector<double> ref((size_t)Hq * D);
    for (int h = 0; h < Hq; ++h) {
        const int g = h / (Hq / Hkv);
        std::vector<double> sc(n_keys);
        double mx = -1e300;
        for (int j = 0; j < n_keys; ++j) {
            double dt = 0;
            for (int e = 0; e < D; ++e) dt += (double)qo[(size_t)h * D + e] * kc[((size_t)g * max_ctx + j) * D + e];
            sc[j] = dt / std::sqrt((double)D);
            mx = std::fmax(mx, sc[j]);
        }
        double L = 0;
        std::vector<double> acc(D, 0.0);
        for (int j = 0; j < n_keys; ++j) {
            double e = std::exp(sc[j] - mx);
            L += e;
            for (int t = 0; t < D; ++t) acc[t] += e * vc[((size_t)g * max_ctx + j) * D + t];
        }
        for (int t = 0; t < D; ++t) ref[(size_t)h * D + t] = acc[t] / L / (1.0 + std::exp(-(double)qg[(size_t)h * 2 * D + D + t]));
    }
    char nm[64];
    std::snprintf(nm, sizeof nm, "attn_decode n=%d S=%d", n_keys, S);
    check(nm, out.data(), ref.data(), ref.size(), 2e-5);
}

static void test_embed() {
    for (int bits : {2, 3, 4, 5, 6, 8}) {
        const int N = 5, K = 192, C = K / 32, G = K / 64;
        // pack per tools/qv_format.py spec (bit-plane stack, planes of width 4/2/1)
        static const std::vector<std::pair<int, int>> planes[9] = {{}, {}, {{2, 0}}, {{2, 0}, {1, 2}}, {{4, 0}}, {{4, 0}, {1, 4}},
                                                                   {{4, 0}, {2, 4}}, {}, {{4, 0}, {4, 4}}};
        std::vector<uint32_t> qw((size_t)N * C * bits, 0);
        std::vector<int> qv((size_t)N * K);
        std::vector<uint16_t> sm((size_t)N * G * 2);
        std::vector<double> ref((size_t)N * K);
        for (auto& x : qv) x = rng() % (1u << bits);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < C; ++c) {
                int wi = 0;
                for (auto [w, sh] : planes[bits]) {
                    for (int j = 0; j < w; ++j) {
                        uint32_t word = 0;
                        for (int f = 0; f < 32 / w; ++f) {
                            const int idx = j + w * f;
                            word |= (uint32_t)((qv[(size_t)r * K + c * 32 + idx] >> sh) & ((1 << w) - 1)) << (w * f);
                        }
                        qw[((size_t)r * C + c) * bits + wi++] = word;
                    }
                }
            }
            for (int g = 0; g < G; ++g) {
                float s = rnd(0.01f, 0.5f), m = rnd(-1, 1);
                sm[((size_t)r * G + g) * 2] = _cvtss_sh(s, 0);
                sm[((size_t)r * G + g) * 2 + 1] = _cvtss_sh(m, 0);
            }
        }
        for (int r = 0; r < N; ++r)
            for (int i = 0; i < K; ++i)
                ref[(size_t)r * K + i] = (double)_cvtsh_ss(sm[((size_t)r * G + i / 64) * 2]) * qv[(size_t)r * K + i] +
                                         _cvtsh_ss(sm[((size_t)r * G + i / 64) * 2 + 1]);
        QRows E{qw.data(), sm.data(), K, bits};
        std::vector<float> out(K);
        for (int r = 0; r < N; ++r) {
            embed_row(E, r, out.data());
            char nm[32];
            std::snprintf(nm, sizeof nm, "embed_row %d-bit row %d", bits, r);
            check(nm, out.data(), &ref[(size_t)r * K], K, 1e-6);
        }
    }
}

static void test_argmax() {
    const int n = 248320 + 3;  // vocab-like, odd tail
    for (int trial = 0; trial < 20; ++trial) {
        auto x = rvec(n, -10, 10);
        int pos = rng() % n;
        x[pos] = 50.f;  // unique max...
        if (trial % 2) x[(pos + 1000) % n] = 50.f;  // ...or a tie: smaller index must win
        int ref = 0;
        for (int i = 0; i < n; ++i) if (x[i] > x[ref]) ref = i;
        const int P = 7;
        float pv[P]; int pi[P];
        for (int p = 0; p < P; ++p) argmax_range(x.data(), (int)((long)n * p / P), (int)((long)n * (p + 1) / P), &pv[p], &pi[p]);
        int got;
        argmax_merge(pv, pi, P, &got);
        if (got != ref) { std::printf("argmax FAIL trial %d got %d ref %d\n", trial, got, ref); ++fails; return; }
    }
    std::puts("argmax (ranged+merge, ties)  ok");
}

int main() {
    test_math();
    test_elementwise();
    test_conv();
    test_gdn();
    test_attn(1, 4);
    test_attn(37, 4);
    test_attn(1000, 16);
    test_attn(4096, 80);
    test_embed();
    test_argmax();
    std::printf("%s (%d failures)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}
