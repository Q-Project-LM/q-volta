// qv_ink_cpu - driver for the Inkling-2-Mini CPU engine (src/runtime/cpu_inkling_engine.cpp).
//   dump  : qv_ink_cpu --model X.qv --ids ids.i32 --dump DIR [--gen 32] [--threads 32]
//           writes DIR/{hidden.f32 [50][T][2304] (emb, 48 applications, final norm), router.i8 [48][T][8], logits_last.f32,
//           gen_ids.i32, gen_top20_ids.i32 [N][20], gen_top20_val.f32 [N][20], gen_router.i8 [N][48][8]}  (compare: tests/check_inkling_cpu.py)
//   bench : qv_ink_cpu --model X.qv --ids ids.i32 --bench --ctx 128,512,2048 [--n 64] [--chunk 256] [--threads 32]
//           per context: timed prefill of ctx tokens (tok/s) and timed greedy decode of n tokens (tok/s, ms/token), JSON lines.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_inkling_engine.h"
#include "qv/engine.h"

static std::vector<int> read_i32(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::vector<int> v;
    int x;
    while (f.read((char*)&x, 4)) v.push_back(x);
    return v;
}
template <class T> static void write_vec(const std::string& p, const std::vector<T>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char*)v.data(), (std::streamsize)(v.size() * sizeof(T)));
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    std::string model, ids_path, dump, ctxs = "128,512,2048";
    bool nll = false;
    std::string nll_out;
    bool bench = false;
    int gen = 0, n = 64, threads = 32, chunk = 256, max_ctx = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--model") model = next();
        else if (a == "--ids") ids_path = next();
        else if (a == "--dump") dump = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--bench") bench = true;
        else if (a == "--nll") nll = true;
        else if (a == "--nll-out") nll_out = next();  // per-position NLL of ids[pos+1] (float32, one value per predicted token), for loop/long-context studies
        else if (a == "--ctx") ctxs = next();
        else if (a == "--n") n = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--max-ctx") max_ctx = std::stoi(next());
        else throw std::runtime_error("unknown option " + a);
    }
    try {
        std::vector<int> ids = read_i32(ids_path);
        qv::ModelFile mf = qv::ModelFile::open(model);
        qv::EngineOptions opt;
        opt.threads = threads;
        opt.prefill_chunk = chunk;
        opt.dump_hidden = !dump.empty();
        std::vector<int> cl;
        if (bench) for (size_t p = 0; p < ctxs.size();) { cl.push_back(std::stoi(ctxs.substr(p))); p = ctxs.find(',', p); if (p == std::string::npos) break; ++p; }
        opt.max_ctx = max_ctx ? max_ctx : (bench ? *std::max_element(cl.begin(), cl.end()) + n + 8 : (int)ids.size() + gen + 8);
        double t0 = now();
        double nll_sum = 0; long nll_n = 0;
        std::vector<float> nll_pos;
        if (nll) {
            opt.row_logits = [&](int pos, const float* lg) {
                if (pos + 1 >= (int)ids.size()) return;
                float m = lg[0];
                for (int i = 1; i < 200058; ++i) m = std::max(m, lg[i]);
                double z = 0;
                for (int i = 0; i < 200058; ++i) z += std::exp((double)lg[i] - m);
                const double nl = -((double)lg[ids[pos + 1]] - m - std::log(z));
                nll_sum += nl;
                if (!nll_out.empty()) { if ((int)nll_pos.size() <= pos) nll_pos.resize(pos + 1, 0.f); nll_pos[pos] = (float)nl; }
                nll_n++;
            };
        }
        auto eng = qv::make_cpu_engine(mf, opt);
        auto* info = dynamic_cast<qv::InklingCpuInfo*>(eng.get());
        std::fprintf(stderr, "loaded in %.1f s, decode bytes/token %.3f GB\n", now() - t0, info->decode_bytes_per_token() / 1e9);
        {
            std::ifstream sm("/proc/self/smaps_rollup");
            for (std::string l; std::getline(sm, l);) if (l.rfind("AnonHugePages", 0) == 0 || l.rfind("Rss", 0) == 0) std::fprintf(stderr, "%s\n", l.c_str());
        }
        if (nll) {
            t0 = now();
            eng->prefill(ids.data(), (int)ids.size(), 0);
            if (!nll_out.empty()) { std::ofstream o(nll_out, std::ios::binary); o.write((const char*)nll_pos.data(), nll_pos.size() * 4); }
            std::printf("{\"nll\": %.5f, \"n\": %ld, \"seconds\": %.1f}\n", nll_sum / nll_n, nll_n, now() - t0);
            return 0;
        }
        if (bench) {
            for (int ctx : cl) {
                if (ctx > (int)ids.size()) throw std::runtime_error("--ids has fewer tokens than --ctx");
                eng->reset();
                double tp = now();
                eng->prefill(ids.data(), ctx, 0);
                tp = now() - tp;
                std::vector<const char*> nm; std::vector<double> ms; std::vector<long> cn;
                info->phase_report(nm, ms, cn);
                for (size_t i = 0; i < nm.size(); ++i) std::printf("  prefill phase %-14s %9.1f ms  (%.1f%%)\n", nm[i], ms[i], 100.0 * ms[i] / (tp * 1e3));
                std::vector<int> out;
                int tok = eng->argmax();
                eng->decode_greedy(tok, ctx, 4, out, false);  // warm-up
                tok = out.back();
                info->phase_report(nm, ms, cn);
                double td = now();
                eng->decode_greedy(tok, ctx + 4, n, out, false);
                td = now() - td;
                std::printf("{\"ctx\": %d, \"prefill_s\": %.4f, \"prefill_tok_s\": %.2f, \"decode_n\": %d, \"decode_s\": %.4f, \"decode_tok_s\": %.3f, \"ms_per_token\": %.2f, \"gb_s\": %.2f}\n",
                            ctx, tp, ctx / tp, n, td, n / td, 1e3 * td / n, info->decode_bytes_per_token() * n / td / 1e9);
                info->phase_report(nm, ms, cn);
                for (size_t i = 0; i < nm.size(); ++i) std::printf("  phase %-14s %9.2f ms/token  (%ld calls)\n", nm[i], ms[i] / n, cn[i]);
                std::fflush(stdout);
            }
            return 0;
        }
        // dump mode
        const int T = (int)ids.size();
        t0 = now();
        eng->prefill(ids.data(), T, 0);
        std::fprintf(stderr, "prefill %d tokens: %.2f s\n", T, now() - t0);
        write_vec(dump + "/hidden.f32", eng->hidden_dump());
        write_vec(dump + "/router.i8", info->router_dump());
        std::vector<float> lg;
        eng->logits(lg);
        write_vec(dump + "/logits_last.f32", lg);
        std::vector<int> gids, t20i;
        std::vector<float> t20v;
        std::vector<int8_t> grt;
        auto top20 = [&](const std::vector<float>& l) {
            std::vector<int> ix(l.size());
            for (size_t i = 0; i < ix.size(); ++i) ix[i] = (int)i;
            std::partial_sort(ix.begin(), ix.begin() + 20, ix.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
            for (int i = 0; i < 20; ++i) { t20i.push_back(ix[i]); t20v.push_back(l[ix[i]]); }
            return ix[0];
        };
        int tok = top20(lg);
        for (int s = 0; s < gen; ++s) {
            gids.push_back(tok);
            if (s == gen - 1) break;
            eng->step(tok, T + s);
            const auto& rd = info->router_dump();
            grt.insert(grt.end(), rd.begin(), rd.end());
            eng->logits(lg);
            tok = top20(lg);
        }
        write_vec(dump + "/gen_ids.i32", gids);
        write_vec(dump + "/gen_top20_ids.i32", t20i);
        write_vec(dump + "/gen_top20_val.f32", t20v);
        write_vec(dump + "/gen_router.i8", grt);
        std::fprintf(stderr, "done\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
