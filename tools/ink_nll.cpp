// qv_ink_nll - teacher-forced negative log-likelihood by position for the Inkling CUDA engine (docs/LOOPS.md).
//   qv_ink_nll --model X.qv --ids doc.i32 --kv int4|fp16 --max-ctx N --windows 0,4096,16384,65536 [--win 512] [--out nll.f32]
// The context is filled by the batched prefill (the path every real prompt takes, KV written in the selected format), and
// the NLL is measured over `win` tokens after each window start with the DECODE graph path (step()), i.e. the code path that
// generates tokens.  The prefill resumes after each window, so one pass covers all windows.  Prints one JSON line per window
// (mean NLL, standard error, token count) and optionally writes the per-position NLL (float32, NaN where not measured) to --out.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "qv/engine.h"

static std::vector<int> read_i32(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::vector<int> v;
    int x;
    while (f.read((char*)&x, 4)) v.push_back(x);
    return v;
}

static double token_nll(const std::vector<float>& lg, int tok) {
    float m = lg[0];
    for (float v : lg) m = std::max(m, v);
    double z = 0;
    for (float v : lg) z += std::exp((double)v - m);
    return -((double)lg[tok] - m - std::log(z));
}

int main(int argc, char** argv) {
    std::string model, ids_path, out, wins = "0", kv = "int4";
    int max_ctx = 0, win = 512, chunk = 2048;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--model") model = next();
        else if (a == "--ids") ids_path = next();
        else if (a == "--out") out = next();
        else if (a == "--windows") wins = next();
        else if (a == "--win") win = std::stoi(next());
        else if (a == "--kv") kv = next();
        else if (a == "--max-ctx") max_ctx = std::stoi(next());
        else if (a == "--prefill-chunk") chunk = std::stoi(next());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    std::vector<int> ids = read_i32(ids_path), starts;
    for (size_t p = 0; p < wins.size();) { starts.push_back(std::stoi(wins.substr(p))); p = wins.find(',', p); if (p == std::string::npos) break; ++p; }
    std::sort(starts.begin(), starts.end());
    const int N = (int)ids.size();
    if (starts.back() + win + 1 > N) throw std::runtime_error("--ids too short for the last window");
    qv::ModelFile mf = qv::ModelFile::open(model);
    qv::EngineOptions opt;
    opt.max_ctx = max_ctx ? max_ctx : starts.back() + win + 16;
    opt.prefill_chunk = chunk;
    opt.kv_mode = kv == "int4" ? 1 : kv == "fp16" ? 2 : 0;
    auto eng = qv::make_cuda_engine(mf, opt);
    eng->reset();
    std::vector<float> per(N, NAN), lg;
    int pos = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int s : starts) {
        if (s > pos) { eng->prefill(ids.data() + pos, s - pos, pos); pos = s; }
        // logits for ids[s]: after prefill they are the last row; for s == 0 there is nothing yet
        double sum = 0, sum2 = 0;
        int cnt = 0;
        if (pos > 0) {
            eng->logits(lg);
            const double nl = token_nll(lg, ids[pos]);
            per[pos - 1] = (float)nl; sum += nl; sum2 += nl * nl; ++cnt;
        }
        for (int j = 0; j < win; ++j) {
            eng->step(ids[pos], pos);
            eng->logits(lg);
            const double nl = token_nll(lg, ids[pos + 1]);
            per[pos] = (float)nl; sum += nl; sum2 += nl * nl; ++cnt;
            ++pos;
        }
        const double mean = sum / cnt, var = std::max(0.0, sum2 / cnt - mean * mean);
        std::printf("{\"kv\": \"%s\", \"start\": %d, \"win\": %d, \"n\": %d, \"nll\": %.5f, \"se\": %.5f, \"elapsed_s\": %.1f}\n", kv.c_str(), s, win, cnt, mean,
                    std::sqrt(var / cnt), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        std::fflush(stdout);
    }
    if (!out.empty()) { std::ofstream o(out, std::ios::binary); o.write((const char*)per.data(), per.size() * 4); }
    return 0;
}
