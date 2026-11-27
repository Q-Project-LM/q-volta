// Inkling checkpoints with MTP on: L1 (save_checkpoint) and L2 (save_kv_prefix + save_checkpoint into a trashed engine) must restore a state
// that continues bit-identically: same greedy ids, same drafter acceptance counts, same next-step logits as the uninterrupted run.
// Usage: test_ink_ckpt MODEL.qv MTP.qv PROMPT.i32 [SPLIT=0.6] [N=48] [DEPTH=3] [OTHER.i32]   Needs the GPU; skips (exit 0) without a model.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "qv/engine.h"

static std::vector<int> read_i32(const char* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<int> v;
    int x;
    while (f.read((char*)&x, 4)) v.push_back(x);
    return v;
}
struct Out { std::vector<int> ids; std::vector<float> logits; long cycles = 0, accepted = 0; };
static Out gen(qv::Engine& e, int pos, int n) {
    qv::MtpStats s0, s1;
    e.mtp_stats(s0);
    Out o;
    o.ids = {e.argmax()};
    std::vector<int> rest;
    e.decode_greedy(o.ids[0], pos, n - 1, rest, false);
    o.ids.insert(o.ids.end(), rest.begin(), rest.end());
    e.step(o.ids.back(), pos + n - 1);
    e.logits(o.logits);
    e.mtp_stats(s1);
    o.cycles = s1.cycles - s0.cycles; o.accepted = s1.accepted - s0.accepted;
    return o;
}
static bool same(const char* tag, const Out& a, const Out& b) {
    const bool ok = a.ids == b.ids && a.logits == b.logits && a.cycles == b.cycles && a.accepted == b.accepted;
    std::printf("%s: ids identical=%d logits identical=%d cycles %ld/%ld accepted %ld/%ld -> %s\n", tag, (int)(a.ids == b.ids), (int)(a.logits == b.logits),
                a.cycles, b.cycles, a.accepted, b.accepted, ok ? "ok" : "MISMATCH");
    return ok;
}
int main(int argc, char** argv) {
    if (argc < 4 || !std::ifstream(argv[1])) { std::printf("test_ink_ckpt: no model, skipped\n"); return 0; }
    auto prompt = read_i32(argv[3]);
    const double split = argc > 4 ? std::atof(argv[4]) : 0.6;
    const int n = argc > 5 ? std::atoi(argv[5]) : 48, depth = argc > 6 ? std::atoi(argv[6]) : 3;
    std::vector<int> other = argc > 7 ? read_i32(argv[7]) : std::vector<int>(prompt.rbegin(), prompt.rend());
    auto mf = qv::ModelFile::open(argv[1]);
    qv::EngineOptions opt;
    opt.max_ctx = std::getenv("QV_TEST_MAXCTX") ? std::atoi(std::getenv("QV_TEST_MAXCTX")) : 8192;
    opt.kv_mode = 1; opt.mtp = true; opt.mtp_depth = depth; opt.mtp_file = argv[2];
    auto eng = qv::make_cuda_engine(mf, opt);
    qv::Engine& e = *eng;
    if (!e.has_checkpoint() || !e.has_kv_checkpoint()) { std::printf("FAIL: no checkpoint support with MTP\n"); return 1; }
    const int T = (int)prompt.size(), P = std::max(16, (int)(T * split));
    e.set_mtp_enabled(true);
    e.reset();
    e.prefill(prompt.data(), P, 0);
    std::vector<uint8_t> ck1;
    if (!e.save_checkpoint(ck1)) { std::printf("FAIL: save_checkpoint\n"); return 1; }
    e.prefill(prompt.data() + P, T - P, P);
    std::vector<uint8_t> ck2, kv2;
    e.save_kv_prefix(kv2, T);
    e.save_checkpoint(ck2);
    Out ref = gen(e, T, n);
    int fail = 0;
    // L1: restore the mid-prompt checkpoint (global KV stays in VRAM), re-prefill the tail, generate again
    if (!e.restore_checkpoint(ck1)) { std::printf("FAIL: restore_checkpoint\n"); return 1; }
    e.prefill(prompt.data() + P, T - P, P);
    fail += !same("L1 restore", ref, gen(e, T, n));
    // L2: trash the whole engine with another prompt, then restore KV prefix + end checkpoint
    e.reset();
    const int To = std::min<int>((int)other.size(), T);
    e.prefill(other.data(), To, 0);
    gen(e, To, 16);
    if (!e.restore_kv_prefix(kv2) || !e.restore_checkpoint(ck2)) { std::printf("FAIL: L2 restore\n"); return 1; }
    fail += !same("L2 restore after trash", ref, gen(e, T, n));
    // informational: a prompt prefilled whole (chunk grid from 0) vs the same prompt prefilled in two pieces at the split point (what a restored
    // checkpoint implies). Batched prefill numerics may depend on the chunk boundaries; this is the floor for "cached == uncached" at the server.
    e.reset();
    e.prefill(prompt.data(), T, 0);
    const Out whole = gen(e, T, n);
    std::printf("info: whole-prompt prefill vs split-at-%d prefill: ids identical=%d logits identical=%d\n", P, (int)(whole.ids == ref.ids), (int)(whole.logits == ref.logits));
    std::printf("prompt %d split %d ckpt %.1f MB kv-prefix %.1f MB: %s\n", T, P, ck1.size() / 1048576.0, kv2.size() / 1048576.0, fail ? "FAIL" : "PASS");
    return fail != 0;
}
