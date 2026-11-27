// Scripted engine for HTTP-layer tests (`qv_server --backend mock`): no model weights. After a prompt it replays a fixed token
// script (the tokenisation of --mock-text) and then emits EOS (or loops with --mock-loop). Logits are peaked on the scripted token with a
// few deterministic runners-up, so top-k / top-p / logprobs / seeded sampling behave like on a real model. Honours the
// engine contract (prefill leaves logits of the last row, step(token,pos), reset()).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>

#include "qv/engine.h"

namespace qv {

class MockEngine : public Engine {
   public:
    MockEngine(int vocab, std::vector<int> script, int eos, bool loop, int delay_us, int n_seq = 1) : script_(std::move(script)), eos_(eos), loop_(loop), delay_us_(delay_us), st_(std::max(1, n_seq)) {
        cfg_.vocab = vocab;
        cfg_.eos = eos;
        cfg_.arch = "mock";
        if (script_.empty()) script_.push_back(eos);
    }
    int n_seq() const override { return (int)st_.size(); }
    void select_seq(int i) override { if (i < 0 || i >= (int)st_.size()) throw std::runtime_error("sequence index out of range"); cur_ = i; }
    void step(int tok, int pos) override {
        delay(1);
        Seq& s = st_[cur_];
        s.k = pos + 1 - s.prompt_end;
        if ((int)s.kv.size() <= pos) s.kv.resize((size_t)pos + 1, -1);
        s.kv[pos] = tok;
    }
    void prefill(const int* tokens, int T, int pos0) override {
        delay(T / 8 + 1);
        Seq& s = st_[cur_];
        s.prompt_end = pos0 + T;
        s.k = 0;
        if ((int)s.kv.size() < pos0 + T) s.kv.resize((size_t)pos0 + T, -1);
        for (int i = 0; i < T; ++i) s.kv[pos0 + i] = tokens ? tokens[i] : -1;
    }
    void prefill_embeds(const int*, const float*, const unsigned char* mask, int T, int pos0) override {
        for (int i = 0; i < T; ++i) n_embeds_ += mask[i] ? 1 : 0;
        prefill(nullptr, T, pos0);
    }
    long n_embeds_ = 0;  // image rows received through prefill_embeds (tests)
    void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool) override {
        out.resize(n);
        int tok = first;
        for (int i = 0; i < n; ++i) { step(tok, start_pos + i); tok = argmax(); out[i] = tok; }
    }
    bool has_speculative_decode() const override { return true; }
    bool has_constrained_mtp() const override { return true; }
    void decode_greedy_constrained(int first, int start_pos, int n, std::vector<int>& out, TokenConstraint* tc) override {
        out.clear();
        std::vector<float> lg;
        int tok = first;
        for (int i = 0; i < n; ++i) {
            step(tok, start_pos + i);
            int cand = argmax();
            if (tc && !tc->try_accept(cand)) {
                logits(lg);
                if (!tc->awaiting()) tc->mask_logits(lg.data(), lg.size());
                cand = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
                (void)tc->try_accept(cand);
            }
            out.push_back(cand);
            tok = cand;
            if (tc && (tc->is_eog(cand) || tc->is_dead())) break;
        }
    }
    void decode_speculative(int first, int start_pos, int n, const SpecSamplingParams& sp, std::vector<int>& out) override {
        decode_speculative_constrained(first, start_pos, n, sp, out, nullptr);
    }
    void decode_speculative_constrained(int first, int start_pos, int n, const SpecSamplingParams& sp, std::vector<int>& out, TokenConstraint* tc) override {
        out.clear();
        std::mt19937_64 rng(sp.seed);
        std::vector<float> lg;
        SpecDist dist;
        int tok = first;
        for (int i = 0; i < n; ++i) {
            step(tok, start_pos + i);
            logits(lg);
            build_dist_dense(lg.data(), cfg_.vocab, sp, dist, tc);
            tok = sample_dist(dist, rng);
            if (tc) (void)tc->try_accept(tok);
            out.push_back(tok);
            if (tc && (tc->is_eog(tok) || tc->is_dead())) break;
        }
    }
    int argmax() override { return scripted(); }
    void logits(std::vector<float>& out) override {
        const int V = cfg_.vocab;
        out.resize(V);
        for (int i = 0; i < V; ++i) out[i] = -6.0f + (float)((uint32_t)(i * 2654435761u) >> 20) / 4096.0f * 2.0f;  // deterministic noise in [-6,-4)
        int t = scripted();
        out[t] = 20.0f;
        const float ru[3] = {18.0f, 17.0f, 16.0f};
        for (int j = 0; j < 3; ++j) { int a = (t + 1 + j * 7919) % V; if (a != t) out[a] = ru[j]; }
    }
    void reset() override { st_[cur_].k = 0; st_[cur_].prompt_end = 0; st_[cur_].kv.clear(); }
    bool has_state_save() const override { return true; }
    void state_save() override { saved_ = st_[cur_]; }
    void state_restore() override { st_[cur_] = saved_; }
    bool has_checkpoint() const override { return true; }
    size_t checkpoint_bytes() const override { return sizeof(int) * 2; }
    bool save_checkpoint(std::vector<uint8_t>& out) override {
        out.resize(sizeof(int) * 2);
        std::memcpy(out.data(), &st_[cur_].k, sizeof(int));
        std::memcpy(out.data() + sizeof(int), &st_[cur_].prompt_end, sizeof(int));
        ++ckpt_saves_;
        return true;
    }
    bool restore_checkpoint(const std::vector<uint8_t>& in) override {
        if (in.size() != sizeof(int) * 2) return false;
        std::memcpy(&st_[cur_].k, in.data(), sizeof(int));
        std::memcpy(&st_[cur_].prompt_end, in.data() + sizeof(int), sizeof(int));
        ++ckpt_restores_;
        return true;
    }
    bool has_kv_checkpoint() const override { return true; }
    size_t kv_checkpoint_bytes(int n_tokens) const override {
        if (n_tokens < 0) return 0;
        return sizeof(uint32_t) + sizeof(int) + (size_t)n_tokens * mock_kv_bytes_per_tok_;
    }
    bool save_kv_prefix(std::vector<uint8_t>& out, int n_tokens) override {
        if (n_tokens < 0 || n_tokens > (int)st_[cur_].kv.size()) return false;
        const size_t min_b = sizeof(uint32_t) + sizeof(int) + (size_t)n_tokens * sizeof(int);
        const size_t pad_b = sizeof(uint32_t) + sizeof(int) + (size_t)n_tokens * mock_kv_bytes_per_tok_;
        out.assign(std::max(min_b, pad_b), 0);
        const uint32_t magic = 0x4D4B5631u;
        std::memcpy(out.data(), &magic, sizeof(magic));
        std::memcpy(out.data() + sizeof(magic), &n_tokens, sizeof(int));
        if (n_tokens > 0) std::memcpy(out.data() + sizeof(magic) + sizeof(int), st_[cur_].kv.data(), (size_t)n_tokens * sizeof(int));
        ++kv_saves_;
        return true;
    }
    bool restore_kv_prefix(const std::vector<uint8_t>& in) override {
        if (in.size() < sizeof(uint32_t) + sizeof(int)) return false;
        uint32_t magic = 0;
        int n_tokens = -1;
        std::memcpy(&magic, in.data(), sizeof(magic));
        std::memcpy(&n_tokens, in.data() + sizeof(magic), sizeof(int));
        if (magic != 0x4D4B5631u || n_tokens < 0) return false;
        if (in.size() < sizeof(uint32_t) + sizeof(int) + (size_t)n_tokens * sizeof(int)) return false;
        st_[cur_].kv.resize((size_t)n_tokens);
        if (n_tokens > 0) std::memcpy(st_[cur_].kv.data(), in.data() + sizeof(magic) + sizeof(int), (size_t)n_tokens * sizeof(int));
        ++kv_restores_;
        return true;
    }
    const std::vector<int>& mock_kv(int sq = 0) const { return st_[sq].kv; }
    long ckpt_saves_ = 0, ckpt_restores_ = 0, kv_saves_ = 0, kv_restores_ = 0;
    size_t mock_kv_bytes_per_tok_ = sizeof(int);
    const std::vector<float>& hidden_dump() const override { return empty_; }
    const std::vector<float>& logits_dump() const override { return empty_; }
    void sync() override {}
    size_t device_bytes_weights() const override { return 0; }
    size_t device_bytes_state() const override { return 0; }

   private:
    std::vector<int> script_;
    int eos_;
    bool loop_;
    int delay_us_;
    struct Seq { int k = 0, prompt_end = 0; std::vector<int> kv; };
    std::vector<Seq> st_;
    Seq saved_;
    int cur_ = 0;
    std::vector<float> empty_;
    int scripted() const {
        const int k_ = st_[cur_].k;
        if (k_ < (int)script_.size()) return script_[k_];
        if (loop_) return script_[k_ % script_.size()];
        return eos_;
    }
    void delay(int units) const { if (delay_us_ > 0) std::this_thread::sleep_for(std::chrono::microseconds((long)delay_us_ * units)); }
};

}  // namespace qv

#include "vision/vision.h"
namespace qv {
// Test encoder (`--vision-file mock`): deterministic rows, hidden size 8, no weights.
class MockVisionEncoder : public vision::Encoder {
   public:
    MockVisionEncoder() { cfg_.text_hidden = 8; }
    const char* backend() const override { return "mock"; }
    void encode(const vision::Pixels& p, float* out) override {
        for (int i = 0; i < p.tokens() * 8; ++i) out[i] = 0.01f * (float)(i % 7);
        last_ms = 0.1;
    }
};
}  // namespace qv
