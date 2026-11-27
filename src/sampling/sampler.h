// Token sampling on host logits, following llama.cpp's sampler chain (common/sampling.cpp, llama-sampler.cpp):
//   logit_bias -> penalties (repeat/frequency/presence) -> dry -> top_n_sigma -> top_k -> typical_p -> top_p -> min_p -> xtc -> temperature
// (the order is configurable through `samplers`, like llama-server's "samplers" request field), mirostat v1/v2 as an alternative
// to the chain, and a final draw with std::mt19937 + uniform_real_distribution<double> exactly like llama_sampler_dist.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace qv {

// Hook for grammar / JSON-schema constrained sampling (implemented by src/grammar when it is available).
struct LogitsConstraint {
    virtual ~LogitsConstraint() = default;
    virtual void apply(std::vector<float>& logits) = 0;  // set logits of disallowed tokens to -inf
    virtual void accept(int token) = 0;                  // advance the constraint with the sampled token
};


// Hook used by grammar / JSON-schema decoding (src/grammar/sampler_hook.h): operates on raw float logits.
struct LogitMask {
    virtual ~LogitMask() = default;
    virtual void apply(float* logits, size_t n) = 0;  // set disallowed logits to -inf
    virtual void accept(int tok) = 0;                 // the chosen token was committed
};
namespace detail {
struct MaskConstraint : LogitsConstraint {
    std::shared_ptr<LogitMask> m;
    explicit MaskConstraint(std::shared_ptr<LogitMask> mm) : m(std::move(mm)) {}
    void apply(std::vector<float>& lg) override { m->apply(lg.data(), lg.size()); }
    void accept(int t) override { m->accept(t); }
};
}  // namespace detail

struct SamplingParams {
    // llama-server defaults (common_params_sampling)
    float temperature = 0.8f;       // <= 0 means greedy
    int top_k = 40;                 // <= 0 = disabled
    float top_p = 0.95f;            // 1.0 = disabled
    float min_p = 0.05f;            // 0.0 = disabled
    float typical_p = 1.0f;         // 1.0 = disabled
    float top_n_sigma = -1.0f;      // -1 = disabled
    float xtc_probability = 0.0f;
    float xtc_threshold = 0.1f;
    int min_keep = 0;
    float dynatemp_range = 0.0f;
    float dynatemp_exponent = 1.0f;
    int repeat_last_n = 64;         // -1 = whole context
    float repeat_penalty = 1.0f;
    float presence_penalty = 0.0f;  // llama: logit -= presence * (count > 0) over the last repeat_last_n tokens
    float frequency_penalty = 0.0f; // llama: logit -= frequency * count
    float dry_multiplier = 0.0f;
    float dry_base = 1.75f;
    int dry_allowed_length = 2;
    int dry_penalty_last_n = 64;    // 0 = disabled, -1 = whole context
    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};
    std::vector<std::vector<int>> dry_breaker_tokens;  // tokenised sequence breakers (first token is what matters)
    int mirostat = 0;               // 0 off, 1 v1, 2 v2
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;
    bool has_seed = false;
    uint64_t seed = 0;
    std::vector<std::pair<int, float>> logit_bias;
    std::vector<std::string> samplers = {"penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"};

    bool greedy() const { return temperature <= 0.0f && mirostat == 0; }
    bool has_penalties() const { return presence_penalty != 0 || frequency_penalty != 0 || repeat_penalty != 1.0f || dry_multiplier > 0; }
    // True when the engine's own argmax() gives the answer (no host logits needed).
    bool engine_argmax_ok() const { return greedy() && !has_penalties() && logit_bias.empty(); }
    // True when Engine::decode_speculative matches Sampler's filter & temperature chain.
    bool speculative_ok() const {
        if (mirostat != 0 || has_penalties() || !logit_bias.empty()) return false;
        if (typical_p < 1.0f || top_n_sigma > 0.0f || xtc_probability > 0.0f || dynatemp_range > 0.0f) return false;
        int prev_stage = -1;
        for (const auto& s : samplers) {
            int stage = -1;
            if (s == "top_k" && top_k > 0) stage = 0;
            else if (s == "top_p" && top_p < 1.0f) stage = 1;
            else if (s == "min_p" && min_p > 0.0f) stage = 2;
            else if (s == "temperature") stage = 3;
            if (stage >= 0) {
                if (stage < prev_stage) return false;
                prev_stage = stage;
            }
        }
        return true;
    }
};

struct TokenLogprob {
    int token = -1;
    float logprob = 0;                                  // log-softmax of the raw logits
    std::vector<std::pair<int, float>> top;             // best alternatives, descending
};

class Sampler {
   public:
    Sampler(const SamplingParams& p, const std::vector<int>& prompt_tokens);
    bool want_post_ = false;  // keep the post-chain candidate probabilities (llama post_sampling_probs)
    void set_constraint(std::shared_ptr<LogitsConstraint> c) { constraint_ = std::move(c); }
    void set_mask(std::shared_ptr<LogitMask> m) { constraint_ = std::make_shared<detail::MaskConstraint>(std::move(m)); }
    bool has_mask() const { return constraint_ != nullptr; }
    // logits are modified in place. Does not record the token; call accept().
    int sample(std::vector<float>& logits);
    void accept(int tok);
    // log-probability info from the *raw* logits (call before sample()).
    static TokenLogprob logprob(const std::vector<float>& raw, int chosen, int top_n);
    // Probability of `tok` after the whole chain (llama post_sampling_probs); valid after sample().
    float last_prob(int tok) const;
    const std::vector<std::pair<int, float>>& last_candidates() const { return last_cands_; }  // (id, p) after the chain, sorted desc
    uint64_t next_seed() { return ((uint64_t)rng_() << 32) ^ (uint64_t)rng_(); }

   private:
    struct Cand { int id; float logit; float p; };
    SamplingParams p_;
    std::vector<int> hist_;  // prompt + generated tokens
    std::mt19937 rng_;
    float mu_;
    std::shared_ptr<LogitsConstraint> constraint_;
    std::vector<std::pair<int, float>> last_cands_;
    std::vector<Cand> cur_;
    bool sorted_ = false;
    void apply_penalties(std::vector<float>& lg) const;
    void apply_dry(std::vector<float>& lg) const;
    void softmax();
    void sort_desc();
    void keep_best_one();
    void s_top_k(int k);
    void s_top_p(float p);
    void s_min_p(float p);
    void s_typical(float p);
    void s_top_n_sigma(float n);
    void s_xtc();
    void s_temp();
    int draw();
    int mirostat_v2();
    int mirostat_v1();
};

}  // namespace qv
