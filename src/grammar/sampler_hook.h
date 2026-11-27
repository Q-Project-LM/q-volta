// Adapters: GrammarState -> sampling hook, and vocabulary construction from qv::Tokenizer.
#pragma once
#include <algorithm>
#include <memory>
#include <vector>

#include "grammar/grammar.h"
#include "qv/spec_sampling.h"
#include "sampling/sampler.h"
#include "tokenizer/tokenizer.h"

namespace qv::grammar {

class GrammarLogitMask final : public LogitMask {
   public:
    explicit GrammarLogitMask(GrammarState st) : st_(std::move(st)) {}
    void apply(float* logits, size_t n) override { st_.mask_logits(logits, n); }
    void accept(int tok) override { st_.accept_token(tok); }
    GrammarState& state() { return st_; }

   private:
    GrammarState st_;
};

class GrammarTokenConstraint final : public qv::TokenConstraint {
   public:
    GrammarTokenConstraint(GrammarState st, const std::vector<int>& stop_ids)
        : st_(std::move(st)), stops_(stop_ids) {}
    bool awaiting() const override { return st_.awaiting_trigger(); }
    bool is_dead() const override { return st_.is_dead(); }
    bool is_eog(int tok) const override {
        return std::find(stops_.begin(), stops_.end(), tok) != stops_.end();
    }
    bool probe_allowed(int tok) const override {
        if (st_.awaiting_trigger()) return true;
        if (st_.is_dead() || tok < 0) return false;
        if (!cached_mask_) cached_mask_ = st_.allowed();
        return cached_mask_->allowed((size_t)tok);
    }
    bool try_accept(int tok) override {
        if (st_.awaiting_trigger()) {
            (void)st_.accept_token(tok);
            cached_mask_.reset();
            return true;
        }
        if (st_.is_dead()) return false;
        GrammarState next = st_.fork();
        if (!next.accept_token(tok)) return false;
        st_ = std::move(next);
        cached_mask_.reset();
        return true;
    }
    void mask_logits(float* logits, size_t n) override { st_.mask_logits(logits, n); }

   private:
    mutable GrammarState st_;
    std::vector<int> stops_;
    mutable std::shared_ptr<const TokenMask> cached_mask_;
};

// Pieces are the raw token bytes (special tokens: literal text). `eog_ids` are end-of-generation tokens (eos, <|im_end|>, ...).
inline std::shared_ptr<const VocabIndex> vocab_from_tokenizer(const Tokenizer& tok, const std::vector<int>& eog_ids) {
    std::vector<std::string> pieces((size_t)tok.vocab_size());
    std::vector<uint8_t> eog(pieces.size(), 0);
    for (int i = 0; i < tok.vocab_size(); ++i) pieces[(size_t)i] = tok.token_bytes(i);
    for (int e : eog_ids) if (e >= 0 && (size_t)e < eog.size()) eog[(size_t)e] = 1;
    return VocabIndex::build(pieces, eog);
}

}  // namespace qv::grammar
