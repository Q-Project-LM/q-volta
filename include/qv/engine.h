// Public inference API (backend-independent).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "qv/model.h"
#include "qv/spec_sampling.h"

namespace qv {

struct EngineOptions {
    int max_ctx = 4096;        // context of EACH sequence (the server divides --ctx-size by the number of slots)
    int n_seq = 1;             // independent sequences sharing the weights (Inkling CUDA / Qwen CUDA / CPU engines); see Engine::select_seq
    int prefill_chunk = 2048;  // tokens per batched prefill chunk
    int threads = 32;          // CPU backend: worker threads incl. the caller (RAM bandwidth saturates near 32 here)
    bool mtp = false;          // load the MTP head and use it for exact speculative greedy decoding (CPU and CUDA backends)
    std::string mtp_file;      // Inkling CUDA backend: sidecar container with the MTP (DSpark) drafter (tools/convert_inkling_mtp.py); mtp_depth 1..7 drafts
    int mtp_depth = 1;         // CUDA backend: drafts per verify cycle (1..3); verify processes depth+1 tokens
    int kv_mode = 0;           // CUDA backend KV cache: 0 = fp32 (default), 1 = int4 (64-token blocks + fp16 window, attention/kv4.h), 2 = fp16
    int kv4_kmode = 0;         // int4 K quantization: 0 = per-channel (KIVI), 1 = per-token groups of 64 channels
    // CUDA backend, eval only: if set, prefill() computes the logits of EVERY row and calls this with (position, vocab logits)
    std::function<void(int, const float*)> row_logits;
    bool dump_hidden = false;  // record residual stream after embed + each layer (debug / golden tests)
};

// Counters of the MTP speculative decode path (all zero when MTP is off).
struct MtpStats {
    long cycles = 0;      // verify passes
    long accepted = 0;    // drafts accepted
    long drafted = 0;     // drafts verified (CUDA backend; CPU: == cycles)
    long plain_steps = 0; // single-token steps (tail of the generation)
    double draft_ms = 0, verify_ms = 0, plain_ms = 0, prefill_mtp_ms = 0;
};

class Engine {
   public:
    virtual ~Engine() = default;
    // Process one token at position `pos` (== number of tokens already in the context).
    virtual void step(int token, int pos) = 0;
    // Batched prefill of T tokens at positions [pos0, pos0+T); leaves logits of the last token ready.
    virtual void prefill(const int* tokens, int T, int pos0) = 0;
    // Vision (Inkling engines): batched prefill where positions with mask[t] != 0 take their input embedding from `emb` instead of the token
    // embedding (+ embed_norm): `emb` holds one hidden-size float row per masked position, packed in order. tokens[t] at masked positions is
    // only a placeholder id. Leaves logits of the last token ready. Other engines throw.
    virtual void prefill_embeds(const int* tokens, const float* emb, const unsigned char* mask, int T, int pos0) {
        (void)tokens; (void)emb; (void)mask; (void)T; (void)pos0;
        throw std::runtime_error("this engine does not support image embeddings");
    }
    // Greedy decode n tokens; first = token at start_pos (argmax of the logits after the context).
    virtual void decode_greedy(int first, int start_pos, int n, std::vector<int>& out, bool use_graph) = 0;
    // Lossless speculative sampling of n tokens (temperature > 0).
    virtual bool has_speculative_decode() const { return false; }
    virtual void decode_speculative(int first, int start_pos, int n, const SpecSamplingParams& sp, std::vector<int>& out) {
        (void)first; (void)start_pos; (void)n; (void)sp; (void)out;
        throw std::runtime_error("this engine does not support speculative sampling");
    }
    // Constrained MTP variants: verify-time TokenConstraint; may return fewer than `n` tokens if `tc->is_eog(tok)` or `tc->is_dead()`.
    virtual bool has_constrained_mtp() const { return false; }
    virtual void decode_greedy_constrained(int first, int start_pos, int n, std::vector<int>& out, TokenConstraint* tc) {
        (void)tc;
        decode_greedy(first, start_pos, n, out, false);
    }
    virtual void decode_speculative_constrained(int first, int start_pos, int n, const SpecSamplingParams& sp, std::vector<int>& out, TokenConstraint* tc) {
        (void)tc;
        decode_speculative(first, start_pos, n, sp, out);
    }
    virtual int argmax() = 0;                         // greedy next token from the last step's logits
    virtual void logits(std::vector<float>& out) = 0; // copy last logits to host
    virtual void reset() = 0;                         // clear recurrent/KV state
    virtual const std::vector<float>& hidden_dump() const = 0;
    virtual const std::vector<float>& logits_dump() const = 0;  // per-token logits of prefill() when dump_hidden
    virtual void sync() = 0;
    // research hooks (CPU backend, branch p9/tier): snapshot/restore the recurrent state (GDN + conv; KV is rewritten by replay),
    // and switch single-token attention to sparse block retrieval ("off" or "sink,recent,block,k[,agg]").
    virtual bool has_state_save() const { return false; }
    virtual void state_save() {}
    virtual void state_restore() {}
    // Host-RAM prefix checkpoint of the CURRENT sequence (mutable recurrent/conv state, ring KV windows, last logits).
    // Safe to restore as long as non-ring KV positions [0, checkpoint_pos) have not been overwritten since save_checkpoint().
    virtual bool has_checkpoint() const { return false; }
    virtual bool save_checkpoint(std::vector<uint8_t>& out) { (void)out; return false; }
    virtual bool restore_checkpoint(const std::vector<uint8_t>& in) { (void)in; return false; }
    virtual size_t checkpoint_bytes() const { return 0; }
    // Host-RAM snapshot of the CURRENT sequence's non-ring KV cache prefix [0, n_tokens),
    // paired with save_checkpoint()/restore_checkpoint() for independent parked context slots.
    virtual bool has_kv_checkpoint() const { return false; }
    virtual size_t kv_checkpoint_bytes(int n_tokens) const { (void)n_tokens; return 0; }
    virtual bool save_kv_prefix(std::vector<uint8_t>& out, int n_tokens) { (void)out; (void)n_tokens; return false; }
    virtual bool restore_kv_prefix(const std::vector<uint8_t>& in) { (void)in; return false; }
    virtual void set_sparse(const std::string&) {}
    virtual void set_mtp_enabled(bool) {}  // runtime switch (needs EngineOptions::mtp at construction)
    virtual void mtp_stats(MtpStats& s) const { s = MtpStats(); }
    // Inkling engines: integer debug output of the last step (router expert ids [48 applications][8]) when EngineOptions::dump_hidden.
    // For arch inkling2_loop, hidden_dump() holds the LAST step only: [50][2304] = embedding, 48 layer-application outputs, final-norm output.
    virtual const std::vector<int>& aux_dump() const { static const std::vector<int> none; return none; }
    // Multi-sequence contract: an engine built with EngineOptions::n_seq = N owns N independent sequences (KV cache, recurrent/conv states, last
    // logits, token/position state) over one copy of the weights. All other calls (step / prefill / argmax / logits / reset / state_*) act on
    // the CURRENT sequence, chosen by select_seq(i) (0 <= i < n_seq(); initially 0). Switching is host-side bookkeeping (no device sync, no
    // copy) and the results of a sequence are bit-identical to running it alone. Calls must come from one thread at a time (the server
    // serialises them with its engine turn lock). Engines that cannot hold more than one sequence return 1 and ignore select_seq.
    virtual int n_seq() const { return 1; }
    virtual void select_seq(int) {}
    virtual size_t device_bytes_weights() const = 0;
    virtual size_t device_bytes_state() const = 0;
    const Config& config() const { return cfg_; }

   protected:
    Config cfg_;
};

std::unique_ptr<Engine> make_cuda_engine(const ModelFile& mf, const EngineOptions& opt);  // needs the qv_cuda library
std::unique_ptr<Engine> make_cuda_inkling_engine(const ModelFile& mf, const EngineOptions& opt);  // arch inkling2_loop (dispatched by make_cuda_engine)
std::unique_ptr<Engine> make_cpu_engine(const ModelFile& mf, const EngineOptions& opt);   // needs qv_cpu_engine

}  // namespace qv
