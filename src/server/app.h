// qv_server application: llama-server compatible routes on top of a single stateful Q-Volta engine.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

#include <map>

#include "core/json_doc.h"
#include "core/tool_format.h"
#include "grammar/grammar.h"
#include "qv/engine.h"
#include "sampling/sampler.h"
#include "server/chat_render.h"
#include "server/checkpoint_pool.h"
#include "server/compat_api.h"
#include "server/http.h"
#include "server/options.h"
#include "server/presets.h"
#include "tokenizer/tokenizer.h"
#include "vision/vision.h"

namespace qv::server {

using jd::Value;
using Clock = std::chrono::steady_clock;

struct ApiError {
    int status;
    std::string message, type;
    Value extra;  // object merged into the error object (n_prompt_tokens, n_ctx, ...)
};
[[noreturn]] void throw_error(int status, const std::string& msg, const std::string& type = "");
std::string error_json(const ApiError& e);

enum class Kind { Native, OaiCmpl, OaiChat, Infill };

struct TokProb {
    int id;
    std::string text, bytes_text;
    std::vector<int> bytes;
    float logprob, prob;
    std::vector<std::tuple<int, std::string, std::vector<int>, float>> top;  // id, text, bytes, logprob or prob
};

struct ImagePart {       // one image inside Task::prompt: tokens [start, start+n) are image placeholders
    int start = 0, n = 0;
    std::shared_ptr<vision::Pixels> px;
};

struct Task {
    Kind kind = Kind::Native;
    std::vector<ImagePart> images;  // chat requests with image_url parts; prompt holds negative pseudo ids at these positions (content-hash keyed)
    std::vector<int> prompt;
    std::string prompt_text;
    int n_predict = -1;
    std::vector<std::string> stops;
    bool cache_prompt = true, stream = false, return_tokens = false, timings_per_token = false, post_probs = false, ignore_eos = false;
    int id_slot = -1, n_probs = 0, n_keep = 0, n_choices = 1;
    SamplingParams sp;
    std::string grammar;
    bool grammar_lazy = false, json_object = false;
    std::vector<std::string> trigger_words;
    // chat tools
    Value tools;
    bool parse_tools = false;
    ToolFormat tool_fmt = ToolFormat::Generic;
    Value json_schema, response_fields, logit_bias_echo, stop_echo;
    // chat
    bool include_usage = false, thinking_open = false;
    std::string reasoning_format = "none";  // none | deepseek | deepseek-legacy (resolved)
    std::string think_start = "<think>", think_end = "</think>";
    int reasoning_budget = -1;  // --reasoning-budget N (N > 0): after N reasoning tokens the thinking block is closed by the server
    std::string reasoning_budget_msg;
    bool inkling_msgs = false;  // Inkling message stream (<|content_thinking|> ... <|end_message|>)
    std::string model_echo;
    bool logprobs_oai = false;
    // streaming extras (llama-server: return_progress, sse_ping_interval, X-Conversation-Id)
    bool return_progress = false;
    int sse_ping_interval = 30;  // seconds of silence before an SSE comment ping, <= 0 disables
    std::function<void(int total, int cache, int processed, double ms)> on_progress;
    std::shared_ptr<std::atomic<bool>> reasoning_end;  // set by POST /v1/chat/completions/control {action: reasoning_end}
};

struct SlotInfo {
    int id = 0;
    bool processing = false, has_task = false;
    long id_task = -1;
    int n_prompt = 0, n_prompt_processed = 0, n_prompt_cache = 0, n_decoded = 0, n_remain = 0;
    bool has_next_token = false, has_new_line = false;
    Value params;
};

struct GenResult {
    std::string content;
    std::vector<int> tokens;
    std::vector<TokProb> probs;
    std::string stop_type = "none", stopping_word;  // none | eos | word | limit
    bool truncated = false, aborted = false, has_new_line = false;
    int n_prompt = 0, n_cache = 0, n_prompt_eval = 0, n_decoded = 0, tokens_cached = 0, id_slot = 0;
    double prompt_ms = 0, predicted_ms = 0;
};

struct Chunk {
    std::string text;
    std::vector<int> tokens;
    std::vector<TokProb> probs;
    int n_decoded = 0;
    int n_cache = 0, n_prompt_eval = 0;     // timing snapshot (timings_per_token)
    double prompt_ms = 0, predicted_ms = 0;
};

// Resumable stream buffer (llama-server server-stream.cpp): a streaming request carrying X-Conversation-Id keeps generating after the
// client disconnects; the SSE bytes are kept (4 MiB ring, 300 s after completion) and can be replayed from an offset via GET /v1/stream.
class StreamSession {
   public:
    explicit StreamSession(std::string id);
    bool append(const char* data, size_t len);
    void finalize();
    void cancel() { cancelled_.store(true); }
    bool cancelled() const { return cancelled_.load(); }
    bool done() const;
    size_t total() const;
    size_t dropped() const;
    long long completed_at() const;
    // drains from `offset` (advanced) into `sink` until finalized or should_stop; false when the offset was dropped from the ring
    bool read_from(size_t& offset, const std::function<bool(const char*, size_t)>& sink, const std::function<bool()>& should_stop);
    std::string id;
    long long started_at;

   private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::string buf_;
    size_t dropped_ = 0;
    bool done_ = false;
    long long completed_ = 0;
    std::atomic<bool> cancelled_{false};
};
using StreamSessionPtr = std::shared_ptr<StreamSession>;

// FIFO (ticket) mutex: threads get the engine in arrival order, so concurrent sequences interleave their engine operations round-robin
// (one decode step or one prefill chunk each) and a long prefill cannot starve the others.
class FairMutex {
   public:
    void lock() {
        std::unique_lock<std::mutex> lk(m_);
        const unsigned long t = next_++;
        cv_.wait(lk, [&] { return serving_ == t; });
    }
    void unlock() {
        { std::lock_guard<std::mutex> lk(m_); ++serving_; }
        cv_.notify_all();
    }
    int waiting() { std::lock_guard<std::mutex> lk(m_); return (int)(next_ - serving_); }

   private:
    std::mutex m_;
    std::condition_variable cv_;
    unsigned long next_ = 0, serving_ = 0;
};

class App {
   public:
    Options o;
    Tokenizer tok;
    std::unique_ptr<Engine> eng;
    std::unique_ptr<ChatTemplates> tmpl;
    std::unique_ptr<vision::Encoder> venc;  // null = no image input
    vision::SourceOptions vsrc;
    int image_token_id = -1;
    std::string backend_name, tokenizer_path, model_id, model_template, model_arch;
    std::string template_source;       // "override" | "embedded" | "sidecar" | "none" (where model_template came from)
    std::string model_meta_json;       // "meta" object of the .qv index (compact JSON, "" when absent)
    std::vector<std::string> embedded_blobs;  // names of the blobs stored in the .qv
    std::string embedded_applied;      // human-readable list of embedded generation defaults that were applied
    std::vector<int> stop_ids;
    int n_ctx = 4096, chunk = 256, n_slots = 4, mtp_chunk = 8;
    std::atomic<bool> ready{false};
    long long started_at = 0;
    std::string build_info = "qv-dev";
    int bos_id = -1, eos_id = -1;
    int fim_pre = -1, fim_suf = -1, fim_mid = -1, fim_rep = -1, fim_sep = -1;

    void init_slots();
    void set_sequences(bool multi);  // call with mu_ held (or from init_slots)
    int shared_chunk = 1024;  // prefill chunk (tokens) used while other slots are running, so their decode steps are not stalled by one huge chunk
    std::mutex venc_mu_;
    int slot_ctx = 4096;  // context of each slot / sequence (== n_ctx reported to clients): --ctx-size / n_slots when the engine holds one sequence per slot
    bool multi_seq = false;  // engine holds n_slots sequences and runs requests concurrently; false: one sequence, requests run one at a time
    // ---- model presets / routing (--models-preset)
    Options base_opts;
    std::vector<ModelPreset> presets;
    std::string active_preset_id;
    bool has_presets() const { return !presets.empty(); }
    const ModelPreset* find_preset(const std::string& name) const;
    void load_model(const Options& mo, const std::string& forced_id = "");
    void unload_model();
    long acquire_model_lease(const std::string& requested_model, const std::function<bool()>& alive);
    void release_model_lease(long ticket);

    // --sleep-idle-seconds: the engine is destroyed after N idle seconds (VRAM / RAM freed) and rebuilt by `reload_engine` on the next task
    std::function<std::unique_ptr<Engine>()> reload_engine;
    std::atomic<bool> sleeping{false};
    void start_sleep_watcher();
    void stop_sleep_watcher();
    int cached_n_embd = 0;
    long long cached_weight_bytes = 0;
    void route(const http::Request& req, http::Conn& c);

   private:
    std::thread sleep_thr_;
    std::atomic<bool> sleep_stop_{false};
    std::atomic<long long> last_task_ms_{0};
    void ensure_engine();  // under the engine lease: wake from sleep
    // ---- engine queue + slots
    // Admission is a single FIFO of waiters (slot requests, model users and exclusive admin leases) served by dispatch_locked():
    //  - a slot waiter gets a free slot (its id_slot, else the idle slot with the longest cached prompt prefix, else an empty/least recently used one);
    //    at most `conc_limit_` slots run at once (n_slots when the engine holds one sequence per slot, else 1);
    //  - an exclusive waiter (model load/unload, sleep, slot actions on single-sequence engines) waits until no slot runs and no model user is
    //    registered, and blocks everything queued behind it.
    struct Waiter { long ticket; int kind; int want; const std::vector<int>* prompt; bool user; bool granted; int sid; };  // kind: 0 slot, 1 exclusive, 2 shared (model user)
    std::mutex mu_;
    mutable std::mutex state_mu_;  // guards active model metadata/tokenizer/template during hot-swap vs /models and /props reads
    std::condition_variable cv_;
    long next_ticket_ = 0;
    std::deque<Waiter> waiters_;
    std::vector<char> slot_busy_;
    std::atomic<int> n_busy_{0};
    std::atomic<int> n_decoding_{0};  // running slots that are past their prompt (decoding)
    int n_users_ = 0, conc_limit_ = 1;
    bool excl_held_ = false;
    std::vector<long long> slot_used_;  // last use (monotonic counter) per slot, for LRU
    long long use_clock_ = 0;
    void dispatch_locked();
    int pick_slot_locked(const Waiter& w) const;
    long wait_granted(int kind, int want, const std::vector<int>* prompt, bool user, const std::function<bool()>& alive, int* sid_out);
    long user_acquire(const std::function<bool()>& alive);
    void user_release();
    void lease_downgrade();  // exclusive -> model user
    int slot_acquire(const Task& t, bool user, const std::function<bool()>& alive);  // slot id, or -1 if the client left while waiting
    void slot_release(int sid);
    bool engine_idle_locked() const { return waiters_.empty() && n_busy_ == 0 && n_users_ == 0 && !excl_held_; }
    std::mutex reload_mu_;
    FairMutex eng_mu_;  // engine operations (prefill chunk / decode step) of the concurrently running sequences
    std::vector<SlotInfo> slots_;
    std::vector<std::vector<int>> slot_cached_;  // per sequence: tokens inside the engine (logits valid for exactly this context)
    std::vector<std::vector<PromptCheckpoint>> slot_ckpts_;
    std::vector<char> slot_cacheable_;  // per sequence: whether current VRAM context is eligible for L2 parking
    ParkedCheckpointPool parked_pool_;  // L2 host-RAM parked context pool (LRU, RAM-capped)
    int seq_of(int sid) const { return multi_seq ? sid : 0; }
    std::vector<int>& cached_of(int sid) { return slot_cached_[seq_of(sid)]; }
    std::vector<PromptCheckpoint>& ckpts_of(int sid) { return slot_ckpts_[seq_of(sid)]; }
    int owner_slot_ = 0;
    long task_counter_ = 0;
    // metrics
    struct Metrics {
        long long t_start = 0;
        double prompt_tokens_total = 0, prompt_cached_total = 0, prompt_seconds_total = 0, predicted_total = 0, predicted_seconds_total = 0;
        long long n_decode_total = 0, n_tokens_max = 0;
        double b_prompt_tokens = 0, b_prompt_s = 0, b_pred_tokens = 0, b_pred_s = 0, b_busy_sum = 0, b_decode = 0;
    } m_;
    std::string slot_path(const std::string& fname) const;

    // ---- helpers
    std::string piece_text(int id) const;
    std::string piece_bytes(int id, std::vector<int>& bytes) const;
    Value gen_settings(const Task& t, int flavor) const;
    Value timings_json(const GenResult& r) const;
    Value timings_json(int cache, int prompt_n, double prompt_ms, int pred_n, double pred_ms) const;
    // reasoning control (POST /v1/chat/completions/control)
    std::map<std::string, std::shared_ptr<std::atomic<bool>>> controls_;
    void handle_control(const http::Request&, http::Conn&);
    // resumable streams
    std::mutex sm_;
    std::map<std::string, StreamSessionPtr> sessions_;
    StreamSessionPtr stream_create(const std::string& id);
    StreamSessionPtr stream_get(const std::string& id);
    void handle_stream_get(const http::Request&, http::Conn&);
    void handle_stream_lookup(const http::Request&, http::Conn&);
    void handle_stream_delete(const http::Request&, http::Conn&);
    Value probs_json(const std::vector<TokProb>& v, bool post) const;
    Task parse_task(const Value& body, Kind kind);
    void parse_sampling(const Value& body, Task& t);
    std::vector<int> tokenize_prompt_value(const Value& p);
    std::string template_prefix_json(const Value&);
    void finalize_constraints(Task& t);  // json_schema / response_format -> GBNF (validated)
    std::shared_ptr<grammar::Matcher> get_matcher(const std::string& gbnf);
    std::mutex gm_;
    std::shared_ptr<const grammar::VocabIndex> vocab_;
    std::map<std::string, std::shared_ptr<grammar::Matcher>> matchers_;

    // ---- generation
    long lease_acquire(const std::function<bool()>& alive);  // exclusive FIFO lease (waits for all slots to go idle); -1 if the client left while waiting
    void lease_release(long ticket);
    struct Admission;
    void prefill_with_images(const Task& t, int off, int n, const std::vector<std::vector<float>>& emb);
    GenResult generate(Task& t, const std::function<bool()>& alive, const std::function<bool(const Chunk&)>& on_chunk, const std::function<void(int)>& on_start, long existing_ticket = -1);
    // ---- handlers
    void handle_completion(const http::Request&, http::Conn&, Kind, ApiFmt fmt = ApiFmt::Oai, const Value* body_in = nullptr);
    void handle_responses(const http::Request&, http::Conn&);
    void handle_count_tokens(const http::Request&, http::Conn&, ApiFmt fmt);
    void handle_infill(const http::Request&, http::Conn&);
    void handle_tokenize(const http::Request&, http::Conn&);
    void handle_detokenize(const http::Request&, http::Conn&);
    void handle_apply_template(const http::Request&, http::Conn&);
    void handle_props(http::Conn&);
    void handle_slots(const http::Request&, http::Conn&);
    void handle_slot_action(const http::Request&, http::Conn&, int id);
    void handle_metrics(http::Conn&);
    void handle_models(http::Conn&);
    void handle_model_load(const http::Request&, http::Conn&, bool load);
    void handle_messages(const http::Request&, http::Conn&);
    void respond_json(http::Conn& c, int status, const Value& v, const http::Headers& extra = {});
    bool serve_webui(const http::Request& req, http::Conn& c, const std::string& path);
    static bool is_frontend_path(const std::string& path);
    Value model_json() const;
};

}  // namespace qv::server
