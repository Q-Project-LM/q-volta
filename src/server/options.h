// Command line / environment parsing for qv_server. Accepts every llama-server flag spelling (and LLAMA_ARG_* environment variable)
// of llama.cpp 9d6f04f; flags that have no meaning for Q-Volta are parsed (so their arguments are consumed correctly), warned about
// and ignored. The spelling table is generated from `llama-server --help` (tools/gen_llama_flags.py -> llama_flags.inc).
#pragma once
#include <set>
#include <string>
#include <vector>

namespace qv::server {

struct Options {
    // ---- model / engine
    std::string model;                 // Q-Volta model prefix (PREFIX.qv.json/.qv.bin) or a .qv path
    std::string models_preset;         // --models-preset PATH: INI preset file for multi-model routing
    int models_max = -1;               // --models-max N (accepted; qv_server keeps 1 active engine in memory)
    bool preload = false;              // preset option: preload / load-on-startup = true loads this model at startup
    std::string backend = "auto";      // cuda | cpu | auto | mock
    std::string tokenizer;             // "" = use the tokenizer embedded in the .qv, then sidecar search (docs/QV_FORMAT.md)
    bool embedded_defaults = true;     // apply generation_config temperature/top_k/top_p of the .qv as default sampling (--no-embedded-defaults disables)
    std::set<std::string> given;       // canonical names of flags given on the command line / environment
    int n_gpu_layers = -1;             // -1 = unset ("auto")
    bool ngl_given = false;
    int threads = 0;                   // 0 = min(32, cores)
    int ctx = 0;                       // 0 = default 4096
    int batch = 0, ubatch = 0, prefill_chunk = 0;
    int kv_mode = 0, kv4_kmode = 0;
    bool mtp = false;
    int mtp_depth = 1;
    bool mtp_grammar = false;          // preset/CLI: allow MTP decode under active GBNF grammar (tool_calls, json_schema)
    // ---- vision (Inkling image input; docs/INKLING_VISION.md)
    std::string vision_file;           // sidecar .qv; "" = auto (<model stem>-vision.qv, then <dir>/<family>-vision.qv), "none" = off, "mock" = test encoder
    bool allow_remote_images = false;  // http(s) image_url fetching (SSRF-guarded); data: URLs and files below --image-root always work
    std::string image_root = ".";      // local image files / file:// URLs are only served from below this directory ("" disables)
    int image_max_side = 4096;
    std::string image_token = "<|unused_200054|>";
    std::string mtp_file;  // Inkling: MTP sidecar container (default <model>-mtp.qv)
    // ---- http
    std::string host = "127.0.0.1";
    int port = 9931;                   // llama-server default of the reference commit
    int http_threads = -1;
    int timeout = 3600;
    int sse_ping_interval = 30;
    int sleep_idle_seconds = -1;
    std::string path, api_prefix;
    bool webui = true;
    std::string ui_config;             // --ui-config JSON (returned as props.ui_settings)
    bool ui_mcp_proxy = false;
    std::string tools;
    std::vector<std::string> api_keys;
    std::vector<std::string> alias;    // first entry is the model id reported by the API
    std::string cors_origins = "*", cors_methods = "GET, POST, DELETE, OPTIONS", cors_headers = "*";
    bool cors_credentials = true;
    // ---- server behaviour
    int n_parallel = -1;               // -1 = auto -> 4 slots (llama default with unified KV)
    bool cache_prompt = true;
    int checkpoint_slots = 4;          // --checkpoint-slots N: L2 host-RAM parked context slots (0 disables L2 pool)
    int checkpoint_ram_mb = 8192;      // --checkpoint-ram-mb MB: max total RAM for L2 checkpoint pool (0 disables L2 pool)
    int checkpoint_min_tokens = 4096;  // --checkpoint-min-tokens N: min context tokens to activate L1/L2 checkpoints
    bool metrics = false, slots = true, props = false, embedding = false;
    std::string slot_save_path;
    std::string chat_template, chat_template_file, chat_template_kwargs;
    bool jinja = true;
    std::string reasoning_format = "auto";  // auto | none | deepseek | deepseek-legacy
    int reasoning = -1;                     // -1 auto, 0 off, 1 on
    int reasoning_budget = -1;
    std::string reasoning_budget_message;  // injected before the end-of-thinking marker when the budget is exhausted
    std::string reasoning_effort;
    int max_queue = -1;                // Q-Volta: -1 = unbounded (llama behaviour), N>=0 = 503 beyond N waiting requests
    bool log_disable = false;
    int verbosity = 3;
    bool special = false;
    // ---- default generation settings (llama common_params_sampling defaults)
    int n_predict = -1;
    float temp = 0.8f;
    int top_k = 40;
    float top_p = 0.95f, min_p = 0.05f, typical_p = 1.0f, top_n_sigma = -1.0f;
    float xtc_probability = 0.0f, xtc_threshold = 0.1f;
    float repeat_penalty = 1.0f, presence_penalty = 0.0f, frequency_penalty = 0.0f;
    int repeat_last_n = 64;
    float dry_multiplier = 0.0f, dry_base = 1.75f;
    int dry_allowed_length = 2, dry_penalty_last_n = 64;
    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};
    bool dry_breakers_user = false;
    std::string grammar, json_schema;  // server-wide defaults (--grammar, --json-schema)
    int mirostat = 0;
    float mirostat_tau = 5.0f, mirostat_eta = 0.1f;
    float dynatemp_range = 0.0f, dynatemp_exponent = 1.0f;
    long long seed = -1;
    int min_keep = 0;
    bool ignore_eos = false;
    std::vector<std::string> samplers = {"penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"};
    std::vector<std::pair<int, float>> logit_bias;
    // ---- mock backend (tests)
    std::string mock_text = "Hello! This is the Q-Volta mock engine speaking. It replays a fixed script.";
    bool mock_loop = false;
    int mock_delay_us = 0;
    // bookkeeping
    std::vector<std::string> warnings;
    std::string cmdline;
};

// Returns 0 on success, otherwise the process exit code (message already printed). `exit_now` is set for --help/--version.
int parse_args(int argc, char** argv, Options& o, bool& exit_now);
bool flag_known(const std::string& spelling);
// Apply a single key=value option from a preset INI to `o`. Returns false if the option has no effect / is unknown (`msg` set).
bool apply_preset_option(Options& o, const std::string& key, const std::string& val, std::string& msg);

}  // namespace qv::server
