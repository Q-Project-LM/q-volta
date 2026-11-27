#include "server/options.h"

#include "core/json_doc.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

namespace qv::server {

namespace {

struct FlagEntry {
    std::vector<std::string> names;
    int arg;  // 0 none, 1 required value, 2 optional value (on|off|auto)
    std::string env;
};

const std::vector<FlagEntry>& table() {
    static const std::vector<FlagEntry> t = {
#include "server/llama_flags.inc"
    };
    return t;
}

// Q-Volta's own spellings: name -> {canonical, arg kind}
struct OwnFlag { const char* name; const char* canon; int arg; };
const OwnFlag kOwn[] = {
    {"--backend", "backend", 1}, {"--kv", "kv", 1}, {"--kv4-kmode", "kv4-kmode", 1}, {"--mtp", "mtp", 0}, {"--no-mtp", "no-mtp", 0},
    {"--mtp-grammar", "mtp-grammar", 0}, {"--no-mtp-grammar", "no-mtp-grammar", 0},
    {"--mtp-depth", "mtp-depth", 1}, {"--mtp-file", "mtp-file", 1}, {"--prefill-chunk", "prefill-chunk", 1}, {"--tokenizer", "tokenizer", 1}, {"--no-embedded-defaults", "no-embedded-defaults", 0},
    {"--model-name", "alias", 1}, {"--max-ctx", "ctx-size", 1}, {"--http-threads", "threads-http", 1}, {"--max-queue", "max-queue", 1},
    {"--default-max-tokens", "predict", 1}, {"--quiet", "log-disable", 0}, {"--mock-text", "mock-text", 1},
    {"--mock-loop", "mock-loop", 0}, {"--vision-file", "vision-file", 1}, {"--allow-remote-images", "allow-remote-images", 0}, {"--image-root", "image-root", 1},
    {"--image-max-side", "image-max-side", 1}, {"--image-token", "image-token", 1}, {"--mlock", "mlock", 0}, {"--no-mlock", "no-mlock", 0}, {"--mmap", "mmap", 0}, {"--no-mmap", "no-mmap", 0}, {"--mock-delay-us", "mock-delay-us", 1},
    {"--checkpoint-slots", "checkpoint-slots", 1}, {"--checkpoint-ram-mb", "checkpoint-ram-mb", 1}, {"--checkpoint-min-tokens", "checkpoint-min-tokens", 1}, {"--checkpoint-align", "checkpoint-align", 1},
};

bool truthy(const std::string& v) { return v == "1" || v == "on" || v == "enabled" || v == "true" || v == "yes"; }
bool falsey(const std::string& v) { return v == "0" || v == "off" || v == "disabled" || v == "false" || v == "no"; }

struct Resolved { const FlagEntry* e = nullptr; const OwnFlag* own = nullptr; std::string canon; bool neg = false; int arg = 0; };

bool is_neg_spelling(const std::string& s) { return s.rfind("--no-", 0) == 0 || s.rfind("-no-", 0) == 0 || s == "-nocb" || s == "-nkvo"; }

bool resolve(const std::string& a, Resolved& r) {
    for (auto& o : kOwn)
        if (a == o.name) { r.own = &o; r.canon = o.canon; r.arg = o.arg; return true; }
    for (auto& e : table()) {
        for (auto& n : e.names) {
            if (n != a) continue;
            r.e = &e;
            r.arg = e.arg;
            r.neg = is_neg_spelling(a);
            for (auto& m : e.names) if (m.rfind("--", 0) == 0 && !is_neg_spelling(m)) { r.canon = m.substr(2); break; }
            if (r.canon.empty()) r.canon = e.names[0].substr(e.names[0].rfind("--", 0) == 0 ? 2 : 1);
            return true;
        }
    }
    return false;
}

std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> v;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, d)) v.push_back(cur);
    return v;
}

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error(m); }

int to_int(const std::string& v, const std::string& flag) {
    try { size_t p; int x = std::stoi(v, &p); if (p != v.size()) throw 1; return x; } catch (...) { fail("invalid value '" + v + "' for " + flag); }
}
float to_f(const std::string& v, const std::string& flag) {
    try { size_t p; float x = std::stof(v, &p); if (p != v.size()) throw 1; return x; } catch (...) { fail("invalid value '" + v + "' for " + flag); }
}
std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) fail("failed to open file '" + p + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Flags handled by apply(); everything else in the table is "accepted, ignored".
const std::set<std::string>& handled() {
    static const std::set<std::string> s = {
        "model", "models-preset", "models-max", "ctx-size", "threads", "threads-http", "n-gpu-layers", "gpu-layers", "batch-size", "ubatch-size", "host", "port", "api-key", "api-key-file",
        "alias", "parallel", "cont-batching", "cache-prompt", "jinja", "chat-template", "chat-template-file", "chat-template-kwargs", "reasoning-format",
        "reasoning", "reasoning-budget", "reasoning-budget-message", "reasoning-effort", "metrics", "slots", "props", "embedding", "predict", "temp", "top-k", "top-p", "min-p",
        "typical", "top-nsigma", "xtc-probability", "xtc-threshold", "repeat-penalty", "repeat-last-n", "presence-penalty", "frequency-penalty",
        "dry-multiplier", "dry-base", "dry-allowed-length", "dry-penalty-last-n", "dry-sequence-breaker", "mirostat", "mirostat-lr", "mirostat-ent",
        "dynatemp-range", "dynatemp-exp", "seed", "samplers", "ignore-eos", "logit-bias", "timeout", "path", "ui", "log-disable", "verbose", "log-verbosity",
        "slot-save-path", "special", "cors-origins", "cors-methods", "cors-headers", "cors-credentials", "api-prefix", "min-keep", "grammar", "grammar-file", "json-schema", "json-schema-file", "help", "version",
        "backend", "kv", "kv4-kmode", "mtp", "no-mtp", "mtp-grammar", "no-mtp-grammar", "mtp-depth", "mtp-file", "prefill-chunk", "tokenizer", "no-embedded-defaults", "max-queue", "mock-text", "mock-loop", "mock-delay-us",
        "checkpoint-slots", "checkpoint-ram-mb", "checkpoint-min-tokens", "checkpoint-align",
        "reasoning-preserve", "context-shift", "warmup", "flash-attn", "mlock", "mmap", "no-mmap", "perf", "escape", "verbosity", "log-colors", "log-timestamps",
        "log-prefix", "kv-unified", "cache-idle-slots", "lora", "lora-scaled", "ssl-key-file", "ssl-cert-file", "hf-repo", "model-url", "docker-repo",
        "hf-file", "hf-token", "offline", "no-mlock", "vision-file", "allow-remote-images", "image-root", "image-max-side", "image-token",
        "ui-config", "ui-config-file", "sse-ping-interval", "sleep-idle-seconds", "ui-mcp-proxy", "tools",
    };
    return s;
}

void apply(Options& o, const Resolved& r, const std::string& flag, const std::string& val) {
    const std::string& c = r.canon;
    auto warn = [&](const std::string& m) { o.warnings.push_back(m); };
    if (c == "model") o.model = val;
    else if (c == "models-preset") o.models_preset = val;
    else if (c == "models-max") o.models_max = to_int(val, flag);
    else if (c == "ctx-size") o.ctx = to_int(val, flag);
    else if (c == "threads") o.threads = to_int(val, flag);
    else if (c == "threads-http") o.http_threads = to_int(val, flag);
    else if (c == "n-gpu-layers" || c == "gpu-layers") {
        o.ngl_given = true;
        if (val == "auto" || val == "all") o.n_gpu_layers = 999;
        else o.n_gpu_layers = to_int(val, flag);
    } else if (c == "batch-size") o.batch = to_int(val, flag);
    else if (c == "ubatch-size") o.ubatch = to_int(val, flag);
    else if (c == "host") o.host = val;
    else if (c == "port") o.port = to_int(val, flag);
    else if (c == "api-key") { for (auto& k : split(val, ',')) if (!k.empty()) o.api_keys.push_back(k); }
    else if (c == "api-key-file") {
        std::istringstream is(read_file(val));
        std::string ln;
        while (std::getline(is, ln)) {
            while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' ')) ln.pop_back();
            if (!ln.empty() && ln[0] != '#') o.api_keys.push_back(ln);
        }
    } else if (c == "alias") { o.alias.clear(); for (auto& a : split(val, ',')) if (!a.empty()) o.alias.push_back(a); }
    else if (c == "parallel") o.n_parallel = to_int(val, flag);
    else if (c == "cont-batching") {}
    else if (c == "cache-prompt") o.cache_prompt = !r.neg;
    else if (c == "jinja") o.jinja = !r.neg;
    else if (c == "chat-template") o.chat_template = val;
    else if (c == "chat-template-file") o.chat_template_file = val;
    else if (c == "chat-template-kwargs") o.chat_template_kwargs = val;
    else if (c == "reasoning-format") {
        if (val == "separate") o.reasoning_format = "deepseek";
        else if (val == "auto" || val == "none" || val == "deepseek" || val == "deepseek-legacy") o.reasoning_format = val;
        else fail("invalid --reasoning-format '" + val + "' (auto, none, deepseek, deepseek-legacy)");
    } else if (c == "reasoning") o.reasoning = val == "on" || truthy(val) ? 1 : (val == "off" || falsey(val) ? 0 : -1);
    else if (c == "reasoning-budget") o.reasoning_budget = to_int(val, flag);
    else if (c == "reasoning-budget-message") o.reasoning_budget_message = val;
    else if (c == "reasoning-effort") o.reasoning_effort = val == "default" ? "" : val;
    else if (c == "metrics") o.metrics = true;
    else if (c == "slots") o.slots = !r.neg;
    else if (c == "props") o.props = true;
    else if (c == "embedding") o.embedding = true;
    else if (c == "predict") o.n_predict = to_int(val, flag);
    else if (c == "temp") o.temp = to_f(val, flag);
    else if (c == "top-k") o.top_k = to_int(val, flag);
    else if (c == "top-p") o.top_p = to_f(val, flag);
    else if (c == "min-p") o.min_p = to_f(val, flag);
    else if (c == "typical") o.typical_p = to_f(val, flag);
    else if (c == "top-nsigma") o.top_n_sigma = to_f(val, flag);
    else if (c == "xtc-probability") o.xtc_probability = to_f(val, flag);
    else if (c == "xtc-threshold") o.xtc_threshold = to_f(val, flag);
    else if (c == "repeat-penalty") o.repeat_penalty = to_f(val, flag);
    else if (c == "repeat-last-n") o.repeat_last_n = to_int(val, flag);
    else if (c == "presence-penalty") o.presence_penalty = to_f(val, flag);
    else if (c == "frequency-penalty") o.frequency_penalty = to_f(val, flag);
    else if (c == "dry-multiplier") o.dry_multiplier = to_f(val, flag);
    else if (c == "dry-base") o.dry_base = to_f(val, flag);
    else if (c == "dry-allowed-length") o.dry_allowed_length = to_int(val, flag);
    else if (c == "dry-penalty-last-n") o.dry_penalty_last_n = to_int(val, flag);
    else if (c == "dry-sequence-breaker") { if (!o.dry_breakers_user) { o.dry_sequence_breakers.clear(); o.dry_breakers_user = true; } if (val != "none") o.dry_sequence_breakers.push_back(val); }
    else if (c == "mirostat") o.mirostat = to_int(val, flag);
    else if (c == "mirostat-lr") o.mirostat_eta = to_f(val, flag);
    else if (c == "mirostat-ent") o.mirostat_tau = to_f(val, flag);
    else if (c == "dynatemp-range") o.dynatemp_range = to_f(val, flag);
    else if (c == "dynatemp-exp") o.dynatemp_exponent = to_f(val, flag);
    else if (c == "seed") { try { o.seed = std::stoll(val); } catch (...) { fail("invalid value '" + val + "' for " + flag); } }
    else if (c == "samplers") { o.samplers.clear(); for (auto& s : split(val, ';')) if (!s.empty()) o.samplers.push_back(s == "typical_p" ? "typ_p" : s == "top-k" ? "top_k" : s == "top-p" ? "top_p" : s == "min-p" ? "min_p" : s); }
    else if (c == "ignore-eos") o.ignore_eos = true;
    else if (c == "logit-bias") {
        size_t p = val.find_first_of("+-", 1);
        if (p == std::string::npos) fail("invalid --logit-bias '" + val + "' (TOKEN_ID(+/-)BIAS)");
        o.logit_bias.emplace_back(to_int(val.substr(0, p), flag), to_f(val.substr(p), flag));
    } else if (c == "timeout") o.timeout = to_int(val, flag);
    else if (c == "path") o.path = val;
    else if (c == "ui") o.webui = !r.neg;
    else if (c == "ui-config" || c == "ui-config-file") {
        o.ui_config = c == "ui-config" ? val : read_file(val);
        if (!o.ui_config.empty()) { try { (void)jd::parse(o.ui_config); } catch (const std::exception& e) { fail(std::string("failed to parse UI config: ") + e.what()); } }
    }
    else if (c == "ui-mcp-proxy") { if (!r.neg) warn("--ui-mcp-proxy: the CORS proxy (/cors-proxy) is not implemented by qv_server; the endpoint answers 403 as in a default llama-server"); }
    else if (c == "sleep-idle-seconds") o.sleep_idle_seconds = to_int(val, flag);
    else if (c == "sse-ping-interval") o.sse_ping_interval = to_int(val, flag);
    else if (c == "tools") { o.tools = val; if (!val.empty()) warn("--tools: server-side built-in tools (/tools: shell, file read/write) are deliberately not implemented by qv_server; the endpoint answers 403"); }
    else if (c == "log-disable") o.log_disable = true;
    else if (c == "verbose") o.verbosity = 4;
    else if (c == "log-verbosity" || c == "verbosity") o.verbosity = to_int(val, flag);
    else if (c == "slot-save-path") o.slot_save_path = val;
    else if (c == "special") o.special = true;
    else if (c == "cors-origins") o.cors_origins = val;
    else if (c == "cors-methods") o.cors_methods = val;
    else if (c == "cors-headers") o.cors_headers = val;
    else if (c == "cors-credentials") o.cors_credentials = !r.neg;
    else if (c == "api-prefix") o.api_prefix = val;
    else if (c == "min-keep") o.min_keep = to_int(val, flag);
    else if (c == "grammar") o.grammar = val;
    else if (c == "grammar-file") o.grammar = read_file(val);
    else if (c == "json-schema") o.json_schema = val;
    else if (c == "json-schema-file") o.json_schema = read_file(val);
    else if (c == "backend") o.backend = val;
    else if (c == "kv") { if (val == "int4") o.kv_mode = 1; else if (val == "fp16") o.kv_mode = 2; else if (val == "fp32") o.kv_mode = 0; else fail("--kv fp32|fp16|int4"); }
    else if (c == "kv4-kmode") o.kv4_kmode = val == "token" ? 1 : 0;
    else if (c == "mtp") o.mtp = true;
    else if (c == "no-mtp") o.mtp = false;
    else if (c == "mtp-grammar") o.mtp_grammar = true;
    else if (c == "no-mtp-grammar") o.mtp_grammar = false;
    else if (c == "mtp-depth") o.mtp_depth = to_int(val, flag);
    else if (c == "mtp-file") o.mtp_file = val;
    else if (c == "prefill-chunk") o.prefill_chunk = to_int(val, flag);
    else if (c == "tokenizer") o.tokenizer = val;
    else if (c == "no-embedded-defaults") o.embedded_defaults = false;
    else if (c == "max-queue") o.max_queue = to_int(val, flag);
    else if (c == "vision-file") o.vision_file = val;
    else if (c == "allow-remote-images") o.allow_remote_images = true;
    else if (c == "image-root") o.image_root = val;
    else if (c == "image-max-side") o.image_max_side = to_int(val, flag);
    else if (c == "image-token") o.image_token = val;
    else if (c == "mock-text") o.mock_text = val;
    else if (c == "mock-loop") o.mock_loop = true;
    else if (c == "mock-delay-us") o.mock_delay_us = to_int(val, flag);
    else if (c == "checkpoint-slots") o.checkpoint_slots = std::max(0, to_int(val, flag));
    else if (c == "checkpoint-ram-mb") o.checkpoint_ram_mb = std::max(0, to_int(val, flag));
    else if (c == "checkpoint-align") o.checkpoint_align = to_int(val, flag);
    else if (c == "checkpoint-min-tokens") o.checkpoint_min_tokens = std::max(0, to_int(val, flag));
    else if (c == "lora" || c == "lora-scaled") fail("LoRA adapters are not applied at runtime by qv_server: merge the adapter at conversion time (tools/convert.py --lora DIR); refusing to start with " + flag);
    else if (c == "ssl-key-file" || c == "ssl-cert-file") fail("the server is built without SSL support (" + flag + ")");
    else if (c == "hf-repo" || c == "model-url" || c == "docker-repo") fail(flag + ": downloading models is not supported; pass a local Q-Volta model with -m");
    else if (c == "context-shift") { if (!r.neg) warn("--context-shift: the recurrent GatedDeltaNet state cannot be shifted; generation stops at the context limit (ignored)"); }
    else if (c == "kv-unified" || c == "cache-idle-slots" || c == "warmup" || c == "flash-attn" || c == "mlock" || c == "no-mlock" || c == "mmap" || c == "no-mmap" || c == "perf" || c == "escape" || c == "log-colors" ||
             c == "log-timestamps" || c == "log-prefix" || c == "reasoning-preserve" || c == "hf-file" || c == "hf-token" || c == "offline") {}
    else warn("flag " + flag + " is accepted for llama-server compatibility but has no effect in qv_server");
}

}  // namespace

bool flag_known(const std::string& s) { Resolved r; return resolve(s, r); }

bool apply_preset_option(Options& o, const std::string& key, const std::string& val, std::string& msg) {
    msg.clear();
    std::string flag = "--" + key;
    Resolved r;
    if (!resolve(flag, r)) return false;
    if (r.canon == "models-preset" || r.canon == "models-max" || r.canon == "host" || r.canon == "port") {
        msg = "server-wide option '" + key + "' is ignored inside a preset section";
        return false;
    }
    if (!handled().count(r.canon) && !r.own) {
        msg = "option '" + key + "' has no effect in qv_server";
        return false;
    }
    std::string v = val;
    if (r.arg == 0) {
        if (falsey(v)) {
            Resolved nr;
            if (resolve("--no-" + r.canon, nr)) r = nr;
            else if (r.canon == "mtp" || r.canon == "cache-prompt" || r.canon == "jinja" || r.canon == "slots" || r.canon == "ui" || r.canon == "cors-credentials") { r.neg = true; }
            else { msg = "option '" + key + "' cannot be disabled with false"; return false; }
        } else if (!v.empty() && !truthy(v)) {
            throw std::runtime_error("invalid boolean value '" + v + "' for '" + key + "'");
        }
    } else if (r.arg == 2 && v.empty()) {
        v = "on";
    }
    size_t prev_warns = o.warnings.size();
    apply(o, r, flag, v);
    if (o.warnings.size() > prev_warns) {
        msg = o.warnings.back();
        o.warnings.resize(prev_warns);
    }
    o.given.insert(r.canon);
    return true;
}

int parse_args(int argc, char** argv, Options& o, bool& exit_now) {
    exit_now = false;
    for (int i = 0; i < argc; ++i) { if (i) o.cmdline += ' '; o.cmdline += argv[i]; }
    std::set<std::string> given;  // canonical names given on the command line
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i], inline_val;
            bool has_inline = false;
            if (a.rfind("--", 0) == 0) {
                size_t eq = a.find('=');
                if (eq != std::string::npos) { inline_val = a.substr(eq + 1); a = a.substr(0, eq); has_inline = true; }
            }
            Resolved r;
            if (!resolve(a, r)) { std::fprintf(stderr, "error: invalid argument: %s\n", a.c_str()); return 1; }
            if (r.canon == "help") { exit_now = true; return -1; }
            if (r.canon == "version") { exit_now = true; return -2; }
            std::string val;
            if (r.arg == 1) {
                if (has_inline) val = inline_val;
                else if (i + 1 < argc) val = argv[++i];
                else { std::fprintf(stderr, "error: expected value for argument: %s\n", a.c_str()); return 1; }
            } else if (r.arg == 2) {
                if (has_inline) val = inline_val;
                else if (i + 1 < argc && (truthy(argv[i + 1]) || falsey(argv[i + 1]) || std::string(argv[i + 1]) == "auto")) val = argv[++i];
                else val = "on";
            }
            given.insert(r.canon);
            if (!handled().count(r.canon) && !r.own) o.warnings.push_back("flag " + a + " is accepted for llama-server compatibility but has no effect in qv_server");
            else apply(o, r, a, val);
        }
        // environment (command line wins)
        for (auto& e : table()) {
            if (e.env.empty()) continue;
            std::string canon;
            for (auto& m : e.names) if (m.rfind("--", 0) == 0 && !is_neg_spelling(m)) { canon = m.substr(2); break; }
            if (canon.empty()) canon = e.names[0].substr(1);
            if (given.count(canon)) continue;
            for (auto& en : split(e.env, ' ')) {
                const char* v = std::getenv(en.c_str());
                if (!v) continue;
                Resolved r;
                r.e = &e; r.canon = canon; r.arg = e.arg;
                std::string val = v;
                if (e.arg == 0) {
                    bool has_neg = false;
                    for (auto& m : e.names) if (is_neg_spelling(m)) has_neg = true;
                    if (falsey(val)) { if (!has_neg) continue; r.neg = true; }
                    else if (!truthy(val)) continue;
                } else if (e.arg == 2 && val.empty()) val = "on";
                if (!handled().count(canon)) { o.warnings.push_back("environment variable " + en + " is accepted for llama-server compatibility but has no effect in qv_server"); continue; }
                apply(o, r, "env " + en, val);
                given.insert(canon);
                break;
            }
        }
        if (o.alias.empty()) o.alias.clear();
        o.given = given;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

}  // namespace qv::server
