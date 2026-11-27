// Request parsing, generation loop and the completion / chat / infill handlers.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "server/app.h"
#include "grammar/json_schema.h"
#include "grammar/sampler_hook.h"

namespace qv::server {

namespace {

const char* tname(const Value& v) {
    switch (v.type()) {
        case Value::Type::Null: return "null";
        case Value::Type::Bool: return "boolean";
        case Value::Type::Int: case Value::Type::Num: return "number";
        case Value::Type::Str: return "string";
        case Value::Type::Arr: return "array";
        default: return "object";
    }
}
[[noreturn]] void type_err(const char* field, const char* want, const Value& got) {
    throw_error(400, std::string("Field '") + field + "': [json.exception.type_error.302] type must be " + want + ", but is " + tname(got));
}

// llama.cpp json_value(): missing / null -> default, wrong type -> 400 with nlohmann's message
struct Body {
    const Value& b;
    const Value* g(const char* k) const { const Value* v = b.find(k); return (v && !v->is_null()) ? v : nullptr; }
    double num(const char* k, double d) const { auto* v = g(k); if (!v) return d; if (!v->is_number()) type_err(k, "number", *v); return v->as_double(); }
    int integer(const char* k, int d) const { auto* v = g(k); if (!v) return d; if (!v->is_number()) type_err(k, "number", *v); return (int)v->as_int(); }
    bool boolean(const char* k, bool d) const { auto* v = g(k); if (!v) return d; if (!v->is_bool()) type_err(k, "boolean", *v); return v->as_bool(); }
    std::string str(const char* k, const std::string& d) const { auto* v = g(k); if (!v) return d; if (!v->is_string()) type_err(k, "string", *v); return v->as_string(); }
};

std::string valid_utf8(const std::string& s) {  // replace invalid sequences with U+FFFD (for token "text" fields)
    std::string o;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        bool ok = n > 0 && i + n <= s.size();
        for (int k = 1; ok && k < n; ++k) ok = ((unsigned char)s[i + k] & 0xC0) == 0x80;
        if (ok) { o.append(s, i, n); i += n; } else { o += "\xEF\xBF\xBD"; ++i; }
    }
    return o;
}

Value int_array(const std::vector<int>& v) { Value a = Value::array(); for (int x : v) a.push(x); return a; }

}  // namespace

[[noreturn]] void throw_error(int status, const std::string& msg, const std::string& type) {
    std::string t = type;
    if (t.empty()) {
        switch (status) {
            case 400: t = "invalid_request_error"; break;
            case 401: t = "authentication_error"; break;
            case 403: t = "permission_error"; break;
            case 404: t = "not_found_error"; break;
            case 501: t = "not_supported_error"; break;
            case 503: t = "unavailable_error"; break;
            default: t = "server_error";
        }
    }
    throw ApiError{status, msg, t, Value()};
}

std::string error_json(const ApiError& e) {
    Value err = Value::object();
    err.set("code", e.status);
    err.set("message", e.message);
    err.set("type", e.type);
    if (e.extra.is_object()) for (auto& m : e.extra.members()) err.set(m.first, m.second);
    Value o = Value::object();
    o.set("error", std::move(err));
    return o.dump();
}

// ------------------------------------------------------------------------------------------------ token text helpers
std::string App::piece_text(int id) const {
    std::string b = tok.token_bytes(id);
    return valid_utf8(b);
}
std::string App::piece_bytes(int id, std::vector<int>& bytes) const {
    std::string b = tok.token_bytes(id);
    bytes.clear();
    for (unsigned char c : b) bytes.push_back(c);
    return valid_utf8(b);
}

// ------------------------------------------------------------------------------------------------ JSON builders
Value App::gen_settings(const Task& t, int flavor) const {  // flavor 0: completion response, 1: /slots params and /props defaults
    const SamplingParams& s = t.sp;
    Value o = Value::object();
    auto f = [](float x) { return Value((double)x); };
    o.set("seed", (long long)(s.has_seed ? s.seed : 4294967295ull));
    o.set("temperature", f(s.temperature));
    o.set("dynatemp_range", f(s.dynatemp_range));
    o.set("dynatemp_exponent", f(s.dynatemp_exponent));
    o.set("top_k", s.top_k);
    o.set("top_p", f(s.top_p));
    o.set("min_p", f(s.min_p));
    o.set("top_n_sigma", f(s.top_n_sigma));
    o.set("xtc_probability", f(s.xtc_probability));
    o.set("xtc_threshold", f(s.xtc_threshold));
    o.set("typical_p", f(s.typical_p));
    o.set("repeat_last_n", s.repeat_last_n);
    o.set("repeat_penalty", f(s.repeat_penalty));
    o.set("presence_penalty", f(s.presence_penalty));
    o.set("frequency_penalty", f(s.frequency_penalty));
    o.set("dry_multiplier", f(s.dry_multiplier));
    o.set("dry_base", f(s.dry_base));
    o.set("dry_allowed_length", s.dry_allowed_length);
    o.set("dry_penalty_last_n", s.dry_penalty_last_n);
    if (flavor == 0) { Value a = Value::array(); for (auto& x : s.dry_sequence_breakers) a.push(x); o.set("dry_sequence_breakers", std::move(a)); }
    o.set("mirostat", s.mirostat);
    o.set("mirostat_tau", f(s.mirostat_tau));
    o.set("mirostat_eta", f(s.mirostat_eta));
    o.set("adaptive_target", -1.0);
    o.set("adaptive_decay", f(0.9f));
    if (flavor == 0) { Value a = Value::array(); for (auto& x : t.stops) a.push(x); o.set("stop", std::move(a)); }
    o.set("max_tokens", t.n_predict);
    o.set("n_predict", t.n_predict);
    o.set("n_keep", t.n_keep);
    o.set("n_discard", 0);
    o.set("ignore_eos", t.ignore_eos);
    o.set("stream", t.stream);
    if (flavor == 0) {
        Value lb = Value::array();
        for (auto& [id, b] : s.logit_bias) { Value p = Value::array(); p.push(id); p.push(f(b)); lb.push(std::move(p)); }
        o.set("logit_bias", std::move(lb));
    }
    o.set("n_probs", t.n_probs);
    o.set("min_keep", s.min_keep);
    if (flavor == 0) {
        o.set("grammar", t.grammar);
        o.set("grammar_lazy", t.grammar_lazy);
        {
        Value gt = Value::array();
        for (auto& w : t.trigger_words) { Value x = Value::object(); x.set("type", 1); x.set("value", w); x.set("token", -1); gt.push(std::move(x)); }
        o.set("grammar_triggers", std::move(gt));
    }
        o.set("preserved_tokens", Value::array());
    }
    o.set("chat_format", "Content-only");
    o.set("reasoning_format", t.reasoning_format);
    o.set("reasoning_in_content", false);
    o.set("generation_prompt", "");
    Value sa = Value::array();
    for (auto& x : s.samplers) sa.push(x);
    o.set("samplers", std::move(sa));
    o.set("speculative.types", "none");
    o.set("timings_per_token", t.timings_per_token);
    o.set("post_sampling_probs", t.post_probs);
    o.set("backend_sampling", false);
    o.set("lora", Value::array());
    return o;
}

Value App::timings_json(const GenResult& r) const { return timings_json(r.n_cache, r.n_prompt_eval, r.prompt_ms, r.n_decoded, r.predicted_ms); }

Value App::timings_json(int cache, int pn, double pms, int dn, double dms) const {
    struct { int n_cache, n_prompt_eval, n_decoded; double prompt_ms, predicted_ms; } r{cache, pn, dn, pms, dms};
    Value o = Value::object();
    o.set("cache_n", r.n_cache);
    o.set("prompt_n", r.n_prompt_eval);
    o.set("prompt_ms", r.prompt_ms);
    o.set("prompt_per_token_ms", r.n_prompt_eval > 0 ? r.prompt_ms / r.n_prompt_eval : 0.0);
    o.set("prompt_per_second", r.prompt_ms > 0 ? 1e3 * r.n_prompt_eval / r.prompt_ms : 0.0);
    o.set("predicted_n", r.n_decoded);
    o.set("predicted_ms", r.predicted_ms);
    o.set("predicted_per_token_ms", r.n_decoded > 0 ? r.predicted_ms / r.n_decoded : 0.0);
    o.set("predicted_per_second", r.predicted_ms > 0 ? 1e3 * r.n_decoded / r.predicted_ms : 0.0);
    return o;
}

Value App::probs_json(const std::vector<TokProb>& v, bool post) const {
    Value arr = Value::array();
    for (auto& p : v) {
        Value e = Value::object();
        e.set("id", p.id);
        e.set("token", p.text);
        e.set("bytes", int_array(p.bytes));
        e.set(post ? "prob" : "logprob", (double)(post ? p.prob : p.logprob));
        Value top = Value::array();
        for (auto& [id, text, bytes, x] : p.top) {
            Value t = Value::object();
            t.set("id", id);
            t.set("token", text);
            t.set("bytes", int_array(bytes));
            t.set(post ? "prob" : "logprob", (double)x);
            top.push(std::move(t));
        }
        e.set(post ? "top_probs" : "top_logprobs", std::move(top));
        arr.push(std::move(e));
    }
    return arr;
}

// ------------------------------------------------------------------------------------------------ request parsing
std::vector<int> App::tokenize_prompt_value(const Value& p) {
    auto check = [&](const Value& x) {
        if (!x.is_int()) throw_error(400, "prompt must be a string, a list of tokens, or a mixed list of strings and tokens");
        if (x.as_int() < 0 || x.as_int() >= tok.vocab_size()) throw_error(400, "invalid token id in prompt");
        return (int)x.as_int();
    };
    if (p.is_string()) return tok.encode(p.as_string(), true);
    if (p.is_array()) {
        std::vector<int> out;
        for (auto& x : p.items()) {
            if (x.is_string()) { auto t = tok.encode(x.as_string(), true); out.insert(out.end(), t.begin(), t.end()); }
            else out.push_back(check(x));
        }
        return out;
    }
    if (p.is_object()) {
        const Value* ps = p.find("prompt_string");
        if (ps && ps->is_string()) return tok.encode(ps->as_string(), true);
    }
    throw_error(400, "\"prompt\" must be a string, an array of token ids or an array of prompts");
}

void App::parse_sampling(const Value& bv, Task& t) {
    Body b{bv};
    SamplingParams& s = t.sp;
    s.temperature = (float)b.num("temperature", o.temp);
    s.top_k = b.integer("top_k", o.top_k);
    s.top_p = (float)b.num("top_p", o.top_p);
    s.min_p = (float)b.num("min_p", o.min_p);
    s.typical_p = (float)b.num("typical_p", o.typical_p);
    s.top_n_sigma = (float)b.num("top_n_sigma", o.top_n_sigma);
    s.xtc_probability = (float)b.num("xtc_probability", o.xtc_probability);
    s.xtc_threshold = (float)b.num("xtc_threshold", o.xtc_threshold);
    s.min_keep = b.integer("min_keep", o.min_keep);
    s.dynatemp_range = (float)b.num("dynatemp_range", o.dynatemp_range);
    s.dynatemp_exponent = (float)b.num("dynatemp_exponent", o.dynatemp_exponent);
    s.repeat_last_n = b.integer("repeat_last_n", o.repeat_last_n);
    s.repeat_penalty = (float)b.num("repeat_penalty", b.g("repetition_penalty") ? b.num("repetition_penalty", 1.0) : o.repeat_penalty);
    s.presence_penalty = (float)b.num("presence_penalty", o.presence_penalty);
    s.frequency_penalty = (float)b.num("frequency_penalty", o.frequency_penalty);
    s.dry_multiplier = (float)b.num("dry_multiplier", o.dry_multiplier);
    s.dry_base = (float)b.num("dry_base", o.dry_base);
    s.dry_allowed_length = b.integer("dry_allowed_length", o.dry_allowed_length);
    s.dry_penalty_last_n = b.integer("dry_penalty_last_n", o.dry_penalty_last_n);
    s.dry_sequence_breakers = o.dry_sequence_breakers;
    if (const Value* v = b.g("dry_sequence_breakers")) {
        if (!v->is_array()) type_err("dry_sequence_breakers", "array", *v);
        s.dry_sequence_breakers.clear();
        for (auto& x : v->items()) if (x.is_string()) s.dry_sequence_breakers.push_back(x.as_string());
    }
    if (tok.vocab_size() > 0) for (auto& br : s.dry_sequence_breakers) { auto ids = tok.encode(br, false); if (!ids.empty()) s.dry_breaker_tokens.push_back(ids); }
    s.mirostat = b.integer("mirostat", o.mirostat);
    s.mirostat_tau = (float)b.num("mirostat_tau", o.mirostat_tau);
    s.mirostat_eta = (float)b.num("mirostat_eta", o.mirostat_eta);
    long long seed = (long long)b.num("seed", (double)o.seed);
    s.has_seed = seed >= 0 && seed != 4294967295ll;
    s.seed = (uint64_t)seed;
    s.samplers = o.samplers;
    if (const Value* v = b.g("samplers")) {
        if (v->is_array()) {
            s.samplers.clear();
            for (auto& x : v->items()) if (x.is_string()) s.samplers.push_back(x.as_string());
        } else type_err("samplers", "array", *v);
        // llama drops unknown names: an invalid list leaves an empty chain (distribution sampling on raw logits)
        static const char* known[] = {"penalties", "dry", "top_n_sigma", "top_k", "typ_p", "typical_p", "top_p", "min_p", "xtc", "temperature"};
        std::vector<std::string> keep;
        for (auto& x : s.samplers) {
            std::string n = x == "typical_p" ? "typ_p" : x;
            for (auto* k : known) if (n == k) { keep.push_back(n); break; }
        }
        s.samplers = keep;
    }
    t.ignore_eos = b.boolean("ignore_eos", o.ignore_eos);
    s.logit_bias = o.logit_bias;
    if (const Value* v = b.g("logit_bias")) {
        auto add = [&](const Value& key, const Value& val) {
            float bias = val.is_bool() ? (val.as_bool() ? 0.0f : -INFINITY) : (val.is_number() ? (float)val.as_double() : 0.0f);
            if (val.is_bool() && val.as_bool()) return;
            if (key.is_int()) s.logit_bias.emplace_back((int)key.as_int(), bias);
            else if (key.is_string()) for (int id : tok.encode(key.as_string(), false)) s.logit_bias.emplace_back(id, bias);
        };
        if (v->is_array()) { for (auto& e : v->items()) if (e.is_array() && e.size() == 2) add(e.items()[0], e.items()[1]); }
        else if (v->is_object()) { for (auto& m : v->members()) { try { add(Value((long long)std::stoll(m.first)), m.second); } catch (...) { add(Value(m.first), m.second); } } }
        else type_err("logit_bias", "array", *v);
    }
    if (t.ignore_eos) for (int id : stop_ids) s.logit_bias.emplace_back(id, -INFINITY);
    t.n_probs = b.integer("n_probs", 0);
    t.post_probs = b.boolean("post_sampling_probs", false);
    t.n_keep = b.integer("n_keep", 0);
    t.cache_prompt = b.boolean("cache_prompt", o.cache_prompt);
    t.id_slot = b.integer("id_slot", -1);
    t.timings_per_token = b.boolean("timings_per_token", false);
    t.return_progress = b.boolean("return_progress", false);
    if (b.boolean("reasoning_control", false)) t.reasoning_end = std::make_shared<std::atomic<bool>>(false);
    t.sse_ping_interval = b.integer("sse_ping_interval", o.sse_ping_interval);
    t.return_tokens = b.boolean("return_tokens", false);
    t.stream = b.boolean("stream", false);
    t.n_predict = b.integer("n_predict", b.integer("max_tokens", b.integer("max_completion_tokens", o.n_predict)));
    if (const Value* v = b.g("n_predict")) (void)v; else if (const Value* m = b.g("max_tokens")) (void)m;
    t.grammar = b.str("grammar", o.grammar);
    if (const Value* v = b.g("json_schema")) t.json_schema = *v;
    else if (!o.json_schema.empty()) { try { t.json_schema = jd::parse(o.json_schema); } catch (...) {} }
    if (const Value* v = b.g("stop")) {
        if (v->is_string()) t.stops.push_back(v->as_string());
        else if (v->is_array()) { for (auto& x : v->items()) { if (!x.is_string()) type_err("stop", "string", x); t.stops.push_back(x.as_string()); } }
        else type_err("stop", "array", *v);
        t.stops.erase(std::remove(t.stops.begin(), t.stops.end(), std::string()), t.stops.end());
    }
    if (const Value* v = b.g("response_fields")) t.response_fields = *v;
    // native endpoints: `grammar` wins over `json_schema` silently (llama-server); the chat path rejects the combination below
}

Task App::parse_task(const Value& body, Kind kind) {
    if (!body.is_object()) throw_error(400, "request body must be a JSON object");
    Body b{body};
    Task t;
    t.kind = kind;
    parse_sampling(body, t);
    std::string rf = o.reasoning_format;
    if (kind == Kind::OaiChat) rf = b.str("reasoning_format", rf);
    if (rf == "auto") rf = "deepseek";
    if (rf != "none" && rf != "deepseek" && rf != "deepseek-legacy") throw_error(500, "Unknown reasoning format: " + rf);  // llama-server answers 500 here
    t.reasoning_format = kind == Kind::Native || kind == Kind::Infill ? (o.reasoning_format == "auto" ? "deepseek" : o.reasoning_format) : rf;
    t.model_echo = model_id;
    if (kind == Kind::OaiChat) {
        const Value* tools = b.g("tools");
        bool has_tools = tools && tools->is_array() && tools->size() > 0;
        Value tc = body.find("tool_choice") ? *body.find("tool_choice") : Value("auto");
        std::string tool_choice = "auto", forced_name;
        if (tc.is_string()) tool_choice = tc.as_string();
        else if (tc.is_object()) {
            tool_choice = "required";
            if (const Value* f = tc.find("function"); f && f->is_object()) forced_name = Body{*f}.str("name", "");
            else forced_name = Body{tc}.str("name", "");
        } else if (!tc.is_null()) throw_error(400, "Invalid tool_choice: " + tc.dump());
        if (tool_choice != "auto" && tool_choice != "none" && tool_choice != "required") throw_error(400, "Invalid tool_choice: " + tool_choice);
        if (!o.jinja) {
            if (has_tools) throw_error(400, "tools param requires --jinja flag");
            if (tool_choice != "auto") throw_error(400, "tool_choice param requires --jinja flag");
        }
        if (tools && !tools->is_array()) throw_error(400, "'tools' must be an array of objects");
        if (!t.json_schema.is_null() && !t.grammar.empty()) throw_error(500, "Cannot use both json_schema and grammar");  // llama-server: 500 server_error
        if (const Value* r = b.g("response_format")) {
            std::string type = r->is_object() ? Body{*r}.str("type", "") : "";
            if (type == "json_object") {
                const Value* sch = r->find("schema");
                if (sch) t.json_schema = *sch; else if (t.json_schema.is_null() && t.grammar.empty()) t.json_object = true;
            } else if (type == "json_schema") {
                const Value* w = r->find("json_schema");
                if (w && w->is_object()) { const Value* sch = w->find("schema"); t.json_schema = sch ? *sch : Value::object(); }
                else { const Value* sch = r->find("schema"); t.json_schema = sch ? *sch : Value::object(); }
            } else if (!type.empty() && type != "text")
                throw_error(400, "response_format type must be one of \"text\", \"json_object\" or \"json_schema\", but got: " + type);
        }
        std::vector<std::string> image_urls;
        const Value* msgs = body.find("messages");
        if (!msgs) throw_error(400, "'messages' is required");
        if (!msgs->is_array()) throw_error(400, "Expected 'messages' to be an array");
        for (auto& m : msgs->items()) {
            std::string role = m.is_object() ? Body{m}.str("role", "") : "";
            const Value* c = m.is_object() ? m.find("content") : nullptr;
            if (role != "assistant" && !c) throw_error(400, "All non-assistant messages must contain 'content'");
            if (role == "assistant") {
                if (!c && !m.find("tool_calls")) throw_error(400, "Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (c && !c->is_string() && !c->is_null() && !c->is_array()) throw_error(400, "Expected 'content' to be a string or an array");
            if (c && c->is_array()) {
                for (auto& part : c->items()) {
                    std::string pt = part.is_object() ? Body{part}.str("type", "") : "";
                    if (pt == "input_audio" || pt == "video_url")
                        throw_error(500, "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj", "server_error");
                    if (pt == "image_url" || pt == "input_image" || pt == "image") {
                        if (!venc) throw_error(500, "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj", "server_error");
                        std::string url;
                        if (const Value* iu = part.find("image_url")) url = iu->is_string() ? iu->as_string() : (iu->is_object() && iu->find("url") && iu->find("url")->is_string() ? iu->find("url")->as_string() : "");
                        else if (const Value* iu2 = part.find("url"); iu2 && iu2->is_string()) url = iu2->as_string();
                        else if (const Value* iu3 = part.find("image"); iu3 && iu3->is_string()) url = iu3->as_string();
                        if (url.empty()) throw_error(400, "image part has no url (expected {\"type\":\"image_url\",\"image_url\":{\"url\":...}})");
                        image_urls.push_back(url);
                    }
                }
            }
        }
        // decode + preprocess every image now (CPU, cheap) so that bad images are a clean 400 before anything streams
        std::vector<std::shared_ptr<vision::Pixels>> images;
        std::vector<uint64_t> image_hash;
        for (auto& u : image_urls) {
            try {
                std::vector<uint8_t> bytes = vision::load_image_bytes(u, vsrc);
                uint64_t h = 1469598103934665603ull;
                for (uint8_t b : bytes) { h ^= b; h *= 1099511628211ull; }
                vision::Rgb8 im = vision::decode_image(bytes.data(), bytes.size(), vsrc.limits);
                images.push_back(std::make_shared<vision::Pixels>(vision::preprocess(im, venc->cfg())));
                image_hash.push_back(h);
            } catch (const vision::ImageError& e) {
                throw_error(400, std::string("image: ") + e.what());
            }
        }
        ChatInput ci;
        ci.messages = *msgs;
        if (has_tools) ci.tools = *tools;
        ci.tool_choice = tool_choice;
        ci.parallel_tool_calls = b.boolean("parallel_tool_calls", false);
        ci.add_generation_prompt = b.boolean("add_generation_prompt", true);
        Value kwargs = Value::object();
        if (!o.chat_template_kwargs.empty()) { try { kwargs = jd::parse(o.chat_template_kwargs); } catch (...) {} }
        if (const Value* k = b.g("chat_template_kwargs")) {
            if (!k->is_object()) type_err("chat_template_kwargs", "object", *k);
            for (auto& m : k->members()) kwargs.set(m.first, m.second);
        }
        if (o.reasoning == 0) ci.enable_thinking = false;
        if (o.reasoning_budget == 0) ci.enable_thinking = false;
        if (const Value* v = kwargs.find("enable_thinking")) {
            if (v->is_bool()) ci.enable_thinking = v->as_bool();
            else if (v->is_string()) throw_error(400, "invalid type for \"enable_thinking\" (expected boolean, got string)");
        }
        ci.reasoning_effort = o.reasoning_effort;
        if (const Value* v = kwargs.find("reasoning_effort"); v && v->is_string()) ci.reasoning_effort = v->as_string();
        if (const Value* re = b.g("reasoning_effort")) {
            if (re->is_string()) {
                std::string e = re->as_string();
                if (e == "none") { ci.enable_thinking = false; ci.reasoning_effort.clear(); }
                else if (!e.empty()) ci.reasoning_effort = e == "high" ? "xhigh" : e == "minimal" ? "low" : e;
            }
        }
        ci.kwargs = kwargs;
        ChatOutput co;
        try { co = tmpl->apply(ci); }
        catch (const ApiError&) { throw; }
        catch (const std::exception& e) { throw_error(400, e.what()); }
        t.prompt_text = co.prompt;
        t.prompt = tok.encode(co.prompt, true);
        if (!images.empty() || (venc && image_token_id >= 0 && std::count(t.prompt.begin(), t.prompt.end(), image_token_id) > 0)) {
            // each template placeholder token becomes one token per image embedding row (HF Inkling2Processor.replace_image_token)
            size_t np = std::count(t.prompt.begin(), t.prompt.end(), image_token_id);
            if (np != images.size()) throw_error(400, "the chat template produced " + std::to_string(np) + " image placeholders for " + std::to_string(images.size()) + " images");
            std::vector<int> ex;
            size_t k = 0;
            for (int id : t.prompt) {
                if (id != image_token_id) { ex.push_back(id); continue; }
                ImagePart ip;
                ip.start = (int)ex.size();
                ip.n = images[k]->tokens();
                ip.px = images[k];
                for (int j = 0; j < ip.n; ++j) {  // negative pseudo id: unique per (image content, row) so the prompt cache can only match the identical image
                    uint64_t h = (image_hash[k] ^ ((uint64_t)(j + 1) * 0x9E3779B97F4A7C15ull)) * 0xff51afd7ed558ccdull;
                    ex.push_back(-1 - (int)((h >> 33) & 0x3fffffff));
                }
                t.images.push_back(ip);
                ++k;
            }
            t.prompt = std::move(ex);
        }
        t.thinking_open = co.thinking_open;
        t.think_start = co.think_start;
        t.think_end = co.think_end;
        t.reasoning_budget = o.reasoning_budget;
        t.reasoning_budget_msg = o.reasoning_budget_message;
        t.stops.insert(t.stops.end(), co.additional_stops.begin(), co.additional_stops.end());
        t.tool_fmt = co.tool_format;
        if (has_tools) {
            t.tools = *tools;
            t.parse_tools = tool_choice != "none";
            if (t.parse_tools && t.grammar.empty() && t.json_schema.is_null() && !t.json_object) {
                grammar::ToolGrammarOptions go;
                go.format = co.tool_format;
                go.forced_name = forced_name;
                go.parallel = ci.parallel_tool_calls;
                go.thinking_forced_open = co.thinking_open;
                go.think_end = co.think_end;
                try {
                    grammar::ToolGrammar tg = grammar::tool_call_grammar(*tools, go);
                    if (tool_choice == "required") t.grammar = tg.gbnf;
                    else if (!tg.trigger_words.empty() && !tg.gbnf_lazy.empty()) { t.grammar = tg.gbnf_lazy; t.grammar_lazy = true; t.trigger_words = tg.trigger_words; }
                } catch (const std::exception& e) { throw_error(400, std::string("tool grammar: ") + e.what()); }
            }
        }
        if (b.boolean("logprobs", false)) {
            if (has_tools && t.stream) throw_error(400, "logprobs is not supported with tools + stream");
            t.n_probs = b.integer("top_logprobs", 20);
        } else if (b.g("top_logprobs")) throw_error(400, "top_logprobs requires logprobs to be set to true");
        t.n_choices = std::max(1, b.integer("n", 1));
        if (const Value* so = b.g("stream_options"); so && so->is_object()) t.include_usage = Body{*so}.boolean("include_usage", false);
    } else if (kind == Kind::OaiCmpl) {
        const Value* p = body.find("prompt");
        if (!p) throw_error(400, "[json.exception.out_of_range.403] key 'prompt' not found");
        if (const Value* lp = b.g("logprobs"); lp && lp->is_number()) t.n_probs = (int)lp->as_int();
        t.n_choices = std::max(1, b.integer("n", 1));
    }
    finalize_constraints(t);
    return t;
}

void App::finalize_constraints(Task& t) {
    try {
        if (t.grammar.empty() && t.json_object) t.grammar = grammar::json_object_grammar();
        else if (t.grammar.empty() && !t.json_schema.is_null()) t.grammar = grammar::json_schema_to_grammar(t.json_schema);
        if (!t.grammar.empty()) (void)grammar::Grammar::parse(t.grammar);  // validate now: bad grammars are a 400, not a mid-stream failure
    } catch (const std::exception& e) {
        throw_error(400, std::string("Failed to parse grammar: ") + e.what());
    }
}

std::shared_ptr<grammar::Matcher> App::get_matcher(const std::string& gbnf) {
    std::lock_guard<std::mutex> lk(gm_);
    auto it = matchers_.find(gbnf);
    if (it != matchers_.end()) return it->second;
    if (!vocab_) vocab_ = grammar::vocab_from_tokenizer(tok, stop_ids);
    if (matchers_.size() >= 16) matchers_.clear();
    auto m = std::make_shared<grammar::Matcher>(grammar::Grammar::parse(gbnf), vocab_);
    matchers_[gbnf] = m;
    return m;
}

// ------------------------------------------------------------------------------------------------ generation
int App::pick_slot_locked(const Waiter& w) const {
    if (!multi_seq) {
        if (n_busy_ >= 1) return -1;
        return (w.want >= 0 && w.want < n_slots) ? w.want : owner_slot_;
    }
    if (w.want >= 0 && w.want < n_slots) return slot_busy_[w.want] ? -1 : w.want;
    int best = -1;
    size_t best_len = 0;
    for (int s = 0; s < n_slots; ++s) {  // 1. idle slot whose cached context (or prefix checkpoint) is a prefix of the prompt (longest wins)
        if (slot_busy_[s]) continue;
        auto consider = [&](const std::vector<int>& c) {
            if (w.prompt && !c.empty() && c.size() <= w.prompt->size() && c.size() > best_len && std::equal(c.begin(), c.end(), w.prompt->begin())) { best = s; best_len = c.size(); }
        };
        consider(slot_cached_[s]);
        for (const auto& ck : slot_ckpts_[s]) consider(ck.tokens);
    }
    if (best >= 0) return best;
    for (int s = 0; s < n_slots; ++s) if (!slot_busy_[s] && slot_cached_[s].empty() && slot_ckpts_[s].empty()) return s;  // 2. an empty slot
    for (int s = 0; s < n_slots; ++s)                                                          // 3. least recently used
        if (!slot_busy_[s] && (best < 0 || slot_used_[s] < slot_used_[best])) best = s;
    return best;
}

void App::dispatch_locked() {
    bool excl_ahead = false;
    for (Waiter& w : waiters_) {
        if (w.granted) continue;
        if (w.kind == 1) {
            if (!excl_held_ && n_busy_ == 0 && n_users_ == 0) { w.granted = true; excl_held_ = true; }
            excl_ahead = true;
        } else if (w.kind == 2) {
            if (!excl_held_ && !excl_ahead) { w.granted = true; ++n_users_; }
        } else {
            if (excl_held_ || (excl_ahead && !w.user)) continue;  // a model user (w.user) may finish its request while an exclusive waiter waits for it
            if (n_busy_ >= conc_limit_) continue;
            int sid = pick_slot_locked(w);
            if (sid < 0) continue;
            w.granted = true; w.sid = sid; slot_busy_[sid] = 1; ++n_busy_;
        }
    }
}

long App::wait_granted(int kind, int want, const std::vector<int>* prompt, bool user, const std::function<bool()>& alive, int* sid_out) {
    std::unique_lock<std::mutex> lk(mu_);
    const long ticket = next_ticket_++;
    waiters_.push_back(Waiter{ticket, kind, want, prompt, user, false, -1});
    dispatch_locked();
    auto find = [&] { for (auto it = waiters_.begin(); it != waiters_.end(); ++it) if (it->ticket == ticket) return it; return waiters_.end(); };
    if (o.max_queue >= 0 && kind != 2) {
        int pending = 0;
        for (const Waiter& w : waiters_) pending += !w.granted;
        auto it = find();
        if (!it->granted && pending > o.max_queue) {
            waiters_.erase(it);
            dispatch_locked();
            cv_.notify_all();
            throw_error(503, "The server is busy (request queue is full). Retry later.");
        }
    }
    cv_.notify_all();
    for (;;) {
        auto it = find();
        if (it->granted) {
            if (sid_out) *sid_out = it->sid;
            waiters_.erase(it);
            return ticket;
        }
        cv_.wait_for(lk, std::chrono::milliseconds(100));
        it = find();
        if (!it->granted && !alive()) {
            waiters_.erase(it);
            dispatch_locked();
            cv_.notify_all();
            return -1;
        }
    }
}

long App::lease_acquire(const std::function<bool()>& alive) { return wait_granted(1, -1, nullptr, false, alive, nullptr); }
void App::lease_release(long) {
    std::lock_guard<std::mutex> lk(mu_);
    excl_held_ = false;
    dispatch_locked();
    cv_.notify_all();
}
long App::user_acquire(const std::function<bool()>& alive) { return wait_granted(2, -1, nullptr, false, alive, nullptr); }
void App::user_release() {
    std::lock_guard<std::mutex> lk(mu_);
    --n_users_;
    dispatch_locked();
    cv_.notify_all();
}
void App::lease_downgrade() {
    std::lock_guard<std::mutex> lk(mu_);
    excl_held_ = false;
    ++n_users_;
    dispatch_locked();
    cv_.notify_all();
}
int App::slot_acquire(const Task& t, bool user, const std::function<bool()>& alive) {
    int sid = -1;
    long tk = wait_granted(0, t.id_slot, &t.prompt, user, alive, &sid);
    return tk < 0 ? -1 : sid;
}
void App::slot_release(int sid) {
    std::lock_guard<std::mutex> lk(mu_);
    slot_busy_[sid] = 0;
    --n_busy_;
    slot_used_[sid] = ++use_clock_;
    dispatch_locked();
    cv_.notify_all();
}

// prefill prompt tokens [off, off+n) of a task with images: placeholder rows take the encoder rows (negative ids mark them)
void App::prefill_with_images(const Task& t, int off, int n, const std::vector<std::vector<float>>& emb) {
    const int H = venc->cfg().text_hidden;
    std::vector<int> ids(t.prompt.begin() + off, t.prompt.begin() + off + n);
    std::vector<unsigned char> mask(n, 0);
    std::vector<float> rows;
    for (size_t k = 0; k < t.images.size(); ++k) {
        const ImagePart& ip = t.images[k];
        const int a = std::max(off, ip.start), b = std::min(off + n, ip.start + ip.n);
        for (int p = a; p < b; ++p) {
            mask[p - off] = 1;
            ids[p - off] = image_token_id >= 0 ? image_token_id : 0;
            const float* src = emb[k].data() + (size_t)(p - ip.start) * H;
            rows.insert(rows.end(), src, src + H);
        }
    }
    eng->prefill_embeds(ids.data(), rows.data(), mask.data(), n, off);
}

GenResult App::generate(Task& t, const std::function<bool()>& alive, const std::function<bool(const Chunk&)>& on_chunk, const std::function<void(int)>& on_start, long existing_ticket) {
    GenResult r;
    int busy_sum = 0;  // sum over decoded tokens of the number of running slots (n_busy_slots_per_decode)
    // ---- admission: FIFO queue for a free slot (slot i == engine sequence i; engines with one sequence run one request at a time)
    const int sid = slot_acquire(t, existing_ticket >= 0, alive);
    if (sid < 0) { r.aborted = true; return r; }
    struct Release { App& a; int sid; ~Release() { a.last_task_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); a.slot_release(sid); } } rel{*this, sid};
    try { ensure_engine(); } catch (const std::exception& e) { throw_error(503, std::string("model reload failed: ") + e.what(), "unavailable_error"); }
    std::vector<int>& cached_ = cached_of(sid);
    // one engine turn: select this slot's sequence and hold the engine until the guard dies (FIFO fair across the running slots)
    struct EngOp {
        std::unique_lock<FairMutex> lk;
        EngOp(App& a, int sid) : lk(a.eng_mu_) { a.eng->select_seq(a.seq_of(sid)); }
    };
    {
        std::lock_guard<std::mutex> lk(mu_);
        SlotInfo& s = slots_[sid];
        s.processing = true; s.has_task = true; s.id_task = task_counter_++;
        s.n_prompt = (int)t.prompt.size(); s.n_prompt_processed = 0; s.n_prompt_cache = 0; s.n_decoded = 0; s.n_remain = t.n_predict;
        s.has_next_token = true; s.has_new_line = false;
        s.params = gen_settings(t, 1);
        owner_slot_ = sid;
    }
    r.id_slot = sid;
    struct SlotGuard {
        App& a; int sid;
        ~SlotGuard() { std::lock_guard<std::mutex> lk(a.mu_); a.slots_[sid].processing = false; a.slots_[sid].has_next_token = false; }
    } sg{*this, sid};

    auto now = [] { return Clock::now(); };
    auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const int N = (int)t.prompt.size();
    r.n_prompt = N;
    if (!alive()) { r.aborted = true; return r; }
    try {
        const int sq = seq_of(sid);
        auto& ckpts_ = ckpts_of(sid);
        constexpr int kCkptTail = 16;
        constexpr size_t kCkptMinDelta = 1024;
        const size_t max_ckpts = multi_seq ? 1u : 2u;
        const int min_ckpt_toks = std::max(1, o.checkpoint_min_tokens);
        const bool l2_enabled = o.checkpoint_slots > 0 && o.checkpoint_ram_mb > 0 &&
                                eng->has_checkpoint() && eng->has_kv_checkpoint();
        const size_t max_ram_bytes = (size_t)o.checkpoint_ram_mb * (1024ull * 1024ull);

        auto park_current_slot = [&]() {
            if (!l2_enabled || !slot_cacheable_[sq] || (int)cached_.size() < min_ckpt_toks) return;
            for (int id : cached_) if (id < 0) return;  // multimodal placeholder tokens cannot be safely parked
            int best_pre_ci = -1, exact_end_ci = -1;
            size_t best_pre_len = 0;
            for (int ci = 0; ci < (int)ckpts_.size(); ++ci) {
                const size_t len = ckpts_[ci].tokens.size();
                if (len == cached_.size()) exact_end_ci = ci;
                else if (len < cached_.size() && len > best_pre_len) {
                    best_pre_len = len;
                    best_pre_ci = ci;
                }
            }
            const size_t kv_est = eng->kv_checkpoint_bytes((int)cached_.size());
            if (kv_est == 0) return;
            size_t est = kv_est;
            if (best_pre_ci >= 0) est += ckpts_[best_pre_ci].bytes();
            if (exact_end_ci >= 0) est += ckpts_[exact_end_ci].bytes();
            else est += cached_.size() * sizeof(int) + eng->checkpoint_bytes();
            if (!parked_pool_.reserve_for(est, o.checkpoint_slots, max_ram_bytes)) return;

            auto tp_park0 = now();
            ParkedSlot ps;
            ps.kv_tokens = (int)cached_.size();
            std::vector<uint8_t> end_blob;
            bool ok_save = false;
            {
                EngOp op(*this, sid);
                ok_save = eng->save_kv_prefix(ps.kv_blob, ps.kv_tokens);
                if (ok_save && exact_end_ci < 0) ok_save = eng->save_checkpoint(end_blob);
            }
            if (!ok_save) return;
            if (best_pre_ci >= 0) ps.points.push_back(ckpts_[best_pre_ci]);
            if (exact_end_ci >= 0) ps.points.push_back(ckpts_[exact_end_ci]);
            else ps.points.push_back(PromptCheckpoint{cached_, std::move(end_blob)});
            const size_t n_pts = ps.points.size();
            const size_t ps_bytes = ps.bytes();
            const int ps_toks = ps.kv_tokens;
            if (parked_pool_.insert(std::move(ps), o.checkpoint_slots, max_ram_bytes) &&
                o.verbosity >= 3 && !o.log_disable) {
                std::fprintf(stderr,
                             "[qv_server] slot %d parked L2 checkpoint: tokens=%d points=%zu (%.1f MB, %.0fms, pool=%zu/%d %.1f MB)\n",
                             sid, ps_toks, n_pts, ps_bytes / (1024.0 * 1024.0), ms(tp_park0, now()),
                             parked_pool_.size(), o.checkpoint_slots, parked_pool_.total_bytes() / (1024.0 * 1024.0));
            }
        };

        size_t start = 0;
        if (t.cache_prompt && !cached_.empty() && cached_.size() <= (size_t)N && std::equal(cached_.begin(), cached_.end(), t.prompt.begin())) {
            start = cached_.size();
        } else if (t.cache_prompt) {
            int best_l1_ci = -1;
            size_t best_l1_len = 0;
            if (!ckpts_.empty() && eng->has_checkpoint()) {
                for (int ci = 0; ci < (int)ckpts_.size(); ++ci) {
                    const auto& toks = ckpts_[ci].tokens;
                    if (!toks.empty() && toks.size() <= (size_t)N && toks.size() > best_l1_len &&
                        std::equal(toks.begin(), toks.end(), t.prompt.begin())) {
                        best_l1_len = toks.size();
                        best_l1_ci = ci;
                    }
                }
            }
            ParkedCheckpointPool::Match l2m;
            const bool have_l2 = l2_enabled && parked_pool_.take_best_match(t.prompt, best_l1_len, l2m);
            if (have_l2) {
                park_current_slot();
                auto tp_rest0 = now();
                const size_t l2_bytes = l2m.slot.bytes();
                bool ok = false;
                {
                    EngOp op(*this, sid);
                    ok = eng->restore_kv_prefix(l2m.slot.kv_blob) &&
                         eng->restore_checkpoint(l2m.slot.points[l2m.point_idx].blob);
                }
                if (ok) {
                    start = l2m.match_len;
                    cached_ = l2m.slot.points[l2m.point_idx].tokens;
                    ckpts_.clear();
                    for (auto& pt : l2m.slot.points) {
                        if (pt.tokens.size() <= start) ckpts_.push_back(std::move(pt));
                    }
                    std::sort(ckpts_.begin(), ckpts_.end(),
                              [](const PromptCheckpoint& a, const PromptCheckpoint& b) { return a.tokens.size() < b.tokens.size(); });
                    if (ckpts_.size() > max_ckpts) {
                        ckpts_.erase(ckpts_.begin(), ckpts_.end() - (std::ptrdiff_t)max_ckpts);
                    }
                    if (o.verbosity >= 3 && !o.log_disable) {
                        std::fprintf(stderr, "[qv_server] slot %d restored L2 checkpoint: cached=%zu/%d (%.1f MB, %.0fms)\n",
                                     sid, start, N, l2_bytes / (1024.0 * 1024.0), ms(tp_rest0, now()));
                    }
                }
            } else if (best_l1_ci >= 0) {
                bool ok = false;
                { EngOp op(*this, sid); ok = eng->restore_checkpoint(ckpts_[best_l1_ci].blob); }
                if (ok) {
                    start = best_l1_len;
                    cached_ = ckpts_[best_l1_ci].tokens;
                    ckpts_.erase(
                        std::remove_if(ckpts_.begin(), ckpts_.end(),
                                       [&](const PromptCheckpoint& c) { return c.tokens.size() > start; }),
                        ckpts_.end());
                }
            }
        }
        if (start == 0) {
            park_current_slot();
            { EngOp op(*this, sid); eng->reset(); }
            cached_.clear();
            ckpts_.clear();
        }
        if (start == 0) slot_cacheable_[sq] = (t.cache_prompt && t.images.empty()) ? 1 : 0;
        else if (!t.images.empty()) slot_cacheable_[sq] = 0;
        else if (t.cache_prompt) slot_cacheable_[sq] = 1;
        r.n_cache = (int)start;
        r.n_prompt_eval = N - (int)start;
        { std::lock_guard<std::mutex> lk(mu_); slots_[sid].n_prompt_cache = (int)start; }
        on_start(sid);
        auto tp0 = now();
        if (t.on_progress) t.on_progress(N, (int)start, (int)start, 0.0);
        std::vector<std::vector<float>> img_emb;  // encoder output per image (only images with rows at or after `start`)
        for (auto& ip : t.images) {
            img_emb.emplace_back();
            if (ip.start + ip.n <= (int)start) continue;
            if (!alive()) { r.aborted = true; return r; }
            img_emb.back().resize((size_t)ip.n * venc->cfg().text_hidden);
            { std::lock_guard<std::mutex> vlk(venc_mu_); venc->encode(*ip.px, img_emb.back().data()); }
        }
        const int ckpt_min_prompt = std::max(kCkptTail + 1, o.checkpoint_min_tokens);
        size_t prev_ck_len = 0;
        for (const auto& c : ckpts_) prev_ck_len = std::max(prev_ck_len, c.tokens.size());
        // Chunk grid: batched prefill numerics depend on the chunk boundaries (keys of earlier chunks are read from the int4 cache, keys inside a chunk
        // are exact fp16), so a restored checkpoint reproduces an uncached run bit for bit only if the checkpoint sits on the chunk grid and the rest of
        // the prompt continues on that grid. --checkpoint-align (default: the prefill chunk) cuts checkpoints on multiples of it; the cost is that up to
        // one grid step of the prompt tail is prefilled again on the next turn.
        const int align_g = o.checkpoint_align < 0 ? chunk : o.checkpoint_align;
        size_t cut_tgt = (size_t)std::max(0, N - kCkptTail);
        if (align_g > 0) cut_tgt = cut_tgt / (size_t)align_g * (size_t)align_g;
        const bool want_ckpt = t.cache_prompt && t.images.empty() && eng->has_checkpoint() &&
                               N >= ckpt_min_prompt && cut_tgt > start && cut_tgt > 0 &&
                               (prev_ck_len == 0 || cut_tgt >= prev_ck_len + (align_g > 0 ? (size_t)1 : kCkptMinDelta));
        const size_t ckpt_cut = want_ckpt ? cut_tgt : 0;
        bool ckpt_done = !want_ckpt;
        for (size_t off = start; off < (size_t)N;) {
            if (!alive()) { r.aborted = true; return r; }
            // prefill budget per engine turn: with other slots running, smaller chunks so their decode steps are not stalled for long
            const int ch = (multi_seq && n_busy_ > 1) ? std::min(chunk, std::max(1, shared_chunk)) : chunk;
            int n = (int)std::min<size_t>(ch, N - off);
            if (align_g > 0) n = std::min(n, align_g - (int)(off % (size_t)align_g));  // keep every chunk boundary on the grid (the first chunk after a restore may be short)
            if (!ckpt_done && off < ckpt_cut && off + (size_t)n > ckpt_cut) n = (int)(ckpt_cut - off);
            { EngOp op(*this, sid);
              if (t.images.empty()) eng->prefill(t.prompt.data() + off, n, (int)off);
              else prefill_with_images(t, (int)off, n, img_emb); }
            cached_.insert(cached_.end(), t.prompt.begin() + off, t.prompt.begin() + off + n);
            { std::lock_guard<std::mutex> lk(mu_); slots_[sid].n_prompt_processed = (int)(off + n - start); }
            if (t.on_progress) t.on_progress(N, (int)start, (int)(off + n), ms(tp0, now()));
            off += n;
            if (!ckpt_done && off == ckpt_cut) {
                PromptCheckpoint ck;
                ck.tokens = cached_;
                bool saved = false;
                { EngOp op(*this, sid); saved = eng->save_checkpoint(ck.blob); }
                if (saved) {
                    if (ckpts_.size() >= max_ckpts) ckpts_.back() = std::move(ck);
                    else ckpts_.push_back(std::move(ck));
                }
                ckpt_done = true;
            }
        }
        r.prompt_ms = ms(tp0, now());
        if (t.n_predict == 0) { r.stop_type = "limit"; r.tokens_cached = (int)cached_.size(); goto done; }

        {
            n_decoding_.fetch_add(1);
            struct DecGuard { std::atomic<int>& c; ~DecGuard() { c.fetch_sub(1); } } dec_guard{n_decoding_};
            std::vector<int> sprompt;
            for (int id : t.prompt) if (id >= 0) sprompt.push_back(id);
            Sampler sampler(t.sp, t.images.empty() ? t.prompt : sprompt);
            const bool constrained = !t.grammar.empty();
            std::shared_ptr<grammar::GrammarLogitMask> gmask;
            if (constrained) {
                grammar::Triggers trig;
                trig.words = t.trigger_words;
                gmask = std::make_shared<grammar::GrammarLogitMask>(grammar::GrammarState(get_matcher(t.grammar), trig));
                sampler.set_mask(gmask);
            }
            const bool grammar_active_init = gmask && !gmask->state().awaiting_trigger();
            const bool mtp_gram = o.mtp && o.mtp_grammar && gmask && eng->has_constrained_mtp();
            bool argmax_ok = t.sp.engine_argmax_ok() && !grammar_active_init;  // fetch()/finish() fast argmax (never bypasses active grammar)
            bool chunk_greedy_ok = t.sp.engine_argmax_ok() && (!grammar_active_init || mtp_gram);
            bool spec_ok = !t.sp.engine_argmax_ok() && t.sp.speculative_ok() && eng->has_speculative_decode() && (!grammar_active_init || mtp_gram)
                           // A lazy grammar that fires inside a speculative round needs a rollback; without a fast state save that is a
                           // full re-prefill (4 s at 8K context), so such requests stay on the plain path unless --mtp-grammar handles it.
                           && (!gmask || mtp_gram || eng->has_state_save());
            sampler.want_post_ = t.post_probs && t.n_probs > 0;
            Tokenizer::Decoder dec(tok);
            std::string pending;      // decoded text not yet emitted (held back for stop-string / utf-8)
            std::vector<float> lg, raw;
            std::vector<int> lookahead;
            size_t la = 0;
            size_t la_base_pos = 0;  // cached_.size() right before the current lookahead chunk (position of `first`)
            bool la_saved = false;   // whether eng->state_save() was taken at la_base_pos
            const bool want_probs = t.n_probs > 0;
            bool use_chunk = o.mtp && (chunk_greedy_ok || spec_ok) && !want_probs && !t.reasoning_end;
            std::vector<int> forced;
            bool drain_la = false;           // lazy grammar fired inside a greedy MTP lookahead whose remaining tokens the grammar accepts (see below)
            const bool spec_was_ok = spec_ok;
            int reason_toks = 0;
            bool budget_hit = false;
            auto in_reasoning = [&]() -> bool {  // is the model currently inside its thinking block (prompt-opened or self-opened, not yet closed)?
                const std::string seen = r.content + pending;
                if (t.tool_fmt == ToolFormat::Inkling) {
                    const size_t a = seen.rfind("<|content_thinking|>"), b = seen.rfind("<|end_message|>");
                    if (a == std::string::npos) return t.thinking_open && b == std::string::npos;
                    return b == std::string::npos || b < a;
                }
                return (t.thinking_open || seen.find(t.think_start) != std::string::npos) && seen.find(t.think_end) == std::string::npos;
            };
            if (std::getenv("QV_MTP_DEBUG"))
                std::fprintf(stderr, "[qv_server] use_chunk=%d (mtp=%d argmax_ok=%d spec_ok=%d mtp_gram=%d constrained=%d lazy=%d probs=%d)\n",
                             (int)use_chunk, (int)o.mtp, (int)argmax_ok, (int)spec_ok, (int)mtp_gram, (int)constrained,
                             (int)(gmask && gmask->state().awaiting_trigger()), (int)want_probs);
            Chunk ck;
            bool hit_stop_word = false;
            const int budget = t.n_predict;  // <0: unlimited
            auto td0 = now();
            auto stamp = [&](Chunk& k) { k.n_cache = r.n_cache; k.n_prompt_eval = r.n_prompt_eval; k.prompt_ms = r.prompt_ms; k.predicted_ms = ms(td0, now()); };
            // engine part of picking the next token (caller holds the engine turn): argmax id, or -1 with the logits copied to `lg`
            auto fetch = [&]() -> int {
                if (argmax_ok && !want_probs) return eng->argmax();
                eng->logits(lg);
                return -1;
            };
            // host part (sampling, logprobs): runs outside the engine turn so other slots can use the GPU meanwhile
            auto finish = [&](int a, TokProb& tp, bool& have_tp) -> int {
                have_tp = false;
                if (a >= 0) return a;
                if (want_probs) raw = lg;
                int tkn = argmax_ok ? (int)(std::max_element(lg.begin(), lg.end()) - lg.begin()) : sampler.sample(lg);
                if (want_probs) {
                    have_tp = true;
                    tp.id = tkn;
                    tp.text = piece_bytes(tkn, tp.bytes);
                    if (t.post_probs) {
                        tp.prob = sampler.last_prob(tkn);
                        int k = 0;
                        for (auto& [id, p] : sampler.last_candidates()) { if (k++ >= t.n_probs) break; std::vector<int> by; std::string tx = piece_bytes(id, by); tp.top.emplace_back(id, tx, by, p); }
                    } else {
                        auto info = Sampler::logprob(raw, tkn, t.n_probs);
                        tp.logprob = info.logprob;
                        for (auto& [id, v] : info.top) { std::vector<int> by; std::string tx = piece_bytes(id, by); tp.top.emplace_back(id, tx, by, v); }
                    }
                }
                return tkn;
            };
            TokProb tp;
            bool have_tp = false;
            int tk;
            { int a; { EngOp op(*this, sid); a = fetch(); } tk = finish(a, tp, have_tp); }
            for (int n = 0;;) {
                ++n;
                r.n_decoded = n;
                { std::lock_guard<std::mutex> lk(mu_); slots_[sid].n_decoded = n; slots_[sid].n_remain = budget < 0 ? -1 : budget - n; }
                busy_sum += n_busy_.load();
                r.tokens.push_back(tk);
                const bool is_eos = std::find(stop_ids.begin(), stop_ids.end(), tk) != stop_ids.end();
                if (is_eos) { r.stop_type = "eos"; r.tokens_cached = (int)cached_.size(); break; }
                const bool was_awaiting = gmask && gmask->state().awaiting_trigger();
                sampler.accept(tk);
                const bool trigger_fired = was_awaiting && !gmask->state().awaiting_trigger();
                std::string piece = dec.push(tk);
                if (!o.special && t.tool_fmt != ToolFormat::Inkling && tok.is_added(tk)) { std::string b = tok.token_bytes(tk); if (b.size() > 4 && b.rfind("<|", 0) == 0 && b.compare(b.size() - 2, 2, "|>") == 0) piece.clear(); }
                pending += piece;
                // stop strings
                size_t hit = std::string::npos;
                std::string hit_word;
                for (auto& s : t.stops) { size_t p = pending.find(s); if (p != std::string::npos && p < hit) { hit = p; hit_word = s; } }
                // (stop strings may also span previously emitted text: llama checks the accumulated text; we hold back partial matches so no span exists)
                ck.tokens.push_back(tk);
                if (have_tp) { ck.probs.push_back(tp); r.probs.push_back(tp); }
                ck.n_decoded = n;
                if (hit != std::string::npos) {
                    pending.resize(hit);
                    hit_stop_word = true;
                    r.stop_type = "word";
                    r.stopping_word = hit_word;
                    ck.text = pending;
                    r.content += pending;
                    pending.clear();
                    stamp(ck);
                    if (!on_chunk(ck)) r.aborted = true;
                    ck = Chunk();
                    break;
                }
                size_t keep = 0;
                for (auto& s : t.stops)
                    for (size_t k = std::min(pending.size(), s.size() - 1); k > keep; --k)
                        if (pending.compare(pending.size() - k, k, s, 0, k) == 0) { keep = k; break; }
                std::string out = pending.substr(0, pending.size() - keep);
                pending.erase(0, pending.size() - keep);
                if (!out.empty() || have_tp) {
                    ck.text = out;
                    r.content += out;
                    stamp(ck);
                    if (!on_chunk(ck)) { r.aborted = true; break; }
                    ck = Chunk();
                }
                if (budget >= 0 && n >= budget) { r.stop_type = "limit"; r.tokens_cached = (int)cached_.size(); break; }
                if ((int)cached_.size() + 1 >= n_ctx) { r.stop_type = "limit"; r.truncated = true; break; }
                if (!alive()) { r.aborted = true; break; }
                if (trigger_fired) {
                    argmax_ok = false;
                    if (!mtp_gram) {
                        chunk_greedy_ok = false;
                        spec_ok = false;
                        // The lookahead tokens after the trigger were drawn WITHOUT the grammar mask. If the grammar accepts every one of them the
                        // masked draw would have produced the same token (an allowed global argmax is the masked argmax), so no rollback is needed:
                        // drain the lookahead, then continue one token at a time under the mask. Engines without state_save (Inkling) would
                        // otherwise reset and re-prefill the whole context here (a 60K-token prompt = ~43 s for a 60-token tool call).
                        bool drain = false;
                        if (la < lookahead.size() && t.sp.greedy() && !spec_was_ok && !std::getenv("QV_NO_DRAIN")) {
                            grammar::GrammarState probe = gmask->state().fork();
                            drain = true;
                            for (size_t j = la; j < lookahead.size() && drain; ++j) drain = probe.accept_token(lookahead[j]);
                        }
                        if (drain) {
                            drain_la = true;
                            if (std::getenv("QV_MTP_DEBUG"))
                                std::fprintf(stderr, "[qv_server] lazy grammar triggered at gen=%d (lookahead of %zu tokens grammar-valid, no rollback)\n", n, lookahead.size() - la);
                        } else
                        use_chunk = false;
                        if (!drain && la < lookahead.size()) {
                            // `tk` is lookahead[la - 1], already present in cached_ at index (la_base_pos + la).
                            // Roll engine back to right after `tk` (cached_[0 .. keep_len)) and compute masked next token.
                            const size_t keep_len = la_base_pos + la + 1;
                            const size_t dropped = cached_.size() - keep_len;
                            if (std::getenv("QV_MTP_DEBUG"))
                                std::fprintf(stderr, "[qv_server] lazy grammar triggered at gen=%d (rollback=%zu tokens, fast=%d)\n",
                                             n, dropped, (int)la_saved);
                            lookahead.clear();
                            la = 0;
                            int fa = -1;
                            if (la_saved) {
                                cached_.resize(keep_len);
                                EngOp op(*this, sid);
                                eng->state_restore();
                                for (size_t p = la_base_pos; p < keep_len; ++p) eng->step(cached_[p], (int)p);
                                fa = fetch();
                            } else {
                                std::vector<int> replay(cached_.begin(), cached_.begin() + keep_len);
                                { EngOp op(*this, sid); eng->reset(); }
                                cached_.clear();
                                for (size_t k = 0; k < t.images.size(); ++k) {
                                    if (!img_emb[k].empty()) continue;
                                    if (!alive()) { r.aborted = true; break; }
                                    img_emb[k].resize((size_t)t.images[k].n * venc->cfg().text_hidden);
                                    std::lock_guard<std::mutex> vlk(venc_mu_);
                                    venc->encode(*t.images[k].px, img_emb[k].data());
                                }
                                if (r.aborted) break;
                                for (size_t off = 0; off < keep_len;) {
                                    if (!alive()) { r.aborted = true; break; }
                                    const int ch = (multi_seq && n_busy_ > 1) ? std::min(chunk, std::max(1, shared_chunk)) : chunk;
                                    const size_t lim = (!t.images.empty() && off < (size_t)N) ? (size_t)N : keep_len;
                                    const int cnt = (int)std::min<size_t>(ch, lim - off);
                                    { EngOp op(*this, sid);
                                      if (!t.images.empty() && off < (size_t)N) prefill_with_images(t, (int)off, cnt, img_emb);
                                      else eng->prefill(replay.data() + off, cnt, (int)off); }
                                    cached_.insert(cached_.end(), replay.begin() + off, replay.begin() + off + cnt);
                                    off += cnt;
                                }
                                if (r.aborted) break;
                                { EngOp op(*this, sid); fa = fetch(); }
                            }
                            tk = finish(fa, tp, have_tp);
                            continue;
                        }
                        // `tk` was either the very first token or the last token of lookahead: not yet fed into `eng`/`cached_`.
                        if (!drain) {
                            if (std::getenv("QV_MTP_DEBUG"))
                                std::fprintf(stderr, "[qv_server] lazy grammar triggered at gen=%d (no rollback)\n", n);
                            lookahead.clear();
                            la = 0;
                        }
                    } else if (std::getenv("QV_MTP_DEBUG")) {
                        std::fprintf(stderr, "[qv_server] lazy grammar triggered inside mtp_gram at gen=%d (continuing MTP)\n", n);
                    }
                }
                // ---- next token
                if (use_chunk) {
                    if (la < lookahead.size()) { tk = lookahead[la++]; continue; }
                    if (drain_la) { drain_la = false; use_chunk = false; lookahead.clear(); la = 0; }  // lookahead consumed: tk is its last token, not yet fed
                }
                if (use_chunk) {
                    int room = std::min(mtp_chunk, n_ctx - (int)cached_.size());
                    if (budget >= 0) room = std::min(room, budget - n);
                    if (room >= 2) {
                        la_base_pos = cached_.size();
                        la_saved = gmask != nullptr && !mtp_gram && !multi_seq && eng->has_state_save();
                        std::unique_ptr<grammar::GrammarTokenConstraint> gtc;
                        if (mtp_gram) gtc = std::make_unique<grammar::GrammarTokenConstraint>(gmask->state().fork(), stop_ids);
                        if (chunk_greedy_ok) {
                            EngOp op(*this, sid);
                            if (la_saved) eng->state_save();
                            if (gtc) eng->decode_greedy_constrained(tk, (int)la_base_pos, room, lookahead, gtc.get());
                            else eng->decode_greedy(tk, (int)la_base_pos, room, lookahead, false);
                        } else {
                            auto has_sampler = [&](const char* name) {
                                return std::find(t.sp.samplers.begin(), t.sp.samplers.end(), name) != t.sp.samplers.end();
                            };
                            SpecSamplingParams ssp;
                            ssp.temperature = t.sp.temperature;
                            ssp.top_k = has_sampler("top_k") ? t.sp.top_k : 0;
                            ssp.top_p = has_sampler("top_p") ? t.sp.top_p : 1.0f;
                            ssp.min_p = has_sampler("min_p") ? t.sp.min_p : 0.0f;
                            ssp.min_keep = t.sp.min_keep;
                            ssp.seed = sampler.next_seed();
                            EngOp op(*this, sid);
                            if (la_saved) eng->state_save();
                            if (gtc) eng->decode_speculative_constrained(tk, (int)la_base_pos, room, ssp, lookahead, gtc.get());
                            else eng->decode_speculative(tk, (int)la_base_pos, room, ssp, lookahead);
                        }
                        if (!lookahead.empty()) {
                            cached_.push_back(tk);
                            for (size_t i = 0; i + 1 < lookahead.size(); ++i) cached_.push_back(lookahead[i]);
                            la = 0;
                            tk = lookahead[la++];
                            continue;
                        }
                    }
                }
                int fa;
                {  // one engine turn per token: step + fetch the result (argmax id / logits) of this sequence
                    EngOp op(*this, sid);
                    eng->step(tk, (int)cached_.size());
                    fa = fetch();
                }
                cached_.push_back(tk);
                // --reasoning-budget N: after N tokens inside the thinking block close it (optional message first), like llama-server's budget sampler
                if (forced.empty() && !budget_hit && t.reasoning_budget > 0 && in_reasoning()) {
                    if (++reason_toks >= t.reasoning_budget) {
                        budget_hit = true;
                        const bool ink = t.tool_fmt == ToolFormat::Inkling;
                        const std::string tail = ink ? "<|end_message|><|message_model|><|content_text|>" : "\n" + t.think_end + "\n\n";
                        auto ids = tok.encode((t.reasoning_budget_msg.empty() ? std::string() : (ink ? t.reasoning_budget_msg : "\n" + t.reasoning_budget_msg)) + tail, true);
                        forced.assign(ids.begin(), ids.end());
                    }
                }
                if (forced.empty() && t.reasoning_end && t.reasoning_end->exchange(false)) {
                    // "skip reasoning": close the thinking block by feeding the end marker as if the model had written it
                    const std::string seen = r.content + pending;
                    const bool inside = (t.thinking_open || seen.find(t.think_start) != std::string::npos) && seen.find(t.think_end) == std::string::npos;
                    if (inside) { auto ids = tok.encode("\n" + t.think_end + "\n\n", true); forced.assign(ids.begin(), ids.end()); }
                }
                if (!forced.empty()) { tk = forced.front(); forced.erase(forced.begin()); have_tp = false; }
                else tk = finish(fa, tp, have_tp);
            }
            if (!r.aborted && !hit_stop_word) {  // flush held-back text
                pending += dec.flush();
                if (!pending.empty() || !ck.tokens.empty()) {
                    ck.text = pending;
                    r.content += pending;
                    stamp(ck);
                    on_chunk(ck);
                }
            }
            r.predicted_ms = ms(td0, now());
            if (std::getenv("QV_MTP_DEBUG")) { qv::MtpStats ms_; eng->mtp_stats(ms_); std::fprintf(stderr, "[qv_server] mtp stats: cycles=%ld accepted=%ld plain_steps=%ld\n", ms_.cycles, ms_.accepted, ms_.plain_steps); }
            if (r.tokens_cached == 0) r.tokens_cached = (int)cached_.size();
        }
    } catch (const ApiError&) {
        throw;
    } catch (const std::exception& e) {
        { EngOp op(*this, sid); eng->reset(); }
        cached_.clear();
        ckpts_of(sid).clear();
        slot_cacheable_[seq_of(sid)] = 0;
        std::fprintf(stderr, "[qv_server] engine error: %s\n", e.what());
        throw_error(500, std::string("engine error: ") + e.what());
    }
done:
    r.has_new_line = r.content.find('\n') != std::string::npos;
    {
        std::lock_guard<std::mutex> lk(mu_);
        m_.prompt_tokens_total += r.n_prompt_eval;
        m_.prompt_cached_total += r.n_cache;
        m_.prompt_seconds_total += r.prompt_ms / 1e3;
        m_.predicted_total += r.n_decoded;
        m_.predicted_seconds_total += r.predicted_ms / 1e3;
        m_.n_decode_total += r.n_decoded;
        m_.n_tokens_max = std::max<long long>(m_.n_tokens_max, (long long)cached_.size());
        m_.b_prompt_tokens += r.n_prompt_eval; m_.b_prompt_s += r.prompt_ms / 1e3; m_.b_pred_tokens += r.n_decoded; m_.b_pred_s += r.predicted_ms / 1e3;
        m_.b_busy_sum += busy_sum; m_.b_decode += r.n_decoded;
    }
    if (o.verbosity >= 3 && !o.log_disable)
        std::fprintf(stderr, "[qv_server] slot %d task done: prompt=%d cached=%d gen=%d stop=%s%s prompt=%.0fms gen=%.0fms (%.2f tok/s)\n", sid, N, r.n_cache, r.n_decoded,
                     r.stop_type.c_str(), r.aborted ? " (client aborted)" : "", r.prompt_ms, r.predicted_ms, r.predicted_ms > 0 ? 1e3 * r.n_decoded / r.predicted_ms : 0.0);
    return r;
}

}  // namespace qv::server
