// Routing, middleware and the non-generation endpoints (health, props, slots, metrics, models, tokenize, ...).
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <type_traits>

#ifdef QV_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "server/app.h"
#include "server/mock_engine.h"

namespace qv::server {

namespace {
const char* kJson = "application/json; charset=utf-8";

std::string read_all(const std::string& p, bool& ok) {
    std::ifstream f(p, std::ios::binary);
    ok = (bool)f;
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::string mime_of(const std::string& p) {
    auto ends = [&](const char* e) { size_t n = std::strlen(e); return p.size() >= n && p.compare(p.size() - n, n, e) == 0; };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".js") || ends(".mjs")) return "text/javascript; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".json")) return "application/json; charset=utf-8";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".png")) return "image/png";
    if (ends(".ico")) return "image/x-icon";
    if (ends(".txt")) return "text/plain; charset=utf-8";
    return "application/octet-stream";
}
}  // namespace

void App::init_slots() {
    std::lock_guard<std::mutex> lk(mu_);
    n_slots = o.n_parallel < 0 ? 4 : std::max(1, o.n_parallel);
    slots_.assign(n_slots, SlotInfo());
    for (int i = 0; i < n_slots; ++i) slots_[i].id = i;
    owner_slot_ = 0;
    set_sequences(false);
    if (m_.t_start == 0) m_.t_start = (long long)std::time(nullptr);
}

// Slot <-> sequence layout: with a multi-sequence engine every slot is an engine sequence with its own `--ctx-size / n_slots` tokens of context
// and requests run concurrently; otherwise the slots are logical (one sequence of `--ctx-size` tokens, requests run one at a time).
void App::set_sequences(bool multi) {
    multi_seq = multi && n_slots > 1;
    slot_ctx = multi_seq ? std::max(1, o.ctx / n_slots) : o.ctx;
    n_ctx = slot_ctx;
    conc_limit_ = multi_seq ? n_slots : 1;
    slot_busy_.assign(n_slots, 0);
    slot_used_.assign(n_slots, 0);
    slot_cached_.assign(multi_seq ? n_slots : 1, std::vector<int>());
    slot_ckpts_.assign(multi_seq ? n_slots : 1, {});
    slot_cacheable_.assign(multi_seq ? n_slots : 1, 0);
    parked_pool_.clear();
}

Value App::model_json() const {
    Value m = Value::object();
    m.set("id", model_id);
    Value al = Value::array();
    for (auto& a : o.alias.empty() ? std::vector<std::string>{model_id} : o.alias) al.push(a);
    m.set("aliases", std::move(al));
    m.set("tags", Value::array());
    m.set("object", "model");
    Value arch = Value::object();
    Value in = Value::array(), out = Value::array();
    in.push("text"); if (venc) in.push("image"); out.push("text");
    arch.set("input_modalities", std::move(in));
    arch.set("output_modalities", std::move(out));
    m.set("architecture", std::move(arch));
    m.set("created", started_at);
    m.set("owned_by", "q-volta");
    Value meta = Value::object();
    meta.set("vocab_type", 2);
    meta.set("n_vocab", tok.vocab_size());
    meta.set("n_ctx", n_ctx);
    long long n_ctx_train = model_arch == "mock" ? n_ctx : 262144;
    Value qvm = Value::object();
    if (!model_meta_json.empty()) {
        try {
            qvm = jd::parse(model_meta_json);
            if (const Value* cl = qvm.find("context_length")) if (cl->is_number() && cl->as_int() > 0) n_ctx_train = cl->as_int();
        } catch (...) {}
    }
    meta.set("n_ctx_train", n_ctx_train);
    meta.set("n_embd", cached_n_embd);
    meta.set("n_params", 0);
    meta.set("size", cached_weight_bytes);
    meta.set("ftype", model_arch == "mock" ? "mock" : "Q-Volta");
    m.set("meta", std::move(meta));
    if (!embedded_blobs.empty() || !model_meta_json.empty()) {  // Q-Volta extension (only for self-contained files, so plain models keep the llama-server schema)
        Value q = Value::object();
        q.set("meta", std::move(qvm));
        Value bl = Value::array();
        for (auto& n : embedded_blobs) bl.push(n);
        q.set("blobs", std::move(bl));
        q.set("tokenizer", tokenizer_path.rfind("embedded:", 0) == 0 ? "embedded" : "external");
        q.set("chat_template", template_source);
        m.set("qv", std::move(q));
    }
    return m;
}

void App::handle_models(http::Conn& c) {
    if (has_presets()) {
        std::lock_guard<std::mutex> slk(state_mu_);
        Value models = Value::array();
        Value data = Value::array();
        for (const auto& p : presets) {
            const bool loaded = (p.id == active_preset_id && eng != nullptr);
            Value m;
            if (loaded) {
                m = model_json();
                m.set("id", p.id);
                Value al = Value::array();
                al.push(p.id);
                for (auto& a : p.opts.alias) if (a != p.id) al.push(a);
                m.set("aliases", std::move(al));
            } else {
                m = Value::object();
                m.set("id", p.id);
                Value al = Value::array();
                al.push(p.id);
                for (auto& a : p.opts.alias) if (a != p.id) al.push(a);
                m.set("aliases", std::move(al));
                m.set("tags", Value::array());
                m.set("object", "model");
                m.set("created", started_at);
                m.set("owned_by", "q-volta");
                Value meta = Value::object();
                meta.set("n_ctx", p.opts.ctx > 0 ? p.opts.ctx : 4096);
                m.set("meta", std::move(meta));
            }
            Value st = Value::object();
            st.set("value", loaded ? "loaded" : "unloaded");
            m.set("status", std::move(st));
            data.push(std::move(m));

            Value e = Value::object();
            e.set("name", p.id);
            e.set("model", p.id);
            e.set("type", "model");
            Value det = Value::object();
            det.set("format", "qv");
            e.set("details", std::move(det));
            models.push(std::move(e));
        }
        Value r = Value::object();
        r.set("models", std::move(models));
        r.set("object", "list");
        r.set("data", std::move(data));
        respond_json(c, 200, r);
        return;
    }
    Value m = model_json();
    Value models = Value::array();
    Value e = Value::object();
    e.set("name", model_id);
    e.set("model", model_id);
    e.set("modified_at", "");
    e.set("size", "");
    e.set("digest", "");
    e.set("type", "model");
    e.set("description", "");
    Value tags = Value::array();
    tags.push("");
    e.set("tags", std::move(tags));
    Value caps = Value::array();
    caps.push("completion");
    e.set("capabilities", std::move(caps));
    e.set("parameters", "");
    Value det = Value::object();
    det.set("parent_model", "");
    det.set("format", "qv");
    det.set("family", "");
    Value fam = Value::array();
    fam.push("");
    det.set("families", std::move(fam));
    det.set("parameter_size", "");
    det.set("quantization_level", "");
    e.set("details", std::move(det));
    models.push(std::move(e));
    Value r = Value::object();
    r.set("models", std::move(models));
    r.set("object", "list");
    Value data = Value::array();
    data.push(std::move(m));
    r.set("data", std::move(data));
    respond_json(c, 200, r);
}

void App::handle_model_load(const http::Request& req, http::Conn& c, bool load) {
    if (!has_presets()) throw_error(400, "router mode is not active (start with --models-preset)");
    Value body = Value::object();
    if (!req.body.empty()) {
        try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
        if (!body.is_object()) throw_error(400, "request body must be a JSON object");
    }
    std::string mname;
    if (const Value* mv = body.find("model"); mv && mv->is_string()) mname = mv->as_string();
    auto alive = [&] { return !c.peer_closed(); };
    if (load) {
        if (mname.empty()) throw_error(400, "missing 'model' field");
        long ticket = acquire_model_lease(mname, alive);
        if (ticket == -2) { c.close_after = true; return; }
        struct Rel { App& a; long t; ~Rel() { if (t >= 0) a.release_model_lease(t); } } rel{*this, ticket};
        Value r = Value::object();
        r.set("success", true);
        r.set("model", active_preset_id);
        respond_json(c, 200, r);
    } else {
        long ticket = lease_acquire(alive);
        if (ticket < 0) { c.close_after = true; return; }
        struct Rel { App& a; long t; ~Rel() { a.lease_release(t); } } rel{*this, ticket};
        if (!active_preset_id.empty() && (mname.empty() || mname == active_preset_id || find_preset(mname) == find_preset(active_preset_id))) unload_model();
        Value r = Value::object();
        r.set("success", true);
        respond_json(c, 200, r);
    }
}

void App::handle_props(http::Conn& c) {
    std::lock_guard<std::mutex> slk(state_mu_);
    Value r = Value::object();
    Task dt;
    parse_sampling(Value::object(), dt);
    dt.n_predict = o.n_predict;
    dt.reasoning_format = o.reasoning_format == "auto" ? "deepseek" : o.reasoning_format;
    Value dg = Value::object();
    dg.set("params", gen_settings(dt, 1));
    dg.set("n_ctx", n_ctx);
    r.set("default_generation_settings", std::move(dg));
    r.set("total_slots", n_slots);
    r.set("model_alias", model_id);
    r.set("model_ftype", model_arch == "mock" ? "mock" : "Q-Volta");
    r.set("model_path", o.model);
    Value mod = Value::object();
    mod.set("vision", venc != nullptr);
    mod.set("video", false);
    mod.set("audio", false);
    r.set("modalities", std::move(mod));
    r.set("media_marker", "<__media_qv__>");
    r.set("endpoint_slots", o.slots);
    r.set("endpoint_props", o.props);
    r.set("endpoint_metrics", o.metrics);
    r.set("ui", o.webui);
    {
        Value us = Value::object();
        if (!o.ui_config.empty()) { try { us = jd::parse(o.ui_config); } catch (...) {} }
        r.set("ui_settings", std::move(us));
    }
    r.set("chat_template", tmpl ? tmpl->source() : std::string());
    if (!embedded_blobs.empty() || !model_meta_json.empty()) {  // Q-Volta extension: only for self-contained .qv files (docs/QV_FORMAT.md)
        r.set("qv_chat_template_source", template_source);
        r.set("qv_tokenizer", tokenizer_path.rfind("embedded:", 0) == 0 ? "embedded" : tokenizer_path);
        Value a = Value::array();
        for (auto& n : embedded_blobs) a.push(n);
        r.set("qv_embedded_blobs", std::move(a));
        r.set("qv_embedded_defaults", embedded_applied);
    }
    r.set("chat_template_caps", tmpl ? tmpl->caps() : Value::object());
    r.set("bos_token", bos_id >= 0 ? tok.token_bytes(bos_id) : std::string());
    r.set("eos_token", eos_id >= 0 ? tok.token_bytes(eos_id) : std::string());
    r.set("build_info", build_info);
    r.set("is_sleeping", sleeping.load());
    r.set("cors_proxy_enabled", false);
    respond_json(c, 200, r);
}

void App::handle_slots(const http::Request& req, http::Conn& c) {
    if (!o.slots) throw_error(501, "This server does not support slots endpoint. Start it with `--slots`", "not_supported_error");
    Value arr = Value::array();
    int idle = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& s : slots_) {
            Value e = Value::object();
            e.set("id", s.id);
            e.set("n_ctx", n_ctx);
            e.set("speculative", false);
            e.set("is_processing", s.processing);
            if (!s.processing) ++idle;
            if (s.has_task) {
                e.set("id_task", (long long)s.id_task);
                e.set("n_prompt_tokens", s.n_prompt);
                e.set("n_prompt_tokens_processed", s.n_prompt_processed);
                e.set("n_prompt_tokens_cache", s.n_prompt_cache);
                e.set("params", s.params);
                Value nt = Value::object();
                nt.set("has_next_token", s.has_next_token);
                nt.set("has_new_line", s.has_new_line);
                nt.set("n_remain", s.n_remain);
                nt.set("n_decoded", s.n_decoded);
                Value na = Value::array();
                na.push(std::move(nt));
                e.set("next_token", std::move(na));
            }
            arr.push(std::move(e));
        }
    }
    if (!req.param("fail_on_no_slot").empty() && idle == 0) throw_error(503, "no slot available", "unavailable_error");
    respond_json(c, 200, arr);
}

std::string App::slot_path(const std::string& f) const { return o.slot_save_path + (o.slot_save_path.empty() || o.slot_save_path.back() == '/' ? "" : "/") + f; }

void App::handle_slot_action(const http::Request& req, http::Conn& c, int id) {
    if (o.slot_save_path.empty()) throw_error(501, "This server does not support slots action. Start it with `--slot-save-path`", "not_supported_error");
    if (id < 0 || id >= n_slots) throw_error(400, "Invalid slot ID");
    std::string action = req.param("action");
    if (action != "save" && action != "restore" && action != "erase") throw_error(400, "Invalid action");
    auto alive = [&] { return !c.peer_closed(); };
    Value body = Value::object();
    if (action != "erase") {
        try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
        const Value* f = body.is_object() ? body.find("filename") : nullptr;
        if (!f || !f->is_string()) throw_error(400, "[json.exception.out_of_range.403] key 'filename' not found");
        const std::string& fn = f->as_string();
        if (fn.empty() || fn.find('/') != std::string::npos || fn.find('\\') != std::string::npos || fn.find("..") != std::string::npos) throw_error(400, "Invalid filename");
    }
    Task st;  // slot action = a (non-generating) task pinned to this slot: waits until the slot is idle
    st.id_slot = id;
    const int sid = slot_acquire(st, false, alive);
    if (sid < 0) return;
    struct Rel { App& a; int s; ~Rel() { a.slot_release(s); } } rel{*this, sid};
    auto t0 = Clock::now();
    auto ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
    Value r = Value::object();
    r.set("id_slot", id);
    ensure_engine();
    std::vector<int>& cached_ = cached_of(id);
    struct EngTurn { std::unique_lock<FairMutex> lk; EngTurn(App& a, int sl) : lk(a.eng_mu_) { a.eng->select_seq(a.seq_of(sl)); } };
    if (action == "erase") {
        long n = (long)cached_.size();
        { EngTurn tn(*this, id); eng->reset(); }
        cached_.clear();
        ckpts_of(id).clear();
        slot_cacheable_[seq_of(id)] = 0;
        if (!multi_seq) parked_pool_.clear();
        r.set("n_erased", n);
    } else if (action == "save") {
        for (int c : cached_) if (c < 0) throw_error(400, "Unable to save slot: it holds image tokens (images are not serialised)");
        const std::string fn = body.find("filename")->as_string();
        FILE* f = std::fopen(slot_path(fn).c_str(), "wb");
        if (!f) throw_error(500, "Unable to save slot", "server_error");
        // Q-Volta slot file: the token sequence of the slot. The recurrent GatedDeltaNet state is not serialised; restore re-evaluates the tokens.
        const char magic[8] = {'Q', 'V', 'S', 'L', 'O', 'T', '1', '\n'};
        int32_t n = (int32_t)cached_.size();
        std::fwrite(magic, 1, 8, f);
        std::fwrite(&n, 4, 1, f);
        std::fwrite(cached_.data(), 4, cached_.size(), f);
        long bytes = std::ftell(f);
        std::fclose(f);
        r.set("filename", fn);
        r.set("n_saved", (long)cached_.size());
        r.set("n_written", bytes);
        Value tm = Value::object();
        tm.set("save_ms", ms());
        r.set("timings", std::move(tm));
    } else {
        const std::string fn = body.find("filename")->as_string();
        FILE* f = std::fopen(slot_path(fn).c_str(), "rb");
        if (!f) throw_error(400, "Unable to restore slot, no available space in KV cache or invalid slot save file");
        char magic[8];
        int32_t n = 0;
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "QVSLOT1\n", 8) != 0 || std::fread(&n, 4, 1, f) != 1 || n < 0 || n >= n_ctx) { std::fclose(f); throw_error(400, "Unable to restore slot, no available space in KV cache or invalid slot save file"); }
        std::vector<int> toks(n);
        if (n && std::fread(toks.data(), 4, n, f) != (size_t)n) { std::fclose(f); throw_error(400, "Unable to restore slot, no available space in KV cache or invalid slot save file"); }
        std::fclose(f);
        { EngTurn tn(*this, id); eng->reset(); }
        cached_.clear();
        ckpts_of(id).clear();
        for (size_t off = 0; off < toks.size(); off += chunk) {
            int k = (int)std::min<size_t>(chunk, toks.size() - off);
            { EngTurn tn(*this, id); eng->prefill(toks.data() + off, k, (int)off); }
            cached_.insert(cached_.end(), toks.begin() + off, toks.begin() + off + k);
        }
        slot_cacheable_[seq_of(id)] = 1;
        r.set("filename", fn);
        r.set("n_restored", n);
        r.set("n_read", 12 + 4 * (long)n);
        Value tm = Value::object();
        tm.set("restore_ms", ms());
        r.set("timings", std::move(tm));
    }
    respond_json(c, 200, r);
}

void App::handle_metrics(http::Conn& c) {
    if (!o.metrics) throw_error(501, "This server does not support metrics endpoint. Start it with `--metrics`", "not_supported_error");
    std::ostringstream os;
    Metrics m;
    int processing = 0, deferred;
    {
        std::lock_guard<std::mutex> lk(mu_);
        m = m_;
        for (auto& s : slots_) processing += s.processing;
        deferred = 0;
        for (auto& w : waiters_) deferred += (!w.granted && w.kind == 0);
        m_.b_prompt_tokens = m_.b_prompt_s = m_.b_pred_tokens = m_.b_pred_s = m_.b_busy_sum = m_.b_decode = 0;  // gauges: window between scrapes
    }
    auto metric = [&](const char* type, const char* name, const char* help, double v) {
        os << "# HELP llamacpp:" << name << " " << help << "\n# TYPE llamacpp:" << name << " " << type << "\nllamacpp:" << name << " ";
        char b[64];
        std::snprintf(b, sizeof b, "%.10g", v);
        os << b << "\n";
    };
    metric("counter", "prompt_tokens_total", "Number of prompt tokens processed, excluding cached tokens", m.prompt_tokens_total);
    metric("counter", "prompt_tokens_cached_total", "Number of prompt tokens reused from the cache", m.prompt_cached_total);
    metric("counter", "prompt_seconds_total", "Total time spent processing prompts", m.prompt_seconds_total);
    metric("counter", "tokens_predicted_total", "Number of generation tokens processed", m.predicted_total);
    metric("counter", "tokens_predicted_seconds_total", "Total time spent generating tokens", m.predicted_seconds_total);
    metric("counter", "n_decode_total", "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding", (double)m.n_decode_total);
    metric("counter", "n_tokens_max", "Largest observed sequence length (prompt + generation)", (double)m.n_tokens_max);
    metric("counter", "spec_decode_num_draft_tokens_total", "Speculative: Total draft tokens generated", 0);
    metric("counter", "spec_decode_num_accepted_tokens_total", "Speculative: Total draft tokens accepted by the target model", 0);
    metric("counter", "spec_decode_num_drafts_total", "Speculative: Total speculative decoding verification steps", 0);
    metric("gauge", "prompt_tokens_seconds", "Average prompt throughput in tokens/s", m.b_prompt_s > 0 ? m.b_prompt_tokens / m.b_prompt_s : 0);
    metric("gauge", "predicted_tokens_seconds", "Average generation throughput in tokens/s", m.b_pred_s > 0 ? m.b_pred_tokens / m.b_pred_s : 0);
    metric("gauge", "requests_processing", "Number of requests processing", processing);
    metric("gauge", "requests_deferred", "Number of requests deferred", deferred);
    metric("gauge", "n_busy_slots_per_decode", "Average number of busy slots per llama_decode() call", m.b_decode > 0 ? m.b_busy_sum / m.b_decode : 0);
    c.respond(200, "text/plain; version=0.0.4", os.str(), {{"Process-Start-Time-Unix", std::to_string(m.t_start)}});
}

void App::handle_tokenize(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    if (!body.is_object()) throw_error(400, "[json.exception.type_error.302] type must be object, but is " + std::string(body.is_array() ? "array" : "string"));
    std::string req_model;
    if (const Value* mv = body.find("model"); mv && mv->is_string()) req_model = mv->as_string();
    auto alive = [&] { return !c.peer_closed(); };
    long ticket = acquire_model_lease(req_model, alive);
    if (ticket == -2) { c.close_after = true; return; }
    struct Rel { App& a; long t; ~Rel() { if (t >= 0) a.release_model_lease(t); } } rel{*this, ticket};
    auto tn = [](const Value& v) { return v.is_string() ? "string" : v.is_number() ? "number" : v.is_bool() ? "boolean" : v.is_array() ? "array" : "object"; };
    std::string content;
    if (const Value* v = body.find("content"); v && !v->is_null()) {
        if (!v->is_string()) throw_error(400, std::string("[json.exception.type_error.302] type must be string, but is ") + tn(*v));
        content = v->as_string();
    }
    auto flag = [&](const char* k, bool d) { const Value* v = body.find(k); return v && v->is_bool() ? v->as_bool() : d; };
    bool with_pieces = flag("with_pieces", false), parse_special = flag("parse_special", true);
    flag("add_special", false);  // Qwen tokenizers add no BOS/EOS: add_special has no effect
    std::vector<int> ids = tok.encode(content, parse_special);
    Value toks = Value::array();
    for (int id : ids) {
        if (!with_pieces) { toks.push(id); continue; }
        Value e = Value::object();
        e.set("id", id);
        std::string b = tok.token_bytes(id);
        bool valid = true;
        {
            Value probe;
            std::string s = piece_text(id);
            valid = s == b;
        }
        if (valid) e.set("piece", b);
        else { Value a = Value::array(); for (unsigned char ch : b) a.push((int)ch); e.set("piece", std::move(a)); }
        toks.push(std::move(e));
    }
    Value r = Value::object();
    r.set("tokens", std::move(toks));
    respond_json(c, 200, r);
}

void App::handle_detokenize(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    if (!body.is_object()) throw_error(400, "[json.exception.type_error.302] type must be object, but is " + std::string(body.is_array() ? "array" : "string"));
    std::string req_model;
    if (const Value* mv = body.find("model"); mv && mv->is_string()) req_model = mv->as_string();
    auto alive = [&] { return !c.peer_closed(); };
    long ticket = acquire_model_lease(req_model, alive);
    if (ticket == -2) { c.close_after = true; return; }
    struct Rel { App& a; long t; ~Rel() { if (t >= 0) a.release_model_lease(t); } } rel{*this, ticket};
    std::vector<int> ids;
    if (const Value* v = body.find("tokens"); v && !v->is_null()) {
        if (!v->is_array()) throw_error(400, std::string("[json.exception.type_error.302] type must be array, but is ") + (v->is_string() ? "string" : v->is_number() ? "number" : v->is_bool() ? "boolean" : "object"));
        for (auto& x : v->items()) {
            if (!x.is_number()) throw_error(400, "[json.exception.type_error.302] type must be number, but is string");
            long long id = x.as_int();
            if (id >= 0 && id < tok.vocab_size()) ids.push_back((int)id);
        }
    }
    Value r = Value::object();
    r.set("content", tok.decode(ids));
    respond_json(c, 200, r);
}

void App::handle_apply_template(const http::Request& req, http::Conn& c) {
    Value body;
    try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
    std::string req_model;
    if (body.is_object()) if (const Value* mv = body.find("model"); mv && mv->is_string()) req_model = mv->as_string();
    auto alive = [&] { return !c.peer_closed(); };
    long ticket = acquire_model_lease(req_model, alive);
    if (ticket == -2) { c.close_after = true; return; }
    struct Rel { App& a; long t; ~Rel() { if (t >= 0) a.release_model_lease(t); } } rel{*this, ticket};
    Task t = parse_task(body, Kind::OaiChat);
    Value r = Value::object();
    r.set("prompt", t.prompt_text);
    respond_json(c, 200, r);
}

void App::route(const http::Request& req, http::Conn& c) {
    // ---- headers common to every response
    std::string origin = req.header("origin");
    c.base_headers = {{"Server", "qv_server"}};
    if (o.cors_credentials && o.cors_origins == "*") c.base_headers.push_back({"Access-Control-Allow-Origin", origin});
    else if (o.cors_origins != "localhost") c.base_headers.push_back({"Access-Control-Allow-Origin", o.cors_origins});
    try {
        if (req.method == "OPTIONS") {
            c.respond(200, "text/html", "", {{"Access-Control-Allow-Credentials", o.cors_credentials ? "true" : "false"}, {"Access-Control-Allow-Methods", o.cors_methods}, {"Access-Control-Allow-Headers", o.cors_headers}});
            return;
        }
        std::string path = req.path;
        if (!o.api_prefix.empty()) {
            if (path.rfind(o.api_prefix, 0) == 0 && (path.size() == o.api_prefix.size() || path[o.api_prefix.size()] == '/')) path = path.substr(o.api_prefix.size());
            else throw_error(404, "File Not Found", "not_found_error");
            if (path.empty()) path = "/";
        }
        // UI assets are public and load while the model is still loading (llama-server: frontend_paths)
        const bool frontend = is_frontend_path(path) || path == "/sw.js" || path == "/service-worker.js";
        // ---- model loading
        if (!ready.load() && !frontend) throw_error(503, "Loading model", "unavailable_error");
        // ---- API key (public: /health, /v1/health, UI assets)
        const bool is_public = path == "/health" || path == "/v1/health" || frontend;
        if (!o.api_keys.empty() && !is_public) {
            std::string k = req.header("authorization");
            if (k.empty()) k = req.header("x-api-key");
            if (k.rfind("Bearer ", 0) == 0) k = k.substr(7);
            if (std::find(o.api_keys.begin(), o.api_keys.end(), k) == o.api_keys.end()) throw_error(401, "Invalid API Key", "authentication_error");
        }
        const std::string& m = req.method;
        const bool get = m == "GET" || m == "HEAD", post = m == "POST";
        if (get && (path == "/health" || path == "/v1/health")) { Value v = Value::object(); v.set("status", "ok"); respond_json(c, 200, v); return; }
        if (get && path == "/metrics") { handle_metrics(c); return; }
        if (get && path == "/props") { handle_props(c); return; }
        if (post && path == "/props") {
            if (!o.props) throw_error(501, "This server does not support changing global properties. Start it with `--props`", "not_supported_error");
            Value v = Value::object(); v.set("success", true); respond_json(c, 200, v); return;
        }
        if (get && (path == "/models" || path == "/v1/models")) { handle_models(c); return; }
        if (post && (path == "/models/load" || path == "/v1/models/load")) { handle_model_load(req, c, true); return; }
        if (post && (path == "/models/unload" || path == "/v1/models/unload")) { handle_model_load(req, c, false); return; }
        if (post && (path == "/completion" || path == "/completions")) { handle_completion(req, c, Kind::Native); return; }
        if (post && path == "/v1/completions") { handle_completion(req, c, Kind::OaiCmpl); return; }
        if (post && (path == "/chat/completions" || path == "/v1/chat/completions")) { handle_completion(req, c, Kind::OaiChat); return; }
        if (post && path == "/infill") { handle_infill(req, c); return; }
        if (post && path == "/tokenize") { handle_tokenize(req, c); return; }
        if (post && path == "/detokenize") { handle_detokenize(req, c); return; }
        if (post && path == "/apply-template") { handle_apply_template(req, c); return; }
        if (post && (path == "/chat/completions/input_tokens" || path == "/v1/chat/completions/input_tokens")) { handle_count_tokens(req, c, ApiFmt::Oai); return; }
        if (post && (path == "/responses/input_tokens" || path == "/v1/responses/input_tokens")) { handle_count_tokens(req, c, ApiFmt::Responses); return; }
        if (post && path == "/v1/chat/completions/control") { handle_control(req, c); return; }
        if (post && path == "/v1/messages/count_tokens") { handle_count_tokens(req, c, ApiFmt::Anthropic); return; }
        if (post && path == "/v1/messages") { handle_messages(req, c); return; }
        if (post && (path == "/v1/responses" || path == "/responses")) { handle_responses(req, c); return; }
        if (get && path == "/slots") { handle_slots(req, c); return; }
        if (post && path.rfind("/slots/", 0) == 0) {
            int id;
            try { id = std::stoi(path.substr(7)); } catch (...) { throw_error(400, "Invalid slot ID"); }
            handle_slot_action(req, c, id);
            return;
        }
        if (get && path == "/lora-adapters") { respond_json(c, 200, Value::array()); return; }
        if (post && path == "/lora-adapters") {
            Value body;
            try { body = jd::parse(req.body); } catch (const jd::ParseError& e) { throw_error(400, std::string("[json.exception.parse_error.101] parse error: ") + e.what()); }
            if (!body.is_array()) throw_error(400, "Request body must be an array");
            if (body.size() > 0) throw_error(400, "invalid adapter id");
            Value v = Value::object(); v.set("success", true); respond_json(c, 200, v); return;
        }
        if (post && (path == "/embedding" || path == "/embeddings" || path == "/v1/embeddings"))
            throw_error(501, o.embedding ? "Q-Volta engines do not expose pooled embeddings; /embeddings is not implemented" : "This server does not support embeddings. Start it with `--embeddings`", "not_supported_error");
        if (post && (path == "/rerank" || path == "/reranking" || path == "/v1/rerank" || path == "/v1/reranking"))
            throw_error(501, "This server does not support reranking. Start it with `--reranking`", "not_supported_error");
        if (post && (path == "/v1/audio/transcriptions" ||
                     path == "/v1/audio/speech" || path == "/tts" || path == "/v1/systemone"))
            throw_error(501, "Endpoint " + path + " is not implemented by qv_server", "not_supported_error");
        if (get && path == "/v1/stream") { handle_stream_get(req, c); return; }
        if (post && path == "/v1/streams/lookup") { handle_stream_lookup(req, c); return; }
        if (m == "DELETE" && path == "/v1/stream") { handle_stream_delete(req, c); return; }
        if (get && (frontend || (o.webui && !o.path.empty())) && serve_webui(req, c, path)) return;
        // features that are off by default answer 403 (llama-server: server_setup_features)
        if ((get || post) && (path == "/tools" || path == "/cors-proxy")) {
            c.respond(403, kJson, R"({"error":{"message":"this feature is disabled","type":"feature_disabled"}})");
            return;
        }
        throw_error(404, "File Not Found", "not_found_error");
    } catch (const ApiError& e) {
        if (c.streaming && !c.broken) { c.write_chunk("data: " + error_json(e) + "\n\n"); c.end_chunks(); c.close_after = true; }
        else if (!c.broken) {
            std::string body = error_json(e);
            if (e.status == 404 || e.status == 401 || (e.status == 503 && e.message == "Loading model")) {  // httplib's error handler / the llama-server middlewares use a different key order
                Value err = Value::object();
                err.set("message", e.message); err.set("type", e.type); err.set("code", e.status);
                Value o2 = Value::object(); o2.set("error", std::move(err)); body = o2.dump();
            }
            c.respond(e.status, kJson, body);
        }
    } catch (const std::exception& e) {
        if (c.streaming && !c.broken) { c.write_chunk(std::string("data: ") + error_json(ApiError{500, e.what(), "server_error", Value()}) + "\n\n"); c.end_chunks(); }
        else if (!c.broken) c.respond(500, kJson, error_json(ApiError{500, e.what(), "server_error", Value()}));
        c.close_after = true;
    }
}

}  // namespace qv::server

namespace qv::server {
static long long mono_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }

void App::ensure_engine() {
    last_task_ms_ = mono_ms();
    std::lock_guard<std::mutex> rl(reload_mu_);
    if (eng) return;
    auto t0 = Clock::now();
    std::fprintf(stderr, "[qv_server] server is exiting sleeping state (reloading the model)\n");
    eng = reload_engine();
    eng->sync();
    for (auto& c : slot_cached_) c.clear();
    for (auto& v : slot_ckpts_) v.clear();
    for (auto& b : slot_cacheable_) b = 0;
    parked_pool_.clear();
    sleeping = false;
    std::fprintf(stderr, "[qv_server] model reloaded in %.1fs\n", std::chrono::duration<double>(Clock::now() - t0).count());
}

void App::start_sleep_watcher() {
    if (o.sleep_idle_seconds <= 0 || !reload_engine) return;
    last_task_ms_ = mono_ms();
    sleep_thr_ = std::thread([this] {
        while (!sleep_stop_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (!ready || sleeping || !eng) continue;
            if (mono_ms() - last_task_ms_ < 1000LL * o.sleep_idle_seconds) continue;
            long ticket;
            {
                std::unique_lock<std::mutex> lk(mu_);
                if (!engine_idle_locked()) continue;  // busy
            }
            ticket = lease_acquire([] { return true; });  // exclusive lease: nothing else can start while we unload
            if (ticket < 0) continue;
            {
                struct Rel { App& a; long t; ~Rel() { a.lease_release(t); } } rel{*this, ticket};
                if (mono_ms() - last_task_ms_ >= 1000LL * o.sleep_idle_seconds && eng) {
                    std::fprintf(stderr, "[qv_server] server is entering sleeping state (idle for %d s): model unloaded\n", o.sleep_idle_seconds);
                    eng.reset();
                    for (auto& c : slot_cached_) c.clear();
                    for (auto& v : slot_ckpts_) v.clear();
                    for (auto& b : slot_cacheable_) b = 0;
                    parked_pool_.clear();
                    sleeping = true;
                }
            }
        }
    });
}
void App::stop_sleep_watcher() { sleep_stop_ = true; if (sleep_thr_.joinable()) sleep_thr_.join(); }

const ModelPreset* App::find_preset(const std::string& name) const {
    if (presets.empty()) return nullptr;
    if (name.empty()) {
        if (!active_preset_id.empty()) {
            for (const auto& p : presets) if (p.id == active_preset_id) return &p;
        }
        return &presets[0];
    }
    for (const auto& p : presets) if (p.id == name) return &p;
    for (const auto& p : presets) {
        for (const auto& a : p.opts.alias) if (a == name) return &p;
    }
    return nullptr;
}

void App::unload_model() {
    eng.reset();
    std::lock_guard<std::mutex> slk(state_mu_);
    venc.reset();
    tmpl.reset();
    {
        std::lock_guard<std::mutex> lk(gm_);
        vocab_.reset();
        matchers_.clear();
    }
    for (auto& c : slot_cached_) c.clear();
    for (auto& v : slot_ckpts_) v.clear();
    for (auto& b : slot_cacheable_) b = 0;
    parked_pool_.clear();
    tok = Tokenizer();
    model_id.clear();
    model_template.clear();
    model_meta_json.clear();
    embedded_blobs.clear();
    embedded_applied.clear();
    template_source = "none";
    tokenizer_path.clear();
    model_arch.clear();
    stop_ids.clear();
    bos_id = eos_id = -1;
    image_token_id = -1;
    fim_pre = fim_suf = fim_mid = fim_rep = fim_sep = -1;
    cached_n_embd = 0;
    cached_weight_bytes = 0;
    sleeping = false;
    active_preset_id.clear();
}

void App::load_model(const Options& mo_in, const std::string& forced_id) {
    unload_model();
    try {
        std::unique_lock<std::mutex> slk(state_mu_);
        o = mo_in;
        if (o.ctx <= 0) o.ctx = 4096;
        n_ctx = o.ctx;  // refined by set_sequences() once the backend is known
        model_id = !forced_id.empty() ? forced_id : (!o.alias.empty() ? o.alias[0] : (o.model.empty() ? std::string("qv-mock") : o.model));
        slk.unlock();
        init_slots();
        for (auto& w : o.warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());

        auto exists = [](const std::string& p) { std::ifstream f(p, std::ios::binary); return (bool)f; };
        std::string stem = o.model;
        if (stem.size() > 3 && stem.compare(stem.size() - 3, 3, ".qv") == 0 && !exists(stem)) stem.resize(stem.size() - 3);

        std::optional<qv::ModelFile> emb;
        if (!o.model.empty()) {
            try { if (exists(stem) || exists(stem + ".qv") || exists(stem + ".qv.json")) emb = qv::ModelFile::open(stem); }
            catch (const std::exception&) {}
        }
        slk.lock();
        if (emb) {
            for (auto& [n, b] : emb->blobs) embedded_blobs.push_back(n);
            model_meta_json = emb->meta();
        }
        std::string tpath = o.tokenizer;
        bool tok_embedded = false;
        if (tpath.empty() && emb && emb->has_blob("tokenizer.json")) {
            try { tok = qv::Tokenizer::from_memory(emb->blob("tokenizer.json")); }
            catch (const std::exception& e) { throw std::runtime_error(std::string("embedded tokenizer: ") + e.what()); }
            tok_embedded = true;
            tpath = "embedded:" + o.model;
        }
        if (tpath.empty() && !o.model.empty()) {
            std::string base = o.model;
            if (base.size() > 3 && base.compare(base.size() - 3, 3, ".qv") == 0) base.resize(base.size() - 3);
            size_t sl = base.find_last_of('/');
            std::string dir = sl == std::string::npos ? "." : base.substr(0, sl);
            for (std::string cand : {base + ".tokenizer.json", dir + "/../Qwen3.8-27B/tokenizer.json", dir + "/tokenizer.json"})
                if (exists(cand)) { tpath = cand; break; }
        }
        if (tpath.empty()) throw std::runtime_error("no tokenizer: the model file has no embedded tokenizer.json (embed one with tools/qv_embed.py) and none was found next to it; pass --tokenizer PATH");
        if (!tok_embedded) {
            try { tok = qv::Tokenizer::load(tpath); }
            catch (const std::exception& e) { throw std::runtime_error(std::string("tokenizer: ") + e.what()); }
        }
        tokenizer_path = tpath;
        for (const char* s : {"<|im_end|>", "<|endoftext|>", "<|content_model_end_sampling|>"}) { int id = tok.token_id(s); if (id >= 0) stop_ids.push_back(id); }
        eos_id = tok.token_id("<|im_end|>");
        if (eos_id < 0) eos_id = tok.token_id("<|content_model_end_sampling|>");
        bos_id = tok.token_id("<|endoftext|>");
        if (emb && emb->has_blob("generation_config.json")) {
            try {
                qv::jd::Value gc = qv::jd::parse(emb->blob("generation_config.json"));
                std::vector<int> eos;
                if (const qv::jd::Value* e = gc.find("eos_token_id")) { if (e->is_array()) for (auto& v : e->items()) eos.push_back((int)v.as_int()); else if (e->is_number()) eos.push_back((int)e->as_int()); }
                for (int id : eos)
                    if (id >= 0 && id < tok.vocab_size() && std::find(stop_ids.begin(), stop_ids.end(), id) == stop_ids.end()) stop_ids.push_back(id);
                if (!eos.empty() && eos_id < 0 && eos[0] >= 0 && eos[0] < tok.vocab_size()) eos_id = eos[0];
                if (const qv::jd::Value* b = gc.find("bos_token_id")) if (b->is_number() && bos_id < 0) bos_id = (int)b->as_int();
                if (o.embedded_defaults) {
                    auto num = [&](const char* k, const char* flag, auto& dst) {
                        const qv::jd::Value* v = gc.find(k);
                        if (!v || !v->is_number() || o.given.count(flag)) return;
                        dst = (std::decay_t<decltype(dst)>)v->as_double();
                        embedded_applied += std::string(embedded_applied.empty() ? "" : ", ") + flag + "=" + v->dump();
                    };
                    num("temperature", "temp", o.temp);
                    num("top_k", "top-k", o.top_k);
                    num("top_p", "top-p", o.top_p);
                }
            } catch (const std::exception& e) { std::fprintf(stderr, "warning: embedded generation_config.json ignored: %s\n", e.what()); }
        }
        fim_pre = tok.token_id("<|fim_prefix|>");
        fim_suf = tok.token_id("<|fim_suffix|>");
        fim_mid = tok.token_id("<|fim_middle|>");
        fim_rep = tok.token_id("<|repo_name|>");
        fim_sep = tok.token_id("<|file_sep|>");
        template_source = "none";
        if (emb && emb->has_blob("chat_template.jinja")) {
            model_template = std::string(emb->blob("chat_template.jinja"));
            template_source = "embedded";
        } else if (!tok_embedded) {
            std::string tp = tpath.substr(0, tpath.find_last_of('/') == std::string::npos ? 0 : tpath.find_last_of('/')) + "/chat_template.jinja";
            std::ifstream f(tp, std::ios::binary);
            if (f) { std::stringstream ss; ss << f.rdbuf(); model_template = ss.str(); template_source = "sidecar"; }
        }
        {
            std::string warn, over = o.chat_template;
            if (!o.chat_template_file.empty()) {
                std::ifstream f(o.chat_template_file, std::ios::binary);
                if (!f) throw std::runtime_error("failed to open chat template file " + o.chat_template_file);
                std::stringstream ss; ss << f.rdbuf(); over = ss.str();
            }
            if (!over.empty()) template_source = "override";
            tmpl = std::make_unique<ChatTemplates>(model_template, over, o.jinja, warn);
            if (!warn.empty()) std::fprintf(stderr, "warning: %s\n", warn.c_str());
        }
        if (emb && !embedded_blobs.empty()) {
            std::string names;
            for (auto& n : embedded_blobs) names += (names.empty() ? "" : ", ") + n;
            std::fprintf(stderr, "[qv_server] embedded in the model: %s; tokenizer=%s, chat template=%s%s%s\n", names.c_str(), tok_embedded ? "embedded" : "explicit/sidecar", template_source.c_str(),
                         embedded_applied.empty() ? "" : ", sampling defaults from generation_config: ", embedded_applied.c_str());
        }
        slk.unlock();

        std::string backend = o.backend;
#ifdef QV_HAVE_CUDA
        if (backend == "auto") {
            int n = 0;
            backend = (!(o.ngl_given && o.n_gpu_layers == 0) && cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? "cuda" : "cpu";
        }
#else
        if (backend == "auto") backend = "cpu";
#endif
        if (backend != "cpu" && backend != "cuda" && backend != "mock") throw std::runtime_error("--backend must be cuda, cpu, auto or mock");

        {  // sequences: all engines hold one sequence per slot (except with --mtp: one sequence, requests are serialised)
            const bool can_multi = true;  // every engine (CUDA, CPU, mock) holds one sequence per slot
            if (o.mtp && o.n_parallel > 1) std::fprintf(stderr, "warning: --mtp supports one sequence: --parallel %d slots share it and requests run one at a time\n", o.n_parallel);
            std::lock_guard<std::mutex> lk(mu_);
            set_sequences(can_multi && !o.mtp);
        }
        qv::EngineOptions eo;
        eo.max_ctx = slot_ctx;
        eo.n_seq = multi_seq ? n_slots : 1;
        eo.kv_mode = o.kv_mode;
        eo.kv4_kmode = o.kv4_kmode;
        eo.mtp = o.mtp;
        eo.mtp_depth = o.mtp_depth;
        eo.threads = o.threads > 0 ? o.threads : (int)std::min(32u, std::max(1u, std::thread::hardware_concurrency()));
        chunk = o.prefill_chunk > 0 ? o.prefill_chunk : (o.ubatch > 0 ? o.ubatch : (backend == "cuda" ? 2048 : 256));
        eo.prefill_chunk = chunk;
        mtp_chunk = (backend == "cuda") ? 16 : 8;
        if (const char* e = std::getenv("QV_MTP_CHUNK")) mtp_chunk = std::clamp(std::atoi(e), 2, 64);
        if (const char* e = std::getenv("QV_SHARED_CHUNK")) shared_chunk = std::max(1, std::atoi(e));

        Options mo = o;
        auto make_engine = [this, backend, stem, eo, mo]() mutable -> std::unique_ptr<qv::Engine> {
            std::unique_ptr<qv::Engine> e;
            if (backend == "mock") {
                if (const char* ms = std::getenv("QV_MOCK_LOAD_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(ms)));
                auto script = tok.encode(mo.mock_text, false);
                e = std::make_unique<qv::MockEngine>(tok.vocab_size(), script, eos_id, mo.mock_loop, mo.mock_delay_us, eo.n_seq);
            } else {
                if (mo.model.empty()) throw std::runtime_error("no model: pass -m MODEL (or set LLAMA_ARG_MODEL)");
                qv::ModelFile mf = qv::ModelFile::open(stem);
                if (eo.mtp && mf.cfg.arch == "inkling2_loop") {
                    std::string base = stem.size() > 3 && stem.compare(stem.size() - 3, 3, ".qv") == 0 ? stem.substr(0, stem.size() - 3) : stem;
                    auto ex = [](const std::string& f) { return std::ifstream(f).good(); };
                    if (!mo.mtp_file.empty()) eo.mtp_file = mo.mtp_file;
                    else if (ex(base + "-mtp.qv")) eo.mtp_file = base + "-mtp.qv";
                    else {
                        const size_t d = base.rfind('-'), sl = base.rfind('/');
                        const std::string alt = d != std::string::npos && (sl == std::string::npos || d > sl) ? base.substr(0, d) + "-mtp.qv" : "";
                        if (!alt.empty() && ex(alt)) eo.mtp_file = alt;
                        else throw std::runtime_error("--mtp: MTP sidecar not found (tried " + base + "-mtp.qv" + (alt.empty() ? "" : " and " + alt) + "); pass --mtp-file PATH");
                    }
                    mtp_chunk = 32;
                    if (eo.mtp_depth == 1) eo.mtp_depth = 3;
                }
                if (backend == "cpu") {
#ifdef QV_HAVE_CPU
                    e = qv::make_cpu_engine(mf, eo);
#else
                    throw std::runtime_error("this build has no CPU engine");
#endif
                } else {
#ifdef QV_HAVE_CUDA
                    e = qv::make_cuda_engine(mf, eo);
#else
                    throw std::runtime_error("this build has no CUDA engine (configure with -DQV_CUDA=ON)");
#endif
                }
            }
            return e;
        };

        auto new_eng = make_engine();
        std::string arch = backend == "mock" ? "mock" : new_eng->config().arch;
        int ceos = backend == "mock" ? -1 : new_eng->config().eos;
        int hidden = backend == "mock" ? 0 : new_eng->config().hidden;
        long long wbytes = backend == "mock" ? 0 : (long long)new_eng->device_bytes_weights();
        new_eng->sync();

        // Inkling is sampled at temperature 1.0 (its recommended setting; colder sampling loops more, docs/LOOPS.md). This replaces the
        // llama.cpp default of 0.8 unless --temp / a preset sets it or the model ships a generation_config temperature.
        if (arch == "inkling2_loop" && !o.given.count("temp") && embedded_applied.find("temp=") == std::string::npos) o.temp = 1.0f;
        // ---- vision sidecar (Inkling image input)
        std::unique_ptr<vision::Encoder> new_venc;
        std::string vf = o.vision_file;
        if (vf.empty() && !o.model.empty() && arch == "inkling2_loop") {
            std::string base = o.model;
            if (base.size() > 3 && base.compare(base.size() - 3, 3, ".qv") == 0) base.resize(base.size() - 3);
            const size_t sl = base.find_last_of('/');
            const std::string dir = sl == std::string::npos ? "." : base.substr(0, sl);
            std::string fam = base;
            if (const size_t d = fam.find_last_of('-'); d != std::string::npos && d > sl + 1) fam.resize(d);
            for (const std::string& cand : {base + "-vision.qv", fam + "-vision.qv", dir + "/inkling2-mini-vision.qv"})
                if (exists(cand)) { vf = cand; break; }
        }
        if (vf == "mock") {
            new_venc = std::make_unique<qv::MockVisionEncoder>();
        } else if (!vf.empty() && vf != "none") {
            const bool cu = backend == "cuda";
            new_venc = qv::vision::Encoder::create(vf, cu);
            std::fprintf(stderr, "[qv_server] vision: %s (%s encoder, image tokens = ceil(h/40)*ceil(w/40), remote images %s, local root '%s')\n", vf.c_str(), new_venc->backend(),
                         o.allow_remote_images ? "ALLOWED" : "off", o.image_root.c_str());
        }

        slk.lock();
        eng = std::move(new_eng);
        model_arch = arch;
        if (ceos >= 0) {
            if (std::find(stop_ids.begin(), stop_ids.end(), ceos) == stop_ids.end()) stop_ids.push_back(ceos);
            if (eos_id < 0) eos_id = ceos;
        }
        cached_n_embd = hidden;
        cached_weight_bytes = wbytes;
        reload_engine = make_engine;
        backend_name = backend;
        venc = std::move(new_venc);
        if (venc) {
            image_token_id = tok.token_id(o.image_token);
            if (image_token_id < 0) throw std::runtime_error("image token " + o.image_token + " is not in the tokenizer");
            vsrc.allow_remote = o.allow_remote_images;
            vsrc.limits.max_side = o.image_max_side;
            std::error_code ec;
            vsrc.local_root = o.image_root.empty() ? "" : std::filesystem::absolute(o.image_root, ec).string();
        }
        active_preset_id = forced_id;
    } catch (...) {
        unload_model();
        throw;
    }
}

// Model lease: requests of the active model run concurrently (each registers as a "model user"); a model switch needs the engine to itself.
long App::acquire_model_lease(const std::string& requested_model, const std::function<bool()>& alive) {
    if (!has_presets()) return -1;
    const ModelPreset* p = find_preset(requested_model);
    if (!p) throw_error(404, "The model '" + requested_model + "' does not exist", "model_not_found");
    for (;;) {
        long ticket = user_acquire(alive);
        if (ticket < 0) return -2;
        if (active_preset_id == p->id && eng) return ticket;
        user_release();
        ticket = lease_acquire(alive);
        if (ticket < 0) return -2;
        if (active_preset_id == p->id && eng) { lease_downgrade(); return ticket; }
        auto t0 = Clock::now();
        std::string prev = active_preset_id;
        try {
            load_model(p->opts, p->id);
        } catch (const std::exception& e) {
            lease_release(ticket);
            throw_error(503, std::string("failed to load model '") + p->id + "': " + e.what(), "unavailable_error");
        }
        if (o.verbosity >= 3 && !o.log_disable) {
            if (prev.empty())
                std::fprintf(stderr, "[qv_server] loaded preset '%s' in %.1fs (backend=%s, n_ctx=%d, mtp=%d)\n",
                             p->id.c_str(), std::chrono::duration<double>(Clock::now() - t0).count(), backend_name.c_str(), n_ctx, (int)o.mtp);
            else
                std::fprintf(stderr, "[qv_server] switched model '%s' -> '%s' in %.1fs (backend=%s, n_ctx=%d, mtp=%d)\n",
                             prev.c_str(), p->id.c_str(), std::chrono::duration<double>(Clock::now() - t0).count(), backend_name.c_str(), n_ctx, (int)o.mtp);
        }
        lease_downgrade();
        return ticket;
    }
}

void App::release_model_lease(long ticket) {
    if (ticket < 0) return;
    last_task_ms_ = mono_ms();
    user_release();
}

}  // namespace qv::server
