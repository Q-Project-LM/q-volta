// Grammars for forced tool calls (tool_choice = required / {"function":{"name":..}}) per wire format.
#include <algorithm>
#include <map>
#include <stdexcept>

#include "grammar/json_schema.h"

namespace qv::grammar {

namespace {
std::string hex_class_char(unsigned char c) {
    static const char* h = "0123456789ABCDEF";
    std::string s = "\\x";
    s += h[c >> 4];
    s += h[c & 15];
    return s;
}
}  // namespace

std::string add_until_rule(SchemaGrammarBuilder& b, const std::string& name, const std::string& lit) {
    if (lit.empty()) throw std::invalid_argument("add_until_rule: empty literal");
    const size_t n = lit.size();
    // KMP automaton over bytes of the literal
    std::vector<size_t> fail(n + 1, 0);
    for (size_t i = 1, k = 0; i < n; ++i) {
        while (k && lit[i] != lit[k]) k = fail[k];
        if (lit[i] == lit[k]) ++k;
        fail[i + 1] = k;
    }
    auto next = [&](size_t k, unsigned char c) {
        while (k && (unsigned char)lit[k] != c) k = fail[k];
        if ((unsigned char)lit[k] == c) ++k;
        return k;
    };
    std::string alphabet;
    for (char c : lit) if (alphabet.find(c) == std::string::npos) alphabet += c;
    auto rn = [&](size_t k) { return name + "-s" + std::to_string(k); };
    for (size_t k = 0; k < n; ++k) {
        std::map<size_t, std::string> by_target;
        for (char c : alphabet) by_target[next(k, (unsigned char)c)] += hex_class_char((unsigned char)c);
        std::string body, nonzero;
        for (auto& [t, chars] : by_target) {
            if (t == 0) continue;
            nonzero += chars;
            if (!body.empty()) body += " | ";
            body += "[" + chars + "]" + (t == n ? "" : " " + rn(t));
        }
        if (!body.empty()) body += " | ";
        body += "[^" + nonzero + "] " + rn(0);
        b.add_rule(rn(k), body);
    }
    return b.add_rule(name, rn(0));
}

namespace {
struct ToolDef {
    std::string name;
    jd::Value params;
};

std::vector<ToolDef> collect_tools(const jd::Value& tools, const std::string& forced) {
    std::vector<ToolDef> out;
    if (!tools.is_array()) throw std::invalid_argument("tools must be an array");
    for (auto& t : tools.items()) {
        const jd::Value* fn = t.is_object() ? t.find("function") : nullptr;
        if (!fn) fn = &t;
        const jd::Value* nm = fn->is_object() ? fn->find("name") : nullptr;
        if (!nm || !nm->is_string()) continue;
        if (!forced.empty() && nm->as_string() != forced) continue;
        ToolDef d;
        d.name = nm->as_string();
        const jd::Value* p = fn->find("parameters");
        if (p && p->is_object()) d.params = *p;
        else { d.params = jd::Value::object(); d.params.set("type", "object"); }
        out.push_back(std::move(d));
    }
    if (out.empty()) throw std::invalid_argument(forced.empty() ? "no tools given" : "forced tool '" + forced + "' not found in tools");
    return out;
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) { if (i) o += sep; o += v[i]; }
    return o;
}
std::string q(const std::string& s) { return gbnf_literal(s); }

// Inkling: the chat template prints call arguments with `dictsort` (keys in byte order), optional keys included, so the model writes them in that order.
// The generic schema rule emits required keys first and then the optional ones in declaration order, which rejects a correct call such as
// {"path":..,"pattern":..} (pattern required, path optional). Top-level properties are therefore declared in byte order, each required or optional in place.
// Returns "" when the schema has no explicit properties or allows additional ones (the generic rule is used then).
std::string inkling_args_rule(SchemaGrammarBuilder& b, const std::string& base, const jd::Value& params) {
    const jd::Value* pr = params.find("properties");
    if (!pr || !pr->is_object() || pr->members().empty()) return "";
    if (const jd::Value* ap = params.find("additionalProperties"); ap && !(ap->is_bool() && !ap->as_bool())) return "";
    std::vector<std::string> req;
    if (const jd::Value* r = params.find("required"); r && r->is_array()) for (auto& x : r->items()) if (x.is_string()) req.push_back(x.as_string());
    std::vector<std::pair<std::string, const jd::Value*>> props;
    for (auto& m : pr->members()) props.emplace_back(m.first, &m.second);
    std::stable_sort(props.begin(), props.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    const size_t n = props.size();
    auto cn = [&](size_t i, bool first) { return base + "-c" + std::to_string(i) + (first ? "f" : "n"); };
    std::vector<std::string> kv(n);
    for (size_t i = 0; i < n; ++i) {
        const std::string kn = base + "-k" + std::to_string(i);
        const std::string val = b.add_schema(kn + "-v", *props[i].second, &params);
        kv[i] = b.add_rule(kn, q("\"" + props[i].first + "\"") + " ws " + q(":") + " ws " + val);
    }
    b.add_rule(cn(n, true), "\"\"");
    b.add_rule(cn(n, false), "\"\"");
    for (size_t i = n; i-- > 0;) {
        const bool required = std::find(req.begin(), req.end(), props[i].first) != req.end();
        for (bool first : {true, false}) {
            const std::string item = (first ? std::string() : q(",") + " ws ") + kv[i] + " ws " + cn(i + 1, false);
            b.add_rule(cn(i, first), required ? item : "(" + item + " | " + cn(i + 1, first) + ")");
        }
    }
    return b.add_rule(base + "-ordered", q("{") + " ws " + cn(0, true) + " ws " + q("}"));
}
}  // namespace

ToolGrammar tool_call_grammar(const jd::Value& tools, const ToolGrammarOptions& opt) {
    auto defs = collect_tools(tools, opt.forced_name);
    SchemaGrammarBuilder b;
    b.add_rule("ws", "[ \\t\\n]*");
    std::vector<std::string> calls;       // rule names of full calls
    std::vector<std::string> lazy_calls;  // same, as they start at the trigger word
    int idx = 0;
    for (auto& d : defs) {
        std::string base = "tool-" + gbnf_rule_name(d.name) + (defs.size() > 1 ? "-" + std::to_string(idx) : "");
        ++idx;
        std::string call, lazy;
        if (opt.format == ToolFormat::QwenXml) {
            std::string body;
            const jd::Value* props = d.params.find("properties");
            std::vector<std::string> req;
            if (const jd::Value* r = d.params.find("required"); r && r->is_array()) for (auto& x : r->items()) if (x.is_string()) req.push_back(x.as_string());
            if (props && props->is_object()) {
                for (auto& [k, sch] : props->members()) {
                    bool required = std::find(req.begin(), req.end(), k) != req.end();
                    std::string pn = base + "-p-" + gbnf_rule_name(k);
                    std::string val;
                    const jd::Value* ty = sch.is_object() ? sch.find("type") : nullptr;
                    const jd::Value* en = sch.is_object() ? sch.find("enum") : nullptr;
                    bool all_str_enum = en && en->is_array() && !en->items().empty();
                    if (all_str_enum) for (auto& e : en->items()) if (!e.is_string()) all_str_enum = false;
                    std::string pre = q("<parameter=" + k + ">\n");
                    std::string item;
                    if (all_str_enum) {
                        std::vector<std::string> alts;
                        for (auto& e : en->items()) alts.push_back(q(e.as_string()));
                        item = pre + " (" + join(alts, " | ") + ") " + q("\n</parameter>\n");
                    } else if (ty && ty->is_string() && ty->as_string() == "string") {
                        std::string u = add_until_rule(b, pn + "-text", "\n</parameter>");
                        item = pre + " " + u + " " + q("\n");
                    } else {
                        std::string v = b.add_schema(pn, sch, &d.params);
                        item = pre + " " + v + " " + q("\n</parameter>\n");
                    }
                    body += (required ? " " + item : " (" + item + ")?");
                }
            }
            call = q("<tool_call>\n<function=" + d.name + ">\n") + body + " " + q("</function>\n</tool_call>");
            lazy = call;
        } else {
            std::string args;
            if (opt.format == ToolFormat::Inkling) args = inkling_args_rule(b, base + "-oargs", d.params);
            if (args.empty()) args = b.add_schema(base + "-args", d.params);
            const std::string nm = q(d.name);
            auto json_call = [&](const std::string& key) {
                return q("{") + " ws " + q("\"name\"") + " ws " + q(":") + " ws " + q("\"" + d.name + "\"") + " ws " + q(",") + " ws " + q("\"" + key + "\"") + " ws " + q(":") + " ws " + args + " ws " + q("}");
            };
            switch (opt.format) {
                case ToolFormat::Hermes: call = q("<tool_call>\n") + " " + json_call("arguments") + " " + q("\n</tool_call>"); lazy = call; break;
                case ToolFormat::Mistral: call = json_call("arguments"); lazy = call; break;
                case ToolFormat::Generic: call = json_call("arguments"); lazy = call; break;
                case ToolFormat::Inkling:
                    call = q("<|message_model|>" + d.name + "<|content_invoke_tool_json|>") + " " + json_call("args") + " " + q("<|end_message|>");
                    lazy = q("<|content_invoke_tool_json|>") + " " + json_call("args") + " " + q("<|end_message|>");
                    break;
                default: break;
            }
        }
        calls.push_back(b.add_rule(base, call));
        lazy_calls.push_back(b.add_rule(base + "-lazy", lazy));
    }
    ToolGrammar out;
    auto build_root = [&](const std::vector<std::string>& cs, bool lazy) {
        std::string one = b.add_rule(lazy ? "call-lazy" : "call", join(cs, " | "));
        std::string root;
        switch (opt.format) {
            case ToolFormat::Mistral:
                root = q("[TOOL_CALLS][") + " " + one + (opt.parallel ? " (" + q(",") + " ws " + one + ")*" : "") + " " + q("]");
                break;
            case ToolFormat::Generic:
                if (opt.parallel) root = q("{") + " ws " + q("\"tool_calls\"") + " ws " + q(":") + " ws " + q("[") + " ws " + one + " (ws " + q(",") + " ws " + one + ")* ws " + q("]") + " ws " + q("}");
                else root = q("{") + " ws " + q("\"tool_call\"") + " ws " + q(":") + " ws " + one + " ws " + q("}");
                break;
            default:
                root = opt.parallel ? one + " (ws " + one + ")*" : one;
        }
        return root;
    };
    std::string root = build_root(calls, false);
    if (opt.thinking_forced_open) {
        std::string u = add_until_rule(b, "reasoning", opt.think_end);
        root = u + " ws " + root;
    }
    b.add_rule("root", root);
    out.gbnf = b.build();
    // lazy variant: starts at the trigger word
    std::string lroot = build_root(lazy_calls, true);
    // rebuild a standalone text for lazy: replace root
    {
        std::string text = b.build();
        size_t p = text.rfind("\nroot ::= ");
        p = p == std::string::npos ? 0 : p + 1;
        size_t e = text.find('\n', p);
        text.replace(p, e - p, "root ::= " + lroot);
        out.gbnf_lazy = text;
    }
    switch (opt.format) {
        case ToolFormat::Hermes: case ToolFormat::QwenXml: out.trigger_words = {"<tool_call>"}; break;
        case ToolFormat::Mistral: out.trigger_words = {"[TOOL_CALLS]"}; break;
        case ToolFormat::Inkling: out.trigger_words = {"<|content_invoke_tool_json|>"}; break;
        default: break;
    }
    return out;
}

}  // namespace qv::grammar
