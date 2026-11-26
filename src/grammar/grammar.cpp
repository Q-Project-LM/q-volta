// GBNF parser (port of llama.cpp's llama_grammar_parser) and the pushdown-stack matcher with cached state transitions and
// trie-based allowed-token masks.
#include "grammar/grammar.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <unordered_map>

namespace qv::grammar {

namespace {
enum : uint8_t { T_END = 0, T_ALT = 1, T_REF = 2, T_CHAR = 3, T_NOT = 4, T_RNG = 5, T_ALTCH = 6, T_ANY = 7 };
struct Elem {
    uint8_t t;
    uint32_t v;
};
using Rule = std::vector<Elem>;
constexpr uint32_t MAX_REPETITION_THRESHOLD = 2000;
constexpr uint32_t RECURSIVE_LIMIT = 1024;

bool is_digit_char(char c) { return c >= '0' && c <= '9'; }
bool is_word_char(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' || is_digit_char(c); }

std::pair<uint32_t, const char*> decode_utf8_ptr(const char* src) {
    static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t first = (uint8_t)*src;
    int len = lookup[first >> 4];
    uint8_t mask = (uint8_t)((1 << (8 - len)) - 1);
    uint32_t value = first & mask;
    const char* end = src + len;
    const char* pos = src + 1;
    for (; pos < end && *pos; pos++) value = (value << 6) + ((uint8_t)*pos & 0x3F);
    return {value, pos};
}

std::pair<uint32_t, const char*> parse_hex(const char* src, int size) {
    const char* pos = src;
    const char* end = src + size;
    uint32_t value = 0;
    for (; pos < end && *pos; pos++) {
        value <<= 4;
        char c = *pos;
        if (c >= 'a' && c <= 'f') value += c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') value += c - 'A' + 10;
        else if (c >= '0' && c <= '9') value += c - '0';
        else break;
    }
    if (pos != end) throw GrammarError(std::string("expecting ") + std::to_string(size) + " hex chars at " + src);
    return {value, pos};
}

const char* parse_space(const char* src, bool newline_ok) {
    const char* pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' || (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') {
            while (*pos && *pos != '\r' && *pos != '\n') pos++;
        } else pos++;
    }
    return pos;
}
const char* parse_name(const char* src) {
    const char* pos = src;
    while (is_word_char(*pos)) pos++;
    if (pos == src) throw GrammarError(std::string("expecting name at ") + src);
    return pos;
}
const char* parse_int(const char* src) {
    const char* pos = src;
    while (is_digit_char(*pos)) pos++;
    if (pos == src) throw GrammarError(std::string("expecting integer at ") + src);
    return pos;
}
std::pair<uint32_t, const char*> parse_char(const char* src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return {'\t', src + 2};
            case 'r': return {'\r', src + 2};
            case 'n': return {'\n', src + 2};
            case '\\': case '"': case '[': case ']': case '-': return {(uint32_t)src[1], src + 2};
            default: throw GrammarError(std::string("unknown escape at ") + src);
        }
    } else if (*src) return decode_utf8_ptr(src);
    throw GrammarError("unexpected end of input");
}

struct Parser {
    std::map<std::string, uint32_t> symbol_ids;
    std::vector<Rule> rules;

    uint32_t get_symbol_id(const char* src, size_t len) {
        uint32_t next = (uint32_t)symbol_ids.size();
        return symbol_ids.emplace(std::string(src, len), next).first->second;
    }
    uint32_t generate_symbol_id(const std::string& base) {
        uint32_t next = (uint32_t)symbol_ids.size();
        symbol_ids[base + "_" + std::to_string(next)] = next;
        return next;
    }
    void add_rule(uint32_t id, const Rule& r) {
        if (rules.size() <= id) rules.resize(id + 1);
        rules[id] = r;
    }

    const char* parse_alternates(const char* src, const std::string& rule_name, uint32_t rule_id, bool nested, uint32_t depth) {
        Rule rule;
        const char* pos = parse_sequence(src, rule_name, rule, nested, depth);
        while (*pos == '|') {
            rule.push_back({T_ALT, 0});
            pos = parse_space(pos + 1, true);
            pos = parse_sequence(pos, rule_name, rule, nested, depth);
        }
        rule.push_back({T_END, 0});
        add_rule(rule_id, rule);
        return pos;
    }

    const char* parse_sequence(const char* src, const std::string& rule_name, Rule& rule, bool nested, uint32_t depth) {
        size_t last_sym_start = rule.size();
        const char* pos = src;
        uint64_t n_prev_rules = 1;
        auto handle_rep = [&](uint64_t min_times, uint64_t max_times) {
            bool no_max = max_times == UINT64_MAX;
            if (last_sym_start == rule.size()) throw GrammarError(std::string("expecting preceding item to */+/?/{ at ") + pos);
            Rule prev_rule(rule.begin() + last_sym_start, rule.end());
            uint64_t total_rules = 1;
            if (!no_max && max_times > 0) total_rules = max_times;
            else if (min_times > 0) total_rules = min_times;
            if (n_prev_rules * total_rules > MAX_REPETITION_THRESHOLD)
                throw GrammarError("number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please reduce the number of repetitions or rule complexity");
            if (min_times == 0) rule.resize(last_sym_start);
            else for (uint64_t i = 1; i < min_times; i++) rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            uint32_t last_rec = 0;
            uint64_t n_opt = no_max ? 1 : max_times - min_times;
            Rule rec(prev_rule);
            for (uint64_t i = 0; i < n_opt; i++) {
                rec.resize(prev_rule.size());
                uint32_t rec_id = generate_symbol_id(rule_name);
                if (i > 0 || no_max) rec.push_back({T_REF, no_max ? rec_id : last_rec});
                rec.push_back({T_ALT, 0});
                rec.push_back({T_END, 0});
                add_rule(rec_id, rec);
                last_rec = rec_id;
            }
            if (n_opt > 0) rule.push_back({T_REF, last_rec});
            n_prev_rules *= total_rules;
        };

        while (*pos) {
            if (*pos == '"') {
                pos++;
                last_sym_start = rule.size();
                n_prev_rules = 1;
                while (*pos != '"') {
                    if (!*pos) throw GrammarError("unexpected end of input");
                    auto cp = parse_char(pos);
                    pos = cp.second;
                    rule.push_back({T_CHAR, cp.first});
                }
                pos = parse_space(pos + 1, nested);
            } else if (*pos == '[') {
                pos++;
                uint8_t start_type = T_CHAR;
                if (*pos == '^') { pos++; start_type = T_NOT; }
                last_sym_start = rule.size();
                n_prev_rules = 1;
                while (*pos != ']') {
                    if (!*pos) throw GrammarError("unexpected end of input");
                    auto cp = parse_char(pos);
                    pos = cp.second;
                    uint8_t type = last_sym_start < rule.size() ? (uint8_t)T_ALTCH : start_type;
                    rule.push_back({type, cp.first});
                    if (pos[0] == '-' && pos[1] != ']') {
                        if (!pos[1]) throw GrammarError("unexpected end of input");
                        auto ec = parse_char(pos + 1);
                        pos = ec.second;
                        rule.push_back({T_RNG, ec.first});
                    }
                }
                pos = parse_space(pos + 1, nested);
            } else if (*pos == '<' || *pos == '!') {
                throw GrammarError("token elements (<[id]>) are not supported");
            } else if (is_word_char(*pos)) {
                const char* name_end = parse_name(pos);
                uint32_t ref = get_symbol_id(pos, name_end - pos);
                pos = parse_space(name_end, nested);
                last_sym_start = rule.size();
                n_prev_rules = 1;
                rule.push_back({T_REF, ref});
            } else if (*pos == '(') {
                if (depth >= RECURSIVE_LIMIT) throw GrammarError("grammar recursion depth limit exceeded (max " + std::to_string(RECURSIVE_LIMIT) + "), please reduce nesting depth");
                pos = parse_space(pos + 1, true);
                uint32_t before = (uint32_t)symbol_ids.size();
                uint32_t sub = generate_symbol_id(rule_name);
                pos = parse_alternates(pos, rule_name, sub, true, depth + 1);
                n_prev_rules = std::max(1u, (uint32_t)symbol_ids.size() - before);
                last_sym_start = rule.size();
                rule.push_back({T_REF, sub});
                if (*pos != ')') throw GrammarError(std::string("expecting ')' at ") + pos);
                pos = parse_space(pos + 1, nested);
            } else if (*pos == '.') {
                last_sym_start = rule.size();
                n_prev_rules = 1;
                rule.push_back({T_ANY, 0});
                pos = parse_space(pos + 1, nested);
            } else if (*pos == '*') { pos = parse_space(pos + 1, nested); handle_rep(0, UINT64_MAX); }
            else if (*pos == '+') { pos = parse_space(pos + 1, nested); handle_rep(1, UINT64_MAX); }
            else if (*pos == '?') { pos = parse_space(pos + 1, nested); handle_rep(0, 1); }
            else if (*pos == '{') {
                pos = parse_space(pos + 1, nested);
                if (!is_digit_char(*pos)) throw GrammarError(std::string("expecting an int at ") + pos);
                const char* ie = parse_int(pos);
                uint64_t min_times = std::stoull(std::string(pos, ie - pos));
                pos = parse_space(ie, nested);
                uint64_t max_times = UINT64_MAX;
                if (*pos == '}') {
                    max_times = min_times;
                    pos = parse_space(pos + 1, nested);
                } else if (*pos == ',') {
                    pos = parse_space(pos + 1, nested);
                    if (is_digit_char(*pos)) {
                        const char* ie2 = parse_int(pos);
                        max_times = std::stoull(std::string(pos, ie2 - pos));
                        pos = parse_space(ie2, nested);
                    }
                    if (*pos != '}') throw GrammarError(std::string("expecting '}' at ") + pos);
                    pos = parse_space(pos + 1, nested);
                } else throw GrammarError(std::string("expecting ',' at ") + pos);
                if (min_times > MAX_REPETITION_THRESHOLD) throw GrammarError("number of repetitions exceeds sane defaults, please reduce the number of repetitions");
                if (max_times != UINT64_MAX && max_times > MAX_REPETITION_THRESHOLD) max_times = UINT64_MAX;
                handle_rep(min_times, max_times);
            } else break;
        }
        return pos;
    }

    const char* parse_rule(const char* src) {
        const char* name_end = parse_name(src);
        const char* pos = parse_space(name_end, false);
        size_t name_len = name_end - src;
        uint32_t id = get_symbol_id(src, name_len);
        std::string name(src, name_len);
        if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) throw GrammarError(std::string("expecting ::= at ") + pos);
        pos = parse_space(pos + 3, true);
        pos = parse_alternates(pos, name, id, false, 0);
        if (*pos == '\r') pos += pos[1] == '\n' ? 2 : 1;
        else if (*pos == '\n') pos++;
        else if (*pos) throw GrammarError(std::string("expecting newline or end at ") + pos);
        return parse_space(pos, true);
    }
};

bool is_eos(const Elem& e) { return e.t == T_END || e.t == T_ALT; }

bool detect_left_recursion(const std::vector<Rule>& rules, size_t idx, std::vector<bool>& visited, std::vector<bool>& in_progress, std::vector<bool>& may_be_empty, uint32_t depth = 0) {
    if (in_progress[idx]) return true;
    if (depth > RECURSIVE_LIMIT) throw GrammarError("grammar rules are chained more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    in_progress[idx] = true;
    const Rule& rule = rules[idx];
    bool at_start = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (is_eos(rule[i])) {
            if (at_start) { may_be_empty[idx] = true; break; }
            at_start = true;
        } else at_start = false;
    }
    bool recurse = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (rule[i].t == T_REF && recurse) {
            if (detect_left_recursion(rules, rule[i].v, visited, in_progress, may_be_empty, depth + 1)) return true;
            if (!may_be_empty[rule[i].v]) recurse = false;
        } else if (is_eos(rule[i])) recurse = true;
        else recurse = false;
    }
    in_progress[idx] = false;
    visited[idx] = true;
    return false;
}

}  // namespace

// ============================================================================ compiled grammar + state machine
struct Impl {
    std::vector<Elem> e;               // all rules concatenated
    std::vector<uint32_t> rule_start;
    std::vector<std::string> names;
    uint32_t root = 0;

    std::mutex mu;
    struct St {
        std::vector<std::vector<uint32_t>> stacks;
        bool complete = false;
        std::array<int32_t, 128> ascii;  // -2 = not computed yet
        std::unordered_map<uint32_t, int32_t> other;
        int8_t uniform = -1;             // 1: every code point >= 128 behaves identically in this state
        int32_t hi_next = -2;            // successor on any non-ASCII code point (uniform states)
        St() { ascii.fill(-2); }
    };
    std::vector<std::unique_ptr<St>> states;
    std::unordered_map<std::string, int32_t> intern_map;
    std::unordered_map<uint64_t, bool> partial_cache;
    int32_t init_sid = 0;
    static constexpr size_t MAX_STATES = 500'000;

    // returns {matched, next position}
    std::pair<bool, uint32_t> match_char(uint32_t pos, uint32_t chr) const {
        bool found = false;
        bool positive = e[pos].t == T_CHAR || e[pos].t == T_ANY;
        do {
            if (e[pos + 1].t == T_RNG) {
                found = found || (e[pos].v <= chr && chr <= e[pos + 1].v);
                pos += 2;
            } else if (e[pos].t == T_ANY) {
                found = true;
                pos += 1;
            } else {
                found = found || e[pos].v == chr;
                pos += 1;
            }
        } while (e[pos].t == T_ALTCH);
        return {found == positive, pos};
    }
    bool match_partial(uint32_t pos, uint32_t pv, int rem) const {
        bool positive = e[pos].t == T_CHAR || e[pos].t == T_ANY;
        if (rem < 0 || (rem == 1 && pv < 2)) return false;
        uint32_t low = pv << (rem * 6);
        uint32_t high = low | ((1u << (rem * 6)) - 1);
        if (low == 0) {
            if (rem == 2) low = 1u << 11;
            else if (rem == 3) low = 1u << 16;
        }
        do {
            if (e[pos + 1].t == T_RNG) {
                if (e[pos].v <= high && low <= e[pos + 1].v) return positive;
                pos += 2;
            } else if (e[pos].t == T_ANY) {
                return true;
            } else {
                if (low <= e[pos].v && e[pos].v <= high) return positive;
                pos += 1;
            }
        } while (e[pos].t == T_ALTCH);
        return !positive;
    }

    void advance_stack(const std::vector<uint32_t>& stack, std::set<std::vector<uint32_t>>& out) const {
        std::vector<std::vector<uint32_t>> todo{stack};
        std::set<std::vector<uint32_t>> seen;
        while (!todo.empty()) {
            std::vector<uint32_t> cur = std::move(todo.back());
            todo.pop_back();
            if (!seen.insert(cur).second) continue;
            if (cur.empty()) { out.insert(cur); continue; }
            uint32_t pos = cur.back();
            switch (e[pos].t) {
                case T_REF: {
                    uint32_t sub = rule_start[e[pos].v];
                    for (;;) {
                        std::vector<uint32_t> next(cur.begin(), cur.end() - 1);
                        if (!is_eos(e[pos + 1])) next.push_back(pos + 1);
                        if (!is_eos(e[sub])) next.push_back(sub);
                        todo.push_back(std::move(next));
                        while (!is_eos(e[sub])) sub++;
                        if (e[sub].t == T_ALT) sub++;
                        else break;
                    }
                    break;
                }
                case T_CHAR: case T_NOT: case T_ANY: out.insert(cur); break;
                default: throw GrammarError("internal: stack left on a non-terminal element");
            }
        }
    }

    int32_t intern(std::set<std::vector<uint32_t>>& set) {
        std::string key;
        for (auto& s : set) {
            key.append((const char*)s.data(), s.size() * 4);
            key.push_back('\xff'); key.push_back('\xff'); key.push_back('\xff'); key.push_back('\xff');
        }
        auto it = intern_map.find(key);
        if (it != intern_map.end()) return it->second;
        if (states.size() >= MAX_STATES) throw GrammarError("grammar state explosion (too many distinct states)");
        auto st = std::make_unique<St>();
        for (auto& s : set) {
            st->stacks.push_back(s);
            if (s.empty()) st->complete = true;
        }
        int32_t id = (int32_t)states.size();
        states.push_back(std::move(st));
        intern_map.emplace(std::move(key), id);
        return id;
    }

    int32_t compute_step(int32_t sid, uint32_t cp) {
        std::set<std::vector<uint32_t>> out;
        // copy: advance_stack does not touch `states`, but intern() below may reallocate it
        const auto stacks = states[sid]->stacks;
        for (auto& stack : stacks) {
            if (stack.empty()) continue;
            uint32_t pos = stack.back();
            auto m = match_char(pos, cp);
            if (m.first) {
                std::vector<uint32_t> ns(stack.begin(), stack.end() - 1);
                if (!is_eos(e[m.second])) ns.push_back(m.second);
                advance_stack(ns, out);
            }
        }
        return intern(out);
    }
    int32_t step(int32_t sid, uint32_t cp) {
        if (sid == 0) return 0;
        St& st = *states[sid];
        if (cp < 128) {
            int32_t slot = st.ascii[cp];
            if (slot == -2) { int32_t r = compute_step(sid, cp); states[sid]->ascii[cp] = r; return r; }
            return slot;
        }
        if (st.uniform < 0) st.uniform = is_uniform(st) ? 1 : 0;
        if (st.uniform) {
            if (st.hi_next == -2) { int32_t r = compute_step(sid, 0x4E00); states[sid]->hi_next = r; return r; }
            return st.hi_next;
        }
        auto it = st.other.find(cp);
        if (it != st.other.end()) return it->second;
        int32_t r = compute_step(sid, cp);
        states[sid]->other[cp] = r;
        return r;
    }
    // no element on top of any stack mentions a code point >= 128 -> all non-ASCII code points are treated alike
    bool is_uniform(const St& st) const {
        for (auto& stack : st.stacks) {
            if (stack.empty()) continue;
            uint32_t pos = stack.back();
            do {
                if (e[pos].v >= 128 && e[pos].t != T_ANY) return false;
                pos += e[pos + 1].t == T_RNG ? 2 : 1;
                if (e[pos - 1].t == T_RNG && e[pos - 1].v >= 128) return false;
            } while (e[pos].t == T_ALTCH);
        }
        return true;
    }
    bool partial_ok(int32_t sid, uint32_t pv, int rem) {
        {
            St& st = *states[sid];
            if (st.uniform < 0) st.uniform = is_uniform(st) ? 1 : 0;
            if (st.uniform) {
                if (rem == 1 && pv < 2) return false;
                if (st.hi_next == -2) st.hi_next = compute_step(sid, 0x4E00);
                return st.hi_next != 0;
            }
        }
        uint64_t key = ((uint64_t)sid << 32) | (pv << 2) | (uint32_t)rem;
        auto it = partial_cache.find(key);
        if (it != partial_cache.end()) return it->second;
        bool ok = false;
        for (auto& stack : states[sid]->stacks) {
            if (stack.empty()) continue;
            if (match_partial(stack.back(), pv, rem)) { ok = true; break; }
        }
        partial_cache[key] = ok;
        return ok;
    }
};

// ============================================================================ Grammar
Grammar Grammar::parse(std::string_view gbnf, std::string_view root) {
    std::string text(gbnf);
    Parser p;
    const char* pos = parse_space(text.c_str(), true);
    while (*pos) pos = p.parse_rule(pos);
    for (size_t i = 0; i < p.rules.size(); ++i) {
        const Rule& rule = p.rules[i];
        if (rule.empty()) {
            std::string nm;
            for (auto& kv : p.symbol_ids) if (kv.second == i) nm = kv.first;
            throw GrammarError("Undefined rule identifier '" + nm + "'");
        }
        for (auto& el : rule)
            if (el.t == T_REF && (el.v >= p.rules.size() || p.rules[el.v].empty())) {
                for (auto& kv : p.symbol_ids) if (kv.second == el.v) throw GrammarError("Undefined rule identifier '" + kv.first + "'");
                throw GrammarError("Undefined rule");
            }
    }
    auto it = p.symbol_ids.find(std::string(root));
    if (it == p.symbol_ids.end()) throw GrammarError("grammar does not define the root symbol '" + std::string(root) + "'");
    {
        std::vector<bool> visited(p.rules.size()), in_progress(p.rules.size()), may_be_empty(p.rules.size());
        for (size_t i = 0; i < p.rules.size(); i++)
            if (!visited[i] && detect_left_recursion(p.rules, i, visited, in_progress, may_be_empty))
                throw GrammarError("unsupported grammar, left recursion detected for nonterminal at index " + std::to_string(i));
    }
    auto impl = std::make_shared<Impl>();
    impl->names.resize(p.rules.size());
    for (auto& kv : p.symbol_ids) impl->names[kv.second] = kv.first;
    for (auto& r : p.rules) {
        impl->rule_start.push_back((uint32_t)impl->e.size());
        impl->e.insert(impl->e.end(), r.begin(), r.end());
    }
    impl->e.push_back({T_END, 0});  // sentinel so e[pos + 1] is always valid
    impl->root = it->second;
    // initial state: expand the alternatives of the root rule
    std::set<std::vector<uint32_t>> init;
    {
        uint32_t sub = impl->rule_start[impl->root];
        for (;;) {
            std::vector<uint32_t> st;
            if (!is_eos(impl->e[sub])) st.push_back(sub);
            impl->advance_stack(st, init);
            while (!is_eos(impl->e[sub])) sub++;
            if (impl->e[sub].t == T_ALT) sub++;
            else break;
        }
    }
    impl->states.push_back(std::make_unique<Impl::St>());  // sid 0 = dead
    impl->intern_map.emplace(std::string(), 0);
    impl->init_sid = impl->intern(init);
    Grammar g;
    g.impl_ = impl;
    return g;
}

size_t Grammar::n_rules() const { return impl_ ? impl_->rule_start.size() : 0; }

std::string Grammar::dump() const {
    if (!impl_) return "";
    const Impl& m = *impl_;
    std::string out;
    std::vector<std::string> nm(m.names);
    for (auto& n : nm) for (auto& ch : n) if (ch == '_') ch = '-';  // generated sub-rule names must be re-parseable
    auto lit = [](uint32_t c) {
        std::string s;
        if (c == '"' || c == '\\') { s += '\\'; s += (char)c; }
        else if (c == '\n') s += "\\n";
        else if (c == '\r') s += "\\r";
        else if (c == '\t') s += "\\t";
        else if (c < 0x20 || c == 0x7f) { char b[8]; snprintf(b, sizeof b, "\\x%02X", c); s += b; }
        else if (c < 0x80) s += (char)c;
        else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 63)); }
        else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
        else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 63)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
        return s;
    };
    auto cls = [&](uint32_t c) {
        std::string s;
        if (c == ']' || c == '[' || c == '\\' || c == '-' || c == '^') { s += '\\'; s += (char)c; }
        else s = lit(c);
        if (c == '"') s = "\"";
        return s;
    };
    for (size_t r = 0; r < m.rule_start.size(); ++r) {
        out += nm[r] + " ::= ";
        uint32_t p = m.rule_start[r];
        bool in_lit = false, seq_empty = true;
        auto close_lit = [&] { if (in_lit) { out += "\" "; in_lit = false; } };
        for (;; ++p) {
            const Elem& el = m.e[p];
            if (el.t == T_END) { close_lit(); if (seq_empty) out += "\"\" "; break; }
            if (el.t == T_ALT) { close_lit(); if (seq_empty) out += "\"\" "; out += "| "; seq_empty = true; continue; }
            seq_empty = false;
            if (el.t == T_CHAR && m.e[p + 1].t != T_RNG && m.e[p + 1].t != T_ALTCH) {
                if (!in_lit) { out += "\""; in_lit = true; }
                out += lit(el.v);
                continue;
            }
            close_lit();
            if (el.t == T_REF) out += nm[el.v] + " ";
            else if (el.t == T_ANY) out += ". ";
            else if (el.t == T_CHAR || el.t == T_NOT) {
                out += el.t == T_NOT ? "[^" : "[";
                uint32_t q = p;
                for (;;) {
                    out += cls(m.e[q].v);
                    if (m.e[q + 1].t == T_RNG) { out += "-" + cls(m.e[q + 1].v); q += 2; } else q += 1;
                    if (m.e[q].t != T_ALTCH) break;
                }
                out += "] ";
                p = q - 1;
            }
        }
        out += "\n";
    }
    return out;
}

// ============================================================================ VocabIndex
std::shared_ptr<const VocabIndex> VocabIndex::build(const std::vector<std::string>& pieces, const std::vector<uint8_t>& eog) {
    auto vi = std::make_shared<VocabIndex>();
    vi->pieces_ = pieces;
    vi->eog_ = eog;
    vi->eog_.resize(pieces.size(), 0);
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < pieces.size(); ++i) {
        if (vi->eog_[i]) vi->eog_ids.push_back(i);
        else if (!pieces[i].empty() && pieces[i][0] != 0) ids.push_back(i);
    }
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return pieces[a] < pieces[b]; });
    vi->sorted_ids = ids;
    vi->nodes.push_back({0, 0, 0, 0});
    std::vector<uint32_t> path{0};
    const std::string* prev = nullptr;
    static const std::string empty;
    prev = &empty;
    for (size_t k = 0; k < ids.size(); ++k) {
        const std::string& s = pieces[ids[k]];
        size_t l = 0;
        while (l < s.size() && l < prev->size() && s[l] == (*prev)[l]) ++l;
        while (path.size() - 1 > l) {
            vi->nodes[path.back()].end = (uint32_t)vi->nodes.size();
            path.pop_back();
        }
        for (size_t d = l; d < s.size(); ++d) {
            path.push_back((uint32_t)vi->nodes.size());
            vi->nodes.push_back({0, 0, 0, (uint8_t)s[d]});
        }
        Node& n = vi->nodes[path.back()];
        if (n.tok_cnt == 0) n.tok_begin = (uint32_t)k;
        n.tok_cnt++;
        vi->max_depth = std::max<uint32_t>(vi->max_depth, (uint32_t)s.size());
        prev = &s;
    }
    while (!path.empty()) {
        vi->nodes[path.back()].end = (uint32_t)vi->nodes.size();
        path.pop_back();
    }
    return vi;
}

// ============================================================================ Matcher (mask cache)
struct Matcher::Cache {
    std::mutex mu;
    struct Entry {
        std::shared_ptr<const TokenMask> mask;
        uint64_t stamp;
    };
    std::unordered_map<uint64_t, Entry> map;
    uint64_t clock = 0;
    size_t bytes = 0;
    uint64_t hits = 0, misses = 0, compute_us = 0;
    std::shared_ptr<const TokenMask> all_allowed;
};

Matcher::Matcher(Grammar g, std::shared_ptr<const VocabIndex> vocab, MatcherOptions opt)
    : grammar_(std::move(g)), vocab_(std::move(vocab)), opt_(opt), cache_(std::make_shared<Cache>()) {
    if (!grammar_.valid()) throw GrammarError("Matcher: invalid grammar");
}

Matcher::Stats Matcher::stats() const {
    Stats s;
    std::lock_guard<std::mutex> lk(cache_->mu);
    s.mask_hits = cache_->hits;
    s.mask_misses = cache_->misses;
    s.mask_compute_us = cache_->compute_us;
    s.cached_masks = cache_->map.size();
    s.cache_bytes = cache_->bytes;
    std::lock_guard<std::mutex> lk2(grammar_.impl()->mu);
    s.states = grammar_.impl()->states.size();
    return s;
}

namespace {
struct DfsState {
    int32_t sid;
    uint32_t pv;
    int8_t rem;
};

std::shared_ptr<TokenMask> compute_mask(Impl& g, const VocabIndex& vi, int32_t sid, uint32_t pv, int rem) {
    auto mask = std::make_shared<TokenMask>();
    mask->n = vi.size();
    mask->bits.assign((vi.size() + 63) / 64, 0);
    auto set = [&](uint32_t id) {
        uint64_t& w = mask->bits[id >> 6];
        uint64_t b = 1ull << (id & 63);
        if (!(w & b)) { w |= b; mask->n_allowed++; }
    };
    const auto& nodes = vi.nodes;
    const uint32_t N = (uint32_t)nodes.size();
    std::vector<uint32_t> ends(vi.max_depth + 2);
    std::vector<DfsState> sts(vi.max_depth + 2);
    int top = 0;
    ends[0] = N;
    sts[0] = {sid, pv, (int8_t)rem};
    for (uint32_t i = 1; i < N;) {
        while (ends[top] <= i) --top;
        const auto& nd = nodes[i];
        const DfsState& s = sts[top];
        uint8_t b = nd.byte;
        DfsState ns;
        bool ok = true;
        if (s.rem == 0) {
            if (b < 0x80) {
                int32_t r = g.step(s.sid, b);
                ok = r != 0;
                ns = {r, 0, 0};
            } else {
                int n = (b >> 4) == 0xC || (b >> 4) == 0xD ? 1 : (b >> 4) == 0xE ? 2 : (b >> 4) == 0xF ? 3 : -1;
                if (n < 0 || b == 0xC0 || b == 0xC1) ok = false;
                else {
                    uint32_t v = b & ((1u << (7 - n)) - 1);
                    ok = g.partial_ok(s.sid, v, n);
                    ns = {s.sid, v, (int8_t)n};
                }
            }
        } else {
            if ((b >> 6) != 2) ok = false;
            else {
                uint32_t v = (s.pv << 6) | (b & 0x3F);
                int rem2 = s.rem - 1;
                if (rem2 == 0) {
                    int32_t r = g.step(s.sid, v);
                    ok = r != 0;
                    ns = {r, 0, 0};
                } else {
                    ok = g.partial_ok(s.sid, v, rem2);
                    ns = {s.sid, v, (int8_t)rem2};
                }
            }
        }
        if (!ok) { i = nd.end; continue; }
        for (uint32_t k = 0; k < nd.tok_cnt; ++k) set(vi.sorted_ids[nd.tok_begin + k]);
        if (nd.end > i + 1) {
            ++top;
            sts[top] = ns;
            ends[top] = nd.end;
        }
        ++i;
    }
    if (rem == 0 && g.states[sid]->complete)
        for (uint32_t id : vi.eog_ids) set(id);
    return mask;
}
}  // namespace

// ============================================================================ GrammarState
GrammarState::GrammarState(std::shared_ptr<Matcher> m, Triggers lazy) : m_(std::move(m)), trig_(std::move(lazy)) {
    impl_ = m_->grammar().impl();
    sid_ = impl_->init_sid;
    awaiting_ = !trig_.empty();
}
GrammarState::GrammarState(const Grammar& g) {
    impl_ = g.impl();
    if (!impl_) throw GrammarError("invalid grammar");
    sid_ = impl_->init_sid;
}

bool GrammarState::is_dead() const { return dead_ || sid_ == 0; }

bool GrammarState::is_complete() const {
    if (awaiting_) return true;
    if (is_dead() || rem_ != 0) return false;
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->states[sid_]->complete;
}

bool GrammarState::accept(const std::vector<uint32_t>& cps) {
    if (awaiting_) {
        std::string s;
        for (uint32_t c : cps) {
            if (c < 0x80) s += (char)c;
            else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 63)); }
            else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
            else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 63)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
        }
        return accept(s);
    }
    if (dead_) return false;
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (uint32_t c : cps) {
        sid_ = impl_->step(sid_, c);
        if (sid_ == 0) { dead_ = true; return false; }
    }
    return true;
}

bool GrammarState::feed(std::string_view s) {
    if (dead_) return false;
    static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (unsigned char b : s) {
        uint32_t cp;
        if (rem_ > 0) {
            if ((b >> 6) != 2) { dead_ = true; return false; }
            pv_ = (pv_ << 6) | (b & 0x3F);
            if (--rem_ > 0) continue;
            cp = pv_;
            pv_ = 0;
        } else {
            int n = lookup[b >> 4] - 1;
            if (n < 0) { dead_ = true; return false; }
            if (n == 0) cp = b;
            else {
                pv_ = b & ((1u << (7 - n)) - 1);
                rem_ = (int8_t)n;
                continue;
            }
        }
        sid_ = impl_->step(sid_, cp);
        if (sid_ == 0) { dead_ = true; return false; }
    }
    // a trailing partial sequence must still be completable
    if (rem_ > 0 && !impl_->partial_ok(sid_, pv_, rem_)) { dead_ = true; return false; }
    return true;
}

bool GrammarState::accept(std::string_view piece) {
    if (awaiting_) {
        size_t max_w = 0;
        for (const auto& w : trig_.words) max_w = std::max(max_w, w.size());
        const size_t beg = trig_buf_.size();
        trig_buf_.append(piece);
        const size_t search_from = (beg > max_w) ? (beg - max_w) : 0;
        for (const auto& w : trig_.words) {
            size_t f = trig_buf_.find(w, search_from);
            if (f != std::string::npos) {
                awaiting_ = false;
                std::string replay = trig_buf_.substr(f);
                trig_buf_.clear();
                trig_buf_.shrink_to_fit();
                return feed(replay);
            }
        }
        if (max_w > 1 && trig_buf_.size() >= max_w) {
            trig_buf_.erase(0, trig_buf_.size() - (max_w - 1));
        } else if (max_w <= 1) {
            trig_buf_.clear();
        }
        return true;
    }
    return feed(piece);
}

bool GrammarState::accept_token(int id) {
    if (!m_ || !m_->vocab()) throw GrammarError("accept_token needs a vocabulary");
    const auto& vi = *m_->vocab();
    if (id < 0 || (size_t)id >= vi.size()) return false;
    if (awaiting_) {
        if (std::find(trig_.tokens.begin(), trig_.tokens.end(), id) != trig_.tokens.end()) {
            awaiting_ = false;
            trig_buf_.clear();
            return feed(vi.piece(id));
        }
        return accept(std::string_view(vi.piece(id)));
    }
    if (vi.is_eog(id)) return is_complete();
    return feed(vi.piece(id));
}

std::shared_ptr<const TokenMask> GrammarState::allowed() {
    if (!m_ || !m_->vocab()) throw GrammarError("allowed() needs a vocabulary");
    Matcher::Cache& c = *m_->cache_;
    const VocabIndex& vi = *m_->vocab();
    if (awaiting_) {
        std::lock_guard<std::mutex> lk(c.mu);
        if (!c.all_allowed) {
            auto all = std::make_shared<TokenMask>();
            all->n = vi.size();
            all->bits.assign((vi.size() + 63) / 64, ~0ull);
            all->n_allowed = (uint32_t)vi.size();
            c.all_allowed = all;
        }
        return c.all_allowed;
    }
    if (dead_ || sid_ == 0) {
        auto none = std::make_shared<TokenMask>();
        none->n = vi.size();
        none->bits.assign((vi.size() + 63) / 64, 0);
        return none;
    }
    uint64_t key = ((uint64_t)sid_ << 32) | (pv_ << 2) | (uint32_t)rem_;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        auto it = c.map.find(key);
        if (it != c.map.end()) {
            it->second.stamp = ++c.clock;
            ++c.hits;
            return it->second.mask;
        }
    }
    auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<TokenMask> mask;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        mask = compute_mask(*impl_, vi, sid_, pv_, rem_);
    }
    uint64_t us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(c.mu);
    ++c.misses;
    c.compute_us += us;
    size_t sz = mask->bits.size() * 8 + 64;
    c.map[key] = {mask, ++c.clock};
    c.bytes += sz;
    if (c.bytes > m_->opt_.max_cache_bytes) {  // evict the oldest quarter... until under 75 %
        std::vector<std::pair<uint64_t, uint64_t>> order;
        order.reserve(c.map.size());
        for (auto& kv : c.map) order.emplace_back(kv.second.stamp, kv.first);
        std::sort(order.begin(), order.end());
        for (auto& o : order) {
            if (c.bytes <= m_->opt_.max_cache_bytes * 3 / 4) break;
            if (o.second == key) continue;
            c.bytes -= c.map[o.second].mask->bits.size() * 8 + 64;
            c.map.erase(o.second);
        }
    }
    return mask;
}

bool GrammarState::token_allowed(int id) {
    if (id < 0) return false;
    return allowed()->allowed((size_t)id);
}

void GrammarState::mask_logits(float* logits, size_t n_vocab) {
    if (awaiting_) return;
    auto mask = allowed();
    const float ninf = -std::numeric_limits<float>::infinity();
    size_t n = std::min(n_vocab, mask->n);
    if (mask->n_allowed * 16 < n) {
        // sparse: remember the allowed logits, wipe everything, restore
        std::vector<std::pair<uint32_t, float>> keep;
        keep.reserve(mask->n_allowed);
        for (size_t w = 0; w < mask->bits.size(); ++w) {
            uint64_t bits = mask->bits[w];
            while (bits) {
                uint32_t id = (uint32_t)(w * 64 + __builtin_ctzll(bits));
                bits &= bits - 1;
                if (id < n) keep.emplace_back(id, logits[id]);
            }
        }
        std::fill(logits, logits + n_vocab, ninf);
        for (auto& k : keep) logits[k.first] = k.second;
        return;
    }
    for (size_t w = 0; w < mask->bits.size(); ++w) {
        uint64_t bad = ~mask->bits[w];
        if (!bad) continue;
        size_t base = w * 64;
        if (base >= n) break;
        while (bad) {
            size_t id = base + __builtin_ctzll(bad);
            bad &= bad - 1;
            if (id < n) logits[id] = ninf;
        }
    }
    for (size_t id = mask->n; id < n_vocab; ++id) logits[id] = ninf;
}

}  // namespace qv::grammar
