// Unit tests for src/grammar: GBNF parser/matcher, JSON-schema conversion, tool grammars, trie masks vs brute force (real Qwen3.8 tokenizer),
// and mask latency measurements.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>

#include "grammar/grammar.h"
#include "grammar/json_schema.h"
#include "grammar/sampler_hook.h"

using namespace qv::grammar;
namespace jd = qv::jd;

static int g_fail = 0, g_n = 0;
#define CHECK(c) do { ++g_n; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CHECKM(c, m) do { ++g_n; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d: %s  [%s]\n", __FILE__, __LINE__, #c, std::string(m).c_str()); } } while (0)

static bool matches(const Grammar& g, const std::string& s) {
    GrammarState st(g);
    return st.accept(s) && st.is_complete();
}
static bool prefix_ok(const Grammar& g, const std::string& s) {
    GrammarState st(g);
    return st.accept(s);
}
static bool gm(const std::string& gbnf, const std::string& s) { return matches(Grammar::parse(gbnf), s); }
static bool throws_parse(const std::string& gbnf) {
    try { Grammar::parse(gbnf); } catch (const GrammarError&) { return true; }
    return false;
}
static jd::Value J(const char* s) { return jd::parse(s); }
static Grammar schema_g(const char* schema, SchemaOptions o = {}) { return Grammar::parse(json_schema_to_grammar(J(schema), o)); }
static bool sm(const Grammar& g, const char* s) { return matches(g, s); }

static void test_parser() {
    CHECK(gm("root ::= \"ab\" | \"cd\"", "ab"));
    CHECK(gm("root ::= \"ab\" | \"cd\"", "cd"));
    CHECK(!gm("root ::= \"ab\" | \"cd\"", "a"));
    CHECK(!gm("root ::= \"ab\" | \"cd\"", "abc"));
    CHECK(prefix_ok(Grammar::parse("root ::= \"abc\""), "ab"));
    CHECK(!prefix_ok(Grammar::parse("root ::= \"abc\""), "ac"));
    CHECK(gm("root ::= [a-c]+ \"!\"", "abcba!"));
    CHECK(!gm("root ::= [a-c]+ \"!\"", "!"));
    CHECK(gm("root ::= [^a-c]* ", "xyz"));
    CHECK(gm("root ::= [^a-c]* ", ""));
    CHECK(!gm("root ::= [^a-c]* ", "xbz"));
    CHECK(gm("root ::= \"a\"{2,3}", "aa"));
    CHECK(gm("root ::= \"a\"{2,3}", "aaa"));
    CHECK(!gm("root ::= \"a\"{2,3}", "a"));
    CHECK(!gm("root ::= \"a\"{2,3}", "aaaa"));
    CHECK(gm("root ::= \"a\"{2,}", "aaaaaaa"));
    CHECK(gm("root ::= \"a\"{3}", "aaa"));
    CHECK(!gm("root ::= \"a\"{3}", "aa"));
    CHECK(gm("root ::= (\"ab\" | \"c\")? \"d\"", "d"));
    CHECK(gm("root ::= (\"ab\" | \"c\")? \"d\"", "abd"));
    CHECK(gm("root ::= (\"ab\" | \"c\")? \"d\"", "cd"));
    CHECK(gm("root ::= . .", "xy"));
    CHECK(gm("root ::= .", "\xC3\xA9"));  // any code point
    CHECK(gm("root ::= [\\u00e9-\\u00ff]", "\xC3\xA9"));
    CHECK(!gm("root ::= [\\u00e9-\\u00ff]", "e"));
    CHECK(gm("root ::= \"\\x41\\u0042\\U00000043\"", "ABC"));
    CHECK(gm("root ::= \"\xE2\x82\xAC\"", "\xE2\x82\xAC"));
    CHECK(gm("# comment\nroot ::= a b # trailing\na ::= \"x\"\nb ::= \"y\"\n", "xy"));
    CHECK(gm("root ::= item (\",\" item)*\nitem ::= [0-9]+", "1,22,333"));
    CHECK(gm("root ::= (\n  \"a\"\n  | \"b\"\n)\n", "b"));
    CHECK(gm("root ::= \"a\\n\\t\\\"\"", "a\n\t\""));
    // recursion (right) is fine
    CHECK(gm("root ::= \"(\" root \")\" | \"x\"", "((x))"));
    CHECK(!gm("root ::= \"(\" root \")\" | \"x\"", "((x)"));
    // errors
    CHECK(throws_parse("root ::= root \"a\""));
    CHECK(throws_parse("root ::= a"));
    CHECK(throws_parse("foo ::= \"a\""));
    CHECK(throws_parse("root ::= \"a"));
    CHECK(throws_parse("root ::= [a-"));
    CHECK(throws_parse("root ::= *"));
    CHECK(throws_parse("root ::= (\"a\""));
    CHECK(throws_parse("root ::= \"a\"{3000}"));
    CHECK(throws_parse("root ::= a\na ::= root \"x\" | \"y\"\n"));
    // dump round trip behaves identically
    auto g = Grammar::parse("root ::= \"ab\"? [0-9]{2,4} (\"x\" | [^a-z])\n");
    auto g2 = Grammar::parse(g.dump());
    for (const char* s : {"12x", "ab123!", "1", "ab12345x", "123", "ab1234Z"}) CHECK(matches(g, s) == matches(g2, s));
    // dead state, completeness
    GrammarState st(Grammar::parse("root ::= \"ab\""));
    CHECK(!st.is_complete());
    CHECK(st.accept("a"));
    CHECK(!st.is_complete());
    CHECK(st.accept("b"));
    CHECK(st.is_complete());
    CHECK(!st.accept("c"));
    CHECK(st.is_dead());
    // partial UTF-8 across accept() calls
    GrammarState u(Grammar::parse("root ::= \"\xE2\x82\xAC\""));
    CHECK(u.accept("\xE2"));
    CHECK(!u.is_complete());
    CHECK(u.accept("\x82"));
    CHECK(u.accept("\xAC"));
    CHECK(u.is_complete());
    GrammarState u2(Grammar::parse("root ::= \"a\""));
    CHECK(!u2.accept("\xE2"));  // no stack can take a 3-byte char
    // code point accept
    GrammarState cp(Grammar::parse("root ::= \"\xE2\x82\xAC\""));
    CHECK(cp.accept(std::vector<uint32_t>{0x20AC}) && cp.is_complete());
}

static const char* kObj = R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}},"required":["name","age"]})";

static void test_schema() {
    auto g = schema_g(kObj);
    CHECK(sm(g, R"({"name": "bob", "age": 3})"));
    CHECK(sm(g, R"({"name":"bob","age":-3,"tags":["a","b"]})"));
    CHECK(sm(g, R"({"name":"b\"ob","age":3,"tags":[]})"));
    CHECK(!sm(g, R"({"age":3})"));
    CHECK(!sm(g, R"({"name":"bob","age":"3"})"));
    CHECK(!sm(g, R"({"name":"bob","age":3.5})"));
    CHECK(!sm(g, R"({"name":"bob","age":3,"extra":1})"));  // additionalProperties unspecified -> closed in llama.cpp when properties given
    CHECK(!sm(g, R"({"age":3,"name":"x"})"));              // fixed property order
    // enum / const
    auto e = schema_g(R"({"enum":["red",1,null,true]})");
    for (const char* s : {"\"red\"", "1", "null", "true"}) CHECK(sm(e, s));
    CHECK(!sm(e, "\"blue\""));
    auto c = schema_g(R"({"const":{"a":1}})");
    CHECK(sm(c, "{\"a\":1}"));
    CHECK(!sm(c, "{\"a\":2}"));
    // anyOf / oneOf / type list
    auto u = schema_g(R"({"anyOf":[{"type":"integer"},{"type":"string"},{"type":"null"}]})");
    CHECK(sm(u, "12") && sm(u, "\"x\"") && sm(u, "null") && !sm(u, "true"));
    auto u2 = schema_g(R"({"oneOf":[{"type":"boolean"},{"type":"number"}]})");
    CHECK(sm(u2, "true") && sm(u2, "-1.5e3") && !sm(u2, "\"s\""));
    auto tl = schema_g(R"({"type":["string","null"]})");
    CHECK(sm(tl, "null") && sm(tl, "\"a\"") && !sm(tl, "1"));
    // $ref / definitions / $defs
    auto r = schema_g(R"({"$ref":"#/definitions/node","definitions":{"node":{"type":"object","properties":{"v":{"type":"integer"},"next":{"$ref":"#/definitions/node"}},"required":["v"]}}})");
    CHECK(sm(r, R"({"v":1})"));
    CHECK(sm(r, R"({"v":1,"next":{"v":2,"next":{"v":3}}})"));
    CHECK(!sm(r, R"({"v":1,"next":{}})"));
    auto r2 = schema_g(R"({"type":"object","properties":{"a":{"$ref":"#/$defs/x"}},"required":["a"],"$defs":{"x":{"enum":["p","q"]}}})");
    CHECK(sm(r2, R"({"a":"q"})") && !sm(r2, R"({"a":"z"})"));
    // optional properties (all optional)
    auto o = schema_g(R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}}})");
    for (const char* s : {"{}", "{\"a\":1}", "{\"b\":1}", "{\"c\":1}", "{\"a\":1,\"c\":2}", "{\"a\":1,\"b\":2,\"c\":3}", "{\"b\":2,\"c\":3}"}) CHECKM(sm(o, s), s);
    CHECK(!sm(o, "{\"a\":1,\"a\":2}") && !sm(o, "{\"c\":1,\"a\":1}") && !sm(o, "{\"a\":1,}"));
    // required + optional mix
    auto m = schema_g(R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},"required":["b"]})");
    for (const char* s : {"{\"b\":1}", "{\"b\":1,\"a\":1}", "{\"b\":1,\"c\":1}", "{\"b\":1,\"a\":0,\"c\":2}"}) CHECKM(sm(m, s), s);
    CHECK(!sm(m, "{\"a\":1}") && !sm(m, "{}"));
    // additionalProperties
    auto ap = schema_g(R"({"type":"object","additionalProperties":{"type":"integer"}})");
    CHECK(sm(ap, "{}") && sm(ap, R"({"x":1,"y":2})") && !sm(ap, R"({"x":"s"})"));
    auto ap2 = schema_g(R"({"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":true})");
    CHECK(sm(ap2, R"({"id":1,"foo":[1,{"a":null}],"bar":"x"})") && sm(ap2, R"({"id":1})") && !sm(ap2, R"({"foo":1})"));
    // additionalProperties false with no properties = empty object only
    auto ap3 = schema_g(R"({"type":"object","additionalProperties":false})");
    CHECK(sm(ap3, "{}"));
    // arrays
    auto a = schema_g(R"({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3})");
    CHECK(!sm(a, "[]") && !sm(a, "[1]") && sm(a, "[1,2]") && sm(a, "[1, 2, 3]") && !sm(a, "[1,2,3,4]"));
    auto a0 = schema_g(R"({"type":"array","items":{"type":"string"},"minItems":0,"maxItems":2})");
    CHECK(sm(a0, "[]") && sm(a0, "[\"a\"]") && sm(a0, "[\"a\",\"b\"]") && !sm(a0, "[\"a\",\"b\",\"c\"]"));
    auto at = schema_g(R"({"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}]})");
    CHECK(sm(at, "[1,\"a\"]") && !sm(at, "[\"a\",1]"));
    // strings
    auto s = schema_g(R"({"type":"string","minLength":2,"maxLength":4})");
    CHECK(!sm(s, "\"a\"") && sm(s, "\"ab\"") && sm(s, "\"abcd\"") && !sm(s, "\"abcde\"") && sm(s, "\"\xC3\xA9\xC3\xA9\""));
    CHECK(sm(s, "\"a\\n\"") && !sm(s, "\"a\nb\""));
    auto d = schema_g(R"({"type":"string","format":"date"})");
    CHECK(sm(d, "\"2024-02-29\"") && !sm(d, "\"2024-13-01\"") && !sm(d, "\"24-1-1\""));
    auto dt = schema_g(R"({"type":"string","format":"date-time"})");
    CHECK(sm(dt, "\"2024-02-29T12:30:00Z\"") && sm(dt, "\"2024-02-29T12:30:00.123+01:00\"") && !sm(dt, "\"2024-02-29\""));
    auto uu = schema_g(R"({"type":"string","format":"uuid"})");
    CHECK(sm(uu, "\"123e4567-e89b-12d3-a456-426614174000\"") && !sm(uu, "\"123\""));
    // patterns
    auto p = schema_g(R"({"type":"string","pattern":"^[a-z]{2}-[0-9]+$"})");
    CHECK(sm(p, "\"ab-12\"") && !sm(p, "\"a-12\"") && !sm(p, "\"ab-\"") && !sm(p, "\"AB-1\""));
    auto p2 = schema_g(R"({"type":"string","pattern":"^(foo|bar)baz?$"})");
    CHECK(sm(p2, "\"foobaz\"") && sm(p2, "\"barba\"") && !sm(p2, "\"bazz\""));
    auto p3 = schema_g(R"({"type":"string","pattern":"abc"})");  // unanchored -> any string
    CHECK(sm(p3, "\"zzz\""));
    // integers with ranges
    auto ir = schema_g(R"({"type":"integer","minimum":-5,"maximum":120})");
    for (int v : {-6, -5, -1, 0, 1, 9, 10, 99, 100, 120, 121, 1000, 200}) {
        bool ok = v >= -5 && v <= 120;
        CHECKM(sm(ir, std::to_string(v).c_str()) == ok, "int range " + std::to_string(v));
    }
    CHECK(!sm(ir, "007"));
    auto imin = schema_g(R"({"type":"integer","minimum":18})");
    for (int v : {0, 17, 18, 19, 99, 100, 123456, -1}) CHECKM(sm(imin, std::to_string(v).c_str()) == (v >= 18), "int min " + std::to_string(v));
    auto imax = schema_g(R"({"type":"integer","maximum":42})");
    for (int v : {-1000, -1, 0, 41, 42, 43, 100}) CHECKM(sm(imax, std::to_string(v).c_str()) == (v <= 42), "int max " + std::to_string(v));
    auto iex = schema_g(R"({"type":"integer","exclusiveMinimum":0,"exclusiveMaximum":10})");
    for (int v : {0, 1, 9, 10}) CHECKM(sm(iex, std::to_string(v).c_str()) == (v >= 1 && v <= 9), "int excl " + std::to_string(v));
    // numbers / booleans / null / any
    auto n = schema_g(R"({"type":"number"})");
    CHECK(sm(n, "0") && sm(n, "-12.5") && sm(n, "1e10") && sm(n, "1.5E-3") && !sm(n, "01") && !sm(n, ".5") && !sm(n, "1."));
    auto any = schema_g("{}");
    CHECK(sm(any, "{\"a\":[1,2,{\"b\":null}],\"c\":\"x\"}") && sm(any, "[true,false]") && sm(any, "3"));
    // allOf
    auto al = schema_g(R"({"allOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},{"type":"object","properties":{"b":{"type":"string"}},"required":["b"]}]})");
    CHECK(sm(al, R"({"a":1,"b":"x"})") && !sm(al, R"({"a":1})"));
    // compact mode forbids whitespace
    SchemaOptions co;
    co.compact = true;
    auto cg = schema_g(R"({"type":"array","items":{"type":"integer"}})", co);
    CHECK(sm(cg, "[1,2]") && !sm(cg, "[1, 2]"));
    // errors
    bool thrown = false;
    try { json_schema_to_grammar(J(R"({"type":"widget"})")); } catch (const std::invalid_argument&) { thrown = true; }
    CHECK(thrown);
    thrown = false;
    try { json_schema_to_grammar(J(R"({"$ref":"#/definitions/missing"})")); } catch (const std::invalid_argument&) { thrown = true; }
    CHECK(thrown);
    // json_object builtin
    auto jo = Grammar::parse(json_object_grammar());
    CHECK(sm(jo, "{}") && sm(jo, R"({"a": [1, 2.5, "x", null, true], "b": {"c": {}}})") && !sm(jo, "[1]") && !sm(jo, "{\"a\":}") && !sm(jo, "{a:1}"));
}

static void test_tools() {
    const char* tools = R"([
      {"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["c","f"]},"days":{"type":"integer"}},"required":["city"]}}},
      {"type":"function","function":{"name":"noop","parameters":{"type":"object","properties":{}}}}])";
    jd::Value t = J(tools);
    auto mk = [&](qv::ToolFormat f, const std::string& forced = "", bool par = false) {
        ToolGrammarOptions o;
        o.format = f;
        o.forced_name = forced;
        o.parallel = par;
        return tool_call_grammar(t, o);
    };
    {
        auto tg = mk(qv::ToolFormat::Hermes);
        auto g = Grammar::parse(tg.gbnf);
        CHECK(matches(g, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>"));
        CHECK(matches(g, "<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>"));
        CHECK(!matches(g, "<tool_call>\n{\"name\": \"other\", \"arguments\": {}}\n</tool_call>"));
        CHECK(!matches(g, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"days\": 3}}\n</tool_call>"));
        CHECK(!matches(g, "hello"));
        CHECK(tg.trigger_words.size() == 1 && tg.trigger_words[0] == "<tool_call>");
        auto gl = Grammar::parse(tg.gbnf_lazy);
        CHECK(matches(gl, "<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>"));
        auto gf = Grammar::parse(mk(qv::ToolFormat::Hermes, "noop").gbnf);
        CHECK(matches(gf, "<tool_call>\n{\"name\":\"noop\",\"arguments\":{}}\n</tool_call>") && !matches(gf, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"x\"}}\n</tool_call>"));
        auto gp = Grammar::parse(mk(qv::ToolFormat::Hermes, "", true).gbnf);
        CHECK(matches(gp, "<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>"));
        CHECK(!matches(g, "<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>"));
        ToolGrammarOptions o;
        o.format = qv::ToolFormat::Hermes;
        o.thinking_forced_open = true;
        auto gt = Grammar::parse(tool_call_grammar(t, o).gbnf);
        CHECK(matches(gt, "I should call it.\n</think>\n\n<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>"));
        CHECK(!matches(gt, "<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>"));
    }
    {
        auto g = Grammar::parse(mk(qv::ToolFormat::QwenXml).gbnf);
        CHECK(matches(g, "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\nCity\n</parameter>\n<parameter=unit>\nc\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"));
        CHECK(matches(g, "<tool_call>\n<function=get_weather>\n<parameter=city>\nx\n</parameter>\n</function>\n</tool_call>"));
        CHECK(matches(g, "<tool_call>\n<function=noop>\n</function>\n</tool_call>"));
        CHECK(!matches(g, "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"));
        CHECK(!matches(g, "<tool_call>\n<function=get_weather>\n<parameter=city>\nx\n</parameter>\n<parameter=unit>\nz\n</parameter>\n</function>\n</tool_call>"));
        CHECK(!matches(g, "<tool_call>\n<function=get_weather>\n<parameter=city>\nx\n</parameter>\n<parameter=days>\nabc\n</parameter>\n</function>\n</tool_call>"));
    }
    {
        auto g = Grammar::parse(mk(qv::ToolFormat::Mistral).gbnf);
        CHECK(matches(g, "[TOOL_CALLS][{\"name\": \"noop\", \"arguments\": {}}]"));
        CHECK(!matches(g, "[TOOL_CALLS][{\"name\": \"nope\", \"arguments\": {}}]"));
        auto gp = Grammar::parse(mk(qv::ToolFormat::Mistral, "", true).gbnf);
        CHECK(matches(gp, "[TOOL_CALLS][{\"name\": \"noop\", \"arguments\": {}}, {\"name\": \"noop\", \"arguments\": {}}]"));
    }
    {
        auto g = Grammar::parse(mk(qv::ToolFormat::Inkling).gbnf);
        CHECK(matches(g, "<|message_model|>noop<|content_invoke_tool_json|>{\"name\":\"noop\",\"args\":{}}<|end_message|>"));
        CHECK(!matches(g, "<|message_model|>noop<|content_invoke_tool_json|>{\"name\":\"noop\",\"arguments\":{}}<|end_message|>"));
        auto gl = Grammar::parse(mk(qv::ToolFormat::Inkling).gbnf_lazy);
        CHECK(matches(gl, "<|content_invoke_tool_json|>{\"name\":\"noop\",\"args\":{}}<|end_message|>"));
        // arguments are written in `dictsort` (byte) order by the chat template, whatever the schema's declaration order (p28/loops)
        CHECK(matches(g, "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"P\",\"days\":3,\"unit\":\"c\"}}<|end_message|>"));
        CHECK(matches(g, "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"P\",\"unit\":\"c\"}}<|end_message|>"));
        CHECK(matches(g, "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"P\"}}<|end_message|>"));
        CHECK(!matches(g, "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"days\":3}}<|end_message|>"));   // city is required
        CHECK(!matches(g, "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"unit\":\"c\",\"city\":\"P\"}}<|end_message|>"));  // not byte order
    }
    {
        auto g = Grammar::parse(mk(qv::ToolFormat::Generic).gbnf);
        CHECK(matches(g, "{\"tool_call\": {\"name\": \"noop\", \"arguments\": {}}}"));
        auto gp = Grammar::parse(mk(qv::ToolFormat::Generic, "", true).gbnf);
        CHECK(matches(gp, "{\"tool_calls\": [{\"name\": \"noop\", \"arguments\": {}}, {\"name\": \"noop\", \"arguments\": {}}]}"));
    }
    // until rule
    SchemaGrammarBuilder b;
    add_until_rule(b, "u", "</abc>");
    b.add_rule("root", "u");
    auto ug = Grammar::parse(b.build());
    CHECK(matches(ug, "hello </ab \n</</abc>") && matches(ug, "</abc>") && !matches(ug, "x</abc>y") && !matches(ug, "no end") && matches(ug, "</</abc>"));
    bool thrown = false;
    try { ToolGrammarOptions o; o.forced_name = "zzz"; tool_call_grammar(t, o); } catch (const std::invalid_argument&) { thrown = true; }
    CHECK(thrown);
}

// Brute-force reference: token allowed iff its bytes can be fed from the state (or EOG at completion).
static bool brute_allowed(GrammarState st, const qv::grammar::VocabIndex& vi, size_t id) {
    if (vi.is_eog(id)) return st.is_complete();
    const std::string& p = vi.piece(id);
    if (p.empty() || p[0] == 0) return false;
    return st.accept(p);
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void test_small_vocab_equivalence() {
    std::vector<std::string> pieces = {"a", "ab", "abc", "b", "{", "{\"", "\"", "\":", "1", "12", "\xC3", "\xC3\xA9", "\xA9", "\xE2\x82", "\xE2\x82\xAC", "", " ", "}", "\xFF", "<eos>"};
    std::vector<uint8_t> eog(pieces.size(), 0);
    eog.back() = 1;
    auto vi = VocabIndex::build(pieces, eog);
    for (const char* gb : {"root ::= \"ab\" [a-c]*", "root ::= \"{\\\"\" [a-z]+ \"\\\":\" [0-9]+ \"}\"", "root ::= [\\u00e9-\\u20ff]+ \"1\"?", "root ::= . ."}) {
        auto m = std::make_shared<Matcher>(Grammar::parse(gb), vi);
        GrammarState st(m);
        for (int depth = 0; depth < 4; ++depth) {
            auto mask = st.allowed();
            int pick = -1;
            for (size_t i = 0; i < pieces.size(); ++i) {
                CHECKM(mask->allowed(i) == brute_allowed(st, *vi, i), std::string(gb) + " tok " + std::to_string(i) + " depth " + std::to_string(depth));
                if (pick < 0 && mask->allowed(i) && !vi->is_eog(i)) pick = (int)i;
            }
            if (pick < 0) break;
            CHECK(st.accept_token(pick));
        }
    }
}

static std::string fmt(double v) { char b[32]; std::snprintf(b, sizeof b, "%.3f", v); return b; }

static void test_qwen_vocab() {
    std::ifstream f(QV_QWEN_TOKENIZER);
    if (!f) { std::printf("SKIP qwen tokenizer tests (no %s)\n", QV_QWEN_TOKENIZER); return; }
    qv::Tokenizer tk = qv::Tokenizer::load(QV_QWEN_TOKENIZER);
    std::vector<int> eog;
    for (const char* e : {"<|im_end|>", "<|endoftext|>"}) { int id = tk.token_id(e); if (id >= 0) eog.push_back(id); }
    double t0 = now_us();
    auto vi = vocab_from_tokenizer(tk, eog);
    double build_ms = (now_us() - t0) / 1000;
    std::printf("[qwen] vocab=%zu trie_nodes=%zu max_piece=%u eog=%zu build=%.1f ms\n", vi->size(), vi->nodes.size(), vi->max_depth, eog.size(), build_ms);
    CHECK(vi->size() > 150000);

    struct Case { const char* name; Grammar g; std::string sample; };
    std::vector<Case> cases;
    cases.push_back({"json_object", Grammar::parse(json_object_grammar()), R"({"name": "Alice", "age": 30, "tags": ["a", "b"], "nested": {"x": [1, 2.5, null, true]}})"});
    cases.push_back({"schema_obj", schema_g(kObj), R"({"name": "Bobé 中文", "age": 42, "tags": ["x", "y"]})"});
    cases.push_back({"enum+int_range", schema_g(R"({"type":"object","properties":{"c":{"enum":["red","green","blue"]},"n":{"type":"integer","minimum":0,"maximum":1000}},"required":["c","n"]})"),
                     R"({"c": "green", "n": 512})"});
    {
        ToolGrammarOptions o;
        o.format = qv::ToolFormat::QwenXml;
        cases.push_back({"qwen_tool", Grammar::parse(tool_call_grammar(J(R"([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}])"), o).gbnf),
                         "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis, France\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"});
    }
    cases.push_back({"utf8_class", Grammar::parse("root ::= [\\u4e00-\\u9fff]+ \"!\""), "\xE4\xB8\xAD\xE6\x96\x87\xE5\xAD\x97!"});
    for (auto& c : cases) {
        auto m = std::make_shared<Matcher>(c.g, vi);
        auto toks = tk.encode(c.sample, false);
        std::string joined;
        for (int t : toks) joined += tk.token_bytes(t);
        CHECKM(joined == c.sample, std::string(c.name) + " roundtrip");
        // cold pass: every step computes a fresh mask; verify against brute force on a sample of tokens at several steps
        GrammarState st(m);
        std::vector<double> cold;
        std::mt19937 rng(7);
        int verified_steps = 0;
        for (size_t i = 0; i < toks.size(); ++i) {
            double a = now_us();
            auto mask = st.allowed();
            cold.push_back(now_us() - a);
            CHECKM(mask->allowed((size_t)toks[i]), std::string(c.name) + " sample token allowed at step " + std::to_string(i));
            if (i % 5 == 0 || i + 1 == toks.size()) {
                ++verified_steps;
                // full brute force on every 40th token + all allowed tokens
                size_t bad = 0;
                for (size_t id = 0; id < vi->size(); ++id) {
                    bool want_check = mask->allowed(id) || (id % 37 == (rng() % 37));
                    if (!want_check) continue;
                    if (mask->allowed(id) != brute_allowed(st, *vi, id)) ++bad;
                }
                CHECKM(bad == 0, std::string(c.name) + " mask mismatch step " + std::to_string(i) + " bad=" + std::to_string(bad));
            }
            CHECK(st.accept_token(toks[i]));
        }
        CHECKM(st.is_complete(), std::string(c.name) + " complete after sample");
        // EOG allowed exactly when complete
        auto fm = st.allowed();
        CHECK(eog.empty() || fm->allowed((size_t)eog[0]));
        // warm pass: replay, all masks cached
        GrammarState st2(m);
        std::vector<double> warm;
        std::vector<float> logits(vi->size() + 100, 0.5f);
        std::vector<double> apply;
        for (size_t i = 0; i < toks.size(); ++i) {
            double a = now_us();
            auto mask = st2.allowed();
            warm.push_back(now_us() - a);
            double b = now_us();
            st2.mask_logits(logits.data(), logits.size());
            apply.push_back(now_us() - b);
            std::fill(logits.begin(), logits.end(), 0.5f);
            st2.accept_token(toks[i]);
        }
        auto avg = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size(); };
        auto mx = [](const std::vector<double>& v) { double s = 0; for (double x : v) s = std::max(s, x); return s; };
        auto st_stats = m->stats();
        std::printf("[qwen] %-15s steps=%zu cold avg=%s ms max=%s ms | warm lookup avg=%s ms | mask_logits avg=%s ms | states=%zu masks=%zu\n", c.name, toks.size(),
                    fmt(avg(cold) / 1000).c_str(), fmt(mx(cold) / 1000).c_str(), fmt(avg(warm) / 1000).c_str(), fmt(avg(apply) / 1000).c_str(), st_stats.states, st_stats.cached_masks);
        (void)verified_steps;
    }
    // mask_logits semantics + sampler hook
    {
        auto m = std::make_shared<Matcher>(Grammar::parse("root ::= \"yes\" | \"no\""), vi);
        auto hook = std::make_shared<GrammarLogitMask>(GrammarState(m));
        std::vector<float> lg(vi->size(), 1.0f);
        int yes = tk.token_id("yes"), no = tk.token_id("no");
        CHECK(yes >= 0 && no >= 0);
        lg[(size_t)yes] = 5;
        hook->apply(lg.data(), lg.size());
        size_t finite = 0;
        for (float v : lg) if (std::isfinite(v)) ++finite;
        CHECKM(finite >= 2 && finite < 20, "finite=" + std::to_string(finite));  // y, ye, yes, n, no ...
        CHECK(std::isfinite(lg[(size_t)yes]) && std::isfinite(lg[(size_t)no]));
        qv::SamplingParams sp;
        sp.temperature = 0;
        qv::Sampler s(sp, {});
        s.set_mask(std::make_shared<GrammarLogitMask>(GrammarState(m)));
        std::vector<float> l2(vi->size(), 0.0f);
        l2[(size_t)tk.token_id("hello")] = 100;  // unconstrained argmax
        int t = s.sample(l2);
        CHECK(t == yes || t == no || (std::string("yes").rfind(tk.token_bytes(t), 0) == 0) || (std::string("no").rfind(tk.token_bytes(t), 0) == 0));
        s.accept(t);
    }
    // lazy trigger
    {
        auto m = std::make_shared<Matcher>(Grammar::parse("root ::= \"<tool_call>\" [0-9]+ \"</tool_call>\""), vi);
        Triggers tr;
        tr.words = {"<tool_call>"};
        GrammarState st(m, tr);
        CHECK(st.awaiting_trigger());
        CHECK(st.accept("Let me check. ") && st.awaiting_trigger());
        CHECK(st.accept("ok <tool_call>12") && !st.awaiting_trigger());
        CHECK(!st.is_complete());
        CHECK(st.accept("</tool_call>") && st.is_complete());
        CHECK(!st.accept("x"));
    }
    // mask cache bound / stats
    {
        MatcherOptions mo;
        mo.max_cache_bytes = 4 * (vi->size() / 8 + 64);
        auto m = std::make_shared<Matcher>(Grammar::parse(json_object_grammar()), vi, mo);
        GrammarState st(m);
        std::string s = R"({"a": [1, 2, 3, {"b": "c"}], "d": null})";
        for (size_t i = 0; i < s.size(); ++i) { st.allowed(); st.accept(std::string(1, s[i])); }
        CHECK(m->stats().cache_bytes <= mo.max_cache_bytes + vi->size() / 8 + 64);
        CHECK(st.is_complete());
    }
}

int main() {
    test_parser();
    test_schema();
    test_tools();
    test_small_vocab_equivalence();
    test_qwen_vocab();
    std::printf("%d checks, %d failed\n", g_n, g_fail);
    return g_fail ? 1 : 0;
}
