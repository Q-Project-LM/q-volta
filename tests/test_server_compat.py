#!/usr/bin/env python3
"""llama-server compatibility tests for qv_server.

  python3 -m pytest tests/test_server_compat.py -q          (needs build-cpu/qv_server; mock engine, no model weights)
  QV_SERVER=build-cpu/qv_server LLAMA_SERVER=/tmp/llama_srv_build/bin/llama-server LLAMA_GGUF=/tmp/qwen05/qwen2.5-0.5b-instruct-q4_k_m.gguf \
      python3 -m pytest tests/test_server_compat.py -q       (also runs the structural diff against the real llama-server)

Part 1 (ours alone): endpoints, request/response shapes, SSE framing, errors, queue/slots, CLI. Uses the `--backend mock` engine
(scripted tokens through the real tokenizer path) and stdlib http.client only.
Part 2 (differential, skipped without llama-server): the same requests go to both servers and the JSON *structure* (key sets, value
types), status codes, header names and SSE framing are compared. Text values differ by design (different models) and are not compared.
"""
import http.client, json, os, re, signal, socket, subprocess, sys, tempfile, threading, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QV_SERVER = os.environ.get("QV_SERVER", os.path.join(ROOT, "build-cpu", "qv_server"))
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/tmp/llama_srv_build/bin/llama-server")
LLAMA_GGUF = os.environ.get("LLAMA_GGUF", "/tmp/qwen05/qwen2.5-0.5b-instruct-q4_k_m.gguf")
TOKJSON = os.environ.get("QV_TOKENIZER", "/workspace/q-volta-models/Qwen3.8-27B/tokenizer.json")
HAVE_LLAMA = os.path.exists(LLAMA_SERVER) and os.path.exists(LLAMA_GGUF)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Srv:
    def __init__(self, cmd, env=None, wait=True, name="srv"):
        self.port = free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=f".{name}.log", delete=False)
        full = [c.replace("@PORT@", str(self.port)) for c in cmd]
        self.p = subprocess.Popen(full, stdout=self.log, stderr=self.log, env=dict(os.environ, **(env or {})))
        if wait: self.wait_ready()

    def wait_ready(self, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None: raise RuntimeError("server exited: " + open(self.log.name).read()[-2000:])
            try:
                st, _, _ = self.req("GET", "/health")
                if st == 200: return
            except OSError: pass
            time.sleep(0.1)
        raise RuntimeError("server did not become ready")

    def req(self, method, path, body=None, headers=None, raw=False, timeout=60):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, (str, bytes)) else json.dumps(body)
            h.setdefault("Content-Type", "application/json")
        c.request(method, path, data, h)
        r = c.getresponse()
        b = r.read()
        hd = [(k, v) for k, v in r.getheaders()]
        c.close()
        if raw: return r.status, hd, b
        ct = r.getheader("Content-Type", "") or ""
        if b and ct.startswith("application/json"):
            try: return r.status, hd, json.loads(b)
            except ValueError: pass
        return r.status, hd, b

    def sse(self, path, body, headers=None):
        st, hd, b = self.req("POST", path, body, headers, raw=True)
        events = []
        for blk in b.decode().split("\n\n"):
            if not blk: continue
            assert blk.startswith("data: "), "bad SSE framing: %r" % blk[:80]
            d = blk[6:]
            events.append(d if d == "[DONE]" else json.loads(d))
        return st, hd, events, b

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try: self.p.wait(10)
            except subprocess.TimeoutExpired: self.p.kill()


def hdrs(hd): return {k.lower(): v for k, v in hd}


def qv_cmd(*extra, backend="mock"):
    base = [QV_SERVER, "--backend", backend, "--tokenizer", TOKJSON, "--port", "@PORT@", "-c", "16384"]  # 16384 total = 4 slots x 4096 (--ctx-size is divided among the slots)
    return base + list(extra)


@pytest.fixture(scope="module")
def srv():
    s = Srv(qv_cmd("--metrics", "--props", "--slot-save-path", tempfile.mkdtemp(prefix="qvslots_"), "--mock-loop", "--mock-text", "alpha beta gamma delta"), name="qv")
    yield s
    s.stop()


# =============================================================================================== part 1: ours alone
def test_health_and_aliases(srv):
    for p in ("/health", "/v1/health"):
        st, hd, b = srv.req("GET", p)
        assert st == 200 and b == {"status": "ok"}
        assert hdrs(hd)["content-type"] == "application/json; charset=utf-8"
    st, hd, b = srv.req("HEAD", "/health", raw=True)
    assert st == 200 and b == b""


def test_health_503_while_loading():
    s = Srv(qv_cmd(), env={"QV_MOCK_LOAD_MS": "2500"}, wait=False, name="load")
    try:
        t0 = time.time(); seen = None
        while time.time() - t0 < 2:
            try:
                seen = s.req("GET", "/health"); break
            except OSError: time.sleep(0.05)
        assert seen is not None, "port not open while loading"
        st, hd, b = seen
        assert st == 503 and b == {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}}
        st, _, b = s.req("POST", "/completion", {"prompt": "x"})
        assert st == 503
        s.wait_ready()
        assert s.req("GET", "/health")[0] == 200
    finally:
        s.stop()


def test_props_shape(srv):
    st, _, p = srv.req("GET", "/props")
    assert st == 200
    for k in ("default_generation_settings", "total_slots", "model_alias", "model_path", "modalities", "endpoint_slots", "endpoint_props", "endpoint_metrics",
              "chat_template", "chat_template_caps", "bos_token", "eos_token", "build_info", "is_sleeping"):
        assert k in p, k
    assert p["total_slots"] == 4 and p["endpoint_metrics"] is True and p["endpoint_props"] is True
    assert p["default_generation_settings"]["n_ctx"] == 4096
    prm = p["default_generation_settings"]["params"]
    assert prm["temperature"] == pytest.approx(0.8) and prm["top_k"] == 40 and prm["top_p"] == pytest.approx(0.95) and prm["min_p"] == pytest.approx(0.05)
    assert prm["samplers"] == ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]
    assert srv.req("POST", "/props", {})[2] == {"success": True}


def test_models(srv):
    for path in ("/models", "/v1/models"):
        st, _, m = srv.req("GET", path)
        assert st == 200 and m["object"] == "list" and m["data"][0]["object"] == "model" and m["data"][0]["id"]
        assert m["models"][0]["name"] == m["data"][0]["id"]


def test_completion_native_shape(srv):
    st, hd, r = srv.req("POST", "/completion", {"prompt": "Hello", "n_predict": 4, "temperature": 0})
    assert st == 200
    for k in ("index", "content", "tokens", "id_slot", "stop", "model", "tokens_predicted", "tokens_evaluated", "generation_settings", "prompt", "has_new_line",
              "truncated", "stop_type", "stopping_word", "tokens_cached", "timings"):
        assert k in r, k
    assert r["content"] == "alpha beta gamma" or r["content"].startswith("alpha") 
    assert r["tokens_predicted"] == 4 and r["stop_type"] == "limit" and r["stop"] is True and r["tokens"] == [] and r["truncated"] is False
    t = r["timings"]
    for k in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"):
        assert k in t, k
    gs = r["generation_settings"]
    assert gs["n_predict"] == 4 and gs["temperature"] == 0 and gs["stop"] == [] and "dry_sequence_breakers" in gs and gs["logit_bias"] == []


def test_completion_return_tokens_and_prompt_forms(srv):
    st, _, r = srv.req("POST", "/completion", {"prompt": "Hello", "n_predict": 3, "return_tokens": True, "temperature": 0})
    assert len(r["tokens"]) == 3 and all(isinstance(x, int) for x in r["tokens"])
    ids = srv.req("POST", "/tokenize", {"content": "Hello world"})[2]["tokens"]
    for prompt in (ids, ["Hello", ids[1]], {"prompt_string": "Hello world"}):
        st, _, r = srv.req("POST", "/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0})
        assert st == 200 and r["tokens_evaluated"] == len(ids), prompt
    st, _, r = srv.req("POST", "/completion", {"prompt": ["a", "b"], "n_predict": 1, "temperature": 0})
    assert st == 200 and isinstance(r, list) and len(r) == 2 and r[0]["index"] == 0


def test_stop_word_and_eos(srv):
    st, _, r = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 20, "stop": [" gamma"], "temperature": 0})
    assert r["stop_type"] == "word" and r["stopping_word"] == " gamma" and r["content"] == "alpha beta"
    s2 = Srv(qv_cmd("--mock-text", "short"), name="eos")
    try:
        r = s2.req("POST", "/completion", {"prompt": "x", "n_predict": 20, "temperature": 0})[2]
        assert r["stop_type"] == "eos" and r["content"] == "short" and r["tokens_predicted"] == 2  # script token + EOS
        r = s2.req("POST", "/completion", {"prompt": "x", "n_predict": 20, "temperature": 0, "ignore_eos": True, "logit_bias": []})[2]
        assert r["stop_type"] == "limit" and r["tokens_predicted"] == 20
    finally:
        s2.stop()


def test_n_probs_and_logit_bias(srv):
    r = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 2, "temperature": 0, "n_probs": 3})[2]
    cp = r["completion_probabilities"]
    assert len(cp) == 2 and set(cp[0]) == {"id", "token", "bytes", "logprob", "top_logprobs"} and len(cp[0]["top_logprobs"]) == 3
    assert cp[0]["top_logprobs"][0]["logprob"] == pytest.approx(cp[0]["logprob"])
    r = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 2, "temperature": 0, "n_probs": 2, "post_sampling_probs": True})[2]
    assert "prob" in r["completion_probabilities"][0] and "top_probs" in r["completion_probabilities"][0]
    tid = srv.req("POST", "/tokenize", {"content": "zzz"})[2]["tokens"][0]
    r = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 1, "temperature": 0, "logit_bias": [[tid, 100.0]], "return_tokens": True})[2]
    assert r["tokens"] == [tid]


def test_seed_determinism_and_sampling_params(srv):
    body = {"prompt": "x", "n_predict": 6, "temperature": 5.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42}
    a = srv.req("POST", "/completion", body)[2]["content"]
    b = srv.req("POST", "/completion", body)[2]["content"]
    assert a == b
    body["seed"] = 43
    assert srv.req("POST", "/completion", body)[2]["content"] != a
    for extra in ({"typical_p": 0.5}, {"mirostat": 2}, {"mirostat": 1}, {"xtc_probability": 1.0, "xtc_threshold": 0.01}, {"top_n_sigma": 1.0},
                  {"dry_multiplier": 0.8}, {"dynatemp_range": 0.5}, {"samplers": ["top_k", "temperature"]}, {"repeat_penalty": 1.3}):
        st, _, r = srv.req("POST", "/completion", dict(body, **extra))
        assert st == 200 and r["tokens_predicted"] == 6, extra
    # penalties flip the scripted token when strong enough
    a = srv.req("POST", "/completion", {"prompt": "x", "n_predict": 8, "temperature": 0, "frequency_penalty": 1000})[2]["content"]
    assert a != srv.req("POST", "/completion", {"prompt": "x", "n_predict": 8, "temperature": 0})[2]["content"]


def test_prompt_cache_reporting(srv):
    p = "cache test prompt number one " * 4
    r1 = srv.req("POST", "/completion", {"prompt": p, "n_predict": 2, "temperature": 0})[2]
    r2 = srv.req("POST", "/completion", {"prompt": p + "more", "n_predict": 2, "temperature": 0})[2]
    # the engine state after r1 holds prompt+generated tokens; r2's prompt only extends the prompt, so it cannot reuse a longer cached sequence
    assert r1["timings"]["cache_n"] == 0
    # continuing from the exact cached sequence reuses it
    full = srv.req("POST", "/tokenize", {"content": p})[2]["tokens"]
    r3 = srv.req("POST", "/completion", {"prompt": full, "n_predict": 2, "temperature": 0})[2]
    r4 = srv.req("POST", "/completion", {"prompt": full, "n_predict": 2, "temperature": 0, "cache_prompt": False})[2]
    assert r4["timings"]["cache_n"] == 0 and r4["timings"]["prompt_n"] == len(full)
    # r3 follows r2 (different prompt) -> prefix mismatch -> full re-evaluation; run twice so that the 2nd one can reuse
    r5 = srv.req("POST", "/completion", {"prompt": full, "n_predict": 2, "temperature": 0})[2]
    assert r5["tokens_cached"] >= len(full)
    # exact repetition: the engine state ends at prompt+gen[:-1]; a new request that starts with that exact sequence reuses it
    text = r5["content"]
    st, _, usage = srv.req("POST", "/v1/completions", {"prompt": full, "max_tokens": 1, "temperature": 0})
    assert usage["usage"]["prompt_tokens_details"]["cached_tokens"] == usage["timings"]["cache_n"]


def test_oai_completions(srv):
    st, hd, r = srv.req("POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 3, "temperature": 0})
    assert st == 200 and r["object"] == "text_completion" and r["id"].startswith("chatcmpl-") and r["system_fingerprint"]
    assert r["choices"][0]["finish_reason"] == "length" and r["choices"][0]["logprobs"] is None and r["choices"][0]["index"] == 0
    assert r["usage"]["completion_tokens"] == 3 and r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + 3 and "timings" in r
    st, hd, ev, raw = srv.sse("/v1/completions", {"prompt": "Hello", "max_tokens": 3, "temperature": 0, "stream": True})
    assert hdrs(hd)["content-type"] == "text/event-stream" and ev[-1] == "[DONE]"
    assert all(e["object"] == "text_completion" for e in ev[:-1]) and ev[-2]["choices"][0]["finish_reason"] == "length" and "usage" in ev[-2]
    assert "".join(e["choices"][0]["text"] for e in ev[:-1]) == "alpha beta gamma"[:len("".join(e["choices"][0]["text"] for e in ev[:-1]))]


def test_chat_non_stream(srv):
    st, hd, r = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "temperature": 0,
                                                          "chat_template_kwargs": {"enable_thinking": False}})
    assert st == 200 and r["object"] == "chat.completion" and r["id"].startswith("chatcmpl-")
    ch = r["choices"][0]
    assert ch["message"]["role"] == "assistant" and ch["message"]["content"].startswith("alpha") and ch["finish_reason"] == "length"
    assert set(r["usage"]) == {"completion_tokens", "prompt_tokens", "total_tokens", "prompt_tokens_details"} and "timings" in r
    # same request through the alias without /v1
    assert srv.req("POST", "/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1})[0] == 200
    # multi-part content, developer role, max_completion_tokens, n=2
    r = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
                                                  "max_completion_tokens": 2, "n": 2, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})[2]
    assert len(r["choices"]) == 2 and [c["index"] for c in r["choices"]] == [0, 1] and r["usage"]["completion_tokens"] == 4


def test_chat_reasoning_formats(srv):
    base = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "temperature": 0}
    r = srv.req("POST", "/v1/chat/completions", base)[2]["choices"][0]["message"]   # thinking template opens <think>: the scripted text is reasoning
    assert r["content"] == "" and r["reasoning_content"].startswith("alpha")
    r = srv.req("POST", "/v1/chat/completions", dict(base, reasoning_format="none"))[2]["choices"][0]["message"]
    assert "reasoning_content" not in r and r["content"].startswith("alpha")
    r = srv.req("POST", "/v1/chat/completions", dict(base, chat_template_kwargs={"enable_thinking": False}))[2]["choices"][0]["message"]
    assert "reasoning_content" not in r and r["content"].startswith("alpha")
    st, _, r = srv.req("POST", "/v1/chat/completions", dict(base, reasoning_format="bogus"))
    assert st == 500 and r["error"]["type"] == "server_error"   # llama-server answers 500 for an unknown format


def test_chat_stream_framing(srv):
    st, hd, ev, raw = srv.sse("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "temperature": 0, "stream": True,
                                                        "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}})
    h = hdrs(hd)
    assert st == 200 and h["content-type"] == "text/event-stream" and h["transfer-encoding"] == "chunked" and h.get("x-accel-buffering") == "no"
    assert ev[-1] == "[DONE]" and raw.endswith(b"data: [DONE]\n\n")
    assert ev[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    fin = ev[-3]
    assert fin["choices"][0]["finish_reason"] == "length" and fin["choices"][0]["delta"] == {}
    usage = ev[-2]
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == 4 and "timings" in usage
    ids = {e["id"] for e in ev[:-1]}
    assert len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-")
    assert all(e["object"] == "chat.completion.chunk" and "system_fingerprint" in e for e in ev[:-1])
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in ev[1:-3])
    assert text.startswith("alpha")
    # no include_usage: timings ride on the finish chunk
    st, hd, ev, raw = srv.sse("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2, "stream": True})
    assert ev[-2]["choices"][0]["finish_reason"] == "length" and "usage" not in ev[-2] and "timings" in ev[-2]


def test_chat_logprobs_and_tools_error(srv):
    r = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2, "temperature": 0, "logprobs": True, "top_logprobs": 2,
                                                  "chat_template_kwargs": {"enable_thinking": False}})[2]
    lp = r["choices"][0]["logprobs"]["content"]
    assert len(lp) == 2 and set(lp[0]) == {"id", "token", "bytes", "logprob", "top_logprobs"} and len(lp[0]["top_logprobs"]) == 2
    st, _, e = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "top_logprobs": 2})
    assert st == 400 and e["error"]["message"] == "top_logprobs requires logprobs to be set to true"
    st, _, e = srv.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "tool_choice": "bogus", "tools": [WEATHER]})
    assert st == 400 and e["error"]["message"] == "Invalid tool_choice: bogus"


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
NOTHINK = {"enable_thinking": False}


@pytest.fixture(scope="module")
def tsrv():
    s = Srv(qv_cmd("--mock-text", CALL), name="tools")
    yield s
    s.stop()


def chat_body(**kw):
    return dict({"messages": [{"role": "user", "content": "weather in Paris?"}], "max_tokens": 120, "temperature": 0, "chat_template_kwargs": NOTHINK}, **kw)


def test_tools_non_stream(tsrv):
    st, _, r = tsrv.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER]))
    assert st == 200
    ch = r["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] == ""
    tc = ch["message"]["tool_calls"]
    assert len(tc) == 1 and tc[0]["type"] == "function" and tc[0]["id"] and tc[0]["function"]["name"] == "get_weather"
    assert json.loads(tc[0]["function"]["arguments"]) == {"city": "Paris"}
    # tool_choice none: tools are rendered but the output is not parsed
    r = tsrv.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], tool_choice="none"))[2]["choices"][0]
    assert "tool_calls" not in r["message"] and "<tool_call>" in r["message"]["content"] and r["finish_reason"] != "tool_calls"


def test_tools_stream(tsrv):
    st, hd, ev, raw = tsrv.sse("/v1/chat/completions", chat_body(tools=[WEATHER], stream=True))
    assert st == 200 and ev[-1] == "[DONE]"
    deltas = [e["choices"][0]["delta"] for e in ev[:-1] if e["choices"]]
    assert deltas[0] == {"role": "assistant", "content": None}
    tcs = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert tcs[0]["index"] == 0 and tcs[0]["id"] and tcs[0]["type"] == "function" and tcs[0]["function"]["name"] == "get_weather"
    assert all("id" not in t and "name" not in t["function"] for t in tcs[1:])
    assert json.loads("".join(t["function"]["arguments"] for t in tcs)) == {"city": "Paris"}
    assert not any(d.get("content") for d in deltas[1:])
    fin = [e for e in ev[:-1] if e["choices"] and e["choices"][0]["finish_reason"]]
    assert len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == "tool_calls" and fin[0]["choices"][0]["delta"] == {}


def test_tools_parallel_and_history():
    s = Srv(qv_cmd("--mock-text", CALL + "\n" + CALL.replace("Paris", "Rome")), name="par")
    try:
        r = s.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], parallel_tool_calls=True))[2]["choices"][0]["message"]["tool_calls"]
        assert [json.loads(t["function"]["arguments"])["city"] for t in r] == ["Paris", "Rome"] and r[0]["id"] != r[1]["id"]
        hist = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}]
        ap = s.req("POST", "/apply-template", {"messages": hist, "tools": [WEATHER]})
        assert ap[0] == 200 and "get_weather" in ap[2]["prompt"] and "sunny" in ap[2]["prompt"] and "<tools>" in ap[2]["prompt"]
        assert s.req("POST", "/v1/chat/completions", chat_body(messages=hist, tools=[WEATHER]))[0] == 200
    finally:
        s.stop()


def test_forced_and_required_tool_choice_use_grammar():
    s = Srv(qv_cmd("--mock-text", "I will not call any tool, just talking."), name="forced")
    try:
        for tc in ("required", {"type": "function", "function": {"name": "get_weather"}}):
            r = s.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], tool_choice=tc, max_tokens=300))[2]["choices"][0]
            assert r["finish_reason"] == "tool_calls" and r["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
            assert isinstance(json.loads(r["message"]["tool_calls"][0]["function"]["arguments"]), dict)
        # tool_choice auto: lazy grammar, plain text passes through untouched
        r = s.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], max_tokens=300))[2]["choices"][0]
        assert r["message"]["content"].startswith("I will not") and "tool_calls" not in r["message"]
    finally:
        s.stop()


def test_response_format_and_grammar():
    s = Srv(qv_cmd("--mock-text", "hello world"), name="grammar")
    try:
        r = s.req("POST", "/v1/chat/completions", chat_body(response_format={"type": "json_object"}, max_tokens=300))[2]["choices"][0]
        assert isinstance(json.loads(r["message"]["content"]), dict) and r["finish_reason"] == "stop"
        sch = {"type": "object", "properties": {"n": {"type": "integer"}, "tag": {"enum": ["a", "b"]}}, "required": ["n", "tag"]}
        for rf in ({"type": "json_schema", "json_schema": {"name": "x", "schema": sch}}, {"type": "json_object", "schema": sch}):
            r = s.req("POST", "/v1/chat/completions", chat_body(response_format=rf, max_tokens=300))[2]["choices"][0]
            v = json.loads(r["message"]["content"])
            assert isinstance(v["n"], int) and v["tag"] in ("a", "b")
        r = s.req("POST", "/completion", {"prompt": "x", "grammar": 'root ::= "yes" | "no"', "n_predict": 10, "temperature": 0})[2]
        assert r["content"] in ("yes", "no") and r["generation_settings"]["grammar"] == 'root ::= "yes" | "no"'
        r = s.req("POST", "/completion", {"prompt": "x", "json_schema": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "n_predict": 40, "temperature": 0.7})[2]
        v = json.loads(r["content"]); assert len(v) == 2 and all(isinstance(x, int) for x in v)
        r = s.req("POST", "/completion", {"prompt": "x", "grammar": 'root ::= [0-9]+', "n_predict": 6, "temperature": 1.0, "seed": 3})[2]
        assert r["content"].isdigit()
        st, _, e = s.req("POST", "/completion", {"prompt": "x", "grammar": "root ::= undefined-rule"})
        assert st == 400 and e["error"]["type"] == "invalid_request_error"
        assert s.req("POST", "/v1/chat/completions", chat_body(grammar="root ::= \"a\"", json_schema={}))[0] == 500
        assert s.req("POST", "/completion", {"prompt": "x", "n_predict": 3, "temperature": 0, "grammar": "root ::= \"a\"", "json_schema": {}})[2]["content"] == "a"
        assert s.req("POST", "/v1/chat/completions", chat_body(response_format={"type": "bogus"}))[0] == 400
        st, _, ev, _ = s.sse("/v1/chat/completions", chat_body(response_format={"type": "json_object"}, stream=True, max_tokens=300))
        txt = "".join(e["choices"][0]["delta"].get("content") or "" for e in ev[:-1] if e["choices"])
        assert isinstance(json.loads(txt), dict)
    finally:
        s.stop()


def test_cli_grammar_flags():
    s = Srv(qv_cmd("--mock-text", "zzz", "--grammar", 'root ::= "ok"'), name="cligr")
    try:
        assert s.req("POST", "/completion", {"prompt": "x", "n_predict": 10, "temperature": 0})[2]["content"] == "ok"
    finally:
        s.stop()
    s = Srv(qv_cmd("--mock-text", "zzz", "--json-schema", '{"type":"boolean"}'), name="clijs")
    try:
        assert s.req("POST", "/completion", {"prompt": "x", "n_predict": 10, "temperature": 0})[2]["content"] in ("true", "false")
    finally:
        s.stop()


def test_chat_template_flags(tmp_path):
    s = Srv(qv_cmd("--chat-template", "chatml"), name="chatml")
    try:
        p = s.req("POST", "/apply-template", {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]})[2]["prompt"]
        assert p == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
    finally:
        s.stop()
    f = tmp_path / "t.jinja"
    f.write_text("{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}{% if add_generation_prompt %}[assistant] {% endif %}")
    s = Srv(qv_cmd("--chat-template-file", str(f)), name="tfile")
    try:
        assert s.req("POST", "/apply-template", {"messages": [{"role": "user", "content": "U"}]})[2]["prompt"] == "[user] U\n[assistant] "
        assert s.req("GET", "/props")[2]["chat_template"].startswith("{% for m")
    finally:
        s.stop()
    bad = run_cli(["--backend", "mock", "--tokenizer", TOKJSON, "--chat-template", "{% for %}"])
    assert bad.returncode == 1


def test_jinja_matches_native_renderer():
    a = Srv(qv_cmd(), name="jinja"); b = Srv(qv_cmd("--no-jinja"), name="native")
    try:
        convs = [{"messages": [{"role": "user", "content": "Hi"}]},
                 {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]},
                 {"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"enable_thinking": False}},
                 {"messages": [{"role": "user", "content": [{"type": "text", "text": "p1"}, {"type": "text", "text": "p2"}]}], "reasoning_effort": "low"}]
        for c in convs:
            pa, pb = a.req("POST", "/apply-template", c)[2], b.req("POST", "/apply-template", c)[2]
            assert pa["prompt"] == pb["prompt"], (c, pa, pb)
        assert b.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tools": [WEATHER]})[0] == 400
        assert a.req("GET", "/props")[2]["chat_template_caps"]["supports_tools"] is True
    finally:
        a.stop(); b.stop()


def test_error_shapes(srv):
    def err(m, p, body, status, typ, msg=None):
        st, hd, e = srv.req(m, p, body)
        assert st == status, (p, body, st, e)
        assert set(e) == {"error"} and e["error"]["code"] == status and e["error"]["type"] == typ and set(e["error"]) >= {"code", "message", "type"}, e
        if msg: assert msg in e["error"]["message"], e
        h = hdrs(hd)
        assert h["content-type"] == "application/json; charset=utf-8" and h.get("connection") == "close"
    err("POST", "/v1/chat/completions", "{bad", 400, "invalid_request_error", "parse_error")
    err("POST", "/v1/chat/completions", {}, 400, "invalid_request_error", "'messages' is required")
    err("POST", "/v1/chat/completions", {"messages": "x"}, 400, "invalid_request_error", "Expected 'messages' to be an array")
    err("POST", "/v1/chat/completions", {"messages": [{"role": "user"}]}, 400, "invalid_request_error", "All non-assistant messages must contain 'content'")
    err("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": "a"}, 400, "invalid_request_error", "Field 'max_tokens'")
    err("POST", "/completion", {}, 400, "invalid_request_error", "'prompt' not found")
    err("POST", "/completion", {"prompt": "x", "temperature": "x"}, 400, "invalid_request_error", "Field 'temperature'")
    err("POST", "/completion", {"prompt": "x", "stream": "x"}, 400, "invalid_request_error", "Field 'stream'")
    err("GET", "/nonexistent", None, 404, "not_found_error", "File Not Found")
    err("GET", "/completion", None, 404, "not_found_error")
    err("POST", "/embedding", {"content": "x"}, 501, "not_supported_error", "--embeddings")
    err("POST", "/rerank", {}, 501, "not_supported_error")
    err("POST", "/tokenize", {"content": 5}, 400, "invalid_request_error", "type must be string")
    err("POST", "/detokenize", {"tokens": "x"}, 400, "invalid_request_error", "type must be array")
    big = "word " * 5000
    st, _, e = srv.req("POST", "/completion", {"prompt": big, "n_predict": 1})
    assert st == 400 and e["error"]["type"] == "exceed_context_size_error" and e["error"]["n_ctx"] == 4096 and e["error"]["n_prompt_tokens"] > 4096
    assert "exceeds the available context size" in e["error"]["message"]


def test_tokenize_detokenize_apply_template(srv):
    r = srv.req("POST", "/tokenize", {"content": "Hello world"})[2]
    assert list(r) == ["tokens"] and all(isinstance(x, int) for x in r["tokens"])
    rp = srv.req("POST", "/tokenize", {"content": "Hello world", "with_pieces": True})[2]["tokens"]
    assert [x["id"] for x in rp] == r["tokens"] and [x["piece"] for x in rp] == ["Hello", " world"]
    assert srv.req("POST", "/tokenize", {})[2] == {"tokens": []}
    special = srv.req("POST", "/tokenize", {"content": "<|im_start|>", "parse_special": True})[2]["tokens"]
    plain = srv.req("POST", "/tokenize", {"content": "<|im_start|>", "parse_special": False})[2]["tokens"]
    assert len(special) == 1 and len(plain) > 1
    assert srv.req("POST", "/detokenize", {"tokens": r["tokens"]})[2] == {"content": "Hello world"}
    assert srv.req("POST", "/detokenize", {})[2] == {"content": ""}
    ap = srv.req("POST", "/apply-template", {"messages": [{"role": "user", "content": "Hi"}]})[2]
    assert list(ap) == ["prompt"] and ap["prompt"].startswith("<|im_start|>") and ap["prompt"].endswith("<|im_start|>assistant\n<think>\n")
    assert srv.req("POST", "/apply-template", {"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"enable_thinking": False}})[2]["prompt"].endswith("</think>\n\n")
    assert srv.req("POST", "/apply-template", {})[0] == 400


def test_infill(srv):
    st, _, r = srv.req("POST", "/infill", {"input_prefix": "def f(", "input_suffix": "):\n    pass", "n_predict": 2, "temperature": 0})
    assert st == 200 and "<|fim_prefix|>" in r["prompt"] and "<|fim_suffix|>" in r["prompt"] and r["prompt"].endswith("<|fim_middle|>")
    assert srv.req("POST", "/infill", {"input_suffix": "x"})[0] == 400


def test_slots_and_actions(srv):
    st, _, s = srv.req("GET", "/slots")
    assert st == 200 and len(s) == 4 and all(x["n_ctx"] == 4096 and x["speculative"] is False for x in s)
    srv.req("POST", "/completion", {"prompt": "slot test", "n_predict": 2, "temperature": 0, "id_slot": 2})
    s = srv.req("GET", "/slots")[2]
    assert s[2]["is_processing"] is False and s[2]["id_task"] >= 0 and "params" in s[2] and s[2]["next_token"][0]["n_decoded"] == 2
    assert srv.req("GET", "/slots?fail_on_no_slot=1")[0] == 200
    r = srv.req("POST", "/completion", {"prompt": "slot test", "n_predict": 1, "id_slot": 2})[2]
    assert r["id_slot"] == 2
    st, _, sv = srv.req("POST", "/slots/0?action=save", {"filename": "a.bin"})
    assert st == 200 and sv["id_slot"] == 0 and sv["filename"] == "a.bin" and sv["n_saved"] > 0 and "save_ms" in sv["timings"]
    assert srv.req("POST", "/slots/0?action=erase")[2] == {"id_slot": 0, "n_erased": sv["n_saved"]}
    st, _, rs = srv.req("POST", "/slots/0?action=restore", {"filename": "a.bin"})
    assert st == 200 and rs["n_restored"] == sv["n_saved"] and "restore_ms" in rs["timings"]
    assert srv.req("POST", "/slots/0?action=save", {"filename": "../x"})[0] == 400
    assert srv.req("POST", "/slots/9?action=erase")[0] == 400
    assert srv.req("POST", "/slots/0?action=bogus")[0] == 400
    assert srv.req("POST", "/slots/0?action=restore", {"filename": "missing.bin"})[0] == 400


def test_metrics(srv):
    st, hd, b = srv.req("GET", "/metrics", raw=True)
    t = b.decode()
    assert st == 200 and hdrs(hd)["content-type"] == "text/plain; version=0.0.4" and "process-start-time-unix" in hdrs(hd)
    for name in ("prompt_tokens_total", "prompt_tokens_cached_total", "prompt_seconds_total", "tokens_predicted_total", "tokens_predicted_seconds_total", "n_decode_total",
                 "n_tokens_max", "prompt_tokens_seconds", "predicted_tokens_seconds", "requests_processing", "requests_deferred", "n_busy_slots_per_decode"):
        assert re.search(r"^# TYPE llamacpp:%s (counter|gauge)$" % name, t, re.M), name
    assert float(re.search(r"^llamacpp:tokens_predicted_total (\S+)", t, re.M).group(1)) > 0
    s2 = Srv(qv_cmd(), name="nometrics")
    try:
        st, _, e = s2.req("GET", "/metrics")
        assert st == 501 and e["error"]["type"] == "not_supported_error" and "--metrics" in e["error"]["message"]
        assert s2.req("POST", "/props", {})[0] == 501
        assert s2.req("POST", "/slots/0?action=save", {"filename": "x"})[0] == 501
    finally:
        s2.stop()


def test_cors_options_and_headers(srv):
    st, hd, b = srv.req("OPTIONS", "/v1/chat/completions", raw=True, headers={"Origin": "http://example.com"})
    h = hdrs(hd)
    assert st == 200 and b == b"" and h["access-control-allow-origin"] == "http://example.com" and h["access-control-allow-credentials"] == "true"
    assert "POST" in h["access-control-allow-methods"] and h["access-control-allow-headers"] == "*" and h["content-type"] == "text/html"
    st, hd, _ = srv.req("GET", "/health", headers={"Origin": "http://example.com"})
    assert hdrs(hd)["access-control-allow-origin"] == "http://example.com"
    assert hdrs(hd)["keep-alive"] == "timeout=5, max=100"


def test_chunked_request_body(srv):
    s = socket.create_connection(("127.0.0.1", srv.port))
    body = json.dumps({"prompt": "x", "n_predict": 1, "temperature": 0}).encode()
    s.sendall(b"POST /completion HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n" +
              ("%x\r\n" % len(body)).encode() + body + b"\r\n0\r\n\r\n")
    data = b""
    while True:
        d = s.recv(65536)
        if not d: break
        data += d
    assert data.startswith(b"HTTP/1.1 200")


def test_api_key():
    s = Srv(qv_cmd("--api-key", "k1,k2"), name="auth")
    try:
        assert s.req("GET", "/health")[0] == 200 and s.req("GET", "/v1/health")[0] == 200
        st, _, e = s.req("GET", "/props")
        assert st == 401 and e == {"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}
        assert s.req("GET", "/props", headers={"Authorization": "Bearer k2"})[0] == 200
        assert s.req("GET", "/props", headers={"Authorization": "k1"})[0] == 200
        assert s.req("GET", "/props", headers={"X-Api-Key": "k1"})[0] == 200
        assert s.req("GET", "/props", headers={"Authorization": "Bearer nope"})[0] == 401
        assert s.req("POST", "/completion", {"prompt": "x", "n_predict": 1})[0] == 401
    finally:
        s.stop()
    kf = tempfile.NamedTemporaryFile("w", delete=False); kf.write("# comment\nfilekey\n"); kf.close()
    s = Srv(qv_cmd("--api-key-file", kf.name), name="authfile")
    try:
        assert s.req("GET", "/props")[0] == 401 and s.req("GET", "/props", headers={"Authorization": "Bearer filekey"})[0] == 200
    finally:
        s.stop()


def test_queue_slots_and_disconnect():
    s = Srv(qv_cmd("--metrics", "-np", "1", "--mock-loop", "--mock-delay-us", "20000"), name="queue")
    try:
        res = {}
        def long_req(name, n):
            res[name] = s.req("POST", "/completion", {"prompt": name, "n_predict": n, "temperature": 0}, timeout=60)
        t1 = threading.Thread(target=long_req, args=("first", 25)); t1.start(); time.sleep(0.2)
        t2 = threading.Thread(target=long_req, args=("second", 3)); t2.start(); time.sleep(0.2)
        sl = s.req("GET", "/slots")[2]
        assert len(sl) == 1 and sum(1 for x in sl if x["is_processing"]) == 1
        m = s.req("GET", "/metrics", raw=True)[2].decode()
        assert re.search(r"^llamacpp:requests_processing 1$", m, re.M) and re.search(r"^llamacpp:requests_deferred 1$", m, re.M), m
        t1.join(); t2.join()
        assert res["first"][0] == 200 and res["second"][0] == 200 and res["first"][2]["tokens_predicted"] == 25
        # client disconnect while generating: next request must still work and the slot becomes idle
        c = socket.create_connection(("127.0.0.1", s.port))
        body = json.dumps({"prompt": "x", "n_predict": 500, "temperature": 0, "stream": True}).encode()
        c.sendall(b"POST /completion HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        c.recv(100); c.close()
        t0 = time.time()
        while time.time() - t0 < 10 and any(x["is_processing"] for x in s.req("GET", "/slots")[2]): time.sleep(0.1)
        assert not any(x["is_processing"] for x in s.req("GET", "/slots")[2])
        assert s.req("POST", "/completion", {"prompt": "x", "n_predict": 2, "temperature": 0})[0] == 200
    finally:
        s.stop()


def test_concurrent_requests_interleave():
    """-np 3: three requests run at the same time (one engine sequence each); a fourth waits in the queue; aborting one frees its slot."""
    s = Srv(qv_cmd("--metrics", "-np", "3", "--mock-loop", "--mock-text", "alpha beta gamma delta", "--mock-delay-us", "20000"), name="conc")
    try:
        res, t_first = {}, {}
        def stream(name, n):
            c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=60)
            c.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": name}], "max_tokens": n, "stream": True, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}), {"Content-Type": "application/json"})
            r = c.getresponse(); t0 = time.time(); evs = []
            for line in r.read().decode().split("\n\n"):
                if line.startswith("data: ") and "[DONE]" not in line: evs.append(json.loads(line[6:]))
            res[name] = (time.time() - t0, evs)
        names = ["a", "b", "c", "d"]
        ths = [threading.Thread(target=stream, args=(n, 30)) for n in names[:3]]
        for t in ths: t.start()
        time.sleep(0.3)
        sl = s.req("GET", "/slots")[2]
        assert sum(1 for x in sl if x["is_processing"]) == 3, sl
        t4 = threading.Thread(target=stream, args=("d", 5)); t4.start(); time.sleep(0.2)
        m = s.req("GET", "/metrics", raw=True)[2].decode()
        assert re.search(r"^llamacpp:requests_processing 3$", m, re.M) and re.search(r"^llamacpp:requests_deferred 1$", m, re.M), m
        for t in ths + [t4]: t.join()
        # three 30-token streams run concurrently: wall time ~ 30 steps of 20 ms (+ prefill), NOT 3x; each got its full, ordered content
        tmax = max(res[n][0] for n in "abc")
        assert tmax < 30 * 0.020 * 2.2 + 1.0, res
        for n in "abc":
            txt = "".join((e["choices"][0]["delta"].get("content") or "") for e in res[n][1] if e["choices"])
            assert txt.startswith("alpha beta gamma delta"), (n, txt)
            assert sum(1 for e in res[n][1] if e["choices"] and e["choices"][0]["delta"].get("content")) >= 20
        assert res["d"][1]
        # abort one stream: its slot frees while the others finish
        done = {}
        def victim():
            c = socket.create_connection(("127.0.0.1", s.port))
            body = json.dumps({"prompt": "v", "n_predict": 500, "temperature": 0, "stream": True}).encode()
            c.sendall(b"POST /completion HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
            c.recv(200); c.close()
        other = threading.Thread(target=stream, args=("e", 25)); other.start(); victim(); other.join()
        t0 = time.time()
        while time.time() - t0 < 10 and any(x["is_processing"] for x in s.req("GET", "/slots")[2]): time.sleep(0.1)
        assert not any(x["is_processing"] for x in s.req("GET", "/slots")[2])
        assert len(res["e"][1]) > 20
    finally:
        s.stop()


def test_slots_cache_prompt_per_slot():
    """Two conversations alternate: each keeps its own slot and KV prefix (cache_prompt), the server does not re-prefill either."""
    s = Srv(qv_cmd("-np", "2", "--mock-loop"), name="cache")
    try:
        def go(p):
            return s.req("POST", "/completion", {"prompt": p, "n_predict": 2, "temperature": 0, "cache_prompt": True})[2]
        a1 = go("conversation A first turn"); b1 = go("something else entirely B")
        assert a1["id_slot"] != b1["id_slot"]
        a2 = go("conversation A first turn" + a1["content"] + " and a second turn")   # continues A: its cached tokens are a prefix
        assert a2["id_slot"] == a1["id_slot"] and a2["timings"]["cache_n"] > 0, a2
    finally:
        s.stop()


def test_max_queue_503():
    s = Srv(qv_cmd("--max-queue", "0", "-np", "1", "--mock-loop", "--mock-delay-us", "30000"), name="mq")
    try:
        r = {}
        t = threading.Thread(target=lambda: r.update(a=s.req("POST", "/completion", {"prompt": "x", "n_predict": 20, "temperature": 0}))); t.start(); time.sleep(0.2)
        st, _, e = s.req("POST", "/completion", {"prompt": "y", "n_predict": 1})
        assert st == 503 and e["error"]["type"] == "unavailable_error"
        t.join()
        assert r["a"][0] == 200
    finally:
        s.stop()


def test_context_limit_truncated():
    s = Srv(qv_cmd("-c", "64", "-np", "1", "--mock-loop", "--mock-text", "a b c d"), name="ctx")
    try:
        r = s.req("POST", "/completion", {"prompt": "x", "n_predict": -1, "temperature": 0})[2]
        assert r["stop_type"] == "limit" and r["truncated"] is True and r["tokens_predicted"] + r["tokens_evaluated"] >= 62
    finally:
        s.stop()


def test_mtp_chunk_path_matches_plain():
    a = Srv(qv_cmd("--mock-loop"), name="plain"); b = Srv(qv_cmd("--mock-loop", "--mtp"), name="mtp")
    try:
        body = {"prompt": "x", "n_predict": 21, "temperature": 0, "stop": ["delta"]}
        ra, rb = a.req("POST", "/completion", body)[2], b.req("POST", "/completion", body)[2]
        assert ra["content"] == rb["content"] and ra["stop_type"] == rb["stop_type"]
        body = {"prompt": "x", "n_predict": 21, "temperature": 0}
        assert a.req("POST", "/completion", body)[2]["content"] == b.req("POST", "/completion", body)[2]["content"]
    finally:
        a.stop(); b.stop()


def test_mtp_lazy_grammar_tools():
    def read_log(srv):
        srv.stop()
        with open(srv.log.name, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # 1. Plain text response with tools (tool_choice=auto): MTP runs at full chunk speed
    plain_txt = "I will not call any tool, just writing a long plain answer with many words."
    a = Srv(qv_cmd("--mock-text", plain_txt), name="lazy_plain")
    b = Srv(qv_cmd("--mock-text", plain_txt, "--mtp"), env={"QV_MTP_DEBUG": "1"}, name="lazy_mtp")
    try:
        ra = a.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], max_tokens=64))[2]["choices"][0]
        rb = b.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], max_tokens=64))[2]["choices"][0]
        assert ra["message"] == rb["message"] and ra["finish_reason"] == rb["finish_reason"]
        assert "tool_calls" not in rb["message"]
        log_b = read_log(b)
        assert "use_chunk=1" in log_b and "lazy=1" in log_b
    finally:
        a.stop(); b.stop()

    # 2. Mid-chunk trigger word followed by invalid tokens: rollback + grammar masking must match plain 1-token path
    mid_txt = "Sure, let me check the current weather for your city right now.\n<tool_call>INVALID GARBAGE"
    a = Srv(qv_cmd("--mock-text", mid_txt), name="trig_plain")
    b = Srv(qv_cmd("--mock-text", mid_txt, "--mtp"), env={"QV_MTP_DEBUG": "1"}, name="trig_mtp")
    try:
        ra = a.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], max_tokens=200))[2]["choices"][0]
        rb = b.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], max_tokens=200))[2]["choices"][0]
        assert rb["finish_reason"] == "tool_calls"
        assert rb["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert isinstance(json.loads(rb["message"]["tool_calls"][0]["function"]["arguments"]), dict)
        assert rb["message"]["content"] == ra["message"]["content"]
        assert rb["finish_reason"] == ra["finish_reason"]
        assert rb["message"]["tool_calls"][0]["function"] == ra["message"]["tool_calls"][0]["function"]
        log_b = read_log(b)
        assert "use_chunk=1" in log_b and "lazy grammar triggered" in log_b
    finally:
        a.stop(); b.stop()

    # 3. Trigger at the very first token (no rollback branch)
    head_txt = "<tool_call>INVALID GARBAGE"
    a = Srv(qv_cmd("--mock-text", head_txt), name="head_plain")
    b = Srv(qv_cmd("--mock-text", head_txt, "--mtp"), env={"QV_MTP_DEBUG": "1"}, name="head_mtp")
    try:
        ra = a.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], max_tokens=200))[2]["choices"][0]
        rb = b.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], max_tokens=200))[2]["choices"][0]
        assert rb["finish_reason"] == "tool_calls"
        assert rb["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert isinstance(json.loads(rb["message"]["tool_calls"][0]["function"]["arguments"]), dict)
        assert rb["message"]["content"] == ra["message"]["content"]
        assert rb["message"]["tool_calls"][0]["function"] == ra["message"]["tool_calls"][0]["function"]
    finally:
        a.stop(); b.stop()

    # 4. Required (non-lazy) grammar must keep MTP disabled
    b = Srv(qv_cmd("--mock-text", head_txt, "--mtp"), env={"QV_MTP_DEBUG": "1"}, name="req_mtp")
    try:
        r = b.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], tool_choice="required", max_tokens=200))[2]["choices"][0]
        assert r["finish_reason"] == "tool_calls"
        log_b = read_log(b)
        assert "use_chunk=0" in log_b and "lazy=0" in log_b
    finally:
        b.stop()


def test_mtp_speculative_temp_gt_zero():
    def read_log(srv):
        srv.stop()
        with open(srv.log.name, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # 1. Plain chat with temperature > 0: speculative MTP is active and reproducible with the same seed
    plain_txt = "Speculative sampling works accurately even when temperature is above zero and tokens vary."
    b = Srv(qv_cmd("--mock-text", plain_txt, "--mtp"), env={"QV_MTP_DEBUG": "1"}, name="spec_plain")
    try:
        req_body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 48,
                    "temperature": 0.7, "top_k": 20, "top_p": 0.95, "min_p": 0.05, "seed": 42,
                    "chat_template_kwargs": NOTHINK}
        r1 = b.req("POST", "/v1/chat/completions", req_body)[2]["choices"][0]
        r2 = b.req("POST", "/v1/chat/completions", req_body)[2]["choices"][0]
        assert r1["message"] == r2["message"] and r1["finish_reason"] == r2["finish_reason"]
        # With presence_penalty != 0, engine falls back to host sampler (spec_ok=0)
        pen_body = dict(req_body, presence_penalty=0.5)
        assert b.req("POST", "/v1/chat/completions", pen_body)[0] == 200
        log_b = read_log(b)
        assert "use_chunk=1 (mtp=1 argmax_ok=0 spec_ok=1" in log_b
        assert "use_chunk=0 (mtp=1 argmax_ok=0 spec_ok=0" in log_b
    finally:
        b.stop()

    # 2. Lazy grammar tools with temperature > 0: speculative MTP until trigger, then rollback + grammar mask
    mid_txt = "Checking weather in Paris now.\n<tool_call>INVALID GARBAGE"
    b = Srv(qv_cmd("--mock-text", mid_txt, "--mtp"), env={"QV_MTP_DEBUG": "1"}, name="spec_tools")
    try:
        req_body = {"messages": [{"role": "user", "content": "weather in Paris?"}], "max_tokens": 200,
                    "temperature": 0.7, "top_k": 20, "top_p": 0.95, "min_p": 0.05, "seed": 123,
                    "tools": [WEATHER], "chat_template_kwargs": NOTHINK}
        r = b.req("POST", "/v1/chat/completions", req_body)[2]["choices"][0]
        assert r["finish_reason"] == "tool_calls"
        assert r["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert isinstance(json.loads(r["message"]["tool_calls"][0]["function"]["arguments"]), dict)
        log_b = read_log(b)
        assert "use_chunk=1 (mtp=1 argmax_ok=0 spec_ok=1 constrained=1 lazy=1" in log_b
        assert "lazy grammar triggered" in log_b
    finally:
        b.stop()


def test_mtp_grammar_enabled(tmp_path):
    def read_log(srv):
        srv.stop()
        with open(srv.log.name, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # 1. Required grammar (tool_choice="required"): T=0 matches plain 1-token output & T=0.7 keeps MTP active
    bad_txt = "I refuse to call any tool."
    a = Srv(qv_cmd("--mock-text", bad_txt), name="mg_plain")
    b = Srv(qv_cmd("--mock-text", bad_txt, "--mtp", "--mtp-grammar"), env={"QV_MTP_DEBUG": "1"}, name="mg_mtp")
    try:
        r0 = a.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], tool_choice="required", max_tokens=200))[2]["choices"][0]
        r1 = b.req("POST", "/v1/chat/completions", chat_body(tools=[WEATHER], tool_choice="required", max_tokens=200))[2]["choices"][0]
        assert r0["finish_reason"] == "tool_calls" and r1["finish_reason"] == "tool_calls"
        assert r1["message"]["content"] == r0["message"]["content"]
        assert r1["message"]["tool_calls"][0]["function"] == r0["message"]["tool_calls"][0]["function"]
        assert isinstance(json.loads(r1["message"]["tool_calls"][0]["function"]["arguments"]), dict)

        r_spec = b.req("POST", "/v1/chat/completions",
                       {"messages": [{"role": "user", "content": "weather in Paris?"}], "max_tokens": 200,
                        "temperature": 0.7, "top_k": 20, "top_p": 0.95, "min_p": 0.05, "seed": 99,
                        "tools": [WEATHER], "tool_choice": "required", "chat_template_kwargs": NOTHINK})[2]["choices"][0]
        assert r_spec["finish_reason"] == "tool_calls"
        assert r_spec["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert isinstance(json.loads(r_spec["message"]["tool_calls"][0]["function"]["arguments"]), dict)

        r_json = b.req("POST", "/v1/chat/completions", chat_body(response_format={"type": "json_object"}, max_tokens=120))[2]["choices"][0]
        assert isinstance(json.loads(r_json["message"]["content"]), dict)

        log_b = read_log(b)
        assert "use_chunk=1 (mtp=1 argmax_ok=0 spec_ok=0 mtp_gram=1" in log_b
        assert "use_chunk=1 (mtp=1 argmax_ok=0 spec_ok=1 mtp_gram=1" in log_b
    finally:
        a.stop(); b.stop()

    # 2. Auto tools via presets.ini (mtp-grammar = true): no rollback when <tool_call> triggers
    mid_txt = "Sure, let me check the weather for you.\n<tool_call>INVALID GARBAGE"
    ini = tmp_path / "presets.ini"
    ini.write_text(f"[*]\nbackend = mock\nspec = mtp\nmtp-grammar = true\n[test-model]\nmock-text = {json.dumps(mid_txt)}\n", encoding="utf-8")
    p = Srv([QV_SERVER, "--models-preset", str(ini), "--tokenizer", TOKJSON, "--port", "@PORT@", "-c", "16384"],
            env={"QV_MTP_DEBUG": "1"}, name="mg_preset")
    try:
        r = p.req("POST", "/v1/chat/completions", chat_body(model="test-model", tools=[WEATHER], max_tokens=200))[2]["choices"][0]
        assert r["finish_reason"] == "tool_calls" and r["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert isinstance(json.loads(r["message"]["tool_calls"][0]["function"]["arguments"]), dict)
        log_p = read_log(p)
        assert "mtp_gram=1" in log_p and "continuing MTP" in log_p and "rollback=" not in log_p
    finally:
        p.stop()


# ----- CLI / environment
def run_cli(args, env=None, wait=True):
    return subprocess.run([QV_SERVER] + args, capture_output=True, text=True, timeout=30, env=dict(os.environ, **(env or {})))


def test_cli_help_version_and_unknown():
    r = run_cli(["--help"]); assert r.returncode == 0 and "llama-server compatible" in r.stdout
    r = run_cli(["--version"]); assert r.returncode == 0 and r.stdout.startswith("version:")
    r = run_cli(["--definitely-not-a-flag"]); assert r.returncode == 1 and "invalid argument" in r.stderr
    r = run_cli(["-c"]); assert r.returncode == 1 and "expected value" in r.stderr


def test_cli_llama_flags_are_accepted_and_warned():
    s = Srv(qv_cmd("-ngl", "99", "-fa", "on", "--mlock", "-ub", "128", "-b", "512", "--no-mmap", "-ctk", "q8_0", "--cache-reuse", "256", "-sps", "0.5", "--rope-scale", "2",
                   "--threads-http", "8", "--timeout", "100", "--no-webui", "--log-colors", "off", "-cb", "--no-context-shift", "--swa-full", "--spec-default",
                   "-a", "myalias,other", "--chat-template-kwargs", '{"enable_thinking": false}', "--temp", "0.1", "--top-k", "7", "-n", "3", "--reasoning-budget", "-1"),
            name="flags")
    try:
        p = s.req("GET", "/props")[2]
        assert p["model_alias"] == "myalias" and s.req("GET", "/v1/models")[2]["data"][0]["id"] == "myalias" and s.req("GET", "/v1/models")[2]["data"][0]["aliases"] == ["myalias", "other"]
        prm = p["default_generation_settings"]["params"]
        assert prm["temperature"] == pytest.approx(0.1) and prm["top_k"] == 7 and prm["n_predict"] == 3
        r = s.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})[2]
        assert r["usage"]["completion_tokens"] == 3 and "reasoning_content" not in r["choices"][0]["message"]  # kwargs default: thinking off
        log = open(s.log.name).read()
        assert "warning:" in log and "-ctk" in log and "--swa-full" in log
    finally:
        s.stop()


def test_cli_environment_variables():
    port = free_port()
    p = subprocess.Popen([QV_SERVER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(
        os.environ, LLAMA_ARG_PORT=str(port), LLAMA_ARG_CTX_SIZE="1234", LLAMA_ARG_ALIAS="envmodel", LLAMA_ARG_N_PARALLEL="3", LLAMA_ARG_ENDPOINT_METRICS="1",
        LLAMA_ARG_ENDPOINT_SLOTS="0", LLAMA_ARG_BACKEND="mock", LLAMA_ARG_THREADS="2", LLAMA_ARG_CACHE_PROMPT="0"))
    # LLAMA_ARG_BACKEND is not a llama variable: the backend has no env, so this process must fail cleanly (no model) -- use a wrapper instead
    p.wait(15)
    assert p.returncode == 1
    s = Srv([QV_SERVER, "--backend", "mock", "--tokenizer", TOKJSON], env={"LLAMA_ARG_PORT": "@ENVPORT@"}, wait=False) if False else None
    port = free_port()
    p = subprocess.Popen([QV_SERVER, "--backend", "mock", "--tokenizer", TOKJSON], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(
        os.environ, LLAMA_ARG_PORT=str(port), LLAMA_ARG_CTX_SIZE="1234", LLAMA_ARG_ALIAS="envmodel", LLAMA_ARG_N_PARALLEL="3", LLAMA_ARG_ENDPOINT_METRICS="1",
        LLAMA_ARG_ENDPOINT_SLOTS="0", LLAMA_ARG_CACHE_PROMPT="0", LLAMA_API_KEY="envkey"))
    try:
        t0 = time.time(); ok = False
        while time.time() - t0 < 30 and not ok:
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=2); c.request("GET", "/props", headers={"Authorization": "Bearer envkey"}); r = c.getresponse(); b = json.loads(r.read()); ok = r.status == 200
            except (OSError, ValueError): time.sleep(0.2)
        assert ok and b["total_slots"] == 3 and b["model_alias"] == "envmodel" and b["endpoint_slots"] is False and b["endpoint_metrics"] is True
        assert b["default_generation_settings"]["n_ctx"] == 1234 // 3  # --ctx-size is shared by the slots
    finally:
        p.send_signal(signal.SIGTERM); p.wait(10)


def test_cli_refuses_unsafe_ignores():
    for args in (["--lora", "x.gguf"], ["--ssl-key-file", "k", "--ssl-cert-file", "c"], ["-hf", "a/b"]):
        r = run_cli(["--backend", "mock", "--tokenizer", TOKJSON] + args)
        assert r.returncode == 1 and "error:" in r.stderr, args


# =============================================================================================== part 1b: web UI, streams, Anthropic / Responses
import gzip, hashlib

GZ = {"Accept-Encoding": "gzip"}


def ui_get(s, path, headers=None, method="GET"):
    st, hd, b = s.req(method, path, headers=headers, raw=True)
    return st, hdrs(hd), b


def test_webui_routes_and_headers():
    s = Srv(qv_cmd(), name="ui")
    try:
        st, h, b = ui_get(s, "/", GZ)
        assert st == 200
        assert h["content-type"] == "text/html; charset=utf-8" and h["content-encoding"] == "gzip"
        assert h["cache-control"] == "no-cache"
        assert h["cross-origin-embedder-policy"] == "require-corp" and h["cross-origin-opener-policy"] == "same-origin"
        assert re.fullmatch(r'"[0-9a-f]{64}"', h["etag"]) and h["etag"] == '"%s"' % hashlib.sha256(b).hexdigest()   # ETag = SHA-256 of the served (gzip) bytes
        html = gzip.decompress(b).decode()
        assert "<!doctype html" in html.lower() and "_app/immutable" in html
        assert ui_get(s, "/index.html", GZ)[2] == b
        # gzip-only like llama-server: a client without gzip gets 415
        st, h2, b2 = ui_get(s, "/")
        assert st == 415 and h2["content-type"] == "text/plain" and b2 == b"Error: gzip is not supported by this browser"
        # conditional request
        st, h3, b3 = ui_get(s, "/", dict(GZ, **{"If-None-Match": h["etag"]}))
        assert st == 304 and b3 == b"" and h3["etag"] == h["etag"]
        assert ui_get(s, "/", dict(GZ, **{"If-None-Match": "W/" + h["etag"]}))[0] == 304
        assert ui_get(s, "/", dict(GZ, **{"If-None-Match": '"other"'}))[0] == 200
        st, hd, body = ui_get(s, "/", GZ, method="HEAD")
        assert st == 200 and body == b"" and int(hd["content-length"]) == len(b)
        # hashed asset: immutable; revalidated files: no-cache without ETag; content types by extension
        js = re.search(r'_app/immutable/entry/start\.[A-Za-z0-9_-]+\.js', html)
        assert js, "entry script not referenced by index.html"
        st, h4, b4 = ui_get(s, "/" + js.group(0), GZ)
        assert st == 200 and h4["content-type"] == "application/javascript" and h4["content-encoding"] == "gzip"
        assert h4["cache-control"] == "public, max-age=31536000, immutable" and "etag" in h4
        assert b"import" in gzip.decompress(b4)
        for path, ct in (("/build.json", "application/json"), ("/sw.js", "application/javascript"), ("/service-worker.js", "application/javascript"),
                         ("/manifest.webmanifest", "application/manifest+json"), ("/_app/version.json", "application/json")):
            st, h5, b5 = ui_get(s, path, GZ)
            assert st == 200 and h5["content-type"] == ct and h5["cache-control"] == "no-cache" and "etag" not in h5 and h5["content-encoding"] == "gzip", path
        assert json.loads(gzip.decompress(ui_get(s, "/build.json", GZ)[2]))["version"]
        for path, ct in (("/favicon.svg", "image/svg+xml"), ("/favicon.ico", "image/x-icon"), ("/pwa-192x192.png", "image/png")):
            st, h6, _ = ui_get(s, path, GZ)
            assert st == 200 and h6["content-type"] == ct, path
        assert ui_get(s, "/nope.js", GZ)[0] == 404
        st, _, b = s.req("GET", "/nope.js", headers=GZ)
        assert st == 404 and b == {"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}}
        # props advertise the UI
        _, _, p = s.req("GET", "/props")
        assert p["ui"] is True and p["ui_settings"] == {} and p["cors_proxy_enabled"] is False
        # features that are off by default
        for m in ("GET", "POST"):
            for path in ("/tools", "/cors-proxy"):
                st, _, b = s.req(m, path, {} if m == "POST" else None)
                assert st == 403 and b == {"error": {"message": "this feature is disabled", "type": "feature_disabled"}}, (m, path)
    finally:
        s.stop()


def test_webui_assets_served_while_loading():
    s = Srv(qv_cmd(), env={"QV_MOCK_LOAD_MS": "2500"}, wait=False, name="uiload")
    try:
        t0 = time.time(); got = None
        while time.time() - t0 < 2:
            try: got = ui_get(s, "/", GZ); break
            except OSError: time.sleep(0.05)
        assert got and got[0] == 200, "the UI must load while the model loads (it shows the loading state)"
        assert s.req("GET", "/props")[0] == 503
        s.wait_ready()
    finally:
        s.stop()


def test_webui_disabled():
    s = Srv(qv_cmd("--no-webui"), name="noui")
    try:
        for path in ("/", "/index.html", "/_app/version.json", "/build.json", "/favicon.svg"):
            st, h, b = ui_get(s, path, GZ)
            assert st == 404, path
            assert json.loads(b) == {"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}}
        # a browser that used the UI keeps a service worker: the update check is answered with a self-removing one
        for path in ("/sw.js", "/service-worker.js"):
            st, h, b = ui_get(s, path)
            assert st == 200 and h["content-type"] == "application/javascript" and h["cache-control"] == "no-cache" and b"registration.unregister()" in b
        assert s.req("GET", "/props")[2]["ui"] is False
    finally:
        s.stop()
    s = Srv(qv_cmd("--ui"), name="ui2")           # --ui / --webui spellings, env var
    try: assert ui_get(s, "/", GZ)[0] == 200
    finally: s.stop()
    s = Srv(qv_cmd(), env={"LLAMA_ARG_UI": "0"}, name="uienv")
    try: assert ui_get(s, "/", GZ)[0] == 404
    finally: s.stop()


def test_webui_with_api_key():
    s = Srv(qv_cmd("--api-key", "sekret"), name="uikey")
    try:
        for path in ("/", "/index.html", "/build.json", "/_app/version.json", "/manifest.webmanifest", "/health"):
            assert ui_get(s, path, GZ)[0] == 200, path           # the UI shell and /health are public, as in llama-server
        assert s.req("GET", "/props")[0] == 401 and s.req("GET", "/v1/models")[0] == 401
        assert s.req("GET", "/props", headers={"Authorization": "Bearer sekret"})[0] == 200
        assert s.req("GET", "/tools")[0] == 401
    finally:
        s.stop()
    s = Srv(qv_cmd("--no-webui", "--api-key", "sekret"), name="noui_key")
    try: assert ui_get(s, "/", GZ)[0] == 404                    # public path: passes the key check, then 404 (llama-server frontend_paths)
    finally: s.stop()


def test_webui_path_dir_and_api_prefix():
    d = tempfile.mkdtemp(prefix="qvpublic_")
    open(os.path.join(d, "index.html"), "w").write("<html>custom</html>")
    open(os.path.join(d, "a.js"), "w").write("var x=1;")
    s = Srv(qv_cmd("--path", d), name="uipath")
    try:
        st, h, b = ui_get(s, "/")
        assert st == 200 and b == b"<html>custom</html>" and h["content-type"].startswith("text/html") and "content-encoding" not in h
        st, h, b = ui_get(s, "/a.js")
        assert st == 200 and b == b"var x=1;" and "javascript" in h["content-type"]
        assert ui_get(s, "/_app/version.json")[0] == 404         # embedded assets are not served when --path is given
        assert ui_get(s, "/../etc/passwd")[0] == 404
        assert ui_get(s, "/sw.js")[0] == 200                      # self-removing worker when --path replaces the UI
    finally:
        s.stop()
    s = Srv(qv_cmd("--api-prefix", "/qv"), wait=False, name="prefix")
    try:
        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                if s.req("GET", "/qv/health")[0] == 200: break
            except OSError: pass
            time.sleep(0.1)
        assert ui_get(s, "/qv/", GZ)[0] == 200 and ui_get(s, "/qv/index.html", GZ)[0] == 200
        assert ui_get(s, "/", GZ)[0] == 404
        assert s.req("GET", "/qv/props")[0] == 200
    finally:
        s.stop()


def test_ui_config_flags():
    s = Srv(qv_cmd("--ui-config", '{"showThoughtInProgress": true, "x": [1, 2]}'), name="uicfg")
    try: assert s.req("GET", "/props")[2]["ui_settings"] == {"showThoughtInProgress": True, "x": [1, 2]}
    finally: s.stop()
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); f.write('{"theme": "dark"}'); f.close()
    s = Srv(qv_cmd("--webui-config-file", f.name), name="uicfgf")
    try: assert s.req("GET", "/props")[2]["ui_settings"] == {"theme": "dark"}
    finally: s.stop()
    s = Srv(qv_cmd(), env={"LLAMA_ARG_UI_CONFIG": '{"a": 1}'}, name="uicfge")
    try: assert s.req("GET", "/props")[2]["ui_settings"] == {"a": 1}
    finally: s.stop()
    r = subprocess.run([c.replace("@PORT@", str(free_port())) for c in qv_cmd("--ui-config", "{bad")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "UI config" in r.stderr


def sse_events(raw):
    """Parse event-style SSE ('event: x\\ndata: {...}') into [(event, json)]."""
    out = []
    for blk in raw.decode().split("\n\n"):
        if not blk.strip(): continue
        ev, data = None, None
        for ln in blk.split("\n"):
            if ln.startswith("event: "): ev = ln[7:]
            elif ln.startswith("data: "): data = json.loads(ln[6:])
        out.append((ev, data))
    return out


@pytest.fixture(scope="module")
def msrv():
    s = Srv(qv_cmd("--mock-text", "I should greet.\n</think>\n\nHello there, friend!", "--metrics"), name="msg")
    yield s
    s.stop()


def test_anthropic_messages(msrv):
    st, h, b = msrv.req("POST", "/v1/messages", {"model": "x", "max_tokens": 64, "temperature": 0, "system": "be brief", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
    assert st == 200 and hdrs(h)["content-type"] == "application/json; charset=utf-8"
    assert b["type"] == "message" and b["role"] == "assistant" and b["stop_reason"] == "end_turn" and b["stop_sequence"] is None
    assert [x["type"] for x in b["content"]] == ["thinking", "text"] and b["content"][0]["signature"] == ""
    assert b["content"][1]["text"] == "Hello there, friend!"
    assert set(b["usage"]) == {"cache_read_input_tokens", "input_tokens", "output_tokens"} and b["usage"]["output_tokens"] > 0
    st, _, b = msrv.req("POST", "/v1/messages", {"max_tokens": 3, "temperature": 0, "messages": [{"role": "user", "content": "hi"}]})
    assert st == 200 and b["stop_reason"] == "max_tokens" and b["usage"]["output_tokens"] == 3
    st, _, raw = msrv.req("POST", "/v1/messages", {"max_tokens": 64, "temperature": 0, "stream": True, "messages": [{"role": "user", "content": "hi"}]}, raw=True)
    assert st == 200
    ev = sse_events(raw)
    types = [e for e, _ in ev]
    assert types[0] == "message_start" and types[-2:] == ["message_delta", "message_stop"] and all(d["type"] == e for e, d in ev)
    assert [d["content_block"]["type"] for e, d in ev if e == "content_block_start"] == ["thinking", "text"]
    assert [d["index"] for e, d in ev if e == "content_block_stop"] == [0, 1]
    assert "".join(d["delta"]["text"] for e, d in ev if e == "content_block_delta" and d["delta"]["type"] == "text_delta") == "Hello there, friend!"
    assert any(d["delta"]["type"] == "signature_delta" for e, d in ev if e == "content_block_delta")
    md = [d for e, d in ev if e == "message_delta"][0]
    assert md["delta"] == {"stop_reason": "end_turn", "stop_sequence": None} and md["usage"]["output_tokens"] > 0
    st, _, b = msrv.req("POST", "/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "hi"}]})
    assert st == 200 and list(b) == ["input_tokens"] and b["input_tokens"] > 0
    st, _, b = msrv.req("POST", "/v1/messages", {"max_tokens": 5})
    assert st == 400 and b["error"]["message"] == "'messages' is required"


def test_anthropic_tool_use_roundtrip():
    s = Srv(qv_cmd("--mock-text", CALL), name="atool")
    try:
        tools = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
        body = {"max_tokens": 200, "temperature": 0, "tools": tools, "chat_template_kwargs": NOTHINK, "messages": [{"role": "user", "content": "weather in Paris?"}]}
        st, _, b = s.req("POST", "/v1/messages", body)
        assert st == 200 and b["stop_reason"] == "tool_use"
        tu = [x for x in b["content"] if x["type"] == "tool_use"][0]
        assert tu["name"] == "get_weather" and tu["input"] == {"city": "Paris"} and tu["id"]
        st, _, raw = s.req("POST", "/v1/messages", dict(body, stream=True), raw=True)
        ev = sse_events(raw)
        assert [d["content_block"]["type"] for e, d in ev if e == "content_block_start"] == ["tool_use"]
        assert "".join(d["delta"]["partial_json"] for e, d in ev if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta") .replace(" ", "") == '{"city":"Paris"}'
        assert [d for e, d in ev if e == "message_delta"][0]["delta"]["stop_reason"] == "tool_use"
        # tool_result round trip is accepted
        msgs = body["messages"] + [{"role": "assistant", "content": [{"type": "tool_use", "id": tu["id"], "name": "get_weather", "input": {"city": "Paris"}}]},
                                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu["id"], "content": "sunny"}]}]
        st, _, b = s.req("POST", "/v1/messages", dict(body, messages=msgs))
        assert st == 200
    finally:
        s.stop()


def test_responses_api(msrv):
    st, h, b = msrv.req("POST", "/v1/responses", {"input": "hi", "max_output_tokens": 64, "temperature": 0})
    assert st == 200 and b["object"] == "response" and b["status"] == "completed" and b["id"].startswith("resp_")
    assert [o["type"] for o in b["output"]] == ["reasoning", "message"]
    assert b["output"][1]["content"][0] == {"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello there, friend!"}
    assert set(b["usage"]) == {"input_tokens", "output_tokens", "total_tokens", "input_tokens_details"}
    st, _, raw = msrv.req("POST", "/v1/responses", {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}], "stream": True, "temperature": 0}, raw=True)
    ev = sse_events(raw)
    types = [e for e, _ in ev]
    assert types[:2] == ["response.created", "response.in_progress"] and types[-1] == "response.completed"
    for want in ("response.output_item.added", "response.reasoning_text.delta", "response.content_part.added", "response.output_text.delta", "response.output_text.done",
                 "response.content_part.done", "response.output_item.done"):
        assert want in types, want
    assert "".join(d["delta"] for e, d in ev if e == "response.output_text.delta") == "Hello there, friend!"
    done = ev[-1][1]["response"]
    assert done["status"] == "completed" and [o["type"] for o in done["output"]] == ["reasoning", "message"]
    assert msrv.req("POST", "/v1/responses/input_tokens", {"input": "hi"})[2]["object"] == "response.input_tokens"
    st, _, b = msrv.req("POST", "/v1/responses", {"input": "x", "previous_response_id": "r1"})
    assert st == 400 and "previous_response_id" in b["error"]["message"]
    st, _, b = msrv.req("POST", "/v1/responses", {})
    assert st == 400 and b["error"]["message"] == "'input' is required"


def test_stream_progress_timings_and_ping():
    s = Srv(qv_cmd("--mock-text", "one two three four", "--mock-delay-us", "1200000"), name="ping")
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 3, "return_progress": True, "timings_per_token": True, "temperature": 0,
                "sse_ping_interval": 1, "chat_template_kwargs": NOTHINK}
        st, _, raw = s.req("POST", "/v1/chat/completions", body, raw=True)
        txt = raw.decode()
        assert st == 200 and ":\n\n" in txt, "SSE comment ping while the stream is silent"
        ev = [json.loads(x[6:]) for x in txt.split("\n\n") if x.startswith("data: {")]
        prog = [e["prompt_progress"] for e in ev if "prompt_progress" in e]
        assert len(prog) >= 2 and set(prog[0]) == {"total", "cache", "processed", "time_ms"} and prog[0]["processed"] == prog[0]["cache"] and prog[-1]["processed"] == prog[-1]["total"]
        tm = [e["timings"] for e in ev if "timings" in e and e.get("choices") and e["choices"][0]["delta"].get("content")]
        assert tm and set(tm[0]) == {"cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"}
        assert [t["predicted_n"] for t in tm] == sorted(t["predicted_n"] for t in tm)
    finally:
        s.stop()


def test_resumable_stream_conversation_id():
    s = Srv(qv_cmd("--mock-text", "alpha beta gamma delta epsilon zeta eta theta", "--mock-delay-us", "150000"), name="resume")
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "temperature": 0, "chat_template_kwargs": NOTHINK}
        c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=30)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json", "X-Conversation-Id": "conv1::m"})
        r = c.getresponse()
        first = r.read1(4096) if hasattr(r, "read1") else r.read(1)
        assert b"data:" in first
        c.close()                                                   # the browser tab goes away mid-generation
        st, _, lk = s.req("POST", "/v1/streams/lookup", {"conversation_ids": ["conv1", "other"]})
        assert st == 200 and len(lk) == 1 and lk[0]["conversation_id"] == "conv1::m" and set(lk[0]) == {"conversation_id", "is_done", "total_bytes", "started_at", "completed_at"}
        st, h, raw = s.req("GET", "/v1/stream?conv_id=conv1::m&from=0", raw=True)         # replay + live tail until the generation finishes
        assert st == 200 and hdrs(h)["content-type"] == "text/event-stream"
        txt = raw.decode()
        assert txt.rstrip().endswith("data: [DONE]")
        got = "".join(json.loads(x[6:])["choices"][0]["delta"].get("content") or "" for x in txt.split("\n\n") if x.startswith("data: {") and json.loads(x[6:]).get("choices"))
        assert got == "alpha beta gamma delta epsilon zeta eta theta"
        st, _, lk = s.req("POST", "/v1/streams/lookup", {"conversation_ids": ["conv1"]})
        assert lk[0]["is_done"] is True and lk[0]["total_bytes"] == len(raw) + 0 or lk[0]["total_bytes"] >= len(raw)
        off = len(raw.decode().split("\n\n")[0]) + 2
        st, _, raw2 = s.req("GET", "/v1/stream?conv_id=conv1::m&from=%d" % off, raw=True)
        assert st == 200 and raw2 == raw[off:]
        assert s.req("GET", "/v1/stream?conv_id=nope")[0] == 404 and s.req("GET", "/v1/stream")[0] == 400
        st, _, b = s.req("DELETE", "/v1/stream?conv_id=conv1::m", raw=True)
        assert st == 204 and s.req("GET", "/v1/stream?conv_id=conv1::m")[0] == 404
        # explicit stop while generating: DELETE cancels the producer
        c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=30)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json", "X-Conversation-Id": "conv2"})
        r = c.getresponse(); r.read1(4096); c.close()
        t0 = time.time()
        assert s.req("DELETE", "/v1/stream?conv_id=conv2", raw=True)[0] == 204
        while time.time() - t0 < 5:
            sl = s.req("GET", "/slots")[2]
            if not any(x["is_processing"] for x in sl): break
            time.sleep(0.1)
        assert not any(x["is_processing"] for x in s.req("GET", "/slots")[2]), "DELETE /v1/stream must stop the generation"
        # without the header a disconnect aborts the generation (existing behaviour)
    finally:
        s.stop()


def test_reasoning_control_endpoint():
    s = Srv(qv_cmd("--mock-text", "hmm " * 60, "--mock-delay-us", "30000"), name="ctl")
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "reasoning_control": True, "temperature": 0}
        c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=30)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse(); first = r.read1(4096).decode()
        cid = re.search(r'"id":"(chatcmpl-[A-Za-z0-9]+)"', first).group(1)
        st, _, b = s.req("POST", "/v1/chat/completions/control", {"id": cid, "action": "reasoning_end"})
        assert st == 200 and b == {"success": True}
        rest = r.read().decode(); c.close()
        assert '"content":"' in rest and '"reasoning_content"' in first + rest, "after reasoning_end the rest of the answer is plain content"
        assert s.req("POST", "/v1/chat/completions/control", {"id": "nope", "action": "reasoning_end"})[2] == {"success": False, "message": "no active completion for this id"}
        assert s.req("POST", "/v1/chat/completions/control", {"id": "x", "action": "bogus"})[0] == 400
        assert s.req("POST", "/v1/chat/completions/control", {"action": "reasoning_end"})[0] == 400
    finally:
        s.stop()


def test_sleep_idle_seconds():
    s = Srv(qv_cmd("--sleep-idle-seconds", "1", "--metrics"), name="sleep")
    try:
        assert s.req("GET", "/props")[2]["is_sleeping"] is False
        t0 = time.time()
        while time.time() - t0 < 6 and not s.req("GET", "/props")[2]["is_sleeping"]: time.sleep(0.2)
        assert s.req("GET", "/props")[2]["is_sleeping"] is True
        assert s.req("GET", "/health")[0] == 200 and s.req("GET", "/v1/models")[0] == 200 and s.req("GET", "/metrics")[0] == 200
        assert s.req("GET", "/props")[2]["is_sleeping"] is True          # exempt endpoints do not wake the model
        st, _, b = s.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
        assert st == 200
        assert s.req("GET", "/props")[2]["is_sleeping"] is False
    finally:
        s.stop()


def test_models_preset_routing(tmp_path):
    ini = tmp_path / "presets.ini"
    ini.write_text(
        "[*]\n"
        "backend = mock\n"
        f"tokenizer = {TOKJSON}\n"
        "max-concurrency = 2\n"
        "default-max-tokens = 4\n"
        "[m1]\n"
        "model = dummy1.qv\n"
        "mock-text = alpha\n"
        "[m2]\n"
        "model = dummy2.qv\n"
        "mock-text = beta\n"
        "max-context = 8192\n"
    )
    s = Srv([QV_SERVER, "--port", "@PORT@", "--models-preset", str(ini)], name="preset")
    try:
        st, _, m = s.req("GET", "/v1/models")
        assert st == 200
        by_id = {d["id"]: d for d in m["data"]}
        assert set(by_id) == {"m1", "m2"}
        assert by_id["m1"]["status"]["value"] == "unloaded"
        assert by_id["m2"]["status"]["value"] == "unloaded"

        st, _, r1 = s.req("POST", "/v1/chat/completions", {"model": "m1", "messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"enable_thinking": False}})
        assert st == 200 and r1["model"] == "m1" and "alpha" in r1["choices"][0]["message"]["content"]
        by_id = {d["id"]: d for d in s.req("GET", "/v1/models")[2]["data"]}
        assert by_id["m1"]["status"]["value"] == "loaded" and by_id["m2"]["status"]["value"] == "unloaded"

        st, _, r2 = s.req("POST", "/v1/chat/completions", {"model": "m2", "messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"enable_thinking": False}})
        assert st == 200 and r2["model"] == "m2" and "beta" in r2["choices"][0]["message"]["content"]
        assert s.req("GET", "/props")[2]["default_generation_settings"]["n_ctx"] == 8192 // 2  # max-context is shared by the 2 slots (max-concurrency)

        by_id = {d["id"]: d for d in s.req("GET", "/v1/models")[2]["data"]}
        assert by_id["m1"]["status"]["value"] == "unloaded" and by_id["m2"]["status"]["value"] == "loaded"

        assert s.req("POST", "/v1/chat/completions", {"model": "nope", "messages": [{"role": "user", "content": "hi"}]})[0] == 404
        assert s.req("POST", "/models/unload", {})[0] == 200
        assert all(d["status"]["value"] == "unloaded" for d in s.req("GET", "/v1/models")[2]["data"])
        assert s.req("POST", "/models/load", {"model": "m1"})[0] == 200
        assert s.req("GET", "/v1/models")[2]["data"][0]["status"]["value"] == "loaded"
    finally:
        s.stop()


# =============================================================================================== part 2: structural diff vs llama-server
def shape(v):
    if isinstance(v, dict): return {k: shape(x) for k, x in v.items()}
    if isinstance(v, list): return [shape(v[0])] if v else []
    if isinstance(v, bool): return "bool"
    if v is None: return "null"
    if isinstance(v, (int, float)): return "number"
    return "string"


def diff(a, b, path=""):
    """a = llama shape, b = ours. returns list of (kind, path)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b: out.append(("missing", f"{path}/{k}"))
            else: out += diff(a[k], b[k], f"{path}/{k}")
        for k in b:
            if k not in a: out.append(("extra", f"{path}/{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if a and b: out += diff(a[0], b[0], path + "[]")
    elif a != b:
        out.append(("type", f"{path}: llama={a} ours={b}"))
    return out


# documented, intentional differences: (case, kind, json-path regex). Anything else fails the test.
ALLOWED = [
    (r"POST /v1/chat/completions thinking", "(extra|missing)", r".*/(reasoning_content|content)$"),   # ours extracts reasoning for the Qwen3.8 thinking template
    (r"GET /props", "(missing|extra|type)", r"^/(chat_template_caps/.*|media_marker|model_ftype)$"),
    (r"GET /v1/models|GET /models", "(missing|extra|type)", r"^/(data\[\]/(meta/.*)|models\[\]/details/.*)$"),
    (r".*", "(missing|extra|type)", r"^/(generation_settings|default_generation_settings/params)/(chat_format|reasoning_format|samplers)$"),
]
HDR_IGNORE = {"server", "date", "content-length", "access-control-allow-origin", "keep-alive", "connection", "accept-ranges", "process-start-time-unix"}


def allowed(case, kind, path):
    return any(re.fullmatch(c, case) and re.fullmatch(k, kind) and re.fullmatch(p, path) or (re.fullmatch(c, case) and re.fullmatch(k, kind) and re.search(p, path)) for c, k, p in ALLOWED)


CASES = []
def case(name, method, path, body=None, stream=False, headers=None):
    CASES.append((name, method, path, body, stream, headers))


CHAT = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}  # (llama ignores the kwarg: Qwen2.5 template)
for m, p in (("GET", "/health"), ("GET", "/v1/health"), ("GET", "/props"), ("GET", "/slots"), ("GET", "/v1/models"), ("GET", "/models"), ("GET", "/metrics"),
             ("GET", "/lora-adapters"), ("GET", "/nonexistent"), ("GET", "/completion"), ("DELETE", "/props")):
    case(f"{m} {p}", m, p)
case("POST /props", "POST", "/props", {})
case("POST /completion", "POST", "/completion", {"prompt": "Hello", "n_predict": 3, "temperature": 0})
case("POST /completion n_probs", "POST", "/completion", {"prompt": "Hello", "n_predict": 2, "temperature": 0, "n_probs": 2})
case("POST /completion return_tokens", "POST", "/completion", {"prompt": "Hello", "n_predict": 2, "temperature": 0, "return_tokens": True})
case("POST /completion stop", "POST", "/completion", {"prompt": "Hello", "n_predict": 6, "temperature": 0, "stop": ["zzzz"]})
case("POST /completion stream", "POST", "/completion", {"prompt": "Hello", "n_predict": 3, "temperature": 0, "stream": True}, stream=True)
case("POST /completion stream n_probs", "POST", "/completion", {"prompt": "Hello", "n_predict": 2, "temperature": 0, "stream": True, "n_probs": 2}, stream=True)
case("POST /completion tokens", "POST", "/completion", {"prompt": [9707, 1879], "n_predict": 2, "temperature": 0})
case("POST /completion mixed", "POST", "/completion", {"prompt": ["Hello", 1879], "n_predict": 2, "temperature": 0})
case("POST /completion samplers", "POST", "/completion", {"prompt": "Hello", "n_predict": 2, "typical_p": 0.9, "mirostat": 0, "xtc_probability": 0.0, "seed": 7})
case("POST /v1/completions", "POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 3, "temperature": 0})
case("POST /v1/completions logprobs", "POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 2, "temperature": 0, "logprobs": 2})
case("POST /v1/completions stream", "POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 3, "temperature": 0, "stream": True}, stream=True)
case("POST /v1/chat/completions", "POST", "/v1/chat/completions", CHAT)
case("POST /chat/completions", "POST", "/chat/completions", CHAT)
case("POST /v1/chat/completions stream", "POST", "/v1/chat/completions", dict(CHAT, stream=True), stream=True)
case("POST /v1/chat/completions stream usage", "POST", "/v1/chat/completions", dict(CHAT, stream=True, stream_options={"include_usage": True}), stream=True)
case("POST /v1/chat/completions logprobs", "POST", "/v1/chat/completions", dict(CHAT, logprobs=True, top_logprobs=2))
case("POST /v1/chat/completions n=2", "POST", "/v1/chat/completions", dict(CHAT, n=2))
case("POST /v1/chat/completions parts", "POST", "/v1/chat/completions", {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "Hi"}]}], "max_tokens": 2, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
case("POST /v1/chat/completions stop", "POST", "/v1/chat/completions", dict(CHAT, stop=["zzzz"]))
case("POST /v1/chat/completions response_format", "POST", "/v1/chat/completions", dict(CHAT, response_format={"type": "text"}))
case("POST /v1/chat/completions max_completion_tokens", "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_completion_tokens": 2, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
case("POST /tokenize", "POST", "/tokenize", {"content": "Hello world"})
case("POST /tokenize pieces", "POST", "/tokenize", {"content": "Hello world", "with_pieces": True, "add_special": True})
case("POST /tokenize empty", "POST", "/tokenize", {})
case("POST /detokenize", "POST", "/detokenize", {"tokens": [9707, 1879]})
case("POST /detokenize empty", "POST", "/detokenize", {})
case("POST /apply-template", "POST", "/apply-template", {"messages": [{"role": "user", "content": "Hi"}]})
case("POST /v1/chat/completions/input_tokens", "POST", "/v1/chat/completions/input_tokens", {"messages": [{"role": "user", "content": "Hi"}]})
case("POST /infill", "POST", "/infill", {"input_prefix": "def f(", "input_suffix": "):\n    pass", "n_predict": 2, "temperature": 0})
case("POST /infill missing", "POST", "/infill", {"input_suffix": "x"})
case("POST /slots save (disabled)", "POST", "/slots/0?action=erase", {})
case("POST /embedding", "POST", "/embedding", {"content": "hi"})
case("POST /v1/embeddings", "POST", "/v1/embeddings", {"input": "hi"})
case("POST /rerank", "POST", "/rerank", {})
case("POST /lora-adapters", "POST", "/lora-adapters", [])
# error cases
case("ERR bad json", "POST", "/v1/chat/completions", "{bad")
case("ERR no messages", "POST", "/v1/chat/completions", {})
case("ERR messages not array", "POST", "/v1/chat/completions", {"messages": "x"})
case("ERR content missing", "POST", "/v1/chat/completions", {"messages": [{"role": "user"}]})
case("ERR max_tokens type", "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": "a"})
case("ERR completion no prompt", "POST", "/completion", {})
case("ERR v1 completion no prompt", "POST", "/v1/completions", {})
case("ERR temperature type", "POST", "/completion", {"prompt": "x", "temperature": "x"})
case("ERR stream type", "POST", "/completion", {"prompt": "x", "stream": "x"})
case("ERR tokenize type", "POST", "/tokenize", {"content": 123})
case("ERR detokenize type", "POST", "/detokenize", {"tokens": "x"})
case("ERR apply-template empty", "POST", "/apply-template", {})
case("ERR top_logprobs", "POST", "/v1/chat/completions", dict(CHAT, top_logprobs=2))
case("ERR ctx exceeded", "POST", "/completion", {"prompt": "word " * 6000, "n_predict": 1})
case("ERR reasoning_format", "POST", "/v1/chat/completions", dict(CHAT, reasoning_format="bogus"))
case("POST /v1/chat/completions thinking", "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "temperature": 0})
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
TCHAT = dict(CHAT, messages=[{"role": "user", "content": "What is the weather in Paris?"}], max_tokens=80, tools=TOOLS, tool_choice="required")
case("POST /v1/chat/completions tools required", "POST", "/v1/chat/completions", TCHAT)
case("POST /v1/chat/completions tools required stream", "POST", "/v1/chat/completions", dict(TCHAT, stream=True), stream=True)
case("POST /v1/chat/completions tools forced", "POST", "/v1/chat/completions", dict(TCHAT, tool_choice={"type": "function", "function": {"name": "get_weather"}}))
case("POST /v1/chat/completions tools parallel", "POST", "/v1/chat/completions", dict(TCHAT, parallel_tool_calls=True))
case("POST /v1/chat/completions tools none", "POST", "/v1/chat/completions", dict(CHAT, tools=TOOLS, tool_choice="none"))
case("POST /v1/chat/completions tools history", "POST", "/v1/chat/completions", dict(CHAT, tools=TOOLS, messages=[
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}]))
case("POST /apply-template tools", "POST", "/apply-template", {"messages": [{"role": "user", "content": "Hi"}], "tools": TOOLS})
case("POST /v1/chat/completions json_object", "POST", "/v1/chat/completions", dict(CHAT, max_tokens=60, response_format={"type": "json_object"}))
case("POST /v1/chat/completions json_schema", "POST", "/v1/chat/completions", dict(CHAT, max_tokens=60, response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}}))
case("POST /v1/chat/completions json_object stream", "POST", "/v1/chat/completions", dict(CHAT, max_tokens=60, response_format={"type": "json_object"}, stream=True), stream=True)
case("POST /completion grammar", "POST", "/completion", {"prompt": "Hello", "n_predict": 5, "temperature": 0, "grammar": 'root ::= "yes" | "no"'})
case("POST /completion json_schema", "POST", "/completion", {"prompt": "Hello", "n_predict": 30, "temperature": 0, "json_schema": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}})
case("ERR bad tool_choice", "POST", "/v1/chat/completions", dict(CHAT, tools=TOOLS, tool_choice="bogus"))
case("ERR bad response_format", "POST", "/v1/chat/completions", dict(CHAT, response_format={"type": "bogus"}))
case("POST /completion grammar+schema", "POST", "/completion", {"prompt": "x", "n_predict": 3, "grammar": 'root ::= "a"', "json_schema": {}})
case("ERR chat grammar and schema", "POST", "/v1/chat/completions", dict(CHAT, grammar='root ::= "a"', json_schema={}))
case("OPTIONS", "OPTIONS", "/v1/chat/completions")
case("HEAD /health", "HEAD", "/health")


def parse_resp(srv, c):
    name, method, path, body, stream, headers = c
    st, hd, raw = srv.req(method, path, body, headers, raw=True)
    h = hdrs(hd)
    ct = h.get("content-type", "")
    if stream and st == 200:
        evs = []
        for blk in raw.decode().split("\n\n"):
            if blk: evs.append(blk[6:] if blk.startswith("data: ") else "BADFRAME:" + blk)
        shapes = []
        for e in evs:
            if e != "[DONE]" and not e.startswith("BADFRAME"):   # tool-call argument continuations: chunk count depends on the generated text, not on the protocol
                j = json.loads(e)
                d = j["choices"][0].get("delta", {}) if j.get("choices") else {}
                if d.get("tool_calls") and "id" not in d["tool_calls"][0]: continue
            s = e if (e == "[DONE]" or e.startswith("BADFRAME")) else shape(json.loads(e))
            if not shapes or shapes[-1] != s: shapes.append(s)
        return st, h, ("sse", shapes, evs)
    if ct.startswith("application/json") and raw:
        return st, h, ("json", shape(json.loads(raw)), None)
    return st, h, ("raw", raw.decode(errors="replace")[:0], None)


@pytest.fixture(scope="module")
def pair():
    if not HAVE_LLAMA: pytest.skip("llama-server / gguf not available (set LLAMA_SERVER, LLAMA_GGUF)")
    slots = tempfile.mkdtemp(prefix="cmp_slots_")
    common = ["--metrics", "-c", "4096"]
    ll = Srv([LLAMA_SERVER, "-m", LLAMA_GGUF, "--port", "@PORT@", "-t", "4", "--no-webui"] + common, name="llama")
    ours = Srv(qv_cmd(*common, "--mock-text", "Hello! How can I help you today? I am a mock model."), name="qv")
    yield ll, ours
    ll.stop(); ours.stop()


@pytest.mark.parametrize("c", CASES, ids=[c[0] for c in CASES])
def test_structure_matches_llama(pair, c):
    ll, ours = pair
    name = c[0]
    if name == "POST /v1/chat/completions parts": pass
    sa, ha, ra = parse_resp(ll, c)
    sb, hb, rb = parse_resp(ours, c)
    problems = []
    if sa != sb: problems.append(f"status llama={sa} ours={sb}")
    ca, cb = ha.get("content-type"), hb.get("content-type")
    if ca != cb: problems.append(f"content-type llama={ca!r} ours={cb!r}")
    for hname in sorted((set(ha) ^ set(hb)) - HDR_IGNORE):
        problems.append(f"header {hname}: llama={'yes' if hname in ha else 'no'} ours={'yes' if hname in hb else 'no'}")
    if ra[0] != rb[0]: problems.append(f"body kind llama={ra[0]} ours={rb[0]}")
    elif ra[0] == "json":
        problems += [f"{k} {p}" for k, p in diff(ra[1], rb[1]) if not allowed(name, k, p)]
    elif ra[0] == "sse":
        la, lb = ra[1], rb[1]
        if len(la) != len(lb): problems.append(f"SSE shape-sequence length llama={len(la)} ours={len(lb)}")
        for i, (x, y) in enumerate(zip(la, lb)):
            if isinstance(x, str) or isinstance(y, str):
                if x != y: problems.append(f"SSE event {i}: llama={x!r:.60} ours={y!r:.60}")
            else: problems += [f"SSE event {i}: {k} {p}" for k, p in diff(x, y) if not allowed(name, k, p)]
    assert not problems, "\n" + "\n".join(problems)


PW_PY = os.environ.get("PLAYWRIGHT_PY", "/tmp/pwenv/bin/python")
PW_BROWSERS = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/tmp/pwbrowsers")
BROWSER_SCRIPT = os.path.join(ROOT, "tests", "webui_browser_check.py")


def run_browser(url, *flags):
    out = tempfile.mkdtemp(prefix="webui_shots_")
    r = subprocess.run([PW_PY, BROWSER_SCRIPT, url, out] + list(flags), capture_output=True, text=True, timeout=180, env=dict(os.environ, PLAYWRIGHT_BROWSERS_PATH=PW_BROWSERS))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), out


@pytest.mark.skipif(not (os.path.exists(PW_PY) and os.path.isdir(PW_BROWSERS)), reason="playwright + chromium not installed (see tests/webui_browser_check.py)")
def test_webui_in_browser():
    """The real UI in headless Chromium: loads, shows the model, streams an answer with reasoning, runs a browser tool call, stops a generation."""
    s = Srv(qv_cmd("--mock-text", "The user says hi.\n</think>\n\nHello from **Q-Volta**!\n\n1. one\n2. two\n", "--temp", "0", "--top-k", "1", "--mock-delay-us", "20000", "-a", "mock-model"), name="browser")
    try:
        res, out = run_browser(f"http://127.0.0.1:{s.port}/")
        assert not res["page_errors"], res["page_errors"]
        assert "Hello from Q-Volta!" in res["text"].replace("\n", " ") or "Hello from" in res["text"], res["text"][-400:]
        calls = {(m, p.split("?")[0]): st for m, p, st in res["requests"]}
        assert calls[("GET", "props")] == 200 and calls[("GET", "v1/models")] == 200 and calls[("POST", "v1/chat/completions")] == 200 and calls[("POST", "v1/streams/lookup")] == 200
        assert calls[("GET", "tools")] == 403            # server tools are disabled by default, the UI copes
        assert "Settings" in res["settings_text"] and "API Key" in res["settings_text"]
        assert "Based on the llama.cpp web UI" in res["settings_text"] and "Q-Volta" in res["text"] + res["settings_text"]   # themed UI keeps the upstream credit
        for f in ("webui_home.png", "webui_streaming.png", "webui_chat.png", "webui_settings.png"): assert os.path.getsize(os.path.join(out, f)) > 5000
    finally:
        s.stop()
    s = Srv(qv_cmd("--mock-text", "I will check the time.\n</think>\n\n" + "<tool_call>\n<function=get_datetime>\n</function>\n</tool_call>", "--temp", "0", "--top-k", "1", "--mock-delay-us", "20000"), name="browser_tool")
    try:
        res, _ = run_browser(f"http://127.0.0.1:{s.port}/", "--tool", "--prompt", "what time is it?", "--wait", "5")
        n_chat = sum(1 for m, p, st in res["requests"] if p == "v1/chat/completions")
        assert n_chat >= 2, "the UI must run the browser-side tool and send the tool result back (2nd completion request)"
        assert "Current time" in res["text"]
    finally:
        s.stop()
    s = Srv(qv_cmd("--mock-text", "word " * 200, "--mock-delay-us", "100000"), name="browser_stop")
    try:
        res, _ = run_browser(f"http://127.0.0.1:{s.port}/", "--stop")
        assert not res["page_errors"]
        dels = [r for r in res["requests"] if r[0] == "DELETE" and r[1].startswith("v1/stream")]
        assert dels and dels[0][2] == 204, res["requests"]
        t0 = time.time()
        while time.time() - t0 < 5 and any(x["is_processing"] for x in s.req("GET", "/slots")[2]): time.sleep(0.1)
        assert not any(x["is_processing"] for x in s.req("GET", "/slots")[2]), "Stop in the UI must end the generation on the server"
    finally:
        s.stop()


# ---- web UI routes and Anthropic / Responses endpoints against the real llama-server (UI enabled in both)
UI_PATHS = ["/", "/index.html", "/build.json", "/sw.js", "/service-worker.js", "/manifest.webmanifest", "/_app/version.json", "/favicon.svg", "/favicon.ico", "/nope.js", "/tools", "/cors-proxy"]
UI_HDRS = ("content-type", "content-encoding", "cache-control", "cross-origin-embedder-policy", "cross-origin-opener-policy")


@pytest.fixture(scope="module")
def ui_pair():
    if not HAVE_LLAMA: pytest.skip("llama-server / gguf not available (set LLAMA_SERVER, LLAMA_GGUF)")
    ll = Srv([LLAMA_SERVER, "-m", LLAMA_GGUF, "--port", "@PORT@", "-t", "4", "-c", "2048"], name="llama_ui")
    ours = Srv(qv_cmd("--mock-text", "Hello! How can I help you today? I am a mock model."), name="qv_ui")
    yield ll, ours
    ll.stop(); ours.stop()


@pytest.mark.parametrize("enc", ["gzip", "identity"])
@pytest.mark.parametrize("path", UI_PATHS)
def test_ui_routes_match_llama(ui_pair, path, enc):
    ll, ours = ui_pair
    hd = {"Accept-Encoding": enc}
    a, b = ui_get(ll, path, hd), ui_get(ours, path, hd)
    problems = []
    if a[0] != b[0]: problems.append(f"status llama={a[0]} ours={b[0]}")
    for k in UI_HDRS:
        if a[1].get(k) != b[1].get(k): problems.append(f"{k}: llama={a[1].get(k)!r} ours={b[1].get(k)!r}")
    if ("etag" in a[1]) != ("etag" in b[1]): problems.append("etag presence differs")
    if "etag" in a[1] and not re.fullmatch(r'"[0-9a-f]{64}"', b[1].get("etag", "")): problems.append("etag format")
    for hname in sorted((set(a[1]) ^ set(b[1])) - HDR_IGNORE - {"etag"}): problems.append(f"header {hname} only in {'llama' if hname in a[1] else 'ours'}")
    if a[0] in (404, 403, 415): 
        if a[2] != b[2]: problems.append(f"error body llama={a[2][:80]!r} ours={b[2][:80]!r}")
    if path == "/sw.js" and a[0] == 200 and a[2] != b[2]: problems.append("sw.js differs")
    assert not problems, "\n" + "\n".join(problems)


def test_ui_conditional_and_hashed_assets_match_llama(ui_pair):
    ll, ours = ui_pair
    for srv_ in (ll, ours):
        st, h, b = ui_get(srv_, "/", GZ)
        assert ui_get(srv_, "/", dict(GZ, **{"If-None-Match": h["etag"]}))[0] == 304
    entries = []
    for srv_ in (ll, ours):
        html = gzip.decompress(ui_get(srv_, "/", GZ)[2]).decode()
        m = re.search(r'_app/immutable/entry/start\.[A-Za-z0-9_-]+\.js', html)
        assert m
        entries.append(m.group(0))
    a, b = ui_get(ll, "/" + entries[0], GZ), ui_get(ours, "/" + entries[1], GZ)
    assert a[0] == b[0] == 200
    for k in UI_HDRS + ("etag",): assert (k in a[1]) == (k in b[1]) and (k == "etag" or a[1].get(k) == b[1].get(k)), k


def test_ui_disabled_and_api_key_match_llama():
    if not HAVE_LLAMA: pytest.skip("llama-server / gguf not available")
    for extra in (["--no-webui"], ["--api-key", "k"], ["--no-webui", "--api-key", "k"]):
        ll = Srv([LLAMA_SERVER, "-m", LLAMA_GGUF, "--port", "@PORT@", "-t", "4", "-c", "2048"] + extra, name="llama_uiflag")
        ours = Srv(qv_cmd("--mock-text", "x", *extra), name="qv_uiflag")
        try:
            for path in ("/", "/index.html", "/build.json", "/sw.js", "/service-worker.js", "/_app/version.json", "/health", "/props", "/tools", "/v1/models", "/nope"):
                a, b = ui_get(ll, path, GZ), ui_get(ours, path, GZ)
                assert a[0] == b[0], (extra, path, a[0], b[0])
                assert a[1].get("content-type") == b[1].get("content-type"), (extra, path)
                if a[0] in (401, 403, 404): assert a[2] == b[2], (extra, path, a[2], b[2])
                if path == "/sw.js" and a[0] == 200: assert a[2] == b[2]
        finally:
            ll.stop(); ours.stop()


def collapse(evs):
    """[(event, data)] -> list of (event, key-shape) with consecutive duplicates removed."""
    out = []
    for e, d in evs:
        item = (e, json.dumps(shape(d), sort_keys=True))
        if not out or out[-1] != item: out.append(item)
    return out


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path,body", [
    ("/v1/messages", {"max_tokens": 6, "temperature": 0, "system": "be brief", "messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"enable_thinking": False}}),
    ("/v1/responses", {"input": "Hi", "max_output_tokens": 6, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}),
])
def test_anthropic_and_responses_match_llama(pair, path, body, stream):
    ll, ours = pair
    body = dict(body, stream=stream)
    a, b = ll.req("POST", path, body, raw=True), ours.req("POST", path, body, raw=True)
    assert a[0] == b[0] == 200, (a[0], b[0], a[2][:200], b[2][:200])
    assert hdrs(a[1]).get("content-type") == hdrs(b[1]).get("content-type")
    if not stream:
        problems = [f"{k} {p}" for k, p in diff(shape(json.loads(a[2])), shape(json.loads(b[2]))) if not (k == "extra" and p.endswith(("/id",)))]
        assert not problems, problems
    else:
        ea, eb = collapse(sse_events(a[2])), collapse(sse_events(b[2]))
        assert ea == eb, "\nllama: %s\nours:  %s" % (ea, eb)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"] + sys.argv[1:]))


def test_reasoning_budget_and_length_inside_reasoning():
    """finish_reason=length inside the thinking block: content "" + reasoning_content (as llama-server); --reasoning-budget closes the block."""
    s = Srv(qv_cmd("--mock-loop", "--mock-text", "alpha beta gamma delta"), name="rlen")
    try:
        r = s.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}], "max_tokens": 5})[2]["choices"][0]
        assert r["finish_reason"] == "length" and r["message"]["content"] == "" and r["message"]["reasoning_content"].startswith("alpha beta")
    finally:
        s.stop()
    s = Srv(qv_cmd("--mock-loop", "--mock-text", "alpha beta gamma delta", "--reasoning-budget", "6", "--reasoning-budget-message", "Time is up."), name="rbud")
    try:
        r = s.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}], "max_tokens": 40})[2]["choices"][0]["message"]
        assert r["reasoning_content"].endswith("Time is up.") and r["content"].strip() != ""
    finally:
        s.stop()
