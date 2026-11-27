#!/usr/bin/env python3
"""Loop benchmark for qv_server (docs/LOOPS.md).

Measures how often a generation degenerates into repetition when sampled at temperature 1.0 with the server's own default
sampling (nothing but `temperature`, `seed` and `max_tokens` is sent unless --param says so), reasoning on, through the real
OpenAI-compatible HTTP endpoint.

  loop_bench.py run    --url http://127.0.0.1:8000 --tokenizer tokenizer.json --cells short_chat,agent_8k --n 50 --out r.jsonl --label base
  loop_bench.py report r.jsonl [more.jsonl ...]            per (label, cell) loop rate with Wilson 95% intervals
  loop_bench.py selftest                                   detector / Wilson / prompt-builder checks (no server)

Loop detector (`detect_loop`): an n-gram of >= 16 tokens that occurs >= 4 times (non-overlapping) in the last 512 tokens.
Checking n = 16 is sufficient: any longer repeated n-gram contains a repeated 16-gram.  A run that reaches the token cap
without EOS is additionally classified `cap_loop` (repeating tail) or `cap_other` (no repetition detected).  By default a
generation is aborted the moment the detector fires (the server sees a client abort); --no-abort-on-loop lets it run on.

Prompt cells (all deterministic, built from fixed text and the repository files):
  short_chat            20 short everyday prompts
  reason_math           12 competition-style word problems / puzzles
  reason_code           12 small debugging / implementation tasks
  agent_8k/32k/64k/...  a tool-using coding-agent transcript (system prompt, tool definitions, assistant read_file calls with file
                        contents as tool results) of ~8K/32K/64K prompt tokens, ending with one of 6 debugging tasks
Generation i of a cell uses prompt i mod len(prompts) and seed seed0 + i, so two runs with the same arguments see the same
prompts and seeds.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

# ------------------------------------------------------------------------------------------------------------ detector
LOOP_N = 16
LOOP_REPS = 4
LOOP_WINDOW = 512


def detect_loop(ids, n=LOOP_N, reps=LOOP_REPS, window=LOOP_WINDOW):
    """Return None or (period_start, count, gram) when an n-gram repeats >= reps times (non-overlapping) in the last `window` ids."""
    tail = list(ids[-window:])
    if len(tail) < n * reps:
        return None
    pos = {}
    for i in range(len(tail) - n + 1):
        pos.setdefault(tuple(tail[i:i + n]), []).append(i)
    best = None
    for g, ps in pos.items():
        if len(ps) < reps:
            continue
        cnt, last = 0, -n
        for p in ps:
            if p >= last + n:
                cnt += 1
                last = p
        if cnt >= reps and (best is None or cnt > best[1]):
            best = (ps[0], cnt, g)
    return best


def repeat_coverage(ids, n=LOOP_N, reps=3, window=LOOP_WINDOW):
    """Fraction of the last `window` ids covered by n-grams that occur >= reps times in that window (1.0 = pure repetition)."""
    tail = list(ids[-window:])
    if len(tail) < n:
        return 0.0
    pos = {}
    for i in range(len(tail) - n + 1):
        pos.setdefault(tuple(tail[i:i + n]), []).append(i)
    cov = [False] * len(tail)
    for ps in pos.values():
        if len(ps) >= reps:
            for p in ps:
                for j in range(p, p + n):
                    cov[j] = True
    return sum(cov) / len(tail)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ------------------------------------------------------------------------------------------------------------ prompts
SHORT_CHAT = [
    "Write a haiku about rain.", "Explain what a hash map is in two paragraphs.", "Give me three tips for learning a new language.",
    "What is the difference between TCP and UDP?", "Suggest a name for a bakery that sells sourdough and write a one-line slogan.",
    "Summarise the plot of Romeo and Juliet in five sentences.", "How do I reverse a linked list? Explain the idea first.",
    "Why is the sky blue?", "Write a polite email declining a meeting invitation.", "List five uses for a Raspberry Pi.",
    "What are the pros and cons of electric cars?", "Explain recursion to a ten year old.",
    "Translate 'Where is the train station?' into French, German and Spanish.", "Describe how a bill becomes law in the United States.",
    "Give me a simple recipe for pancakes.", "What causes the seasons on Earth?", "Write a limerick about a cat that loves coffee.",
    "Explain the CAP theorem briefly.", "How does public key cryptography work?", "Recommend a weekly workout plan for a beginner.",
]
REASON_MATH = [
    "A train leaves city A at 9:00 going 60 km/h. Another leaves city B, 300 km away, at 10:00 going 90 km/h toward A. When do they meet?",
    "How many positive integers less than 1000 are divisible by 3 or 5 but not by 15?",
    "Find all real x such that x^2 - 5x + 6 < 0 and |x - 3| > 1/2.",
    "A bag has 5 red and 7 blue balls. Three are drawn without replacement. What is the probability that exactly two are red?",
    "Prove that the sum of the first n odd numbers is n^2, then use it to compute 1+3+5+...+199.",
    "What is the remainder when 7^2024 is divided by 13?",
    "Two circles of radius 5 have centres 6 apart. What is the area of their intersection? Give a closed form.",
    "Solve the system: x + y + z = 6, x*y + y*z + z*x = 11, x*y*z = 6.",
    "A rectangle has perimeter 36 and its diagonal is 13. Find its area.",
    "How many ways can 8 people be seated at a round table if two particular people must not sit next to each other?",
    "Find the smallest positive integer n such that n! is divisible by 10^7.",
    "Let a_1 = 1 and a_{n+1} = a_n + 1/a_n. Estimate a_100 and justify the estimate.",
]
REASON_CODE = [
    "Write a Python function that returns the longest palindromic substring of a string, and explain its complexity.",
    "This C code crashes sometimes: `char* s = malloc(5); strcpy(s, \"hello\"); printf(\"%s\", s); free(s);` Explain why and fix it.",
    "Implement an LRU cache in Python with O(1) get and put, without using OrderedDict.",
    "Why does `for i in range(3): fns.append(lambda: i)` give [2, 2, 2] when called? Show two ways to fix it.",
    "Write a SQL query returning the second highest salary per department from table emp(id, dept, salary).",
    "A Go program deadlocks: two goroutines each lock mutex A then B, and B then A. Explain and give a fix.",
    "Write a function that merges overlapping intervals in a list of (start, end) pairs and prove it is correct.",
    "Explain why this JavaScript prints 'undefined' and fix it: `const o = {n: 1, f() { return () => this.m; }}; console.log(o.f()());`",
    "Write a C++ template function `topk` that returns the k largest elements of a range using a heap.",
    "My binary search sometimes loops forever: `while lo < hi: mid = (lo+hi)//2; if a[mid] < x: lo = mid else: hi = mid`. Why? Fix it.",
    "Write a regular expression that matches valid IPv4 addresses and explain each part.",
    "Implement Dijkstra's algorithm in Rust using BinaryHeap and explain why Reverse is needed.",
]
AGENT_TASKS = [
    "The int4 KV attention path gives slightly different logits from the fp16 path at long positions. Look at the code you have read and tell me where a "
    "systematic error could come from. Be concrete: name the function and the line of reasoning.",
    "A user reports that generation sometimes repeats the same paragraph forever at temperature 1.0. Using the files above, list the places in the sampler, "
    "server and engine that could cause this, then pick the most likely one and propose a fix.",
    "Write a unit test for the sampler's top_p handling and explain which edge cases it covers. Debug anything suspicious in the implementation first.",
    "The prefix cache reports cached=0 for consecutive turns of the same conversation. Trace how the cache decides to reuse a prefix and find why it might miss.",
    "Review the speculative decoding code for correctness: is the accepted token distribution identical to plain sampling? Check the residual sampling step.",
    "Explain how the server turns a chat request into engine calls, and find one bug or fragile spot in that path. Then show the minimal patch.",
]
AGENT_SYSTEM = (
    "You are a coding agent working in the repository of an inference engine. You can call tools to read files, search and run commands. "
    "Think step by step, prefer reading code over guessing, and keep answers concise and concrete. When you are done, answer the user directly."
)
AGENT_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the repository.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository for a regular expression.",
                                      "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                                                     "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return its output.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
AGENT_FILES = ["src/sampling/sampler.cpp", "include/qv/spec_sampling.h", "src/server/app_gen.cpp", "src/runtime/cuda_inkling_engine.cu",
               "docs/INKLING_SPEC.md", "docs/INKLING_ENGINE.md", "docs/SERVER.md", "src/server/app_core.cpp", "src/runtime/cuda_engine.cu",
               "src/backends/cuda/inkling/ink_ops.cu", "docs/PERFORMANCE.md", "docs/INKLING_MTP.md", "src/runtime/cpu_inkling_engine.cpp",
               "src/server/app_handlers.cpp", "docs/LONG_CONTEXT.md", "src/server/app_stream.cpp", "docs/INKLING_DESIGN.md", "src/backends/cuda/inkling/ink_prefill.cu",
               "src/grammar/grammar.cpp", "src/jinja/eval.cpp", "src/server/compat_api.cpp", "docs/SERVER_COMPAT.md", "docs/INKLING_FORMAT.md",
               "src/tokenizer/tokenizer.cpp", "src/model/model_file.cpp", "tools/convert_inkling.py", "tools/inkling_ref.py", "docs/ROUTING.md"]


def agent_messages(repo, tok, target_tokens, task):
    """Chat transcript of about `target_tokens` prompt tokens: files read by the agent, then the task (last message)."""
    msgs = [{"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "I need help with this repository. Start by reading the relevant files, then I will give you the task."}]
    used = len(tok.encode(AGENT_SYSTEM).ids) + 600  # tool definitions, template, task
    k = 0
    for rel in AGENT_FILES * 3:  # repeat the list only if the repository is too small (never for <= 64K)
        if used >= target_tokens:
            break
        p = os.path.join(repo, rel)
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8", errors="replace").read()
        room = target_tokens - used
        ids = tok.encode(txt).ids
        if len(ids) > room:  # cut at a line boundary
            txt = txt[:max(0, int(len(txt) * room / len(ids)))]
            txt = txt[:txt.rfind("\n") + 1] if "\n" in txt else txt
            ids = tok.encode(txt).ids
        k += 1
        cid = "call_%d" % k
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function",
                     "function": {"name": "read_file", "arguments": json.dumps({"path": rel})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "// " + rel + "\n" + txt})
        used += len(ids) + 40
    msgs.append({"role": "user", "content": task + " Do not call any more tools: answer from the files you have already read."})
    return msgs


def cell_prompt(cell, i, repo, tok):
    """(messages, tools) for generation i of a cell."""
    if cell == "short_chat":
        return [{"role": "user", "content": SHORT_CHAT[i % len(SHORT_CHAT)]}], None
    if cell == "reason_math":
        return [{"role": "user", "content": REASON_MATH[i % len(REASON_MATH)]}], None
    if cell == "reason_code":
        return [{"role": "user", "content": REASON_CODE[i % len(REASON_CODE)]}], None
    if cell.startswith("agent_") and cell.endswith("k"):
        tgt = int(cell[6:-1]) * 1024
        return agent_messages(repo, tok, tgt, AGENT_TASKS[i % len(AGENT_TASKS)]), AGENT_TOOLS
    raise SystemExit("unknown cell " + cell)


# ------------------------------------------------------------------------------------------------------------ client
def post_stream(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def one_generation(url, tok, messages, tools, seed, args):
    body = {"model": "qv", "messages": messages, "temperature": args.temperature, "max_tokens": args.max_tokens, "stream": True,
            "seed": seed, "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    for kv in args.param:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        body[k] = v
    t0 = time.time()
    think, text = [], []
    n_chunks = 0
    finish = None
    loop = None
    usage = {}
    first_t = None
    checked = 0
    onset = None
    resp = post_stream(url, body, args.timeout)
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("reasoning_content"):
                    think.append(d["reasoning_content"])
                    n_chunks += 1
                if d.get("content"):
                    text.append(d["content"])
                    n_chunks += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if first_t is None and (think or text):
                    first_t = time.time() - t0
            if n_chunks - checked >= 16:
                checked = n_chunks
                cur = detect_loop(tail_ids(tok, think, text))
                if cur and onset is None:
                    onset, loop = n_chunks, cur
                if onset is not None and not args.no_abort_on_loop and n_chunks - onset >= args.confirm:
                    finish = "loop_abort"  # the tail has been observed for `confirm` more tokens (strong/weak classification below)
                    break
    finally:
        resp.close()
    full_ids = tok.encode("".join(think) + "".join(text), add_special_tokens=False).ids
    if loop is None:
        loop = detect_loop(full_ids)
    cov = repeat_coverage(full_ids)
    return {"cov": cov, "onset": onset, "think": "".join(think), "text": "".join(text), "ids": full_ids, "finish": finish, "loop": loop, "usage": usage,
            "ttft": first_t, "secs": time.time() - t0, "in_think": bool(think) and not text}


def tail_ids(tok, think, text, chars=8000):
    s = "".join(think) + "".join(text)
    return tok.encode(s[-chars:], add_special_tokens=False).ids


def classify(r, max_tokens):
    n = len(r["ids"])
    if r["loop"]:
        return "loop"
    if r["finish"] in ("stop", "tool_calls"):
        return "eos"
    if r["finish"] == "length" or n >= max_tokens - 2:
        return "cap_other"
    return "other:%s" % r["finish"]


def cmd_run(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    repo = args.repo
    out = open(args.out, "a")
    cells = args.cells.split(",")
    for cell in cells:
        built = {}
        for i in range(args.start, args.n):
            key = i % 1000
            seed = args.seed0 + i
            idx = i % (6 if cell.startswith("agent_") else 20)
            if idx not in built:
                built[idx] = cell_prompt(cell, idx, repo, tok)
            msgs, tools = built[idx]
            try:
                r = one_generation(args.url, tok, msgs, tools, seed, args)
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                print("request failed: %s" % e, file=sys.stderr)
                rec = {"label": args.label, "cell": cell, "i": i, "seed": seed, "error": str(e)}
                out.write(json.dumps(rec) + "\n"); out.flush()
                continue
            kind = classify(r, args.max_tokens)
            lp = r["loop"]
            rec = {"label": args.label, "cell": cell, "i": i, "seed": seed, "prompt_idx": idx, "kind": kind, "n_tokens": len(r["ids"]),
                   "finish": r["finish"], "prompt_tokens": r["usage"].get("prompt_tokens"), "secs": round(r["secs"], 1),
                   "in_think": r["in_think"], "think_chars": len(r["think"]), "text_chars": len(r["text"]),
                   "loop_count": lp[1] if lp else 0, "onset": r["onset"], "cov": round(r["cov"], 3), "strong": bool(lp and r["cov"] >= 0.5), "loop_gram": tok.decode(list(lp[2]))[:200] if lp else None,
                   "tail": (r["think"] + r["text"])[-300:] if kind != "eos" else None}
            if args.keep_text:
                rec["think"], rec["text"] = r["think"], r["text"]
            out.write(json.dumps(rec) + "\n"); out.flush()
            print("[%s %s %3d] %-9s tokens=%5d prompt=%s %.0fs" % (args.label, cell, i, kind, rec["n_tokens"], rec["prompt_tokens"], r["secs"]), flush=True)
    return 0


def cmd_report(args):
    rows = []
    for p in args.files:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    seen, uniq = set(), []
    for r in rows:  # resumed sweeps may repeat a generation: keep the first record of each (label, cell, i)
        k = (r["label"], r["cell"], r["i"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    rows = uniq
    keys = []
    for r in rows:
        k = (r["label"], r["cell"])
        if k not in keys:
            keys.append(k)
    print("| label | cell | n | loops | loop rate | Wilson 95% | strong | strong rate | strong Wilson 95% | cap_other | eos | median tokens | errors |")
    print("|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|")
    for lab, cell in keys:
        rs = [r for r in rows if r["label"] == lab and r["cell"] == cell]
        ok = [r for r in rs if "kind" in r]
        n = len(ok)
        loops = sum(r["kind"] == "loop" for r in ok)
        lo, hi = wilson(loops, n)
        toks = sorted(r["n_tokens"] for r in ok)
        med = toks[len(toks) // 2] if toks else 0
        st = sum(bool(r.get("strong")) for r in ok)
        slo, shi = wilson(st, n)
        print("| %s | %s | %d | %d | %.1f%% | %.1f-%.1f%% | %d | %.1f%% | %.1f-%.1f%% | %d | %d | %d | %d |" % (
            lab, cell, n, loops, 100.0 * loops / max(n, 1), 100 * lo, 100 * hi, st, 100.0 * st / max(n, 1), 100 * slo, 100 * shi, sum(r["kind"] == "cap_other" for r in ok),
            sum(r["kind"] == "eos" for r in ok), med, len(rs) - n))
    return 0


def cmd_selftest(args):
    rep = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    base = list(range(100, 400))
    assert detect_loop(base) is None
    assert detect_loop(base + rep * 4) is not None and detect_loop(base + rep * 4)[1] == 4
    assert detect_loop(base + rep * 3) is None  # only 3 repeats
    assert detect_loop(base + rep * 3 + [1, 2] + rep) is not None  # 4 repeats, not adjacent
    assert detect_loop(base + [7] * 80) is not None  # single-token run: non-overlapping 16-grams
    assert detect_loop(base[:200] + rep * 4 + base[200:] + list(range(1000, 1600))) is None  # repeats scrolled out of the 512-token window
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and abs(hi - 0.0713) < 5e-4, (lo, hi)
    lo, hi = wilson(5, 50)
    assert abs(lo - 0.0435) < 1e-3 and abs(hi - 0.2136) < 1e-3, (lo, hi)
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--tokenizer", default="/workspace/q-volta-models/Inkling-2-Mini/tokenizer.json")
    r.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r.add_argument("--cells", default="short_chat,reason_math,reason_code,agent_8k,agent_32k,agent_64k")
    r.add_argument("--n", type=int, default=50, help="generations per cell")
    r.add_argument("--start", type=int, default=0, help="first generation index (resume)")
    r.add_argument("--seed0", type=int, default=1000)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--max-tokens", type=int, default=8192)
    r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--param", action="append", default=[], help="extra request field key=value (JSON value), e.g. top_k=20")
    r.add_argument("--label", default="run")
    r.add_argument("--out", required=True)
    r.add_argument("--no-abort-on-loop", action="store_true")
    r.add_argument("--confirm", type=int, default=400, help="keep generating this many tokens after the detector first fires; the loop is 'strong' if >= 50%% of the last 512 tokens are then covered by repeated 16-grams")
    r.add_argument("--keep-text", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("files", nargs="+")
    sub.add_parser("selftest")
    args = ap.parse_args()
    return {"run": cmd_run, "report": cmd_report, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
