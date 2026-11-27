#!/usr/bin/env python3
"""Long-context agent-traffic replay for qv_server (p30/longctx).

Modes (stdlib only; run with `python3 -I`):
  replay   two (or more) interleaved agent sessions: a system prompt + tool declarations + a large initial document,
           then tool-call-like turns that append 50-500 new tokens each; the assistant reply the server returns is
           appended back the way an OpenAI-style client does (reasoning dropped, or kept with --history keep).
           Per request: prompt_n / cache_n / prefill ms / prefill tok/s / decode tok/s / wall ms.
  sweep    one fresh prompt per length (decode tok/s and prefill tok/s versus context length; server level).
  verify   run the same scripted conversation with cache_prompt true and false and compare the greedy replies.

Text comes from repo files (distinct, seeded shuffle per session) so nothing is repeated inside a context.
Results: one JSON line per request in --out (default stdout summary only) plus a markdown table on stdout.
"""
import argparse, json, os, random, sys, time, urllib.request, urllib.error

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file chunk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]
SYSTEM = ("You are a coding agent working in a C++/CUDA repository. Use the tools read_file and grep to inspect code, "
          "think briefly, then either call a tool or answer in one short paragraph. Keep every answer under 80 words.")


def http(url, payload=None, timeout=3600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP %d from %s: %s" % (e.code, url, e.read()[:500]))


def corpus(root, seed):
    exts = (".md", ".cpp", ".cu", ".h", ".py", ".inc", ".txt", ".json")
    files = []
    for d, dn, fn in os.walk(root):
        dn[:] = [x for x in dn if x not in (".git", "build", "third_party", "webui", "webui-theme", "node_modules", "logs", "results")]
        for f in fn:
            p = os.path.join(d, f)
            if f.endswith(exts) and 2000 < os.path.getsize(p) < 400000:
                files.append(p)
    files.sort()
    random.Random(seed).shuffle(files)
    return files


class Feed:
    """Endless supply of distinct text chunks (chars)."""
    def __init__(self, root, seed):
        self.files, self.i, self.buf = corpus(root, seed), 0, ""
        self.rng = random.Random(seed * 7 + 1)
    def take(self, n):
        while len(self.buf) < n:
            if self.i >= len(self.files):
                raise RuntimeError("corpus exhausted (use a larger --corpus-root)")
            p = self.files[self.i]; self.i += 1
            with open(p, errors="replace") as f:
                self.buf += "\n=== %s ===\n" % p + f.read().replace("<|", "< |").replace("|>", "| >")  # keep special-token text out of the prompt
        out, self.buf = self.buf[:n], self.buf[n:]
        return out


def chat(base, messages, max_tokens, cache_prompt=True, temperature=0.0, slot=None, extra=None, tools=True):
    payload = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
               "cache_prompt": cache_prompt, "stream_options": {"include_usage": True}}
    if tools: payload["tools"] = TOOLS
    if slot is not None: payload["id_slot"] = slot
    if extra: payload.update(extra)
    t0 = time.monotonic(); first = None
    text, reasoning, calls, timings, usage, fin = [], [], {}, None, None, None
    with http(base + "/v1/chat/completions", payload) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            raw = line[6:].strip()
            if raw == b"[DONE]": break
            d = json.loads(raw)
            if "error" in d: raise RuntimeError(str(d["error"]))
            if d.get("timings"): timings = d["timings"]
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if first is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls")): first = time.monotonic() - t0
                if dl.get("content"): text.append(dl["content"])
                if dl.get("reasoning_content"): reasoning.append(dl["reasoning_content"])
                for tc in dl.get("tool_calls") or []:
                    e = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": "", "args": ""})
                    if tc.get("id"): e["id"] = tc["id"]
                    f = tc.get("function", {})
                    e["name"] += f.get("name") or ""; e["args"] += f.get("arguments") or ""
                if c.get("finish_reason"): fin = c["finish_reason"]
    tm = timings or {}
    row = {"prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "prompt_ms": tm.get("prompt_ms"),
           "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
           "decode_tps": tm.get("predicted_per_second"), "ttft_s": first, "wall_s": time.monotonic() - t0, "finish": fin}
    if usage: row["usage_prompt"] = usage.get("prompt_tokens")
    pn, pm = row["prompt_n"], row["prompt_ms"]
    # llama.cpp timings: prompt_n = tokens actually evaluated (prompt minus cache_n)
    row["prefill_tps"] = (pn / (pm / 1000.0)) if pn and pm and pm > 0 else None
    return row, "".join(text), "".join(reasoning), [calls[k] for k in sorted(calls)]


def calib_chars_per_token(base, feed_root):
    f = Feed(feed_root, 99)
    s = f.take(40000)
    r = http(base + "/tokenize", {"content": s})
    n = len(json.loads(r.read())["tokens"])
    return len(s) / n


class Session:
    def __init__(self, name, args, cpt, seed):
        self.name, self.args, self.cpt = name, args, cpt
        self.feed = Feed(args.corpus_root, seed)
        self.rng = random.Random(seed)
        self.msgs = [{"role": "system", "content": SYSTEM}]
        self.turn = 0
        self.call_id = 0
    def start(self, target_tokens):
        doc = self.feed.take(int(target_tokens * self.cpt))
        for _ in range(2):  # re-calibrate on this session's own text (chars/token differs between code and prose)
            n = len(json.loads(http(self.args.url + "/tokenize", {"content": doc}).read())["tokens"])
            if abs(n - target_tokens) < 0.01 * target_tokens: break
            doc = doc[:int(len(doc) * target_tokens / n)] if n > target_tokens else doc + self.feed.take(int(len(doc) * (target_tokens - n) / n))
        self.msgs.append({"role": "user", "content": "Repository dump follows. Study it, then tell me which file you would read next.\n" + doc})
    def step(self, base, cache_prompt, slot=None):
        row, text, reasoning, calls = chat(base, self.msgs, self.args.max_new, cache_prompt=cache_prompt, slot=slot)
        row.update(session=self.name, turn=self.turn, reply_chars=len(text), reasoning_chars=len(reasoning), n_calls=len(calls))
        self.turn += 1
        am = {"role": "assistant", "content": text}
        if self.args.history == "keep" and reasoning: am["reasoning_content"] = reasoning
        if calls:
            try:
                tcs = [{"id": c["id"] or "call_%d" % i, "type": "function", "function": {"name": c["name"], "arguments": json.loads(c["args"] or "{}")}} for i, c in enumerate(calls)]
            except Exception:
                tcs = None
        else:
            tcs = None
        if not tcs:  # model answered in text: still make the turn tool-call-shaped (synthetic call)
            self.call_id += 1
            tcs = [{"id": "call_s%d" % self.call_id, "type": "function", "function": {"name": "read_file", "arguments": {"path": "src/x%d.cpp" % self.call_id, "offset": 0}}}]
        am["tool_calls"] = tcs
        self.msgs.append(am)
        n_new = self.rng.randint(self.args.new_min, self.args.new_max)
        for tc in tcs:
            self.msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": self.feed.take(int(n_new * self.cpt))})
        return row


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(("%.1f" % r[c]) if isinstance(r.get(c), float) else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def emit(args, rows):
    if args.out:
        with open(args.out, "a") as f:
            for r in rows: f.write(json.dumps(r) + "\n")


def mode_replay(args):
    cpt = args.chars_per_token or calib_chars_per_token(args.url, args.corpus_root)
    print("chars/token = %.3f" % cpt, flush=True)
    starts = [int(x) for x in args.start_tokens.split(",")]
    sess = [Session(chr(65 + i), args, cpt, 100 + i) for i in range(len(starts))]
    rows = []
    for s, st in zip(sess, starts): s.start(st)
    for turn in range(args.turns):
        for s in sess:
            slot = None
            if args.pin_slots: slot = sess.index(s) % args.pin_slots
            row = s.step(args.url, not args.no_cache_prompt, slot)
            row["label"] = args.label
            rows.append(row)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    emit(args, rows)
    print(table([dict(r, prompt=(r["prompt_n"] or 0) + (r["cache_n"] or 0)) for r in rows],
                ["session", "turn", "prompt", "cache_n", "prompt_ms", "prefill_tps", "predicted_n", "decode_tps", "wall_s"]))
    hit = sum(1 for r in rows[len(sess):] if (r["cache_n"] or 0) > 0.5 * ((r["prompt_n"] or 0) + (r["cache_n"] or 0)))
    print("hit-rate (turns after the first of each session, cache_n > 50%% of prompt): %d/%d" % (hit, len(rows) - len(sess)))
    ts = sorted(r["wall_s"] for r in rows[len(sess):])
    if ts: print("turn wall-latency s: median %.2f  p90 %.2f  max %.2f" % (ts[len(ts) // 2], ts[int(len(ts) * 0.9)], ts[-1]))


def mode_sweep(args):
    cpt = args.chars_per_token or calib_chars_per_token(args.url, args.corpus_root)
    rows = []
    for L in [int(x) for x in args.lengths.split(",")]:
        for rep in range(args.reps):
            s = Session("S", args, cpt, 500 + L + rep)
            s.start(L)
            row, text, _, _ = chat(args.url, s.msgs, args.max_new, cache_prompt=False, tools=False)
            row.update(target=L, rep=rep, label=args.label, prompt=(row["prompt_n"] or 0) + (row["cache_n"] or 0))
            rows.append(row)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    emit(args, rows)
    print(table(rows, ["target", "rep", "prompt", "prompt_ms", "prefill_tps", "predicted_n", "decode_tps"]))


def mode_verify(args):
    """cache_prompt on vs off (and off vs off, the determinism floor): same scripted conversation, compare the greedy replies."""
    cpt = args.chars_per_token or calib_chars_per_token(args.url, args.corpus_root)
    res = {}
    for name, cp in (("cached", True), ("uncached", False), ("uncached2", False)):
        sess = [Session(chr(65 + i), args, cpt, 100 + i) for i in range(2)]
        for s, st in zip(sess, [int(x) for x in args.start_tokens.split(",")]): s.start(st)
        replies = []
        for _ in range(args.turns):
            for s in sess:
                row, text, reasoning, calls = chat(args.url, s.msgs, args.max_new, cache_prompt=cp)
                replies.append((text, reasoning, [(c["name"], c["args"]) for c in calls]))  # tool-call ids are random per request: not compared
                s.turn += 1  # same deterministic shape fed back in every run (the model's own text only enters through `text`)
                s.msgs.append({"role": "assistant", "content": text, "tool_calls": [{"id": "call_v%d" % s.turn, "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.cpp"}}}]})
                s.msgs.append({"role": "tool", "tool_call_id": "call_v%d" % s.turn, "content": s.feed.take(int(s.rng.randint(args.new_min, args.new_max) * cpt))})
                print("%s %s turn %d cache_n=%s prompt_n=%s" % (name, s.name, s.turn, row["cache_n"], row["prompt_n"]), flush=True)
        res[name] = replies
    def cmp(x, y, tag):
        ok = True
        for i, (u, v) in enumerate(zip(res[x], res[y])):
            if u != v:
                ok = False
                t1, t2 = (u[1] or "") + (u[0] or ""), (v[1] or "") + (v[0] or "")
                k = next((j for j in range(min(len(t1), len(t2))) if t1[j] != t2[j]), min(len(t1), len(t2)))
                print("  reply %d differs at char %d: %r vs %r" % (i, k, t1[max(0, k - 20):k + 30], t2[max(0, k - 20):k + 30]))
        print("VERIFY %s identical: %s (%d replies)" % (tag, ok, len(res[x])))
        return ok
    c1 = cmp("cached", "uncached", "cached vs uncached")
    c2 = cmp("uncached", "uncached2", "uncached vs uncached (determinism floor)")
    sys.exit(0 if c1 else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["replay", "sweep", "verify"])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--corpus-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--chars-per-token", type=float, default=0.0)
    ap.add_argument("--start-tokens", default="60000,60000", help="initial document size per session (tokens)")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--new-min", type=int, default=50)
    ap.add_argument("--new-max", type=int, default=500)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--history", choices=["drop", "keep"], default="drop", help="whether the client sends reasoning_content back")
    ap.add_argument("--no-cache-prompt", action="store_true")
    ap.add_argument("--pin-slots", type=int, default=0, help="send id_slot = session index %% N (0: let the server choose)")
    ap.add_argument("--lengths", default="4096,16384,32768,65536,85000,128000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    {"replay": mode_replay, "sweep": mode_sweep, "verify": mode_verify}[args.mode](args)


if __name__ == "__main__":
    main()
