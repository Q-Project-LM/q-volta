#!/usr/bin/env python3
"""Decode tok/s through qv_server at default sampling for three prompt classes (code, tool-call/JSON agent turn, prose), several runs each.
Usage: bench_spec_server.py [--port 8000] [--runs 3] [--max-tokens 256] [--ctx-pad N] [--label X]
Prints one JSON line per request (decode tok/s from the server's own timings: predicted_n / predicted_ms, plus client-side tok/s after the first token)."""
import argparse, json, time, urllib.request

PROMPTS = {
    'code': 'Write a Python module implementing an LRU cache class with get, put and a decorator version, with type hints and docstrings. Code only.',
    'tool': None,
    'prose': 'Write a detailed essay of about 400 words on why the seasons change on Earth, covering axial tilt, hemispheres and the equinoxes.',
}
TOOLS = [{"type": "function", "function": {"name": "search_files", "description": "Search files in a repository by regex and return matching lines",
          "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["pattern"]}}},
         {"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]


def run(port, kind, max_tokens, pad, temperature):
    if kind == 'tool':
        msgs = [{"role": "user", "content": "Find every place where the function parse_config is called in src/ and then write a file notes.md listing them as a markdown table."}]
        extra = {"tools": TOOLS}
    else:
        msgs = [{"role": "user", "content": PROMPTS[kind]}]
        extra = {}
    if pad:
        filler = ("The quick brown fox jumps over the lazy dog. " * 40 + "\n") * (pad // 450 + 1)
        msgs.insert(0, {"role": "system", "content": "Background notes (ignore):\n" + filler[:pad * 4]})
    payload = {'messages': msgs, 'max_tokens': max_tokens, 'stream': True, **extra}
    if temperature is not None:
        payload['temperature'] = temperature
    t0 = time.monotonic(); first = None; n = 0; timings = None; targs = {}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b'data: '): continue
            raw = line[6:].strip()
            if raw == b'[DONE]': break
            d = json.loads(raw)
            if 'timings' in d: timings = d['timings']
            for c in d.get('choices', []):
                dl = c.get('delta', {})
                for tc in dl.get('tool_calls') or []:
                    targs[tc.get('index', 0)] = targs.get(tc.get('index', 0), '') + ((tc.get('function') or {}).get('arguments') or '')
                if dl.get('content') or dl.get('reasoning_content') or dl.get('tool_calls'):
                    if first is None: first = time.monotonic()
                    n += 1
    el = time.monotonic() - t0
    row = {'kind': kind, 'chunks': n, 'ttft_s': (first - t0) if first else None, 'elapsed_s': el}
    if kind == 'tool':
        ok = bool(targs)
        for a_ in targs.values():
            try: json.loads(a_)
            except Exception: ok = False
        row['tool_calls'] = len(targs); row['tool_json_ok'] = ok
    if first and n > 1: row['client_chunks_per_s'] = (n - 1) / (time.monotonic() - first)
    if timings:
        row['predicted_n'] = timings.get('predicted_n'); row['decode_tok_s'] = timings.get('predicted_per_second')
    return row


ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8000); ap.add_argument('--runs', type=int, default=3); ap.add_argument('--max-tokens', type=int, default=256)
ap.add_argument('--ctx-pad', type=int, default=0); ap.add_argument('--label', default=''); ap.add_argument('--temperature', type=float, default=None)
ap.add_argument('--kinds', default='code,tool,prose')
a = ap.parse_args()
for kind in a.kinds.split(','):
    for i in range(a.runs):
        row = run(a.port, kind, a.max_tokens, a.ctx_pad, a.temperature)
        row['run'] = i; row['label'] = a.label
        print(json.dumps(row), flush=True)
