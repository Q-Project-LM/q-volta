#!/usr/bin/env python3
"""Decode speed of a tool-call turn vs context (p28/loops): same agent transcript with and without `tools`.
  tool_speed.py --url http://127.0.0.1:8100 --ctx 0,32,64 [--tools both|on|off] [--max-tokens 160] [--reps 2]
ctx is in K tokens of tool-result history (0 = no history).  Prints prompt tokens, generated tokens, finish reason and the decode rate measured from
the first streamed delta to the last (client side), plus the server's own numbers are in its log ("task done ... tok/s").
"""
import argparse, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_bench as lb
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8100")
ap.add_argument("--tokenizer", default="/workspace/q-volta-models/Inkling-2-Mini/tokenizer.json")
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--ctx", default="0,32")
ap.add_argument("--tools", default="both")
ap.add_argument("--max-tokens", type=int, default=160)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--think", default="off", help="off: chat_template_kwargs enable_thinking=false; on: template default")
ap.add_argument("--task", default="Use the grep tool to find where `speculative_ok` is defined, then report. Call the tool now.")
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer)
for ck in [int(x) for x in a.ctx.split(",")]:
    msgs = lb.agent_messages(a.repo, tok, ck * 1024, a.task)[:-1] if ck else [{"role": "system", "content": lb.AGENT_SYSTEM}]
    msgs.append({"role": "user", "content": a.task})
    for use_tools in ([True, False] if a.tools == "both" else [a.tools == "on"]):
        for rep in range(a.reps):
            body = {"model": "qv", "messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens, "stream": True, "seed": a.seed + rep,
                    "stream_options": {"include_usage": True}}
            if a.think == "off":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            if use_tools:
                body["tools"] = lb.AGENT_TOOLS
            t0 = time.time(); first = None; n = 0; fin = None; usage = {}
            req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                for raw in r:
                    l = raw.decode().strip()
                    if not l.startswith("data:") or l.endswith("[DONE]"): continue
                    ev = json.loads(l[5:])
                    if ev.get("usage"): usage = ev["usage"]
                    for c in ev.get("choices", []):
                        d = c.get("delta", {})
                        if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                            if first is None: first = time.time()
                            n += 1
                        if c.get("finish_reason"): fin = c["finish_reason"]
            t1 = time.time()
            ct = usage.get("completion_tokens", n)
            print("ctx=%2dK tools=%-3s rep=%d prompt=%s completion=%s finish=%s ttft=%.1fs decode=%.1fs rate=%.1f tok/s" % (
                ck, "on" if use_tools else "off", rep, usage.get("prompt_tokens"), ct, fin, (first or t1) - t0, t1 - (first or t1),
                (ct - 1) / max(t1 - (first or t1), 1e-9)), flush=True)
