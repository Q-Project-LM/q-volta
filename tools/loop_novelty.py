#!/usr/bin/env python3
"""Soft repetition metric for loop_bench records written with --keep-text: fraction of word 6-grams in the last 1500 words that already occurred earlier in the
same generation (0 = all new, 1 = pure repetition).  Complements the strict 16-token detector; prints per (label, cell, kind) mean and max."""
import json, sys, collections

def novelty_rep(text, tail=1500, n=6):
    w = text.split()
    if len(w) < tail + n:
        tail = max(len(w) // 2, n)
    head, last = w[:-tail], w[-tail:]
    seen = set(tuple(head[i:i + n]) for i in range(len(head) - n + 1))
    g = [tuple(last[i:i + n]) for i in range(len(last) - n + 1)]
    return sum(x in seen for x in g) / max(len(g), 1)

if __name__ == "__main__":
    acc = collections.defaultdict(list)
    for p in sys.argv[1:]:
        for l in open(p):
            r = json.loads(l)
            if "think" in r and r["n_tokens"] > 1500:
                acc[(r["label"], r["cell"], r["kind"])].append(novelty_rep(r["think"] + r["text"]))
    for k, v in sorted(acc.items()):
        print("%-14s %-12s %-9s n=%3d mean=%.3f max=%.3f" % (k + (len(v), sum(v) / len(v), max(v))))
