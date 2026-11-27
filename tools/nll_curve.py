#!/usr/bin/env python3
"""Per-position NLL curves from float32 arrays written by qv_ink_nll / qv_ink_cpu --nll-out (NaN or 0 = not measured).
  nll_curve.py A.f32 [B.f32 ...] --bin 4096   mean NLL per position bin for each file, and paired difference to the first file."""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--bin", type=int, default=4096)
a = ap.parse_args()
arrs = [np.fromfile(f, dtype=np.float32) for f in a.files]
n = max(len(x) for x in arrs)
arrs = [np.pad(x, (0, n - len(x)), constant_values=np.nan) for x in arrs]
arrs = [np.where(x == 0, np.nan, x) for x in arrs]
print("bin_start " + " ".join("%12s" % f.split("/")[-1][:12] for f in a.files) + "   n   " + "  diff_vs_first(se)")
for s in range(0, n, a.bin):
    sl = [x[s:s + a.bin] for x in arrs]
    ok = np.all([~np.isnan(x) for x in sl], axis=0)
    if ok.sum() == 0:
        continue
    row = "%8d " % s + " ".join("%12.4f" % np.mean(x[ok]) for x in sl) + " %5d" % ok.sum()
    for x in sl[1:]:
        d = (x - sl[0])[ok]
        row += "  %+.4f(%.4f)" % (d.mean(), d.std() / np.sqrt(len(d)))
    print(row)
