#!/usr/bin/env python3 -I
"""Requantize the expert tensors of an existing Inkling .qv to fewer bits (data-free: dequantize -> clipped RTN, meta8).

Used to build small profiles for RAM-limited devices (phone) from a released q3a5 file without the BF16 checkpoint or
calibration data. Everything that is not an expert tensor is copied bit-exactly. Error compounds (3-bit -> 2-bit), so this is a
quality-for-size trade; check with `qv_ink_cpu --nll`.

  python3 -I tools/qv_requant.py --src in.qv --out out.qv --bits 2 [--keep-layers 0,1,22,23] [--hf-dir DIR]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qv_format as F
import qv_quant as Q
import qv_blobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=2); ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--keep-layers", default="", help="comma list of layers whose experts keep the source bits")
    ap.add_argument("--hf-dir", help="dir with tokenizer.json / chat_template.jinja / generation_config.json to embed")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    keep = {int(x) for x in a.keep_layers.split(",") if x}
    r = F.Reader(a.src)
    w = F.ContainerWriter(a.out)
    t0 = time.time()
    for name, t in r.tensors.items():
        if "mlp.experts" in name and t["kind"] == "q":
            L = int(name.split("layers.")[1].split(".")[0])
            if L in keep:
                copy(r, w, name, t); continue
            E, N, K = t["shape"]
            pk_all, sm_all, rs_all = [], [], []
            for e0 in range(0, E, a.batch):
                Wd = torch.stack([r.dequant(name, e) for e in range(e0, min(E, e0 + a.batch))]).to(a.device)
                q, S, M, rs = Q.rtn_batched(Wd, a.bits, a.group, clip=True, meta8=True)
                for b in range(q.shape[0]):
                    pk_all.append(F.pack(q[b].cpu(), a.bits)); sm_all.append(Q.pack_meta(S[b], M[b], rs[b]).cpu()); rs_all.append(rs[b].cpu())
            w.add_quant(name, torch.stack(pk_all), torch.stack(sm_all), a.bits, [E, N, K], a.group, torch.stack(rs_all))
            print(f"{name} -> {a.bits}b  {time.time()-t0:.0f}s  {w.off/2**30:.2f} GiB", flush=True)
        else:
            copy(r, w, name, t)
    cfg = dict(r.config)
    cfg["quant"] = dict(cfg.get("quant", {}), requant_expert_bits=a.bits, requant_group=a.group, requant_keep_layers=sorted(keep), requant_from=os.path.basename(a.src))
    blobs = meta = None
    if a.hf_dir:
        blobs, meta = qv_blobs.collect_hf_dir(a.hf_dir)
        meta.update(model_name=os.path.splitext(os.path.basename(a.out))[0], base_model="thinkingmachines/Inkling-2-Mini",
                    quantization=f"requant of {os.path.basename(a.src)}: experts {a.bits}b/g{a.group} RTN+clip (data-free)")
    w.close(cfg, blobs=blobs, meta=meta)
    print(f"FINISHED {a.out}: {os.path.getsize(a.out)/2**30:.3f} GiB {time.time()-t0:.0f}s")


def copy(r, w, name, t):
    ent = dict(t)
    for k_off, k_nb in (("off_q", "nbytes_q"), ("off_sm", "nbytes_sm"), ("off_rs", "nbytes_rs"), ("off", "nbytes")):
        if k_off in t:
            arr = np.array(r.mm[r.data_off + t[k_off]: r.data_off + t[k_off] + t[k_nb]])
            ent[k_off] = w._blob(arr)
    w.tensors[name] = ent


if __name__ == "__main__":
    main()
