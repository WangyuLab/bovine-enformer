#!/usr/bin/env python3
# L1: ISM (saturation mutagenesis) + gradient saliency variant scoring
import os, sys, time, datetime
import numpy as np
import torch
import torch.nn as nn

PROJ = "/storage/public/home/2022010213/cattle_vareffect"
GENOME = "/storage/public/home/2022010213/camel_project/genome/cattle_Bos_taurus.ARS-UCD1.2.dna.toplevel.fa"
FAI = PROJ + "/data/h1/cattle_ARS-UCD1.2.fai"
LAB = PROJ + "/data/h1/mammary_effect_labels.tsv"
START = int(sys.argv[1]); END = int(sys.argv[2])
OUT = PROJ + "/data/h3/ism_%d_%d.tsv" % (START, END)
SEQ_LEN = 196608; HALF = SEQ_LEN // 2; NB = 17

def log(m): print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), m), flush=True)

# --- data ---
recs = []
with open(LAB) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        recs.append((p[0], p[1]))
recs = recs[START:END]
log("records %d-%d: %d" % (START, END, len(recs)))

fai = {}
with open(FAI) as f:
    for line in f:
        p = line.split("\t")
        fai[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
fa = open(GENOME, "rb")
def fetch(chrom, start, end):
    if chrom not in fai: return ""
    slen, off, lb, ll = fai[chrom]
    start = max(1, start); end = min(slen, end)
    if start > end: return ""
    chunks, posi = [], start
    while posi <= end:
        row = (posi - 1) // lb; col = (posi - 1) % lb
        fa.seek(off + row * ll + col)
        n = min(lb - col, end - posi + 1)
        chunks.append(fa.read(n)); posi += n
    return b"".join(chunks).decode().upper()

LUT = np.full(256, 4, dtype=np.uint8)
LUT[ord('A')] = 0; LUT[ord('C')] = 1; LUT[ord('G')] = 2; LUT[ord('T')] = 3
def one_hot(seq):
    idx = LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    oh = np.zeros((4, len(seq)), dtype=np.float32)
    for c in range(4):
        oh[c] = (idx == c)
    return torch.from_numpy(oh)

# --- model: trunk + production MLP head ---
from enformer_pytorch import Enformer
log("loading model ...")
model = Enformer.from_pretrained(PROJ + "/models/enformer-official-rough")
dev = "cuda"
model = model.to(dev).eval()
trunk = model._trunk.to(dev).to(torch.bfloat16)
ck = torch.load(PROJ + "/models/bovine_enformer/head_mlp.pt", map_location="cpu", weights_only=False)
K = len(ck["ctx_names"])
head = nn.Sequential(nn.Linear(3072, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, K))
head.load_state_dict(ck["head_mlp"]); head.eval().to(dev)
log("model ready, K=%d" % K)

B2C = {"A": 0, "C": 1, "G": 2, "T": 3}
ALTS = "ACGT"

def head_center(x):
    y = trunk(x)
    y = y[:, y.shape[1] // 2 - NB // 2: y.shape[1] // 2 + NB // 2 + 1, :].mean(dim=1)
    return torch.sigmoid(head(y.float()))

written = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            written.add(line.split("\t")[0])
log("resume from %d" % len(written))

t0 = time.time(); done = 0
with open(OUT, "a") as out:
    if not written:
        out.write("variant_id\tref\talt\talt_dmaxabs\talt_l2\talt_argmaxctx\tism_maxabs\tism_mean_l2\tgrad_mag\tgrad_dir\tgrad_dir_ctx\n")
    for chrom, vid in recs:
        if vid in written: continue
        parts = vid.split("_")
        pos = int(parts[1]); ref, alt = parts[2], parts[3]
        if ref not in B2C or alt not in B2C: continue
        s = fetch(chrom, pos - HALF, pos + HALF - 1)
        if len(s) < SEQ_LEN:
            s = s + "N" * (SEQ_LEN - len(s))
        x_ref = one_hot(s)
        # 4-sequence batch: ref + 3 alternative bases at center
        alts3 = [b for b in ALTS if b != ref]
        xs = [x_ref]
        for b in alts3:
            xa = x_ref.clone()
            xa[:, HALF] = 0.0
            xa[B2C[b], HALF] = 1.0
            xs.append(xa)
        xb = torch.stack(xs).permute(0, 2, 1).contiguous().to(dev).to(torch.bfloat16)
        with torch.no_grad():
            P = head_center(xb)              # (4, 59)
        deltas = (P[1:] - P[0]).float().cpu().numpy()   # (3, 59)
        # alt-specific
        ai = alts3.index(alt)
        d_alt = deltas[ai]
        alt_l2 = float(np.sqrt((d_alt ** 2).sum()))
        alt_arg = int(np.abs(d_alt).argmax())
        alt_dmax = float(d_alt[alt_arg])
        # ISM summaries across all 3 alts
        ism_maxabs = float(np.abs(deltas).max())
        ism_mean_l2 = float(np.mean(np.sqrt((deltas ** 2).sum(axis=1))))
        # gradient saliency (forward+backward on ref)
        xg = x_ref.permute(1, 0).unsqueeze(0).contiguous().to(dev)
        xg.requires_grad_(True)
        Pg = head_center(xg.to(torch.bfloat16))
        loss = Pg.sum()
        g = torch.autograd.grad(loss, xg, retain_graph=True)[0][0, HALF, :].float().cpu().numpy()  # (4,)
        grad_mag = float(np.abs(g).sum())
        dir_vec = np.zeros(4, dtype=np.float32); dir_vec[B2C[alt]] = 1.0; dir_vec[B2C[ref]] = -1.0
        grad_dir = float((g * dir_vec).sum())
        # per-ctx directional gradient at the max-|delta| ctx
        head.zero_grad()
        loss2 = Pg[0, alt_arg]
        g2 = torch.autograd.grad(loss2, xg, retain_graph=False)[0][0, HALF, :].float().cpu().numpy()
        grad_dir_ctx = float((g2 * dir_vec).sum())
        out.write("%s\t%s\t%s\t%.5f\t%.5f\t%d\t%.5f\t%.5f\t%.6f\t%.6f\t%.6f\n" % (
            vid, ref, alt, alt_dmax, alt_l2, alt_arg, ism_maxabs, ism_mean_l2,
            grad_mag, grad_dir, grad_dir_ctx))
        done += 1
        if done % 500 == 0:
            out.flush()
            rate = done / max(time.time() - t0, 1)
            log("%d done, %.2f var/s, ETA %.1f h" % (done, rate, (len(recs) - done) / max(rate, 0.01) / 3600))
log("ISM_DONE -> %s" % OUT)
print("ISM_DONE", flush=True)
