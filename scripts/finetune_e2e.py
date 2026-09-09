#!/usr/bin/env python3
# H3c stage B: end-to-end fine-tune (unfreeze last 3 transformer layers + head)
import os, sys, time, datetime
import numpy as np
import torch
import torch.nn as nn

PROJ = "/storage/public/home/2022010213/cattle_vareffect"
GENOME = "/storage/public/home/2022010213/camel_project/genome/cattle_Bos_taurus.ARS-UCD1.2.dna.toplevel.fa"
FAI = PROJ + "/data/h1/cattle_ARS-UCD1.2.fai"
CRE = PROJ + "/data/atlas/cre_list.tsv"
LAB = PROJ + "/data/atlas/ctx_labels.npy"
WEIGHTS = PROJ + "/models/enformer-official-rough"
OUTDIR = PROJ + "/models/bovine_enformer"
os.makedirs(OUTDIR, exist_ok=True)
SEQ_LEN = 196608
HALF = SEQ_LEN // 2
N_SUB = 60000          # subsample for stage B budget
EPOCHS = 2
BS = 2
ACCUM = 8

def log(m): print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), m), flush=True)

# --- data ---
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
    chunks = []; posi = start
    while posi <= end:
        row = (posi - 1) // lb; col = (posi - 1) % lb
        fa.seek(off + row * ll + col)
        n = min(lb - col, end - posi + 1)
        chunks.append(fa.read(n)); posi += n
    return b"".join(chunks).decode().upper()

recs = []
with open(CRE) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        recs.append((p[0], int(p[1])))
Y = np.load(LAB).astype(np.float32)
alive = Y.sum(axis=0) > 0
Y = Y[:, alive]
K = Y.shape[1]
log("records %d, alive ctx %d" % (len(recs), K))

rng = np.random.RandomState(42)
chrs_all = np.array([r[0] for r in recs])
va_mask = np.array([c == "4" for c in chrs_all])
tr_mask = ~np.array([c in {"1", "2", "3", "4"} for c in chrs_all])
tr_idx = np.where(tr_mask)[0]
rng.shuffle(tr_idx)
tr_idx = tr_idx[:N_SUB]
va_idx = np.where(va_mask)[0][:6000]
log("stageB train %d val %d" % (len(tr_idx), len(va_idx)))

LUT = np.full(256, 4, dtype=np.uint8)
LUT[ord('A')] = 0; LUT[ord('C')] = 1; LUT[ord('G')] = 2; LUT[ord('T')] = 3
def one_hot(seq):
    idx = LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    oh = np.zeros((4, len(seq)), dtype=np.float32)
    for c in range(4):
        oh[c] = (idx == c)
    return torch.from_numpy(oh)

# --- model ---
from enformer_pytorch import Enformer
log("loading enformer ...")
model = Enformer.from_pretrained(WEIGHTS)
dev = "cuda"
model = model.to(dev)
# freeze all, then unfreeze last 3 transformer layers + final_pointwise
for p in model.parameters():
    p.requires_grad = False
for li in [8, 9, 10]:
    for p in model.transformer[li].parameters():
        p.requires_grad = True
for p in model.final_pointwise.parameters():
    p.requires_grad = True

class BovineHead(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.h = nn.Linear(dim, k)
    def forward(self, x):
        return self.h(x)

DIM = model.final_pointwise[-1].out_features if hasattr(model.final_pointwise[-1], "out_features") else 3072
head = BovineHead(DIM, K).to(dev)
# init from stage A if available
try:
    ck = torch.load(PROJ + "/models/bovine_enformer/stageA_head.pt", map_location="cpu", weights_only=False)
    head.h.load_state_dict(ck["head"] if "head" in ck else ck)
    log("stage A head loaded as init")
except Exception as e:
    log("stage A head not found, random init: %s" % e)

try:
    model.gradient_checkpointing_enable()
    log("gradient checkpointing on")
except Exception as e:
    log("no checkpointing: %s" % e)

params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr": 1e-4},
          {"params": head.parameters(), "lr": 5e-4}]
opt = torch.optim.AdamW(params, weight_decay=1e-4)
pos_rate = Y[tr_idx].mean(axis=0)
pos_w = torch.from_numpy(np.clip((1 - pos_rate) / np.clip(pos_rate, 1e-4, 1), 1.0, 50.0)).float().to(dev)
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

NB = 17
trunk = model._trunk

def run_batch(idxs, train=True):
    tot_loss, n = 0.0, 0
    for b in range(0, len(idxs), BS):
        sel = idxs[b:b + BS]
        xs = []
        for i in sel:
            chrom, center = recs[i]
            s = fetch(chrom, center - HALF, center + HALF - 1)
            if len(s) < SEQ_LEN:
                s = s + "N" * (SEQ_LEN - len(s))
            xs.append(one_hot(s))
        xb = torch.stack(xs).permute(0, 2, 1).contiguous().to(dev)
        yb = torch.from_numpy(Y[sel]).to(dev)
        if train:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = trunk(xb)[:, trunk(xb).shape[1] // 2 - NB // 2: trunk(xb).shape[1] // 2 + NB // 2 + 1, :].mean(dim=1)
                loss = lossf(head(out), yb) / ACCUM
            loss.backward()
            tot_loss += loss.item() * ACCUM * len(sel)
        else:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = trunk(xb)[:, trunk(xb).shape[1] // 2 - NB // 2: trunk(xb).shape[1] // 2 + NB // 2 + 1, :].mean(dim=1)
                loss = lossf(head(out), yb)
            tot_loss += loss.item() * len(sel)
        n += len(sel)
        if train and (b // BS + 1) % ACCUM == 0:
            opt.step(); opt.zero_grad()
    return tot_loss / max(n, 1)

def val_auc():
    from sklearn.metrics import roc_auc_score
    model.eval(); head.eval()
    ps, ts = [], []
    with torch.no_grad():
        for b in range(0, len(va_idx), BS):
            sel = va_idx[b:b + BS]
            xs = []
            for i in sel:
                chrom, center = recs[i]
                s = fetch(chrom, center - HALF, center + HALF - 1)
                if len(s) < SEQ_LEN:
                    s = s + "N" * (SEQ_LEN - len(s))
                xs.append(one_hot(s))
            xb = torch.stack(xs).permute(0, 2, 1).contiguous().to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = trunk(xb)
                out = out[:, out.shape[1] // 2 - NB // 2: out.shape[1] // 2 + NB // 2 + 1, :].mean(dim=1)
                p = torch.sigmoid(head(out))
            ps.append(p.float().cpu().numpy()); ts.append(Y[sel])
    P = np.concatenate(ps); T = np.concatenate(ts)
    aucs = [roc_auc_score(T[:, k], P[:, k]) for k in range(K) if T[:, k].sum() >= 20 and (1 - T[:, k]).sum() >= 20]
    return float(np.mean(aucs)), len(aucs)

best = -1
for ep in range(1, EPOCHS + 1):
    model.train(); head.train()
    t0 = time.time()
    perm = tr_idx.copy()
    rng.shuffle(perm)
    # process in chunks of ACCUM*BS with logging
    chunk = ACCUM * BS
    tl, nb = 0.0, 0
    for c0 in range(0, len(perm), chunk):
        sub = perm[c0:c0 + chunk]
        l = run_batch(sub, train=True)
        tl += l; nb += 1
        if nb % 50 == 0:
            rate = nb * chunk / max(time.time() - t0, 1)
            eta = (len(perm) - nb * chunk) / max(rate, 1e-6) / 3600
            log("ep%d %d/%d loss=%.4f %.1f seq/s ETA %.1fh" % (ep, nb * chunk, len(perm), tl / nb, rate, eta))
    va, nv = val_auc()
    log("epoch %d done: train_loss=%.4f val_macroAUROC=%.4f (ctx %d)" % (ep, tl / max(nb, 1), va, nv))
    if va > best:
        best = va
        torch.save({"trunk_unfrozen": {n: p.cpu() for n, p in model.transformer.state_dict().items()},
                    "final_pointwise": {n: p.cpu() for n, p in model.final_pointwise.state_dict().items()},
                    "head": head.h.state_dict(), "alive_ctx": alive.tolist(), "val_auc": va},
                   OUTDIR + "/stageB_best.pt")
        log("saved best (val %.4f)" % va)
log("STAGEB_DONE best val %.4f" % best)
print("STAGEB_DONE", flush=True)
