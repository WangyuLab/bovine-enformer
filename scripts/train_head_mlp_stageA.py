#!/usr/bin/env python3
# Production head: 2-layer MLP on cached embeddings + per-ctx metrics + calibration
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import glob

D = "/home/wangyu/cattle_h1/h3c/"
def log(m): print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), m), flush=True)

log("loading embeddings ...")
parts, metas = [], []
DIM = 3072
for f in sorted(glob.glob(D + "cre_emb_*.npy")):
    parts.append(np.asarray(np.memmap(f, dtype=np.float16, mode="r")).reshape(-1, DIM))
    metas.append(pd.read_csv(f.replace(".npy", ".meta.tsv"), sep="\t", dtype={"chrom": str}))
X = np.concatenate(parts).astype(np.float32)
meta = pd.concat(metas, ignore_index=True)
Y = np.load("/home/wangyu/cattle_h1/atlas/ctx_labels.npy").astype(np.float32)
names = pd.read_csv("/home/wangyu/cattle_h1/atlas/ctx_label_names.tsv", header=None)[0].tolist()
alive = Y.sum(axis=0) > 0
Y = Y[:, alive]
names = [n for n, a in zip(names, alive) if a]
K = Y.shape[1]
chrs = meta["chrom"].values
te = np.array([c in {"1", "2", "3"} for c in chrs])
va = np.array([c == "4" for c in chrs])
tr = ~(te | va)
log("X %s Y %s | train %d val %d test %d" % (str(X.shape), str(Y.shape), tr.sum(), va.sum(), te.sum()))

dev = "cpu"
model = nn.Sequential(nn.Linear(DIM, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, K))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
pos_rate = Y[tr].mean(axis=0)
pos_w = torch.from_numpy(np.clip((1 - pos_rate) / np.clip(pos_rate, 1e-4, 1), 1.0, 50.0)).float()
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

Xtr = torch.from_numpy(X[tr]); Ytr = torch.from_numpy(Y[tr])
Xva = torch.from_numpy(X[va]); Yva = torch.from_numpy(Y[va])
Xte = torch.from_numpy(X[te]); Yte = torch.from_numpy(Y[te])

def macro_auc(Xd, Td):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(torch.cat([model(Xd[i:i + 16384]) for i in range(0, Xd.shape[0], 16384)])).numpy()
    t = Td.numpy()
    per = {}
    for k in range(K):
        if t[:, k].sum() >= 20 and (1 - t[:, k]).sum() >= 20:
            per[k] = roc_auc_score(t[:, k], p[:, k])
    return float(np.mean(list(per.values()))), per, p

BS = 4096
best, best_state, patience = -1, None, 0
for ep in range(1, 81):
    model.train()
    perm = torch.randperm(Xtr.shape[0])
    for b in range(0, Xtr.shape[0], BS):
        idx = perm[b:b + BS]
        loss = lossf(model(Xtr[idx]), Ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    va_auc, _, _ = macro_auc(Xva, Yva)
    if ep % 5 == 0 or ep == 1:
        log("ep %d val %.4f" % (ep, va_auc))
    if va_auc > best:
        best, best_state, patience = va_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 8:
            log("early stop ep %d" % ep); break

model.load_state_dict(best_state)
te_auc, per_ctx, pte = macro_auc(Xte, Yte)
log("=== PROD HEAD test macro AUROC %.4f (%d ctx) ===" % (te_auc, len(per_ctx)))

rows = [{"ctx": names[k], "auroc": v} for k, v in sorted(per_ctx.items(), key=lambda x: -x[1])]
pdf = pd.DataFrame(rows)
pdf.to_csv(D + "prod_head_perctx.tsv", sep="\t", index=False)
log("top5: " + "; ".join("%s %.3f" % (r["ctx"][:30], r["auroc"]) for r in rows[:5]))
log("bot5: " + "; ".join("%s %.3f" % (r["ctx"][:30], r["auroc"]) for r in rows[-5:]))
torch.save({"head_mlp": best_state, "ctx_names": names, "alive_idx": np.where(alive)[0].tolist(),
            "arch": "3072-512-GELU-drop0.2-%d" % K, "val_auc": best, "test_auc": te_auc},
           D + "bovine_head_mlp.pt")
log("saved bovine_head_mlp.pt")
print("PROD_HEAD_DONE", flush=True)
