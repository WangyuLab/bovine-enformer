python3 <<'PYEOF'

import glob, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

D = "/home/wangyu/cattle_h1/h3c/"
A = "/home/wangyu/cattle_h1/atlas/"
def log(m): print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), m), flush=True)

DIM = 3072
parts, metas = [], []
for f in sorted(glob.glob(D + "cre_emb_*.npy")):
    parts.append(np.asarray(np.memmap(f, dtype=np.float16, mode="r")).reshape(-1, DIM))
    metas.append(pd.read_csv(f.replace(".npy", ".meta.tsv"), sep="\t", dtype={"chrom": str}))
Xc = np.concatenate(parts)
meta_c = pd.concat(metas, ignore_index=True)
log("CRE emb: %s" % str(Xc.shape))

bparts, bmetas = [], []
for f in sorted(glob.glob(A + "bg_emb_?_4.npy")):
    bparts.append(np.asarray(np.memmap(f, dtype=np.float16, mode="r")).reshape(-1, DIM))
    bmetas.append(pd.read_csv(f.replace(".npy", ".meta.tsv"), sep="\t", dtype={"chrom": str}))
Xb = np.concatenate(bparts)
meta_b = pd.concat(bmetas, ignore_index=True)
log("BG emb: %s" % str(Xb.shape))

X = np.concatenate([Xc, Xb])
chrs = np.concatenate([meta_c["chrom"].values, meta_b["chrom"].values])
is_bg = np.r_[np.zeros(len(Xc), bool), np.ones(len(Xb), bool)]

Y = np.load(A + "ctx_labels.npy").astype(np.float32)
names = pd.read_csv(A + "ctx_label_names.tsv", header=None)[0].tolist()
alive = Y.sum(axis=0) > 0
Y = Y[:, alive]
names = [n for n, a in zip(names, alive) if a]
K = Y.shape[1]
Yfull = np.zeros((len(X), K), dtype=np.float32)
Yfull[:len(Xc)] = Y
log("X %s Y %s K %d (cre %d, bg %d)" % (str(X.shape), str(Yfull.shape), K, (~is_bg).sum(), is_bg.sum()))

te = np.array([c in {"1", "2", "3"} for c in chrs])
va = np.array([c == "4" for c in chrs])
tr = ~(te | va)
cre_mask = ~is_bg

def T(idx):
    return torch.from_numpy(X[idx].astype(np.float32))
Yfull_t = torch.from_numpy(Yfull)

tr_i = np.nonzero(tr)[0]
va_i = np.nonzero(va)[0]
te_i = np.nonzero(te)[0]
Xva = T(va_i); Yva = Yfull_t[va_i]; va_cre = cre_mask[va]
Xte = T(te_i); Yte = Yfull_t[te_i]; te_cre = cre_mask[te]; te_bg = is_bg[te]

model = nn.Sequential(nn.Linear(DIM, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, K))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
pos_rate = Yfull[tr & cre_mask].mean(axis=0)
pos_w = torch.from_numpy(np.clip((1 - pos_rate) / np.clip(pos_rate, 1e-4, 1), 1.0, 50.0)).float()
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

def eval_val():
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xva[va_cre])).numpy()
        pb = torch.sigmoid(model(Xva[~va_cre])).numpy()
    t = Yva[va_cre].numpy()
    aucs = [roc_auc_score(t[:, k], pv[:, k]) for k in range(K)
            if t[:, k].sum() >= 10 and (1 - t[:, k]).sum() >= 10]
    return float(np.mean(aucs)), float(pb.mean())

BS = 4096
best, best_state, patience = -1, None, 0
for ep in range(1, 61):
    model.train()
    perm = np.random.RandomState(ep).permutation(len(tr_i))
    for b in range(0, len(tr_i), BS):
        idxp = perm[b:b + BS]
        idx = tr_i[idxp]
        loss = lossf(model(T(idx)), Yfull_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    va_auc, bgm = eval_val()
    if ep % 3 == 0 or ep == 1:
        log("ep %d val cre-AUROC %.4f | bg P(open) %.3f" % (ep, va_auc, bgm))
    if va_auc > best:
        best, best_state, patience = va_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 8:
            log("early stop %d" % ep); break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pc = torch.sigmoid(torch.cat([model(Xte[te_cre][i:i + 16384]) for i in range(0, te_cre.sum(), 16384)])).numpy()
    pb = torch.sigmoid(torch.cat([model(Xte[te_bg][i:i + 16384]) for i in range(0, te_bg.sum(), 16384)])).numpy()
t = Yte[te_cre].numpy()
aucs = [roc_auc_score(t[:, k], pc[:, k]) for k in range(K) if t[:, k].sum() >= 10 and (1 - t[:, k]).sum() >= 10]
test_auc = float(np.mean(aucs))
sep = roc_auc_score(np.r_[np.ones(len(pc)), np.zeros(len(pb))], np.r_[pc.max(axis=1), pb.max(axis=1)])
log("=== V1.0 HEAD: test cre macro AUROC %.4f ===" % test_auc)
log("=== background P(open): mean %.4f p95 %.4f ===" % (pb.mean(), np.quantile(pb, 0.95)))
log("=== cre-vs-bg separation: %.4f ===" % sep)
torch.save({"head_mlp": best_state, "ctx_names": names, "arch": "3072-512-GELU-drop0.2-%d" % K,
            "val_auc": best, "test_auc": test_auc, "bg_p_open_mean": float(pb.mean()),
            "trained": "478k CRE center + 100k bg, v1.0"},
           D + "bovine_head_v10.pt")
log("saved bovine_head_v10.pt")
print("V10_DONE", flush=True)

PYEOF
