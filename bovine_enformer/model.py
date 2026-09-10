"""Bovine-Enformer: bovine cell-context accessibility prediction from sequence."""
import os
import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 196608
HALF = SEQ_LEN // 2
NB = 17

_LUT = np.full(256, 4, dtype=np.uint8)
_LUT[ord('A')] = 0; _LUT[ord('C')] = 1; _LUT[ord('G')] = 2; _LUT[ord('T')] = 3


def one_hot(seq):
    idx = _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    oh = np.zeros((4, len(seq)), dtype=np.float32)
    for c in range(4):
        oh[c] = (idx == c)
    return oh


class BovineEnformer:
    def __init__(self, enformer_dir, head_path, device=None):
        from enformer_pytorch import Enformer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = Enformer.from_pretrained(enformer_dir).eval()
        self.trunk = model._trunk.to(self.device)
        if self.device == "cuda":
            self.trunk = self.trunk.to(torch.bfloat16)
        ck = torch.load(head_path, map_location="cpu", weights_only=False)
        self.ctx_names = ck["ctx_names"]
        K = len(self.ctx_names)
        dim = ck["head_mlp"]["0.weight"].shape[1]
        self.head = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, K))
        self.head.load_state_dict(ck["head_mlp"])
        self.head.eval().to(self.device)
        self.meta = {k: ck.get(k) for k in ("arch", "val_auc", "test_auc")}

    @torch.no_grad()
    def _embed(self, seqs):
        outs = []
        for i in range(0, len(seqs), 8):
            batch = []
            for s in seqs[i:i + 8]:
                if len(s) > SEQ_LEN:
                    c = len(s) // 2
                    s = s[c - HALF:c + HALF]
                pad = SEQ_LEN - len(s)
                s = "N" * (pad // 2) + s + "N" * (pad - pad // 2)
                batch.append(one_hot(s))
            xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1).contiguous().to(self.device)
            if self.device == "cuda":
                xb = xb.to(torch.bfloat16)
            y = self.trunk(xb)
            y = y[:, y.shape[1] // 2 - NB // 2: y.shape[1] // 2 + NB // 2 + 1, :].mean(dim=1)
            outs.append(y.float())
        return torch.cat(outs)

    @torch.no_grad()
    def predict_accessibility(self, seqs):
        emb = self._embed(seqs)
        return torch.sigmoid(self.head(emb)).cpu().numpy()

    @torch.no_grad()
    def score_variant(self, ref_seq, alt_base, center=None):
        c = center if center is not None else len(ref_seq) // 2
        alt_seq = ref_seq[:c] + alt_base + ref_seq[c + 1:]
        p = self.predict_accessibility([ref_seq, alt_seq])
        delta = p[1] - p[0]
        return {"acc_ref": p[0], "acc_alt": p[1], "delta": delta,
                "max_abs_delta": float(np.abs(delta).max()),
                "l2_delta": float(np.sqrt((delta ** 2).sum())),
                "argmax_ctx": self.ctx_names[int(np.abs(delta).argmax())]}
