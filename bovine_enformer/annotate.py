"""Variant functional annotation: atlas open-peak overlap + TF motif disruption."""
import re
import numpy as np

_B2C = {"A": 0, "C": 1, "G": 2, "T": 3}
_BG = np.array([0.29, 0.21, 0.21, 0.29])  # cattle ~42% GC


class PeakAnnotator:
    """Overlap queries against the BovineSCellRegAtlas unified CRE catalog."""

    def __init__(self, cre_annot_path):
        import pandas as pd
        df = pd.read_csv(cre_annot_path, sep="\t")
        self.by_chr = {}
        for chrom, sub in df.groupby("chr"):
            self.by_chr[chrom] = sub.sort_values("start").reset_index(drop=True)

    def query(self, chrom, pos):
        """pos: 1-based. Returns dict with peak annotation or {'in_peak': 0}."""
        key = "chr" + str(chrom)
        if key not in self.by_chr:
            key = str(chrom)
        if key not in self.by_chr:
            return {"in_peak": 0}
        sub = self.by_chr[key]
        starts = sub["start"].values
        ends = sub["end"].values
        hits = np.nonzero((starts < pos) & (ends >= pos))[0]
        if not len(hits):
            # distance to nearest CRE
            i = np.searchsorted(starts, pos)
            d = None
            if i > 0:
                d = pos - ends[i - 1]
            if i < len(starts):
                d2 = starts[i] - pos
                d = d2 if d is None else min(d, d2)
            return {"in_peak": 0, "cre_dist": int(d) if d is not None else -1}
        j = hits[np.argmax(sub["score"].values[hits])]
        row = sub.iloc[j]
        return {"in_peak": 1, "cre_id": row["peak_id"], "cre_score": float(row["score"]),
                "cre_n_ctx": int(row["n_ctx"]) if row["n_ctx"] == row["n_ctx"] else 0,
                "cre_n_tissue": int(row["n_tissue"]) if row["n_tissue"] == row["n_tissue"] else 0,
                "cre_level": str(row["Level_fig4d_CPM1"]),
                "cre_marker_n": int(row["mk_n"]) if row["mk_n"] == row["mk_n"] else 0,
                "cre_cons5": int(row["cons_all5"]),
                "p2g_genes": row["p2g_genes"]}


class MotifAnnotator:
    """JASPAR PWM scanning with ref/alt disruption scoring (log-odds, both strands)."""

    def __init__(self, pfm_path, rel_threshold=0.80):
        self.motifs = []
        self.rel_t = rel_threshold
        cur_id = cur_name = None
        mat = {}
        for line in open(pfm_path):
            line = line.strip()
            if line.startswith(">"):
                if cur_id and mat.get("A"):
                    self.motifs.append((cur_id, cur_name, self._lo(mat)))
                parts = line[1:].split("\t")
                cur_id, cur_name = parts[0], parts[1] if len(parts) > 1 else parts[0]
                mat = {}
            elif line and line[0] in "ACGT":
                mat[line[0]] = [float(x) for x in re.findall(r"[-\d.]+", line)]
        if cur_id and mat.get("A"):
            self.motifs.append((cur_id, cur_name, self._lo(mat)))

    def _lo(self, mat):
        P = np.stack([mat[b] for b in "ACGT"]).astype(np.float32)
        P = (P + 0.8) / (P.sum(axis=0, keepdims=True) + 3.2)
        return np.log2(P / _BG[:, None])

    def _scan(self, seq_int, pwm):
        w = pwm.shape[1]
        W = len(seq_int)
        out = np.zeros(W - w + 1, dtype=np.float32)
        for j in range(w):
            out += pwm[seq_int[j:j + W - w + 1], j]
        return out

    def annotate(self, ref_seq, alt_base, center=None):
        """Returns motif hits overlapping the variant + best disruption/creation."""
        c = center if center is not None else len(ref_seq) // 2
        alt_seq = ref_seq[:c] + alt_base + ref_seq[c + 1:]
        sr = np.array([_B2C.get(b, 0) for b in ref_seq], dtype=np.uint8)
        sa = np.array([_B2C.get(b, 0) for b in alt_seq], dtype=np.uint8)
        n_hits = 0
        best = {"disrupt": (0.0, None), "create": (0.0, None)}
        for mid, name, pwm in self.motifs:
            w = pwm.shape[1]
            maxs = pwm.max(axis=0).sum(); mins = pwm.min(axis=0).sum()
            thr = mins + self.rel_t * (maxs - mins)
            for P in (pwm, pwm[::-1, ::-1]):
                Sr = self._scan(sr, P); Sa = self._scan(sa, P)
                lo, hi = max(c - w + 1, 0), min(c, Sr.shape[0] - 1)
                if lo > hi:
                    continue
                seg_r = Sr[lo:hi + 1]
                j = int(seg_r.argmax())
                br = seg_r[j]
                if br < thr:
                    continue
                n_hits += 1
                d = float(Sa[lo + j] - br)
                if d < best["disrupt"][0]:
                    best["disrupt"] = (d, name + "(" + mid + ")")
                if d > best["create"][0] and Sa[lo + j] >= thr:
                    best["create"] = (d, name + "(" + mid + ")")
        return {"motif_n_hits": n_hits,
                "motif_max_disrupt": best["disrupt"][0], "top_disrupt_tf": best["disrupt"][1] or "",
                "motif_max_create": best["create"][0], "top_create_tf": best["create"][1] or "",
                "motif_abs_delta_max": max(abs(best["disrupt"][0]), abs(best["create"][0]))}


def approx_effect(be_l2, peak_ann, motif_ann):
    """Heuristic v1 evidence score (0-3) + tier. To be calibrated on credible-set data."""
    s_be = 1.0 / (1.0 + np.exp(-8.0 * float(be_l2)))      # 0-1 perturbation evidence
    s_peak = 1.0 if peak_ann.get("in_peak", 0) else 0.0
    s_motif = min(1.0, motif_ann.get("motif_abs_delta_max", 0.0) / 2.0)
    total = s_be + s_peak + s_motif
    tier = "低" if total < 1.0 else ("中" if total < 2.0 else "高")
    return round(float(total), 3), tier
