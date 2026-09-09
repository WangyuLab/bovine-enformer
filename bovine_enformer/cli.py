#!/usr/bin/env python3
"""bovine-enformer CLI."""
import argparse
import numpy as np
from bovine_enformer.model import BovineEnformer, SEQ_LEN, HALF
from bovine_enformer.annotate import PeakAnnotator, MotifAnnotator, approx_effect


def read_fasta(path):
    seqs, names, cur = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur: seqs.append("".join(cur).upper()); cur = []
                names.append(line[1:].split()[0])
            else:
                cur.append(line)
    if cur: seqs.append("".join(cur).upper())
    return names, seqs


def make_fetch(fai_path, genome_path):
    fai = {}
    with open(fai_path) as f:
        for line in f:
            p = line.split("\t")
            fai[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    fa = open(genome_path, "rb")
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
    return fetch


def main():
    ap = argparse.ArgumentParser(prog="bovine-enformer")
    ap.add_argument("cmd", choices=["seq", "region", "var"])
    ap.add_argument("--fa"); ap.add_argument("--bed"); ap.add_argument("--variants")
    ap.add_argument("--genome"); ap.add_argument("--fai")
    ap.add_argument("--enformer", default="/storage/public/home/2022010213/cattle_vareffect/models/enformer-official-rough")
    ap.add_argument("--head", default="/storage/public/home/2022010213/cattle_vareffect/models/bovine_enformer/head_v10.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flank", type=int, default=250)
    ap.add_argument("--cre", default="/storage/public/home/2022010213/cattle_vareffect/models/bovine_enformer/data/cre_annot.tsv")
    ap.add_argument("--pfm", default="/storage/public/home/2022010213/cattle_vareffect/models/bovine_enformer/data/jaspar2024_mammal876.pfms.txt")
    a = ap.parse_args()

    be = BovineEnformer(a.enformer, a.head)
    print("[bovine-enformer] %d cell contexts; head val_auc=%.4f test_auc=%.4f" % (
        len(be.ctx_names), be.meta.get("val_auc") or -1, be.meta.get("test_auc") or -1))

    if a.cmd == "seq":
        names, seqs = read_fasta(a.fa)
        P = be.predict_accessibility(seqs)
        with open(a.out, "w") as f:
            f.write("name\t" + "\t".join(be.ctx_names) + "\n")
            for n, row in zip(names, P):
                f.write(n + "\t" + "\t".join("%.4f" % v for v in row) + "\n")
    elif a.cmd == "region":
        fetch = make_fetch(a.fai, a.genome)
        seqs, names = [], []
        with open(a.bed) as f:
            for line in f:
                if line.startswith("#"): continue
                p = line.rstrip("\n").split("\t")
                c, s, e = p[0], int(p[1]), int(p[2])
                mid = (s + e) // 2
                seqs.append(fetch(c, mid - HALF, mid + HALF - 1))
                names.append("%s:%d-%d" % (c, s, e))
        P = be.predict_accessibility(seqs)
        with open(a.out, "w") as f:
            f.write("region\t" + "\t".join(be.ctx_names) + "\n")
            for n, row in zip(names, P):
                f.write(n + "\t" + "\t".join("%.4f" % v for v in row) + "\n")
    else:
        fetch = make_fetch(a.fai, a.genome)
        peak_ann = PeakAnnotator(a.cre)
        motif_ann = MotifAnnotator(a.pfm)
        cols = ["chr", "pos", "ref", "alt",
                "in_peak", "cre_id", "cre_n_ctx", "cre_level", "cre_marker_n", "cre_cons5", "p2g_genes",
                "motif_n_hits", "top_disrupt_tf", "motif_max_disrupt", "top_create_tf", "motif_max_create",
                "be_l2_delta", "be_argmax_ctx", "approx_score", "approx_tier"]
        with open(a.out, "w") as f:
            f.write("\t".join(cols) + "\n")
            for line in open(a.variants):
                if line.startswith("#"): continue
                p = line.split()
                chrom, posi, ref, alt = p[0], int(p[1]), p[2], p[3]
                if len(ref) != 1 or len(alt) != 1: continue
                s = fetch(chrom, posi - a.flank, posi + a.flank)
                if len(s) < 2 * a.flank + 1:
                    s = s + "N" * (2 * a.flank + 1 - len(s))
                r = be.score_variant(s, alt, center=a.flank)
                pk = peak_ann.query(chrom, posi)
                mo = motif_ann.annotate(s, alt, center=a.flank)
                score, tier = approx_effect(r["l2_delta"], pk, mo)
                f.write("%s\t%d\t%s\t%s\t%d\t%s\t%d\t%s\t%d\t%d\t%s\t%d\t%s\t%.4f\t%s\t%.4f\t%.5f\t%s\t%.3f\t%s\n" % (
                    chrom, posi, ref, alt,
                    pk.get("in_peak", 0), pk.get("cre_id", ""), pk.get("cre_n_ctx", 0),
                    pk.get("cre_level", ""), pk.get("cre_marker_n", 0), pk.get("cre_cons5", 0),
                    pk.get("p2g_genes", ""),
                    mo["motif_n_hits"], mo["top_disrupt_tf"], mo["motif_max_disrupt"],
                    mo["top_create_tf"], mo["motif_max_create"],
                    r["l2_delta"], r["argmax_ctx"], score, tier))
    print("[bovine-enformer] written -> " + a.out)


if __name__ == "__main__":
    main()
