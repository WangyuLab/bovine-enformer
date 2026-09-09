#!/usr/bin/env python3
# L3: quantitative CRE x ctx accessibility targets (log1p mean TMM per CRE)
import os, sys, json, time, datetime
import numpy as np

PROJ = "/storage/public/home/2022010213/cattle_vareffect"
BWD = "/storage/public/home/2022010213/bovine_reg_atlas/tracks/bigwig/tmm"
CRE = PROJ + "/data/atlas/cre_list.tsv"
MAP = PROJ + "/data/atlas/ctx_bw_map.json"
SHARD = int(sys.argv[1]); NSHARD = int(sys.argv[2])
OUT = PROJ + "/data/atlas/cre_quant_%d_%d.tsv" % (SHARD, NSHARD)

def log(m): print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), m), flush=True)

mapping = json.load(open(MAP))
ctxs = sorted(mapping.keys())
my = [c for i, c in enumerate(ctxs) if i % NSHARD == SHARD]
log("shard %d/%d: %d ctx" % (SHARD, NSHARD, len(my)))

# load CREs grouped by chr
recs = []
with open(CRE) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        recs.append((p[0], int(p[1]), p[2]))   # chrom, center, peak_id
by_chr = {}
for i, (c, ctr, pid) in enumerate(recs):
    by_chr.setdefault(c, []).append(i)
log("CREs: %d" % len(recs))
# cre intervals from peak_id (1-based start)
def parse_pid(pid):
    a = pid.split(":")[1]
    s, e = a.split("-")
    return int(s), int(e)

import pyBigWig
res = {c: np.zeros(len(recs), dtype=np.float32) for c in my}
t0 = time.time()
for ci, ctx in enumerate(my):
    bws = mapping[ctx]
    opened = [pyBigWig.open(os.path.join(BWD, b + ".TMM.bw")) for b in bws]
    opened = [b for b in opened if b is not None]
    if not opened:
        log("WARN no bw for %s" % ctx); continue
    chroms = opened[0].chroms()
    for chrom0, idxs in sorted(by_chr.items(), key=lambda z: -len(z[1])):
        chrom = "chr" + chrom0 if "chr" + chrom0 in chroms else chrom0
        if chrom not in chroms: continue
        L = chroms[chrom]
        arrs = []
        for bw in opened:
            if chrom in bw.chroms():
                arrs.append(np.nan_to_num(bw.values(chrom, 0, L, numpy=True)))
        if not arrs: continue
        a = np.mean(arrs, axis=0)
        for i in idxs:
            pid = recs[i][2]
            s, e = parse_pid(pid)
            s = max(s - 1, 0); e = min(e, L)
            if e <= s: continue
            res[ctx][i] = np.log1p(a[s:e].mean())
    for bw in opened: bw.close()
    rate = (ci + 1) / max(time.time() - t0, 1)
    log("ctx %d/%d %s done (%.1f ctx/h)" % (ci + 1, len(my), ctx[:40], rate * 3600))

with open(OUT, "w") as f:
    f.write("peak_id\t" + "\t".join(my) + "\n")
    M = np.stack([res[c] for c in my], 1)
    for i, r in enumerate(recs):
        f.write(r[2] + "\t" + "\t".join("%.4f" % v for v in M[i]) + "\n")
log("written %s" % OUT)
print("CRE_QUANT_DONE", flush=True)
