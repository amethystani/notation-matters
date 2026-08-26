"""
Script 11 — Referee 1, Point 1: connecting the behavioural and mechanistic phases.

The referee correctly observed that Phase 1 (four notations, generative LLM) and
Phase 2 (augmented SMILES pairs, ChemBERTa-2 encoder) do not directly speak to one
another: the mechanistic experiment cannot explain the four-notation behaviour
because it is run on a different model and only on SMILES.

This script closes that gap inside a single model. For each molecule we obtain,
from the SAME generative LLM used for the behavioural benchmark:
  (a) its predicted logS in each of the four notations  (from script 10), and
  (b) its hidden representation of each of the four notation strings.
We then test whether cross-notation REPRESENTATIONAL divergence predicts
cross-notation BEHAVIOURAL divergence, which is the link Phase 2 established for
ChemBERTa-2 on SMILES alone (Spearman rho = -0.371).

Usage:  python3 11_crossnotation_embedding_bridge.py [model_tag]
Run on the host with the Ollama endpoint, AFTER script 10 has completed.
"""

import requests, json, time, re, csv, io, os, sys, urllib.parse, random
import numpy as np
import selfies as sf
from itertools import combinations
from scipy.stats import spearmanr, pearsonr

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "nous-hermes2:34b"
EMBED   = "http://localhost:11434/api/embed"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
BENCH   = f"secondmodel_{MODEL.replace(':','_').replace('.','')}_results.json"
OUT     = f"bridge_{MODEL.replace(':','_').replace('.','')}_results.json"
CKPT    = OUT.replace("_results", "_checkpoint")
NOTS    = ["SMILES", "IUPAC", "InChI", "SELFIES"]


def pubchem(smiles, retries=3):
    for _ in range(retries):
        try:
            u = (f"{PUBCHEM}/smiles/{urllib.parse.quote(smiles, safe='')}"
                 f"/property/IUPACName,InChI/JSON")
            r = requests.get(u, timeout=25)
            if r.status_code == 200:
                p = r.json()["PropertyTable"]["Properties"][0]
                return p.get("IUPACName"), p.get("InChI")
            if r.status_code == 404:
                return None, None
        except Exception:
            pass
        time.sleep(1.5)
    return None, None


def embed(text, retries=2):
    for _ in range(retries):
        try:
            r = requests.post(EMBED, json={"model": MODEL, "input": text}, timeout=180)
            e = r.json().get("embeddings")
            if e:
                return np.asarray(e[0], dtype=np.float64)
        except Exception:
            time.sleep(2)
    return None


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── Behavioural results from script 10 ────────────────────────────────────────
if not os.path.exists(BENCH):
    sys.exit(f"ERROR: {BENCH} not found. Run 10_second_model_benchmark.py first.")
bench = json.load(open(BENCH))
rows = bench["results"]
print(f"Loaded {len(rows)} molecules from {BENCH}")
print(f"Model: {MODEL}\n")

done, out = set(), []
if os.path.exists(CKPT):
    out = json.load(open(CKPT))["results"]
    done = {r["smiles"] for r in out}
    print(f"Resuming: {len(out)} already embedded\n")

todo = [r for r in rows if r["smiles"] not in done]
t0 = time.time()

for i, r in enumerate(todo):
    smi = r["smiles"]
    iupac, inchi = pubchem(smi)
    try:
        selfies = sf.encoder(smi)
    except Exception:
        selfies = None
    reps = {"SMILES": smi, "IUPAC": iupac, "InChI": inchi, "SELFIES": selfies}

    embs = {}
    for n in NOTS:
        if reps[n]:
            v = embed(reps[n])
            if v is not None:
                embs[n] = v

    if len(embs) < 2:
        continue

    sims = {f"{a}|{b}": cos(embs[a], embs[b]) for a, b in combinations(NOTS, 2)
            if a in embs and b in embs}

    out.append({
        "smiles": smi,
        "class": r.get("class"),
        "true_logS": r.get("true_logS"),
        "spread_4rep": r.get("spread_4rep"),
        "n_valid_pred": r.get("n_valid"),
        "n_embedded": len(embs),
        "pairwise_cos": {k: round(v, 5) for k, v in sims.items()},
        "mean_cos": round(float(np.mean(list(sims.values()))), 5),
        "min_cos": round(float(min(sims.values())), 5),
    })

    if (i + 1) % 10 == 0:
        json.dump({"results": out}, open(CKPT, "w"))
        el = time.time() - t0
        eta = el / (i + 1) * (len(todo) - i - 1) / 60
        print(f"  [{i+1:4d}/{len(todo)}] mean_cos={out[-1]['mean_cos']:.4f}  ETA {eta:.1f} min",
              flush=True)

# ── The bridge test ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("REPRESENTATIONAL vs BEHAVIOURAL CROSS-NOTATION DIVERGENCE")
print("=" * 70)

paired = [r for r in out if r.get("spread_4rep") is not None and r["n_embedded"] >= 2]
mc = np.array([r["mean_cos"] for r in paired])
mn = np.array([r["min_cos"] for r in paired])
sp = np.array([r["spread_4rep"] for r in paired])
print(f"\nMolecules with both a representation and a behavioural measure: {len(paired)}")

rho, p = spearmanr(mc, sp)
rho2, p2 = spearmanr(mn, sp)
pr, pp = pearsonr(mc, sp)
print(f"\n  mean cross-notation cosine vs four-notation spread:")
print(f"    Spearman rho = {rho:+.4f}   p = {p:.3e}")
print(f"    Pearson  r   = {pr:+.4f}   p = {pp:.3e}")
print(f"  minimum pairwise cosine vs spread:")
print(f"    Spearman rho = {rho2:+.4f}   p = {p2:.3e}")
print(f"\n  (ChemBERTa-2, augmented SMILES pairs, Phase 2: rho = -0.371, p = 1.4e-35)")

print("\n  Dose-response across cosine quintiles:")
q = np.quantile(mc, [0, .2, .4, .6, .8, 1.0])
for i in range(5):
    m = (mc >= q[i]) & (mc <= q[i+1] if i == 4 else mc < q[i+1])
    if m.sum():
        print(f"    cos [{q[i]:.4f},{q[i+1]:.4f}]  n={m.sum():>4}  mean spread={sp[m].mean():.3f}")

print("\n  Mean pairwise similarity by notation pair:")
allp = {}
for r in paired:
    for k, v in r["pairwise_cos"].items():
        allp.setdefault(k, []).append(v)
for k in sorted(allp, key=lambda k: np.mean(allp[k])):
    print(f"    {k:<18} n={len(allp[k]):>4}  mean cos = {np.mean(allp[k]):.4f}")

print("\n  By structural class:")
byc = {}
for r in paired:
    byc.setdefault(r["class"], []).append((r["mean_cos"], r["spread_4rep"]))
print(f"    {'class':<18}{'n':>5}{'mean_cos':>11}{'mean_spread':>13}")
for c in sorted(byc, key=lambda c: np.mean([x[0] for x in byc[c]])):
    v = byc[c]
    print(f"    {c:<18}{len(v):>5}{np.mean([x[0] for x in v]):>11.4f}"
          f"{np.mean([x[1] for x in v]):>13.3f}")

json.dump({"model": MODEL, "n_paired": len(paired),
           "spearman_meancos_spread": {"rho": float(rho), "pval": float(p)},
           "spearman_mincos_spread": {"rho": float(rho2), "pval": float(p2)},
           "pearson_meancos_spread": {"r": float(pr), "pval": float(pp)},
           "pairwise_means": {k: float(np.mean(v)) for k, v in allp.items()},
           "results": out}, open(OUT, "w"), indent=2)
print(f"\n✓ Written: {OUT}")
