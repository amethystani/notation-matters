"""
Script 09 — Re-analyse Phase 1 with the corrected structural classifier.

Referee 3 identified that classify() used re.search(r'[FClBrI]', s), a character
class matching any of F/C/l/B/r/I individually rather than the halogen tokens.
This affects ONLY the class label attached to each molecule; the underlying LLM
predictions are unchanged. This script therefore re-labels the existing Phase 1
results and recomputes the class-stratified statistics, leaving every reported
per-notation figure (MAE, sign-flip rate, four-notation spread) untouched.

Input:  phase1_full_results.json  (from the original Phase 1 run)
Output: phase1_reclassified_stats.json + a printed report

Usage:  python3 09_reclassify_phase1.py [path/to/phase1_full_results.json]
"""

import json, re, sys, os
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import kruskal, mannwhitneyu

IN_FILE  = sys.argv[1] if len(sys.argv) > 1 else "phase1_full_results.json"
OUT_FILE = "phase1_reclassified_stats.json"


# ─── Classifiers ──────────────────────────────────────────────────────────────
def classify_buggy(smiles: str) -> str:
    """The original, as published — retained for the before/after comparison."""
    s = smiles
    if re.search(r'[nNsS].*\d|\d.*[nNsS]', s):            return "heterocyclic"
    if re.search(r'[a-z]', s) and re.search(r'\d', s):    return "aromatic"
    if 'C(=O)O' in s or 'C(O)=O' in s or 'OC(=O)' in s: return "carboxylic/ester"
    if re.search(r'[FClBrI]', s):                          return "halogenated"
    if re.search(r'(?<![a-z])O(?![a-z=\(])', s):          return "alcohol/phenol"
    if re.search(r'(?<![a-z])N(?![a-z=\(+])', s):         return "amine"
    return "aliphatic"


def classify(smiles: str) -> str:
    """Corrected: halogen rule tests for the element tokens, not a character class."""
    s = smiles
    if re.search(r'[nNsS].*\d|\d.*[nNsS]', s):            return "heterocyclic"
    if re.search(r'[a-z]', s) and re.search(r'\d', s):    return "aromatic"
    if 'C(=O)O' in s or 'C(O)=O' in s or 'OC(=O)' in s: return "carboxylic/ester"
    if re.search(r'Cl|Br|F|I', s):                         return "halogenated"
    if re.search(r'(?<![a-z])O(?![a-z=\(])', s):          return "alcohol/phenol"
    if re.search(r'(?<![a-z])N(?![a-z=\(+])', s):         return "amine"
    return "aliphatic"


# ─── Load ─────────────────────────────────────────────────────────────────────
if not os.path.exists(IN_FILE):
    sys.exit(f"ERROR: {IN_FILE} not found.\n"
             f"Retrieve it from the Phase 1 output directory, e.g.\n"
             f"  scp <host>:/home/snu/workspace/notation_research/"
             f"phase1_full_results.json .")

with open(IN_FILE) as f:
    data = json.load(f)
results = data["molecule_results"] if isinstance(data, dict) else data
print(f"Loaded {len(results)} molecules from {IN_FILE}")

# Locate the SMILES and spread fields without assuming exact key names.
probe = results[0]
sm_key = next(k for k in probe if 'smiles' in k.lower())
sp_key = next(k for k in probe if 'spread' in k.lower())
print(f"Using fields: smiles='{sm_key}', spread='{sp_key}'\n")


# ─── Re-label ─────────────────────────────────────────────────────────────────
changed = 0
for r in results:
    old = r.get("class", classify_buggy(r[sm_key]))
    new = classify(r[sm_key])
    r["class_original"], r["class"] = old, new
    if old != new:
        changed += 1

co = Counter(r["class_original"] for r in results)
cn = Counter(r["class"] for r in results)

print("CLASS ASSIGNMENT — original vs corrected")
print(f"{'class':<20}{'orig':>7}{'fixed':>7}{'delta':>8}")
for k in sorted(set(co) | set(cn), key=lambda k: -cn.get(k, 0)):
    d = cn.get(k, 0) - co.get(k, 0)
    print(f"{k:<20}{co.get(k,0):>7}{cn.get(k,0):>7}{d:>+8}")
print(f"\nmolecules whose class changes: {changed}/{len(results)} "
      f"({changed/len(results)*100:.1f}%)")

unchanged = [k for k in cn if cn.get(k, 0) == co.get(k, 0) and cn.get(k, 0) > 0]
print(f"classes with identical membership: {', '.join(sorted(unchanged))}\n")


# ─── Class-stratified statistics (corrected) ──────────────────────────────────
by_cls = defaultdict(list)
for r in results:
    v = r.get(sp_key)
    if v is not None:
        by_cls[r["class"]].append(float(v))

groups = {c: np.array(v) for c, v in by_cls.items() if len(v) >= 5}

print("FOUR-NOTATION SPREAD BY CORRECTED CLASS")
print(f"{'class':<20}{'n':>6}{'mean':>9}{'median':>9}{'%>0.5':>8}")
class_stats = {}
for c in sorted(groups, key=lambda c: -groups[c].mean()):
    a = groups[c]
    pct = float((a > 0.5).mean() * 100)
    class_stats[c] = {"n": int(len(a)), "mean_spread": float(a.mean()),
                      "median_spread": float(np.median(a)), "pct_inconsistent": pct}
    print(f"{c:<20}{len(a):>6}{a.mean():>9.3f}{np.median(a):>9.3f}{pct:>7.1f}%")

kw = None
if len(groups) >= 3:
    H, p = kruskal(*groups.values())
    kw = {"H": float(H), "pval": float(p), "n_groups": len(groups)}
    print(f"\nKruskal-Wallis across {len(groups)} corrected classes: "
          f"H={H:.3f}, p={p:.3e}")

# Pairwise against carboxylic/ester — the class carrying the manuscript's claim
pairwise = {}
if "carboxylic/ester" in groups:
    print("\nMann-Whitney, carboxylic/ester vs. each other class (one-sided):")
    for c, a in sorted(groups.items()):
        if c == "carboxylic/ester":
            continue
        stat, p = mannwhitneyu(groups["carboxylic/ester"], a, alternative="greater")
        pairwise[c] = {"U": float(stat), "pval": float(p)}
        flag = "*" if p < 0.05 else "ns"
        print(f"  vs {c:<20} p={p:.4f} {flag}")

json.dump({
    "n_molecules": len(results),
    "n_class_changed": changed,
    "counts_original": dict(co),
    "counts_corrected": dict(cn),
    "classes_with_identical_membership": sorted(unchanged),
    "class_stats": class_stats,
    "kruskal_wallis": kw,
    "mannwhitney_vs_carboxylic": pairwise,
}, open(OUT_FILE, "w"), indent=2)

print(f"\n✓ Written: {OUT_FILE}")
print("\nInsert into main.tex:")
print("  line 308 — corrected class counts (see counts_corrected above)")
print("  line 401 — Kruskal-Wallis H/p and the Mann-Whitney comparisons")
