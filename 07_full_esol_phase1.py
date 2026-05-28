"""
Script 07 — FULL ESOL Phase 1: Complete 4-notation benchmark
- ALL 1128 ESOL molecules × 4 notations (SMILES, IUPAC, InChI, SELFIES)
- Checkpoint/resume: saves every 25 molecules — safe to interrupt and restart
- Comprehensive statistics: spread, MAE, sign-flip, Fisher exact, Kruskal-Wallis
- Output: phase1_full_results.json + phase1_full_stats.json
"""

import requests, json, time, re, csv, io, os, sys, urllib.parse
import numpy as np
import selfies as sf
from collections import defaultdict, Counter
from scipy.stats import kruskal, mannwhitneyu, fisher_exact
from scipy import stats as scipy_stats

OLLAMA_URL = "http://localhost:11434/api/generate"
ESOL_URL   = ("https://raw.githubusercontent.com/deepchem/deepchem/"
               "master/datasets/delaney-processed.csv")
PUBCHEM    = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
OUT_DIR    = "/home/snu/workspace/notation_research"
CKPT_FILE  = f"{OUT_DIR}/phase1_full_checkpoint.json"
OUT_FILE   = f"{OUT_DIR}/phase1_full_results.json"
STATS_FILE = f"{OUT_DIR}/phase1_full_stats.json"

os.makedirs(OUT_DIR, exist_ok=True)


# ─── Molecule classification (no rdkit) ───────────────────────────────────────
def classify(smiles: str) -> str:
    s = smiles
    if re.search(r'[nNsS].*\d|\d.*[nNsS]', s):            return "heterocyclic"
    if re.search(r'[a-z]', s) and re.search(r'\d', s):    return "aromatic"
    if 'C(=O)O' in s or 'C(O)=O' in s or 'OC(=O)' in s: return "carboxylic/ester"
    if re.search(r'[FClBrI]', s):                          return "halogenated"
    if re.search(r'(?<![a-z])O(?![a-z=\(])', s):          return "alcohol/phenol"
    if re.search(r'(?<![a-z])N(?![a-z=\(+])', s):         return "amine"
    return "aliphatic"


# ─── PubChem lookup with retry ─────────────────────────────────────────────────
def pubchem_lookup(smiles: str, retries: int = 3):
    for attempt in range(retries):
        try:
            encoded = urllib.parse.quote(smiles, safe='')
            url = (f"{PUBCHEM}/smiles/{encoded}"
                   f"/property/IUPACName,InChI/JSON")
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                props = r.json()["PropertyTable"]["Properties"][0]
                return props.get("IUPACName"), props.get("InChI")
            elif r.status_code == 404:
                return None, None
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    return None, None


# ─── SMILES → SELFIES ─────────────────────────────────────────────────────────
def smiles_to_selfies(smiles: str):
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


# ─── Qwen3.6 query with retry ─────────────────────────────────────────────────
PROMPT = (
    "You are a chemistry expert. Predict the water solubility of the molecule "
    "below as logS (log₁₀ molar solubility).\n\n"
    "Molecule ({rep}): {mol}\n\n"
    "Reply with ONLY one decimal number between -10.0 and 2.0. "
    "No text, no units."
)

def query_qwen(rep: str, mol: str, retries: int = 2):
    prompt = PROMPT.format(rep=rep, mol=mol)
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": "qwen3.6:35b-a3b",
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 12}
            }, timeout=90)
            raw = r.json()["response"].strip()
            nums = re.findall(r"-?\d+\.?\d*", raw)
            if nums:
                val = float(nums[0])
                if -10.0 <= val <= 2.0:
                    return val
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
    return None


# ─── Load full ESOL (1128 molecules) ──────────────────────────────────────────
def load_full_esol():
    print("Downloading full ESOL dataset...")
    r = requests.get(ESOL_URL, timeout=30)
    rows = list(csv.DictReader(io.StringIO(r.text)))
    smiles_col = next(c for c in rows[0] if 'smiles' in c.lower())
    logs_col   = next(
        c for c in rows[0]
        if ('measured' in c.lower() or 'log' in c.lower())
        and 'smiles' not in c.lower()
    )
    mols = []
    for row in rows:
        sm = row[smiles_col].strip()
        try:
            ls = float(row[logs_col])
        except ValueError:
            continue
        mols.append((sm, ls, classify(sm)))
    print(f"✓ Loaded {len(mols)} molecules")
    return mols


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(values, len(values), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, ci + (100 - ci) / 2)
    return float(np.mean(values)), float(lo), float(hi)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("PHASE 1 — FULL ESOL: 4-notation notation inconsistency benchmark")
    print("Notations: SMILES | IUPAC | InChI | SELFIES")
    print("=" * 72)

    # ── Resume from checkpoint if available ──────────────────────────────────
    completed = {}
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE) as f:
            ckpt_list = json.load(f)
        completed = {r["smiles"]: r for r in ckpt_list}
        print(f"✓ Resuming: {len(completed)} molecules already completed")

    mols = load_full_esol()
    remaining = [(sm, ls, cls) for sm, ls, cls in mols if sm not in completed]
    print(f"  Remaining: {len(remaining)}/{len(mols)} molecules to process\n")

    # ── Phase A: PubChem lookups ──────────────────────────────────────────────
    print("[Phase A] Fetching IUPAC + InChI from PubChem REST API...")
    iupac_cache = {}
    for i, (sm, ls, cls) in enumerate(remaining):
        iupac, inchi = pubchem_lookup(sm)
        iupac_cache[sm] = (iupac, inchi)
        time.sleep(0.35)          # ≈3 req/s — safely within PubChem rate limit
        if (i + 1) % 100 == 0:
            ok = sum(1 for v in iupac_cache.values() if v[0])
            print(f"  PubChem: {i+1}/{len(remaining)}  resolved={ok}")

    # ── Phase B: SELFIES conversion ───────────────────────────────────────────
    print("\n[Phase B] SMILES → SELFIES (local conversion)...")
    selfies_cache = {sm: smiles_to_selfies(sm) for sm, _, _ in remaining}
    ok_sf = sum(1 for v in selfies_cache.values() if v)
    print(f"  ✓ {ok_sf}/{len(remaining)} SELFIES conversions succeeded")

    # ── Phase C: LLM queries ──────────────────────────────────────────────────
    total = len(remaining)
    print(f"\n[Phase C] Querying Qwen3.6 — {total} molecules × 4 notations "
          f"= {total*4} total queries")
    print("  Checkpoints every 25 molecules. Safe to Ctrl-C and resume.\n")

    batch = list(completed.values())
    start_time = time.time()

    for idx, (sm, true_ls, cls) in enumerate(remaining):
        iupac, inchi = iupac_cache.get(sm, (None, None))
        sel_str      = selfies_cache.get(sm)
        preds        = {}

        for rep, mol_str in [
            ("SMILES",  sm),
            ("IUPAC",   iupac),
            ("InChI",   inchi),
            ("SELFIES", sel_str),
        ]:
            if mol_str is None:
                preds[rep] = None
                continue
            preds[rep] = query_qwen(rep, mol_str)
            time.sleep(0.15)

        valid_all  = [v for v in preds.values() if v is not None]
        valid_3    = [preds[k] for k in ("SMILES","IUPAC","InChI")
                      if preds.get(k) is not None]

        if len(valid_all) < 2:
            continue

        spread_4 = round(max(valid_all) - min(valid_all), 4)
        spread_3 = (round(max(valid_3) - min(valid_3), 4)
                    if len(valid_3) >= 2 else None)

        # Sign flip: true logS < -1 (insoluble) predicted > 0 (highly soluble)
        def flip(v): return (v is not None and v > 0 and true_ls < -1)

        entry = {
            "smiles":   sm,
            "class":    cls,
            "true_logS": round(true_ls, 4),
            "pred_SMILES":  (round(preds["SMILES"],  4)
                             if preds["SMILES"]  is not None else None),
            "pred_IUPAC":   (round(preds["IUPAC"],   4)
                             if preds["IUPAC"]   is not None else None),
            "pred_InChI":   (round(preds["InChI"],   4)
                             if preds["InChI"]   is not None else None),
            "pred_SELFIES": (round(preds["SELFIES"], 4)
                             if preds["SELFIES"] is not None else None),
            "spread_4rep":  spread_4,
            "spread_3rep":  spread_3,
            "sign_flip_SMILES":  flip(preds["SMILES"]),
            "sign_flip_IUPAC":   flip(preds["IUPAC"]),
            "sign_flip_InChI":   flip(preds["InChI"]),
            "sign_flip_SELFIES": flip(preds["SELFIES"]),
            "n_valid":    len(valid_all),
        }
        batch.append(entry)

        # Progress display
        flips = [k for k in ("SMILES","IUPAC","InChI","SELFIES")
                 if entry.get(f"sign_flip_{k}")]
        eta_s  = (time.time() - start_time) / (idx + 1) * (total - idx - 1)
        eta_h  = eta_s / 3600
        status = f"sp4={spread_4:.2f}"
        if flips: status += f" ⚠FLIP({','.join(flips)})"
        print(f"  [{idx+1:4d}/{total}] [{cls:18s}] "
              f"logS={true_ls:6.2f} → {status}  ETA {eta_h:.1f}h")

        # Checkpoint every 25 molecules
        if (idx + 1) % 25 == 0:
            with open(CKPT_FILE, "w") as f:
                json.dump(batch, f)
            print(f"  ─── CHECKPOINT saved ({len(batch)} total) ───")

    # Final save
    with open(CKPT_FILE, "w") as f:
        json.dump(batch, f)
    with open(OUT_FILE, "w") as f:
        json.dump(batch, f, indent=2)

    # ── Comprehensive statistics ──────────────────────────────────────────────
    results = batch
    print("\n" + "=" * 72)
    print(f"FINAL STATISTICS  (N = {len(results)} molecules)")
    print("=" * 72)

    spreads4 = np.array([r["spread_4rep"] for r in results
                         if r["spread_4rep"] is not None])
    spreads3 = np.array([r["spread_3rep"] for r in results
                         if r["spread_3rep"] is not None])

    n_inc4 = int((spreads4 > 0.5).sum())
    n_inc3 = int((spreads3 > 0.5).sum())

    m4, lo4, hi4 = bootstrap_ci(spreads4)
    m3, lo3, hi3 = bootstrap_ci(spreads3)

    print(f"\n  4-notation spread: mean={m4:.3f} [{lo4:.3f},{hi4:.3f}] "
          f"median={np.median(spreads4):.3f}  "
          f"inconsistent={n_inc4}/{len(spreads4)} "
          f"({n_inc4/len(spreads4)*100:.1f}%)")
    print(f"  3-notation spread: mean={m3:.3f} [{lo3:.3f},{hi3:.3f}] "
          f"median={np.median(spreads3):.3f}  "
          f"inconsistent={n_inc3}/{len(spreads3)} "
          f"({n_inc3/len(spreads3)*100:.1f}%)")

    # Per-notation MAE + sign-flip
    notation_stats = {}
    print("\n  Per-notation performance:")
    for rep in ("SMILES", "IUPAC", "InChI", "SELFIES"):
        pf, sf_field = f"pred_{rep}", f"sign_flip_{rep}"
        vr = [r for r in results if r.get(pf) is not None]
        if not vr:
            continue
        mae = float(np.mean([abs(r[pf] - r["true_logS"]) for r in vr]))
        flips = int(sum(r.get(sf_field, False) for r in vr))
        flip_pct = flips / len(vr) * 100
        notation_stats[rep] = {"n": len(vr), "mae": mae,
                                "sign_flips": flips,
                                "sign_flip_pct": flip_pct}
        print(f"    {rep:8s}: n={len(vr):5d}  MAE={mae:.3f}  "
              f"sign_flips={flips:4d}/{len(vr)} ({flip_pct:.1f}%)")

    # Fisher exact: SELFIES vs SMILES sign-flip rate
    smi_n  = notation_stats.get("SMILES",  {})
    sel_n  = notation_stats.get("SELFIES", {})
    if smi_n and sel_n:
        smi_flip = smi_n["sign_flips"]; smi_tot = smi_n["n"]
        sel_flip = sel_n["sign_flips"]; sel_tot = sel_n["n"]
        table = [[smi_flip, smi_tot - smi_flip],
                 [sel_flip, sel_tot - sel_flip]]
        OR, pval_fe = fisher_exact(table, alternative='greater')
        print(f"\n  Fisher exact (SMILES > SELFIES flip rate):")
        print(f"    SMILES {smi_flip}/{smi_tot}  SELFIES {sel_flip}/{sel_tot}")
        print(f"    OR={OR:.3f}  p={pval_fe:.4f}")

    # Kruskal-Wallis across structural classes
    by_cls = defaultdict(list)
    for r in results:
        if r["spread_4rep"] is not None:
            by_cls[r["class"]].append(r["spread_4rep"])

    cls_groups = {c: np.array(v) for c, v in by_cls.items() if len(v) >= 5}
    if len(cls_groups) >= 3:
        H, p_kw = kruskal(*cls_groups.values())
        print(f"\n  Kruskal-Wallis (spread across classes): H={H:.3f}  p={p_kw:.4f}")

    # Class-level breakdown
    print("\n  Class breakdown (4-notation spread):")
    print(f"  {'Class':<22} {'n':>5} {'mean':>7} {'median':>8} "
          f"{'n_incons':>10} {'sf_SMILES':>10} {'sf_SELFIES':>11}")
    class_stats = {}
    for cls in sorted(cls_groups, key=lambda c: -cls_groups[c].mean()):
        cr = [r for r in results if r["class"] == cls]
        arr = cls_groups[cls]
        n_inc = int((arr > 0.5).sum())
        sf_sm = sum(1 for r in cr if r.get("sign_flip_SMILES"))
        sf_se = sum(1 for r in cr if r.get("sign_flip_SELFIES"))
        n_sm  = sum(1 for r in cr if r.get("pred_SMILES") is not None)
        n_se  = sum(1 for r in cr if r.get("pred_SELFIES") is not None)
        class_stats[cls] = {
            "n": len(cr), "mean_spread": float(arr.mean()),
            "median_spread": float(np.median(arr)),
            "n_inconsistent": n_inc,
            "sign_flips_SMILES": sf_sm, "sign_flips_SELFIES": sf_se,
            "n_SMILES": n_sm, "n_SELFIES": n_se,
        }
        print(f"  {cls:<22} {len(cr):>5} {arr.mean():>7.3f} {np.median(arr):>8.3f} "
              f"  {n_inc:4d}/{len(cr):<5d}  {sf_sm:3d}/{n_sm}  {sf_se:4d}/{n_se}")

    # Pairwise class comparisons (halogenated vs others)
    if "halogenated" in cls_groups:
        print("\n  Mann-Whitney pairwise (halogenated vs others):")
        for other_cls, other_arr in sorted(cls_groups.items()):
            if other_cls == "halogenated":
                continue
            stat, p = mannwhitneyu(cls_groups["halogenated"], other_arr,
                                   alternative='greater')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    vs {other_cls:<22} p={p:.4f} {sig}")

    # Build full summary statistics dict
    stats = {
        "n_molecules": len(results),
        "spread_4notation": {
            "mean": m4, "ci_95_lo": lo4, "ci_95_hi": hi4,
            "median": float(np.median(spreads4)),
            "n_inconsistent": n_inc4, "pct_inconsistent": n_inc4/len(spreads4)*100,
        },
        "spread_3notation": {
            "mean": m3, "ci_95_lo": lo3, "ci_95_hi": hi3,
            "median": float(np.median(spreads3)),
            "n_inconsistent": n_inc3, "pct_inconsistent": n_inc3/len(spreads3)*100,
        },
        "per_notation": notation_stats,
        "fisher_exact_SMILES_vs_SELFIES": {
            "OR": float(OR), "pval": float(pval_fe),
        } if smi_n and sel_n else {},
        "kruskal_wallis": {
            "H": float(H), "pval": float(p_kw),
        } if len(cls_groups) >= 3 else {},
        "class_stats": class_stats,
    }

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Results:    {OUT_FILE}")
    print(f"✓ Statistics: {STATS_FILE}")
    print(f"✓ Total molecules processed: {len(results)}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
