"""
Script 14 — Referee 2, Major #1: does the cross-notation effect generalize beyond ESOL?

Runs the Phase 1 protocol on additional MoleculeNet datasets covering BOTH a
regression task and a classification task, as requested.

Key design change from Phase 1: SMILES, InChI and SELFIES are all generated
LOCALLY (RDKit and the selfies library). Only IUPAC names require PubChem, and
that notation is therefore optional here — the three offline notations are
sufficient to test whether cross-notation inconsistency replicates, and avoid a
network dependency that previously produced silent failures under rate limiting.

Datasets:
  FreeSolv       regression      hydration free energy      (~640 molecules)
  Lipophilicity  regression      octanol/water logD         (~4200 molecules)
  BBBP           classification  blood-brain barrier        (~2050 molecules)
  BACE           classification  BACE-1 inhibition          (~1510 molecules)

Usage:  python3 14_multidataset_benchmark.py <model> <dataset> [n_molecules]
"""

import requests, json, time, re, csv, io, os, sys, random, gzip
import numpy as np
import selfies as sf
import warnings; warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6:35b-a3b"
DSNAME  = sys.argv[2] if len(sys.argv) > 2 else "FreeSolv"
NMOL    = int(sys.argv[3]) if len(sys.argv) > 3 else 150
GEN     = "http://localhost:11434/api/generate"
SEED    = 42
random.seed(SEED)

S3 = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/"
DATASETS = {
    "FreeSolv":      (S3+"SAMPL.csv",          "regression",
                      "hydration free energy in water, in kcal/mol",
                      "expt", -26.0, 4.0),
    "Lipophilicity": (S3+"Lipophilicity.csv",  "regression",
                      "octanol/water distribution coefficient logD at pH 7.4",
                      "exp", -2.0, 5.0),
    "BBBP":          (S3+"BBBP.csv",           "classification",
                      "blood-brain barrier penetration", "p_np", 0, 1),
    "BACE":          (S3+"bace.csv",           "classification",
                      "BACE-1 enzyme inhibition", "Class", 0, 1),
}

url, task,描述, target_col, LO, HI = DATASETS[DSNAME]
DESC = 描述
OUT  = f"multids_{DSNAME}_{MODEL.replace(':','_').replace('.','')}.json"
CKPT = OUT.replace(".json", "_ckpt.json")
NOTS = ["SMILES", "InChI", "SELFIES"]     # all generated locally

if task == "regression":
    PROMPT = ("You are a chemistry expert. Predict the {prop} for the molecule below.\n\n"
              "Molecule ({rep}): {mol}\n\n"
              "Reply with ONLY one decimal number between {lo} and {hi}. No text, no units.")
else:
    PROMPT = ("You are a chemistry expert. Does the molecule below show {prop}?\n\n"
              "Molecule ({rep}): {mol}\n\n"
              "Reply with ONLY the single digit 1 for yes or 0 for no. No text.")


def query(rep, mol):
    p = (PROMPT.format(prop=DESC, rep=rep, mol=mol, lo=LO, hi=HI)
         if task == "regression" else PROMPT.format(prop=DESC, rep=rep, mol=mol))
    for _ in range(2):
        try:
            r = requests.post(GEN, json={"model": MODEL, "prompt": p, "stream": False,
                                         "think": False,
                                         "options": {"temperature": 0, "num_predict": 12}},
                              timeout=150)
            raw = r.json().get("response", "").strip()
            if task == "classification":
                m = re.search(r"[01]", raw)
                if m:
                    return int(m.group())
            else:
                nums = re.findall(r"-?\d+\.?\d*", raw)
                if nums:
                    v = float(nums[0])
                    if LO <= v <= HI:
                        return v
        except Exception:
            time.sleep(2)
    return None


# ── load dataset ──────────────────────────────────────────────────────────────
raw = requests.get(url, timeout=120)
text = gzip.decompress(raw.content).decode() if url.endswith(".gz") else raw.text
rows = list(csv.DictReader(io.StringIO(text)))
cols = list(rows[0].keys())
smi_col = next(c for c in cols if 'smiles' in c.lower() or c.lower() == 'mol')
print(f"{DSNAME}: {len(rows)} rows | smiles='{smi_col}' target='{target_col}' task={task}")

recs = []
for r in rows:
    try:
        y = float(r[target_col])
    except (ValueError, KeyError):
        continue
    smi = r[smi_col].strip()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    try:
        inchi = Chem.MolToInchi(m)
    except Exception:
        inchi = None
    try:
        selfies = sf.encoder(Chem.MolToSmiles(m))
    except Exception:
        selfies = None
    if not (inchi and selfies):
        continue
    recs.append({"smiles": Chem.MolToSmiles(m), "InChI": inchi,
                 "SELFIES": selfies, "y": y})

random.shuffle(recs)
recs = recs[:NMOL]
print(f"Usable with all three notations: {len(recs)}\n")

results, done = [], set()
if os.path.exists(CKPT):
    results = json.load(open(CKPT))["results"]
    done = {r["smiles"] for r in results}
    print(f"Resuming: {len(results)} done\n")

todo = [r for r in recs if r["smiles"] not in done]
t0 = time.time()

for i, rec in enumerate(todo):
    reps = {"SMILES": rec["smiles"], "InChI": rec["InChI"], "SELFIES": rec["SELFIES"]}
    out = {"smiles": rec["smiles"], "y_true": rec["y"]}
    for n in NOTS:
        out[f"pred_{n}"] = query(n, reps[n])
    preds = [out[f"pred_{n}"] for n in NOTS if out[f"pred_{n}"] is not None]
    out["n_valid"] = len(preds)
    if task == "regression":
        out["spread"] = round(max(preds) - min(preds), 4) if len(preds) >= 2 else None
    else:
        out["disagree"] = (len(set(preds)) > 1) if len(preds) >= 2 else None
    results.append(out)

    if (i + 1) % 10 == 0:
        json.dump({"results": results}, open(CKPT, "w"))
        eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"  [{i+1}/{len(todo)}] valid={out['n_valid']}/3  ETA {eta:.1f} min", flush=True)

# ── statistics ────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(f"{DSNAME}  ({task})  —  {MODEL}   n = {len(results)}")
print("=" * 68)

per = {}
for n in NOTS:
    v = [r for r in results if r.get(f"pred_{n}") is not None]
    if not v:
        continue
    if task == "regression":
        mae = float(np.mean([abs(r[f"pred_{n}"] - r["y_true"]) for r in v]))
        per[n] = {"n": len(v), "mae": mae}
        print(f"  {n:<9} n={len(v):>4}  MAE={mae:.3f}")
    else:
        acc = float(np.mean([r[f"pred_{n}"] == int(r["y_true"]) for r in v]))
        pos = float(np.mean([r[f"pred_{n}"] == 1 for r in v]))
        per[n] = {"n": len(v), "accuracy": acc, "pred_pos_rate": pos}
        print(f"  {n:<9} n={len(v):>4}  accuracy={acc*100:.1f}%  predicted-positive={pos*100:.1f}%")

full = [r for r in results if r["n_valid"] == len(NOTS)]
summary = {"dataset": DSNAME, "task": task, "model": MODEL,
           "n": len(results), "n_complete": len(full), "per_notation": per}

if task == "regression":
    sp = np.array([r["spread"] for r in full])
    for th in (0.5, 1.0, 2.0):
        print(f"  molecules with 3-notation spread > {th}: {np.mean(sp>th)*100:.1f}%")
    summary["spread"] = {"mean": float(sp.mean()), "median": float(np.median(sp)),
                         "pct_gt_0.5": float(np.mean(sp > 0.5) * 100),
                         "pct_gt_1.0": float(np.mean(sp > 1.0) * 100)}
    print(f"  mean spread = {sp.mean():.3f}  median = {np.median(sp):.3f}")
else:
    dis = float(np.mean([r["disagree"] for r in full]) * 100)
    summary["pct_disagree"] = dis
    print(f"  molecules where the three notations DISAGREE on the label: {dis:.1f}%")

json.dump({**summary, "results": results}, open(OUT, "w"), indent=2)
print(f"\n✓ Written: {OUT}")
