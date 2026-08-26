"""
Script 10 — Referee 1, Point 2: cross-notation benchmark on a second open LLM.

Runs the Phase 1 protocol unchanged (identical prompt template, identical decoding
settings) against a second, architecturally distinct open model, on a subset of
ESOL stratified by the corrected structural classification. The purpose is to test
whether cross-notation inconsistency is a property of Qwen3.6 specifically or of
open LLMs applied to molecular property prediction more generally.

Also records exact model version, quantization, decoding parameters, and a full
accounting of parse/response failures by notation, as the referee requested.

Usage:  python3 10_second_model_benchmark.py [model_tag] [n_per_class]
Run on the host where the Ollama endpoint lives.
"""

import requests, json, time, re, csv, io, os, sys, urllib.parse, random
import numpy as np
import selfies as sf
from collections import defaultdict

MODEL      = sys.argv[1] if len(sys.argv) > 1 else "nous-hermes2:34b"
N_PER_CLS  = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OLLAMA_URL = "http://localhost:11434/api/generate"
ESOL_URL   = ("https://raw.githubusercontent.com/deepchem/deepchem/"
              "master/datasets/delaney-processed.csv")
PUBCHEM    = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
OUT        = f"secondmodel_{MODEL.replace(':','_').replace('.','')}_results.json"
CKPT       = OUT.replace("_results", "_checkpoint")
SEED       = 42
random.seed(SEED); np.random.seed(SEED)

# ── Identical prompt and decoding settings to Phase 1 ─────────────────────────
PROMPT = (
    "You are a chemistry expert. Predict the water solubility of the molecule "
    "below as logS (log₁₀ molar solubility).\n\n"
    "Molecule ({rep}): {mol}\n\n"
    "Reply with ONLY one decimal number between -10.0 and 2.0. "
    "No text, no units."
)
OPTIONS = {"temperature": 0, "num_predict": 12}

# Failure accounting, as requested by Referee 1 Point 2
FAIL = defaultdict(lambda: defaultdict(int))


def classify(s: str) -> str:
    """Corrected classifier (halogen rule tests element tokens, not a char class)."""
    if re.search(r'[nNsS].*\d|\d.*[nNsS]', s):            return "heterocyclic"
    if re.search(r'[a-z]', s) and re.search(r'\d', s):    return "aromatic"
    if 'C(=O)O' in s or 'C(O)=O' in s or 'OC(=O)' in s: return "carboxylic/ester"
    if re.search(r'Cl|Br|F|I', s):                         return "halogenated"
    if re.search(r'(?<![a-z])O(?![a-z=\(])', s):          return "alcohol/phenol"
    if re.search(r'(?<![a-z])N(?![a-z=\(+])', s):         return "amine"
    return "aliphatic"


def query(rep: str, mol: str, retries: int = 2):
    """Returns (value, failure_reason). failure_reason is None on success."""
    prompt = PROMPT.format(rep=rep, mol=mol)
    last = "unknown"
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": MODEL, "prompt": prompt, "stream": False,
                "think": False, "options": OPTIONS}, timeout=120)
            raw = r.json().get("response", "").strip()
            if not raw:
                last = "empty_response"; continue
            nums = re.findall(r"-?\d+\.?\d*", raw)
            if not nums:
                last = "no_number_parsed"; continue
            val = float(nums[0])
            if not (-10.0 <= val <= 2.0):
                last = "out_of_range"; continue
            return val, None
        except requests.Timeout:
            last = "timeout"
        except Exception:
            last = "request_error"
        if attempt < retries - 1:
            time.sleep(2)
    return None, last


def pubchem(smiles: str, retries: int = 3):
    for a in range(retries):
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


# ── Load and stratify ─────────────────────────────────────────────────────────
print(f"Model: {MODEL} | {N_PER_CLS} molecules per class | seed={SEED}")
rows = list(csv.DictReader(io.StringIO(requests.get(ESOL_URL, timeout=60).text)))
sc = next(c for c in rows[0] if 'smiles' in c.lower())
lc = next(c for c in rows[0] if ('measured' in c.lower() or 'log' in c.lower())
          and 'smiles' not in c.lower())

pool = defaultdict(list)
for r in rows:
    try:
        pool[classify(r[sc].strip())].append((r[sc].strip(), float(r[lc])))
    except ValueError:
        continue

subset = []
for cls, items in sorted(pool.items()):
    random.shuffle(items)
    take = items[:N_PER_CLS]
    subset += [(s, l, cls) for s, l in take]
    print(f"  {cls:<18} available={len(items):>4}  sampled={len(take)}")
print(f"Subset total: {len(subset)} molecules\n")

# ── Resume ────────────────────────────────────────────────────────────────────
results, done = [], set()
if os.path.exists(CKPT):
    results = json.load(open(CKPT))["results"]
    done = {r["smiles"] for r in results}
    print(f"Resuming: {len(results)} already done\n")

todo = [m for m in subset if m[0] not in done]
NOTS = ["SMILES", "IUPAC", "InChI", "SELFIES"]
t0 = time.time()

for i, (smi, true_ls, cls) in enumerate(todo):
    iupac, inchi = pubchem(smi)
    try:
        selfies = sf.encoder(smi)
    except Exception:
        selfies = None

    reps = {"SMILES": smi, "IUPAC": iupac, "InChI": inchi, "SELFIES": selfies}
    rec = {"smiles": smi, "true_logS": true_ls, "class": cls}

    for n in NOTS:
        if not reps[n]:
            rec[f"pred_{n}"] = None
            FAIL[n]["notation_unavailable"] += 1
            continue
        val, why = query(n, reps[n])
        rec[f"pred_{n}"] = val
        if val is None:
            FAIL[n][why] += 1
        else:
            rec[f"sign_flip_{n}"] = bool(true_ls < -1 and val > 0)

    preds = [rec[f"pred_{n}"] for n in NOTS if rec.get(f"pred_{n}") is not None]
    rec["n_valid"] = len(preds)
    rec["spread_4rep"] = round(max(preds) - min(preds), 4) if len(preds) >= 2 else None
    results.append(rec)

    if (i + 1) % 10 == 0:
        json.dump({"results": results, "failures": {k: dict(v) for k, v in FAIL.items()}},
                  open(CKPT, "w"))
        el = time.time() - t0
        eta = el / (i + 1) * (len(todo) - i - 1) / 60
        print(f"  [{i+1:4d}/{len(todo)}] {cls:<18} valid={rec['n_valid']}/4  ETA {eta:.1f} min")

# ── Statistics ────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(f"RESULTS — {MODEL}  (n = {len(results)})")
print("=" * 68)
print(f"\n{'notation':<10}{'n_valid':>9}{'MAE':>9}{'sign-flip':>14}")
per = {}
for n in NOTS:
    v = [r for r in results if r.get(f"pred_{n}") is not None]
    if not v:
        print(f"{n:<10}{'0':>9}{'--':>9}{'--':>14}"); continue
    mae = float(np.mean([abs(r[f"pred_{n}"] - r["true_logS"]) for r in v]))
    sf_n = sum(1 for r in v if r.get(f"sign_flip_{n}"))
    per[n] = {"n": len(v), "mae": mae, "sign_flips": sf_n,
              "sign_flip_pct": sf_n / len(v) * 100}
    print(f"{n:<10}{len(v):>9}{mae:>9.3f}{sf_n:>7}/{len(v):<4}{sf_n/len(v)*100:>5.1f}%")

sp = [r["spread_4rep"] for r in results if r.get("spread_4rep") is not None]
inc = float(np.mean([s > 0.5 for s in sp]) * 100)
print(f"\nFour-notation spread: mean={np.mean(sp):.3f}  median={np.median(sp):.2f}  "
      f">0.5 logS in {inc:.1f}% of molecules")

matched = [r for r in results if all(r.get(f"pred_{n}") is not None for n in NOTS)]
print(f"Matched complete-case: {len(matched)}/{len(results)} "
      f"({len(matched)/len(results)*100:.1f}%)")

print("\nFailure accounting by notation (Referee 1, Point 2):")
for n in NOTS:
    tot = sum(FAIL[n].values())
    detail = ", ".join(f"{k}={v}" for k, v in sorted(FAIL[n].items())) or "none"
    print(f"  {n:<9} total={tot:<4} {detail}")

json.dump({"model": MODEL, "seed": SEED, "options": OPTIONS,
           "prompt_template": PROMPT, "n_molecules": len(results),
           "per_notation": per,
           "spread": {"mean": float(np.mean(sp)), "median": float(np.median(sp)),
                      "pct_inconsistent": inc},
           "n_matched": len(matched),
           "failures": {k: dict(v) for k, v in FAIL.items()},
           "results": results}, open(OUT, "w"), indent=2)
print(f"\n✓ Written: {OUT}")
