"""
Script 12 — Model validity probe.

Before accepting any model into the cross-notation benchmark, verify that it is
actually performing the prediction task rather than emitting a notation-dependent
constant. This gate was added after nous-hermes2:34b was found to return a single
identical value for all 181 SELFIES prompts (and 88.8% identical for SMILES),
which produced spurious 0% sign-flip rates that had nothing to do with notation
robustness.

A model passes only if, for every notation, it produces a reasonable diversity of
outputs AND its predictions correlate with ground truth.

Usage:  python3 12_model_validity_probe.py <model_tag> [n_probe]
"""

import requests, json, re, sys, time, urllib.parse, random
import numpy as np
import selfies as sf
import csv, io
from scipy.stats import pearsonr

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
N       = int(sys.argv[2]) if len(sys.argv) > 2 else 40
GEN     = "http://localhost:11434/api/generate"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
ESOL    = ("https://raw.githubusercontent.com/deepchem/deepchem/"
           "master/datasets/delaney-processed.csv")
NOTS    = ["SMILES", "IUPAC", "InChI", "SELFIES"]
random.seed(7)

# Pass thresholds
# NOTE: a low unique-fraction is NOT degeneracy. A model answering on a coarse
# grid (e.g. one decimal place over a narrow range) legitimately reuses values:
# Qwen3.6 yields only 2.4-7.1% unique values across 1000 molecules yet correlates
# with ground truth at Spearman rho = 0.66-0.83. The diagnostics that actually
# separate prediction from constant-emission are the MODAL fraction and the
# correlation with truth. Unique-fraction is reported but not gated on.
MIN_UNIQUE_FRAC = 0.0    # reported only, not a pass criterion
MAX_MODAL_FRAC  = 0.50   # no single value may account for >50% of predictions
MIN_ABS_R       = 0.30   # |Pearson r| with truth must reach 0.30

PROMPT = (
    "You are a chemistry expert. Predict the water solubility of the molecule "
    "below as logS (log₁₀ molar solubility).\n\n"
    "Molecule ({rep}): {mol}\n\n"
    "Reply with ONLY one decimal number between -10.0 and 2.0. "
    "No text, no units."
)
OPTIONS = {"temperature": 0, "num_predict": 12}


def query(rep, mol):
    for _ in range(2):
        try:
            r = requests.post(GEN, json={"model": MODEL, "prompt": PROMPT.format(rep=rep, mol=mol),
                                         "stream": False, "think": False,
                                         "options": OPTIONS}, timeout=120)
            nums = re.findall(r"-?\d+\.?\d*", r.json().get("response", "").strip())
            if nums:
                v = float(nums[0])
                if -10.0 <= v <= 2.0:
                    return v
        except Exception:
            time.sleep(2)
    return None


def pubchem(s):
    for _ in range(3):
        try:
            u = (f"{PUBCHEM}/smiles/{urllib.parse.quote(s, safe='')}"
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


rows = list(csv.DictReader(io.StringIO(requests.get(ESOL, timeout=60).text)))
sc = next(c for c in rows[0] if 'smiles' in c.lower())
lc = next(c for c in rows[0] if ('measured' in c.lower() or 'log' in c.lower())
          and 'smiles' not in c.lower())
mols = []
for r in rows:
    try:
        mols.append((r[sc].strip(), float(r[lc])))
    except ValueError:
        pass
random.shuffle(mols)
probe = mols[:N]

print(f"Validity probe: {MODEL} on {len(probe)} molecules\n")
preds = {n: [] for n in NOTS}
truth = {n: [] for n in NOTS}

for i, (smi, ls) in enumerate(probe):
    iupac, inchi = pubchem(smi)
    try:
        selfies = sf.encoder(smi)
    except Exception:
        selfies = None
    reps = {"SMILES": smi, "IUPAC": iupac, "InChI": inchi, "SELFIES": selfies}
    for n in NOTS:
        if reps[n]:
            v = query(n, reps[n])
            if v is not None:
                preds[n].append(v); truth[n].append(ls)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(probe)}", flush=True)

print(f"\n{'notation':<10}{'n':>5}{'uniq':>6}{'uniq%':>8}{'modal%':>8}{'r':>8}  verdict")
overall = True
report = {}
for n in NOTS:
    p, t = preds[n], truth[n]
    if len(p) < 5:
        print(f"{n:<10}{len(p):>5}   -- insufficient responses"); overall = False; continue
    uniq = len(set(p)); uf = uniq / len(p); mf = max(p.count(z) for z in set(p)) / len(p)
    r = float('nan') if uniq == 1 else pearsonr(p, t)[0]
    ok = (mf <= MAX_MODAL_FRAC) and (abs(r) >= MIN_ABS_R if uniq > 1 else False)
    overall &= ok
    report[n] = {"n": len(p), "unique": uniq, "unique_frac": uf,
                 "modal_frac": mf, "pearson_r": None if uniq == 1 else r, "pass": ok}
    rs = "  n/a" if uniq == 1 else f"{r:>7.3f}"
    print(f"{n:<10}{len(p):>5}{uniq:>6}{uf*100:>7.1f}%{mf*100:>7.1f}%{rs}  {'PASS' if ok else 'FAIL'}")

print(f"\nOVERALL: {'PASS — model is performing the task, safe to benchmark' if overall else 'FAIL — model is not performing the task; do not benchmark'}")
print(f"gated on: modal<={MAX_MODAL_FRAC:.0%}, |r|>={MIN_ABS_R}  (unique% reported only)")
json.dump({"model": MODEL, "n_probe": len(probe), "pass": overall,
           "thresholds": {"min_unique_frac": MIN_UNIQUE_FRAC,
                          "max_modal_frac": MAX_MODAL_FRAC, "min_abs_r": MIN_ABS_R},
           "per_notation": report},
          open(f"validity_{MODEL.replace(':','_').replace('.','')}.json", "w"), indent=2)
