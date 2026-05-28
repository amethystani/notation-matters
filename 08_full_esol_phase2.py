"""
Script 08 — FULL ESOL Phase 2: Large-scale mechanistic analysis
- All valid augmented-SMILES pairs from 1128 ESOL molecules
- ChemBERTa-2 + Ridge head trained on full ESOL
- Representation-level hidden-state replacement at all 7 positions
- Bootstrap CIs (n_boot=2000), Spearman orthogonality, Kruskal-Wallis
- Checkpoint every 100 pairs — safe to interrupt and resume
- Output: phase2_full_results.json + phase2_full_stats.json
"""

import torch, numpy as np, json, requests, csv, io, os, re, sys, time
import selfies as sf
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr, kruskal, wilcoxon
from collections import defaultdict, Counter

MODEL_ID  = "seyonec/ChemBERTa-zinc-base-v1"
ESOL_URL  = ("https://raw.githubusercontent.com/deepchem/deepchem/"
              "master/datasets/delaney-processed.csv")
OUT_DIR   = "/home/snu/workspace/notation_research"
CKPT_FILE = f"{OUT_DIR}/phase2_full_checkpoint.json"
OUT_FILE  = f"{OUT_DIR}/phase2_full_results.json"
STATS_FILE = f"{OUT_DIR}/phase2_full_stats.json"
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Device selection (prefer GPU but fall back gracefully) ───────────────────
def get_device():
    if torch.cuda.is_available():
        free_mb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e6
        if free_mb > 800:           # ChemBERTa needs ~400 MB VRAM
            return "cuda"
    return "cpu"


# ─── Molecule classification ───────────────────────────────────────────────────
def classify(s: str) -> str:
    if re.search(r'[nNsS].*\d|\d.*[nNsS]', s):            return "heterocyclic"
    if re.search(r'[a-z]', s) and re.search(r'\d', s):    return "aromatic"
    if 'C(=O)O' in s or 'C(O)=O' in s or 'OC(=O)' in s: return "carboxylic"
    if re.search(r'[FClBrI]', s):                          return "halogenated"
    if re.search(r'(?<![a-z])O(?![a-z=\(])', s):          return "alcohol"
    if re.search(r'(?<![a-z])N(?![a-z=\(+])', s):         return "amine"
    return "aliphatic"


# ─── SMILES augmentation via SELFIES token rotation ───────────────────────────
def augment_smiles(smiles: str):
    """SELFIES encode → rotate one token → decode → augmented SMILES."""
    try:
        enc    = sf.encoder(smiles)
        tokens = list(sf.split_selfies(enc))
        if len(tokens) > 2:
            rotated = tokens[1:] + [tokens[0]]
            aug = sf.decoder("".join(rotated))
            if aug and aug != smiles:
                return aug
    except Exception:
        pass
    return None


# ─── Embedding utilities ───────────────────────────────────────────────────────
def get_cls_all_layers(model, tokenizer, smiles, device):
    """Return list of CLS hidden states at each of the 7 positions (emb + 6 layers)."""
    inp = tokenizer(smiles, return_tensors="pt",
                    truncation=True, max_length=128)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return [h[0, 0, :].float().cpu().numpy()
            for h in out.hidden_states]   # list of 7 arrays shape (hidden,)


def cos_sim(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def patch_and_predict(model, tokenizer, head, src_hs, smiles_tgt,
                      layer_k, device):
    """
    Forward-pass smiles_tgt with the CLS hidden state at layer_k
    replaced by src_hs[layer_k]. Returns predicted logS from Ridge head.
    """
    patch_v = torch.tensor(src_hs[layer_k],
                           dtype=torch.float32).to(device)

    def make_hook(target_layer):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                hs, rest = output[0].clone(), output[1:]
            else:
                hs, rest = output.clone(), None
            if target_layer == layer_k:
                if hs.dim() == 3:
                    hs[0, 0, :] = patch_v
                else:
                    hs[0, :] = patch_v
            return (hs,) + rest if rest else hs
        return hook

    hooks = [
        layer.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.encoder.layer)
    ]
    inp = tokenizer(smiles_tgt, return_tensors="pt",
                    truncation=True, max_length=128)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    for h in hooks:
        h.remove()
    emb = out.hidden_states[-1][0, 0, :].float().cpu().numpy()
    return float(head.predict(emb.reshape(1, -1))[0])


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=2000, ci=95, seed=SEED):
    rng   = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(values, len(values), replace=True))
             for _ in range(n_boot)]
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, ci + (100 - ci) / 2))
    return float(np.mean(values)), lo, hi


# ─── Load ESOL ────────────────────────────────────────────────────────────────
print("=" * 72)
print("PHASE 2 — FULL ESOL: Large-scale mechanistic analysis")
print("=" * 72)

print("Downloading ESOL dataset...")
r = requests.get(ESOL_URL, timeout=30)
rows    = list(csv.DictReader(io.StringIO(r.text)))
sc      = next(c for c in rows[0] if 'smiles' in c.lower())
lc      = next(c for c in rows[0]
               if ('measured' in c.lower() or 'log' in c.lower())
               and 'smiles' not in c.lower())

all_mols = []
for row in rows:
    sm = row[sc].strip()
    try:
        ls = float(row[lc])
    except ValueError:
        continue
    all_mols.append((sm, ls, classify(sm)))
print(f"✓ Loaded {len(all_mols)} molecules")

# ─── Generate augmented SMILES pairs ──────────────────────────────────────────
print("\nGenerating augmented SMILES pairs (SELFIES token rotation)...")
all_pairs = []
fail = 0
for sm, ls, cls in all_mols:
    aug = augment_smiles(sm)
    if aug and aug != sm:
        all_pairs.append((sm, aug, ls, cls))
    else:
        fail += 1
print(f"✓ Valid pairs: {len(all_pairs)}/{len(all_mols)} "
      f"({len(all_pairs)/len(all_mols)*100:.1f}%)  "
      f"augmentation failures: {fail}")

# ─── Load ChemBERTa-2 ─────────────────────────────────────────────────────────
DEVICE = get_device()
print(f"\nLoading {MODEL_ID}  (device={DEVICE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModel.from_pretrained(MODEL_ID,
                                       output_hidden_states=True)
model     = model.to(DEVICE).eval()
N_LAYERS  = model.config.num_hidden_layers
print(f"✓ Model loaded | layers={N_LAYERS} | "
      f"hidden_size={model.config.hidden_size} | device={DEVICE}")

# ─── Train Ridge head on full ESOL ────────────────────────────────────────────
print(f"\nEmbedding all {len(all_mols)} ESOL molecules for Ridge head training...")
embeddings, logS_vals = [], []
for i, (sm, ls, _) in enumerate(all_mols):
    try:
        hs = get_cls_all_layers(model, tokenizer, sm, DEVICE)
        embeddings.append(hs[-1])
        logS_vals.append(ls)
    except Exception as e:
        pass
    if (i + 1) % 200 == 0:
        print(f"  Embedded: {i+1}/{len(all_mols)}")

X = np.array(embeddings)
y = np.array(logS_vals)
print(f"✓ Embedded {len(X)} molecules (shape {X.shape})")

head   = Ridge(alpha=1.0)
head.fit(X, y)
cv_mae = float(-cross_val_score(
    head, X, y, cv=5, scoring="neg_mean_absolute_error").mean())
print(f"✓ Ridge head trained | 5-fold CV MAE = {cv_mae:.3f} logS")

# ─── Resume from checkpoint ───────────────────────────────────────────────────
done_set = set()
results  = []
if os.path.exists(CKPT_FILE):
    with open(CKPT_FILE) as f:
        ckpt = json.load(f)
    results  = ckpt.get("molecule_results", [])
    done_set = {r["smiles_a"] for r in results}
    print(f"\n✓ Resuming: {len(results)} pairs already done")

remaining_pairs = [(a, b, ls, cls) for a, b, ls, cls in all_pairs
                   if a not in done_set]
print(f"  Remaining: {len(remaining_pairs)} pairs to process\n")

# ─── Main intervention loop ───────────────────────────────────────────────────
print(f"Running representation-level hidden-state replacement "
      f"({len(remaining_pairs)} pairs × {N_LAYERS+1} layers)...")
print("Checkpoints every 100 pairs.\n")

start_t = time.time()
total   = len(remaining_pairs)

for idx, (smi_a, smi_b, true_ls, cls) in enumerate(remaining_pairs):
    try:
        hs_a = get_cls_all_layers(model, tokenizer, smi_a, DEVICE)
        hs_b = get_cls_all_layers(model, tokenizer, smi_b, DEVICE)
    except Exception as e:
        print(f"  [{idx+1}] SKIP (embed error): {e}")
        continue

    layer_sims  = [cos_sim(hs_a[i], hs_b[i]) for i in range(len(hs_a))]
    pred_a      = float(head.predict(hs_a[-1].reshape(1, -1))[0])
    pred_b      = float(head.predict(hs_b[-1].reshape(1, -1))[0])
    gap         = abs(pred_a - pred_b)

    # Test all N_LAYERS+1 positions (embedding layer 0 + transformer layers 1-6)
    layer_imps  = {}
    best_k, best_imp = -1, -999.0
    for k in range(N_LAYERS + 1):
        try:
            pp  = patch_and_predict(model, tokenizer, head,
                                    hs_a, smi_b, k, DEVICE)
            imp = gap - abs(pred_a - pp)
            layer_imps[k] = round(float(imp), 5)
            if imp > best_imp:
                best_k, best_imp = k, imp
        except Exception:
            layer_imps[k] = None

    div_layer = next((i for i, s in enumerate(layer_sims)
                      if s < 0.85), None)

    results.append({
        "smiles_a":       smi_a,
        "smiles_b":       smi_b,
        "class":          cls,
        "true_logS":      round(true_ls, 4),
        "pred_a":         round(pred_a,  4),
        "pred_b":         round(pred_b,  4),
        "gap_unpatched":  round(gap,     4),
        "best_layer":     best_k,
        "best_improvement": round(best_imp, 4),
        "layer_improvements": layer_imps,
        "layer_cosine_sims": [round(s, 4) for s in layer_sims],
        "final_cosine_sim":  round(layer_sims[-1], 4),
        "divergence_layer":  div_layer,
    })

    # Checkpoint every 100 pairs
    if (idx + 1) % 100 == 0:
        with open(CKPT_FILE, "w") as f:
            json.dump({"molecule_results": results}, f)
        elapsed = time.time() - start_t
        eta_h   = elapsed / (idx + 1) * (total - idx - 1) / 3600
        print(f"  [CKPT {idx+1:5d}/{total}] "
              f"gap={gap:.3f}  best_L={best_k}  imp={best_imp:.3f}  "
              f"ETA {eta_h:.2f}h")
    elif (idx + 1) % 10 == 0:
        print(f"  [{idx+1:5d}/{total}] [{cls:14s}] "
              f"gap={gap:.3f}  best_L={best_k}  imp={best_imp:.3f}")

# ─── Final statistics ──────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"FINAL MECHANISTIC STATISTICS  (N = {len(results)} pairs)")
print("=" * 72)

gaps  = np.array([r["gap_unpatched"]    for r in results])
imps  = np.array([r["best_improvement"] for r in results])
fins  = np.array([r["final_cosine_sim"] for r in results])

n_improved  = int((imps > 0).sum())
n_degraded  = int((imps < 0).sum())
n_neutral   = int((imps == 0).sum())

gap_m, gap_lo, gap_hi  = bootstrap_ci(gaps)
imp_m, imp_lo, imp_hi  = bootstrap_ci(imps)
fin_m, fin_lo, fin_hi  = bootstrap_ci(fins)

# Wilcoxon signed-rank test (one-sided: improvement > 0)
wx_stat, wx_p = wilcoxon(imps, alternative='greater')

# Layer frequency (among improved pairs)
best_layers  = [r["best_layer"] for r in results if r["best_improvement"] > 0]
layer_counts = Counter(best_layers)

print(f"\n  N pairs:                    {len(results)}")
print(f"  Head 5-fold CV MAE:         {cv_mae:.3f} logS")
print(f"\n  Avg prediction gap          "
      f"{gap_m:.3f}  95% CI [{gap_lo:.3f},{gap_hi:.3f}]  "
      f"median={float(np.median(gaps)):.3f}")
print(f"  Avg best-layer improvement  "
      f"{imp_m:.3f}  95% CI [{imp_lo:.3f},{imp_hi:.3f}]  "
      f"median={float(np.median(imps)):.3f}")
print(f"  Avg final cosine similarity "
      f"{fin_m:.4f}  95% CI [{fin_lo:.4f},{fin_hi:.4f}]")
print(f"\n  Improved  (imp > 0):  {n_improved}/{len(results)} "
      f"({n_improved/len(results)*100:.1f}%)")
print(f"  Degraded  (imp < 0):  {n_degraded}/{len(results)} "
      f"({n_degraded/len(results)*100:.1f}%)")
print(f"  Neutral   (imp = 0):  {n_neutral}/{len(results)}")
print(f"\n  Wilcoxon signed-rank (imp > 0): "
      f"stat={wx_stat:.1f}  p={wx_p:.4e}")

print("\n  Layer effectiveness (best layer frequency, improved pairs only):")
for k in sorted(layer_counts.keys()):
    pct = layer_counts[k] / len(best_layers) * 100
    bar = "█" * min(40, layer_counts[k])
    print(f"    L{k}:  {layer_counts[k]:5d} ({pct:5.1f}%)  {bar}")

# By-class breakdown
by_cls = defaultdict(list)
for r in results:
    by_cls[r["class"]].append(r)

print(f"\n  Class breakdown:")
print(f"  {'Class':<16} {'n':>5}  {'avg_gap':>8}  {'avg_imp':>8}  "
      f"{'pct_impr':>9}  {'mode_L':>7}")

class_stats = {}
for cls in sorted(by_cls,
                   key=lambda c: -np.mean([r["gap_unpatched"]
                                           for r in by_cls[c]])):
    cr  = by_cls[cls]
    ag  = float(np.mean([r["gap_unpatched"]    for r in cr]))
    ai  = float(np.mean([r["best_improvement"] for r in cr]))
    ni  = sum(1 for r in cr if r["best_improvement"] > 0)
    bl  = [r["best_layer"] for r in cr if r["best_improvement"] > 0]
    ml  = int(max(set(bl), key=bl.count)) if bl else -1
    class_stats[cls] = {
        "n": len(cr), "avg_gap": ag, "avg_improvement": ai,
        "n_improved": ni, "pct_improved": ni/len(cr)*100,
        "mode_layer": ml,
    }
    print(f"  {cls:<16} {len(cr):>5}  {ag:>8.3f}  {ai:>8.3f}  "
          f"  {ni}/{len(cr)} ({ni/len(cr)*100:.0f}%)  L{ml}")

# Spearman consistency–accuracy orthogonality
gaps_s = np.array([r["gap_unpatched"] for r in results])
maes_s = np.array([abs(r["pred_a"] - r["true_logS"]) for r in results])
rho, pval_sp = spearmanr(gaps_s, maes_s)
print(f"\n  Consistency–Accuracy Spearman ρ = {rho:.4f}  p = {pval_sp:.4f}")
print("  → " + ("ORTHOGONAL (p > 0.05)" if pval_sp > 0.05
                 else f"Correlated (ρ={rho:.3f})"))

# Kruskal-Wallis on gap across classes
cls_gap_groups = {c: np.array([r["gap_unpatched"] for r in by_cls[c]])
                  for c in by_cls if len(by_cls[c]) >= 5}
if len(cls_gap_groups) >= 3:
    H_kw, p_kw = kruskal(*cls_gap_groups.values())
    print(f"\n  Kruskal-Wallis (gap across classes): H={H_kw:.3f}  p={p_kw:.4f}")
else:
    H_kw = p_kw = None

# ─── Save results ──────────────────────────────────────────────────────────────
stats_out = {
    "n_pairs":       len(results),
    "cv_mae_head":   cv_mae,
    "n_improved":    n_improved,
    "n_degraded":    n_degraded,
    "pct_improved":  n_improved / len(results) * 100,
    "avg_gap":       gap_m,
    "gap_ci_95":     [gap_lo, gap_hi],
    "median_gap":    float(np.median(gaps)),
    "avg_improvement":    imp_m,
    "imp_ci_95":     [imp_lo, imp_hi],
    "median_improvement": float(np.median(imps)),
    "avg_final_cosim":    fin_m,
    "cosim_ci_95":   [fin_lo, fin_hi],
    "wilcoxon_stat": float(wx_stat),
    "wilcoxon_pval": float(wx_p),
    "layer_frequency": {str(k): int(v)
                        for k, v in layer_counts.most_common()},
    "spearman_rho":  float(rho),
    "spearman_pval": float(pval_sp),
    "kruskal_wallis": {"H": float(H_kw), "pval": float(p_kw)}
                       if H_kw is not None else None,
    "class_stats":   class_stats,
    "molecule_results": results,
}

with open(CKPT_FILE, "w") as f:
    json.dump(stats_out, f)
with open(OUT_FILE, "w") as f:
    json.dump(stats_out, f, indent=2)

# Compact stats (no per-molecule data)
stats_compact = {k: v for k, v in stats_out.items()
                 if k != "molecule_results"}
with open(STATS_FILE, "w") as f:
    json.dump(stats_compact, f, indent=2)

print(f"\n✓ Results:    {OUT_FILE}")
print(f"✓ Statistics: {STATS_FILE}")
print(f"✓ Total pairs processed: {len(results)}")
