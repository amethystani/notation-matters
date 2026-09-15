# Notation Matters

**Cross-representation inconsistency in chemistry language models and its mechanistic origins**

*Published in [Digital Discovery](https://pubs.rsc.org/en/journals/journalissues/dd) (Royal Society of Chemistry)*  
Animesh Mishra · Shiv Nadar Institution of Eminence

---

## Overview

Large language models predict the same molecular property differently depending on how the molecule is written. On the ESOL water-solubility benchmark, **88% of molecules** receive predictions that span more than 0.5 log S units across four standard notations (SMILES, IUPAC, InChI, SELFIES) — without any change to the underlying chemistry. This inconsistency is invisible to standard accuracy metrics and survives reformulation as a binary classification task, where **28% of molecules** still receive a different qualitative answer based solely on notation.

A descriptor-based RandomForest baseline on the same molecules shows a mean cross-notation spread 16× smaller, confirming this is a property of string-consuming models, not molecular machine learning in general.

---

## Repository Structure

```
notation-matters/
├── src/                        experimental scripts
│   ├── notation_benchmark.py       Phase 1: 4-notation LLM benchmark on ESOL
│   ├── mechanistic_analysis.py     Phase 2: ChemBERTa-2 hidden-state analysis
│   ├── reclassify_results.py       corrected structural classification
│   ├── second_model_benchmark.py   benchmark on a second open LLM
│   ├── embedding_bridge.py         cross-notation representation divergence
│   ├── model_validity_probe.py     screen models for degenerate output
│   ├── build_notation_cache.py     cache PubChem notation lookups
│   └── multidataset_benchmark.py   FreeSolv / Lipophilicity / BBBP / BACE
│
├── results/                    derived data (no re-inference needed)
│   ├── phase1_full_results.json    per-molecule predictions, all 4 notations
│   ├── phase1_full_stats.json      summary statistics for Table 1
│   ├── phase1_reclassified_stats.json  corrected structural-class analysis
│   └── extra_datasets_manifest.json    multi-dataset benchmark metadata
│
└── paper/                      manuscript source
    ├── main.tex / main.pdf         final manuscript
    ├── supplementary.tex / .pdf    supplementary information
    ├── rsc.bib                     bibliography
    ├── figures/                    figure source and high-res exports
    └── head_foot/                  RSC journal template assets
```

---

## Requirements

```bash
pip install -r requirements.txt
```

External dependencies:

| Dependency | Purpose | Install |
|---|---|---|
| [Ollama](https://ollama.ai) | Local LLM inference | See ollama.ai |
| Qwen3.6 weights | Primary model | `ollama pull qwen3.6` |
| ChemBERTa-2 | Mechanistic analysis | Auto-downloaded via HuggingFace |

---

## Running the Experiments

### Step 0 — Build the notation cache (do this once)
Fetches IUPAC names and InChI strings from PubChem with rate-limit-safe backoff.
```bash
python src/build_notation_cache.py
```

### Phase 1 — Cross-notation LLM benchmark
Evaluates Qwen3.6 on 1072 ESOL molecules × 4 notations. Checkpoints every 25 molecules; safe to interrupt and resume.
```bash
python src/notation_benchmark.py
```

### Phase 1 (post-hoc) — Corrected structural classification
Re-labels existing Phase 1 results with the fixed halogen classifier; does not re-run inference.
```bash
python src/reclassify_results.py results/phase1_full_results.json
```

### Phase 2 — Mechanistic analysis (ChemBERTa-2)
Hidden-state extraction and layerwise cosine divergence on augmented SMILES pairs.
```bash
python src/mechanistic_analysis.py
```

### Validation — Model validity probe
Run before benchmarking any new model to screen for degenerate constant-output behaviour.
```bash
python src/model_validity_probe.py <model_tag>
```

### Multi-dataset benchmark
Tests whether the effect replicates on FreeSolv, Lipophilicity, BBBP, or BACE.
```bash
python src/multidataset_benchmark.py <model> <dataset> [n_molecules]
# dataset: freesolv | lipophilicity | bbbp | bace
```

### Full pipeline (Phase 1 only)
```bash
bash run_benchmark.sh
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Molecules with spread > 0.5 log S | **88.0%** (unmatched), **92.5%** (matched) |
| Mean 4-notation spread | 1.686 log S |
| Mean spread, descriptor RandomForest baseline | 0.105 log S |
| Molecules with different binary solubility call | **27.8%** |
| Effect replication on FreeSolv | **81.9%** exceeding 0.5-unit spread |
| Mean final-layer cosine similarity (SMILES pairs) | 0.660 |
| Spearman ρ, representation divergence vs. prediction gap | −0.371 |

Effect persists from 0.1 to 2.0 log S thresholds (94.5% → 26.3%) and under a standard-deviation metric (Spearman ρ = 0.987 vs. range).

---

## Data

The ESOL dataset is retrieved at runtime from:
> Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. *J. Chem. Inf. Comput. Sci.*, 44(3), 1000–1005. https://doi.org/10.1021/ci034243x

Derived result files (no re-inference required to reproduce tables and figures) are in `results/`.

---

## Reproducibility Notes

- LLM responses depend on model weights and hardware; minor variations in predictions are expected
- 1072 of 1128 ESOL molecules have a PubChem IUPAC name; the rest are excluded
- 89.4% of SMILES convert to valid SELFIES; the rest are excluded from SELFIES analysis
- All random seeds are fixed in the scripts
- The validity probe (`src/model_validity_probe.py`) should be run on any new model before benchmarking

---

## Citation

```bibtex
@article{Mishra2026,
  author  = {Mishra, Animesh},
  title   = {Notation matters: cross-representation inconsistency in chemistry
             language models and its mechanistic origins},
  journal = {Digital Discovery},
  year    = {2026},
  doi     = {10.1039/DD-COM-05-2026-000309}
}
```

---

## License

MIT — see [LICENSE](LICENSE)

## Contact

Animesh Mishra · [am847@snu.edu.in](mailto:am847@snu.edu.in) · ORCID [0009-0009-1770-6329](https://orcid.org/0009-0009-1770-6329)
