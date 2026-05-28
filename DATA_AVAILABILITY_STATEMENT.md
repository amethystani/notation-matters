# Data Availability Statement
## Manuscript: DD-COM-05-2026-000309
## "Notation Matters: Inconsistency of LLMs in Molecular Representations"

**Date Prepared:** 28 May 2026  
**Corresponding Author:** Animesh Mishra (am847@snu.edu.in)

---

## Public Repository

All experimental code, documentation, and article source are publicly available without embargo at:

**GitHub:** https://github.com/amethystani/notation-matters  
**Accessibility:** Public (no authentication required)  
**License:** MIT

---

## Available Artifacts

### Experimental Scripts
1. **07_full_esol_phase1.py** (17.2 KB)
   - Phase 1: Notation consistency evaluation
   - Input: ESOL dataset (1072 molecules with 4 notations each)
   - Output: Prediction logs, accuracy metrics per notation
   - Runtime: ~2-4 hours on CPU

2. **08_full_esol_phase2.py** (17.6 KB)
   - Phase 2: Ridge regression pooling analysis
   - Input: Phase 1 predictions
   - Output: Ridge coefficients, cross-fold validation metrics
   - Runtime: ~30 minutes

3. **run_full_benchmark.sh** (2.4 KB)
   - Batch execution script for reproducibility
   - Runs both phases with consistent random seeds

### Documentation
- **README.md** — Installation, usage, reproducibility guide
- **main.tex** — Complete 6-page article source
- **rsc.bib** — Bibliography with verified DOIs

### Source Data
- **ESOL Dataset**: Retrieved dynamically via PubChem REST API
  - Citation: Delaney, J. S. (2004). J. Chem. Inf. Comput. Sci., 44(3), 1000–1005
  - DOI: https://doi.org/10.1021/ci034243x

---

## Reproducibility Instructions

```bash
# Install requirements
pip install requests pandas numpy scikit-learn

# Install Ollama and Qwen3.6
# https://ollama.ai
ollama pull qwen3.6

# Run full pipeline
bash run_full_benchmark.sh
```

All results are reproducible from provided scripts. Raw execution logs are generated at runtime.

---

## Accessibility & Retention

| Item | Status |
|------|--------|
| Public Access | ✓ Yes (no embargo) |
| Authentication Required | ✗ No |
| Usage Restrictions | ✗ None (MIT License) |
| Permanent Availability | ✓ GitHub + PubChem |

---

## Contact

**Animesh Mishra**  
am847@snu.edu.in  
ORCID: 0009-0009-1770-6329

*Certified: All data and code are publicly available as of 28 May 2026.*
