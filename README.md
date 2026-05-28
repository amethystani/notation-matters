# Notation Matters: Inconsistency of LLMs in Molecular Representations

Research code and reproducibility materials for the paper published in *Digital Discovery* (2025).

**Manuscript ID:** DD-COM-05-2026-000309  

## Overview

This repository contains all experimental scripts and analysis code for evaluating the robustness of large language models (LLMs) to molecular notation variations. The work benchmarks Qwen3.6 and ChemBERTa-2 on SMILES, IUPAC, InChI, and SELFIES representations of molecules from the ESOL water solubility dataset.

## Files

### Experimental Scripts
- `07_full_esol_phase1.py` – Phase 1: Notation consistency evaluation across LLMs
- `08_full_esol_phase2.py` – Phase 2: Ridge regression pooling of equivalent representations
- `run_full_benchmark.sh` – Batch execution script for both phases

### Article
- `royal-society-of-chemistry-article-template/main.tex` – LaTeX source (6-page Communication)
- `royal-society-of-chemistry-article-template/rsc.bib` – BibTeX bibliography

## Requirements

```bash
pip install requests pandas numpy scikit-learn
```

### External Dependencies
- **Ollama** (for local LLM inference): https://ollama.ai
- **Qwen3.6** model: `ollama pull qwen3.6`
- **ChemBERTa-2** (via Hugging Face): Pre-installed via script imports

### Data
- **ESOL Dataset**: Delaney, J. S. (2004). J. Chem. Inf. Comput. Sci., 44(3), 1000–1005. https://doi.org/10.1021/ci034243x
- Retrieved via PubChem REST API (see scripts)

## Running the Experiments

### Phase 1: Notation Consistency
```bash
python 07_full_esol_phase1.py
```
Evaluates prediction accuracy across four molecular notations (SMILES, IUPAC, InChI, SELFIES) for 1072 molecules.

### Phase 2: Ridge Regression Pooling
```bash
python 08_full_esol_phase2.py
```
Learns per-notation prediction heads and tests whether a single Ridge regressor can fuse them.

### Full Pipeline
```bash
bash run_full_benchmark.sh
```
Executes both phases sequentially with consistent random seeds.

## Reproducibility Notes

- **LLM Inference**: Scripts use Ollama for Qwen3.6. Results depend on model weights and may vary slightly across hardware/inference parameters.
- **Statistical Testing**: All p-values and test statistics are reported in the paper.
- **IUPAC Lookup**: 56 of 1128 ESOL molecules have no PubChem IUPAC name; these are skipped (final dataset: 1072 molecules).
- **SELFIES Validity**: 89.4% of SMILES convert to valid SELFIES; remaining molecules are excluded from SELFIES analysis.

## Citation

```bibtex
@article{Mishra2025,
  author    = {Mishra, Animesh},
  title     = {Notation Matters: Inconsistency of LLMs in Molecular Representations},
  journal   = {Digital Discovery},
  year      = {2025},
  volume    = {X},
  pages     = {1--6}
}
```

## Author

**Animesh Mishra**  
B.Tech Student, School of Engineering, Shiv Nadar Institution of Eminence  
Email: am847@snu.edu.in  
ORCID: [0009-0009-1770-6329](https://orcid.org/0009-0009-1770-6329)

## License

Code is provided for research and reproducibility purposes under the MIT License.

## Issues & Questions

For questions about reproducibility, please contact am847@snu.edu.in.
