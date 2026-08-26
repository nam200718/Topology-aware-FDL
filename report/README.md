# UROP Final Report — HEP

## Contents
- 
eport/main.tex — LaTeX source (IEEEtran, technical). Compile with:
  `
  cd report
  latexmk -pdf main.tex
  # or
  pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex
  `
- 
eport/references.bib — BibTeX
- 
eport/figures/ — 8 figures (hardware_efficiency_profile.png, convergence_*.png, byzantine_label_flip.png, pareto_accuracy_cost.png, etc.)
- paper/main.tex — identical content kept for archival (same as report, with UROP title page)

## Before submitting (deadline Aug 28)
1. Edit 
eport/main.tex:12-20 — replace [Your Full Name], [Your Institution], [Supervisor Name], your.email@..., https://github.com/[your-username]/Topology-aware-FDL
2. Replace 
eport/references.bib placeholder if needed (current 25 entries from HEP)
3. Set Acknowledgements GitHub URL or note ZIP submission
4. Compile to PDF and check all Figure 1-4 / Table I-VIII captions render

## Grading mapping (UROP 100%)
- Writing/presentation 20% → IEEEtran, numbered captions, microtype, booktabs
- Creativity 20% → anchored binomial-style weighting, S-AFR, variance-adaptive β_{c,k}
- Implementation 30% → src/core/hierarchical_ensemble_engine.py, updater.py, loss.py, DirectML+CPU fallback
- Experimental eval 30% → Tables I-VIII, 5 heterogeneity regimes, 50-client scale, Byzantine matrix (71.58% vs 31.31% at f=40%)

## Source code
Submit either:
- ZIP of this repo (src/, configs/, scripts/, 	ests/), or
- GitHub link: https://github.com/[your-username]/Topology-aware-FDL

All numbers in report match outputs/byzantine_matrix_20260826_120000/matrix_results.csv and paper/figures/byzantine_label_flip.png (71.58% HEP, 62.87% Ditto at f=40%).
