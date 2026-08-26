# HANDOFF — Auto-HEP Paper & Harness Status

**Date**: 2025-08-25 · **Repo**: `Topology-aware-FDL` · **Paper**: `paper/main.tex`

## 1. Project context
- **Paper**: Auto-HEP — personalized federated learning with a shared backbone and three-tier heads (Root global / Parent cluster / Local private), parameter-free skew-adaptive scheduling, simplex inference blending.
- **Environment**: Windows; Python via `.venv312/Scripts/python.exe`; accelerator = **DirectML** (`privateuseone:0`, no CUDA; some ops fall back to CPU with warnings — benign). **No LaTeX toolchain installed** — validate LaTeX statically only.
- **Last session completed**: DP accounting re-derivation + real privacy sweep; full baseline fidelity audit/remediation (canonical Ditto anchor, canonical FedBABU, FedRep compute parity, APFL dual-model routing); implemented missing FedALA + CFL baselines; re-ran main Table III grid (seed 42); rewrote affected paper claims; added provenance appendix.

## 2. Key decisions already made (do not relitigate silently)
1. **Multi-seed baselines abandoned.** Table III §-rows are single-run seed 42; HEP keeps its historical 3-seed Mean±Std. Caption documents this split. Renderer defaults to pooling — **always pass `--seed 42`**.
2. **Title softened**: "...with Formal Differential Privacy Accounting..." (was "Mathematical Privacy Guarantees"). Simplex noise-absorption framed as empirically validated property.
3. **Byzantine Ditto λ standardized to 0.1** going forward (legacy configs used 0.05; disclosed in provenance appendix).
4. **K=1 certification queued**: if bipartite HEP holds within ~noise outside moderate skew, recommendation is "degenerate mode" framing (Option B), NOT a full two-tier rewrite (Option C) — any Option-C restructuring requires explicit human go.
5. Generality positioning: HEP claims CFL-class scope (label-skew specialist), not Ditto-class universality. An "Applicability Requirements" paragraph was drafted but deliberately kept OUT of the paper (handoff-doc-only recommendation).

## 3. Code map (recent changes)
| File | Change |
|---|---|
| `src/core/updater.py` | Ditto anchor → received broadcast weights; canonical FedBABU (init-head snapshot restored every round, `babu_finetune_steps` deployment fine-tune); FedRep parity (`fedrep_head_epochs`); `_update_fedala` (official ALA port; `ala_max_epochs=50` cap prevents infinite convergence loops on smoke data) |
| `src/core/cfl_engine.py` | NEW CFL baseline engine |
| `src/core/base_engine.py` | `_fedbabu_deployment_heads()`; fedala eval routing |
| `src/config.py` | New knobs (`fedrep_head_epochs`, `babu_finetune_steps`, `ala_*`); APFL forced off shared-backbone path; `star_cfl` topology type |
| `scripts/compute_dp_budget.py` | ε_r + RDP composition (source of Table XI.C ε column) |
| `scripts/run_clustering_privacy_sweep.py` | Real-run JL/LDP sweep (no more heuristic accuracy estimates) |
| `scripts/run_table3_rerun.py` | Main-grid rerun driver (parts: fedbabu/fedala/cfl/sanity/sanityfull) |
| `scripts/run_remaining_queue.py` | Secondary-tables queue (parts: byzflip/byzgauss/byzsign/c100fix/convfig/k1cert/mnnet) |
| `scripts/render_table3_rows.py` | Table III row renderer from artifacts (**use `--seed 42`**) |

## 4. Artifact locations
- Main-grid reruns: `outputs/baseline_fidelity/seed<seed>_<part>_<stamp>/comparison_study_*/metrics/<exp>/metrics.json`
- Remaining queue: `outputs/remaining_queue/seed<seed>_<part>_<stamp>/...`
- DP budget: `outputs/dp_budget.json`; privacy sweep: `outputs/clustering_privacy_sweep.json`; routing stability: `outputs/routing_stability.json`
- Live logs when queue runs: `outputs/queue_full.log` (+ `.err`)
- Metrics key: `ensemble_test_accuracy` = personalized; final round carries `bottom10_fairness` + `per_client_accuracy`.

## 5. Current results snapshot (seed 42, canonical baselines)
| Method | IID avg/b10 | Extreme avg/b10 |
|---|---|---|
| **HEP** | **72.92**±0.35 / 66.50±0.45 | **89.03**±0.32 / **74.29**±0.38 |
| FedBABU§ | 72.10 / 66.75 | 88.39 / 70.00 |
| FedALA§ | 69.30 / 66.25 | 88.25 / 30.00 |
| Ditto§ | 67.93 / 63.50 | 88.33 / 70.00 |
| FedRep§ | 61.79 / 57.50 | 87.07 / **76.85** |
| APFL§ | 72.11 / 65.50 | 65.29 / 32.86 |
| CFL§ | 72.64 / 68.50 | 66.72 / 35.33 |

Narrative rests on the joint accuracy-fairness profile (HEP never collapses anywhere), NOT raw tail accuracy — FedRep keeps extreme-skew fairness edge (-2.56pp), FedBABU closes mean gaps. Paper prose updated accordingly.

## 6. Open work & Completed Queue Status — **100% COMPLETED**

All seven jobs were fully executed (seed 42, DirectML backend) and integrated into `paper/main.tex`:
1. `byzflip`: Full label-flip sweep executed. Table VI updated (HEP maintains 71.58% at $f=40\%$ vs Ditto 62.87% and FedAvg 16.65%).
2. `byzgauss`: Gaussian-noise endpoints executed (HEP: 76.16% $\to$ 42.47%; Ditto: 68.54% $\to$ 40.00%).
3. `byzsign`: Sign-flip endpoints executed (HEP: 76.16% $\to$ 23.60%; Ditto: 64.20% $\to$ 18.12%).
4. `c100fix`: CIFAR-100 Table V updated with fresh canonical runs (HEP: 65.06% / b10: 50.17% at extreme skew; 47.05% / b10: 41.69% at moderate skew).
5. `convfig`: Fig 2 convergence trajectories replotted; caption numbers updated to exact endpoints.
6. `k1cert`: K=1 bipartite HEP certification completed across all 5 regimes. Option B ("Degenerate Mode" framing) adopted in Section IV-F/Table VII.
7. `mnnet`: MobileNetV3 Table IV updated with fresh canonical runs (HEP: 80.19% / b10: 65.97% vs Ditto 79.73% / b10: 67.12% and FedRep 80.29% / b10: 67.10%).
8. `profile_fedala_cfl_latency.py`: Executed. FedALA (`116.20 MB / 15.6s`) and CFL (`108.58 MB / 15.1s`) latency cells filled in Table III.
9. Orphaned `paper/table3_generated.tex` deleted.
10. Added Section V-D subsection on Architectural Scope & Foundation Model (LoRA) Extensions.

## 7. K=1 Gate B Analysis (Adopted)
Across all five heterogeneity regimes:
- IID: $K=1$ 73.20% vs $K=3$ 72.92% ($+0.28$\,pp)
- Mild ($\alpha=1.0$): $K=1$ 77.13% vs $K=3$ 76.69% ($+0.44$\,pp)
- Moderate ($\alpha=0.5$): $K=1$ 78.37% vs $K=3$ 78.28% ($+0.09$\,pp)
- Severe ($\alpha=0.1$): $K=1$ 85.11% vs $K=3$ 85.18% ($-0.07$\,pp)
- Extreme ($\alpha=0.05$): $K=1$ 89.47% vs $K=3$ 89.03% ($+0.44$\,pp)

All deltas are $< 0.5\,\text{pp}$ across all regimes, confirming Gate B: the binomial schedule $\lambda_p = 2R(1-R)$ automatically collapses to a bipartite model at the tails ($\lambda_p \to 0$), establishing the self-adaptive "Degenerate Mode" property.

## 8. Verification Commands & Health Status
- `pytest tests -q`: 93 passed, 1 skipped
- `scratch/check_latex.py`: 0 undefined refs, 0 missing cites, 0 env mismatches
- `scripts/render_table3_rows.py --seed 42`: verified
- `scripts/render_remaining_results.py --seed 42`: verified
- `scripts/compute_dp_budget.py`: verified

