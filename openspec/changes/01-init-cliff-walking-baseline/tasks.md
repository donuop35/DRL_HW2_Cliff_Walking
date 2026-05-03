# Tasks: 01-init-cliff-walking-baseline

## Status: DONE

### Phase 1 — Project Scaffolding
- [x] Write proposal.md
- [x] Write design.md
- [x] Create .gitignore
- [x] Create requirements.txt
- [x] Create README.md skeleton
- [x] Create directory structure

### Phase 2 — Implementation
- [x] src/utils.py — seed management, moving average
- [x] src/environment.py — custom env fallback (Gymnasium v0 deprecated, custom used)
- [x] src/agents.py — TabularAgent (Q-learning + SARSA)
- [x] src/train.py — training loop + run_experiment()
- [x] src/evaluate.py — metrics (convergence, AULC, final reward)
- [x] src/plot.py — all visualization functions
- [x] scripts/run_experiments.py — orchestrator

### Phase 3 — Experiments
- [x] Install dependencies (gymnasium, numpy, matplotlib, pandas, scipy)
- [x] Run baseline experiments (20 seeds x 1000 episodes x 2 algos)
- [x] Run epsilon sensitivity (3 epsilons x 10 seeds x 2 algos)
- [x] Save all raw results to results/raw/ (CSV + NPY + JSON)
- [x] Generate all figures to results/figures/ (6 figures)

### Phase 4 — Report
- [x] Write report/hw2_report.md (all 9 sections + appendix)
- [x] Embed figures in report

### Phase 5 — Tests
- [x] tests/test_smoke.py — 14/14 smoke tests PASS

### Phase 6 — Verification & Commit
- [x] Verify all outputs exist
- [x] git add + commit (f10a88b, 67 files)
- [x] git push -> https://github.com/donuop35/DRL_HW2_Cliff_Walking.git
- [x] Archive this change

## Key Results
- SARSA: final=-20.92+/-2.37, convergence=548+/-224 ep, AULC=-46.78
- Q-Learning: final=-50.37+/-6.56, never converged (MA threshold), AULC=-66.04
