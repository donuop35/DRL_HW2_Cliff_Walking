# Tasks: 01-init-cliff-walking-baseline

## Status: IN_PROGRESS

### Phase 1 — Project Scaffolding
- [x] Write proposal.md
- [x] Write design.md
- [ ] Create .gitignore
- [ ] Create requirements.txt
- [ ] Create README.md skeleton
- [ ] Create directory structure

### Phase 2 — Implementation
- [ ] src/utils.py — seed management, moving average
- [ ] src/environment.py — custom env fallback
- [ ] src/agents.py — TabularAgent (Q-learning + SARSA)
- [ ] src/train.py — training loop + run_experiment()
- [ ] src/evaluate.py — metrics (convergence, AULC, final reward)
- [ ] src/plot.py — all visualization functions
- [ ] scripts/run_experiments.py — orchestrator

### Phase 3 — Experiments
- [ ] Install dependencies (gymnasium, numpy, matplotlib, pandas)
- [ ] Run baseline experiments (20 seeds × 1000 episodes × 2 algos)
- [ ] Run epsilon sensitivity (3 epsilons × 20 seeds × 2 algos)
- [ ] Save all raw results to results/raw/
- [ ] Generate all figures to results/figures/

### Phase 4 — Report
- [ ] Write report/hw2_report.md (all 9 sections)
- [ ] Embed figures in report

### Phase 5 — Tests
- [ ] tests/test_smoke.py — smoke tests for env, agent, training

### Phase 6 — Verification & Commit
- [ ] Verify all outputs exist
- [ ] git add + commit + push
- [ ] Archive this change
