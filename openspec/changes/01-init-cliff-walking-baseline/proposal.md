# Proposal: 01-init-cliff-walking-baseline

## Status: PROPOSED

## Problem Statement
HW2 requires implementing and comparing tabular Q-learning vs tabular SARSA on the Cliff Walking GridWorld environment, analyzing convergence speed, policy differences, stability, and producing a submittable report.

## Goals
1. Set up the project structure (src/, scripts/, results/, report/, tests/)
2. Implement a shared training skeleton with Q-learning and SARSA update rules separated
3. Run multi-seed experiments (20 seeds × 1000 episodes) with baseline α=0.1, γ=0.9, ε=0.1
4. Run epsilon sensitivity analysis (ε ∈ {0.01, 0.1, 0.2})
5. Produce learning curves, policy visualizations, convergence comparison, and a full report

## Scope
- Use Gymnasium CliffWalking-v0; fallback to custom 4×12 env if API issues
- Tabular methods only (no neural networks)
- Reproducible via fixed seed management
- Report in Markdown (hw2_report.md) with LaTeX math

## Out of Scope
- Deep RL methods
- Hyperparameter sweep beyond epsilon analysis
- PDF generation (nice-to-have)

## Risks
- Gymnasium API differences (old/new step() return signature)
- Windows path issues

## Decision: APPROVED → proceed to design
