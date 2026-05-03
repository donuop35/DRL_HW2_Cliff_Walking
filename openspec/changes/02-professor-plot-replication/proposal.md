# Proposal: 02-professor-plot-replication

## Status: PROPOSED → APPROVED

## Problem Statement
教授展示的參考圖有不同的超參數設定（α=0.5, 50 runs, 500 episodes），
且同時展示「自己實驗的實線」與「Sutton & Barto 教科書參考虛線」。
目前 baseline 實驗（α=0.1, 20 seeds）無法重現此圖樣式。

## Goals
1. 以 α=0.5, ε=0.1, γ=1.0（Sutton 原始設定用 γ=1），50 seeds × 500 episodes 重新跑實驗
2. 建立 `plot_professor_style()` 函數，精確重現教授圖的視覺風格
3. 加入 Sutton & Barto 教科書數位化參考曲線作為虛線
4. 更新報告嵌入新圖

## Spec Delta
- gamma: 原 0.9 → 新實驗另外用 1.0（Sutton 原設定）；baseline 0.9 保留
- alpha: 新增 alpha=0.5 實驗組
- seeds: 新增 50 seeds 版本
- episodes: 500（與教授圖一致）

## Sutton Reference Curves
來自 Sutton & Barto RL 2nd Ed. Example 6.6 / Figure 6.4 的數位化近似值，
以 hardcoded 陣列形式嵌入 plot 模組，標記為 "Sarsa, Sutton Pub." 和 "Q-learning, Sutton Pub."
