# DRL HW2 — Cliff Walking: Q-Learning vs SARSA

## 概述
本專案實作並比較 **Tabular Q-Learning**（off-policy）與 **Tabular SARSA**（on-policy）在 Cliff Walking GridWorld 環境中的學習行為，包含收斂速度、策略差異、穩定性分析，以及探索參數（ε）敏感性實驗。

## 環境
- Gymnasium `CliffWalking-v0`（4×12 grid）
- 超參數：α=0.1, γ=0.9, ε=0.1
- 20 random seeds × 1000 episodes

## 快速開始

```bash
pip install -r requirements.txt
python scripts/run_experiments.py
```

結果將儲存於 `results/raw/` 與 `results/figures/`，報告見 `report/hw2_report.md`。

## 目錄結構
```
├── src/             # 核心程式碼（環境、代理、訓練、評估、繪圖）
├── scripts/         # 實驗執行腳本
├── results/         # 實驗結果（raw CSV/JSON + figures）
├── report/          # 作業報告
├── tests/           # 煙霧測試
└── requirements.txt
```

## 課程資訊
中興大學資工系深度強化學習 HW2
