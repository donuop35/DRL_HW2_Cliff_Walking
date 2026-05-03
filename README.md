# 深度強化學習：Cliff Walking 實驗分析
> 比較 Tabular Q-Learning (Off-policy) 與 SARSA (On-policy)

![Professor Style Plot](results/figures/07_professor_style.png)

## 📊 專案概述
本儲存庫提供了在經典的 **Cliff Walking GridWorld** 環境中，對 **Q-Learning** 與 **SARSA** 演算法的完整實作與比較分析。

我們探討了這兩種演算法在收斂速度、學習穩定性以及最終策略（Policy）品質上的根本差異，特別著重於它們如何處理危險狀態附近的探索風險（ε-greedy）。

## 🚀 關鍵發現
| 指標 | Q-Learning (Off-policy) | SARSA (On-policy) |
| :--- | :--- | :--- |
| **策略行為** | 冒險型 (緊貼懸崖邊緣) | 保守型 (安全路徑) |
| **收斂特性** | 較慢 / 穩定性較低 | **較快 / 高度穩定** |
| **探索風險** | 高 (懲罰值 -100) | 低 (安全導航) |
| **ε=0 時的最佳性** | 是 (理論最短路徑) | 否 (次優路徑) |

## 🎨 視覺化結果

### 1. 策略比較
觀察 SARSA 如何學習遠離懸崖的「安全」路徑，而 Q-Learning 則堅持選擇風險較高的最短路徑。

| Q-Learning (冒險邊緣) | SARSA (安全路線) |
| :---: | :---: |
| ![Q-Learning Policy](results/figures/04_policy_qlearning.png) | ![SARSA Policy](results/figures/04_policy_sarsa.png) |

### 2. 學習穩定性與收斂
由於 SARSA 具有「在線探索覺知（Online Exploration Awareness）」的特性，其表現出顯著較低的變異數與更快的收斂速度。

| 累積獎勵曲線 (Learning Curves) | 穩定性分析 (Stability) |
| :---: | :---: |
| ![Learning Curves](results/figures/01_learning_curves.png) | ![Stability](results/figures/06_stability.png) |

## 🛠️ 開始使用

### 環境需求
- Python 3.8+
- Gymnasium
- NumPy
- Matplotlib

### 安裝步驟
```bash
git clone https://github.com/donuop35/DRL_HW2_Cliff_Walking.git
cd DRL_HW2_Cliff_Walking
pip install -r requirements.txt
```

### 執行實驗
執行主程式以重新產生所有結果與圖表：
```bash
python scripts/run_experiments.py
python scripts/run_professor_plot.py
```

## 📂 目錄結構
```text
.
├── src/                # 核心 RL 實作 (Env, Agent, Train, Plot)
├── scripts/            # 實驗執行腳本
├── results/            # 原始數據與生成的影像
├── report/             # 完整學術報告 (PDF/MD)
├── tests/              # 系統煙霧測試 (Smoke Tests)
└── requirements.txt
```

---
*專為深度強化學習 (DRL) 分析開發。所有結果皆可透過固定的隨機種子（Random Seeds）重複驗證。*
