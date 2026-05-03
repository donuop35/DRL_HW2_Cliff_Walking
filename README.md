# Deep Reinforcement Learning: Cliff Walking Analysis
> Comparison between Tabular Q-Learning (Off-policy) and SARSA (On-policy)

![Professor Style Plot](results/figures/07_professor_style.png)

## 📊 Project Overview
This repository provides a comprehensive implementation and comparative analysis of **Q-Learning** and **SARSA** algorithms within the classic **Cliff Walking GridWorld** environment. 

We explore the fundamental differences in convergence speed, learning stability, and final policy quality, particularly focusing on how these algorithms handle exploration risk (ε-greedy) near hazardous states.

## 🚀 Key Findings
| Metric | Q-Learning (Off-policy) | SARSA (On-policy) |
| :--- | :--- | :--- |
| **Strategy** | Aggressive (Cliff Edge) | Conservative (Safe Path) |
| **Convergence** | Slower / Less Stable | **Faster / Highly Stable** |
| **Exploration Risk** | High (Penalty -100) | Low (Safe Navigation) |
| **Optimal for ε=0** | Yes (Theoretical Shortest) | No (Suboptimal Path) |

## 🎨 Visualized Results

### 1. Policy Comparison
Observe how SARSA learns a "safe" path away from the cliff, while Q-Learning insists on the risky shortest path.

| Q-Learning (Risky Edge) | SARSA (Safe Route) |
| :---: | :---: |
| ![Q-Learning Policy](results/figures/04_policy_qlearning.png) | ![SARSA Policy](results/figures/04_policy_sarsa.png) |

### 2. Learning Stability & Convergence
SARSA exhibits significantly lower variance and faster convergence due to its online exploration awareness.

| Cumulative Rewards | Stability Analysis |
| :---: | :---: |
| ![Learning Curves](results/figures/01_learning_curves.png) | ![Stability](results/figures/06_stability.png) |

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- Gymnasium
- NumPy
- Matplotlib

### Installation
```bash
git clone https://github.com/donuop35/DRL_HW2_Cliff_Walking.git
cd DRL_HW2_Cliff_Walking
pip install -r requirements.txt
```

### Run Experiments
Execute the main orchestrator to reproduce all results and figures:
```bash
python scripts/run_experiments.py
python scripts/run_professor_plot.py
```

## 📂 Directory Structure
```text
.
├── src/                # Core RL implementation (Env, Agent, Train, Plot)
├── scripts/            # Experiment orchestrators
├── results/            # Raw data and generated high-res figures
├── report/             # Comprehensive academic reports (PDF/MD)
├── tests/              # System smoke tests
└── requirements.txt
```

---
*Developed for DRL Analysis. All results are reproducible via fixed random seeds.*
