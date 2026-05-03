# Design: 01-init-cliff-walking-baseline

## Status: APPROVED

## Architecture

### Environment
- Primary: `gymnasium.make("CliffWalking-v0")`
  - 4×12 grid, states 0-47, actions {0=Up,1=Right,2=Down,3=Left}
  - Reward: -1/step, -100 cliff fall (reset to S), episode ends at Goal
  - Handle new gymnasium API: `obs, info = env.reset()`, `obs, reward, terminated, truncated, info = env.step(a)`
- Fallback: `src/environment.py` — custom MinimalCliffWalking if gym unavailable

### Agent (src/agents.py)
```
class TabularAgent:
    __init__(alpha, gamma, epsilon, n_states, n_actions, seed)
    select_action(state) -> epsilon-greedy
    update(s, a, r, s_next, a_next=None, done=False)  # a_next=None → Q-learning; provided → SARSA
    get_greedy_policy() -> array[n_states]
    reset_qtable()
```

### Training (src/train.py)
```
train_agent(env_fn, agent, n_episodes, seed) -> episode_rewards[]
run_experiment(algo, alpha, gamma, epsilon, n_episodes, seeds) -> results dict
```

### Evaluation & Plots (src/evaluate.py, src/plot.py)
- Moving average (window=50)
- Final 100-episode mean ± std across seeds
- Convergence metric: first episode where MA(50) ≥ threshold (-20) sustained for 50 eps
- Area Under Learning Curve (AULC) for first 500 eps
- Policy grid visualization with arrows + path highlight
- Epsilon sensitivity: bar chart of final reward across algos × epsilons

### Data Flow
```
scripts/run_experiments.py
  → train_agent() × (2 algos × 20 seeds)
  → results/raw/rewards_{algo}_{seed}.csv
  → results/raw/summary.json
  → src/plot.py → results/figures/*.png
  → report/hw2_report.md (auto-generated sections + manual review)
```

## File Structure
```
DRL_HW2_Cliff_Walking/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── environment.py   # custom env fallback
│   ├── agents.py        # TabularAgent
│   ├── train.py         # training loop
│   ├── evaluate.py      # metrics
│   ├── plot.py          # visualization
│   └── utils.py         # seed management, helpers
├── scripts/
│   └── run_experiments.py
├── results/
│   ├── raw/
│   └── figures/
├── report/
│   └── hw2_report.md
├── tests/
│   └── test_smoke.py
└── .gitignore
```

## Key Algorithmic Correctness
- Q-learning: `Q[s,a] += α * (r + γ * max(Q[s',:]) - Q[s,a])` — uses max over next state
- SARSA: `Q[s,a] += α * (r + γ * Q[s', a'] - Q[s,a])` — uses actual next action from ε-greedy
- Terminal state: target = r (no bootstrap)
- Cliff fall: reward=-100, reset to S, done=False (episode continues)
- Goal: done=True
