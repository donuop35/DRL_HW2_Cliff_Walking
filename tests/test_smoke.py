"""
test_smoke.py — Smoke tests verifying env, agent, and training loop are functional.
"""

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─── Environment ──────────────────────────────────────────────────────────────
def test_custom_env_reset():
    from src.environment import CliffWalkingEnv
    env = CliffWalkingEnv(seed=0)
    obs, info = env.reset()
    assert obs == 36, f"Expected start state 36 (row3,col0), got {obs}"
    assert isinstance(info, dict)


def test_custom_env_step_actions():
    from src.environment import CliffWalkingEnv
    env = CliffWalkingEnv(seed=0)
    env.reset()
    # From Start (36): move Right should go to cliff → reward -100, reset to 36
    obs, reward, terminated, truncated, info = env.step(1)  # Right
    assert reward == -100.0, f"Expected -100 for cliff, got {reward}"
    assert obs == 36, f"Expected reset to start state 36, got {obs}"
    assert not terminated


def test_custom_env_goal_terminal():
    from src.environment import CliffWalkingEnv
    env = CliffWalkingEnv(seed=0)
    # Manually set state near goal: row 2, col 11 → step Down → Goal
    env._state = env._encode(2, 11)
    obs, reward, terminated, truncated, _ = env.step(2)  # Down
    assert terminated, "Goal should terminate episode"
    assert obs == env._encode(3, 11)


def test_custom_env_n_states_actions():
    from src.environment import CliffWalkingEnv
    env = CliffWalkingEnv()
    assert env.N_STATES == 48
    assert env.N_ACTIONS == 4


# ─── Agent ────────────────────────────────────────────────────────────────────
def test_agent_qlearning_update():
    from src.agents import TabularAgent
    agent = TabularAgent("qlearning", alpha=0.1, gamma=0.9, epsilon=0.0, n_states=48, n_actions=4, seed=0)
    # Q all zeros; update with r=−1, next_state=37, done=False
    agent.update(36, 1, -1.0, 37, done=False)
    # TD target = -1 + 0.9 * max(Q[37]) = -1 + 0 = -1
    # Q[36,1] += 0.1 * (-1 - 0) = -0.1
    assert abs(agent.Q[36, 1] - (-0.1)) < 1e-9, f"Got {agent.Q[36, 1]}"


def test_agent_sarsa_update():
    from src.agents import TabularAgent
    agent = TabularAgent("sarsa", alpha=0.1, gamma=0.9, epsilon=0.0, n_states=48, n_actions=4, seed=0)
    agent.update(36, 1, -1.0, 37, a_next=0, done=False)
    assert abs(agent.Q[36, 1] - (-0.1)) < 1e-9


def test_agent_terminal_update():
    from src.agents import TabularAgent
    agent = TabularAgent("qlearning", alpha=0.1, gamma=0.9, epsilon=0.0, n_states=48, n_actions=4, seed=0)
    agent.update(47, 0, 0.0, 47, done=True)
    # Terminal: target = r = 0; Q[47,0] += 0.1 * (0 - 0) = 0
    assert agent.Q[47, 0] == 0.0


def test_agent_epsilon_greedy():
    from src.agents import TabularAgent
    rng = np.random.default_rng(42)
    agent = TabularAgent("qlearning", alpha=0.1, gamma=0.9, epsilon=0.5, n_states=48, n_actions=4, seed=42)
    # Set Q so greedy would always pick action 0
    agent.Q[0, 0] = 100.0
    actions = [agent.select_action(0) for _ in range(200)]
    greedy_ratio = actions.count(0) / 200
    # Should be roughly 0.5 (greedy) + 0.5 * 0.25 (random chance of 0) ≈ 0.625
    assert 0.4 < greedy_ratio < 0.9, f"epsilon-greedy ratio unexpected: {greedy_ratio}"


# ─── Training Loop ────────────────────────────────────────────────────────────
def test_training_runs_custom_env():
    from src.train import train_agent
    rewards = train_agent(
        algo="qlearning", alpha=0.1, gamma=0.9, epsilon=0.1,
        n_episodes=10, seed=0, use_gymnasium=False,
    )
    assert len(rewards) == 10
    assert all(r <= -1.0 for r in rewards), f"Unexpected rewards: {rewards}"


def test_sarsa_training_runs_custom_env():
    from src.train import train_agent
    rewards = train_agent(
        algo="sarsa", alpha=0.1, gamma=0.9, epsilon=0.1,
        n_episodes=10, seed=0, use_gymnasium=False,
    )
    assert len(rewards) == 10
    assert all(r <= -1.0 for r in rewards)


def test_rewards_converge_roughly():
    """After 500 episodes, mean reward should be better than -200 (sanity check)."""
    from src.train import train_agent
    rewards = train_agent(
        algo="qlearning", alpha=0.1, gamma=0.9, epsilon=0.1,
        n_episodes=500, seed=0, use_gymnasium=False,
    )
    final_mean = np.mean(rewards[-100:])
    assert final_mean > -200, f"Final mean too low: {final_mean}"


# ─── Utils ────────────────────────────────────────────────────────────────────
def test_moving_average():
    from src.utils import moving_average
    data = [1.0] * 100
    ma = moving_average(data, window=10)
    assert len(ma) == 100
    assert abs(ma[50] - 1.0) < 1e-9


def test_find_convergence_episode_never():
    from src.utils import find_convergence_episode
    data = np.full(100, -50.0)
    ep = find_convergence_episode(data, threshold=-20.0, window=10, sustained=10)
    assert ep == 100  # never converged


def test_find_convergence_episode_immediate():
    from src.utils import find_convergence_episode
    data = np.full(200, -15.0)  # always above threshold
    ep = find_convergence_episode(data, threshold=-20.0, window=10, sustained=10)
    assert ep == 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
