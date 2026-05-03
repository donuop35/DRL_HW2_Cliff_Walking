"""
train.py — Training loop shared by Q-learning and SARSA.

The only algorithmic difference between the two is in TabularAgent.update(),
so we share the entire episode loop here.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional


def make_env(use_gymnasium: bool = True):
    """
    Return a (env, n_states, n_actions) tuple.
    Tries Gymnasium CliffWalking-v1 first; falls back to custom env.
    """
    if use_gymnasium:
        for version in ["CliffWalking-v1", "CliffWalking-v0"]:
            try:
                import gymnasium as gym
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    env = gym.make(version)
                n_states = env.observation_space.n
                n_actions = env.action_space.n
                return env, n_states, n_actions
            except Exception:
                continue
        print("[make_env] Gymnasium unavailable, using custom env.")

    from src.environment import CliffWalkingEnv
    env = CliffWalkingEnv()
    return env, env.N_STATES, env.N_ACTIONS


def run_episode(env, agent, seed: Optional[int] = None) -> float:
    """
    Run one full episode and return the total undiscounted reward.
    Compatible with both Gymnasium and custom env APIs.
    """
    # Reset
    reset_result = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result

    total_reward = 0.0
    done = False

    # For SARSA: select first action before entering loop
    action = agent.select_action(state)

    while not done:
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:  # custom env returns 4-tuple (for compatibility)
            next_state, reward, terminated, truncated = step_result[:4]
            done = terminated or truncated

        total_reward += reward

        # Select next action BEFORE update (needed for SARSA)
        next_action = agent.select_action(next_state)

        agent.update(
            s=state,
            a=action,
            r=reward,
            s_next=next_state,
            a_next=next_action if agent.algo == "sarsa" else None,
            done=done,
        )

        state = next_state
        action = next_action  # SARSA advances; Q-learning ignores a_next anyway

    return total_reward


def train_agent(
    algo: str,
    alpha: float,
    gamma: float,
    epsilon: float,
    n_episodes: int,
    seed: int,
    use_gymnasium: bool = True,
) -> np.ndarray:
    """
    Train one agent for n_episodes using a single seed.
    Returns array of per-episode total rewards.
    """
    from src.agents import TabularAgent

    env, n_states, n_actions = make_env(use_gymnasium)
    agent = TabularAgent(
        algo=algo,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        n_states=n_states,
        n_actions=n_actions,
        seed=seed,
    )

    rewards = np.zeros(n_episodes, dtype=np.float64)
    for ep in range(n_episodes):
        rewards[ep] = run_episode(env, agent)

    # Close gymnasium env if applicable
    if hasattr(env, 'close'):
        env.close()

    return rewards


def run_experiment(
    algo: str,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    n_episodes: int = 1000,
    seeds: Optional[List[int]] = None,
    use_gymnasium: bool = True,
) -> np.ndarray:
    """
    Run multiple seeds and return reward matrix of shape (n_seeds, n_episodes).
    """
    if seeds is None:
        seeds = list(range(20))

    n_seeds = len(seeds)
    matrix = np.zeros((n_seeds, n_episodes), dtype=np.float64)

    for i, seed in enumerate(seeds):
        print(f"  [{algo}] seed {seed:3d} ({i+1}/{n_seeds}) ...", end="", flush=True)
        matrix[i] = train_agent(
            algo=algo,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            n_episodes=n_episodes,
            seed=seed,
            use_gymnasium=use_gymnasium,
        )
        print(f" done. final_reward={matrix[i, -1]:.1f}")

    return matrix


def get_final_policy(
    algo: str,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    n_episodes: int = 1000,
    seed: int = 0,
    use_gymnasium: bool = True,
) -> "np.ndarray":
    """Train with a fixed seed and return the final greedy policy array."""
    from src.agents import TabularAgent

    env, n_states, n_actions = make_env(use_gymnasium)
    agent = TabularAgent(
        algo=algo,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        n_states=n_states,
        n_actions=n_actions,
        seed=seed,
    )
    for _ in range(n_episodes):
        run_episode(env, agent)

    if hasattr(env, 'close'):
        env.close()

    return agent.greedy_policy()
