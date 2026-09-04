"""
evaluate.py — head-to-head comparison of all three controllers.

Runs the fixed-timer baseline, the trained Q-learning agent, and the
value-iteration planner against the SAME seeded TrafficEnv scenario, then:
    - prints a summary table (avg wait, max queue length, switch count)
    - plots average cars waiting over time for all three, on one graph
    - plots the Q-learning training reward curve (proof learning happened)
    - plots a rush-hour bonus comparison: Q-learning's switching behaviour
      vs the baseline's fixed switching, across the three traffic phases

Why a separate evaluation seed from the training seed:
    Evaluating the agent on the exact same seed it trained on would only
    prove it memorized that one trajectory, not that it learned a general
    policy. TRAIN_SEED and EVAL_SEED are deliberately different.

Why all three controllers use the SAME eval seed:
    TrafficEnv's arrivals are positional (draw at step i depends only on i,
    not on prior actions — verified while building value_iteration.py), so
    seeding all three controllers' environments identically guarantees they
    face the exact same arrival sequence. Any difference in outcome is then
    attributable only to the controller's decisions, not to luck.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # write PNG files directly, no GUI window required
import matplotlib.pyplot as plt

from traffic_env import TrafficEnv
from q_learning_agent import QLearningAgent
from baseline_controller import FixedTimerController
from value_iteration import ValueIteration

RATE_NS = 0.8
RATE_EW = 0.5
MAX_STEPS = 100
TRAIN_SEED = 7
EVAL_SEED = 123
NUM_TRAINING_EPISODES = 500
EPSILON_DECAY_EPISODES = 300
BASELINE_SWITCH_EVERY = 10

PLOTS_DIR = "plots"


def run_episode(env: TrafficEnv, choose_action_fn):
    """
    Runs exactly one episode against env, calling choose_action_fn(state) to
    pick each action. Returns:
        waiting_history: total cars waiting (NS+EW) after each step
        switch_count: how many times the action was SWITCH
        max_queue: the largest single-road queue seen at any step
    """
    state = env.reset()
    waiting_history = []
    switch_count = 0
    max_queue = 0
    done = False

    while not done:
        action = choose_action_fn(state)
        state, _reward, done = env.step(action)
        s = env.state
        waiting_history.append(s.cars_waiting_ns + s.cars_waiting_ew)
        max_queue = max(max_queue, s.cars_waiting_ns, s.cars_waiting_ew)
        if action == "SWITCH":
            switch_count += 1

    return waiting_history, switch_count, max_queue


def print_summary_table(rows: list[dict]) -> None:
    header = f"{'Controller':<16}{'Avg wait':>12}{'Max queue':>12}{'Switches':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<16}{r['avg_wait']:>12.2f}"
            f"{r['max_queue']:>12d}{r['switches']:>12d}"
        )


def plot_waiting_comparison(results: dict) -> None:
    """One graph, one line per controller: total cars waiting at each step
    of the same evaluation episode. Lower and flatter is better."""
    plt.figure(figsize=(10, 5))
    for name, (waiting_history, _switches, _max_q) in results.items():
        plt.plot(waiting_history, label=name)
    plt.xlabel("Step")
    plt.ylabel("Total cars waiting (NS + EW)")
    plt.title(f"Cars waiting over time (rate_ns={RATE_NS}, rate_ew={RATE_EW}, seed={EVAL_SEED})")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "waiting_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_training_reward(training_rewards: list[float]) -> None:
    """Per-episode total reward across training. A rising (less negative)
    trend is the direct evidence that Q-learning is actually learning."""
    window = 20
    rolling_avg = np.convolve(
        training_rewards, np.ones(window) / window, mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(training_rewards, alpha=0.3, label="Per-episode reward")
    plt.plot(
        range(window - 1, len(training_rewards)),
        rolling_avg,
        label=f"{window}-episode rolling average",
        linewidth=2,
    )
    plt.xlabel("Training episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning training progress")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_reward.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_rush_hour_switching(agent: QLearningAgent) -> None:
    """
    Bonus comparison: run the Q-learning agent and the fixed-timer baseline
    on the SAME rush-hour episode, and plot cumulative switch count over
    time with the phase boundaries marked. The baseline's line is a
    perfectly straight staircase (it switches every N steps no matter what);
    if the Q-learning line's slope visibly changes across phases, that's
    the agent adapting to the changing NS/EW traffic split.

    Note: value_iteration.py does not support rush_hour (see its module
    docstring — the planner's model assumes fixed arrival rates), so this
    comparison only covers Q-learning vs the baseline.
    """
    rush_train_env = TrafficEnv(
        rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS,
        rush_hour=True, seed=TRAIN_SEED + 1,
    )
    rush_agent = QLearningAgent(seed=TRAIN_SEED + 1, epsilon_decay_episodes=EPSILON_DECAY_EPISODES)
    rush_agent.train(rush_train_env, NUM_TRAINING_EPISODES)

    def q_switch_flags(env):
        state = env.reset()
        flags = []
        done = False
        while not done:
            action = rush_agent.best_action(state)
            state, _r, done = env.step(action)
            flags.append(1 if action == "SWITCH" else 0)
        return flags

    def baseline_switch_flags(env):
        controller = FixedTimerController(switch_every=BASELINE_SWITCH_EVERY)
        state = env.reset()
        controller.reset()
        flags = []
        done = False
        while not done:
            action = controller.choose_action(state)
            state, _r, done = env.step(action)
            flags.append(1 if action == "SWITCH" else 0)
        return flags

    q_env = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, rush_hour=True, seed=EVAL_SEED)
    base_env = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, rush_hour=True, seed=EVAL_SEED)

    q_flags = q_switch_flags(q_env)
    base_flags = baseline_switch_flags(base_env)

    q_cumulative = np.cumsum(q_flags)
    base_cumulative = np.cumsum(base_flags)

    phase_length = MAX_STEPS // 3
    plt.figure(figsize=(10, 5))
    plt.plot(q_cumulative, label="Q-learning (adaptive)")
    plt.plot(base_cumulative, label="Fixed-timer baseline")
    plt.axvline(phase_length, color="gray", linestyle="--", alpha=0.6)
    plt.axvline(phase_length * 2, color="gray", linestyle="--", alpha=0.6)
    plt.text(phase_length / 2, plt.ylim()[1] * 0.02, "NS-heavy", ha="center")
    plt.text(phase_length * 1.5, plt.ylim()[1] * 0.02, "balanced", ha="center")
    plt.text(phase_length * 2.5, plt.ylim()[1] * 0.02, "EW-heavy", ha="center")
    plt.xlabel("Step")
    plt.ylabel("Cumulative switch count")
    plt.title("Rush-hour: switching behaviour across traffic phases")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "rushhour_switching.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Train the Q-learning agent -----------------------------------
    print(f"Training Q-learning agent for {NUM_TRAINING_EPISODES} episodes...")
    train_env = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, seed=TRAIN_SEED)
    agent = QLearningAgent(seed=TRAIN_SEED, epsilon_decay_episodes=EPSILON_DECAY_EPISODES)
    training_rewards = agent.train(train_env, NUM_TRAINING_EPISODES)
    print(f"  first 50 episodes avg reward: {np.mean(training_rewards[:50]):.1f}")
    print(f"  last 50 episodes avg reward:  {np.mean(training_rewards[-50:]):.1f}")

    # --- Solve the value-iteration planner -----------------------------
    print("Solving value iteration planner...")
    planner = ValueIteration(rate_ns=RATE_NS, rate_ew=RATE_EW)
    planner.solve()
    print(f"  converged after {planner.iterations_run} iterations")

    # --- Evaluate all three on the SAME seeded scenario -----------------
    eval_env_q = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, seed=EVAL_SEED)
    q_result = run_episode(eval_env_q, agent.best_action)

    eval_env_base = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, seed=EVAL_SEED)
    baseline = FixedTimerController(switch_every=BASELINE_SWITCH_EVERY)
    baseline.reset()
    base_result = run_episode(eval_env_base, baseline.choose_action)

    eval_env_vi = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, seed=EVAL_SEED)
    vi_result = run_episode(eval_env_vi, lambda state: planner.choose_action(eval_env_vi))

    results = {
        "Q-learning": q_result,
        "Baseline": base_result,
        "Value iteration": vi_result,
    }

    # --- Summary table ---------------------------------------------------
    print()
    rows = []
    for name, (waiting_history, switches, max_q) in results.items():
        rows.append({
            "name": name,
            "avg_wait": float(np.mean(waiting_history)),
            "max_queue": max_q,
            "switches": switches,
        })
    print_summary_table(rows)

    # --- Plots -------------------------------------------------------
    plot_waiting_comparison(results)
    plot_training_reward(training_rewards)
    plot_rush_hour_switching(agent)


if __name__ == "__main__":
    main()
