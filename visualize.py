"""
visualize.py — simple step-by-step text demo of the intersection.

Trains a Q-learning agent (same hyperparameters as evaluate.py), then runs
it against a fixed-seed environment, printing one frame per step: an ASCII
bar for each road's queue length and which light is green. This is meant
to be watched in a terminal, not analyzed like evaluate.py's plots — it's
the "look at it actually work" demo, not the measurement tool.

Kept deliberately simple (plain print() + time.sleep(), no new
dependencies) per CONSTRAINTS.md's "no unnecessary abstraction" rule and
because this file is explicitly optional in the original project spec.
"""

from __future__ import annotations

import time

from traffic_env import TrafficEnv
from q_learning_agent import QLearningAgent

RATE_NS = 0.8
RATE_EW = 0.5
TRAIN_MAX_STEPS = 100
TRAIN_SEED = 7
NUM_TRAINING_EPISODES = 500
EPSILON_DECAY_EPISODES = 300

DEMO_SEED = 999           # different from TRAIN_SEED, same reasoning as evaluate.py:
                          # a demo on the training seed would just replay a memorized path
DEMO_MAX_STEPS = 30
STEP_DELAY_SECONDS = 0.4


def render_frame(step: int, ns_count: int, ew_count: int, light: str, action: str, reward: float) -> None:
    """Prints one text 'frame': an ASCII bar per road (one '#' per waiting
    car) and which road currently has green."""
    ns_marker = "GREEN" if light == "NS" else "red"
    ew_marker = "GREEN" if light == "EW" else "red"
    print(f"Step {step:>3} | action={action:<6} reward={reward:>6.1f}")
    print(f"  NS [{ns_marker:>5}] {'#' * ns_count} ({ns_count})")
    print(f"  EW [{ew_marker:>5}] {'#' * ew_count} ({ew_count})")
    print()


def main() -> None:
    print(f"Training a Q-learning agent for {NUM_TRAINING_EPISODES} episodes before the demo...")
    train_env = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=TRAIN_MAX_STEPS, seed=TRAIN_SEED)
    agent = QLearningAgent(seed=TRAIN_SEED, epsilon_decay_episodes=EPSILON_DECAY_EPISODES)
    agent.train(train_env, NUM_TRAINING_EPISODES)
    print("Done training. Starting demo...\n")

    env = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=DEMO_MAX_STEPS, seed=DEMO_SEED)
    state = env.reset()

    for step in range(DEMO_MAX_STEPS):
        action = agent.best_action(state)
        state, reward, done = env.step(action)
        s = env.state
        render_frame(step, s.cars_waiting_ns, s.cars_waiting_ew, s.current_light, action, reward)
        time.sleep(STEP_DELAY_SECONDS)
        if done:
            break


if __name__ == "__main__":
    main()
