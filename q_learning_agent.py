"""
q_learning_agent.py — Tabular Q-learning agent, built from scratch (no RL libraries).

This file implements the Q-learning algorithm itself: a Q-table, an
epsilon-greedy action-selection rule, the Bellman update, and a training
loop that runs the agent against a TrafficEnv for many episodes.

Why tabular Q-learning (not DQN / a neural network)?
    The state space here is small and finite (54 discretized states x 2
    actions = 108 table entries — see traffic_env.discretize_state). A
    dictionary can hold every possible (state, action) pair directly, so
    there is no need to approximate the Q-function with a neural network.
    Tabular Q-learning is also exact and easy to inspect/debug (you can
    print the whole table), which matters for a project meant to be fully
    understood, not just working.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from traffic_env import ACTIONS, TrafficEnv

# A discretized state is the 4-tuple returned by TrafficEnv.discretize_state,
# e.g. ("LOW", "MEDIUM", "NS", "SHORT"). Used only for type hints below.
StateType = Tuple[str, str, str, str]


class QLearningAgent:
    """
    Tabular Q-learning agent.

    The Q-table is a dict mapping (state, action) -> estimated value of
    taking that action in that state and acting optimally afterwards. This
    is exactly the Q-function from the theory: Q(s, a).

    Using a dict (keyed by tuples) instead of a NumPy array indexed by
    integers keeps the state representation human-readable (you can print
    Q[("LOW", "HIGH", "NS", "SHORT"), "SWITCH"] directly) and avoids having
    to write a separate state -> integer-index mapping. `defaultdict(float)`
    means any (state, action) pair not yet seen is treated as having value
    0.0 the first time it's looked up, which is the standard Q-learning
    initialization (optimistic-vs-zero initialization is a real design
    choice — see DECISIONS.md).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 500,
        seed: int | None = None,
    ) -> None:
        """
        alpha (learning rate):
            How much each new experience overwrites the old Q-value estimate.
            0.1 is a common, conservative default — small enough that a
            single noisy transition (remember, arrivals are Poisson/random)
            doesn't wildly swing an estimate, large enough that learning
            still happens in a few hundred episodes.
        gamma (discount factor):
            How much future reward matters relative to immediate reward.
            0.95 is close to 1 (far-sighted) because traffic control is
            inherently about long-run queue management, not just this
            step's cars — a policy that only cares about the next step
            would never learn to hold a light a bit longer to fully drain
            a queue. It's not exactly 1 so that Q-values stay finite/stable
            over long episodes.
        epsilon_start / epsilon_end / epsilon_decay_episodes:
            Epsilon-greedy exploration schedule. Start at 1.0 (always
            explore randomly — the agent knows nothing yet, so acting
            greedily on a blank Q-table would just lock in arbitrary
            early guesses). Decay linearly down to epsilon_end (0.05) over
            epsilon_decay_episodes episodes, then hold at that floor. The
            floor is kept above 0 (rather than decaying all the way to 0)
            because traffic arrivals are stochastic — the agent should
            keep sampling a small amount forever in case rare states show
            up later that its current policy handles badly.
        seed:
            Seeds this agent's own random generator (used for epsilon-greedy
            coin flips and random action choices), kept separate from the
            environment's RNG so training reproducibility doesn't depend on
            how many random draws the environment happens to make.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self._rng = np.random.default_rng(seed)

        # The Q-table. defaultdict(float) means Q[new_key] reads as 0.0
        # instead of raising KeyError — every state/action pair implicitly
        # starts at value 0 until the agent actually visits it.
        self.q_table: Dict[Tuple[StateType, str], float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    def epsilon_for_episode(self, episode: int) -> float:
        """
        Linear decay from epsilon_start to epsilon_end over
        epsilon_decay_episodes, then held flat at epsilon_end.

        Linear (rather than exponential) decay is used because it's the
        easiest schedule to reason about and justify out loud: "epsilon
        drops by a fixed amount each episode until it hits the floor."
        Exponential decay is also common but adds a decay-rate
        hyperparameter that's harder to tune/explain for the same result.
        """
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_end
        # Fraction of the way through the decay period, from 0.0 to 1.0.
        progress = episode / self.epsilon_decay_episodes
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: StateType, epsilon: float) -> str:
        """
        Epsilon-greedy policy:
            with probability epsilon, act randomly (explore)
            otherwise, act greedily w.r.t. the current Q-table (exploit)

        Exploration is necessary because the agent only learns about
        (state, action) pairs it actually visits — a purely greedy agent
        could get stuck always choosing KEEP just because it happened to
        look slightly better early on, never discovering that SWITCH is
        sometimes better in that state.
        """
        if self._rng.random() < epsilon:
            return self._rng.choice(ACTIONS)
        return self.best_action(state)

    def best_action(self, state: StateType) -> str:
        """
        Returns argmax_a Q(state, a) over the two possible actions.

        Ties (e.g. both actions still at their default 0.0 for a
        never-visited state) are broken by just taking the first action in
        ACTIONS order (KEEP) — simple and deterministic, no need for
        random tie-breaking at this scale.
        """
        q_keep = self.q_table[(state, ACTIONS[0])]
        q_switch = self.q_table[(state, ACTIONS[1])]
        return ACTIONS[0] if q_keep >= q_switch else ACTIONS[1]

    def max_q_value(self, state: StateType) -> float:
        """max_a' Q(state, a') — the value used on the right-hand side of
        the Bellman update below."""
        return max(self.q_table[(state, a)] for a in ACTIONS)

    # ------------------------------------------------------------------
    # The core update rule — Bellman equation for Q-learning
    # ------------------------------------------------------------------

    def update(
        self,
        state: StateType,
        action: str,
        reward: float,
        next_state: StateType,
    ) -> None:
        """
        The Q-learning update rule:

            Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

        Read line by line, mapped to the code below:
            current_q       = Q(s,a)                  the OLD estimate
            best_next_q     = max_a' Q(s',a')          best achievable value
                                                        from the next state
            td_target       = r + gamma * best_next_q  what Q(s,a) "should"
                                                        equal, based on this
                                                        one observed
                                                        transition
            td_error        = td_target - current_q    how wrong the old
                                                        estimate was
            new_q           = current_q + alpha * td_error
                                                        nudge the estimate
                                                        toward the target by
                                                        a fraction (alpha) of
                                                        the error, rather
                                                        than overwriting it
                                                        completely — this is
                                                        what makes learning
                                                        stable under noisy,
                                                        randomly-arriving
                                                        traffic.

        This is model-free: it never uses knowledge of arrival rates or
        transition probabilities, only the single observed (s, a, r, s')
        transition. Contrast with value_iteration.py, which DOES use the
        full known model to compute the optimal policy directly.
        """
        current_q = self.q_table[(state, action)]
        best_next_q = self.max_q_value(next_state)
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_table[(state, action)] = new_q

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env: TrafficEnv, num_episodes: int) -> List[float]:
        """
        Runs the agent against env for num_episodes full episodes, updating
        the Q-table after every single step (online learning, not batched).

        Returns a list of total (summed) reward per episode, so training
        progress can be plotted afterward (evaluate.py). A rising
        (less negative) trend across episodes is the direct evidence that
        learning is actually happening, not just that code runs.
        """
        episode_rewards: List[float] = []

        for episode in range(num_episodes):
            state = env.reset()
            epsilon = self.epsilon_for_episode(episode)
            total_reward = 0.0
            done = False

            while not done:
                action = self.choose_action(state, epsilon)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

        return episode_rewards


if __name__ == "__main__":
    # Quick end-to-end smoke test: train for a modest number of episodes on
    # a fixed-seed environment and print early vs. late average reward, so
    # improvement (or lack of it) is visible without needing evaluate.py yet.
    env = TrafficEnv(rate_ns=0.8, rate_ew=0.5, max_steps=100, seed=7)
    agent = QLearningAgent(seed=7, epsilon_decay_episodes=300)

    rewards = agent.train(env, num_episodes=500)

    first_50 = rewards[:50]
    last_50 = rewards[-50:]
    print(f"Q-table size (state-action pairs visited): {len(agent.q_table)}")
    print(f"Avg reward, first 50 episodes: {np.mean(first_50):.1f}")
    print(f"Avg reward, last 50 episodes:  {np.mean(last_50):.1f}")
    print(f"Epsilon at episode 0:   {agent.epsilon_for_episode(0):.3f}")
    print(f"Epsilon at episode 150: {agent.epsilon_for_episode(150):.3f}")
    print(f"Epsilon at episode 500: {agent.epsilon_for_episode(500):.3f}")
