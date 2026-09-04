"""
value_iteration.py — Exact dynamic-programming planner (stretch goal).

Contrast with q_learning_agent.py:
    Q-learning is MODEL-FREE — it never knows the arrival rates or departure
    rules, only what it observes by acting. Value iteration is the opposite:
    it uses the KNOWN transition model (Poisson arrival rates, fixed
    departure rate) to compute the mathematically optimal policy directly,
    with no trial and error at all. This file is "planning"; the agent file
    is "learning."

Why this needs its own (approximate) model, even though we know the exact
simulator rules:
    TrafficEnv's true state space is infinite (a queue can hold arbitrarily
    many cars), so exact dynamic programming over it is impossible — DP
    needs a FINITE state space to iterate over. This file solves that by
    planning over the same bucketed abstraction (LOW/MEDIUM/HIGH) that
    Q-learning uses for its states, representing each bucket by one
    "typical" car count when computing arrival probabilities. That is a
    real approximation of the true dynamics (see DECISIONS.md) — the
    resulting policy is optimal for this approximate model, not provably
    optimal for the true unbounded-queue environment. It's then evaluated
    on the REAL environment in evaluate.py, same as every other controller,
    so any gap between "planned" and "actual" performance is itself
    meaningful evidence of the model's accuracy.

Scope limitation: this only supports FIXED arrival rates (rush_hour=False).
Rush-hour mode changes arrival rates over the course of an episode, which
would make the transition model non-stationary (probabilities would depend
on the absolute step count, not just the bucketed state) — handling that
would require adding the step count (or phase) to the state, which is left
out here to keep the planner's state space small and its logic simple.
"""

from __future__ import annotations

import itertools
from typing import Dict, Tuple

from scipy.stats import poisson

from traffic_env import (
    ACTION_KEEP,
    ACTION_SWITCH,
    ACTIONS,
    DEPARTURE_RATE,
    SWITCH_PENALTY,
    TIME_SINCE_SWITCH_CAP,
    TrafficEnv,
)

CAR_BUCKETS = ("LOW", "MEDIUM", "HIGH")
LIGHTS = ("NS", "EW")

# One representative raw car count per bucket, used ONLY to compute
# transition probabilities and expected rewards for planning — see the
# module docstring and DECISIONS.md for why this approximation is
# necessary and what it costs. Chosen as: middle of LOW's range (0-3),
# middle of MEDIUM's range (4-8), and a fixed anchor above HIGH's open
# lower bound (9+) that represents "clearly overloaded" without picking an
# arbitrarily huge number.
BUCKET_REPRESENTATIVE_COUNT: Dict[str, int] = {"LOW": 2, "MEDIUM": 6, "HIGH": 12}

# How many possible arrival counts to sum over when computing the Poisson
# transition distribution. Truncating an infinite sum is necessary to
# actually compute anything; 30 is generously large for the arrival rates
# used in this project (probability mass beyond 30 arrivals in one step at
# rate ~1-2 is astronomically small), so truncation error is negligible.
MAX_ARRIVALS_TO_SUM = 30

# A single (state, action) key for this planner's own finite MDP:
# (ns_bucket, ew_bucket, current_light, exact_time_since_switch).
# Note this uses the EXACT time counter (0..20), not the SHORT/MEDIUM/LONG
# bucket — unlike the two stochastic car-count dimensions, time-since-switch
# transitions are fully deterministic given the action, so there is no need
# to approximate it; keeping it exact costs nothing and loses no accuracy.
VIState = Tuple[str, str, str, int]


def _build_road_transition_table(
    rate: float,
) -> Dict[Tuple[str, bool], Tuple[Dict[str, float], float]]:
    """
    For one road (given its arrival rate), precompute, for every
    (current_bucket, is_green) combination:
        - a probability distribution over the NEXT bucket
        - the expected car count after this step (used for reward)

    This is done once per road before running value iteration, since it
    doesn't depend on the light/time part of the state at all — arrivals
    and departures on one road only depend on that road's own bucket and
    whether it currently has green.
    """
    table: Dict[Tuple[str, bool], Tuple[Dict[str, float], float]] = {}

    for bucket in CAR_BUCKETS:
        representative_count = BUCKET_REPRESENTATIVE_COUNT[bucket]
        for is_green in (True, False):
            next_bucket_probs: Dict[str, float] = {b: 0.0 for b in CAR_BUCKETS}
            expected_count_after = 0.0

            for arrivals in range(MAX_ARRIVALS_TO_SUM + 1):
                p_arrivals = poisson.pmf(arrivals, rate)
                count_after_arrival = representative_count + arrivals

                if is_green:
                    departing = min(DEPARTURE_RATE, count_after_arrival)
                    count_after = count_after_arrival - departing
                else:
                    count_after = count_after_arrival

                next_bucket = TrafficEnv._bucket_cars(count_after)
                next_bucket_probs[next_bucket] += p_arrivals
                expected_count_after += p_arrivals * count_after

            # Truncating the (infinite) Poisson sum at MAX_ARRIVALS_TO_SUM
            # loses a tiny sliver of probability mass; renormalize so the
            # distribution sums to exactly 1.0 rather than ~0.999999...
            total_mass = sum(next_bucket_probs.values())
            next_bucket_probs = {
                b: p / total_mass for b, p in next_bucket_probs.items()
            }

            table[(bucket, is_green)] = (next_bucket_probs, expected_count_after)

    return table


class ValueIteration:
    """
    Solves the approximate, bucketed traffic MDP exactly via the value
    iteration algorithm:

        V(s) <- max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) * V(s') ]

    repeated until V stops changing (converges), then the optimal policy
    is read off as policy(s) = argmax_a [ R(s,a) + gamma * sum_s' P(s'|s,a) * V(s') ].

    This is the textbook Bellman OPTIMALITY update — note it looks almost
    identical to the Q-learning update in q_learning_agent.py, but here the
    expectation over next states is computed EXACTLY from the known model
    (sum_s' P(s'|s,a) * V(s')), instead of estimated from a single sampled
    transition. That's the entire difference between planning and learning.
    """

    def __init__(
        self,
        rate_ns: float,
        rate_ew: float,
        gamma: float = 0.95,
        theta: float = 1e-3,
        max_iterations: int = 1000,
    ) -> None:
        """
        rate_ns / rate_ew:
            The TRUE arrival rates from the TrafficEnv being planned for.
            Must match rate_ns/rate_ew passed to TrafficEnv for the
            resulting policy to make sense against it.
        gamma:
            Discount factor — same meaning and same default (0.95) as
            QLearningAgent, so the two methods' notions of "optimal" are
            directly comparable.
        theta:
            Convergence threshold. Value iteration stops once the largest
            change in any state's value between sweeps drops below this.
        max_iterations:
            Safety cap in case theta is set too small to reach in practice.
        """
        self.rate_ns = rate_ns
        self.rate_ew = rate_ew
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.ns_table = _build_road_transition_table(rate_ns)
        self.ew_table = _build_road_transition_table(rate_ew)

        self.states = list(
            itertools.product(
                CAR_BUCKETS, CAR_BUCKETS, LIGHTS, range(TIME_SINCE_SWITCH_CAP + 1)
            )
        )
        self.value: Dict[VIState, float] = {s: 0.0 for s in self.states}
        self.policy: Dict[VIState, str] = {}
        self.iterations_run = 0

    def _next_light_and_time(self, state: VIState, action: str) -> Tuple[str, int]:
        """Deterministic part of the transition: applying `action` to
        (light, time_since_switch) never depends on car counts at all."""
        _, _, light, time_since_switch = state
        if action == ACTION_SWITCH:
            next_light = "EW" if light == "NS" else "NS"
            next_time = 0
        else:
            next_light = light
            next_time = min(time_since_switch + 1, TIME_SINCE_SWITCH_CAP)
        return next_light, next_time

    def _q_value(self, state: VIState, action: str) -> float:
        """
        Computes R(s,a) + gamma * sum_s' P(s'|s,a) * V(s') for one
        (state, action) pair — the right-hand side of the Bellman
        optimality equation, evaluated against the CURRENT value estimates.
        """
        ns_bucket, ew_bucket, _light, _time = state
        next_light, next_time = self._next_light_and_time(state, action)

        is_ns_green = next_light == "NS"
        is_ew_green = next_light == "EW"

        ns_probs, ns_expected_count = self.ns_table[(ns_bucket, is_ns_green)]
        ew_probs, ew_expected_count = self.ew_table[(ew_bucket, is_ew_green)]

        # Expected immediate reward: negative expected total waiting cars,
        # minus the flat switch penalty if this action was SWITCH — the
        # same reward formula as TrafficEnv.step(), just using expected
        # counts under the approximate model instead of the real ones.
        reward = -(ns_expected_count + ew_expected_count)
        if action == ACTION_SWITCH:
            reward -= SWITCH_PENALTY

        # Expected future value: since ns and ew arrivals are independent,
        # the joint next-state distribution factors into the product of
        # the two roads' independent bucket distributions.
        expected_future_value = 0.0
        for next_ns, p_ns in ns_probs.items():
            for next_ew, p_ew in ew_probs.items():
                next_state = (next_ns, next_ew, next_light, next_time)
                expected_future_value += p_ns * p_ew * self.value[next_state]

        return reward + self.gamma * expected_future_value

    def solve(self) -> None:
        """
        Runs the value iteration sweeps until convergence (max change
        across all states drops below theta) or max_iterations is hit.
        After this returns, self.value and self.policy hold the final
        (converged, or best-effort) results.
        """
        for iteration in range(1, self.max_iterations + 1):
            new_value: Dict[VIState, float] = {}
            max_delta = 0.0

            for state in self.states:
                best_q = max(self._q_value(state, a) for a in ACTIONS)
                max_delta = max(max_delta, abs(best_q - self.value[state]))
                new_value[state] = best_q

            self.value = new_value
            self.iterations_run = iteration

            if max_delta < self.theta:
                break

        # One final pass to extract the greedy policy from the converged
        # values (argmax instead of max).
        for state in self.states:
            best_action = max(ACTIONS, key=lambda a: self._q_value(state, a))
            self.policy[state] = best_action

    def choose_action(self, env: TrafficEnv) -> str:
        """
        Looks up the optimal action for env's CURRENT true state.

        This reads env.state directly (raw car counts, exact time counter)
        rather than going through env.discretize_state() — per
        ARCHITECTURE.md, value_iteration.py is the one module allowed to
        do this, since it represents "planning with full knowledge of the
        model," which includes knowing the exact state, not just the
        agent's-eye bucketed view of it.
        """
        ns_bucket = TrafficEnv._bucket_cars(env.state.cars_waiting_ns)
        ew_bucket = TrafficEnv._bucket_cars(env.state.cars_waiting_ew)
        # env's own counter is already capped at TIME_SINCE_SWITCH_CAP, so
        # it's always a valid key into self.policy without extra clamping.
        state: VIState = (
            ns_bucket,
            ew_bucket,
            env.state.current_light,
            env.state.time_since_last_switch,
        )
        return self.policy[state]


if __name__ == "__main__":
    # Smoke test: solve the planner for the same rates used in
    # traffic_env.py's own manual trace, then run its resulting policy
    # against a fresh real environment and print a trace, so the behaviour
    # can be eyeballed the same way as every other file in this project.
    rate_ns, rate_ew = 1.2, 0.6

    planner = ValueIteration(rate_ns=rate_ns, rate_ew=rate_ew)
    planner.solve()
    print(f"Converged after {planner.iterations_run} iterations.")
    print(f"Policy covers {len(planner.policy)} states "
          f"(expected 3*3*2*{TIME_SINCE_SWITCH_CAP + 1} = "
          f"{3 * 3 * 2 * (TIME_SINCE_SWITCH_CAP + 1)}).")

    # A few illustrative states: an empty intersection should KEEP; a
    # badly backed-up red road that's been waiting a while should SWITCH.
    sample_states = [
        ("LOW", "LOW", "NS", 0),
        ("LOW", "HIGH", "NS", 10),
        ("HIGH", "LOW", "EW", 10),
    ]
    print("\nSample policy decisions:")
    for s in sample_states:
        print(f"  state={s} -> action={planner.policy[s]}  (V={planner.value[s]:.1f})")

    print("\nRunning the planned policy against the REAL environment:")
    env = TrafficEnv(rate_ns=rate_ns, rate_ew=rate_ew, max_steps=20, seed=42)
    env.reset()
    print(f"{'step':>4} {'action':>7} | {'raw NS':>6} {'raw EW':>6} {'light':>5} "
          f"{'t_since':>7} | {'reward':>7}")
    print("-" * 70)
    for i in range(20):
        action = planner.choose_action(env)
        _, reward, done = env.step(action)
        s = env.state
        print(f"{i:>4} {action:>7} | {s.cars_waiting_ns:>6} {s.cars_waiting_ew:>6} "
              f"{s.current_light:>5} {s.time_since_last_switch:>7} | {reward:>7.1f}")
        if done:
            break
