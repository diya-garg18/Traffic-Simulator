"""
traffic_env.py — Custom traffic-intersection environment (built from scratch, no Gymnasium).

This file defines the Markov Decision Process (MDP) for a single 4-way
intersection with two conflicting traffic streams: North-South (NS) and
East-West (EW). Exactly one of the two directions has a green light at a
time; the other is red.

Why build this from scratch instead of using Gymnasium?
    The whole point of this coursework project is to be able to explain the
    MDP formulation (state, action, reward, transition dynamics) in an
    interview without hand-waving at a library's internals. Writing every
    line here means there is nothing to hide behind.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

# --- Action space -----------------------------------------------------
# Two actions only: hold the current light, or flip it. Using plain strings
# (not an Enum/IntEnum) keeps this readable on a whiteboard and works fine
# as a dict key for the Q-table later — no abstraction needed for two values.
ACTION_KEEP = "KEEP"
ACTION_SWITCH = "SWITCH"
ACTIONS = (ACTION_KEEP, ACTION_SWITCH)

# --- Tunable constants that define the simulated intersection ---------
DEPARTURE_RATE = 3        # cars that clear per step when their light is green
SWITCH_PENALTY = 5        # extra cost applied when the action is SWITCH
TIME_SINCE_SWITCH_CAP = 20  # raw counter is capped here before discretizing


@dataclass
class State:
    """
    Raw (continuous/unbounded) state of the intersection at a given step.

    cars_waiting_ns / cars_waiting_ew:
        Integer queue lengths on each road. Unbounded in principle (a queue
        can always grow), which is exactly why discretize_state() exists.
    current_light:
        Which road currently has green — "NS" or "EW". EW is implicitly red
        whenever NS is green, and vice versa (only one green at a time).
    time_since_last_switch:
        How many steps have passed since the light last changed. Tracked
        mainly so the agent can learn "don't flicker" behaviour instead of
        switching every single step.
    """
    cars_waiting_ns: int
    cars_waiting_ew: int
    current_light: str
    time_since_last_switch: int


class TrafficEnv:
    """
    A single 4-way intersection simulated as an MDP.

    Standard RL environment interface:
        reset() -> discretized_state
        step(action) -> (next_discretized_state, reward, done)

    The environment holds the TRUE (raw, undiscretized) state internally.
    Discretization is only a lens the agent looks through — the simulator
    itself always tracks exact car counts, which is what makes
    value_iteration.py possible later (it needs the real transition model).
    """

    def __init__(
        self,
        rate_ns: float = 0.5,
        rate_ew: float = 0.5,
        max_steps: int = 200,
        rush_hour: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        rate_ns / rate_ew:
            Poisson arrival rate (expected new cars per step) for each road.
            Kept as separate parameters (not one shared rate) specifically so
            uneven traffic can be simulated — e.g. a busier NS avenue vs a
            quieter EW street.
        max_steps:
            Episode length. Q-learning needs episodes to terminate so we can
            reset and generate many training trajectories; a real
            intersection never "ends", but training does.
        rush_hour:
            When True, arrival rates are overridden by a time-varying
            schedule (see _current_arrival_rates) instead of the fixed
            rate_ns/rate_ew above. Used to test whether the agent adapts to
            changing traffic patterns rather than memorizing one fixed rate.
        seed:
            Optional seed for the internal random generator, so evaluation
            runs can be made reproducible (same arrival sequence every time).
        """
        self.rate_ns = rate_ns
        self.rate_ew = rate_ew
        self.max_steps = max_steps
        self.rush_hour = rush_hour
        self._rng = np.random.default_rng(seed)

        # Populated by reset(); declared here with type hints for clarity.
        self.state: State
        self.current_step: int

        self.reset()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[str, str, str, str]:
        """
        Start a fresh episode: empty queues, NS green by default, light has
        "just switched" (time_since_last_switch = 0), step counter at 0.

        Returns the DISCRETIZED starting state, because that is what the
        agent actually consumes (see discretize_state).
        """
        self.state = State(
            cars_waiting_ns=0,
            cars_waiting_ew=0,
            current_light="NS",
            time_since_last_switch=0,
        )
        self.current_step = 0
        return self.discretize_state(self.state)

    def step(self, action: str) -> Tuple[Tuple[str, str, str, str], float, bool]:
        """
        Advance the simulation by one time step.

        Order of operations matters here and mirrors how a real intersection
        actually behaves within one tick:
            1. New cars arrive on both roads (Poisson), independent of the
               action — arrivals happen regardless of what the light does.
            2. The action is applied: KEEP leaves the light alone (and ages
               time_since_last_switch by one); SWITCH flips which road is
               green and resets time_since_last_switch to 0.
            3. Whichever road now has green discharges up to DEPARTURE_RATE
               cars, capped at how many are actually waiting (can't depart
               cars that don't exist).
            4. Reward is computed from the RESULTING queue state — i.e. the
               consequence of arrivals + the action + departures together.

        Returns (next_discretized_state, reward, done).
        """
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action!r}. Must be one of {ACTIONS}.")

        # --- 1. Arrivals -------------------------------------------------
        rate_ns, rate_ew = self._current_arrival_rates()
        self.state.cars_waiting_ns += int(self._rng.poisson(rate_ns))
        self.state.cars_waiting_ew += int(self._rng.poisson(rate_ew))

        # --- 2. Apply the action ------------------------------------------
        if action == ACTION_SWITCH:
            self.state.current_light = "EW" if self.state.current_light == "NS" else "NS"
            self.state.time_since_last_switch = 0
        else:  # ACTION_KEEP
            self.state.time_since_last_switch = min(
                self.state.time_since_last_switch + 1, TIME_SINCE_SWITCH_CAP
            )

        # --- 3. Departures on the (now current) green road -----------------
        if self.state.current_light == "NS":
            departing = min(DEPARTURE_RATE, self.state.cars_waiting_ns)
            self.state.cars_waiting_ns -= departing
        else:
            departing = min(DEPARTURE_RATE, self.state.cars_waiting_ew)
            self.state.cars_waiting_ew -= departing

        # --- 4. Reward -----------------------------------------------------
        # Negative of total queue length: every waiting car costs the agent
        # 1 point per step, so minimizing total wait = maximizing reward.
        # This is a direct, easy-to-justify proxy for "average wait time"
        # without needing to track individual cars' wait durations.
        #
        # The SWITCH_PENALTY is added on top of that so that flipping the
        # light is never a "free" action. Without it, a policy that
        # oscillates NS/EW every single step can look attractive purely
        # because departures happen on whichever road is momentarily green
        # — in reality that flickering would be useless/unsafe at a real
        # intersection (cars need a minimum green duration to actually
        # cross), so this penalty discourages that degenerate behaviour.
        total_waiting = self.state.cars_waiting_ns + self.state.cars_waiting_ew
        reward = -float(total_waiting)
        if action == ACTION_SWITCH:
            reward -= SWITCH_PENALTY

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.discretize_state(self.state), reward, done

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    def discretize_state(self, state: State) -> Tuple[str, str, str, str]:
        """
        Bucket the raw (unbounded) state into a small, finite set of
        categories so tabular Q-learning has a manageable number of states.

        Why discretization is necessary:
            Tabular Q-learning stores one Q-value per (state, action) pair
            in a dictionary. If cars_waiting_ns and cars_waiting_ew were used
            as raw integers, the state space would be unbounded (queues can
            grow arbitrarily long) — the table could never be fully
            explored, let alone converge, because most exact car counts
            would only ever be visited once. Bucketing each queue into
            {LOW, MEDIUM, HIGH} collapses infinitely many raw states into
            just 3 categories per road, giving a small, finite,
            learnable state space: 3 (NS) x 3 (EW) x 2 (light) x 3 (time)
            = 54 total states. Small enough to explore fully; coarse enough
            to still capture the distinctions that matter for the policy
            (roughly: is a road empty, moderately busy, or overloaded?).
        """
        return (
            self._bucket_cars(state.cars_waiting_ns),
            self._bucket_cars(state.cars_waiting_ew),
            state.current_light,
            self._bucket_time(state.time_since_last_switch),
        )

    @staticmethod
    def _bucket_cars(count: int) -> str:
        """LOW: 0-3, MEDIUM: 4-8, HIGH: 9+. Thresholds chosen so that LOW
        roughly matches 'clears in one green phase' (DEPARTURE_RATE=3), and
        HIGH flags a queue that's clearly outgrowing what a single green
        phase can drain — the two boundaries an intersection controller
        actually cares about."""
        if count <= 3:
            return "LOW"
        elif count <= 8:
            return "MEDIUM"
        else:
            return "HIGH"

    @staticmethod
    def _bucket_time(steps: int) -> str:
        """SHORT: 0-2, MEDIUM: 3-7, LONG: 8+ (of TIME_SINCE_SWITCH_CAP=20).
        SHORT roughly covers 'just switched, don't switch again immediately'.
        LONG flags a light that has been held for a while, which is the
        regime where the agent needs to weigh 'is the other road starving?'
        against the SWITCH_PENALTY."""
        if steps <= 2:
            return "SHORT"
        elif steps <= 7:
            return "MEDIUM"
        else:
            return "LONG"

    # ------------------------------------------------------------------
    # Rush-hour arrival-rate schedule
    # ------------------------------------------------------------------

    def _current_arrival_rates(self) -> Tuple[float, float]:
        """
        Returns (rate_ns, rate_ew) for the current step.

        When rush_hour=False, this is just the fixed rates passed to
        __init__. When rush_hour=True, the episode is split into three
        equal phases so the agent (and the evaluation plots) can show
        adaptation to a changing environment:
            phase 1 (first third):  NS-heavy   (morning commute inbound)
            phase 2 (middle third): balanced   (midday)
            phase 3 (final third):  EW-heavy   (evening commute)
        The exact numbers are deliberately simple multiples of the base
        rates so the effect is visible without needing a large max_steps.
        """
        if not self.rush_hour:
            return self.rate_ns, self.rate_ew

        phase_length = max(self.max_steps // 3, 1)
        phase = min(self.current_step // phase_length, 2)  # clamp to phase 3

        if phase == 0:
            return self.rate_ns * 2.0, self.rate_ew * 0.5
        elif phase == 1:
            return self.rate_ns, self.rate_ew
        else:
            return self.rate_ns * 0.5, self.rate_ew * 2.0


if __name__ == "__main__":
    # Manual smoke test: run a handful of steps and print what happens, so
    # the environment's behaviour can be eyeballed for sanity before any
    # agent is built on top of it (arrivals accumulating, green road
    # draining, SWITCH resetting the timer, reward going more negative as
    # queues grow).
    env = TrafficEnv(rate_ns=1.2, rate_ew=0.6, max_steps=20, seed=42)
    state = env.reset()
    print(f"reset -> discretized state: {state}")
    print(f"{'step':>4} {'action':>7} | {'raw NS':>6} {'raw EW':>6} {'light':>5} "
          f"{'t_since':>7} | {'reward':>7} | discretized")
    print("-" * 80)

    actions_to_try = [
        ACTION_KEEP, ACTION_KEEP, ACTION_KEEP, ACTION_SWITCH,
        ACTION_KEEP, ACTION_KEEP, ACTION_SWITCH, ACTION_SWITCH,
        ACTION_KEEP, ACTION_KEEP,
    ]
    for i, action in enumerate(actions_to_try):
        disc_state, reward, done = env.step(action)
        s = env.state
        print(f"{i:>4} {action:>7} | {s.cars_waiting_ns:>6} {s.cars_waiting_ew:>6} "
              f"{s.current_light:>5} {s.time_since_last_switch:>7} | "
              f"{reward:>7.1f} | {disc_state}")
        if done:
            print("episode done")
            break
