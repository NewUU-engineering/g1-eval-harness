"""
harness/scenarios.py

The standard test tracks. Owned by the harness: every policy runs exactly these.
Seed-dependent variation (push timing/direction jitter, initial joint noise) is
drawn from the trial seed in harness/engine.py, so a given seed means the same
disturbances for every policy.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Push:
    time_s: float                          # nominal start time (jittered per seed)
    force_xyz: Tuple[float, float, float]  # world-frame force on the pelvis (N)
    duration_s: float = 0.1

    @property
    def impulse_ns(self) -> float:
        return float((sum(f * f for f in self.force_xyz) ** 0.5) * self.duration_s)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    friction: float = 1.0                  # sliding friction of floor and feet (MuJoCo default is 1.0)
    pushes: Tuple[Push, ...] = ()
    duration_s: float = 10.0
    # If set, overrides the --cmd-vel given on the command line for this scenario.
    command_override: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pushes"] = [dict(asdict(p), impulse_ns=p.impulse_ns) for p in self.pushes]
        return d


TEST_SCENARIOS = {s.name: s for s in [
    Scenario("flat_nominal",
             "Flat ground, nominal friction, no disturbances. Walk at the commanded velocity."),
    Scenario("stand_still",
             "Zero velocity command. Checks drift and jitter while standing.",
             command_override=(0.0, 0.0, 0.0)),
    Scenario("low_friction",
             "Slippery floor (mu = 0.4, e.g. polished tiles).",
             friction=0.4),
    Scenario("ice",
             "Very slippery floor (mu = 0.2).",
             friction=0.2),
    Scenario("push_lateral_medium",
             "150 N sideways shove on the pelvis for 0.2 s (30 N*s) at t = 4 s.",
             pushes=(Push(4.0, (0.0, 150.0, 0.0), 0.2),)),
    Scenario("push_forward_hard",
             "250 N shove from behind for 0.2 s (50 N*s) at t = 4 s.",
             pushes=(Push(4.0, (250.0, 0.0, 0.0), 0.2),)),
    Scenario("push_backward_medium",
             "150 N shove from the front for 0.2 s (30 N*s) at t = 4 s.",
             pushes=(Push(4.0, (-150.0, 0.0, 0.0), 0.2),)),
]}

DEFAULT_SCENARIOS: List[str] = list(TEST_SCENARIOS.keys())
