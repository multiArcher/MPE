"""Shared MPE return colors. Low is the theoretical worst return, high is the best."""
import json
import math
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

# Pale orange is the lower return, deep orange the upper return.
SCALE = [
    "#FFF3E0", "#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726",
    "#FF9800", "#FB8C00", "#F57C00", "#EF6C00", "#E65100",
]
# Entities are placed in [-1, 1]^2, so the farthest pair is the diagonal.
DIAGONAL = 2 * math.sqrt(2)
SQUARED_DIAGONAL = DIAGONAL ** 2


def colormap():
    return LinearSegmentedColormap.from_list("mpe_return", SCALE)


# Fixed episode-return anchors. Pale orange is the lower value, deep orange the upper.
FIXED_BOUNDS = {
    "simple_spread": (-100.0, 0.0),
    "simple_tag": (0.0, 300.0),
}


def return_bounds(map_name, time_limit, scenario_args=None, common_reward=True,
                  reward_scalarisation="sum"):
    """Episode-return limits for the logged team return.

    simple_spread and simple_tag use fixed anchors so every algorithm shares one
    color scale. Other maps still use the scenario's geometric step range.
    """
    scenario_args = dict(scenario_args or {})
    family = map_name.rsplit("_v", 1)[0]
    if family in FIXED_BOUNDS:
        return FIXED_BOUNDS[family]
    step_low, step_high, n_agents = _step_total(family, scenario_args)
    if common_reward and reward_scalarisation == "mean":
        step_low, step_high = step_low / n_agents, step_high / n_agents
    elif reward_scalarisation not in ("sum", "mean"):
        raise ValueError(f"Unknown reward_scalarisation: {reward_scalarisation}")
    horizon = int(time_limit)
    return step_low * horizon, step_high * horizon


def bounds_for_map(root, map_name):
    """Read the study's saved environment settings and return (low, high)."""
    seen = {}
    for path in sorted(Path(root).glob("runs/*/*/batch_*/effective_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        env_args = config.get("env_args") or {}
        name = env_args.get("map_name") or env_args.get("scenario")
        if name != map_name:
            continue
        key = (
            int(env_args.get("time_limit", 25)),
            bool(config.get("common_reward", True)),
            config.get("reward_scalarisation", "sum"),
            json.dumps(env_args.get("scenario_args") or {}, sort_keys=True),
        )
        seen[key] = key
    if not seen:
        seen[(25, True, "sum", "{}")] = (25, True, "sum", "{}")
    if len(seen) > 1:
        raise ValueError(f"{map_name} has more than one reward configuration in this study")
    time_limit, common_reward, scalarisation, scenario_json = next(iter(seen.values()))
    return return_bounds(map_name, time_limit, json.loads(scenario_json), common_reward, scalarisation)


def _step_total(family, args):
    """Return (lowest sum of agent rewards, highest sum, agent count) for one step."""
    if family == "simple_spread":
        n = int(args.get("N", 3))
        mix = float(args.get("local_ratio", 0.5))
        collisions = 0.0 if args.get("curriculum", False) else 1.0
        low = n * (1 - mix) * (-n * DIAGONAL) + mix * collisions * (-n * (n - 1))
        return low, 0.0, n
    if family == "simple_reference":
        return -2 * DIAGONAL, 0.0, 2
    if family == "simple_speaker_listener":
        return -2 * SQUARED_DIAGONAL, 0.0, 2
    if family == "simple":
        return -SQUARED_DIAGONAL, 0.0, 1
    if family in ("simple_line", "simple_formation"):
        n = int(args.get("N", 4))
        return -2.0 * n, 0.0, n
    if family == "simple_tag":
        good = int(args.get("num_good", 1))
        adversaries = int(args.get("num_adversaries", 3))
        pairs = good * adversaries
        # Each adversary scores every prey-predator contact, while each prey is penalized once.
        high = 10.0 * pairs * max(adversaries - 1, 0)
        low = -20.0 * good
        return low, high, good + adversaries
    if family == "simple_adversary":
        good = int(args.get("N", 2))
        return -good * DIAGONAL, (good - 1) * DIAGONAL, good + 1
    if family == "simple_push":
        return -DIAGONAL, 0.0, 2
    raise ValueError(f"No theoretical return range for MPE map {family}")
