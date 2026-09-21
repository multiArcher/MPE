"""Bring-up check for MPEEnv simple_spread_v3."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np


def test_mpe_shapes():
    from utils.maker import EnvMaker

    env = EnvMaker.make_mpe(
        scenario="simple_spread_v3",
        time_limit=25,
        seed=0,
        common_reward=True,
        reward_scalarisation="sum",
        window_size_x=400,
        args=object(),
    )
    info = env.get_env_info()
    assert info["n_agents"] == 3, info
    assert info["obs_shape"] == 18, info
    assert info["state_shape"] == 54, info
    assert info["n_actions"] == 5, info
    assert info["episode_limit"] == 25, info

    obs, _ = env.reset(seed=0)
    assert len(obs) == 3
    assert all(np.asarray(o).shape == (18,) for o in obs)
    assert env.get_state().shape == (54,)
    assert all(len(a) == 5 for a in env.get_avail_actions())

    actions = np.zeros(3, dtype=np.int64)
    for _ in range(5):
        obs, reward, terminated, truncated, step_info = env.step(actions)
        assert env.n_agents == 3
        assert env.get_obs_size() == 18
        assert env.get_state_size() == 54
        assert env.get_total_actions() == 5
        assert np.isscalar(reward) or np.asarray(reward).shape == ()
        if terminated or truncated:
            break

    # Consecutive reset() without seed must continue the env RNG (gymma-style).
    obs_a, _ = env.reset()
    obs_b, _ = env.reset()
    assert not all(
        np.allclose(a, b) for a, b in zip(obs_a, obs_b)
    ), "reset() reseeds every episode; layouts must vary"

    obs, _ = env.reset()
    terminated = truncated = False
    step_info = {}
    for _ in range(env.episode_limit + 2):
        obs, reward, terminated, truncated, step_info = env.step(actions)
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert step_info["episode_limit"] is True

    env.close()
    print("mpe simple_spread_v3 shape checks passed", info)


if __name__ == "__main__":
    test_mpe_shapes()
