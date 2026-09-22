"""Real MPE2 contract tests; run with python -m pytest tests/test_mpe.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("mpe2")

from envs.mpe_wrapper import MPEWrapper
from utils.maker.env_maker import EnvMaker


@pytest.mark.parametrize("map_name", ["simple_spread_v3", "simple_reference_v3", "simple_speaker_listener_v4"])
def test_matches_native_rewards_and_final_observation(map_name):
    import importlib

    native = importlib.import_module(f"mpe2.{map_name}").parallel_env(max_cycles=3)
    env = MPEWrapper(map_name=map_name, time_limit=3, seed=19)
    try:
        env.reset()
        native.reset(seed=19)
        for t in range(3):
            actions = [0] * env.n_agents
            obs, reward, term, trunc, info = env.step(actions)
            expected, rewards, _, _, _ = native.step(dict(zip(env.agents, actions)))
            assert reward == pytest.approx(np.mean(list(rewards.values())))
            assert not term
            assert trunc == (t == 2)
            assert not info.get("episode_limit", False)
            for i, agent in enumerate(env.agents):
                np.testing.assert_allclose(obs[i][:len(expected[agent])], expected[agent])
                assert obs[i].shape == (env.get_obs_size(),)
            assert env.get_state().shape == (env.get_state_size(),)
            assert np.isfinite(env.get_state()).all()
        assert not native.agents
        # Still valid for selecting the bootstrap action at the final timestep.
        assert all(any(mask) for mask in env.get_avail_actions())
        with pytest.raises(RuntimeError):
            env.step(actions)
    finally:
        env.close()
        native.close()


def test_heterogeneous_spaces_and_seed_sequence():
    env = MPEWrapper(map_name="simple_speaker_listener_v4", seed=13)
    try:
        first, _ = env.reset()
        second, _ = env.reset()
        assert any(not np.array_equal(a, b) for a, b in zip(first, second))
        repeated, _ = env.reset(seed=13)
        np.testing.assert_array_equal(first, repeated)
        assert env.get_total_actions() == 5
        assert [sum(m) for m in env.get_avail_actions()] == [3, 5]
        with pytest.raises(ValueError):
            env.step([4, 0])
    finally:
        env.close()


def test_reward_modes():
    envs = [MPEWrapper(seed=3, common_reward=False),
            MPEWrapper(seed=3, reward_scalarisation="sum"),
            MPEWrapper(seed=3, reward_scalarisation="mean")]
    try:
        for env in envs:
            env.reset()
        rewards = [env.step([0, 0, 0])[1] for env in envs]
        assert rewards[0].shape == (3,)
        assert rewards[1] == pytest.approx(rewards[0].sum())
        assert rewards[2] == pytest.approx(rewards[0].mean())
    finally:
        for env in envs:
            env.close()


def test_delayed_factory_training_eval_and_reset():
    env = EnvMaker.make("delayed_mpe", seed=5, time_limit=4,
                        delay_mean=2, delay_std=0, max_delay=2)
    try:
        env.reset(seed=5)
        np.testing.assert_array_equal(env.get_obs(), env.env.get_obs())
        env.training = False
        env.reset(seed=5)
        initial = env.env.get_obs()
        assert np.all(np.asarray(env.get_obs()) == 0)
        assert np.all(env.get_obs_generation_time() == -1)
        np.testing.assert_array_equal(env.get_state(), env.env.get_state())
        env.step([0, 0, 0])
        env.step([0, 0, 0])
        np.testing.assert_array_equal(env.get_obs(), initial)
        assert env.episode_timestep == 2
        assert np.all(env.get_obs_generation_time() == 0)
        env.reset(seed=5)
        assert env.episode_timestep == 0
        assert np.all(env.get_obs_generation_time() == -1)
    finally:
        env.close()


def test_rejects_continuous_actions():
    with pytest.raises(ValueError, match="discrete"):
        MPEWrapper(scenario_args={"continuous_actions": True})


def test_scenario_alias_overrides_map_name():
    env = MPEWrapper(map_name="simple_spread_v3", scenario="simple_tag_v3", time_limit=3)
    try:
        env.reset()
        assert env.n_agents == 4
        assert env.agents[0].startswith("adversary")
    finally:
        env.close()


def test_timeout_is_terminal_unless_bootstrap_requested():
    env = MPEWrapper(time_limit=2, seed=3)
    boot = MPEWrapper(time_limit=2, seed=3, bootstrap_at_timeout=True)
    try:
        env.reset()
        boot.reset()
        env.step([0, 0, 0])
        boot.step([0, 0, 0])
        _, _, term, trunc, info = env.step([0, 0, 0])
        _, _, b_term, b_trunc, b_info = boot.step([0, 0, 0])
        assert trunc and not term
        assert not info.get("episode_limit", False)
        assert b_trunc and not b_term
        assert b_info.get("episode_limit") is True
    finally:
        env.close()
        boot.close()
