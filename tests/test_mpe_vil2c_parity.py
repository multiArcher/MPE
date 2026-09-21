"""Behavioral comparison with the user's sibling VIL2C checkout, if present."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from envs.mpe import MPEEnv


@pytest.fixture
def reference_class():
    path = ROOT.parent / "VIL2C-Branch" / "src" / "envs" / "mpe_wrapper.py"
    if not path.exists():
        pytest.skip("Sibling VIL2C-Branch checkout is not available")
    spec = importlib.util.spec_from_file_location("envs._vil2c_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MPEWrapper


@pytest.mark.parametrize("scenario", ["simple_spread_v3", "simple_tag_v3"])
@pytest.mark.parametrize("common,aggregation", [(True, "mean"), (True, "sum"), (False, "mean")])
def test_reference_episode_and_seed_parity(reference_class, scenario, common, aggregation):
    options = dict(time_limit=5, seed=71, common_reward=common,
                   reward_scalarisation=aggregation)
    local = MPEEnv(scenario=scenario, **options)
    reference = reference_class(map_name=scenario, **options)
    try:
        assert local.get_env_info() == reference.get_env_info()
        assert local.get_avail_actions() == reference.get_avail_actions()
        # First reset consumes constructor seed; later resets continue RNG state.
        for episode in range(3):
            obs, info = local.reset()
            ref_obs, ref_info = reference.reset()
            np.testing.assert_array_equal(obs, ref_obs)
            assert info == ref_info
            for t in range(5):
                actions = [(episode + t + i) % sum(mask)
                           for i, mask in enumerate(local.get_avail_actions())]
                actual = local.step(torch.tensor(actions).view(-1, 1))
                expected = reference.step(actions)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                assert actual[2:] == expected[2:]
                np.testing.assert_array_equal(local.get_state(), reference.get_state())
                assert local.episode_timestep == t + 1
            assert actual[4]["episode_limit"] is True
            with pytest.raises(RuntimeError, match="Episode has ended"):
                local.step(actions)
        assert local.seed(89) == reference.seed(89)
        np.testing.assert_array_equal(local.reset()[0], reference.reset()[0])
        np.testing.assert_array_equal(local.reset(seed=13)[0], reference.reset(seed=13)[0])
    finally:
        local.close()
        reference.close()


def test_map_name_alias_and_strict_actions():
    env = MPEEnv(scenario="simple_spread_v3", map_name="simple_tag_v3")
    try:
        env.reset()
        assert env.n_agents == 4
        assert env.get_state_size() == 4 * env.get_obs_size()
        with pytest.raises(ValueError, match="Invalid action"):
            env.step([0.5] * 4)
    finally:
        env.close()
