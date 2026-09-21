"""Compare the adapter with real MPE2, including heterogeneous roles."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from envs.mpe import MPEEnv, _ALLOWED_SCENARIOS
from utils.maker import EnvMaker


def make(scenario="simple_tag_v3", **kwargs):
    return MPEEnv(scenario=scenario, time_limit=4, seed=7,
                  common_reward=False, reward_scalarisation="sum", **kwargs)


@pytest.mark.parametrize("scenario", sorted(_ALLOWED_SCENARIOS))
def test_native_observations_rewards_and_actions(scenario):
    env = make(scenario)
    native = importlib.import_module(f"mpe2.{scenario}").parallel_env(max_cycles=4)
    try:
        expected, _ = native.reset(seed=19)
        actual, _ = env.reset(seed=19)
        assert env.agent_ids == native.possible_agents
        for _ in range(4):
            for agent, obs in zip(env.agent_ids, actual):
                np.testing.assert_array_equal(obs[:len(expected[agent])], expected[agent])
                assert not obs[len(expected[agent]):].any()
            actions = [native.action_space(a).n - 1 for a in env.agent_ids]
            expected, rewards, term, trunc, _ = native.step(dict(zip(env.agent_ids, actions)))
            actual, reward, terminated, truncated, _ = env.step(actions)
            np.testing.assert_allclose(reward, [rewards[a] for a in env.agent_ids])
            np.testing.assert_array_equal(env.get_state(), np.concatenate([
                np.pad(expected[a], (0, env.get_obs_size() - len(expected[a])))
                for a in env.agent_ids
            ]))
            for a, mask in zip(env.agent_ids, env.get_avail_actions()):
                assert sum(mask) == native.action_space(a).n
            assert terminated == all(term.values())
            assert truncated == all(trunc.values())
    finally:
        env.close()
        native.close()


def test_tag_dimensions_options_and_reward_sum():
    env = make(scenario_args={"num_good": 2, "num_adversaries": 2})
    try:
        env.reset()
        assert env.agent_ids == ["adversary_0", "adversary_1", "agent_0", "agent_1"]
        sizes = [env._env.observation_space(a).shape[0] for a in env.agent_ids]
        assert env.get_obs_size() == max(sizes)
        assert len(set(sizes)) == 2
        with pytest.raises(ValueError, match="Invalid action"):
            env.step([5] * 4)
        for _ in range(4):
            _, _, _, truncated, _ = env.step([0] * 4)
        assert truncated
    finally:
        env.close()
    summed = MPEEnv("simple_tag_v3", 4, 7, True, "sum")
    individual = make()
    try:
        # Force collisions to test a nonzero adversarial reward aggregation.
        for e in (summed, individual):
            e.reset()
            for index, agent in enumerate(e._env.unwrapped.world.agents):
                agent.state.p_pos[:] = [index * 0.01, 0]
        assert summed.step([0] * 4)[1] == pytest.approx(sum(individual.step([0] * 4)[1]))
    finally:
        summed.close()
        individual.close()


def test_delayed_tag_preserves_native_state_and_rewards():
    delayed = EnvMaker.make_delayed_mpe(
        scenario="simple_tag_v3", time_limit=4, seed=7, common_reward=False,
        reward_scalarisation="sum", delay_mean=2, delay_std=0, max_delay=2,
    )
    direct = make()
    try:
        delayed.training = False
        delayed.reset(seed=19)
        direct.reset(seed=19)
        history = [direct.get_obs()]
        for t in range(1, 5):
            obs, rewards, _, _, _ = delayed.step([1] * 4)
            fresh, expected_rewards, _, _, _ = direct.step([1] * 4)
            history.append(fresh)
            np.testing.assert_allclose(rewards, expected_rewards)
            np.testing.assert_array_equal(delayed.get_state(), direct.get_state())
            np.testing.assert_array_equal(obs, history[t - 2] if t >= 2 else np.zeros_like(obs))
    finally:
        delayed.close()
        direct.close()


def test_invalid_options_are_not_silently_ignored():
    with pytest.raises(TypeError, match="num_good"):
        make(num_good=2)
    with pytest.raises(ValueError, match="discrete"):
        make(scenario_args={"continuous_actions": True})
    with pytest.raises(ValueError, match="module name"):
        make("not_a_scenario")
    with pytest.raises(ValueError, match="time_limit"):
        make(scenario_args={"max_cycles": 99})
