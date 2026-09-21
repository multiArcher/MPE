"""MPE2 adapter matching VIL2C-Branch MPEWrapper semantics.

All roles are controlled; common rewards aggregate all roles. State concatenates
padded observations, and time limits bootstrap via info["episode_limit"].
Local extensions: scenario alias, tensor actions, delay clock, pygame safeguards.
"""

import importlib
import re
import os
import signal
import torch

import numpy as np
from gymnasium.spaces import Discrete

from .multiagentenv import MultiAgentEnv


# Scenarios covered by regression tests; loading accepts valid MPE2 module names.
_ALLOWED_SCENARIOS = {
    "simple_v3", "simple_spread_v3", "simple_tag_v3", "simple_adversary_v3",
    "simple_crypto_v3", "simple_push_v3", "simple_reference_v3",
    "simple_speaker_listener_v4", "simple_world_comm_v3",
}

_CRASH_SIGNALS = ("SIGSEGV", "SIGBUS", "SIGFPE", "SIGABRT")


def _restore_default_crash_signals():
    for name in _CRASH_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (ValueError, OSError):
            pass


def _disable_torch_dynamo():
    try:
        torch._dynamo.config.disable = True
    except Exception:
        pass


class MPEEnv(MultiAgentEnv):
    def __init__(
        self, scenario=None, time_limit=25, seed=None,
        common_reward=True, reward_scalarisation="mean", scenario_args=None,
        render_mode=None, args=None, map_name=None,
        # These SMAC defaults are merged into every environment by main.py.
        window_size_x=None, window_size_y=None, state_timestep_number=False,
    ):
        # Keep existing scenario= launchers; map_name is the VIL2C spelling.
        map_name = map_name if map_name is not None else (scenario or "simple_spread_v3")
        self.scenario = map_name
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        if not re.fullmatch(r"[a-z][a-z0-9_]*_v\d+", map_name):
            raise ValueError("Use an MPE2 module name, e.g. simple_spread_v3")
        if int(time_limit) != time_limit or time_limit <= 0:
            raise ValueError("time_limit must be a positive integer")
        if reward_scalarisation not in ("sum", "mean"):
            raise ValueError("reward_scalarisation must be 'sum' or 'mean'")
        if state_timestep_number:
            raise ValueError("MPE does not support state_timestep_number")
        scenario_args = dict(scenario_args or {})
        if scenario_args.pop("continuous_actions", False):
            raise ValueError("MPEWrapper requires discrete actions")
        if "max_cycles" in scenario_args or "render_mode" in scenario_args:
            raise ValueError("Set time_limit and render_mode in env_args, not scenario_args")
        try:
            module = importlib.import_module(f"mpe2.{map_name}")
        except ModuleNotFoundError as exc:
            if exc.name == "mpe2":
                raise ImportError("MPE requires MPE2: pip install mpe2") from exc
            raise
        self._env = module.parallel_env(
            max_cycles=int(time_limit), continuous_actions=False,
            render_mode=render_mode, **scenario_args,
        )
        _restore_default_crash_signals()
        _disable_torch_dynamo()
        self._episode_t = 0
        self.agents = tuple(self._env.possible_agents)
        self.agent_ids = list(self.agents)
        self.n_agents = len(self.agents)
        self.episode_limit = int(time_limit)
        self.common_reward = common_reward
        self.reward_scalarisation = reward_scalarisation
        self._pending_seed = seed
        spaces = [self._env.action_space(a) for a in self.agents]
        if not all(isinstance(space, Discrete) and space.start == 0 for space in spaces):
            self._env.close()
            raise ValueError("MPEWrapper requires zero-based Discrete action spaces")
        self._action_sizes = [space.n for space in spaces]
        self._n_actions = max(self._action_sizes)
        self._obs_size = max(int(np.prod(self._env.observation_space(a).shape)) for a in self.agents)
        self._obs = None

    def _set_obs(self, observations):
        # possible_agents is stable even when env.agents becomes empty at timeout.
        self._obs = []
        for agent in self.agents:
            obs = np.asarray(observations[agent], dtype=np.float32).reshape(-1)
            self._obs.append(np.pad(obs, (0, self._obs_size - obs.size)))

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(
            seed=self._pending_seed if seed is None else seed, options=options,
        )
        self._pending_seed = None
        self._episode_t = 0
        self._set_obs(obs)
        return self.get_obs(), info

    def step(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(-1)
        if not self._env.agents:
            raise RuntimeError("Episode has ended; call reset() before step()")
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}")
        action_dict = {}
        for agent, action, size in zip(self.agents, actions, self._action_sizes):
            value = int(action)
            if value != action or not 0 <= value < size:
                raise ValueError(f"Invalid action {action} for {agent} (Discrete({size}))")
            action_dict[agent] = value
        obs, rewards, terminations, truncations, _ = self._env.step(action_dict)
        self._episode_t += 1
        self._set_obs(obs)
        terminated = all(terminations[a] for a in self.agents)
        ended = all(terminations[a] or truncations[a] for a in self.agents)
        truncated = ended and not terminated
        if not ended and set(self._env.agents) != set(self.agents):
            raise RuntimeError("MPEWrapper requires a fixed team until episode end")
        reward = np.asarray([rewards[a] for a in self.agents], dtype=np.float32)
        if self.common_reward:
            reward = float(reward.sum() if self.reward_scalarisation == "sum" else reward.mean())
        # Runners use this flag to bootstrap time-limit transitions.
        return self.get_obs(), reward, terminated, truncated, {"episode_limit": truncated}

    def get_obs(self):
        return [obs.copy() for obs in self._obs]

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id].copy()

    def get_obs_size(self):
        return self._obs_size

    def get_state(self):
        return np.concatenate(self._obs).astype(np.float32)

    def get_state_size(self):
        return self.n_agents * self._obs_size

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        size = self._action_sizes[agent_id]
        return [1] * size + [0] * (self._n_actions - size)

    def get_total_actions(self):
        return self._n_actions

    @property
    def episode_timestep(self):
        """Clock used by the local delayed-observation wrapper."""
        return self._episode_t

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed]

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def save_replay(self):
        raise NotImplementedError("MPE replay export is unavailable; use render_mode='rgb_array'")
