"""MPE2 Parallel API adapter for the framework's discrete MultiAgentEnv API.

Ported from VIL2C-Branch `MPEWrapper`. Episode-runner actions may arrive as
tensors; they are converted to numpy before the original discrete checks.
"""

import importlib
import re

import numpy as np
from gymnasium.spaces import Discrete

from .multiagentenv import MultiAgentEnv


class MPEWrapper(MultiAgentEnv):
    def __init__(
        self, map_name="simple_spread_v3", time_limit=25, seed=None,
        common_reward=True, reward_scalarisation="sum", scenario_args=None,
        render_mode=None, args=None, scenario=None, bootstrap_at_timeout=False,
        # These SMAC defaults are merged into every environment by main.py.
        window_size_x=None, window_size_y=None, state_timestep_number=False,
    ):
        # env_args.scenario is an alias used by launchers; if set it selects the map.
        if scenario is not None:
            map_name = scenario
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
                raise ImportError("MPE requires MPE2: pip install -r mpe_requirements.txt") from exc
            raise
        self._env = module.parallel_env(
            max_cycles=int(time_limit), continuous_actions=False,
            render_mode=render_mode, **scenario_args,
        )
        self.agents = tuple(self._env.possible_agents)
        self.n_agents = len(self.agents)
        self.episode_limit = int(time_limit)
        self.common_reward = common_reward
        self.reward_scalarisation = reward_scalarisation
        self.bootstrap_at_timeout = bool(bootstrap_at_timeout)
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
        self._set_obs(obs)
        return self.get_obs(), info

    def step(self, actions):
        if hasattr(actions, "detach"):
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
        self._set_obs(obs)
        terminated = all(terminations[a] for a in self.agents)
        ended = all(terminations[a] or truncations[a] for a in self.agents)
        truncated = ended and not terminated
        if not ended and set(self._env.agents) != set(self.agents):
            raise RuntimeError("MPEWrapper requires a fixed team until episode end")
        reward = np.asarray([rewards[a] for a in self.agents], dtype=np.float32)
        if self.common_reward:
            reward = float(reward.sum() if self.reward_scalarisation == "sum" else reward.mean())
        # MPE is a finite-horizon task: the time limit ends the episode.
        # Setting info["episode_limit"]=True makes PyMARL bootstrap Q(s_{T+1}).
        # That is correct for cut-off continuing tasks such as SMAC, but on MPE
        # every episode times out, so value-based methods (QMIX) explode.
        # Default matches gymma: timeout is terminal. Pass bootstrap_at_timeout=True
        # only if you explicitly want the VIL2C truncation bootstrap.
        info = {"episode_limit": True} if (truncated and self.bootstrap_at_timeout) else {}
        return self.get_obs(), reward, terminated, truncated, info

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

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed]

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def save_replay(self):
        raise NotImplementedError("MPE replay export is unavailable; use render_mode='rgb_array'")
