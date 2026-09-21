import copy
from typing import Any

import numpy as np
import torch

from envs.multiagentenv import MultiAgentEnv
from components.delay_model import DelayModel


class DelayedObservationWrapper(MultiAgentEnv):
    """Delay local observations while leaving state and action availability fresh.

    Delay is an environment property (applied here, identically for every algorithm
    that runs on the delayed env), so benchmarking stays fair. The arrival-time logic
    is delegated to the shared :class:`components.delay_model.DelayModel` (single env =
    batch axis ``b=1``): each step's fresh observation is produced at ``sent_time = t``
    and arrives at ``t + delay``; the agent receives the freshest observation that has
    arrived by step ``t``. When no observation has arrived yet (sampled delay exceeds
    the elapsed steps) the slot is zero-filled and flagged unarrived
    (``generation_time = -1``), so the very first steps honor the delay distribution
    instead of being forced fresh.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        delay_type: str = "gaussian",
        delay_mean: float = 0.0,
        delay_std: float = 0.0,
        max_delay: int = 0,
        delay_per_agent: bool = True,
        delay_during_training: bool = False,
        # seed: int | None = None,
        **_ignored,
    ):
        self.env = env
        self.episode_limit = env.episode_limit
        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]

        self.delay_model = DelayModel(
            delay_type=delay_type,
            delay_mean=delay_mean,
            delay_std=delay_std,
            max_delay=max_delay,
            delay_per_source=delay_per_agent,
        )
        self._max_t = self.episode_limit + 1

        self._training = True
        self.delay_during_training = delay_during_training
        self._time = 0
        self._current_obs: list[np.ndarray] = []
        self._current_delays = np.zeros(self.n_agents, dtype=np.int64)
        self._current_generation_times = np.zeros(self.n_agents, dtype=np.int64)

    @property
    def episode_timestep(self):
        env = self.env
        clock = getattr(env, "episode_timestep", None)
        if clock is not None:
            return int(clock)
        inner = getattr(env, "env", None)
        if inner is not None and hasattr(inner, "_episode_steps"):
            return int(inner._episode_steps)
        return int(self._time)

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, training: bool) -> None:
        self._training = bool(training)

    def reset(self, seed=None, options=None):
        fresh_obs, info = self.env.reset(seed=seed, options=options)
        # if seed is not None:
            # torch.manual_seed(seed)
        self._time = 0
        self.delay_model.reset()
        self._push_and_refresh(fresh_obs)
        return self.get_obs(), info

    def step(self, actions):
        obss, reward, terminated, truncated, info = self.env.step(actions)
        self._time += 1
        self._push_and_refresh(obss)
        return self.get_obs(), reward, terminated, truncated, info

    def get_obs(self):
        return self._copy_obs(self._current_obs)

    def get_obs_agent(self, agent_id):
        return np.array(self._current_obs[agent_id], copy=True)

    def get_obs_delay(self):
        return self._current_delays.copy()

    def get_obs_generation_time(self):
        return self._current_generation_times.copy()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        return self.env.get_total_actions()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return self.env.seed(seed)

    def save_replay(self):
        return self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()

    def _push_and_refresh(self, fresh_obs):
        """Push this step's fresh obs into the delay model and pull the delivered obs.

        The runner sets the training mode. Training bypasses delay unless
        explicitly enabled; evaluation always uses the configured distribution.
        Shapes use the b=1 batch axis: payload [1, n_agents, obs_dim].
        """
        obs_arr = np.asarray(fresh_obs, dtype=np.float32)  # [n_agents, obs_dim]
        payload = torch.from_numpy(obs_arr).unsqueeze(0)   # [1, n, d]
        bypass_delay = self._training and not self.delay_during_training
        self.delay_model.push_step(
            payload,
            self.episode_timestep,
            training=bypass_delay,
            max_t=self._max_t,
            feat_ndims=1,  # Observation features are the last axis, everything else is batch-like.
        )
        delivered, gen_t, delay, _ = self.delay_model.query_step(self.episode_timestep)
        # Squeeze the b=1 axis back to per-agent numpy.
        self._current_obs = [delivered[0, a].numpy().copy() for a in range(self.n_agents)]
        self._current_generation_times = gen_t[0].numpy().astype(np.int64)   # [n], -1 == unarrived
        self._current_delays = delay[0].numpy().astype(np.int64)

    @staticmethod
    def _copy_obs(obs: Any):
        return [np.array(agent_obs, copy=True) for agent_obs in copy.copy(obs)]
