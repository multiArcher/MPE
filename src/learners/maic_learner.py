"""MAIC's one-step Double-Q / QMIX loss plus teammate-model and sparsity terms.

Adapted from mansicer/MAIC
``src/learners/maic_learner.py``. The released optimiser is RMSprop; target
copies update every ``target_update_interval`` episodes. Extra losses are
averaged over the full sequence length, including the bootstrap timestep, as in
the authors' learner.
"""

import copy
from pathlib import Path

import torch as th
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from learners.learner import Learner
from utils.maker import MixerMaker


class MAICLearner(Learner):
    def __init__(self, mac, scheme, logger, args):
        if not args.common_reward:
            raise ValueError("MAIC QMIX requires common_reward=True")
        if getattr(args, "standardise_rewards", False) or getattr(args, "standardise_returns", False):
            raise ValueError("Released MAIC uses unstandardised rewards and returns")
        if args.mixer not in ("qmix", "vdn"):
            raise ValueError("Released MAIC supports qmix or vdn")
        if getattr(args, "target_update_interval", 0) <= 0:
            raise ValueError("MAIC target_update_interval must be a positive episode count")

        self.args = args
        self.mac = mac
        self.logger = logger
        self.device = args.device

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = MixerMaker.make(args.mixer, args) if args.mixer else None
        if self.mixer is not None:
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        else:
            self.target_mixer = None

        self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.mac.train()
        self.target_mac.train()

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        if mask.sum() == 0:
            return

        prepare_for_logging = t_env - self.log_stats_t >= self.args.learner_log_interval

        logs = []
        losses = []
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, returns_ = self.mac.forward(
                batch,
                t=t,
                prepare_for_logging=prepare_for_logging,
                train_mode=True,
                mixer=self.target_mixer,
            )
            mac_out.append(agent_outs)
            if prepare_for_logging and "logs" in returns_:
                logs.append(returns_["logs"])
                del returns_["logs"]
            losses.append(returns_)
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, _ = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out[1:], dim=1)
            target_mac_out = target_mac_out.masked_fill(avail_actions[:, 1:] == 0, -9999999)

            if self.args.double_q:
                mac_out_detach = mac_out.detach().clone()
                mac_out_detach = mac_out_detach.masked_fill(avail_actions == 0, -9999999)
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

            if self.target_mixer is not None:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = chosen_action_qvals - targets
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        external_loss, loss_dict = self._process_loss(losses, batch)
        loss = td_loss + external_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()
            stats = {
                "loss/total_loss": loss,
                "loss/td_loss": td_loss,
                "running/grad_norm": grad_norm,
                "q_values/td_error_abs": masked_td_error.abs().sum() / mask_elems,
                "q_values/q_taken_mean": (chosen_action_qvals * mask).sum()
                / (mask_elems * self.args.n_agents),
                "q_values/target_mean": (targets * mask).sum()
                / (mask_elems * self.args.n_agents),
            }
            stats.update({f"loss/{k}": v for k, v in loss_dict.items()})
            for key, value in stats.items():
                self.logger.log_stat(key, value.item(), t_env)
            self.log_stats_t = t_env

        self.mac.hidden_states = None
        self.target_mac.hidden_states = None

    def _process_loss(self, losses, batch: EpisodeBatch):
        total_loss = 0
        loss_dict = {}
        for item in losses:
            for k, v in item.items():
                if str(k).endswith("loss"):
                    loss_dict[k] = loss_dict.get(k, 0) + v
                    total_loss += v
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / batch.max_seq_length
        if losses:
            total_loss = total_loss / batch.max_seq_length
        return total_loss, loss_dict

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        with th.no_grad():
            for target_param, param in zip(
                self.target_mac.parameters(), self.mac.parameters()
            ):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            if self.mixer is not None:
                for target_param, param in zip(
                    self.target_mixer.parameters(), self.mixer.parameters()
                ):
                    target_param.data.copy_(
                        target_param.data * (1.0 - tau) + param.data * tau
                    )

    def to(self, device):
        self.device = device
        for module in (self.mac, self.target_mac, self.mixer, self.target_mixer):
            if module is not None:
                module.to(device)
        for state in self.optimiser.state.values():
            for key, value in state.items():
                if th.is_tensor(value):
                    state[key] = value.to(device)
        return self

    def cuda(self):
        return self.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        path = Path(path)
        self.mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(path / "mixer.th", map_location="cpu", weights_only=True)
            )
        self.optimiser.load_state_dict(
            th.load(path / "opt.th", map_location="cpu", weights_only=True)
        )
        self._update_targets_hard()
        self.to(self.device)
