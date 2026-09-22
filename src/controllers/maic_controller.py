"""MAIC adapter for this repository's runner and EpisodeBatch API.

Inherits BasicMAC for observation construction, last-action / agent-id input,
hidden-state init and optional observation delay. The agent still consumes a
flat ``(B * n_agents, input_dim)`` tensor, as in mansicer/MAIC.
"""

from copy import copy

import torch as th

from controllers.basic_controller import BasicMAC


class MAICMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        args = copy(args)
        if args.n_agents < 2:
            raise ValueError("MAIC requires at least two agents")
        if args.agent_output_type != "q":
            raise ValueError("This adapter implements the authors' released Q-learning MAIC")
        # Official source reads rnn_hidden_dim; this repo's GRU width is hidden_dim.
        if getattr(args, "rnn_hidden_dim", None) is None:
            args.rnn_hidden_dim = args.hidden_dim
        else:
            args.hidden_dim = args.rnn_hidden_dim
        super().__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(
            ep_batch, t_ep, test_mode=test_mode, train_mode=False
        )
        return self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, losses = self.agent.forward(
            agent_inputs,
            self.hidden_states,
            ep_batch.batch_size,
            test_mode=test_mode,
            **kwargs,
        )

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
                agent_outs = (
                    (1 - self.action_selector.epsilon) * agent_outs
                    + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num
                )
                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), losses
