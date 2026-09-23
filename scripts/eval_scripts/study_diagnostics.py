"""Read-only paired diagnostics; only the loaded policy controls the environment."""
import gzip
import json

import numpy as np
import torch
import torch.nn.functional as F

from modules.bcrbc.evaluation_diagnostics import action_scores


def vector_scores(candidate, reference):
    difference = candidate - reference
    return dict(mse=difference.square().mean(-1), mae=difference.abs().mean(-1),
                cosine=F.cosine_similarity(candidate, reference, dim=-1))


class StudyDiagnostics:
    def __init__(self, directory, batch_size, offset, intervention, feature_groups):
        self.directory = directory
        self.offset = offset
        self.intervention = intervention
        self.groups = feature_groups
        self.reference_encoder = self.reference_dynamics = self.reference_decoder = None
        self.mask_state = None
        self.streak = None
        self.rows = [[] for _ in range(batch_size)]
        self.histograms = [{} for _ in range(batch_size)]
        self.clipped = [0 for _ in range(batch_size)]
        self.decisions = gzip.open(directory / "decisions.jsonl.gz", "wt", encoding="utf-8")

    @torch.no_grad()
    def record(self, runner, active):
        if not active:
            return
        mac, step = runner.mac, runner.t
        data = runner.get_diagnostic_data(active)
        truth = torch.zeros_like(runner.batch["obs"][:, step:step + 1])
        truth[active, 0] = torch.as_tensor(np.asarray([v["observations"] for v in data]), device=mac.device)
        _, actions = mac._build_inputs(runner.batch, slice(step, step + 1))
        reference_z, cache = mac.agent.encode_observations(
            truth, kv_cache=self.reference_encoder, use_kv_cache=True, rope_offset=step)
        self.reference_encoder = mac._trim_cache(cache)
        reference = mac.agent.estimate_clean_z(
            reference_z, actions, torch.ones_like(reference_z[..., :1, :1]),
            start_t=step, kv_cache=self.reference_dynamics, use_kv_cache=True, rope_offset=step)
        self.reference_dynamics = mac._trim_cache(reference["kv_cache"])
        clean_obs, cache = mac.agent.decode_observations(
            reference_z, kv_cache=self.reference_decoder, use_kv_cache=True, rope_offset=step)
        self.reference_decoder = mac._trim_cache(cache)

        history = mac._eval_state["encoded_z"]
        mask_z = history[:, -1:]
        # Decoder uses precisely the current corrected MASK history, as in training.
        mask_obs, cache = mac.agent.decode_observations(history, use_kv_cache=True, rope_offset=0)
        condition = {"kv": cache, "positions": torch.arange(history.shape[1], device=mac.device)}
        reconstructed = mac.agent.decode_observations(mac.decision_z, condition=condition, rope_offset=step)
        available = runner.batch["avail_actions"][:, step]
        reference_q = reference["q_values"][:, 0, :, 0]
        scores = action_scores(reference_q, mac.decision_q_values, available)
        metrics = {"agreement": scores["agreement"].float(), "q_softmax_kl": scores["kl"]}
        reference_value = reference_q.gather(-1, scores["reference_action"][..., None]).squeeze(-1)
        candidate_value = reference_q.gather(-1, scores["candidate_action"][..., None]).squeeze(-1)
        metrics["reference_q_gap"] = reference_value - candidate_value
        for name, z in [("generated_z", mac.decision_z), ("mask_z", mask_z)]:
            for metric, value in vector_scores(z.flatten(-2), reference_z.flatten(-2)).items():
                metrics[f"{name}_{metric}"] = value[:, 0]
        for name, observation in [("generated_obs", reconstructed),
                                  ("mask_obs", mask_obs[:, -1:]), ("reference_obs", clean_obs)]:
            for metric, value in vector_scores(observation, truth).items():
                metrics[f"{name}_{metric}"] = value[:, 0]
            for group, (start, end) in self.groups.items():
                metrics[f"{name}_{group}_mse"] = (observation[..., start:end] - truth[..., start:end]).square().mean(-1)[:, 0]
        metrics["z_mse_improvement"] = metrics["mask_z_mse"] - metrics["generated_z_mse"]
        if self.intervention:
            masked, self.mask_state = mac.history_forward(
                runner.batch, slice(step, step + 1), self.mask_state, generate=False)
            before = action_scores(reference_q, masked["q_values"][:, 0, :, 0], available)
            metrics["intervention_agreement"] = before["agreement"].float()
            metrics["intervention_q_softmax_kl"] = before["kl"]
            metrics["correction"] = (~before["agreement"] & scores["agreement"]).float()
            metrics["damage"] = (before["agreement"] & ~scores["agreement"]).float()
        generation_time = runner.batch["obs_gen_t"][:, step, :, 0]
        missing = generation_time < step
        self.streak = torch.zeros_like(generation_time) if self.streak is None else self.streak
        self.streak = torch.where(missing, self.streak + 1, 0)
        cpu_metrics = {key: value.cpu().tolist() for key, value in metrics.items()}
        for row_index, env_index in enumerate(active):
            self.clipped[env_index] += data[row_index]["clipped_count"]
            for delay in data[row_index]["sampled_delays"]:
                self.histograms[env_index][str(delay)] = self.histograms[env_index].get(str(delay), 0) + 1
            for agent in range(mac.n_agents):
                gen = int(generation_time[env_index, agent])
                row = dict(episode=self.offset + env_index, step=step, agent=agent,
                           missing=bool(missing[env_index, agent]), never_arrived=gen < 0,
                           age=None if gen < 0 else step - gen,
                           missing_length=int(self.streak[env_index, agent]),
                           eligible=bool(missing[env_index, agent] and available[env_index, agent].sum() > 1),
                           actionable=bool(available[env_index, agent].sum() > 1),
                           regime=data[row_index]["regime"], regime_age=data[row_index]["regime_age"],
                           decision_ms=mac.decision_ms,
                           reference_action=int(scores["reference_action"][env_index, agent]),
                           candidate_action=int(scores["candidate_action"][env_index, agent]),
                           actual_action=int(runner.batch["actions"][env_index, step, agent, 0]),
                           **{key: value[env_index][agent] for key, value in cpu_metrics.items()})
                self.decisions.write(json.dumps(row) + "\n")
                self.rows[env_index].append(row)

    def finish(self, returns, lengths, wins):
        self.decisions.close()
        with (self.directory / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for index, rows in enumerate(self.rows):
                eligible = [r for r in rows if r["eligible"]]
                metric_keys = [k for k in rows[0] if k.endswith(("mse", "mae", "cosine"))]
                metric_keys += ["agreement", "q_softmax_kl", "reference_q_gap", "z_mse_improvement"]
                if self.intervention:
                    metric_keys += ["intervention_agreement", "intervention_q_softmax_kl", "correction", "damage"]
                record = dict(episode=self.offset + index, won=bool(wins[index]),
                              episode_return=float(returns[index]), length=int(lengths[index]),
                              decision_count=len(eligible), sample_count=len(rows),
                              missing_count=sum(r["missing"] for r in rows),
                              never_arrived_count=sum(r["never_arrived"] for r in rows),
                              reference_obs_all_mse=sum(r["reference_obs_mse"] for r in rows)/len(rows),
                              clipped_count=self.clipped[index], delay_histogram=self.histograms[index],
                              **{key + "_sum": sum(r[key] for r in eligible) for key in metric_keys})
                stream.write(json.dumps(record) + "\n")

    def log(self, logger, t_env):
        pass


class ReturnDiagnostics:
    """Episode return and packet-delay records for controllers without BCRBC latents."""

    def __init__(self, directory, batch_size, offset):
        self.directory = directory
        self.offset = offset
        self.rows = [[] for _ in range(batch_size)]
        self.histograms = [{} for _ in range(batch_size)]
        self.clipped = [0 for _ in range(batch_size)]
        self.decisions = gzip.open(directory / "decisions.jsonl.gz", "wt", encoding="utf-8")

    @torch.no_grad()
    def record(self, runner, active):
        if not active:
            return
        step = runner.t
        data = runner.get_diagnostic_data(active)
        generation_time = runner.batch["obs_gen_t"][:, step, :, 0]
        missing = generation_time < step
        decision_ms = float(getattr(runner.mac, "decision_ms", 0.0))
        for row_index, env_index in enumerate(active):
            self.clipped[env_index] += data[row_index]["clipped_count"]
            for delay in data[row_index]["sampled_delays"]:
                key = str(int(delay))
                self.histograms[env_index][key] = self.histograms[env_index].get(key, 0) + 1
            for agent in range(runner.mac.n_agents):
                gen = int(generation_time[env_index, agent])
                row = dict(episode=self.offset + env_index, step=int(step), agent=agent,
                           missing=bool(missing[env_index, agent]), never_arrived=gen < 0,
                           age=None if gen < 0 else int(step - gen), eligible=False,
                           regime=data[row_index]["regime"], regime_age=data[row_index]["regime_age"],
                           decision_ms=decision_ms)
                self.decisions.write(json.dumps(row) + "\n")
                self.rows[env_index].append(row)

    def finish(self, returns, lengths, wins):
        self.decisions.close()
        with (self.directory / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for index, rows in enumerate(self.rows):
                record = dict(episode=self.offset + index, won=bool(wins[index]),
                              episode_return=float(returns[index]), length=int(lengths[index]),
                              decision_count=0, sample_count=len(rows),
                              missing_count=sum(row["missing"] for row in rows),
                              never_arrived_count=sum(row["never_arrived"] for row in rows),
                              reference_obs_all_mse=0.0, clipped_count=self.clipped[index],
                              delay_histogram=self.histograms[index])
                stream.write(json.dumps(record) + "\n")

    def log(self, logger, t_env):
        pass
