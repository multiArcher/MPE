"""Rebuild paper tables from completed evaluations; no model or simulator imports."""
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio_interval(rows, key):
    counts = np.array([r["decision_count"] for r in rows], dtype=float)
    sums = np.array([r[key + "_sum"] for r in rows], dtype=float)
    if counts.sum() == 0:
        return None, None, None
    rng = np.random.default_rng(101)
    indices = rng.integers(len(rows), size=(1000, len(rows)))
    denominator = counts[indices].sum(1)
    bootstrap = sums[indices].sum(1)[denominator > 0] / denominator[denominator > 0]
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return sums.sum() / counts.sum(), low, high


def main(root):
    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    summary, episodes, buckets = [], [], defaultdict(lambda: defaultdict(float))
    for model in sorted((root / "runs").iterdir()):
        for condition_dir in sorted(model.iterdir()):
            completed = sorted(condition_dir.glob("batch_*/result.json"))
            if not completed:
                continue
            rows, packets = [], defaultdict(int)
            timing, decision_times = [], []
            for result_path in completed:
                directory = result_path.parent
                job = json.loads((directory / "job.json").read_text())
                config = json.loads((directory / "effective_config.json").read_text())
                metadata = dict(model_id=model.name, condition_id=condition_dir.name,
                                env=config.get("env"),
                                map=config["env_args"].get("scenario") or config["env_args"]["map_name"],
                                train_seed=config.get("seed"),
                                generation_horizon=config.get("bcrbc_generation_horizon"),
                                flow_steps=config.get("bcrbc_flow_steps"))
                timing.append(json.loads(result_path.read_text()))
                batch_rows = list(read_rows(directory / "episodes.jsonl"))
                rows.extend(batch_rows)
                episodes.extend({**metadata, **r, "delay_histogram": json.dumps(r["delay_histogram"])} for r in batch_rows)
                for row in batch_rows:
                    for delay, count in row["delay_histogram"].items():
                        packets[int(delay)] += count
                for row in read_rows(directory / "decisions.jsonl.gz"):
                    # Count batch selection latency once per step, not once per agent.
                    if row["agent"] == 0:
                        decision_times.append(row["decision_ms"])
                    if not row["eligible"]:
                        continue
                    bins = dict(age=-1 if row["age"] is None else row["age"],
                                missing_length=row["missing_length"],
                                error_bin=math.floor(math.log10(max(row["generated_z_mse"], 1e-8))))
                    if row["regime"] is not None:
                        bins["dynamic"] = f"{row['regime']}:{row['regime_age']}"
                    for kind, value in bins.items():
                        key = (model.name, condition_dir.name, metadata["map"], kind, value, row["episode"])
                        aggregate = buckets[key]
                        aggregate["count"] += 1
                        for metric in ("generated_z_mse", "mask_z_mse", "generated_obs_mse",
                                       "mask_obs_mse", "agreement", "q_softmax_kl", "reference_q_gap"):
                            aggregate[metric + "_sum"] += row[metric]
            n = len(rows)
            returns = np.asarray([r["episode_return"] for r in rows], dtype=float)
            return_mean = float(returns.mean())
            return_std = float(returns.std(ddof=1)) if n > 1 else None
            mpe = config.get("env") in ("mpe", "delayed_mpe")
            packet_count = sum(packets.values())
            mean = sum(d*c for d, c in packets.items()) / packet_count
            variance = sum((d-mean)**2*c for d, c in packets.items()) / packet_count
            cumulative = 0
            p95 = 0
            for delay, count in sorted(packets.items()):
                cumulative += count
                if cumulative >= 0.95 * packet_count:
                    p95 = delay
                    break
            sample_count = sum(r["sample_count"] for r in rows)
            family = job["condition"]["kind"]
            if family == "gaussian" and job["condition"]["std"] == 0:
                family = "fixed"
            record = dict(**metadata, distribution=family,
                          condition=json.dumps(job["condition"]), episodes=n,
                          return_mean=return_mean, return_std=return_std,
                          length_mean=np.mean([r["length"] for r in rows]),
                          reference_obs_all_mse=sum(r["reference_obs_all_mse"]*r["sample_count"] for r in rows)/sample_count,
                          decision_count=sum(r["decision_count"] for r in rows),
                          missing_fraction=sum(r["missing_count"] for r in rows)/sample_count,
                          never_arrived_fraction=sum(r["never_arrived_count"] for r in rows)/sample_count,
                          sampled_delay_mean=mean, sampled_delay_std=math.sqrt(variance),
                          sampled_delay_p95=p95,
                          clipped_fraction=sum(r["clipped_count"] for r in rows)/packet_count,
                          sampled_delay_histogram=json.dumps(packets),
                          decision_ms=np.mean(decision_times),
                          hardware=" / ".join(sorted({r.get("device_name", "unknown") for r in timing})),
                          episodes_per_second=n/sum(r["seconds"] for r in timing),
                          diagnostic_inclusive_peak_gib=max(r["diagnostic_inclusive_peak_bytes"] for r in timing)/1024**3)
            if not mpe:
                wins = sum(r["won"] for r in rows)
                p, z = wins / n, 1.96
                center = (p + z * z / (2 * n)) / (1 + z * z / n)
                radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
                record.update(win_rate=p, win_low=center - radius, win_high=center + radius)
            for metric in ("dead_allies", "dead_enemies"):
                values = [r[metric] for r in rows if r.get(metric) is not None]
                record[metric + "_mean"] = np.mean(values) if values else None
            for key in rows[0]:
                if key.endswith("_sum"):
                    metric = key[:-4]
                    record[metric], record[metric+"_low"], record[metric+"_high"] = ratio_interval(rows, metric)
            summary.append(record)
    baselines = {(r["map"], r["model_id"]): r["win_rate"] for r in summary
                 if r["condition_id"] == "fixed_0" and "win_rate" in r}
    for row in summary:
        if "win_rate" not in row:
            continue
        baseline = baselines.get((row["map"], row["model_id"]))
        row["win_drop_from_no_delay"] = None if baseline is None else baseline - row["win_rate"]
    write_csv(tables / "summary.csv", summary)
    write_csv(tables / "episodes.csv", episodes)
    bucket_rows = [dict(model_id=k[0], condition_id=k[1], map=k[2], grouping=k[3],
                        value=k[4], episode=k[5], **v) for k, v in buckets.items()]
    write_csv(tables / "quality_buckets.csv", bucket_rows)
    print(f"Summarized {len(summary)} model/condition pairs into {tables}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
