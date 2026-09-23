"""Equal-weight statistics across independently trained runs."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "plots"))
from study_plotting import aggregate_runs


def main(root):
    tables = root / "tables"
    data = pd.read_csv(tables / "summary.csv")
    excluded = {"train_seed", "generation_horizon", "flow_steps"}
    metrics = [key for key in data.select_dtypes("number")
               if key not in excluded and not key.endswith(("_low", "_high"))]
    group = ["env", "map", "condition_id", "distribution"]
    group = [key for key in group if key in data.columns]
    result = aggregate_runs(data, group, metrics)
    result.to_csv(tables / "run_summary.csv", index=False)
    efficiency_metrics = [key for key in ("decision_ms", "win_rate", "return_mean", "return_std")
                          if key in data.columns]
    efficiency = aggregate_runs(data, [key for key in ("env", "map", "condition_id", "hardware") if key in data.columns],
                                efficiency_metrics)
    efficiency.to_csv(tables / "run_efficiency.csv", index=False)
    bucket_path = tables / "quality_buckets.csv"
    if bucket_path.stat().st_size:
        buckets = pd.read_csv(bucket_path)
        keys = ["map", "model_id", "condition_id", "grouping", "value"]
        runs = buckets.groupby(keys).sum(numeric_only=True).reset_index()
        metrics = []
        for key in buckets:
            if key.endswith("_sum"):
                metric = key[:-4]
                runs[metric] = runs[key] / runs["count"]
                metrics.append(metric)
        runs[keys + ["count"] + metrics].to_csv(tables / "quality_per_run.csv", index=False)
        aggregate_runs(runs, ["map", "condition_id", "grouping", "value"],
                       metrics).to_csv(tables / "run_quality.csv", index=False)
    else:
        pd.DataFrame(columns=["map", "condition_id", "grouping", "value"]).to_csv(
            tables / "run_quality.csv", index=False)
    print(f"Aggregated {data.model_id.nunique()} runs with equal weights", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
