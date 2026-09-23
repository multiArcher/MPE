"""Small shared I/O helpers; each figure's analysis lives in its own script."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


def uses_return(data):
    """MPE reports return; SMAC reports win rate."""
    if "env" not in data.columns or data.empty:
        return False
    return set(data["env"].dropna()) <= {"mpe", "delayed_mpe"}


def percent_axis(ax):
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))


def error_limit(values):
    maximum = pd.Series(values).max()
    return 1.05 * maximum if pd.notna(maximum) and maximum > 0 else 1.0


def aggregate_runs(data, keys, metrics):
    """Each input row represents one run; missing metrics remain missing."""
    grouped = data.groupby(keys, dropna=False)
    columns = {}
    for metric in metrics:
        columns[metric] = grouped[metric].mean()
        columns[metric + "_std"] = grouped[metric].std(ddof=1)
        columns[metric + "_n"] = grouped[metric].count()
    return pd.DataFrame(columns).reset_index()


def band(ax, data, x, metric, label=None):
    data = data.sort_values(x)
    line, = ax.plot(data[x], data[metric], label=label)
    valid = data[metric + "_n"] > 1
    ax.fill_between(data[x].to_numpy(),
                    (data[metric] - data[metric + "_std"]).to_numpy(),
                    (data[metric] + data[metric + "_std"]).to_numpy(),
                    where=valid.to_numpy(), color=line.get_color(), alpha=0.2)
    return line


def load(raw=False):
    root = Path(sys.argv[1])
    name = "summary.csv" if raw else "run_summary.csv"
    return root, pd.read_csv(root / "tables" / name)


def save(root, name, figure, category="evaluation"):
    directory = root / "figures" / category
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight")
    figure.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    plt.close(figure)


def bucket_data(root, grouping):
    data = pd.read_csv(root / "tables/run_quality.csv")
    return data[data.grouping == grouping].copy()
