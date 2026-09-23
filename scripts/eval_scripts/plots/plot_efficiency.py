from study_plotting import load, save, plt, percent_axis, uses_return
import pandas as pd


def main():
    root, raw = load(raw=True)
    summary = pd.read_csv(root / "tables/run_efficiency.csv")
    for map_name, rows in raw.groupby("map"):
        fig, ax = plt.subplots()
        reward = uses_return(rows)
        metric = "return_mean" if reward else "win_rate"
        error = "return_std" if reward else "win_rate_std"
        for hardware, group in rows.groupby("hardware"):
            points = ax.scatter(group.decision_ms, group[metric], alpha=0.3,
                                label=f"{hardware}: individual runs")
            means = summary[(summary["map"] == map_name) & (summary.hardware == hardware)]
            ax.errorbar(means.decision_ms, means[metric],
                        xerr=means.decision_ms_std.fillna(0),
                        yerr=means[error].fillna(0), fmt="D", capsize=2,
                        color=points.get_facecolor()[0], label=f"{hardware}: mean ± SD")
        ax.set(xlabel="Action-selection batch latency (ms, excludes scoring)",
               ylabel="Return" if reward else "Win rate", title=map_name)
        if not reward:
            percent_axis(ax)
        ax.legend(fontsize=7)
        save(root, f"efficiency_{map_name}", fig)


if __name__ == "__main__":
    main()
