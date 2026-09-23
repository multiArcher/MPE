from study_plotting import load, save, plt, percent_axis, error_limit


def main():
    root, data = load()
    metrics = ["agreement", "q_softmax_kl", "reference_q_gap"]
    metrics += [m for m in ("intervention_agreement", "correction", "damage") if m in data]
    for map_name, rows in data.groupby("map"):
        for metric in metrics:
            if metric not in rows:
                continue
            fig, ax = plt.subplots()
            for family, group in rows.groupby("distribution"):
                ax.errorbar(group.sampled_delay_mean, group[metric],
                            yerr=group[metric + "_std"].fillna(0),
                            fmt="o", capsize=2, label=family)
            ax.set(xlabel="Measured packet delay mean", ylabel=metric,
                   title=f"{map_name}: run mean ± SD")
            ax.legend()
            if metric in ("agreement", "intervention_agreement", "correction", "damage"):
                percent_axis(ax)
            else:
                ax.set_ylim(0, error_limit(data[metric] + data[metric + "_std"].fillna(0)))
            save(root, f"policy_{map_name}_{metric}", fig)


if __name__ == "__main__":
    main()
