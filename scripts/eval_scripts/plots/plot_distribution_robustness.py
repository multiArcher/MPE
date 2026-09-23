from study_plotting import load, save, plt, percent_axis, uses_return


def main():
    root, data = load()
    for map_name, rows in data.groupby("map"):
        reward = uses_return(rows)
        metric = "return_mean" if reward else "win_rate"
        error = "return_std" if reward else "win_rate_std"
        fig, ax = plt.subplots()
        for family, group in rows.groupby("distribution"):
            ax.errorbar(group.sampled_delay_mean, group[metric],
                        yerr=group[error].fillna(0), fmt="o",
                        capsize=2, label=family)
        ax.set(xlabel="Measured packet delay mean", ylabel="Return" if reward else "Win rate",
               title=f"{map_name}: run mean ± episode SD" if reward else f"{map_name}: run mean ± SD")
        if not reward:
            percent_axis(ax)
        ax.legend()
        save(root, f"distributions_{map_name}", fig)


if __name__ == "__main__":
    main()
