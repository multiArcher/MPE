from study_plotting import load, save, plt, json, PercentFormatter, uses_return
import numpy as np

from mpe_return_scale import bounds_for_map, colormap


def main():
    root, data = load()
    manifest = json.loads((root / "manifest.json").read_text())
    for family in ("gaussian", "uniform"):
        if family + "_cells" not in manifest:
            continue
        cells = manifest[family + "_cells"]
        means = sorted({c["mean"] for c in cells})
        stds = sorted({c["std"] for c in cells})
        for map_name, rows in data.groupby("map"):
            reward = uses_return(rows)
            metric = "return_mean" if reward else "win_rate"
            values = rows.set_index("condition_id")
            grid = np.full((len(stds), len(means)), np.nan)
            fig, ax = plt.subplots(figsize=(8, 5))
            for cell in cells:
                key = cell["condition_id"]
                if key not in values.index:
                    continue
                row = values.loc[key]
                y, x = stds.index(cell["std"]), means.index(cell["mean"])
                grid[y, x] = row[metric]
                if reward:
                    label = f"{row.return_mean:.1f}±{row.return_std:.1f}"
                    text_color = "black"
                else:
                    label = f"{row.win_rate:.0%}"
                    if row.win_rate_n > 1:
                        label += f"±{100 * row.win_rate_std:.1f}"
                    text_color = "white"
                ax.text(x, y, label, ha="center", va="center", color=text_color, fontsize=8)
            if reward:
                vmin, vmax = bounds_for_map(root, map_name)
                colorbar_format, colorbar_label = None, "Return"
                cmap = colormap()
            else:
                vmin, vmax, colorbar_format = 0, 1, PercentFormatter(1)
                colorbar_label = "Win rate"
                cmap = "viridis"
            im = ax.imshow(grid, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set(xticks=range(len(means)), xticklabels=means,
                   yticks=range(len(stds)), yticklabels=stds,
                   xlabel="Raw mean", ylabel="Raw standard deviation",
                   title=f"{map_name}: {family} (run mean ± SD)")
            fig.colorbar(im, ax=ax, label=colorbar_label, format=colorbar_format)
            stem = "return" if reward else "winrate"
            save(root, f"{stem}_{family}_{map_name}", fig)


if __name__ == "__main__":
    main()
