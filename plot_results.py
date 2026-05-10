#!/usr/bin/env python3
"""
Generate plots for the sample-sort topology comparison study.

Reads CSVs from results/{strong,weak}/<np>/<binary>_results.csv and writes
PNGs to results/plots/.

Run via:
    docker run --rm -v "$PWD:/work" -w /work python:3.11-slim \\
        bash -c 'pip install -q matplotlib numpy && python plot_results.py'
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

NP_LIST = [16, 32, 64, 128, 256]
TOPOLOGIES = ["dragonfly", "fat_tree", "hypercube", "ring", "torus"]
IMPLEMENTATIONS = [
    "sample_sort",                  # agnostic baseline
    "sample_sort_ring",
    "sample_sort_torus",
    "sample_sort_hypercube",
    "sample_sort_fat_tree",
    "sample_sort_dragonfly",
]
IMPL_LABELS = {
    "sample_sort":            "agnostic",
    "sample_sort_ring":       "ring-opt",
    "sample_sort_torus":      "torus-opt",
    "sample_sort_hypercube":  "hypercube-opt",
    "sample_sort_fat_tree":   "fat_tree-opt",
    "sample_sort_dragonfly":  "dragonfly-opt",
}


def read_csv(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out[row["topology"]] = float(row["time_s"])
            except (ValueError, KeyError, TypeError):
                pass
    return out


def load_data():
    data = {"strong": {}, "weak": {}}
    for mode in data:
        for np_val in NP_LIST:
            data[mode][np_val] = {}
            for impl in IMPLEMENTATIONS:
                data[mode][np_val][impl] = read_csv(
                    f"results/{mode}/{np_val}/{impl}_results.csv"
                )
    return data


def plot_agnostic_scaling(data, out_dir):
    for mode in ["strong", "weak"]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for topo in TOPOLOGIES:
            xs, ys = [], []
            for np_val in NP_LIST:
                t = data[mode][np_val]["sample_sort"].get(topo)
                if t is not None:
                    xs.append(np_val)
                    ys.append(t)
            ax.plot(xs, ys, marker="o", linewidth=2, label=topo)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(NP_LIST)
        ax.set_xticklabels([str(x) for x in NP_LIST])
        ax.set_xlabel("Number of ranks (np)")
        ax.set_ylabel("Time (s, log)")
        title = "Strong scaling (TOTAL_N=16M)" if mode == "strong" \
                else "Weak scaling (per_rank=1M)"
        ax.set_title(f"Agnostic sample sort — {title} across topologies")
        ax.legend(title="Topology")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        path = f"{out_dir}/agnostic_{mode}_scaling.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Wrote {path}")


def plot_grid_heatmap(data, out_dir, np_val=256):
    for mode in ["strong", "weak"]:
        matrix = np.full((len(IMPLEMENTATIONS), len(TOPOLOGIES)), np.nan)
        for i, impl in enumerate(IMPLEMENTATIONS):
            for j, topo in enumerate(TOPOLOGIES):
                t = data[mode][np_val][impl].get(topo)
                if t is not None:
                    matrix[i, j] = t

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(
            matrix,
            cmap="viridis_r",
            aspect="auto",
            norm=LogNorm(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix)),
        )
        ax.set_xticks(range(len(TOPOLOGIES)))
        ax.set_xticklabels(TOPOLOGIES, rotation=20)
        ax.set_yticks(range(len(IMPLEMENTATIONS)))
        ax.set_yticklabels([IMPL_LABELS[i] for i in IMPLEMENTATIONS])
        ax.set_xlabel("Topology (network XML used by smpirun)")
        ax.set_ylabel("Implementation")
        title = "Strong" if mode == "strong" else "Weak"
        ax.set_title(
            f"{title} np={np_val} — time (s) per (implementation, topology) cell"
        )
        # Annotate cells with the timing value
        log_mid = (np.log10(np.nanmax(matrix)) + np.log10(np.nanmin(matrix))) / 2
        for i in range(len(IMPLEMENTATIONS)):
            for j in range(len(TOPOLOGIES)):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                color = "white" if np.log10(v) > log_mid else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color=color, fontsize=9)
        # Outline diagonal cells (matching-topology entries)
        for j, topo in enumerate(TOPOLOGIES):
            try:
                i = IMPLEMENTATIONS.index(f"sample_sort_{topo}")
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="red",
                                           linewidth=2.5))
            except ValueError:
                pass
        plt.colorbar(im, ax=ax, label="Time (s, log scale)")
        plt.tight_layout()
        path = f"{out_dir}/grid_{mode}_np{np_val}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Wrote {path}")


def plot_diagonal_speedup(data, out_dir, np_val=256):
    for mode in ["strong", "weak"]:
        speedups = []
        for topo in TOPOLOGIES:
            agn = data[mode][np_val]["sample_sort"].get(topo)
            opt = data[mode][np_val][f"sample_sort_{topo}"].get(topo)
            if agn is None or opt is None or opt == 0:
                speedups.append(0.0)
            else:
                speedups.append(agn / opt)

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(
            TOPOLOGIES,
            speedups,
            color=["#2ecc71" if s >= 1 else "#e74c3c" for s in speedups],
            edgecolor="black",
        )
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5,
                   label="agnostic baseline (1.0×)")
        for bar, s in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{s:.2f}×", ha="center",
                    va="bottom" if s >= 1 else "top", fontsize=11,
                    fontweight="bold")
        ax.set_xlabel("Topology (matching variant on its own network)")
        ax.set_ylabel("Speedup vs. agnostic")
        title = "Strong" if mode == "strong" else "Weak"
        ax.set_title(
            f"{title} np={np_val} — matching-topology optimization speedup over agnostic"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = f"{out_dir}/diagonal_speedup_{mode}_np{np_val}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Wrote {path}")


def plot_per_topology_scaling(data, out_dir):
    """For each topology, plot agnostic vs best-matching-opt across all np."""
    for mode in ["strong", "weak"]:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
        for ax, topo in zip(axes, TOPOLOGIES):
            for impl, label, style in [
                ("sample_sort", "agnostic", "k-o"),
                (f"sample_sort_{topo}", f"{topo}-opt", "r-s"),
            ]:
                ys, xs = [], []
                for np_val in NP_LIST:
                    t = data[mode][np_val][impl].get(topo)
                    if t is not None:
                        xs.append(np_val)
                        ys.append(t)
                ax.plot(xs, ys, style, label=label, linewidth=2, markersize=7)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(NP_LIST)
            ax.set_xticklabels([str(x) for x in NP_LIST], rotation=30)
            ax.set_title(topo)
            ax.set_xlabel("np")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Time (s, log)")
        title = "Strong" if mode == "strong" else "Weak"
        fig.suptitle(
            f"{title} scaling — agnostic vs. matching-topology optimization, per topology",
            fontsize=14,
        )
        plt.tight_layout()
        path = f"{out_dir}/per_topology_{mode}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Wrote {path}")


def main():
    out_dir = "results/plots"
    os.makedirs(out_dir, exist_ok=True)
    data = load_data()

    plot_agnostic_scaling(data, out_dir)
    plot_grid_heatmap(data, out_dir, np_val=256)
    plot_diagonal_speedup(data, out_dir, np_val=256)
    plot_per_topology_scaling(data, out_dir)

    n = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    print(f"\n{n} plots saved to {out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
