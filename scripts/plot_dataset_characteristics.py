#!/usr/bin/env python3
"""Generate the dataset-characteristics figure.

Produces a 2x2 panel figure:
  A. Number of instances per image
  B. Number of categories per image
  C. Dominance per image
  D. Relative bounding-box size (with two zoom insets connected by brackets)

Usage:
    python scripts/plot_dataset_characteristics.py \
        --data data/WIO-ReefFish --out figures
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# ── Paths (set from the CLI in main()) ───────────────────────────────────
DATA = Path()
OUT = Path()

# ── Color palette ────────────────────────────────────────────────────────
C_MAIN = "#2166AC"       # deep blue — primary
C_MID = "#67A9CF"        # medium blue — zoom 1
C_LIGHT = "#D1E5F0"      # pale blue — zoom 2
C_ACCENT = "#B2182B"     # dark red — annotation lines
C_FILL = "#F7F7F7"       # near-white panel background
C_EDGE = "white"

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": False,
})


def load_dataset():
    """Load all annotations across splits, return per-image records."""
    records = []
    for split in ("train", "valid", "test"):
        label_dir = DATA / split / "labels"
        for lf in sorted(label_dir.glob("*.txt")):
            lines = [l.strip() for l in lf.read_text().strip().split("\n") if l.strip()]
            classes, areas = [], []
            for l in lines:
                parts = l.split()
                if len(parts) >= 5:
                    classes.append(int(float(parts[0])))
                    w, h = float(parts[3]), float(parts[4])
                    areas.append(w * h * 100)  # as % of image area
            if not classes:
                continue
            counts = Counter(classes)
            records.append({
                "n_inst": len(classes),
                "n_cat": len(set(classes)),
                "dominance": max(counts.values()) / len(classes) * 100,
                "areas": areas,
            })
    return records


def _add_stat_line(ax, val, label, vertical=True, position="right",
                   fontsize=7.5, offset_x=0, offset_y=0):
    """Add a dashed annotation line with label."""
    if vertical:
        ax.axvline(val, color=C_ACCENT, linewidth=0.9, linestyle="--",
                   alpha=0.75, zorder=5)
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.90 + offset_y
        ha = "left" if position == "right" else "right"
        x_off = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
        x_pos = val + (x_off if position == "right" else -x_off) + offset_x
        ax.text(x_pos, y_pos, label, color=C_ACCENT, fontsize=fontsize,
                va="top", ha=ha, fontstyle="italic")


def _panel_label(ax, letter, x=-0.12, y=1.08):
    """Add bold panel letter (A, B, C, D)."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
            fontfamily="sans-serif")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="Dataset root containing train/valid/test label folders")
    parser.add_argument("--out", type=Path, default=Path("figures"),
                        help="Output directory for the figure (default: figures/)")
    args = parser.parse_args()

    global DATA, OUT
    DATA, OUT = args.data, args.out

    records = load_dataset()
    all_areas = [a for r in records for a in r["areas"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(hspace=0.48, wspace=0.35, left=0.10, right=0.96,
                        top=0.94, bottom=0.08)

    # ── A. Instances per image ───────────────────────────────────────
    ax = axes[0, 0]
    _panel_label(ax, "A")
    n_inst = [r["n_inst"] for r in records]
    max_val = max(n_inst)
    bins = np.arange(0.5, max_val + 1.5, 1)
    ax.hist(n_inst, bins=bins, color=C_MAIN, edgecolor=C_EDGE,
            linewidth=0.3, rwidth=0.85)
    ax.set_xlabel("Instances per image")
    ax.set_ylabel("Number of images")
    ax.set_xlim(0, max_val + 1)
    _add_stat_line(ax, np.mean(n_inst), f"mean = {np.mean(n_inst):.1f}")

    # ── B. Categories per image ──────────────────────────────────────
    ax = axes[0, 1]
    _panel_label(ax, "B")
    n_cat = [r["n_cat"] for r in records]
    cat_counts = Counter(n_cat)
    x_vals = sorted(cat_counts.keys())
    y_vals = [cat_counts[x] for x in x_vals]
    ax.bar(x_vals, y_vals, color=C_MAIN, edgecolor=C_EDGE,
           linewidth=0.3, width=0.7)
    ax.set_xlabel("Families per image")
    ax.set_ylabel("Number of images")
    ax.set_xticks(x_vals)
    _add_stat_line(ax, np.mean(n_cat), f"mean = {np.mean(n_cat):.1f}")

    # ── C. Dominance per image ───────────────────────────────────────
    ax = axes[1, 0]
    _panel_label(ax, "C")
    dom = [r["dominance"] for r in records]
    bins_dom = np.arange(0, 105, 5)
    ax.hist(dom, bins=bins_dom, color=C_MAIN, edgecolor=C_EDGE,
            linewidth=0.3, rwidth=0.88)
    ax.set_xlabel("Most abundant family per image (%)")
    ax.set_ylabel("Number of images")
    ax.set_xlim(0, 105)
    _add_stat_line(ax, np.mean(dom), f"mean = {np.mean(dom):.1f}%",
                   position="left")

    # ── D. Relative bounding-box size ────────────────────────────────
    ax = axes[1, 1]
    _panel_label(ax, "D")
    areas_arr = np.array(all_areas)
    median_area = np.median(areas_arr)

    # Main histogram (0–5%)
    bins_area = np.arange(0, 5.25, 0.25)
    ax.hist(areas_arr[areas_arr <= 5], bins=bins_area, color=C_MAIN,
            edgecolor=C_EDGE, linewidth=0.3, rwidth=0.9)
    ax.set_xlabel("Bounding-box area (% of image)")
    ax.set_ylabel("Number of instances")
    ax.set_xlim(0, 5)
    _add_stat_line(ax, median_area, f"median = {median_area:.2f}%",
                   fontsize=6.5)

    # Inset 1: zoom 0–0.5%
    ax_z1 = ax.inset_axes([0.32, 0.35, 0.65, 0.60])
    bins_z1 = np.arange(0, 0.52, 0.025)
    ax_z1.hist(areas_arr[areas_arr <= 0.5], bins=bins_z1,
               color=C_MID, edgecolor=C_MAIN, linewidth=0.25)
    ax_z1.set_xlim(0, 0.5)
    ax_z1.set_title("0 – 0.5 %", fontsize=7.5, pad=3, fontstyle="italic")
    ax_z1.tick_params(labelsize=6)
    ax_z1.set_ylabel("")
    ax_z1.set_xlabel("")
    for spine in ax_z1.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#555555")
    ax_z1.patch.set_facecolor("white")
    ax_z1.patch.set_alpha(0.95)

    # Inset 2: zoom 0–0.1% (inside inset 1)
    ax_z2 = ax_z1.inset_axes([0.48, 0.35, 0.48, 0.58])
    bins_z2 = np.arange(0, 0.105, 0.005)
    ax_z2.hist(areas_arr[areas_arr <= 0.1], bins=bins_z2,
               color=C_LIGHT, edgecolor=C_MID, linewidth=0.25)
    ax_z2.set_xlim(0, 0.1)
    ax_z2.set_title("0 – 0.1 %", fontsize=6.5, pad=2, fontstyle="italic")
    ax_z2.tick_params(labelsize=5)
    ax_z2.set_ylabel("")
    ax_z2.set_xlabel("")
    for spine in ax_z2.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#888888")
    ax_z2.patch.set_facecolor("white")
    ax_z2.patch.set_alpha(0.95)

    # Shaded zoom regions
    rect1 = mpatches.Rectangle((0, 0), 0.5, ax.get_ylim()[1],
                                linewidth=0.6, edgecolor=C_MID,
                                facecolor=C_MID, alpha=0.06, zorder=0)
    ax.add_patch(rect1)
    rect2 = mpatches.Rectangle((0, 0), 0.1, ax_z1.get_ylim()[1],
                                linewidth=0.5, edgecolor=C_LIGHT,
                                facecolor=C_LIGHT, alpha=0.12, zorder=0)
    ax_z1.add_patch(rect2)

    # ── Save ─────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT / "dataset_characteristics.pdf"
    out_png = OUT / "dataset_characteristics.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
