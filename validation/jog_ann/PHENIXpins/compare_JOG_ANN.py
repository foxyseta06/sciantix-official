#!/usr/bin/env python3
"""Compare the JOG ANN with OpenCalphad on the PHENIX cases, from the SCIANTIX outputs only.

Sets compared:
  OC      validation/jog/test_PHENIXpins_point_*        (OpenCalphad, system pressure: the reference of the JOG model)
  OC 1bar folder given with --oc1 (see make_oc_1bar_cases.py): OpenCalphad at 1 bar, the pressure of the training data
  JOG ANN validation/jog_ann/test_PHENIXpins_point_*    (fission_products.module = NEURALNETWORK)

The JOG thickness is read from the results_summary.txt written by plot_JOG.py (OC) and plot_JOG_ANN.py (ANN, OC 1bar):
run those first. output.txt is used when present, otherwise output_gold.txt.

    python3 compare_JOG_ANN.py [--oc1 DIR]
Writes plots/compare_*.png and compare_summary.txt next to this script.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOG = HERE.parents[1] / "jog"
COLORS = {"OC": "#1C5CAB", "OC 1bar": "#8A8F98", "JOG ANN": "#E0702E"}
COOLDOWN_START_H = 25200.0


def output(case_dir: Path) -> pd.DataFrame:
    f = case_dir / "output.txt"
    if not f.exists():
        f = case_dir / "output_gold.txt"
    return pd.read_csv(f, sep="\t")


def cases(folder: Path) -> dict:
    return {d.name.split("_")[3]: d for d in sorted(folder.glob("test_PHENIXpins_point_*")) if d.is_dir()}


def thickness(summary: Path) -> list:
    """Oxides pre/post cooldown, oxides + metallics pre/post cooldown (um), as written by plot_JOG.py."""
    if not summary.exists():
        return []
    return [float(x) for x in re.findall(r"total = ([0-9.]+) um", summary.read_text())][:4]


def jog_total(df: pd.DataFrame) -> pd.Series:
    return df[[c for c in df.columns if c.startswith("JOG (")]].sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oc1", type=Path, help="folder with the OpenCalphad cases at 1 bar (already run)")
    args = parser.parse_args()

    sets = {"OC": (cases(JOG), JOG / "PHENIXpins" / "results_summary.txt")}
    if args.oc1:
        sets["OC 1bar"] = (cases(args.oc1), args.oc1 / "results_summary.txt")
    sets["JOG ANN"] = (cases(HERE.parent), HERE / "results_summary.txt")
    data = {name: {k: output(d) for k, d in c.items()} for name, (c, _) in sets.items()}
    (HERE / "plots").mkdir(exist_ok=True)
    lines = []

    # 1. JOG thickness
    labels = ["Oxides\nend of irradiation", "Oxides\nafter cooldown",
              "Oxides + metallics\nend of irradiation", "Oxides + metallics\nafter cooldown"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    names = [n for n in sets if thickness(sets[n][1])]
    w = 0.8 / max(len(names), 1)
    lines.append("JOG thickness (um): " + " | ".join(l.replace("\n", " ") for l in labels))
    for j, name in enumerate(names):
        t = thickness(sets[name][1])
        x = np.arange(4) + (j - (len(names) - 1) / 2) * w
        ax.bar(x, t, w * 0.92, color=COLORS[name], label=name)
        for xi, v in zip(x, t):
            ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=9)
        lines.append(f"  {name:8s} " + "  ".join(f"{v:6.1f}" for v in t))
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("JOG thickness (µm)")
    ax.legend(loc="upper left", ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "plots" / "compare_thickness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Total JOG volume fraction at each point
    lines.append("\nTotal JOG volume fraction (/): end of irradiation | end of history")
    for k in sorted(data["OC"]):
        row = []
        for name in data:
            df = data[name][k]
            i_end = np.searchsorted(df["Time (h)"].values, COOLDOWN_START_H, side="right") - 1
            tot = jog_total(df)
            row.append(f"{name} {tot.iloc[i_end]:.4f} | {tot.iloc[-1]:.4f}")
        lines.append(f"  point {k}: " + ";  ".join(row))

    # 3. Released caesium at point 02
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ref = data["OC"]["02"]
    produced = ref["Cs produced (at/m3)"].clip(lower=1.0)
    for name in data:
        df = data[name]["02"]
        p = np.interp(df["Time (h)"], ref["Time (h)"], produced)
        m = df["Time (h)"] < COOLDOWN_START_H
        ax.plot(df["Time (h)"][m] / 1000, 100 * df["Cs released (at/m3)"][m] / p[m],
                "--" if name == "JOG ANN" else "-", color=COLORS[name], lw=2.2, label=name)
    ax.set_xlabel("Time (10³ h)")
    ax.set_ylabel("Cs released / Cs produced (%)")
    ax.set_title("Point 02", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "plots" / "compare_cs_released_02.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Phases at sharp boundaries during the cooldown
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    lines.append("\nEnd of history:")
    for ax, k, col in ((axs[0], "03", "JOG (Cs2MoO4) (/)"), (axs[1], "04", "JOG (Ba3MoO6) (/)")):
        for name in data:
            df = data[name][k]
            m = (df["Time (h)"] > COOLDOWN_START_H - 10) & (df["Temperature (K)"] < 1300)
            ax.plot(df["Temperature (K)"][m], df[col][m], "--" if name == "JOG ANN" else "-",
                    color=COLORS[name], lw=2.2, label=name)
            lines.append(f"  point {k} {col}: {name} {df[col].iloc[-1]:.4f}")
        ax.invert_xaxis()
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Temperature during cooldown (K)")
        ax.set_ylabel(col)
        ax.set_title(f"Point {k}", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False)
    fig.savefig(HERE / "plots" / "compare_cooldown.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    text = "\n".join(lines) + "\n"
    (HERE / "compare_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
