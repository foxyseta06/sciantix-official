#!/usr/bin/env python3
"""Run the post-processing of the JOG model (validation/jog/PHENIXpins/plot_JOG.py) on the JOG ANN cases.

plot_JOG.py is used unchanged: only its case, plot and summary paths are redirected.

    python3 plot_JOG_ANN.py                      # cases in validation/jog_ann (JOG ANN)
    python3 plot_JOG_ANN.py --cases DIR --out DIR  # any folder of test_PHENIXpins_point_* cases (e.g. OpenCalphad at 1 bar)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOT_JOG = HERE.parents[1] / "jog" / "PHENIXpins" / "plot_JOG.py"


def load_plot_jog():
    spec = importlib.util.spec_from_file_location("plot_JOG", PLOT_JOG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=HERE.parent, help="folder with the test_PHENIXpins_point_* cases")
    parser.add_argument("--out", type=Path, default=HERE, help="folder for plots/ and results_summary.txt")
    args, rest = parser.parse_known_args()

    plot_jog = load_plot_jog()
    plot_jog.CASES_DIR = args.cases.resolve()
    plot_jog.PLOTS_DIR = args.out.resolve() / "plots"
    plot_jog.RESULTS_SUMMARY = args.out.resolve() / "results_summary.txt"
    sys.argv = [str(PLOT_JOG)] + rest
    return plot_jog.main()


if __name__ == "__main__":
    sys.exit(main())
