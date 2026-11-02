#!/usr/bin/env python3
"""Create (and optionally run) the OpenCalphad PHENIX cases at 1 bar, the consistent reference for the JOG ANN.

The JOG ANN is trained at 1 bar. In this database only the gas depends on pressure, so OpenCalphad at 1 bar
isolates the error of the surrogate from the effect of pressure. The cases are copies of
validation/jog/test_PHENIXpins_point_*; the only change is the system pressure column of input_history.txt.

    python3 make_oc_1bar_cases.py OUT_DIR [--run]
"""
import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
JOG_CASES = REPO / "validation" / "jog"
PRESSURE_COLUMN = 4  # time, temperature, fission rate, hydrostatic stress, system pressure (Pa), O/M


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--run", action="store_true", help="run sciantix.x on each case (OpenCalphad build)")
    args = parser.parse_args()

    for source in sorted(JOG_CASES.glob("test_PHENIXpins_point_*")):
        case = args.out / source.name
        case.mkdir(parents=True, exist_ok=True)
        for f in source.glob("input_*.txt"):
            shutil.copy(f, case / f.name)
        rows = [line.split() for line in (source / "input_history.txt").read_text().splitlines() if line.strip()]
        for row in rows:
            row[PRESSURE_COLUMN] = "1.00e+05"
        (case / "input_history.txt").write_text("\n".join("   ".join(row) for row in rows) + "\n")
        print("created", case)
        if args.run:
            subprocess.run([str(REPO / "build" / "sciantix.x"), str(case) + "/"], check=True,
                           stdout=open(case / "run.log", "w"), stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
