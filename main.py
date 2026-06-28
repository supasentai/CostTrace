from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PHASES = {
    "prepare": [
        Path("src/costtrace/preparation/audit.py"),
        Path("src/costtrace/preparation/curation.py"),
        Path("src/costtrace/preparation/graph.py"),
        Path("src/costtrace/preparation/profile.py"),
    ],
    "metrics": [
        Path("src/costtrace/analysis/topology.py"),
        Path("src/costtrace/analysis/centrality.py"),
        Path("src/costtrace/analysis/community.py"),
        Path("src/costtrace/analysis/risk.py"),
    ],
    "model": [
        Path("src/costtrace/modeling/graphsage.py"),
    ],
    "budget": [
        Path("src/costtrace/intervention/allocation.py"),
        Path("src/costtrace/intervention/counterfactual.py"),
        Path("src/costtrace/intervention/simulation.py"),
        Path("src/costtrace/intervention/evaluation.py"),
    ],
    "notebooks": [
        Path("src/costtrace/reporting/notebooks.py"),
    ],
}


def run_script(script_path: Path) -> None:
    script_path = ROOT / script_path
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n=== Running {script_path.relative_to(ROOT)} ===", flush=True)
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )
    subprocess.run([sys.executable, str(script_path)], cwd=ROOT, check=True, env=env)


def selected_scripts(phase: str) -> list[Path]:
    if phase == "all":
        ordered = []
        for key in ["prepare", "metrics", "model", "budget", "notebooks"]:
            ordered.extend(PHASES[key])
        return ordered
    return PHASES[phase]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CostTrace analysis pipeline.")
    parser.add_argument(
        "--phase",
        choices=["all", *PHASES.keys()],
        default="all",
        help="Pipeline section to run.",
    )
    args = parser.parse_args()

    for script in selected_scripts(args.phase):
        run_script(script)

    print("\nCostTrace pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
