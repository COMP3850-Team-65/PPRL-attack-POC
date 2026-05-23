from datetime import datetime
import subprocess
from pathlib import Path

from pprl_attack.config import RUNS_DIR


def log_run(notebook, **metrics):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(RUNS_DIR.glob("run*.txt"))
    run_num = len(existing) + 1

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    lines = [
        f"Run {run_num:03d}",
        f"Notebook: {notebook}",
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Git: {sha}",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    path = RUNS_DIR / f"run{run_num:03d}.txt"
    path.write_text("\n".join(lines))
    print(f"Run log: {path}")
