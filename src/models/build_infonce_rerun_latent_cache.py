from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.infonce_rerun_viz_common import (
    BASELINE_CHECKPOINT,
    DATA_ROOT,
    INFO_NCE_CHECKPOINT,
    OUTPUT_DIR,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "models" / "analyze_latent_comparison.py"),
        "--data-root",
        str(DATA_ROOT),
        "--baseline-checkpoint",
        str(BASELINE_CHECKPOINT),
        "--text-checkpoint",
        str(INFO_NCE_CHECKPOINT),
        "--samples-per-class",
        "180",
        "--num-workers",
        "0",
        "--output-dir",
        str(OUTPUT_DIR),
        "--cache-only",
    ]
    env = {
        **os.environ,
        "MPLCONFIGDIR": "/private/tmp/mpl",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
