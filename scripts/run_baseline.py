from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    inference_path = PROJECT_ROOT / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError("Root inference.py is required but was not found.")

    command = [sys.executable, str(inference_path)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if completed.returncode == 0:
        result_file = PROJECT_ROOT / "baseline_results.json"
        legacy_copy = PROJECT_ROOT / "scripts" / "baseline_results.json"
        if result_file.exists():
            shutil.copyfile(result_file, legacy_copy)


if __name__ == "__main__":
    main()
