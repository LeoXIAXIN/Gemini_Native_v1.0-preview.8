#!/usr/bin/env python3
"""Run the maintained src-only verification suites in their proper runtimes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
GMR_PYTHON = ROOT / "runtime-gmr" / "python.exe"
HGPT_PYTHON = ROOT / "runtime-hgpt" / "python.exe"

GMR_SUITES = (
    "audit_src_layout.py",
    "test_chingmu_watchdog.py",
    "test_domain.py",
    "test_application_controller.py",
    "test_http_disconnect.py",
    "test_g1_versions.py",
    "test_sdk_model_integration.py",
    "test_gmr_headless.py",
    "test_remote_handover.py",
    "test_sim_supervisor_regression.py",
    "test_safety_runtime.py",
)
HGPT_SUITES = (
    "test_hgpt_runtime.py",
    "test_keyboard_cmd.py",
    "test_state_store_redis_compat.py",
    "test_unitree_hg_compat.py",
    "test_unitree_official_probe.py",
    "test_unitree_official_control.py",
    "test_unitree_thread.py",
)


def run(interpreter: Path, script: str) -> bool:
    print(f"\n===== {script} =====", flush=True)
    environment = dict(os.environ)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})
    result = subprocess.run(
        [str(interpreter), str(TOOLS / script)],
        cwd=str(ROOT),
        env=environment,
        stdin=subprocess.DEVNULL,
    )
    print(f"----- exit code {result.returncode} -----", flush=True)
    return result.returncode == 0


def main() -> int:
    failures: list[str] = []
    for interpreter in (GMR_PYTHON, HGPT_PYTHON):
        if not interpreter.is_file():
            failures.append(f"missing runtime: {interpreter.relative_to(ROOT)}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    for script in GMR_SUITES:
        if not run(GMR_PYTHON, script):
            failures.append(script)
    for script in HGPT_SUITES:
        if not run(HGPT_PYTHON, script):
            failures.append(script)

    print("\n========================================")
    if failures:
        print("FAILED SUITES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("ALL SRC-ONLY CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
