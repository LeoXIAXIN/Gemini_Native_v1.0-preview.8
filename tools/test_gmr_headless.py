#!/usr/bin/env python3
"""Regression checks for Windows headless GMR imports."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.gmr_headless import load_general_motion_retargeting  # noqa: E402


def main() -> int:
    assert "general_motion_retargeting" not in sys.modules
    assert "mujoco._simulate" not in sys.modules

    retargeter_type = load_general_motion_retargeting()
    retargeter = retargeter_type(
        src_human="bvh_nokov",
        tgt_robot="unitree_g1",
        actual_human_height=1.75,
        verbose=False,
    )

    assert retargeter.model.nq == 36
    assert retargeter_type.__module__ == "_chingmu_gmr_headless.motion_retarget"
    assert "general_motion_retargeting" not in sys.modules
    assert "mujoco._simulate" not in sys.modules

    # Importing bridge definitions must remain headless as well.  The bridge
    # process creates GMR only in main(), after its local state endpoint exists.
    from src.motion import gmr_bridge  # noqa: F401, PLC0415

    assert "mujoco._simulate" not in sys.modules

    native_source = (ROOT / "src" / "controller" / "native_app.py").read_text(
        encoding="utf-8"
    )
    assert "load_general_motion_retargeting" in native_source
    assert "from general_motion_retargeting import GeneralMotionRetargeting" not in native_source

    live_sim_source = (ROOT / "src" / "motion" / "live_sim.py").read_text(
        encoding="utf-8"
    )
    assert "continuing headless" in live_sim_source
    assert "mj_sim.headless = True" in live_sim_source
    print("WINDOWS HEADLESS GMR IMPORT TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
