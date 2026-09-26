#!/usr/bin/env python3
"""Load the released ONNX policy and step both supported G1 MuJoCo models."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.motion.sim_mj import MJSim  # noqa: E402
from src.motion.tracking_policy import Args, MLP_Policy_ONNX  # noqa: E402


def test_policy() -> None:
    model_path = ROOT / "storage" / "ckpts" / "pns_wo_priv216.onnx"
    assert model_path.is_file(), model_path
    policy = MLP_Policy_ONNX(Args(device="cpu", load_path=str(model_path)))
    input_meta = policy.onnx_model.get_inputs()[0]
    obs_shape = tuple(
        value if isinstance(value, int) and value > 0 else 1
        for value in input_meta.shape
    )
    action = policy.infer(np.zeros(obs_shape, dtype=np.float32))
    assert action.shape == (1, 29), action.shape
    assert np.all(np.isfinite(action))


def test_mujoco(version: str) -> None:
    xml = ROOT / "storage" / "assets" / f"unitree_g1_{version}" / "scene_mjx_track.xml"
    assert xml.is_file(), xml
    simulation = MJSim(str(xml), ctrl_dt=0.02, sim_dt=0.001, headless=True)
    assert simulation.mj_model.nu == 29
    assert simulation.mj_model.nq == 36
    simulation.kps = np.ones(29, dtype=np.float64)
    simulation.kds = np.full(29, 0.1, dtype=np.float64)
    simulation.torque_limit = np.full(29, 5.0, dtype=np.float64)
    simulation.init_qpos = np.zeros(36, dtype=np.float64)
    simulation.init_qpos[2] = 0.8
    simulation.init_qpos[3] = 1.0
    state = simulation.reset(simulation.init_state())
    state = simulation.step(state, np.zeros(29, dtype=np.float64))
    assert np.all(np.isfinite(state.mj_data.qpos))
    print(f"  G1 {version}: MuJoCo model loaded and stepped")


def main() -> int:
    test_policy()
    for version in ("4010", "5010"):
        test_mujoco(version)
    print("HUMANOID-GPT RUNTIME TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
