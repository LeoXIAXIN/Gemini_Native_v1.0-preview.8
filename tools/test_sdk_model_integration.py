"""Offline regression for model migration, launch/ARM binding, and trace identity."""

from pathlib import Path
import json
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.application.authorization import (
    RealOutputAuthorizationBroker, RealOutputAuthorizationError,
    binding_from_control_config, ENV_BINDING, ENV_CAPABILITY, ENV_SESSION, ENV_TOKEN,
)
from src.domain.models import AuthorizationContext
from src.domain.process_spec import build_real_pipeline_command, build_sim_pipeline_command
from src.adapters.runtime import default_pipeline_paths
from src.infrastructure.config import FROZEN_DEFAULTS, validate_config
from src.motion.official_standup_trace import OfficialStandupTrace, OfficialStandupTraceError


def main():
    config = validate_config({
        "g1_version": "5010", "execution_mode": "real_only", "debug_mode": False,
        "server_ip": "192.168.123.100", "skeleton_id": 1,
    }, dict(FROZEN_DEFAULTS))
    assert "g1_version" not in config
    assert config["model_profile"] == "unitree_g1_5010"
    config["startup_handover_enabled"] = True
    binding = binding_from_control_config(config)
    assert binding == AuthorizationContext.from_control_config(config).to_dict()
    assert "mode_machine" not in binding and "g1_version" not in binding

    # Use the real broker's stage and binding validation without opening a socket.
    broker = RealOutputAuthorizationBroker()
    broker._ensure_started = lambda: None
    broker._endpoint = "offline-test-only"
    env = broker.issue(binding)
    request = {
        "capability": env[ENV_CAPABILITY], "session": env[ENV_SESSION],
        "token": env[ENV_TOKEN], "binding": json.loads(env[ENV_BINDING]),
        "stage": "launcher",
    }
    wrong_binding = dict(binding, model_profile="unitree_g1_4010")
    try:
        broker._consume(dict(request, binding=wrong_binding))
    except RealOutputAuthorizationError:
        pass
    else:
        raise AssertionError("ARM allowed the model asset to change")
    for stage in ("launcher", "prepare_runner", "runner"):
        request["stage"] = stage
        result = broker._consume(request)
        assert result["consumed"]
        if "next_token" in result:
            request["token"] = result["next_token"]
    try:
        broker._consume(request)
    except RealOutputAuthorizationError:
        pass
    else:
        raise AssertionError("spent ARM capability was reusable")

    paths = default_pipeline_paths()
    for builder in (build_real_pipeline_command, build_sim_pipeline_command):
        cmd = builder(paths, config).command
        assert "--g1-version" not in cmd
        assert cmd[cmd.index("--model-profile") + 1] == config["model_profile"]

    # Preserve legacy trace asset labels/hashes, without deriving mode from them.
    q = np.zeros((100, 29), dtype=np.float32)
    q[:, 3] = np.linspace(1., 0., 100)
    q[:, 9] = np.linspace(1., 0., 100)
    trace = OfficialStandupTrace(
        robot_variant="5010", mode_machine=5, firmware_id="synthetic-test",
        timestamps=np.arange(100) * .02, joint_qpos=q,
        joint_qvel=np.zeros((100, 29)),
        root_quat_wxyz=np.tile([1., 0., 0., 0.], (100, 1)),
        root_gyro=np.zeros((100, 3)), fsm_id=np.full(100, 706), metadata={},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.npz"
        trace.save(path)
        loaded = OfficialStandupTrace.load(path, expected_robot_variant="5010")
        assert loaded.mode_machine == 5 and loaded.content_sha256 == trace.content_sha256
        try:
            OfficialStandupTrace.load(path, expected_robot_variant="4010")
        except OfficialStandupTraceError:
            pass
        else:
            raise AssertionError("trace switched model assets silently")
    print("SDK/MODEL MIGRATION AND AUTHORIZATION INTEGRATION PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
