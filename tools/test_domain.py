#!/usr/bin/env python3
"""Domain and codec tests (read-only; run with runtime-gmr python.exe).

Verifies:
  * round-trip of the CHINGMU frame pickle packet produced by the src adapter;
  * round-trip of the ``action_qpos_g1_packet`` / ``action_mimic_g1_packet``
    JSON formats;
  * safety telemetry decoding with fail-closed behavior;
  * domain-model validation errors.

This test never writes files, never starts processes, never touches config.
"""

from __future__ import annotations

import json
import os
import sys
import pickle

sys.dont_write_bytecode = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from src.domain import (  # noqa: E402
    G1Reference,
    LegacyCodec,
    MocapFrame,
    MotorCommand,
    SafeCommand,
    SafetyStatus,
    ValidationError,
    CodecError,
)
from src.domain.models import (  # noqa: E402
    G1_QPOS_SIZE,
    MIMIC_TARGET_SIZE,
    AuthorizationContext,
)
from src.adapters.chingmu import build_frame_packet  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [ok]   {name}")
    else:
        FAILURES.append(f"{name}: {detail or 'assertion failed'}")
        print(f"  [FAIL] {name}: {detail or 'assertion failed'}")


def expect_raises(name: str, exc_type: type, fn) -> None:
    try:
        fn()
    except exc_type as exc:
        print(f"  [ok]   {name} ({type(exc).__name__})")
        return
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{name}: raised {type(exc).__name__} instead of {exc_type.__name__}")
        print(f"  [FAIL] {name}: raised {type(exc).__name__} instead of {exc_type.__name__}")
        return
    FAILURES.append(f"{name}: did not raise {exc_type.__name__}")
    print(f"  [FAIL] {name}: did not raise {exc_type.__name__}")


def synthetic_frame(segments: int = 63, frame_id: int = 12345):
    rng = np.random.default_rng(7)
    positions = rng.uniform(-500.0, 500.0, size=(segments, 3))
    rotations = rng.uniform(-1.0, 1.0, size=(segments, 4))
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    detected = (rng.uniform(size=(segments,)) > 0.2).astype(int)
    return frame_id, positions, rotations, detected


def main() -> int:
    print("Domain and wire-codec tests")

    # --- 1. frame packet: adapter producer -> codec -> byte-identical re-encode
    frame_id, positions, rotations, detected = synthetic_frame()
    legacy_payload = build_frame_packet(
        frame_id,
        positions.flatten().tolist(),
        rotations.flatten().tolist(),
        detected.tolist(),
    )
    frame = LegacyCodec.decode_chingmu_frame_packet(legacy_payload)
    check("frame decode: frame_id", frame.frame_id == frame_id)
    check("frame decode: segment count", frame.segment_count == 63)
    check(
        "frame decode: positions match",
        np.allclose(frame.positions, positions),
    )
    check(
        "frame decode: rotations match",
        np.allclose(frame.rotations, rotations),
    )
    check(
        "frame decode: detected match",
        np.array_equal(frame.detected, detected.astype(bool)),
    )
    reencoded = LegacyCodec.encode_chingmu_frame_packet(frame)
    check("frame re-encode: byte-identical to adapter producer",
          reencoded == legacy_payload)

    # --- 2. qpos packet round-trip (legacy key order preserved semantically)
    qpos = np.concatenate(
        [
            np.array([0.0, 0.0, 0.793]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(29),
        ]
    )
    legacy_meta = {
        "schema_version": 1,
        "session_id": "0123456789abcdef",
        "sequence": 42,
        "frame_id": frame_id,
        "generated_monotonic": 100.25,
        "valid_until_monotonic": 100.33,
        "robot_model": "unitree_g1_29dof",
        "dof_order_hash": "deadbeef",
    }
    legacy_qpos_text = json.dumps({**legacy_meta, "monotonic": 100.25,
                                   "qpos": qpos.tolist()})
    ref = LegacyCodec.decode_gmr_qpos_packet(legacy_qpos_text)
    check("qpos packet: session/sequence",
          ref.session_id == "0123456789abcdef" and ref.sequence == 42)
    check("qpos packet: qpos 36 values", ref.qpos is not None and ref.qpos.shape == (36,))
    check("qpos packet: dof_position 29", ref.dof_position.shape == (29,))
    check("qpos packet: fresh at t=100.30", ref.is_fresh(100.30))
    check("qpos packet: stale at t=100.40", not ref.is_fresh(100.40))
    check("qpos packet: age", abs(ref.age_seconds(100.35) - 0.1) < 1e-9)
    requantized = json.loads(LegacyCodec.encode_gmr_qpos_packet(ref))
    expected = json.loads(legacy_qpos_text)
    check("qpos packet: re-encode equals legacy JSON",
          requantized == expected)

    # --- 3. mimic packet (33 values, no qpos) round-trip
    mimic = np.linspace(-0.2, 0.2, MIMIC_TARGET_SIZE).astype(np.float32)
    legacy_mimic_text = json.dumps({**legacy_meta, "mimic": mimic.tolist()})
    mimic_ref = LegacyCodec.decode_gmr_mimic_packet(legacy_mimic_text)
    check("mimic packet: mimic 33 values",
          mimic_ref.mimic is not None and mimic_ref.mimic.shape == (33,))
    check("mimic packet: no qpos", mimic_ref.qpos is None)
    check("mimic packet: round-trip",
          json.loads(LegacyCodec.encode_gmr_mimic_packet(mimic_ref))
          == json.loads(legacy_mimic_text))

    # --- 4. safety telemetry decode (fail-closed)
    telemetry = {
        "schema_version": 1,
        "state": "TELEOP",
        "reason": "teleop active",
        "estop_latched": False,
        "fault_latched": False,
        "reference_health": "LIVE",
        "state_source": "mujoco",
        "real_hardware_allowed": False,
        "teleop_weight": 0.9,
        "reference_age_ms": 12.5,
        "strategy_enabled": True,
        "debug_mode": False,
        "bypass_active": False,
        "hard_guards_enabled": True,
        "limited": False,
    }
    status = LegacyCodec.decode_safety_status(telemetry)
    check("safety: decode ok + healthy", status.healthy)
    estop = dict(telemetry, state="E_STOP", estop_latched=True)
    check("safety: estop latched -> not healthy",
          not LegacyCodec.decode_safety_status(estop).healthy)
    bad = dict(telemetry, state="FLYING")
    expect_raises("safety: unknown state rejected",
                  CodecError, lambda: LegacyCodec.decode_safety_status(bad))
    bad_weight = dict(telemetry, teleop_weight=1.5)
    expect_raises("safety: weight out of range rejected",
                  CodecError, lambda: LegacyCodec.decode_safety_status(bad_weight))

    # --- 5. safety command encode
    estop_cmd = SafeCommand(sequence=7, action="estop")
    check("safe command: estop payload",
          LegacyCodec.encode_safe_command(estop_cmd)
          == {"sequence": 7, "action": "estop"})
    reset_cmd = SafeCommand(sequence=8, action="reset",
                            operator_confirmed=True,
                            physical_estop_released=True)
    check("safe command: reset payload",
          LegacyCodec.encode_safe_command(reset_cmd)
          == {"sequence": 8, "action": "reset",
              "operator_confirmed": True, "physical_estop_released": True})
    expect_raises("safe command: reset without confirmations rejected",
                  ValidationError,
                  lambda: SafeCommand(sequence=9, action="reset"))
    decoded_reset = LegacyCodec.decode_safe_command(
        json.dumps(LegacyCodec.encode_safe_command(reset_cmd))
    )
    check(
        "safe command: decode round-trip",
        decoded_reset.sequence == reset_cmd.sequence
        and decoded_reset.action == reset_cmd.action
        and decoded_reset.operator_confirmed == reset_cmd.operator_confirmed
        and decoded_reset.physical_estop_released
        == reset_cmd.physical_estop_released,
    )

    # --- 6. model validation errors
    expect_raises("model: MocapFrame bad shape",
                  ValidationError,
                  lambda: MocapFrame(frame_id=1,
                                     positions=np.zeros((3, 2)),
                                     rotations=np.zeros((3, 4)),
                                     detected=np.zeros(3, dtype=bool)))
    expect_raises("model: G1Reference wrong qpos size",
                  ValidationError,
                  lambda: G1Reference(
                      session_id="s", sequence=1, frame_id=1,
                      generated_monotonic=1.0, valid_until_monotonic=1.1,
                      robot_model="unitree_g1_29dof", dof_order_hash="h",
                      qpos=np.zeros(35)))
    expect_raises("model: MotorCommand wrong dof count",
                  ValidationError,
                  lambda: MotorCommand(target_qpos=np.zeros(28),
                                       timestamp_monotonic=1.0))

    # --- 7. authorization context contains every safety binding field
    config = {
        "backend": "humanoid_gpt",
        "execution_mode": "real_only",
        "debug_mode": False,
        "startup_handover_enabled": True,
        "server_ip": "127.0.0.1",
        "skeleton_id": 0,
        "model_profile": "unitree_g1_5010",
        "unitree_network_interface": "192.168.123.100",
        "unitree_robot_ip": "192.168.123.164",
    }
    context = AuthorizationContext.from_control_config(config)
    check("authorization: binding fields present",
          set(context.to_dict())
          >= {"backend", "execution_mode", "debug_mode", "model_profile",
              "unitree_network_interface", "unitree_robot_ip", "mocap_type",
              "redis_key"})
    check("authorization: redis contract pinned",
          context.redis_key == "action_qpos_g1_packet"
          and context.redis_port == 6379)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("ALL DOMAIN TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
