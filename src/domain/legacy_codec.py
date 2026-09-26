"""LegacyCodec: byte/JSON-level compatibility with established wire formats.

This module owns their layout details:

  * CHINGMU UDP frame packet
    ``pickle.dumps((frame_id:int, positions:[[x,y,z]...], rotations:[[x,y,z,w]...],
    detected:[0|1 ...]), protocol=5)`` — produced by
    ``src.adapters.chingmu.build_frame_packet``.
  * GMR reference packets ``action_qpos_g1_packet`` / ``action_mimic_g1_packet``
    — JSON objects with schema_version=1 (see ``src.domain.models``).
  * Safety channel payloads ``g1_safety_control`` / ``g1_safety_status`` —
    JSON objects with schema_version=1.

Conversions validate fail-closed and raise ``CodecError`` on any mismatch.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np

from src.domain.enums import (
    ReferenceHealth,
    RobotStateSource,
    SafetyState,
)
from src.domain.errors import CodecError, ValidationError
from src.domain.models import (
    G1_QPOS_SIZE,
    MIMIC_TARGET_SIZE,
    G1Reference,
    MocapFrame,
    SafeCommand,
    SafetyStatus,
)


class LegacyCodec:
    """Stateless conversions between domain models and legacy payloads."""

    # ------------------------------------------------------------------ #
    # CHINGMU frame packet (pickle protocol 5)                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def decode_chingmu_frame_packet(payload: bytes) -> MocapFrame:
        if not isinstance(payload, (bytes, bytearray)):
            raise CodecError("frame packet payload must be bytes")
        try:
            raw = pickle.loads(bytes(payload))
        except Exception as exc:  # noqa: BLE001 - legacy pickle may raise anything
            raise CodecError(f"cannot unpickle frame packet: {exc}") from exc
        if not isinstance(raw, tuple) or len(raw) != 4:
            raise CodecError("frame packet must be a 4-item tuple")
        frame_id, positions, rotations, detected = raw
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise CodecError("frame packet frame_id must be an integer")
        try:
            positions = np.asarray(positions, dtype=float)
            rotations = np.asarray(rotations, dtype=float)
            detected = np.asarray(detected)
        except (TypeError, ValueError) as exc:
            raise CodecError(f"frame packet arrays are invalid: {exc}") from exc
        try:
            return MocapFrame(
                frame_id=frame_id,
                positions=positions,
                rotations=rotations,
                detected=detected,
                received_monotonic=None,
            )
        except Exception as exc:  # noqa: BLE001 - ValidationError from models
            raise CodecError(f"frame packet validation failed: {exc}") from exc

    @staticmethod
    def encode_chingmu_frame_packet(frame: MocapFrame) -> bytes:
        if not isinstance(frame, MocapFrame):
            raise CodecError("expected a MocapFrame")
        positions = [
            [float(value) for value in row]
            for row in frame.positions[:, :3].tolist()
        ]
        rotations = [
            [float(value) for value in row]
            for row in frame.rotations[:, :4].tolist()
        ]
        detected = [int(bool(flag)) for flag in frame.detected.tolist()]
        return pickle.dumps(
            (int(frame.frame_id), positions, rotations, detected), protocol=5
        )

    # ------------------------------------------------------------------ #
    # GMR reference packets (JSON, schema_version 1)                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode_reference_packet(
        payload: Any, *, require_mimic: bool
    ) -> G1Reference:
        if isinstance(payload, (bytes, bytearray)):
            payload = bytes(payload).decode("utf-8")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise CodecError(f"reference packet is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise CodecError("reference packet must decode to a JSON object")
        if payload.get("schema_version") != 1:
            raise CodecError("unsupported reference packet schema version")
        required = (
            "session_id",
            "sequence",
            "frame_id",
            "generated_monotonic",
            "valid_until_monotonic",
            "robot_model",
            "dof_order_hash",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise CodecError(f"reference packet is missing fields: {missing}")
        qpos_raw = payload.get("qpos")
        mimic_raw = payload.get("mimic")
        if require_mimic and mimic_raw is None:
            raise CodecError("mimic packet is missing its mimic vector")
        if not require_mimic and qpos_raw is None:
            raise CodecError("qpos packet is missing its qpos vector")
        try:
            sequence = int(payload["sequence"])
            frame_id = int(payload["frame_id"])
            generated = float(payload["generated_monotonic"])
            valid_until = float(payload["valid_until_monotonic"])
        except (TypeError, ValueError) as exc:
            raise CodecError(f"reference packet scalars are invalid: {exc}") from exc
        try:
            return G1Reference(
                session_id=str(payload["session_id"]),
                sequence=sequence,
                frame_id=frame_id,
                generated_monotonic=generated,
                valid_until_monotonic=valid_until,
                robot_model=str(payload["robot_model"]),
                dof_order_hash=str(payload["dof_order_hash"]),
                qpos=np.asarray(qpos_raw, dtype=float)
                if qpos_raw is not None
                else None,
                mimic=np.asarray(mimic_raw, dtype=float)
                if mimic_raw is not None
                else None,
            )
        except Exception as exc:  # noqa: BLE001 - ValidationError from models
            raise CodecError(f"reference packet validation failed: {exc}") from exc

    @staticmethod
    def decode_gmr_qpos_packet(payload: Any) -> G1Reference:
        """Decode the ``action_qpos_g1_packet`` value (str or bytes)."""
        return LegacyCodec._decode_reference_packet(payload, require_mimic=False)

    @staticmethod
    def decode_gmr_mimic_packet(payload: Any) -> G1Reference:
        """Decode the ``action_mimic_g1_packet`` value (str or bytes)."""
        return LegacyCodec._decode_reference_packet(payload, require_mimic=True)

    @staticmethod
    def encode_gmr_qpos_packet(reference: G1Reference) -> str:
        """Re-encode ``action_qpos_g1_packet`` with the legacy key order."""
        if not isinstance(reference, G1Reference):
            raise CodecError("expected a G1Reference")
        if reference.qpos is None:
            raise CodecError("G1Reference has no qpos vector")
        if reference.qpos.shape != (G1_QPOS_SIZE,):
            raise CodecError("G1Reference qpos must have 36 values")
        meta = {
            "schema_version": 1,
            "session_id": reference.session_id,
            "sequence": int(reference.sequence),
            "frame_id": int(reference.frame_id),
            "generated_monotonic": float(reference.generated_monotonic),
            "valid_until_monotonic": float(reference.valid_until_monotonic),
            "robot_model": reference.robot_model,
            "dof_order_hash": reference.dof_order_hash,
        }
        return json.dumps(
            {**meta, "monotonic": float(reference.generated_monotonic),
             "qpos": [float(value) for value in reference.qpos.tolist()]}
        )

    @staticmethod
    def encode_gmr_mimic_packet(reference: G1Reference) -> str:
        if not isinstance(reference, G1Reference):
            raise CodecError("expected a G1Reference")
        if reference.mimic is None:
            raise CodecError("G1Reference has no mimic vector")
        if reference.mimic.shape != (MIMIC_TARGET_SIZE,):
            raise CodecError("mimic vector must have 33 values")
        meta = {
            "schema_version": 1,
            "session_id": reference.session_id,
            "sequence": int(reference.sequence),
            "frame_id": int(reference.frame_id),
            "generated_monotonic": float(reference.generated_monotonic),
            "valid_until_monotonic": float(reference.valid_until_monotonic),
            "robot_model": reference.robot_model,
            "dof_order_hash": reference.dof_order_hash,
        }
        return json.dumps(
            {**meta, "mimic": [float(value) for value in reference.mimic.tolist()]}
        )

    # ------------------------------------------------------------------ #
    # Safety channel (JSON, schema_version 1)                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def encode_safe_command(command: SafeCommand) -> dict[str, Any]:
        if not isinstance(command, SafeCommand):
            raise CodecError("expected a SafeCommand")
        payload: dict[str, Any] = {
            "sequence": int(command.sequence),
            "action": command.action,
        }
        if command.action == "reset":
            payload.update(
                {
                    "operator_confirmed": True,
                    "physical_estop_released": True,
                }
            )
        return payload

    @staticmethod
    def decode_safe_command(payload: Any) -> SafeCommand:
        if isinstance(payload, (bytes, bytearray)):
            payload = bytes(payload).decode("utf-8")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise CodecError(f"safety command is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise CodecError("safety command must decode to a JSON object")
        try:
            return SafeCommand(
                sequence=int(payload["sequence"]),
                action=str(payload["action"]).strip().lower(),
                operator_confirmed=bool(payload.get("operator_confirmed", False)),
                physical_estop_released=bool(
                    payload.get("physical_estop_released", False)
                ),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise CodecError(f"safety command is invalid: {exc}") from exc

    @staticmethod
    def decode_safety_status(payload: Any) -> SafetyStatus:
        """Decode the ``g1_safety_status`` telemetry with legacy-equivalent checks."""
        if isinstance(payload, (bytes, bytearray)):
            payload = bytes(payload).decode("utf-8")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise CodecError(f"safety status is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise CodecError("safety status must decode to a JSON object")
        try:
            state = SafetyState(str(payload["state"]))
            reference_health = ReferenceHealth(str(payload["reference_health"]))
            state_source = RobotStateSource(str(payload["state_source"]))
            teleop_weight = float(payload["teleop_weight"])
            reference_age_ms = float(payload["reference_age_ms"])
            flags = {
                name: bool(payload[name])
                for name in (
                    "estop_latched",
                    "fault_latched",
                    "real_hardware_allowed",
                    "strategy_enabled",
                    "debug_mode",
                    "bypass_active",
                    "hard_guards_enabled",
                    "limited",
                )
            }
            return SafetyStatus(
                state=state,
                reason=str(payload["reason"]),
                estop_latched=flags["estop_latched"],
                fault_latched=flags["fault_latched"],
                reference_health=reference_health,
                state_source=state_source,
                real_hardware_allowed=flags["real_hardware_allowed"],
                teleop_weight=teleop_weight,
                reference_age_ms=reference_age_ms,
                strategy_enabled=flags["strategy_enabled"],
                debug_mode=flags["debug_mode"],
                bypass_active=flags["bypass_active"],
                hard_guards_enabled=flags["hard_guards_enabled"],
                limited=flags["limited"],
                schema_version=1,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise CodecError(f"safety status is invalid: {exc}") from exc
