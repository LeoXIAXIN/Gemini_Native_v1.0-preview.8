"""CHINGMU/GMR RESP reference adapter for Humanoid-GPT deployment.

This module adapts the existing ``action_qpos_g1_packet`` value to the tiny
``MocapBuffer.read()`` interface used by :mod:`src.motion.play_track`. It carries
reference motion only: it never reads simulated motor targets and it has no
Unitree SDK2 or LowCmd dependency.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Mapping

import numpy as np


CHINGMU_QPOS_SCHEMA_VERSION = 1
CHINGMU_G1_QPOS_SIZE = 36
CHINGMU_G1_MODEL = "unitree_g1_29dof"
CHINGMU_G1_DOF_ORDER_HASH = "24863bd4b4afc6d538b1db91d6ae305a41573b79b42f1a59eb298d81fdb52618"


class ChingMuPacketError(ValueError):
    """The Redis value is not a valid G1 reference packet."""


def parse_chingmu_qpos_packet(payload: Any) -> tuple[np.ndarray, float]:
    """Parse one ``action_qpos_g1_packet`` value.

    Returns a normalized 36-value floating-base G1 qpos and its producer-side
    monotonic timestamp.  The parser deliberately does not turn a reference
    into a motor command; real-robot state feedback and policy inference stay
    in Humanoid-GPT's official deployment loop.
    """

    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ChingMuPacketError("CHINGMU Redis packet is not UTF-8") from exc
    if isinstance(payload, str):
        try:
            packet = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ChingMuPacketError("CHINGMU Redis packet is not valid JSON") from exc
    else:
        packet = payload

    if not isinstance(packet, Mapping):
        raise ChingMuPacketError("CHINGMU Redis packet must be a JSON object")
    if packet.get("schema_version") != CHINGMU_QPOS_SCHEMA_VERSION:
        raise ChingMuPacketError("unsupported CHINGMU qpos packet schema")

    timestamp_value = packet.get("generated_monotonic", packet.get("monotonic"))
    if isinstance(timestamp_value, bool):
        raise ChingMuPacketError("CHINGMU packet timestamp must be finite")
    try:
        timestamp = float(timestamp_value)
    except (TypeError, ValueError) as exc:
        raise ChingMuPacketError("CHINGMU packet timestamp is missing or invalid") from exc
    if not math.isfinite(timestamp):
        raise ChingMuPacketError("CHINGMU packet timestamp must be finite")

    try:
        qpos = np.asarray(packet["qpos"], dtype=np.float32)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ChingMuPacketError("CHINGMU packet qpos is missing or invalid") from exc
    if qpos.shape != (CHINGMU_G1_QPOS_SIZE,):
        raise ChingMuPacketError(
            f"CHINGMU packet qpos must have shape ({CHINGMU_G1_QPOS_SIZE},), "
            f"got {qpos.shape}"
        )
    if not np.all(np.isfinite(qpos)):
        raise ChingMuPacketError("CHINGMU packet qpos contains NaN or infinity")

    quaternion_norm = float(np.linalg.norm(qpos[3:7]))
    if not math.isfinite(quaternion_norm) or quaternion_norm < 1e-8:
        raise ChingMuPacketError("CHINGMU root quaternion has near-zero norm")
    qpos = qpos.copy()
    qpos[3:7] /= quaternion_norm
    return qpos, timestamp


class ChingMuRedisMocapBuffer:
    """Read the latest CHINGMU/GMR reference from Redis."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        key: str = "action_qpos_g1_packet",
        *,
        redis_client: Any | None = None,
        require_integrity: bool = False,
        clock=time.monotonic,
        maximum_age_seconds: float = 0.12,
    ) -> None:
        if not str(host).strip():
            raise ValueError("Redis host must be non-empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("Redis port must be between 1 and 65535")
        if not str(key).strip():
            raise ValueError("Redis key must be non-empty")

        self.host = str(host).strip()
        self.port = int(port)
        self.key = str(key).strip()
        self.require_integrity = bool(require_integrity)
        self._clock = clock
        self.maximum_age_seconds = float(maximum_age_seconds)
        if not math.isfinite(self.maximum_age_seconds) or self.maximum_age_seconds <= 0:
            raise ValueError("maximum_age_seconds must be positive and finite")
        self._last_session_id: str | None = None
        self._last_sequence: int | None = None
        if redis_client is None:
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError(
                    "Redis support is required for --mocap-type chingmu_redis; "
                    "install the 'redis' Python package"
                ) from exc
            redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=0,
                decode_responses=False,
                socket_connect_timeout=1.0,
                socket_timeout=0.1,
                # Gemini Native ships a deliberately small loopback-only
                # RESP2 state store.  redis-py 6/7 may otherwise negotiate
                # RESP3 with HELLO, which that bounded server does not expose.
                protocol=2,
            )
            redis_client.ping()
        self._redis = redis_client

    def read(self) -> tuple[np.ndarray, float]:
        payload = self._redis.get(self.key)
        if payload is None:
            raise ChingMuPacketError(
                f"CHINGMU Redis key {self.key!r} has no reference packet"
            )
        qpos, generated = parse_chingmu_qpos_packet(payload)
        if self.require_integrity:
            try:
                packet = json.loads(bytes(payload).decode("utf-8")) if isinstance(
                    payload, (bytes, bytearray)
                ) else json.loads(payload) if isinstance(payload, str) else payload
                session_id = str(packet["session_id"])
                sequence_value = packet["sequence"]
                if isinstance(sequence_value, bool):
                    raise ValueError("sequence must be an integer")
                sequence = int(sequence_value)
                deadline = float(packet["valid_until_monotonic"])
            except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ChingMuPacketError(
                    f"CHINGMU packet integrity metadata is missing or invalid: {exc}"
                ) from exc
            if not re.fullmatch(r"[0-9a-f]{32}", session_id):
                raise ChingMuPacketError("CHINGMU packet session_id is invalid")
            if sequence <= 0:
                raise ChingMuPacketError("CHINGMU packet sequence must be positive")
            if packet.get("robot_model") != CHINGMU_G1_MODEL:
                raise ChingMuPacketError("CHINGMU packet robot model does not match G1 29-DoF")
            if packet.get("dof_order_hash") != CHINGMU_G1_DOF_ORDER_HASH:
                raise ChingMuPacketError("CHINGMU packet joint order hash does not match")
            now = float(self._clock())
            if not math.isfinite(now) or not math.isfinite(deadline):
                raise ChingMuPacketError("CHINGMU packet freshness values must be finite")
            if generated > now + 0.05:
                raise ChingMuPacketError("CHINGMU packet timestamp is in the future")
            if now - generated > self.maximum_age_seconds or now > deadline:
                raise ChingMuPacketError("CHINGMU reference packet is stale")
            if self._last_session_id == session_id and self._last_sequence is not None:
                if sequence < self._last_sequence:
                    raise ChingMuPacketError("CHINGMU packet sequence regressed")
            else:
                self._last_session_id = session_id
                self._last_sequence = None
            self._last_sequence = sequence
        return qpos, generated


__all__ = [
    "CHINGMU_G1_QPOS_SIZE",
    "CHINGMU_QPOS_SCHEMA_VERSION",
    "ChingMuPacketError",
    "ChingMuRedisMocapBuffer",
    "parse_chingmu_qpos_packet",
]
