"""Authenticated local-network envelope for policy-to-gateway targets.

Freshness is measured from the gateway's *local receive time*.  Monotonic
clocks from WSL, a robot PC and Windows are not assumed to share an epoch.
The sender timestamp is retained for diagnostics only; sequence/session and a
short receive-side lifetime prevent replay of a cached target.

Import-cycle note: ``g1_safety.core`` classes are resolved lazily at call time
so importing this module remains lightweight.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
from pathlib import Path as _Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def _core():
    # M-decoupling: the core state machine is the single source of truth in
    # src.safety.g1_core; no legacy package is involved.
    return _importlib.import_module("src.safety.g1_core")


def _validation_error(message: str) -> Exception:
    return _core().ValidationError(message)


SCHEMA_VERSION = 1
MAX_PACKET_BYTES = 32 * 1024
ALLOWED_STATE_SOURCES = frozenset({"mujoco", "unitree_lowstate"})
_SESSION_RE = re.compile(r"[A-Za-z0-9._:-]{8,128}\Z")


def _vector(values: Sequence[float], size: int, label: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise _validation_error(f"{label} must contain {size} finite values")
    return array.tolist()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True)
class PolicyTargetEnvelope:
    session_id: str
    seq: int
    sender_monotonic: float
    valid_for_s: float
    state_source: str
    backend: str
    model: str
    dof_hash: str
    q: Sequence[float]
    dq: Sequence[float]
    tau_ff: Sequence[float]
    kp: Sequence[float]
    kd: Sequence[float]
    active_mask: Sequence[bool]
    deadman: bool
    teleop_source_age_s: float
    teleop_source_valid: bool

    def payload(self) -> dict[str, Any]:
        if not _SESSION_RE.fullmatch(str(self.session_id)):
            raise _validation_error(
                "session_id must be 8..128 safe ASCII characters")
        if not isinstance(self.seq, (int, np.integer)) or int(self.seq) < 0:
            raise _validation_error("sequence must be a non-negative integer")
        if not math.isfinite(float(self.sender_monotonic)):
            raise _validation_error("sender_monotonic must be finite")
        if not 0.002 <= float(self.valid_for_s) <= 0.250:
            raise _validation_error("valid_for_s must be between 2 and 250 ms")
        if self.state_source not in ALLOWED_STATE_SOURCES:
            raise _validation_error(
                "state_source must be mujoco or unitree_lowstate")
        if not str(self.backend).strip() or len(str(self.backend)) > 64:
            raise _validation_error("backend is missing or too long")
        if not str(self.model).strip() or not str(self.dof_hash).strip():
            raise _validation_error("model and dof_hash are required")
        age = float(self.teleop_source_age_s)
        if not math.isfinite(age) or age < 0:
            raise _validation_error("teleop_source_age_s must be non-negative")
        active_values = list(self.active_mask)
        if len(active_values) != 29 or any(
            not isinstance(value, (bool, np.bool_)) for value in active_values
        ):
            raise _validation_error(
                "active_mask must contain 29 JSON booleans")
        active = np.asarray(active_values, dtype=bool)
        if not isinstance(self.deadman, (bool, np.bool_)):
            raise _validation_error("deadman must be a JSON boolean")
        if not isinstance(self.teleop_source_valid, (bool, np.bool_)):
            raise _validation_error("teleop_source_valid must be a JSON boolean")
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "seq": int(self.seq),
            "sender_monotonic": float(self.sender_monotonic),
            "valid_for_s": float(self.valid_for_s),
            "state_source": self.state_source,
            "backend": str(self.backend),
            "model": str(self.model),
            "dof_hash": str(self.dof_hash),
            "q": _vector(self.q, 29, "q"),
            "dq": _vector(self.dq, 29, "dq"),
            "tau_ff": _vector(self.tau_ff, 29, "tau_ff"),
            "kp": _vector(self.kp, 29, "kp"),
            "kd": _vector(self.kd, 29, "kd"),
            "active_mask": active.tolist(),
            "deadman": bool(self.deadman),
            "teleop_source_age_s": age,
            "teleop_source_valid": bool(self.teleop_source_valid),
        }

    def to_policy_command(self, receive_monotonic: float):
        """Create a command timestamped on the gateway's local clock."""

        if not math.isfinite(receive_monotonic):
            raise _validation_error("receive_monotonic must be finite")
        payload = self.payload()
        return _core().PolicyCommand(
            seq=payload["seq"], timestamp=float(receive_monotonic),
            q=payload["q"], dq=payload["dq"], tau_ff=payload["tau_ff"],
            kp=payload["kp"], kd=payload["kd"], model=payload["model"],
            dof_hash=payload["dof_hash"], deadman=payload["deadman"],
            source_valid=True, active_mask=payload["active_mask"],
        )


def encode_policy_target(
    envelope: PolicyTargetEnvelope, *, secret: Optional[bytes] = None
) -> bytes:
    payload = envelope.payload()
    payload_bytes = _canonical_json(payload)
    signature = "" if secret is None else hmac.new(secret, payload_bytes, hashlib.sha256).hexdigest()
    encoded = _canonical_json({"payload": payload, "hmac_sha256": signature})
    if len(encoded) > MAX_PACKET_BYTES:
        raise _validation_error("policy packet exceeds maximum size")
    return encoded


def decode_policy_target(
    packet: bytes, *, secret: Optional[bytes] = None, require_hmac: bool = False
) -> PolicyTargetEnvelope:
    if not isinstance(packet, (bytes, bytearray)) or not 0 < len(packet) <= MAX_PACKET_BYTES:
        raise _validation_error("policy packet has invalid size")
    try:
        wrapper = json.loads(bytes(packet).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _validation_error("policy packet is not valid ASCII JSON") from exc
    if not isinstance(wrapper, dict) or set(wrapper) != {"payload", "hmac_sha256"}:
        raise _validation_error("policy packet wrapper schema mismatch")
    payload = wrapper["payload"]
    signature = wrapper["hmac_sha256"]
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise _validation_error("policy payload schema version mismatch")
    if require_hmac and (secret is None or not signature):
        raise _validation_error("authenticated policy packet required")
    if signature:
        if secret is None:
            raise _validation_error(
                "packet is signed but no verification key was supplied")
        expected = hmac.new(secret, _canonical_json(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(str(signature), expected):
            raise _validation_error("policy packet HMAC mismatch")
    try:
        envelope = PolicyTargetEnvelope(
            session_id=payload["session_id"], seq=payload["seq"],
            sender_monotonic=payload["sender_monotonic"],
            valid_for_s=payload["valid_for_s"], state_source=payload["state_source"],
            backend=payload["backend"], model=payload["model"],
            dof_hash=payload["dof_hash"], q=payload["q"], dq=payload["dq"],
            tau_ff=payload["tau_ff"], kp=payload["kp"], kd=payload["kd"],
            active_mask=payload["active_mask"], deadman=payload["deadman"],
            teleop_source_age_s=payload["teleop_source_age_s"],
            teleop_source_valid=payload["teleop_source_valid"],
        )
    except KeyError as exc:
        raise _validation_error(
            f"policy payload missing field {exc.args[0]}") from exc
    envelope.payload()
    return envelope
