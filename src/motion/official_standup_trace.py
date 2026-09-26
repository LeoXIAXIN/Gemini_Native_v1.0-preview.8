"""Validated, DDS-free storage for a measured official G1 stand-up trace.

The official ``Squat2StandUp`` action is executed by robot firmware.  A trace
captured from LowState can be replayed in MuJoCo for visual QA, but it must not
silently cross G1 hardware variants or accept a truncated/corrupt recording.
This module contains only NumPy and Python standard-library code; importing it
does not construct Unitree clients, DDS participants, or command publishers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


TRACE_SCHEMA_VERSION = 1
TRACE_HZ = 50.0
MIN_TRACE_FRAMES = 50
# Keep the duration long enough that a valid source stays valid after the
# public 50 Hz resampler (49 intervals for a 50-frame minimum).
MIN_TRACE_DURATION_SECONDS = (MIN_TRACE_FRAMES - 1) / TRACE_HZ
MIN_ENDPOINT_JOINT_DELTA = 0.15
OFFICIAL_SQUAT_TO_STAND_FSM_ID = 706

# Legacy trace asset labels are retained to keep existing archive hashes
# readable. They describe the model used for the trace, never the expected
# LowState mode_machine; that field is recorded directly from the robot.
SUPPORTED_TRACE_VARIANTS = frozenset({"4010", "5010"})

LEFT_KNEE = 3
RIGHT_KNEE = 9

_ARRAY_DTYPES = {
    "timestamps": np.dtype("<f8"),
    "joint_qpos": np.dtype("<f4"),
    "joint_qvel": np.dtype("<f4"),
    "root_quat_wxyz": np.dtype("<f4"),
    "root_gyro": np.dtype("<f4"),
    "fsm_id": np.dtype("<i4"),
}
_NPZ_KEYS = frozenset(
    {
        "schema_version",
        "robot_variant",
        "mode_machine",
        "firmware_id",
        *_ARRAY_DTYPES,
        "metadata_json",
        "content_sha256",
    }
)


class OfficialStandupTraceError(ValueError):
    """The trace is incompatible, incomplete, malformed, or corrupt."""


def _canonical_metadata(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(metadata, Mapping):
        raise OfficialStandupTraceError("metadata must be a JSON object")
    try:
        rendered = json.dumps(
            dict(metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(rendered)
    except (TypeError, ValueError) as exc:
        raise OfficialStandupTraceError("metadata is not finite JSON data") from exc
    if not isinstance(decoded, dict):
        raise OfficialStandupTraceError("metadata must encode a JSON object")
    return decoded, rendered


def _hash_part(digest: Any, name: str, payload: bytes) -> None:
    name_bytes = name.encode("utf-8")
    digest.update(len(name_bytes).to_bytes(4, "little"))
    digest.update(name_bytes)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _content_digest(
    *,
    schema_version: int,
    robot_variant: str,
    mode_machine: int,
    firmware_id: str,
    arrays: Mapping[str, np.ndarray],
    metadata_json: str,
) -> str:
    """Hash canonical decoded content, independent of ZIP container metadata."""

    digest = hashlib.sha256()
    for name, value in (
        ("schema_version", int(schema_version)),
        ("robot_variant", robot_variant),
        ("mode_machine", int(mode_machine)),
        ("firmware_id", firmware_id),
        ("metadata_json", metadata_json),
    ):
        _hash_part(
            digest,
            name,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
        )
    for name in _ARRAY_DTYPES:
        array = np.ascontiguousarray(arrays[name])
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        _hash_part(digest, f"{name}:header", header)
        _hash_part(digest, f"{name}:data", array.tobytes(order="C"))
    return digest.hexdigest()


def _scalar(array: np.ndarray, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise OfficialStandupTraceError(f"{name} must be an NPZ scalar")
    return value.item()


def _variant(value: Any) -> str:
    rendered = str(value).strip()
    if rendered.startswith("unitree_g1_"):
        rendered = rendered.removeprefix("unitree_g1_")
    if rendered not in SUPPORTED_TRACE_VARIANTS:
        raise OfficialStandupTraceError(
            f"unsupported trace model asset {value!r}"
        )
    return rendered


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise OfficialStandupTraceError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OfficialStandupTraceError(f"{name} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise OfficialStandupTraceError(f"{name} must be an integer")
    return int(number)


@dataclass(frozen=True)
class OfficialStandupTrace:
    """One measured firmware Squat2StandUp trajectory in 29-DoF order."""

    robot_variant: str
    mode_machine: int
    firmware_id: str
    timestamps: np.ndarray
    joint_qpos: np.ndarray
    joint_qvel: np.ndarray
    root_quat_wxyz: np.ndarray
    root_gyro: np.ndarray
    fsm_id: np.ndarray
    metadata: Mapping[str, Any]
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        variant = _variant(self.robot_variant)
        firmware_id = str(self.firmware_id).strip()
        if not firmware_id:
            raise OfficialStandupTraceError("firmware_id must not be empty")
        try:
            firmware_id.encode("utf-8")
        except UnicodeError as exc:
            raise OfficialStandupTraceError("firmware_id is not valid UTF-8") from exc
        metadata, _ = _canonical_metadata(self.metadata)
        object.__setattr__(self, "robot_variant", variant)
        object.__setattr__(
            self, "mode_machine", _strict_int(self.mode_machine, "mode_machine")
        )
        object.__setattr__(self, "firmware_id", firmware_id)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "schema_version",
            _strict_int(self.schema_version, "schema_version"),
        )
        for name, dtype in _ARRAY_DTYPES.items():
            raw = np.asarray(getattr(self, name))
            if name == "fsm_id":
                try:
                    numeric = np.asarray(raw, dtype=np.float64)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise OfficialStandupTraceError(
                        "fsm_id must contain integers"
                    ) from exc
                if not np.all(np.isfinite(numeric)) or not np.all(
                    numeric == np.floor(numeric)
                ):
                    raise OfficialStandupTraceError(
                        "fsm_id must contain finite integers"
                    )
            value = np.asarray(raw, dtype=dtype)
            object.__setattr__(self, name, np.ascontiguousarray(value).copy())
        self.validate()

    @property
    def frame_count(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def content_sha256(self) -> str:
        _, metadata_json = _canonical_metadata(self.metadata)
        return _content_digest(
            schema_version=self.schema_version,
            robot_variant=self.robot_variant,
            mode_machine=self.mode_machine,
            firmware_id=self.firmware_id,
            arrays={name: getattr(self, name) for name in _ARRAY_DTYPES},
            metadata_json=metadata_json,
        )

    def validate(self, *, expected_robot_variant: str | None = None) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise OfficialStandupTraceError(
                f"unsupported schema_version {self.schema_version}; "
                f"expected {TRACE_SCHEMA_VERSION}"
            )
        if not 1 <= self.mode_machine <= 255:
            raise OfficialStandupTraceError(
                "mode_machine must be a nonzero uint8 received from LowState"
            )
        if expected_robot_variant is not None:
            expected = _variant(expected_robot_variant)
            if self.robot_variant != expected:
                raise OfficialStandupTraceError(
                    f"trace is for G1 {self.robot_variant}, not requested G1 {expected}"
                )

        if self.timestamps.ndim != 1:
            raise OfficialStandupTraceError("timestamps must have shape (N,)")
        n = self.frame_count
        expected_shapes = {
            "joint_qpos": (n, 29),
            "joint_qvel": (n, 29),
            "root_quat_wxyz": (n, 4),
            "root_gyro": (n, 3),
            "fsm_id": (n,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise OfficialStandupTraceError(
                    f"{name} must have shape {shape}, got {getattr(self, name).shape}"
                )
        if n < MIN_TRACE_FRAMES:
            raise OfficialStandupTraceError(
                f"trace has only {n} frames; at least {MIN_TRACE_FRAMES} are required"
            )
        for name in _ARRAY_DTYPES:
            value = getattr(self, name)
            if not np.all(np.isfinite(value)):
                raise OfficialStandupTraceError(f"{name} contains NaN or infinity")
        if np.any(np.diff(self.timestamps) <= 0.0):
            raise OfficialStandupTraceError("timestamps must be strictly increasing")
        if self.duration_seconds < MIN_TRACE_DURATION_SECONDS:
            raise OfficialStandupTraceError(
                f"trace duration {self.duration_seconds:.3f}s is too short"
            )
        quaternion_norms = np.linalg.norm(self.root_quat_wxyz, axis=1)
        if np.any(np.abs(quaternion_norms - 1.0) > 0.05):
            raise OfficialStandupTraceError(
                "root_quat_wxyz must contain normalized wxyz quaternions"
            )
        if np.any(self.fsm_id < 0):
            raise OfficialStandupTraceError("fsm_id contains a negative value")
        if not np.any(self.fsm_id == OFFICIAL_SQUAT_TO_STAND_FSM_ID):
            raise OfficialStandupTraceError(
                "trace never observed official Squat2StandUp FSM 706"
            )

        endpoint_delta = np.abs(self.joint_qpos[-1] - self.joint_qpos[0])
        if float(np.max(endpoint_delta)) < MIN_ENDPOINT_JOINT_DELTA:
            raise OfficialStandupTraceError(
                "first/last poses do not contain a stand-up-sized joint motion"
            )
        for index, label in ((LEFT_KNEE, "left"), (RIGHT_KNEE, "right")):
            extension = float(self.joint_qpos[0, index] - self.joint_qpos[-1, index])
            if extension < MIN_ENDPOINT_JOINT_DELTA:
                raise OfficialStandupTraceError(
                    f"{label} knee did not extend by at least "
                    f"{MIN_ENDPOINT_JOINT_DELTA:.2f} rad"
                )

    def save(self, path: str | os.PathLike[str]) -> Path:
        return save_official_standup_trace(self, path)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_robot_variant: str | None = None,
    ) -> "OfficialStandupTrace":
        return load_official_standup_trace(
            path, expected_robot_variant=expected_robot_variant
        )

    def resample_50hz(self) -> "OfficialStandupTrace":
        return resample_official_standup_trace_50hz(self)


def save_official_standup_trace(
    trace: OfficialStandupTrace, path: str | os.PathLike[str]
) -> Path:
    """Atomically save a validated trace with a canonical content SHA256."""

    if not isinstance(trace, OfficialStandupTrace):
        raise TypeError("trace must be an OfficialStandupTrace")
    trace.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _, metadata_json = _canonical_metadata(trace.metadata)
    payload: dict[str, Any] = {
        "schema_version": np.asarray(trace.schema_version, dtype="<i4"),
        "robot_variant": np.asarray(trace.robot_variant),
        "mode_machine": np.asarray(trace.mode_machine, dtype="<i4"),
        "firmware_id": np.asarray(trace.firmware_id),
        **{name: getattr(trace, name) for name in _ARRAY_DTYPES},
        "metadata_json": np.asarray(metadata_json),
        "content_sha256": np.asarray(trace.content_sha256),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=destination.name + ".", suffix=".tmp", dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(temporary, **payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_official_standup_trace(
    path: str | os.PathLike[str],
    *,
    expected_robot_variant: str | None = None,
) -> OfficialStandupTrace:
    """Load a trace and verify schema, canonical metadata, and content SHA256."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != _NPZ_KEYS:
                missing = sorted(_NPZ_KEYS - keys)
                extra = sorted(keys - _NPZ_KEYS)
                raise OfficialStandupTraceError(
                    f"invalid NPZ fields; missing={missing}, extra={extra}"
                )
            if archive["schema_version"].dtype != np.dtype("<i4"):
                raise OfficialStandupTraceError("schema_version must use int32 storage")
            if archive["mode_machine"].dtype != np.dtype("<i4"):
                raise OfficialStandupTraceError("mode_machine must use int32 storage")
            schema_version = _strict_int(
                _scalar(archive["schema_version"], "schema_version"),
                "schema_version",
            )
            robot_variant = _variant(_scalar(archive["robot_variant"], "robot_variant"))
            mode_machine = _strict_int(
                _scalar(archive["mode_machine"], "mode_machine"), "mode_machine"
            )
            firmware_id = str(_scalar(archive["firmware_id"], "firmware_id"))
            metadata_json = str(_scalar(archive["metadata_json"], "metadata_json"))
            stored_sha256 = str(_scalar(archive["content_sha256"], "content_sha256"))
            if len(stored_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in stored_sha256
            ):
                raise OfficialStandupTraceError("content_sha256 is not lowercase SHA256")
            arrays: dict[str, np.ndarray] = {}
            for name, dtype in _ARRAY_DTYPES.items():
                value = np.asarray(archive[name])
                if value.dtype != dtype:
                    raise OfficialStandupTraceError(
                        f"{name} has dtype {value.dtype.str}, expected {dtype.str}"
                    )
                arrays[name] = np.ascontiguousarray(value).copy()
    except OfficialStandupTraceError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise OfficialStandupTraceError(f"could not read trace {source}: {exc}") from exc

    try:
        decoded_metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise OfficialStandupTraceError("metadata_json is invalid JSON") from exc
    metadata, canonical_json = _canonical_metadata(decoded_metadata)
    if metadata_json != canonical_json:
        raise OfficialStandupTraceError("metadata_json is not canonical JSON")
    calculated_sha256 = _content_digest(
        schema_version=schema_version,
        robot_variant=robot_variant,
        mode_machine=mode_machine,
        firmware_id=firmware_id,
        arrays=arrays,
        metadata_json=metadata_json,
    )
    if not hmac.compare_digest(stored_sha256, calculated_sha256):
        raise OfficialStandupTraceError(
            "trace content SHA256 mismatch; file is corrupt or was modified"
        )
    trace = OfficialStandupTrace(
        schema_version=schema_version,
        robot_variant=robot_variant,
        mode_machine=mode_machine,
        firmware_id=firmware_id,
        metadata=metadata,
        **arrays,
    )
    trace.validate(expected_robot_variant=expected_robot_variant)
    return trace


def resample_official_standup_trace_50hz(
    trace: OfficialStandupTrace,
) -> OfficialStandupTrace:
    """Return a uniform 50 Hz trace using linear/NLERP and held FSM IDs."""

    if not isinstance(trace, OfficialStandupTrace):
        raise TypeError("trace must be an OfficialStandupTrace")
    trace.validate()
    start = float(trace.timestamps[0])
    end = float(trace.timestamps[-1])
    count = int(math.floor((end - start) * TRACE_HZ + 1e-9)) + 1
    timestamps = start + np.arange(count, dtype=np.float64) / TRACE_HZ

    def interpolate_matrix(values: np.ndarray) -> np.ndarray:
        result = np.empty((count, values.shape[1]), dtype=np.float32)
        for column in range(values.shape[1]):
            result[:, column] = np.interp(
                timestamps, trace.timestamps, values[:, column]
            ).astype(np.float32)
        return result

    qpos = interpolate_matrix(trace.joint_qpos)
    qvel = interpolate_matrix(trace.joint_qvel)
    gyro = interpolate_matrix(trace.root_gyro)

    # Consecutive q and -q are the same orientation.  Align signs first so
    # component interpolation follows the short quaternion arc, then normalize
    # every output row (normalized linear interpolation / NLERP).
    source_quat = trace.root_quat_wxyz.astype(np.float64, copy=True)
    for index in range(1, source_quat.shape[0]):
        if float(np.dot(source_quat[index - 1], source_quat[index])) < 0.0:
            source_quat[index] *= -1.0
    quat = np.empty((count, 4), dtype=np.float64)
    for column in range(4):
        quat[:, column] = np.interp(
            timestamps, trace.timestamps, source_quat[:, column]
        )
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 1e-12):
        raise OfficialStandupTraceError("quaternion interpolation became singular")
    quat = (quat / norms[:, None]).astype(np.float32)

    held_indices = np.searchsorted(trace.timestamps, timestamps, side="right") - 1
    held_indices = np.clip(held_indices, 0, trace.frame_count - 1)
    fsm_id = trace.fsm_id[held_indices].astype(np.int32, copy=True)
    metadata = dict(trace.metadata)
    metadata.update(
        {
            "resample_hz": TRACE_HZ,
            "resampled_from_frame_count": trace.frame_count,
            "resampled_from_sha256": trace.content_sha256,
        }
    )
    return OfficialStandupTrace(
        robot_variant=trace.robot_variant,
        mode_machine=trace.mode_machine,
        firmware_id=trace.firmware_id,
        timestamps=timestamps,
        joint_qpos=qpos,
        joint_qvel=qvel,
        root_quat_wxyz=quat,
        root_gyro=gyro,
        fsm_id=fsm_id,
        metadata=metadata,
    )


__all__ = [
    "MIN_ENDPOINT_JOINT_DELTA",
    "MIN_TRACE_DURATION_SECONDS",
    "MIN_TRACE_FRAMES",
    "OFFICIAL_SQUAT_TO_STAND_FSM_ID",
    "OfficialStandupTrace",
    "OfficialStandupTraceError",
    "SUPPORTED_TRACE_VARIANTS",
    "TRACE_HZ",
    "TRACE_SCHEMA_VERSION",
    "load_official_standup_trace",
    "resample_official_standup_trace_50hz",
    "save_official_standup_trace",
]
