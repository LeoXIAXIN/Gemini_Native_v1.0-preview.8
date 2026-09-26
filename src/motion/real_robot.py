"""Real-robot low-level control for Unitree G1.

Modified in July 2026 for the CHINGMU control-console integration and explicit
simulation/real deployment separation.

Provides:
- LowLevelControlG1: read sensor state and send joint-level PD targets.
- RealRobotState: adapter that writes sensor readings into a MuJoCo
  mj_data / State so that G1TrackInferFn.update_state() works unchanged.
"""

from __future__ import annotations

import math
import time
import struct
import threading
import mujoco
import numpy as np

from src.motion.deploy_constants import (
    NUM_JOINT, DEFAULT_QPOS, TOPIC_LOWCMD, TOPIC_LOWSTATE, MotorID,
)
from src.motion.startup_handover import HandoverPhase, StartupHandover

from src.motion.infer_utils import State
from src.motion.tracking_constants import KPs, KDs


# ---------------------------------------------------------------------------
# Unitree SDK helpers (deferred imports keep sim-only usage SDK-free)
# ---------------------------------------------------------------------------

class KeyMap:
    R1 = 0; L1 = 1; start = 2; select = 3
    R2 = 4; L2 = 5; F1 = 6; F2 = 7
    A = 8; B = 9; X = 10; Y = 11
    up = 12; right = 13; down = 14; left = 15


class RemoteController:
    def __init__(self):
        self.lx = self.ly = self.rx = self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # Official HG IDL exposes array[uint8, 40], usually decoded as a list.
        data = bytes(data)
        keys = struct.unpack("<H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("<f", data[4:8])[0]
        self.rx = struct.unpack("<f", data[8:12])[0]
        self.ry = struct.unpack("<f", data[12:16])[0]
        self.ly = struct.unpack("<f", data[20:24])[0]


def _set_motor_velocity_zero(motor_cmd) -> None:
    """Set the SDK motor velocity field across dq/qd naming revisions."""

    if hasattr(motor_cmd, "dq"):
        motor_cmd.dq = 0.0
    elif hasattr(motor_cmd, "qd"):
        motor_cmd.qd = 0.0
    else:
        raise AttributeError("Unitree motor command has neither dq nor qd")


class _UnitreeMotor:
    class MotorMode:
        PR = 0
        AB = 1

    @staticmethod
    def create_damping_cmd(cmd):
        for i in range(len(cmd.motor_cmd)):
            cmd.motor_cmd[i].q = 0
            _set_motor_velocity_zero(cmd.motor_cmd[i])
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 8
            cmd.motor_cmd[i].tau = 0

    @staticmethod
    def create_zero_cmd(cmd):
        for i in range(len(cmd.motor_cmd)):
            cmd.motor_cmd[i].q = 0
            _set_motor_velocity_zero(cmd.motor_cmd[i])
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0

    @staticmethod
    def init_cmd_hg(cmd, mode_machine, mode_pr):
        cmd.mode_machine = mode_machine
        cmd.mode_pr = mode_pr
        for i in range(len(cmd.motor_cmd)):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = 0
            _set_motor_velocity_zero(cmd.motor_cmd[i])
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0


# ---------------------------------------------------------------------------
# Low-level G1 controller
# ---------------------------------------------------------------------------

class LowLevelControlG1:
    """Low-level Unitree G1 motor controller via DDS."""

    def __init__(
        self,
        active_j2m_ids=MotorID.FULL,
        ctrl_dt: float = 0.02,
        debug: bool = False,
        defer_publisher: bool = False,
        lowstate_timeout_seconds: float | None = 20.0,
        should_cancel=None,
        g1_version: str | None = None,
    ):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG, LowState_ as LowStateHG
        from unitree_sdk2py.utils.crc import CRC

        self.debug = debug
        self.defer_publisher = bool(defer_publisher)
        # Deprecated compatibility argument; asset selection is independent of
        # the DDS machine mode and must never decide the command header.
        self.g1_version = g1_version
        # Locked by the first validated LowState callback, before a writer can
        # be created.  Zero is a valid uint8 value, so use None as the sentinel.
        self.expected_mode_machine: int | None = None
        self.ctrl_dt = ctrl_dt
        self.default_qpos = DEFAULT_QPOS.copy()
        self._kps = KPs.copy()
        self._kds = KDs.copy()
        self.active_j2m_ids = list(active_j2m_ids)
        self._crc = CRC()
        self._state_crc = CRC()

        self.joint_qpos = np.zeros(NUM_JOINT, dtype=np.float32)
        self.joint_qvel = np.zeros(NUM_JOINT, dtype=np.float32)
        self.joint_torque = np.zeros(NUM_JOINT, dtype=np.float32)
        self.root_quat = np.zeros(4, dtype=np.float32)
        self.root_gyro = np.zeros(3, dtype=np.float32)

        # Use the official SDK factory and HG IDL for both message directions.
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = _UnitreeMotor.MotorMode.PR
        self.mode_machine_ = 0
        self.remote = RemoteController()
        # The DDS callback may run concurrently with the 50 Hz controller.
        # Every value derived from one LowState packet is committed under this
        # single lock so readers can never combine a new tick/mode with old
        # joints or IMU data (or vice versa).
        self._state_lock = threading.Lock()
        self._last_low_state_tick = None
        self._last_low_state_advanced_at = None
        self._low_state_error = None

        self._LowCmdHG = LowCmdHG
        self._pub = None
        self._last_lowcmd_write_at = None
        self._lowcmd_write_count = 0
        # ``disable_lowcmd_publisher`` is a terminal boundary for the current
        # writer instance.  Keeping this state separate from ``_pub is None``
        # preserves the existing deferred/debug behaviour while making every
        # command method fail explicitly after a real writer has been closed.
        self._lowcmd_publisher_disabled = False
        self._lowcmd_writer_shutdown_unconfirmed = False
        self._sub = ChannelSubscriber(TOPIC_LOWSTATE, LowStateHG)
        self._sub.Init(self._handle_low_state, 10)

        self._wait_low_state(lowstate_timeout_seconds, should_cancel)
        self._validated_sensor_snapshot()
        _UnitreeMotor.init_cmd_hg(
            self.low_cmd, self.expected_mode_machine, self.mode_pr_
        )
        # Keep the public fast_* methods for callers that time individual
        # stages; serialization and CRC now always use the official SDK.
        self._kps_f32 = np.ascontiguousarray(self._kps, dtype=np.float32)
        self._kds_f32 = np.ascontiguousarray(self._kds, dtype=np.float32)
        # Debug/read-only mode never imports or constructs ChannelPublisher.
        if not self.debug and not self.defer_publisher:
            from unitree_sdk2py.core.channel import ChannelPublisher

            publisher = ChannelPublisher(TOPIC_LOWCMD, LowCmdHG)
            self._initialize_publisher(publisher)
            self._pub = publisher

        if self.debug:
            publisher_state = "read-only; LowCmd disabled"
        elif self.defer_publisher:
            publisher_state = "LowCmd gated until high-level release"
        else:
            publisher_state = "LowCmd ready"
        print(f"[LowLevelControlG1] Connected ({publisher_state}).")

    @property
    def has_publisher(self) -> bool:
        """Whether a live LowCmd DDS writer is currently owned.

        A deferred, debug/read-only, or successfully disabled controller returns
        ``False``.  A failed/unavailable SDK ``Close()`` keeps this property
        ``True`` even after the unusable local handle is cleared, because DDS
        writer shutdown was not proven.  The property says nothing about other
        processes' writers; callers must still enforce session ownership before
        enabling a Unitree high-level service.
        """

        return bool(
            self._pub is not None or self._lowcmd_writer_shutdown_unconfirmed
        )

    @property
    def low_state_tick(self) -> int | None:
        """Latest validated device tick, used to prove a post-event sample."""

        with self._state_lock:
            return self._last_low_state_tick

    @property
    def remote_buttons(self) -> tuple[int, ...]:
        """Atomic copy of the 16 physical-remote button states.

        ``self.remote`` remains available for compatibility with older callers,
        but new control code should use this property when making a safety or
        state-transition decision from a concurrent LowState callback.
        """

        with self._state_lock:
            return tuple(int(value) for value in self.remote.button)

    @property
    def last_lowcmd_write_at(self) -> float | None:
        """Monotonic time of the most recent DDS Write confirmed successful."""

        return self._last_lowcmd_write_at

    @property
    def lowcmd_write_count(self) -> int:
        """Number of DDS writes whose SDK return value was exactly ``True``."""

        return self._lowcmd_write_count

    def wait_for_sensor_state_after(
        self,
        previous_tick: int | None,
        *,
        timeout_seconds: float = 0.25,
        should_cancel=None,
    ):
        """Return a copied sensor state from a tick newer than ``previous_tick``."""

        if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be positive and finite")
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            current_tick, state = self._validated_sensor_snapshot()
            advanced = previous_tick is None
            if previous_tick is not None and current_tick is not None:
                delta = (int(current_tick) - int(previous_tick)) & 0xFFFFFFFF
                advanced = 0 < delta < 0x80000000
            if advanced:
                return state
            if should_cancel is not None and should_cancel():
                raise InterruptedError(
                    "cancelled while waiting for post-release Unitree LowState"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "no new Unitree LowState arrived after high-level release"
                )
            time.sleep(min(self.ctrl_dt, max(0.0, deadline - time.monotonic())))

    def wait_for_remote_buttons_after(
        self,
        previous_tick: int | None,
        *,
        timeout_seconds: float = 0.25,
        should_cancel=None,
    ) -> tuple[int, tuple[int, ...]]:
        """Return buttons atomically copied from a newer validated LowState.

        Safety transitions must not combine a tick read from one callback with
        remote buttons from another.  This method applies the same hardware,
        freshness and tick-order checks as the sensor-state path while keeping
        the entire button decision tied to one LowState packet.
        """

        tick, buttons, _state = self.wait_for_remote_gate_state_after(
            previous_tick,
            timeout_seconds=timeout_seconds,
            should_cancel=should_cancel,
        )
        return tick, buttons

    def wait_for_remote_gate_state_after(
        self,
        previous_tick: int | None,
        *,
        timeout_seconds: float = 0.25,
        should_cancel=None,
    ):
        """Return tick, buttons and sensors copied from one newer LowState."""

        if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be positive and finite")
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            current_tick, buttons, state = self._validated_remote_gate_snapshot()
            advanced = previous_tick is None
            if previous_tick is not None:
                delta = (int(current_tick) - int(previous_tick)) & 0xFFFFFFFF
                advanced = 0 < delta < 0x80000000
            if advanced:
                return int(current_tick), buttons, state
            if should_cancel is not None and should_cancel():
                raise InterruptedError(
                    "cancelled while waiting for a new Unitree remote frame"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "no new Unitree LowState arrived for the remote-mode gate"
                )
            time.sleep(min(self.ctrl_dt, max(0.0, deadline - time.monotonic())))

    def enable_lowcmd_publisher(self) -> None:
        """Open the LowCmd path after high-level ownership is released.

        Construction can subscribe to LowState while deliberately deferring the
        publisher.  The official firmware stand-up runs in that receive-only
        interval.  Callers must validate fresh Develop LowState and complete
        their physical authorization gates before invoking this method;
        MotionSwitcher is an optional secondary observation on field firmware.
        """

        if self.debug:
            raise RuntimeError("read-only debug mode cannot enable LowCmd")
        if self._lowcmd_writer_shutdown_unconfirmed:
            raise RuntimeError(
                "previous Unitree LowCmd writer shutdown was not confirmed; "
                "refusing to create another writer in this process"
            )
        if self._pub is not None:
            return
        self._validated_sensor_snapshot()
        from unitree_sdk2py.core.channel import ChannelPublisher

        # No Write() occurs here; the caller's next step is the measured
        # takeover pose.  Keep the original validated session mode.
        _UnitreeMotor.init_cmd_hg(
            self.low_cmd, self.expected_mode_machine, self.mode_pr_
        )
        publisher = ChannelPublisher(TOPIC_LOWCMD, self._LowCmdHG)
        self._initialize_publisher(publisher)
        self._pub = publisher
        self._lowcmd_publisher_disabled = False
        self._lowcmd_writer_shutdown_unconfirmed = False
        print("[LowLevelControlG1] High-level release confirmed; LowCmd enabled.")

    def _initialize_publisher(self, publisher) -> None:
        """Initialize a DDS writer and fail closed after a partial failure."""

        try:
            publisher.Init()
            # Init may yield to DDS callbacks.  Close the unwritten writer if
            # feedback became stale or changed modes during initialization.
            self._validated_sensor_snapshot()
        except BaseException as init_exc:
            close = getattr(publisher, "Close", None)
            if not callable(close):
                self._lowcmd_writer_shutdown_unconfirmed = True
                self._lowcmd_publisher_disabled = True
                raise RuntimeError(
                    "Unitree LowCmd publisher Init() failed and the SDK does "
                    "not expose Close(); writer shutdown cannot be confirmed"
                ) from init_exc
            try:
                close()
            except BaseException as close_exc:
                self._lowcmd_writer_shutdown_unconfirmed = True
                self._lowcmd_publisher_disabled = True
                raise RuntimeError(
                    "Unitree LowCmd publisher Init() failed and Close() also "
                    "failed; writer shutdown cannot be confirmed"
                ) from close_exc
            # The failed writer was positively closed, so the original SDK
            # error is safe to report without permanently poisoning retries.
            raise

    def disable_lowcmd_publisher(self) -> None:
        """Close and forget this controller's LowCmd DDS writer.

        The caller **must first stop and join every policy/control thread** so
        no late frame can race the close.  This method deliberately publishes
        no damping or hold frames; any desired finite shutdown command belongs
        before the join/close boundary.  Unitree high-level motion service
        recovery must not be requested until this method has returned and
        :attr:`has_publisher` is false.

        Repeated calls, including calls in debug/read-only mode, are idempotent.
        If an SDK publisher exists but cannot prove closure through ``Close()``,
        the local command path is still invalidated and an exception is raised
        so callers cannot claim that ownership was safely handed back.
        """

        publisher = self._pub
        # Invalidate the command path before invoking SDK teardown.  Even if
        # Close() raises, no later call through this object may publish again.
        self._pub = None
        self._lowcmd_publisher_disabled = True
        if publisher is None:
            if self._lowcmd_writer_shutdown_unconfirmed:
                raise RuntimeError(
                    "Unitree LowCmd writer shutdown remains unconfirmed; "
                    "the local publisher handle is unavailable and closure "
                    "cannot be claimed"
                )
            return

        close = getattr(publisher, "Close", None)
        if not callable(close):
            self._lowcmd_writer_shutdown_unconfirmed = True
            raise RuntimeError(
                "Unitree LowCmd publisher does not expose Close(); "
                "writer shutdown cannot be confirmed"
            )
        try:
            close()
        except BaseException as exc:
            self._lowcmd_writer_shutdown_unconfirmed = True
            raise RuntimeError(
                "Unitree LowCmd publisher Close() failed; writer shutdown "
                "cannot be confirmed"
            ) from exc
        self._lowcmd_writer_shutdown_unconfirmed = False
        print("[LowLevelControlG1] LowCmd writer closed.")

    def _require_lowcmd_not_disabled(self) -> None:
        if self._lowcmd_writer_shutdown_unconfirmed:
            raise RuntimeError(
                "Unitree LowCmd writer shutdown is unconfirmed; refusing all "
                "further command writes in this process"
            )
        if self._lowcmd_publisher_disabled:
            raise RuntimeError(
                "Unitree LowCmd publisher has been disabled; "
                "enable a new writer only after an authorized ownership handoff"
            )

    def _wait_low_state(self, timeout_seconds: float | None, should_cancel=None):
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("lowstate_timeout_seconds must be positive or None")
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + float(timeout_seconds)
        )
        while True:
            with self._state_lock:
                received = self._last_low_state_tick is not None
            if received:
                return
            if should_cancel is not None and should_cancel():
                raise InterruptedError("Cancelled while waiting for Unitree LowState")
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for Unitree LowState on rt/lowstate "
                        f"after {float(timeout_seconds):.1f}s. Check --net, DDS "
                        "multicast, the robot connection, and confirm the robot "
                        "entered Develop before this task was started."
                    )
                time.sleep(min(self.ctrl_dt, remaining))
            else:
                time.sleep(self.ctrl_dt)

    def _handle_low_state(self, msg):
        now = time.monotonic()
        try:
            if int(msg.crc) != self._state_crc.Crc(msg):
                raise ValueError("CRC mismatch")
            tick = int(msg.tick)
            mode_machine = int(msg.mode_machine)
            if not 0 <= tick <= 0xFFFFFFFF or not 0 <= mode_machine <= 0xFF:
                raise ValueError("tick or mode_machine is outside its SDK field range")
            joint_qpos = np.empty(NUM_JOINT, dtype=np.float32)
            joint_qvel = np.empty(NUM_JOINT, dtype=np.float32)
            joint_torque = np.empty(NUM_JOINT, dtype=np.float32)
            for i, mid in enumerate(MotorID.FULL):
                motor = msg.motor_state[mid]
                joint_qpos[i] = motor.q
                joint_qvel[i] = motor.dq
                joint_torque[i] = motor.tau_est
            root_quat = np.asarray(
                tuple(msg.imu_state.quaternion[:4]), dtype=np.float32
            )
            root_gyro = np.asarray(
                tuple(msg.imu_state.gyroscope[:3]), dtype=np.float32
            )
            remote = RemoteController()
            remote.set(msg.wireless_remote)
            if root_quat.shape != (4,) or root_gyro.shape != (3,):
                raise ValueError("invalid IMU shape")
            values = (root_quat, root_gyro, joint_qpos, joint_qvel, joint_torque)
            if not all(np.all(np.isfinite(value)) for value in values):
                raise ValueError("NaN or infinity")
            quaternion_norm = float(np.linalg.norm(root_quat))
            if not 0.5 <= quaternion_norm <= 1.5:
                raise ValueError(f"invalid IMU quaternion norm: {quaternion_norm:.4f}")
            if not all(math.isfinite(value) for value in (
                remote.lx, remote.ly, remote.rx, remote.ry,
            )):
                raise ValueError("remote contains NaN or infinity")
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError,
                struct.error) as exc:
            with self._state_lock:
                self._low_state_error = f"Invalid Unitree LowState packet: {exc}"
            return

        with self._state_lock:
            if self.expected_mode_machine is None:
                # Learn protocol mode only from a fully validated packet.  A
                # malformed packet before this point must not select a mode.
                self.expected_mode_machine = mode_machine
                self._low_state_error = None
            elif mode_machine != self.expected_mode_machine:
                # A mode change is a session boundary.  Do not silently resume
                # if a later packet switches back; restart the handoff flow.
                self._low_state_error = (
                    f"G1 mode_machine changed: expected {self.expected_mode_machine}, "
                    f"received {mode_machine}"
                )
                return
            if self._last_low_state_tick is None:
                self._last_low_state_advanced_at = now
            else:
                delta = (tick - self._last_low_state_tick) & 0xFFFFFFFF
                if 0 < delta < 0x80000000:
                    self._last_low_state_advanced_at = now
                elif delta >= 0x80000000:
                    self._low_state_error = "Unitree LowState tick regressed"
            self._last_low_state_tick = tick
            self.low_state = msg
            self.mode_machine_ = mode_machine
            self.remote = remote
            self.joint_qpos[:] = joint_qpos
            self.joint_qvel[:] = joint_qvel
            self.joint_torque[:] = joint_torque
            self.root_quat[:] = root_quat
            self.root_gyro[:] = root_gyro

    def _validated_sensor_snapshot_timed(self):
        """Return ``(tick, state, received_at)`` from one atomic LowState frame."""

        with self._state_lock:
            error = self._low_state_error
            mode_machine = self.mode_machine_
            last_advanced = self._last_low_state_advanced_at
            tick = self._last_low_state_tick
            state = (
                self.root_quat.copy(),
                self.root_gyro.copy(),
                self.joint_qpos.copy(),
                self.joint_qvel.copy(),
            )
        if error is not None:
            raise RuntimeError(error)
        if mode_machine != self.expected_mode_machine:
            raise RuntimeError(
                f"G1 mode_machine changed: expected {self.expected_mode_machine}, "
                f"received {mode_machine}"
            )
        if last_advanced is None or time.monotonic() - last_advanced > 0.10:
            raise TimeoutError("Unitree LowState is stale for more than 100 ms")
        if not all(np.all(np.isfinite(values)) for values in state):
            raise RuntimeError("Unitree LowState contains NaN or infinity")
        quaternion_norm = float(np.linalg.norm(state[0]))
        if not 0.5 <= quaternion_norm <= 1.5:
            raise RuntimeError(
                f"Unitree IMU quaternion norm is invalid: {quaternion_norm:.4f}"
            )
        return tick, state, float(last_advanced)

    def _validated_remote_gate_snapshot_timed(self):
        """Return tick, buttons, sensors and time from one fresh locked packet."""

        with self._state_lock:
            error = self._low_state_error
            mode_machine = self.mode_machine_
            last_advanced = self._last_low_state_advanced_at
            tick = self._last_low_state_tick
            buttons = tuple(int(value) for value in self.remote.button)
            state = (
                self.root_quat.copy(),
                self.root_gyro.copy(),
                self.joint_qpos.copy(),
                self.joint_qvel.copy(),
            )
        if error is not None:
            raise RuntimeError(error)
        if mode_machine != self.expected_mode_machine:
            raise RuntimeError(
                f"G1 mode_machine changed: expected {self.expected_mode_machine}, "
                f"received {mode_machine}"
            )
        if tick is None or last_advanced is None:
            raise TimeoutError("Unitree LowState has not produced a valid frame")
        if time.monotonic() - last_advanced > 0.10:
            raise TimeoutError("Unitree LowState is stale for more than 100 ms")
        if len(buttons) != 16 or any(value not in (0, 1) for value in buttons):
            raise RuntimeError("Unitree remote button snapshot is invalid")
        if not all(np.all(np.isfinite(values)) for values in state):
            raise RuntimeError("Unitree LowState contains NaN or infinity")
        quaternion_norm = float(np.linalg.norm(state[0]))
        if not 0.5 <= quaternion_norm <= 1.5:
            raise RuntimeError(
                f"Unitree IMU quaternion norm is invalid: {quaternion_norm:.4f}"
            )
        return int(tick), buttons, state, float(last_advanced)

    def _validated_remote_gate_snapshot(self):
        """Return tick, buttons and sensors from one fresh locked packet."""

        tick, buttons, state, _received_at = (
            self._validated_remote_gate_snapshot_timed()
        )
        return tick, buttons, state

    def _validated_remote_snapshot(self) -> tuple[int, tuple[int, ...]]:
        """Return one fresh tick and its same-packet physical-remote buttons."""

        tick, buttons, _state = self._validated_remote_gate_snapshot()
        return tick, buttons

    def _validated_sensor_snapshot(self):
        """Return ``(tick, state)`` copied atomically from one LowState frame."""

        tick, state, _received_at = self._validated_sensor_snapshot_timed()
        return tick, state

    def _send(self):
        """Legacy send path with the same telemetry gate and SDK CRC."""
        self.fast_publish()

    def get_sensor_state(self):
        """Return a validated, immutable-by-callback copy of one robot frame."""

        _tick, state = self._validated_sensor_snapshot()
        return state

    def get_sensor_state_with_timestamp(self):
        """Return one validated state plus its monotonic callback receive time."""

        _tick, state, received_at = self._validated_sensor_snapshot_timed()
        return (*state, received_at)

    def get_sensor_state_with_timestamp_and_buttons(self):
        """Return sensors, callback time and buttons from one LowState packet."""

        _tick, buttons, state, received_at = (
            self._validated_remote_gate_snapshot_timed()
        )
        return (*state, received_at, buttons)

    def _write_lowcmd_confirmed(self) -> None:
        """Publish once and account it only after an explicit SDK success."""

        self._require_lowcmd_not_disabled()
        publisher = self._pub
        if publisher is None:
            raise RuntimeError("Unitree LowCmd publisher is not initialized")
        result = publisher.Write(self.low_cmd)
        if result is not True:
            raise RuntimeError(
                "Unitree LowCmd DDS Write() did not return True; command "
                f"delivery was not confirmed (returned {result!r})"
            )
        self._last_lowcmd_write_at = time.monotonic()
        self._lowcmd_write_count += 1

    # ------------------------------------------------------------------
    # Command staging API (official SDK CRC)
    # ------------------------------------------------------------------

    def fast_pack_motor_cmd(
        self,
        tar_qpos: np.ndarray,
        kps: np.ndarray | None = None,
        kds: np.ndarray | None = None,
    ) -> None:
        """Write checked PD targets into the official SDK command object."""
        self._require_lowcmd_not_disabled()
        if kps is None:
            kps = self._kps_f32
        elif kps is not self._kps_f32:
            kps = np.ascontiguousarray(kps, dtype=np.float32)
        if kds is None:
            kds = self._kds_f32
        elif kds is not self._kds_f32:
            kds = np.ascontiguousarray(kds, dtype=np.float32)
        tar_qpos = np.ascontiguousarray(tar_qpos, dtype=np.float32)
        if tar_qpos.shape != (NUM_JOINT,):
            raise ValueError(f"motor target must have shape ({NUM_JOINT},), got {tar_qpos.shape}")
        if kps.shape != (NUM_JOINT,) or kds.shape != (NUM_JOINT,):
            raise ValueError("motor gains must match the 29-DoF target shape")
        if not np.all(np.isfinite(tar_qpos)) or float(np.max(np.abs(tar_qpos))) > 10.0:
            raise ValueError("motor target is non-finite or outside the hard numeric envelope")
        if (
            not np.all(np.isfinite(kps))
            or not np.all(np.isfinite(kds))
            or np.any(kps < 0)
            or np.any(kds < 0)
            or float(np.max(kps)) > 1000.0
            or float(np.max(kds)) > 1000.0
        ):
            raise ValueError("motor gains are non-finite or outside the hard numeric envelope")

        # Every transmitted field belongs to this single official SDK object.
        cmd = self.low_cmd
        for mid in self.active_j2m_ids:
            mc = cmd.motor_cmd[mid]
            mc.q = float(tar_qpos[mid])
            _set_motor_velocity_zero(mc)
            mc.tau = 0.0
            mc.kp = float(kps[mid])
            mc.kd = float(kds[mid])

    def fast_pack_damping(self, kd_value: float = 8.0) -> None:
        """Write damping (q=kp=0, kd=kd_value) to all 35 motors.

        Mirrors :py:meth:`_UnitreeMotor.create_damping_cmd` (every motor
        slot, not just the active ones) so that disconnected / passive
        joints also receive the safety damping torque.
        """
        self._require_lowcmd_not_disabled()
        if not math.isfinite(float(kd_value)) or not 0.0 <= float(kd_value) <= 100.0:
            raise ValueError("damping gain must be finite and within 0..100")
        cmd = self.low_cmd
        for i in range(len(cmd.motor_cmd)):
            mc = cmd.motor_cmd[i]
            mc.q = 0.0
            _set_motor_velocity_zero(mc)
            mc.tau = 0.0
            mc.kp = 0.0
            mc.kd = float(kd_value)

    def fast_compute_crc(self) -> int:
        """Stamp the official SDK CRC after restoring the session header."""
        self._require_lowcmd_not_disabled()
        if self.expected_mode_machine is None:
            raise RuntimeError("Unitree LowState has not established a session mode")
        self.low_cmd.mode_machine = self.expected_mode_machine
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        return self.low_cmd.crc

    def fast_publish(self) -> None:
        self._require_lowcmd_not_disabled()
        if not self.debug:
            # Every command path, including legacy _send(), requires fresh
            # feedback with the original session mode immediately before DDS.
            self._validated_sensor_snapshot()
            # Recompute from the actual object at the write boundary, including
            # callers that changed fields after an earlier fast_compute_crc().
            self.fast_compute_crc()
            self._write_lowcmd_confirmed()

    def fast_step(
        self,
        tar_qpos: np.ndarray,
        kps: np.ndarray | None = None,
        kds: np.ndarray | None = None,
    ) -> None:
        """Stage and publish checked PD targets through the official SDK."""
        self._require_lowcmd_not_disabled()
        self.fast_pack_motor_cmd(tar_qpos, kps, kds)
        self.fast_publish()

    def step(self, tar_qpos: np.ndarray, kps=None, kds=None):
        """Send checked PD targets through the official SDK."""
        self.fast_step(tar_qpos, kps, kds)

    def set_motor_damping(self):
        """Send damping through the same feedback and publisher gates."""
        self.fast_pack_damping(kd_value=8.0)
        self.fast_publish()

    def move_to_default_pos(
        self,
        duration: float = 2.0,
        dt: float = 0.02,
        should_cancel=None,
    ) -> bool:
        """Released START->default path retained for non-CHINGMU compatibility."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        if should_cancel is not None and should_cancel():
            return False
        curr_qpos = np.zeros(NUM_JOINT, dtype=np.float32)
        _root_quat, _root_gyro, measured_qpos, _measured_qvel = (
            self.get_sensor_state()
        )
        curr_qpos[:] = measured_qpos
        num_steps = max(1, int(duration / dt))
        init_kps = np.zeros_like(self._kps)
        for i in range(1, num_steps + 1):
            if (
                self.remote_buttons[KeyMap.select] == 1
                or (should_cancel is not None and should_cancel())
            ):
                self.set_motor_damping()
                return False
            alpha = i / num_steps
            _kps = init_kps * (1 - alpha) + self._kps * alpha
            _qpos = curr_qpos * (1 - alpha) + self.default_qpos * alpha
            self.step(_qpos, _kps, self._kds)
            time.sleep(dt)
        return True

    def follow_startup_handover(
        self,
        handover: StartupHandover,
        *,
        started_at: float,
        dt: float = 0.02,
        should_cancel=None,
    ) -> bool:
        """Execute only a planner's measured-to-stand phase.

        The planner is supplied by the caller and remains available afterward
        for its settle and live-reference blend phases.  Publishing still goes
        through :meth:`step`, so debug/read-only mode keeps its existing hard
        guarantee that no LowCmd publisher exists.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        while True:
            if (
                self.remote_buttons[KeyMap.select] == 1
                or (should_cancel is not None and should_cancel())
            ):
                self.set_motor_damping()
                return False
            # Do not keep publishing through a stale/mismatched LowState while
            # the blocking startup phase is active.  This performs the same
            # 100 ms freshness and mode_machine checks as the policy loop.
            self.get_sensor_state()
            sample = handover.sample(time.monotonic() - float(started_at))
            if sample.phase != HandoverPhase.MOVE_TO_STAND:
                # Finish on an exact default-pose command before returning to
                # the caller's settle loop.
                self.step(self.default_qpos, self._kps, self._kds)
                return True
            kps = self._kps * float(sample.stiffness_weight)
            self.step(sample.joint_target, kps, self._kds)
            time.sleep(dt)


# ---------------------------------------------------------------------------
# Adapter: real sensor data -> MuJoCo State for G1TrackInferFn
# ---------------------------------------------------------------------------

class RealRobotState:
    """Creates a MuJoCo State from real-robot sensor readings.

    G1TrackInferFn.update_state() reads qpos, qvel, sensor data, and
    body xpos/xmat from State.mj_data.  We populate a phantom mj_data
    with real sensor readings and call mj_forward() so that FK quantities
    (xpos, xmat, cvel, site data) are consistent.
    """

    def __init__(self, mj_model: mujoco.MjModel):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)

    def build_state(
        self,
        root_quat: np.ndarray,
        root_gyro: np.ndarray,
        joint_qpos: np.ndarray,
        joint_qvel: np.ndarray,
    ) -> State:
        """Populate mj_data from sensor readings and compute FK."""
        d = self.mj_data
        # Root: use nominal standing height (IMU has no position)
        d.qpos[:3] = np.array([0.0, 0.0, 0.78], dtype=np.float32)
        d.qpos[3:7] = root_quat
        d.qpos[7:] = joint_qpos
        # Root angular velocity (body frame) -> cvel for correct keypoint velocities
        d.qvel[3:6] = root_gyro
        d.qvel[6:] = joint_qvel

        # IMU gyro -> write into sensordata for gyro_pelvis sensor
        sensor_id = self.mj_model.sensor("gyro_pelvis").id
        adr = self.mj_model.sensor_adr[sensor_id]
        dim = self.mj_model.sensor_dim[sensor_id]
        d.sensordata[adr: adr + dim] = root_gyro

        mujoco.mj_forward(self.mj_model, d)
        return State(mj_data=d)
