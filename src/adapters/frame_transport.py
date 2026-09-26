"""Local frame transport shared by the isolated CHINGMU SDK processes.

The proprietary SDK must remain in a small system-Python process, while GMR
runs in its Conda environment.  Managed launchers use an AF_UNIX datagram
socket between those two processes so WSL mirrored-networking and firewall
rules cannot interfere with Linux-to-Linux localhost traffic.  UDP remains
available as an explicit compatibility mode for older standalone commands.
"""

from __future__ import annotations

import argparse
import errno
import os
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Union

DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 15150
FRAME_TRANSPORTS = ("unix", "udp")
SocketDestination = Union[str, tuple[str, int]]
FRAME_SOCKET_BASE = Path("/tmp")


def default_frame_socket_path() -> Path:
    """Return a short, per-user socket path shared by both Python runtimes."""

    getuid = getattr(os, "getuid", None)
    uid = getuid() if getuid is not None else 0
    return FRAME_SOCKET_BASE / f"chingmu-gmr-{uid}" / "frames.sock"


def allowed_frame_socket_root() -> Path:
    getuid = getattr(os, "getuid", None)
    uid = getuid() if getuid is not None else 0
    return FRAME_SOCKET_BASE / f"chingmu-gmr-{uid}"


def add_frame_transport_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_transport: str = "udp",
) -> None:
    """Add the common local frame-transport CLI without hiding legacy UDP."""

    if default_transport not in FRAME_TRANSPORTS:
        raise ValueError(f"Unsupported default frame transport: {default_transport}")
    parser.add_argument(
        "--frame-transport",
        choices=FRAME_TRANSPORTS,
        default=default_transport,
        help="Local SDK-to-GMR transport. Managed launchers use 'unix'.",
    )
    parser.add_argument(
        "--frame-socket",
        type=Path,
        default=default_frame_socket_path(),
        help="AF_UNIX datagram path used when --frame-transport=unix.",
    )
    parser.add_argument("--udp-host", default=DEFAULT_UDP_HOST)
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)


def _same_user(st: os.stat_result) -> bool:
    getuid = getattr(os, "getuid", None)
    return getuid is None or st.st_uid == getuid()


def _validate_unix_path(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Frame socket path must be an absolute safe path: {path}")
    if path.parent != allowed_frame_socket_root():
        raise ValueError(
            f"Frame socket must be directly inside {allowed_frame_socket_root()}: {path}"
        )


def _ensure_parent_directory(path: Path) -> None:
    _validate_unix_path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_stat = os.lstat(path.parent)
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise RuntimeError(f"Frame socket parent is not a real directory: {path.parent}")
    if not _same_user(parent_stat):
        raise PermissionError(f"Frame socket parent belongs to another user: {path.parent}")
    if _same_user(parent_stat):
        os.chmod(path.parent, 0o700)


def _unix_receiver_is_active(path: Path) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        probe.connect(str(path))
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ECONNREFUSED):
            return False
        raise
    else:
        return True
    finally:
        probe.close()


def _prepare_unix_receiver_path(path: Path) -> None:
    _ensure_parent_directory(path)
    if not os.path.lexists(path):
        return
    existing = os.lstat(path)
    if stat.S_ISLNK(existing.st_mode) or not stat.S_ISSOCK(existing.st_mode):
        raise RuntimeError(f"Refusing to replace non-socket frame path: {path}")
    if not _same_user(existing):
        raise PermissionError(f"Frame socket belongs to another user: {path}")
    if _unix_receiver_is_active(path):
        raise OSError(errno.EADDRINUSE, "frame socket is already active", str(path))
    path.unlink()


@dataclass
class FrameSender:
    socket: socket.socket
    destination: SocketDestination
    endpoint: str

    def send(self, packet: bytes) -> bool:
        """Queue one frame without allowing a slow consumer to block capture."""

        try:
            return self.socket.sendto(packet, self.destination) == len(packet)
        except BlockingIOError:
            return False
        except OSError as error:
            if error.errno in (
                errno.EAGAIN,
                errno.ENOBUFS,
                errno.ENOENT,
                errno.ECONNREFUSED,
            ):
                return False
            raise

    def close(self) -> None:
        self.socket.close()


@dataclass
class FrameReceiver:
    socket: socket.socket
    endpoint: str
    socket_path: Path | None = None
    socket_device: int | None = None
    socket_inode: int | None = None

    def recvfrom(self, max_size: int) -> tuple[bytes, object]:
        return self.socket.recvfrom(max_size)

    def settimeout(self, timeout: float) -> None:
        self.socket.settimeout(timeout)

    def close(self) -> None:
        self.socket.close()
        if (
            self.socket_path is None
            or self.socket_device is None
            or self.socket_inode is None
        ):
            return
        try:
            current = os.lstat(self.socket_path)
        except FileNotFoundError:
            return
        if (
            stat.S_ISSOCK(current.st_mode)
            and _same_user(current)
            and current.st_dev == self.socket_device
            and current.st_ino == self.socket_inode
        ):
            self.socket_path.unlink()


def open_frame_sender(
    transport: str,
    frame_socket: Path,
    udp_host: str,
    udp_port: int,
) -> FrameSender:
    if transport == "unix":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.setblocking(False)
        path = frame_socket.expanduser()
        _validate_unix_path(path)
        return FrameSender(sock, str(path), f"unix:{path}")
    if transport == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        return FrameSender(sock, (udp_host, udp_port), f"udp:{udp_host}:{udp_port}")
    raise ValueError(f"Unsupported frame transport: {transport}")


def open_frame_receiver(
    transport: str,
    frame_socket: Path,
    udp_host: str,
    udp_port: int,
    *,
    receive_buffer: int = 1 << 20,
) -> FrameReceiver:
    if transport == "unix":
        path = frame_socket.expanduser()
        _prepare_unix_receiver_path(path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer)
        try:
            old_umask = os.umask(0o177)
            try:
                sock.bind(str(path))
            finally:
                os.umask(old_umask)
            os.chmod(path, 0o600)
            identity = os.lstat(path)
        except Exception:
            sock.close()
            raise
        return FrameReceiver(
            sock,
            f"unix:{path}",
            path,
            identity.st_dev,
            identity.st_ino,
        )
    if transport == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer)
        try:
            sock.bind((udp_host, udp_port))
        except Exception:
            sock.close()
            raise
        return FrameReceiver(sock, f"udp:{udp_host}:{udp_port}")
    raise ValueError(f"Unsupported frame transport: {transport}")
