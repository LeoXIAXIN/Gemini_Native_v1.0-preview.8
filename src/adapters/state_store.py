"""StateStore adapter: RESP2 client and stable key contract.

This adapter is transport-only and contains no business decisions.

Read/write via this adapter never touches files; the store itself is the
in-memory RESP server started by the pipeline supervisors.
"""

from __future__ import annotations

# Direct-script bootstrap: the pipeline supervisors launch this file as a
# script; make the src package importable first.
import sys as _sys
from pathlib import Path as _Path

if str(_Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import socket
from typing import Any, Callable, Mapping

from src.domain.enums import BridgeStatus
from src.domain.errors import CodecError

STATE_HOST = "127.0.0.1"
STATE_PORT = 6379
DEFAULT_TIMEOUT_SECONDS = 0.75


class StateStoreKeys:
    """Frozen key names (architecture/PHASE0_INTERFACE_FREEZE.md §6)."""

    BRIDGE_STATUS = "chingmu_gmr_bridge_status"
    QPOS_PACKET = "action_qpos_g1_packet"
    MIMIC_PACKET = "action_mimic_g1_packet"
    SIM_START_COMMAND = "g1_sim_start_command"
    SAFETY_CONTROL = "g1_safety_control"
    SAFETY_STATUS = "g1_safety_status"


ConnectionFactory = Callable[[tuple[str, int], float], Any]


def _encode_command(*parts: str | bytes) -> bytes:
    encoded = [
        part if isinstance(part, bytes) else part.encode("utf-8") for part in parts
    ]
    chunks = [f"*{len(encoded)}\r\n".encode("ascii")]
    for part in encoded:
        chunks.extend((f"${len(part)}\r\n".encode("ascii"), part, b"\r\n"))
    return b"".join(chunks)


class StateStoreClient:
    """Minimal RESP2 client speaking to the legacy in-memory store."""

    def __init__(
        self,
        host: str = STATE_HOST,
        port: int = STATE_PORT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self.address = (str(host), int(port))
        self.timeout = float(timeout)
        self._connect = connection_factory or socket.create_connection

    def _request(self, *parts: str) -> bytes | int | None:
        client = self._connect(self.address, self.timeout)
        try:
            client.settimeout(self.timeout)
            client.sendall(_encode_command(*parts))
            stream = client.makefile("rb")
            first = stream.readline()
            if first.startswith(b"+"):
                return first[1:-2]
            if first.startswith(b":"):
                try:
                    return int(first[1:-2])
                except ValueError as exc:
                    raise CodecError("invalid RESP integer reply") from exc
            if first == b"$-1\r\n":
                return None
            if first.startswith(b"$"):
                try:
                    size = int(first[1:-2])
                except ValueError as exc:
                    raise CodecError("invalid RESP bulk length") from exc
                payload = stream.read(size)
                if stream.read(2) != b"\r\n":
                    raise CodecError("truncated RESP bulk payload")
                return payload
            if first.startswith(b"-"):
                raise CodecError(
                    first[1:-2].decode("utf-8", errors="replace") or "RESP error"
                )
            raise CodecError("unexpected RESP reply type")
        finally:
            client.close()

    def ping(self) -> bool:
        return self._request("PING") == b"PONG"

    def get(self, key: str) -> bytes | None:
        return self._request("GET", key)

    def get_text(self, key: str) -> str | None:
        value = self.get(key)
        return value.decode("utf-8", errors="replace") if value else None

    def set(self, key: str, value: str | bytes, ttl_seconds: float | None = None) -> None:
        payload = value.encode("utf-8") if isinstance(value, str) else value
        if ttl_seconds is None:
            reply = self._request("SET", key, payload)
        else:
            reply = self._request(
                "SET", key, payload, "EX", str(int(ttl_seconds))
            )
        if reply != b"OK":
            raise CodecError(f"SET rejected: {reply!r}")

    def delete(self, *keys: str) -> int:
        reply = self._request("DEL", *keys)
        if isinstance(reply, int):
            return reply
        if reply is None:
            return 0
        try:
            return int(reply)
        except ValueError as exc:
            raise CodecError("invalid DEL reply") from exc


def parse_bridge_status(value: str | bytes | None) -> tuple[BridgeStatus, str]:
    """Classify a ``chingmu_gmr_bridge_status`` value (prefix-based).

    Returns (status, detail) where detail is the ``live:``/``error:`` suffix.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = (value or "").strip()
    if not text:
        return BridgeStatus.STARTING_GMR_RECEIVER, ""
    lowered = text.lower()
    if lowered.startswith(BridgeStatus.LIVE.value + ":"):
        return BridgeStatus.LIVE, text.split(":", 1)[1]
    if lowered.startswith(BridgeStatus.INVALID_SKELETON.value + ":"):
        return BridgeStatus.INVALID_SKELETON, text.split(":", 1)[1]
    if lowered.startswith(BridgeStatus.ERROR.value + ":"):
        return BridgeStatus.ERROR, text.split(":", 1)[1]
    for status in BridgeStatus:
        if lowered == status.value:
            return status, ""
    raise CodecError(f"unknown bridge status value: {text!r}")


# In-memory RESP2 server used by the local pipeline supervisors.

import argparse
from collections.abc import Sequence
import socketserver
import threading
import time
from typing import BinaryIO


class RespProtocolError(ValueError):
    """Raised when a client sends malformed RESP data."""


def _read_line(stream: BinaryIO) -> bytes:
    line = stream.readline()
    if not line:
        raise EOFError
    if not line.endswith(b"\r\n"):
        raise RespProtocolError("RESP line is missing CRLF")
    return line[:-2]


def _read_request(stream: BinaryIO) -> list[bytes]:
    marker = stream.read(1)
    if not marker:
        raise EOFError
    if marker != b"*":
        raise RespProtocolError("only RESP arrays are accepted")
    try:
        count = int(_read_line(stream))
    except ValueError as exc:
        raise RespProtocolError("invalid RESP array length") from exc
    if count < 0 or count > 64:
        raise RespProtocolError("unsupported RESP array length")

    items: list[bytes] = []
    for _ in range(count):
        if stream.read(1) != b"$":
            raise RespProtocolError("only bulk-string array items are accepted")
        try:
            size = int(_read_line(stream))
        except ValueError as exc:
            raise RespProtocolError("invalid RESP bulk-string length") from exc
        if size < 0 or size > 64 * 1024 * 1024:
            raise RespProtocolError("unsupported RESP bulk-string length")
        payload = stream.read(size)
        if len(payload) != size or stream.read(2) != b"\r\n":
            raise RespProtocolError("truncated RESP bulk string")
        items.append(payload)
    return items


def _simple(value: str) -> bytes:
    return b"+" + value.encode("utf-8") + b"\r\n"


def _error(value: str) -> bytes:
    return b"-ERR " + value.encode("utf-8", errors="replace") + b"\r\n"


def _integer(value: int) -> bytes:
    return f":{value}\r\n".encode("ascii")


def _bulk(value: bytes | None) -> bytes:
    if value is None:
        return b"$-1\r\n"
    return f"${len(value)}\r\n".encode("ascii") + value + b"\r\n"


class InMemoryRespStore:
    """Thread-safe byte key/value store."""

    def __init__(self) -> None:
        self._values: dict[bytes, bytes] = {}
        self._expires_at: dict[bytes, float] = {}
        self._lock = threading.RLock()

    def get(self, key: bytes) -> bytes | None:
        with self._lock:
            expires_at = self._expires_at.get(key)
            if expires_at is not None and time.monotonic() >= expires_at:
                self._values.pop(key, None)
                self._expires_at.pop(key, None)
                return None
            return self._values.get(key)

    def set(self, key: bytes, value: bytes, ttl_seconds: float | None = None) -> None:
        with self._lock:
            self._values[key] = value
            if ttl_seconds is None:
                self._expires_at.pop(key, None)
            else:
                self._expires_at[key] = time.monotonic() + ttl_seconds

    def delete(self, keys: Sequence[bytes]) -> int:
        with self._lock:
            removed = 0
            for key in keys:
                if key in self._values:
                    del self._values[key]
                    self._expires_at.pop(key, None)
                    removed += 1
            return removed

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._expires_at.clear()


class RespRequestHandler(socketserver.StreamRequestHandler):
    """RESP2 command handler."""

    server: "RespServer"

    def handle(self) -> None:
        while True:
            try:
                request = _read_request(self.rfile)
            except (EOFError, ConnectionError, OSError):
                return
            except RespProtocolError as exc:
                self.wfile.write(_error(str(exc)))
                self.wfile.flush()
                return

            response, should_close = self.server.execute(request)
            self.wfile.write(response)
            self.wfile.flush()
            if should_close:
                return


class RespServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded local RESP server."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        self.store = InMemoryRespStore()
        super().__init__(address, RespRequestHandler)

    def execute(self, request: list[bytes]) -> tuple[bytes, bool]:
        if not request:
            return _error("empty command"), False
        try:
            command = request[0].decode("ascii").upper()
        except UnicodeDecodeError:
            return _error("command must be ASCII"), False
        args = request[1:]

        if command == "PING":
            if len(args) > 1:
                return _error("wrong number of arguments for 'ping'"), False
            return (_simple("PONG") if not args else _bulk(args[0])), False
        if command == "GET":
            if len(args) != 1:
                return _error("wrong number of arguments for 'get'"), False
            return _bulk(self.store.get(args[0])), False
        if command == "SET":
            if len(args) < 2:
                return _error("wrong number of arguments for 'set'"), False
            ttl_seconds: float | None = None
            index = 2
            while index < len(args):
                option = args[index].decode("ascii", errors="ignore").upper()
                if option not in {"EX", "PX"} or index + 1 >= len(args):
                    return _error("unsupported SET option"), False
                try:
                    duration = int(args[index + 1])
                except ValueError:
                    return _error("invalid SET expiration"), False
                if duration <= 0:
                    return _error("invalid expire time in 'set' command"), False
                ttl_seconds = duration if option == "EX" else duration / 1000.0
                index += 2
            self.store.set(args[0], args[1], ttl_seconds=ttl_seconds)
            return _simple("OK"), False
        if command in {"DEL", "UNLINK"}:
            if not args:
                return _error(f"wrong number of arguments for '{command.lower()}'"), False
            return _integer(self.store.delete(args)), False
        if command == "SELECT":
            if len(args) != 1 or args[0] != b"0":
                return _error("only database 0 is supported"), False
            return _simple("OK"), False
        if command == "FLUSHDB":
            if args:
                return _error("wrong number of arguments for 'flushdb'"), False
            self.store.clear()
            return _simple("OK"), False
        if command == "CLIENT":
            if not args:
                return _error("wrong number of arguments for 'client'"), False
            subcommand = args[0].decode("ascii", errors="ignore").upper()
            if subcommand in {"SETINFO", "SETNAME"}:
                return _simple("OK"), False
            if subcommand == "GETNAME":
                return _bulk(None), False
            return _error("unsupported CLIENT subcommand"), False
        if command == "QUIT":
            return _simple("OK"), True
        return _error(f"unknown command '{command.lower()}'"), False


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loopback-only in-memory RESP2 store for Gemini Native."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing non-loopback host; this store is local-only.")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")

    with RespServer((args.host, args.port)) as server:
        actual_host, actual_port = server.server_address[:2]
        print(
            f"Gemini Native local state store ready on {actual_host}:{actual_port}",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
