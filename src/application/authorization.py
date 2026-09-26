"""Ephemeral, session-bound authorization for experimental Unitree LowCmd.

The browser acknowledgement is intentionally *not* a motor-output capability.
The web supervisor keeps the actual capabilities in memory and exposes a
private per-process broker.  A managed launcher must consume the first stage,
mint a short-lived runner stage immediately before exec, and the real runner
must consume that final stage before importing/initializing Unitree DDS.

This module is standard-library-only so it can be used by every isolated Python
environment in the project.  It never imports Unitree SDK2 and never opens DDS.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import threading
import time
from typing import Any, Mapping
import urllib.parse

AUTH_SCHEMA_VERSION = 1
ENV_BROKER = "CHINGMU_REAL_AUTH_BROKER"
ENV_CAPABILITY = "CHINGMU_REAL_AUTH_CAPABILITY"
ENV_SESSION = "CHINGMU_REAL_AUTH_SESSION"
ENV_TOKEN = "CHINGMU_REAL_AUTH_TOKEN"
ENV_BINDING = "CHINGMU_REAL_AUTH_BINDING"

WEB_TO_LAUNCHER_TTL_SECONDS = 20.0
LAUNCHER_SESSION_TTL_SECONDS = 180.0
RUNNER_TTL_SECONDS = 20.0
MAX_REQUEST_BYTES = 32 * 1024


class RealOutputAuthorizationError(RuntimeError):
    """Raised when a real-output capability is absent, stale, or mismatched."""


def canonical_binding(binding: Mapping[str, Any]) -> str:
    """Return the stable JSON identity used by every authorization stage."""

    if not isinstance(binding, Mapping):
        raise RealOutputAuthorizationError("real-output binding must be an object")
    return json.dumps(
        dict(binding),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def binding_from_control_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select the fields that identify one authorized real-output session."""

    return {
        "schema_version": AUTH_SCHEMA_VERSION,
        "backend": str(config["backend"]),
        "execution_mode": str(config["execution_mode"]),
        "debug_mode": bool(config["debug_mode"]),
        "startup_handover_enabled": bool(config["startup_handover_enabled"]),
        "server_ip": str(config["server_ip"]),
        "skeleton_id": int(config["skeleton_id"]),
        "model_profile": str(config["model_profile"]),
        "unitree_network_interface": str(config["unitree_network_interface"]),
        "unitree_robot_ip": str(config["unitree_robot_ip"]),
        # Production real output is intentionally tied to the managed local
        # CHINGMU Redis reference contract, not an arbitrary live source.
        "mocap_type": "chingmu_redis",
        "redis_host": "127.0.0.1",
        "redis_port": 6379,
        "redis_key": "action_qpos_g1_packet",
    }


def _secure_runtime_directory() -> Path:
    root = Path(f"/tmp/chingmu-real-auth-{os.getuid()}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(root.stat().st_mode)
        if mode != 0o700:
            os.chmod(root, 0o700)
    except OSError as exc:
        raise RealOutputAuthorizationError(
            f"cannot secure real-output authorization directory: {exc}"
        ) from exc
    return root


class RealOutputAuthorizationBroker:
    """In-memory one-shot capability broker owned by the web supervisor."""

    def __init__(
        self,
        *,
        runtime_directory: Path | None = None,
        monotonic=time.monotonic,
    ) -> None:
        self._runtime_directory = runtime_directory
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._socket: socket.socket | None = None
        self._socket_path: Path | None = None
        self._endpoint: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def socket_path(self) -> Path | None:
        return self._socket_path

    @property
    def endpoint(self) -> str | None:
        """Private local endpoint advertised to managed child processes."""

        return self._endpoint

    def _ensure_started(self) -> None:
        if self._socket is not None:
            return
        path: Path | None = None
        if hasattr(socket, "AF_UNIX"):
            root = self._runtime_directory or _secure_runtime_directory()
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(root, 0o700)
            path = root / f"broker-{os.getpid()}-{secrets.token_hex(8)}.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(path))
                os.chmod(path, 0o600)
                listener.listen(8)
                listener.settimeout(0.25)
            except Exception:
                listener.close()
                try:
                    path.unlink()
                except OSError:
                    pass
                raise
            endpoint = str(path)
        else:
            # The official embeddable CPython builds used by Gemini Native on
            # Windows do not expose AF_UNIX.  Keep the broker private to this
            # host by binding an ephemeral IPv4 loopback port.  The random
            # capability, session and token still have to match at every
            # stage; a TCP connection by itself grants no authorization.
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.bind(("127.0.0.1", 0))
                listener.listen(8)
                listener.settimeout(0.25)
            except Exception:
                listener.close()
                raise
            host, port = listener.getsockname()[:2]
            endpoint = f"tcp://{host}:{port}"
        self._socket = listener
        self._socket_path = path
        self._endpoint = endpoint
        self._thread = threading.Thread(
            target=self._serve,
            name="real-output-authorization-broker",
            daemon=True,
        )
        self._thread.start()

    def issue(self, binding: Mapping[str, Any]) -> dict[str, str]:
        """Issue the first, short-lived launcher capability."""

        self._ensure_started()
        capability = secrets.token_urlsafe(24)
        session = secrets.token_urlsafe(24)
        token = secrets.token_urlsafe(32)
        binding_json = canonical_binding(binding)
        now = self._monotonic()
        with self._lock:
            self._discard_expired_locked(now)
            self._records[capability] = {
                "session": session,
                "token": token,
                "binding": binding_json,
                "stage": "launcher",
                "expires": now + WEB_TO_LAUNCHER_TTL_SECONDS,
            }
        assert self._endpoint is not None
        return {
            ENV_BROKER: self._endpoint,
            ENV_CAPABILITY: capability,
            ENV_SESSION: session,
            ENV_TOKEN: token,
            ENV_BINDING: binding_json,
        }

    def revoke(self, capability: str | None) -> None:
        if not capability:
            return
        with self._lock:
            self._records.pop(str(capability), None)

    def close(self) -> None:
        self._stop.set()
        listener, self._socket = self._socket, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        path, self._socket_path = self._socket_path, None
        self._endpoint = None
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass
        with self._lock:
            self._records.clear()

    def _discard_expired_locked(self, now: float) -> None:
        expired = [
            capability
            for capability, record in self._records.items()
            if float(record["expires"]) < now
        ]
        for capability in expired:
            self._records.pop(capability, None)

    def _consume(self, request: Mapping[str, Any]) -> dict[str, Any]:
        capability = str(request.get("capability", ""))
        session = str(request.get("session", ""))
        token = str(request.get("token", ""))
        stage = str(request.get("stage", ""))
        binding_json = canonical_binding(request.get("binding", {}))
        if stage not in {"launcher", "prepare_runner", "runner"}:
            raise RealOutputAuthorizationError("unknown authorization stage")
        now = self._monotonic()
        with self._lock:
            self._discard_expired_locked(now)
            record = self._records.get(capability)
            if record is None:
                raise RealOutputAuthorizationError(
                    "real-output capability is unknown, expired, or already consumed"
                )
            if record["stage"] != stage:
                raise RealOutputAuthorizationError(
                    f"real-output capability is for stage {record['stage']}, not {stage}"
                )
            if not hmac.compare_digest(str(record["session"]), session):
                raise RealOutputAuthorizationError("real-output session does not match")
            if not hmac.compare_digest(str(record["token"]), token):
                raise RealOutputAuthorizationError("real-output token does not match")
            if not hmac.compare_digest(str(record["binding"]), binding_json):
                raise RealOutputAuthorizationError(
                    "real-output configuration changed after ARM G1"
                )

            if stage == "runner":
                self._records.pop(capability, None)
                return {"ok": True, "consumed": True}

            next_token = secrets.token_urlsafe(32)
            next_stage = "prepare_runner" if stage == "launcher" else "runner"
            ttl = (
                LAUNCHER_SESSION_TTL_SECONDS
                if stage == "launcher"
                else RUNNER_TTL_SECONDS
            )
            record.update(
                token=next_token,
                stage=next_stage,
                expires=now + ttl,
            )
            return {
                "ok": True,
                "consumed": True,
                "next_stage": next_stage,
                "next_token": next_token,
            }

    def _serve(self) -> None:
        listener = self._socket
        assert listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(2.0)
                response: dict[str, Any]
                try:
                    payload = b""
                    while b"\n" not in payload:
                        chunk = connection.recv(4096)
                        if not chunk:
                            break
                        payload += chunk
                        if len(payload) > MAX_REQUEST_BYTES:
                            raise RealOutputAuthorizationError(
                                "authorization request is too large"
                            )
                    request = json.loads(payload.split(b"\n", 1)[0].decode("utf-8"))
                    if not isinstance(request, dict):
                        raise RealOutputAuthorizationError(
                            "authorization request must be an object"
                        )
                    response = self._consume(request)
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                try:
                    connection.sendall(
                        json.dumps(response, separators=(",", ":")).encode("utf-8")
                        + b"\n"
                    )
                except OSError:
                    pass


def _load_environment_binding(
    environment: Mapping[str, str], expected: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        binding = json.loads(environment[ENV_BINDING])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RealOutputAuthorizationError(
            "missing or invalid web-issued real-output binding"
        ) from exc
    if not isinstance(binding, dict):
        raise RealOutputAuthorizationError("real-output binding must be an object")
    for key, value in expected.items():
        if binding.get(key) != value:
            raise RealOutputAuthorizationError(
                f"real-output binding mismatch for {key}: "
                f"authorized={binding.get(key)!r}, actual={value!r}"
            )
    return binding


def consume_authorization_from_environment(
    stage: str,
    *,
    expected: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
    timeout: float = 3.0,
) -> str | None:
    """Consume one capability stage and return the next stage token, if any."""

    env = os.environ if environment is None else environment
    binding = _load_environment_binding(env, expected)
    try:
        broker = env[ENV_BROKER]
        capability = env[ENV_CAPABILITY]
        session = env[ENV_SESSION]
        token = env[ENV_TOKEN]
    except KeyError as exc:
        raise RealOutputAuthorizationError(
            "missing web-issued real-output capability; use ARM G1 in the web console"
        ) from exc
    request = {
        "stage": stage,
        "capability": capability,
        "session": session,
        "token": token,
        "binding": binding,
    }
    if broker.startswith("tcp://"):
        parsed = urllib.parse.urlsplit(broker)
        if (
            parsed.scheme != "tcp"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
        ):
            raise RealOutputAuthorizationError(
                "real-output TCP broker must use an IPv4 loopback endpoint"
            )
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        broker_address: str | tuple[str, int] = (
            parsed.hostname,
            parsed.port,
        )
    else:
        if not hasattr(socket, "AF_UNIX"):
            raise RealOutputAuthorizationError(
                "this runtime cannot connect to an AF_UNIX authorization broker"
            )
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        broker_address = broker
    client.settimeout(timeout)
    try:
        client.connect(broker_address)
        client.sendall(json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n")
        payload = b""
        while b"\n" not in payload:
            chunk = client.recv(4096)
            if not chunk:
                break
            payload += chunk
            if len(payload) > MAX_REQUEST_BYTES:
                raise RealOutputAuthorizationError(
                    "authorization broker response is too large"
                )
    except (OSError, socket.timeout) as exc:
        raise RealOutputAuthorizationError(
            f"cannot validate real-output capability with the web supervisor: {exc}"
        ) from exc
    finally:
        client.close()
    try:
        response = json.loads(payload.split(b"\n", 1)[0].decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RealOutputAuthorizationError(
            "invalid response from real-output authorization broker"
        ) from exc
    if not isinstance(response, dict) or response.get("ok") is not True:
        error = response.get("error", "authorization rejected") if isinstance(response, dict) else "authorization rejected"
        raise RealOutputAuthorizationError(str(error))
    next_token = response.get("next_token")
    if next_token is None:
        return None
    if not isinstance(next_token, str) or not next_token:
        raise RealOutputAuthorizationError("authorization broker returned an invalid token")
    return next_token


def require_runner_authorization(
    *,
    backend: str,
    network_interface: str,
    model_profile: str | None = None,
    redis_host: str | None = None,
    redis_port: int | None = None,
    redis_key: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Consume the final stage before a non-debug real runner can open DDS."""

    expected: dict[str, Any] = {
        "backend": str(backend),
        "debug_mode": False,
        "unitree_network_interface": str(network_interface),
    }
    if model_profile is not None:
        expected["model_profile"] = str(model_profile)
    if redis_host is not None:
        expected["redis_host"] = str(redis_host)
    if redis_port is not None:
        expected["redis_port"] = int(redis_port)
    if redis_key is not None:
        expected["redis_key"] = str(redis_key)
    result = consume_authorization_from_environment(
        "runner", expected=expected, environment=environment
    )
    if result is not None:
        raise RealOutputAuthorizationError(
            "runner authorization returned an unexpected additional stage"
        )
    if environment is None:
        # The broker record is already gone.  Remove the spent material from the
        # long-running Runner environment so it cannot leak to later children.
        for name in (
            ENV_BROKER,
            ENV_CAPABILITY,
            ENV_SESSION,
            ENV_TOKEN,
            ENV_BINDING,
        ):
            os.environ.pop(name, None)


def _stage_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consume a managed CHINGMU G1 real-output authorization stage"
    )
    parser.add_argument("stage", choices=("launcher", "prepare_runner"))
    parser.add_argument("--backend", required=True)
    parser.add_argument("--execution-mode", required=True)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--model-profile", required=True)
    args = parser.parse_args(argv)
    try:
        token = consume_authorization_from_environment(
            args.stage,
            expected={
                "backend": args.backend,
                "execution_mode": args.execution_mode,
                "debug_mode": False,
                "unitree_network_interface": args.network_interface,
                "model_profile": args.model_profile,
                "mocap_type": "chingmu_redis",
            },
        )
    except RealOutputAuthorizationError as exc:
        parser.error(str(exc))
    if token is None:
        parser.error("authorization stage did not return the required next token")
    print(token)
    return 0


class AuthorizationService:
    """One-shot real-output authorization facade (no decisions here)."""

    @property
    def env_names(self) -> dict[str, str]:
        return {
            "broker": ENV_BROKER,
            "capability": ENV_CAPABILITY,
            "session": ENV_SESSION,
            "token": ENV_TOKEN,
            "binding": ENV_BINDING,
        }

    def binding_from_control_config(
        self, config: Mapping[str, Any]
    ) -> dict[str, Any]:
        return dict(binding_from_control_config(dict(config)))

    def canonical_binding(self, binding: Mapping[str, Any]) -> str:
        return canonical_binding(dict(binding))

    def make_broker(
        self, runtime_directory: Path | None = None
    ) -> RealOutputAuthorizationBroker:
        return RealOutputAuthorizationBroker(
            runtime_directory=runtime_directory
        )


if __name__ == "__main__":
    raise SystemExit(_stage_cli())
