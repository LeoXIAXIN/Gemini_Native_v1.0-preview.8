"""Fail-closed license checks for starting a new Chingmu Gemini session.

This module deliberately has no process-control behavior.  Call
``LicenseManager.authorize_new_session()`` immediately before starting a new
managed session and reject that start when the returned status is not allowed.
An already-running session is outside this module's lifecycle and is never
terminated here.

The signed license envelope is::

    {
      "algorithm": "RS256",
      "payload": "<base64url of the original UTF-8 JSON bytes>",
      "signature": "<base64url RSA PKCS#1 v1.5 SHA-256 signature>"
    }

The signature covers the decoded payload bytes exactly.  The payload is not
canonicalized or re-serialized before verification.

Only public RSA material is installed with the application::

    {
      "schema": "chingmu-gemini-rsa-public-key-v1",
      "algorithm": "RS256",
      "modulus": "<base64url unsigned big-endian n>",
      "exponent": 65537
    }

No private key belongs in a client package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import argparse
import base64
import hashlib
import hmac
import json
import secrets
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Union
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PUBLIC_KEY_SCHEMA = "chingmu-gemini-rsa-public-key-v1"
LICENSE_SCHEMA = "chingmu-gemini-license-v1"
EXPECTED_PRODUCT = "chingmu-gemini"
EXPECTED_VERSION = "1.0"
RSA_ALGORITHM = "RS256"

DEFAULT_HTTPS_DATE_SOURCES = (
    "https://www.cloudflare.com/cdn-cgi/trace",
    "https://www.microsoft.com/",
    "https://www.aliyun.com/",
    "https://cloud.tencent.com/",
)

_MAX_DOCUMENT_BYTES = 1024 * 1024
_MAX_BASE64_TEXT = 256 * 1024
_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex(
    "3031300d060960864801650304020105000420"
)


class LicenseFormatError(ValueError):
    """A key, envelope, payload, or timestamp has an invalid safe format."""


class TrustedTimeError(RuntimeError):
    """Independent HTTPS Date sources could not establish a trusted UTC time."""


@dataclass(frozen=True)
class TimeSample:
    """A UTC value together with the independent HTTPS authority that supplied it."""

    observed_utc: datetime
    authority: str


RawTimeSample = Union[TimeSample, datetime, str, int, float]
TimeFetcher = Callable[[str, float], RawTimeSample]


@dataclass(frozen=True)
class ProviderAuthorization:
    """Result from a hardware- or server-backed authorization provider.

    ``authenticated`` may be true only after the provider has cryptographically
    verified the supplied nonce and payload digest.  Merely discovering a USB
    VID/PID, serial number, device path, network host, or account identifier is
    not authentication.

    A provider may return a trusted UTC time from a secure hardware clock or
    authenticated server response.  Local wall-clock time is not acceptable.
    """

    authenticated: bool
    trusted_utc: Optional[datetime] = None


class DongleProvider(Protocol):
    """Hardware-backed challenge-response adapter.

    Implementations must verify a response bound to both ``nonce`` and
    ``payload_sha256``.  Device enumeration alone must return unauthenticated.
    """

    def verify_challenge_response(
        self, *, nonce: bytes, payload_sha256: bytes
    ) -> ProviderAuthorization:
        ...


class CloudProvider(Protocol):
    """Authenticated cloud-license adapter reserved for a future service."""

    def request_authorization(
        self, *, nonce: bytes, payload_sha256: bytes
    ) -> ProviderAuthorization:
        ...


@dataclass(frozen=True)
class LicenseStatus:
    """Operator-safe result for a *new* session.

    The status intentionally exposes no license ID, dongle serial, cloud
    subject, machine fingerprint, account identifier, or source URL.
    """

    allowed: bool
    code: str
    message: str
    license_type: Optional[str] = None
    valid_until: Optional[str] = None

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deny(code: str, message: str, license_type: Optional[str] = None) -> LicenseStatus:
    return LicenseStatus(
        allowed=False,
        code=code,
        message=message,
        license_type=license_type,
    )


def _b64url_decode(text: Any, *, field: str) -> bytes:
    if not isinstance(text, str) or not text or len(text) > _MAX_BASE64_TEXT:
        raise LicenseFormatError(f"{field} is not a valid base64url string")
    if "=" in text:
        raise LicenseFormatError(f"{field} must be unpadded base64url")
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise LicenseFormatError(f"{field} must be ASCII base64url") from exc
    allowed = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    if any(character not in allowed for character in encoded):
        raise LicenseFormatError(f"{field} contains non-base64url characters")
    try:
        return base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4))
    except (ValueError, base64.binascii.Error) as exc:
        raise LicenseFormatError(f"{field} is not valid base64url") from exc


def _read_json_object(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_DOCUMENT_BYTES:
            raise LicenseFormatError(f"{description} has an invalid size")
        raw = path.read_bytes()
    except OSError as exc:
        raise LicenseFormatError(f"{description} is unavailable") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LicenseFormatError(f"{description} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LicenseFormatError(f"{description} must be a JSON object")
    return value


def _load_public_key(path: Path) -> tuple[int, int]:
    key = _read_json_object(path, description="public key")
    if key.get("schema") != PUBLIC_KEY_SCHEMA:
        raise LicenseFormatError("public key schema is unsupported")
    if key.get("algorithm") != RSA_ALGORITHM:
        raise LicenseFormatError("public key algorithm is unsupported")

    modulus_bytes = _b64url_decode(key.get("modulus"), field="modulus")
    modulus = int.from_bytes(modulus_bytes, "big")
    exponent = key.get("exponent")
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        raise LicenseFormatError("public key exponent must be an integer")
    if modulus.bit_length() < 2048:
        raise LicenseFormatError("RSA modulus must contain at least 2048 bits")
    if exponent < 3 or exponent % 2 == 0:
        raise LicenseFormatError("RSA public exponent is invalid")
    return modulus, exponent


def _rsa_pkcs1_v1_5_sha256_verify(
    payload_bytes: bytes,
    signature: bytes,
    *,
    modulus: int,
    exponent: int,
) -> bool:
    key_size = (modulus.bit_length() + 7) // 8
    if len(signature) != key_size:
        return False
    signature_integer = int.from_bytes(signature, "big")
    if signature_integer <= 0 or signature_integer >= modulus:
        return False

    encoded_message = pow(signature_integer, exponent, modulus).to_bytes(
        key_size, "big"
    )
    digest_info = _SHA256_DIGEST_INFO_PREFIX + hashlib.sha256(payload_bytes).digest()
    padding_length = key_size - len(digest_info) - 3
    if padding_length < 8:
        return False
    expected = b"\x00\x01" + b"\xff" * padding_length + b"\x00" + digest_info
    return hmac.compare_digest(encoded_message, expected)


def _load_verified_payload(
    license_path: Path, public_key_path: Path
) -> tuple[Mapping[str, Any], bytes]:
    envelope = _read_json_object(license_path, description="license")
    algorithm = envelope.get("algorithm", RSA_ALGORITHM)
    if algorithm != RSA_ALGORITHM:
        raise LicenseFormatError("license algorithm is unsupported")
    payload_bytes = _b64url_decode(envelope.get("payload"), field="payload")
    signature = _b64url_decode(envelope.get("signature"), field="signature")
    if not payload_bytes or len(payload_bytes) > _MAX_DOCUMENT_BYTES:
        raise LicenseFormatError("license payload has an invalid size")

    modulus, exponent = _load_public_key(public_key_path)
    if not _rsa_pkcs1_v1_5_sha256_verify(
        payload_bytes,
        signature,
        modulus=modulus,
        exponent=exponent,
    ):
        raise LicenseFormatError("license signature is invalid")

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LicenseFormatError("signed license payload is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise LicenseFormatError("signed license payload must be a JSON object")
    return payload, payload_bytes


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LicenseFormatError(f"{field} must be an RFC3339 UTC timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise LicenseFormatError(f"{field} is not a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LicenseFormatError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_https_date(url: str, timeout: float) -> TimeSample:
    """Fetch an HTTP Date header over TLS without consulting local wall time."""

    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise TrustedTimeError("trusted time source must use HTTPS")
    request = Request(
        url,
        # GET is used because several otherwise suitable Date authorities
        # reject HEAD.  The body is never read and Range limits cooperative
        # endpoints to one byte.
        method="GET",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Range": "bytes=0-0",
            "User-Agent": "Chingmu-Gemini-License-Time/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()
            final = urlparse(final_url)
            if final.scheme.lower() != "https" or not final.hostname:
                raise TrustedTimeError("trusted time source redirected outside HTTPS")
            date_header = response.headers.get("Date")
    except TrustedTimeError:
        raise
    except Exception as exc:
        raise TrustedTimeError("trusted time source is unavailable") from exc
    if not date_header:
        raise TrustedTimeError("trusted time source omitted its Date header")
    try:
        observed = parsedate_to_datetime(date_header)
    except (TypeError, ValueError) as exc:
        raise TrustedTimeError("trusted time source returned an invalid Date header") from exc
    if observed.tzinfo is None or observed.utcoffset() is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return TimeSample(
        observed_utc=observed.astimezone(timezone.utc),
        authority=final.hostname.lower(),
    )


class HttpsDateArbiter:
    """Establish current UTC from multiple independent HTTPS Date authorities."""

    def __init__(
        self,
        *,
        sources: Sequence[str] = DEFAULT_HTTPS_DATE_SOURCES,
        fetcher: TimeFetcher = fetch_https_date,
        timeout: float = 4.0,
        minimum_sources: int = 2,
        maximum_skew_seconds: float = 300.0,
    ) -> None:
        self.sources = tuple(sources)
        self.fetcher = fetcher
        self.timeout = float(timeout)
        self.minimum_sources = int(minimum_sources)
        self.maximum_skew_seconds = float(maximum_skew_seconds)

    @staticmethod
    def _source_authority(source: str) -> str:
        parsed = urlparse(source)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise TrustedTimeError("all trusted time sources must use HTTPS")
        return parsed.hostname.lower()

    def _normalize_sample(self, raw: RawTimeSample, source: str) -> TimeSample:
        authority = self._source_authority(source)
        if isinstance(raw, TimeSample):
            value = raw.observed_utc
            authority = raw.authority.strip().lower()
            if not authority:
                raise TrustedTimeError("trusted time authority is empty")
        elif isinstance(raw, datetime):
            value = raw
        elif isinstance(raw, str):
            try:
                value = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                value = _parse_utc(raw, field="trusted time")
        elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TrustedTimeError("trusted time source returned an unsupported value")
        else:
            value = datetime.fromtimestamp(float(raw), tz=timezone.utc)
        if value.tzinfo is None or value.utcoffset() is None:
            raise TrustedTimeError("trusted time source returned a timezone-less value")
        return TimeSample(value.astimezone(timezone.utc), authority)

    def current_utc(self) -> datetime:
        if self.minimum_sources < 2:
            raise TrustedTimeError("at least two trusted time sources are required")
        configured_authorities = {
            self._source_authority(source) for source in self.sources
        }
        if len(configured_authorities) < self.minimum_sources:
            raise TrustedTimeError("trusted time sources are not independent")

        by_authority: dict[str, datetime] = {}
        for source in self.sources:
            try:
                sample = self._normalize_sample(
                    self.fetcher(source, self.timeout), source
                )
            except Exception:
                continue
            by_authority.setdefault(sample.authority, sample.observed_utc)

        if len(by_authority) < self.minimum_sources:
            raise TrustedTimeError("too few independent HTTPS Date sources responded")
        timestamps = [value.timestamp() for value in by_authority.values()]
        if max(timestamps) - min(timestamps) > self.maximum_skew_seconds:
            raise TrustedTimeError("independent HTTPS Date sources disagree")
        return datetime.fromtimestamp(statistics.median(timestamps), tz=timezone.utc)


class LicenseManager:
    """Verify the license required to start one new managed session."""

    def __init__(
        self,
        *,
        public_key_path: Union[str, Path],
        license_path: Union[str, Path],
        time_arbiter: Optional[HttpsDateArbiter] = None,
        time_fetcher: Optional[TimeFetcher] = None,
        time_sources: Sequence[str] = DEFAULT_HTTPS_DATE_SOURCES,
        dongle_provider: Optional[DongleProvider] = None,
        cloud_provider: Optional[CloudProvider] = None,
        expected_product: str = EXPECTED_PRODUCT,
        expected_version: str = EXPECTED_VERSION,
    ) -> None:
        if time_arbiter is not None and time_fetcher is not None:
            raise ValueError("pass either time_arbiter or time_fetcher, not both")
        self.public_key_path = Path(public_key_path)
        self.license_path = Path(license_path)
        self.time_arbiter = time_arbiter or HttpsDateArbiter(
            sources=time_sources,
            fetcher=time_fetcher or fetch_https_date,
        )
        self.dongle_provider = dongle_provider
        self.cloud_provider = cloud_provider
        self.expected_product = expected_product
        self.expected_version = expected_version

    def _validate_common_payload(self, payload: Mapping[str, Any]) -> str:
        if payload.get("schema") != LICENSE_SCHEMA:
            raise LicenseFormatError("license payload schema is unsupported")
        if payload.get("product") != self.expected_product:
            raise LicenseFormatError("license is for a different product")
        if payload.get("version") != self.expected_version:
            raise LicenseFormatError("license is for a different product version")
        license_type = payload.get("license_type")
        if license_type not in {"online_trial", "dongle", "cloud"}:
            raise LicenseFormatError("license type is unsupported")
        return license_type

    @staticmethod
    def _validate_window(
        payload: Mapping[str, Any],
        current_utc: datetime,
        *,
        required: bool,
    ) -> Optional[str]:
        has_not_before = "not_before" in payload
        has_not_after = "not_after" in payload
        if required and (not has_not_before or not has_not_after):
            raise LicenseFormatError("online trial requires a signed validity window")
        if has_not_before != has_not_after:
            raise LicenseFormatError("license validity window is incomplete")
        if not has_not_before:
            return None

        not_before = _parse_utc(payload["not_before"], field="not_before")
        not_after = _parse_utc(payload["not_after"], field="not_after")
        if not_after <= not_before:
            raise LicenseFormatError("license validity window is invalid")
        if (
            not isinstance(current_utc, datetime)
            or current_utc.tzinfo is None
            or current_utc.utcoffset() is None
        ):
            raise LicenseFormatError("trusted time must include a UTC offset")
        current = current_utc.astimezone(timezone.utc)
        if current < not_before:
            raise LicenseFormatError("license validity has not started")
        if current > not_after:
            raise LicenseFormatError("license validity has expired")
        return _format_utc(not_after)

    def _authorize_online_trial(
        self, payload: Mapping[str, Any], license_type: str
    ) -> LicenseStatus:
        try:
            trusted_now = self.time_arbiter.current_utc()
        except TrustedTimeError:
            return _deny(
                "trusted_time_unavailable",
                "无法通过独立 HTTPS 时间源确认当前时间。",
                license_type,
            )
        try:
            valid_until = self._validate_window(payload, trusted_now, required=True)
        except LicenseFormatError as exc:
            return _deny("license_time_invalid", str(exc), license_type)
        return LicenseStatus(
            allowed=True,
            code="authorized",
            message="许可证有效。",
            license_type=license_type,
            valid_until=valid_until,
        )

    def _authorize_provider(
        self,
        payload: Mapping[str, Any],
        payload_bytes: bytes,
        license_type: str,
    ) -> LicenseStatus:
        nonce = secrets.token_bytes(32)
        payload_sha256 = hashlib.sha256(payload_bytes).digest()
        try:
            if license_type == "dongle":
                if self.dongle_provider is None:
                    return _deny(
                        "dongle_unconfigured",
                        "加密狗授权提供程序尚未配置。",
                        license_type,
                    )
                result = self.dongle_provider.verify_challenge_response(
                    nonce=nonce,
                    payload_sha256=payload_sha256,
                )
            else:
                if self.cloud_provider is None:
                    return _deny(
                        "cloud_unconfigured",
                        "云端授权提供程序尚未配置。",
                        license_type,
                    )
                result = self.cloud_provider.request_authorization(
                    nonce=nonce,
                    payload_sha256=payload_sha256,
                )
        except Exception:
            return _deny(
                f"{license_type}_provider_error",
                "授权提供程序未能完成加密验证。",
                license_type,
            )

        if not isinstance(result, ProviderAuthorization) or not result.authenticated:
            return _deny(
                f"{license_type}_authentication_failed",
                "加密授权验证失败。",
                license_type,
            )

        has_window = "not_before" in payload or "not_after" in payload
        valid_until: Optional[str] = None
        if has_window:
            if result.trusted_utc is None:
                return _deny(
                    f"{license_type}_trusted_time_missing",
                    "授权提供程序未提供受信时间，无法校验有效期。",
                    license_type,
                )
            try:
                valid_until = self._validate_window(
                    payload, result.trusted_utc, required=False
                )
            except LicenseFormatError as exc:
                return _deny("license_time_invalid", str(exc), license_type)
        return LicenseStatus(
            allowed=True,
            code="authorized",
            message="许可证有效。",
            license_type=license_type,
            valid_until=valid_until,
        )

    def authorize_new_session(self) -> LicenseStatus:
        """Return authorization for a new session; never terminate a process."""

        try:
            payload, payload_bytes = _load_verified_payload(
                self.license_path, self.public_key_path
            )
            license_type = self._validate_common_payload(payload)
        except LicenseFormatError:
            return _deny(
                "license_invalid",
                "许可证缺失、损坏、签名无效或不适用于当前产品。",
            )

        if license_type == "online_trial":
            return self._authorize_online_trial(payload, license_type)
        return self._authorize_provider(payload, payload_bytes, license_type)

    # A descriptive alias for integrations that use "check" terminology.
    check_new_session = authorize_new_session


def evaluate_license(
    public_key_path: Union[str, Path],
    license_path: Union[str, Path],
    *,
    time_arbiter: Optional[HttpsDateArbiter] = None,
    time_fetcher: Optional[TimeFetcher] = None,
    time_sources: Sequence[str] = DEFAULT_HTTPS_DATE_SOURCES,
    dongle_provider: Optional[DongleProvider] = None,
    cloud_provider: Optional[CloudProvider] = None,
    expected_product: str = EXPECTED_PRODUCT,
    expected_version: str = EXPECTED_VERSION,
) -> LicenseStatus:
    """Convenience API used by a web backend immediately before session start."""

    return LicenseManager(
        public_key_path=public_key_path,
        license_path=license_path,
        time_arbiter=time_arbiter,
        time_fetcher=time_fetcher,
        time_sources=time_sources,
        dongle_provider=dongle_provider,
        cloud_provider=cloud_provider,
        expected_product=expected_product,
        expected_version=expected_version,
    ).authorize_new_session()


_DEFAULT_LICENSE_BASE = Path(__file__).resolve().parents[2] / "resources"


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether Chingmu Gemini may start a new session."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the installed signed license for one new session",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=_DEFAULT_LICENSE_BASE,
        help=(
            "installed application directory containing "
            "license/public_key.json and license/ChingmuGemini.license"
        ),
    )
    arguments = parser.parse_args(argv)
    if not arguments.check:
        parser.error("--check is required")

    base_dir = arguments.base_dir.resolve()
    status = evaluate_license(
        base_dir / "license" / "public_key.json",
        base_dir / "license" / "ChingmuGemini.license",
    )
    # This is the same deliberately minimal, operator-safe object returned by
    # the Python API.  No payload/provider identifiers are ever printed.
    print(
        json.dumps(
            status.to_public_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if status.allowed else 4


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns a code and never kills another process."""

    try:
        return _cli(argv)
    except SystemExit:
        raise
    except Exception:
        # Unexpected implementation/runtime errors remain fail-closed and do
        # not leak file paths, IDs, provider details, or exception text.
        status = _deny(
            "license_check_error",
            "许可证检查未能安全完成。",
        )
        print(
            json.dumps(
                status.to_public_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 4


__all__ = [
    "CloudProvider",
    "DEFAULT_HTTPS_DATE_SOURCES",
    "DongleProvider",
    "EXPECTED_PRODUCT",
    "EXPECTED_VERSION",
    "HttpsDateArbiter",
    "LICENSE_SCHEMA",
    "LicenseManager",
    "LicenseStatus",
    "ProviderAuthorization",
    "PUBLIC_KEY_SCHEMA",
    "RSA_ALGORITHM",
    "TimeSample",
    "TrustedTimeError",
    "evaluate_license",
    "fetch_https_date",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
