"""Unitree adapter: Windows DDS interface selection and control boundary.

The adapter does not import the DDS control module in-process. Real LowCmd
control stays inside the isolated ``src.motion.play_track`` process launched
by orchestration.

SAFETY RULE (frozen): this adapter NEVER constructs a LowCmd publisher on its
own.  Debug mode stays receive-only; any real-output path must come through the
one-shot authorization chain.
"""

from __future__ import annotations

from html import escape
import ipaddress
import os
from pathlib import Path
import tempfile
from typing import Any

from src.domain.errors import DomainError


def configure_windows_unitree_interface(
    interface_address: str, *, trace_path: Path | None = None
) -> str:
    """Patch the not-yet-initialized Unitree channel factory for Windows."""

    if os.name != "nt":
        return str(interface_address)

    try:
        address = ipaddress.ip_address(str(interface_address).strip())
    except ValueError as exc:
        raise ValueError(
            "Windows Unitree --net must be the robot-facing adapter IPv4 address"
        ) from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise ValueError("Windows Unitree DDS currently requires IPv4")
    if (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
    ):
        raise ValueError(
            "select the non-loopback Windows Ethernet IPv4 connected to G1"
        )

    if trace_path is None:
        log_root = Path(
            os.environ.get("LOCALAPPDATA", tempfile.gettempdir())
        ) / "ChingmuGeminiNative" / "logs"
        trace_path = log_root / "cyclonedds-real.log"
    trace_path = Path(trace_path).expanduser().resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    # ChannelFactory.Init reads this module global at call time.  No Domain,
    # participant, reader, or writer exists while this configuration is set.
    from unitree_sdk2py.core import channel

    channel.ChannelConfigHasInterface = f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="{escape(str(address), quote=True)}"
                          priority="default"
                          multicast="default"/>
      </Interfaces>
    </General>
    <Tracing>
      <Verbosity>config</Verbosity>
      <OutputFile>{escape(str(trace_path.resolve()))}</OutputFile>
    </Tracing>
  </Domain>
</CycloneDDS>"""
    return str(address)


class UnitreeAdapter:
    """Delegating wrapper; never opens DDS unless explicitly asked to."""

    def configure_windows_interface(self, interface_address: str) -> str:
        """Select the Windows adapter IPv4 for CycloneDDS."""
        return configure_windows_unitree_interface(interface_address)

    @property
    def real_robot_module(self) -> Any:
        """M7b boundary: the legacy DDS module is process-isolated.

        Real LowCmd control runs inside the legacy ``deploy.play_track``
        subprocess (launched by the new orchestration).  The new architecture
        never imports it in-process; callers needing DDS control must go
        through the legacy gates, not through this facade.
        """
        raise DomainError(
            "Unitree real-robot DDS control is process-isolated in the legacy "
            "deploy.play_track subprocess; the src architecture deliberately "
            "does not import deploy/real_robot.py (M7b boundary, see "
            "architecture/MODULE_ROADMAP.md)"
        )
