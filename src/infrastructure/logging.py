"""Control-pipeline log classification."""

from __future__ import annotations

import re

LEVELS = ("info", "warning", "error")
REJECTED_COUNT_RE = re.compile(r"\brejected\s+(?P<count>[0-9]+)\b", re.I)


def classify_component(line: str, backend: str | None) -> str:
    lowered = line.lower()
    if "sender fps" in lowered or "[sdk " in lowered or "vrpn" in lowered:
        return "chingmu"
    if "bridge" in lowered or "ik frame" in lowered or "use ik config" in lowered:
        return "gmr"
    if "humanoid-gpt" in lowered or "[onnx]" in lowered:
        return "humanoid_gpt"
    if "gmr preview" in lowered or "gmr exact preview" in lowered:
        return "gmr"
    if (
        "policy device" in lowered
        or "policy loaded" in lowered
        or "motor id" in lowered
        or "degrees of freedom" in lowered
    ):
        return "twist"
    return backend or "launcher"


def classify_level(line: str) -> str:
    lowered = line.lower()
    if any(
        marker in lowered
        for marker in (
            "traceback",
            "segmentation fault",
            "fatal python error",
            "address already in use",
            "cannot find",
            "did not start",
            "no valid live",
            "invalid_skeleton",
        )
    ):
        return "error"
    if "error" in lowered or "exception" in lowered:
        return "error"
    if "warning" in lowered:
        return "warning"
    rejected = REJECTED_COUNT_RE.search(line)
    if rejected is not None and int(rejected.group("count")) > 0:
        return "warning"
    if "rejected live reference packet" in lowered:
        return "warning"
    return "info"


class LogClassification:
    """Stable names over the moved implementation."""

    def component(self, line: str, backend: str | None = None) -> str:
        return classify_component(line, backend)

    def level(self, line: str) -> str:
        return classify_level(line)
