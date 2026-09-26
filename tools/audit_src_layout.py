#!/usr/bin/env python3
"""Fail if the package drifts away from its src-only deployment layout."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    ROOT / "src" / "controller" / "native_app.py",
    ROOT / "src" / "controller" / "static" / "index.html",
    ROOT / "config" / "gemini_native_config.json",
    ROOT / "resources" / "license" / "public_key.json",
    ROOT / "storage" / "assets.lock.json",
)
FORBIDDEN_TEXT = ("app/", "app\\", "--app-dir")


def main() -> int:
    failures: list[str] = []
    if (ROOT / "app").exists():
        failures.append("the removed app directory exists")
    for path in REQUIRED:
        if not path.is_file():
            failures.append(f"required file missing: {path.relative_to(ROOT)}")

    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_TEXT:
            if marker in text:
                failures.append(
                    f"{path.relative_to(ROOT)} still contains {marker!r}"
                )
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)} does not compile: {exc}")

    if failures:
        print("SRC LAYOUT AUDIT FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("SRC-ONLY LAYOUT AUDIT PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
