"""Check tracked source and optionally history without printing sensitive values."""
from __future__ import annotations

import argparse
from pathlib import PurePosixPath
import re
import subprocess


BLOCKED_PARTS = {"runtime-gmr", "runtime-hgpt", "__pycache__", "node_modules"}
BLOCKED_SUFFIXES = {".license", ".lic", ".pem", ".key", ".pfx", ".p12", ".exe", ".dll", ".zip", ".7z", ".bundle"}
BLOCKED_PREFIXES = ("storage/auth/", "storage/reports/", "storage/logs/", "storage/assets/", "storage/ckpts/")
SECRET_PATTERNS = {
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "github-token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    "aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "embedded-url-password": re.compile(rb"https?://[^\s/:@]+:[^\s/@]{8,}@"),
}
PUBLIC_BRAND_MARKERS = (
    b"chingmu", b"mcavatar", b"cmavatar", b"cmvrpn", b"cmhuman",
    "青木".encode("utf-8"),
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)


def blocked(path: str) -> bool:
    normalized = path.lower()
    parts = PurePosixPath(normalized).parts
    return (
        bool(set(parts) & BLOCKED_PARTS)
        or PurePosixPath(normalized).suffix in BLOCKED_SUFFIXES
        or normalized.startswith(BLOCKED_PREFIXES)
        or normalized == "config/gemini_native_config.json"
        or normalized == "gemini_native_manifest.json"
        or normalized == "resources/license/public_key.json"
        or normalized == "tracker_log.txt"
        or "private_key" in normalized or "private-key" in normalized
        or (PurePosixPath(normalized).name.startswith(".env") and not normalized.endswith(".env.example"))
        or (normalized.startswith("resources/license/") and normalized not in {
            "resources/license/readme.txt"})
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    objects = {}
    for row in git("ls-files", "--stage", "-z").split(b"\0"):
        if row:
            metadata, filename = row.split(b"\t", 1)
            objects[(metadata.split()[1].decode(), filename.decode("utf-8"))] = None
    if args.history:
        for row in git("rev-list", "--objects", "--all").splitlines():
            fields = row.decode("utf-8").split(" ", 1)
            if len(fields) == 2:
                objects[(fields[0], fields[1])] = None
    failures = set()
    checked = 0
    for object_id, path in objects:
        if blocked(path):
            failures.add((path, "excluded-public-path"))
        if not path.startswith(("src/", "tools/")):
            if any(marker in path.lower().encode("utf-8") for marker in PUBLIC_BRAND_MARKERS):
                failures.add((path, "public-brand-path"))
        if git("cat-file", "-t", object_id).strip() != b"blob":
            continue
        content = git("cat-file", "blob", object_id)
        checked += 1
        if not path.startswith(("src/", "tools/")):
            lowered = content.lower()
            if any(marker in lowered for marker in PUBLIC_BRAND_MARKERS):
                failures.add((path, "public-brand-reference"))
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                failures.add((path, label))
    for path, issue in sorted(failures):
        print(f"FAIL {issue}: {path}")
    print(f"Checked {checked} blobs; {len(failures)} findings.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
