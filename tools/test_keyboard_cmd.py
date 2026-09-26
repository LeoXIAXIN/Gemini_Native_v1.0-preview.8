#!/usr/bin/env python3
"""Check printable key mapping without an initialized Pygame video system."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from src.motion.keyboard_cmd import CmdBtn, CmdDim  # noqa: E402


def main() -> int:
    pygame.init()
    pygame.display.quit()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        axis = CmdDim("x", "s", "w")
        button = CmdBtn("kill", chr(96))
    assert (axis.low_key_code, axis.high_key_code) == (pygame.K_s, pygame.K_w)
    assert button.key_code == pygame.K_BACKQUOTE
    pygame.quit()
    print("KEYBOARD KEYCODE REGRESSION PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
