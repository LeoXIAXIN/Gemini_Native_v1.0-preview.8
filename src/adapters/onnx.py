"""ONNX Runtime adapter: facade over the migrated src/motion/tracking_policy.

M-decoupling: no legacy imports.  Requires the isolated hgpt runtime
(onnxruntime); importing this module is always safe, only the wrapped calls
import onnxruntime.
"""

from __future__ import annotations

import importlib
from typing import Any

from src.domain.errors import DomainError


def _policy() -> Any:
    try:
        return importlib.import_module("src.motion.tracking_policy")
    except (ImportError, ModuleNotFoundError) as exc:
        raise DomainError(
            "ONNX policy adapter requires the Humanoid-GPT runtime "
            f"(runtime-hgpt/python.exe): {exc}"
        ) from exc


class OnnxRuntimeAdapter:
    """Delegating wrapper around the migrated ONNX policy loader."""

    def make_args(
        self,
        policy_type: str = "mlp",
        device: str = "cpu",
        load_path: str = "",
    ) -> Any:
        return _policy().Args(
            policy_type=policy_type, device=device, load_path=load_path
        )

    def get_policy_onnx(
        self,
        args: Any,
        use_trt: bool = False,
        strict_trt: bool = False,
    ) -> Any:
        return _policy().get_policy_onnx(
            args, use_trt=use_trt, strict_trt=strict_trt
        )
